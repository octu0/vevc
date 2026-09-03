// rANSContextCodec.swift - Context-conditioned rANS with 12-bit per-symbol precision
// Target: Profile 2 L0 LL tail coefficients (pos 4..15)
// Zero-heap-allocation per block design

import Foundation

// MARK: - 12-bit rANS Constants

let rANSContextScaleBits: UInt32 = 12
let rANSContextScale: UInt32 = 1 << 12 // 4096
let rANSContextLBound: UInt32 = 1 << 15 // 32768

// MARK: - Lookup Tables

final class rANSContextTables: @unchecked Sendable {
    static let shared = rANSContextTables()

    let sigmoidLUT: [Int32]
    let geluLUT: [Int32]

    private init() {
        var sig = [Int32](repeating: 0, count: 8193)
        var i = 0
        while i <= 8192 {
            let z = (Double(i) - 4096.0) / 512.0
            let s = 1.0 / (1.0 + exp(-1.0 * z))
            sig[i] = Int32(round(s * 4096.0))
            i += 1
        }
        self.sigmoidLUT = sig

        var gel = [Int32](repeating: 0, count: 16385)
        var j = 0
        while j <= 16384 {
            let x = (Double(j) - 8192.0) / 1024.0
            let g = 0.5 * x * (1.0 + tanh(sqrt(2.0 / Double.pi) * (x + (0.044715 * x * x * x))))
            gel[j] = Int32(round(g * 4096.0))
            j += 1
        }
        self.geluLUT = gel
    }

    @inline(__always)
    func fastSigmoidQ12(_ xQ12: Int32) -> Int32 {
        let scaled = (xQ12 >> 3) + 4096
        if scaled <= 0 {
            return 0
        }
        if 8192 <= scaled {
            return 4096
        }
        return sigmoidLUT[Int(scaled)]
    }

    @inline(__always)
    func fastGELUQ12(_ xQ12: Int32) -> Int32 {
        let scaled = (xQ12 >> 2) + 8192
        if scaled <= 0 {
            return 0
        }
        if 16384 <= scaled {
            return xQ12
        }
        return geluLUT[Int(scaled)]
    }
}

// MARK: - Flattened Weights (single-array layout for one plane-level borrow)

/// Per-pos weight rows of rANSContextWeights concatenated into single arrays,
/// so callers can borrow w1/b1/w2 once per plane instead of extracting a
/// per-pos array on every predict call. Values are copied unchanged.
final class rANSContextFlatWeights: @unchecked Sendable {
    static let shared = rANSContextFlatWeights()

    let w1Offsets: [Int]
    let w1All: [Int32]
    let b1All: [Int32]
    let w2All: [Int32]
    let b2Q: [Int32]
    let invScalesQ: [Int32]

    private init() {
        let w = rANSContextWeights.shared
        var offs = [Int](repeating: 0, count: 16)
        var w1 = [Int32]()
        var b1 = [Int32](repeating: 0, count: 16 * 32)
        var w2 = [Int32](repeating: 0, count: 16 * 32)
        var pos = 4
        while pos < 16 {
            offs[pos] = w1.count
            w1.append(contentsOf: w.w1FlatQ[pos])
            var h = 0
            while h < 32 {
                b1[(pos * 32) + h] = w.b1Q[pos][h]
                w2[(pos * 32) + h] = w.w2Q[pos][h]
                h += 1
            }
            pos += 1
        }
        self.w1Offsets = offs
        self.w1All = w1
        self.b1All = b1
        self.w2All = w2
        self.b2Q = w.b2Q
        self.invScalesQ = w.invScalesQ
    }
}

// MARK: - rANSContext Workspace (Zero-Allocation per-block Buffer)

final class rANSContextWorkspace: @unchecked Sendable {
    let flatWeights = rANSContextFlatWeights.shared
    let tables = rANSContextTables.shared

    // Buffers for rANS encoding/decoding
    var encodeWords = [UInt16](repeating: 0, count: 128)
    var encodeWordCount: Int = 0

    // Buffers for CDF
    var rawCum = [Int32](repeating: 0, count: 132)
    var freqs = [UInt32](repeating: 0, count: 130)
    var cumFreqs = [UInt32](repeating: 0, count: 131)

    // Buffers for Neural Predictor
    var feat = [Int32](repeating: 0, count: 96)
    var hidden = [Int32](repeating: 0, count: 32)

    // Byte output buffer for block encoding
    var outputBytes = [UInt8](repeating: 0, count: 256)
    var outputByteCount: Int = 0

    // Reusable 16-coeff buffers
    var blockCoeffs = [Int16](repeating: 0, count: 16)
    var cArr = [Int16](repeating: 0, count: 16)
    var topBuf = [Int16](repeating: 0, count: 16)
    var leftBuf = [Int16](repeating: 0, count: 16)
    var syms = [Int](repeating: 0, count: 16)
    var escapes = [Int16](repeating: 0, count: 16)
    var hasEscapes = [Bool](repeating: false, count: 16)

    // Plane-level rANS stream buffers
    var planeWords = [UInt16](repeating: 0, count: 524288)
    var planeWordCount: Int = 0
    var planeEscapes = [UInt8](repeating: 0, count: 65536)
    var planeEscapeCount: Int = 0
    var planeState: UInt32 = rANSContextLBound

    // Plane-level rANS decoder state
    var decodePlaneState: UInt32 = 0
    var decodeWordPtr: UnsafePointer<UInt8>!
    var decodeWordEndPtr: UnsafePointer<UInt8>!
    var decodeEscapePtr: UnsafePointer<UInt8>!
    var decodeEscapeEndPtr: UnsafePointer<UInt8>!

    init() {}

    @inline(__always)
    func resetPlaneEncoder() {
        planeWordCount = 0
        planeEscapeCount = 0
        planeState = rANSContextLBound
    }

    @inline(__always)
    func planeEncodeSymbol(sym: Int, freq: UInt32, cumFreq: UInt32) {
        let maxState = ((rANSContextLBound >> rANSContextScaleBits) << 16) * freq
        while maxState <= planeState {
            if planeWords.count <= planeWordCount {
                planeWords.append(contentsOf: [UInt16](repeating: 0, count: planeWords.count))
            }
            planeWords[planeWordCount] = UInt16(truncatingIfNeeded: planeState & 0xFFFF)
            planeWordCount += 1
            planeState = planeState >> 16
        }
        planeState = ((planeState / freq) << rANSContextScaleBits) + (planeState % freq) + cumFreq
    }

    @inline(__always)
    func planeEncodeEscape(val: Int16) {
        if planeEscapes.count <= (planeEscapeCount + 2) {
            planeEscapes.append(contentsOf: [UInt8](repeating: 0, count: max(1024, planeEscapes.count)))
        }
        let uVal = UInt16(bitPattern: val)
        planeEscapes[planeEscapeCount] = UInt8(truncatingIfNeeded: (uVal >> 8) & 0xFF)
        planeEscapeCount += 1
        planeEscapes[planeEscapeCount] = UInt8(truncatingIfNeeded: uVal & 0xFF)
        planeEscapeCount += 1
    }

    /// Finalize plane encoding into a contiguous [UInt8] array:
    /// Layout:
    /// [4 bytes BigEndian: escapesCount (in bytes)]
    /// [4 bytes BigEndian: planeState]
    /// [planeWordCount * 2 bytes: planeWords emitted in reverse order]
    /// [planeEscapeCount bytes: planeEscapes in forward order]
    func finalizePlaneEncoder() -> [UInt8] {
        var out = [UInt8]()
        let totalBytes = 4 + 4 + (planeWordCount * 2) + planeEscapeCount
        out.reserveCapacity(totalBytes)

        let escCount32 = UInt32(planeEscapeCount)
        out.append(UInt8(truncatingIfNeeded: (escCount32 >> 24) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: (escCount32 >> 16) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: (escCount32 >> 8) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: escCount32 & 0xFF))

        let st = planeState
        out.append(UInt8(truncatingIfNeeded: (st >> 24) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: (st >> 16) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: (st >> 8) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: st & 0xFF))

        var wIdx = planeWordCount - 1
        while 0 <= wIdx {
            let w = planeWords[wIdx]
            out.append(UInt8(truncatingIfNeeded: (w >> 8) & 0xFF))
            out.append(UInt8(truncatingIfNeeded: w & 0xFF))
            wIdx -= 1
        }

        var eIdx = 0
        while eIdx < planeEscapeCount {
            out.append(planeEscapes[eIdx])
            eIdx += 1
        }

        return out
    }

    @inline(__always)
    func initPlaneDecoder(inputPtr: UnsafePointer<UInt8>, totalBytes: Int) throws {
        if totalBytes < 8 {
            throw DecodeError.insufficientData
        }
        let e0 = UInt32(inputPtr[0])
        let e1 = UInt32(inputPtr[1])
        let e2 = UInt32(inputPtr[2])
        let e3 = UInt32(inputPtr[3])
        let escCount = Int((e0 << 24) | (e1 << 16) | (e2 << 8) | e3)

        let s0 = UInt32(inputPtr[4])
        let s1 = UInt32(inputPtr[5])
        let s2 = UInt32(inputPtr[6])
        let s3 = UInt32(inputPtr[7])
        decodePlaneState = (s0 << 24) | (s1 << 16) | (s2 << 8) | s3

        let wordsStart = inputPtr + 8
        let wordsBytes = totalBytes - 8 - escCount
        if wordsBytes < 0 {
            throw DecodeError.insufficientData
        }
        decodeWordPtr = wordsStart
        decodeWordEndPtr = wordsStart + wordsBytes
        decodeEscapePtr = decodeWordEndPtr
        decodeEscapeEndPtr = inputPtr + totalBytes
    }

    @inline(__always)
    func planeDecodeSymbol(freqP: UnsafePointer<UInt32>, cumP: UnsafePointer<UInt32>) -> Int {
        let cum = decodePlaneState & (rANSContextScale - 1)
        var low = 0
        var high = 130
        while low + 1 < high {
            let mid = (low + high) >> 1
            if cumP[mid] <= cum {
                low = mid
            } else {
                high = mid
            }
        }
        let sym = low
        let freq = freqP[sym]
        let cumFreq = cumP[sym]

        let mask = rANSContextScale - 1
        decodePlaneState = (freq * (decodePlaneState >> rANSContextScaleBits)) + (decodePlaneState & mask) - cumFreq
        while decodePlaneState < rANSContextLBound {
            if decodeWordPtr + 1 < decodeWordEndPtr {
                let w = (UInt32(decodeWordPtr[0]) << 8) | UInt32(decodeWordPtr[1])
                decodeWordPtr = decodeWordPtr + 2
                decodePlaneState = (decodePlaneState << 16) | w
            } else {
                decodePlaneState = decodePlaneState << 16
                decodeWordPtr = decodeWordPtr + 2
            }
        }
        return sym
    }

    @inline(__always)
    func planeDecodeEscape() -> Int16 {
        if decodeEscapePtr + 1 < decodeEscapeEndPtr {
            let w = (UInt16(decodeEscapePtr[0]) << 8) | UInt16(decodeEscapePtr[1])
            decodeEscapePtr = decodeEscapePtr + 2
            return Int16(bitPattern: w)
        }
        return 0
    }

    /// Estimate model bits for tail coefficients of a block (pos 4..15).
    /// The caller borrows the workspace buffers and flat weights once outside
    /// its block loop and passes raw pointers down (rule 12'(j)).
    @inline(__always)
    func estimateModelTailBits(
        blockCoeffs: UnsafePointer<Int16>,
        topCoeffs: UnsafePointer<Int16>?,
        leftCoeffs: UnsafePointer<Int16>?,
        tempCoeffs: UnsafePointer<Int16>?,
        isPFrame: Bool,
        plane: Int,
        qstep: Int32,
        featP: UnsafeMutablePointer<Int32>,
        hiddenP: UnsafeMutablePointer<Int32>,
        rawP: UnsafeMutablePointer<Int32>,
        freqP: UnsafeMutablePointer<UInt32>,
        cumP: UnsafeMutablePointer<UInt32>,
        w1AllP: UnsafePointer<Int32>,
        b1AllP: UnsafePointer<Int32>,
        w2AllP: UnsafePointer<Int32>
    ) -> Int {
        var bitCost = 0
        var pos = 4
        while pos < 16 {
            let val = blockCoeffs[pos]
            let (mu, invScale) = predict(
                pos: pos,
                blockCoeffs: blockCoeffs,
                topCoeffs: topCoeffs,
                leftCoeffs: leftCoeffs,
                tempCoeffs: tempCoeffs,
                isPFrame: isPFrame,
                plane: plane,
                qstep: qstep,
                featP: featP,
                hiddenP: hiddenP,
                w1AllP: w1AllP,
                b1AllP: b1AllP,
                w2AllP: w2AllP
            )
            buildCDF(muQ12: mu, invScaleQ12: invScale, rawP: rawP, freqP: freqP, cumP: cumP)
            let sym: Int
            if val < -64 {
                sym = 129
                bitCost += 16
            } else {
                if 64 < val {
                    sym = 129
                    bitCost += 16
                } else {
                    sym = Int(val + 64)
                }
            }
            let freq = freqP[sym]
            if freq <= 1 {
                bitCost += 12
            } else {
                let lz = freq.leadingZeroBitCount
                let msb = 31 - lz
                let frac = Int((UInt64(freq ^ (1 << msb)) << 4) >> UInt64(msb))
                let log2Q4 = (msb << 4) | frac
                let bitsQ4 = (12 << 4) - log2Q4
                let symBits = max(1, (bitsQ4 + 8) >> 4)
                bitCost += symBits
            }
            pos += 1
        }
        return bitCost
    }

    @inline(__always)
    func resetEncoder() {
        encodeWordCount = 0
    }

    @inline(__always)
    func encPutWord(_ word: UInt16) {
        encodeWords[encodeWordCount] = word
        encodeWordCount += 1
    }

    /// Build cumulative frequency table for logistic distribution with Q12 mu and invScale.
    @inline(__always)
    func buildCDF(
        muQ12: Int32,
        invScaleQ12: Int32,
        rawP: UnsafeMutablePointer<Int32>,
        freqP: UnsafeMutablePointer<UInt32>,
        cumP: UnsafeMutablePointer<UInt32>
    ) {
        let totalSyms = 130
        let M: Int32 = 64
        rawP[0] = 0

        // Each sigmoid argument is element-independent: the Q12 arithmetic
        // runs 8 lanes at a time, the table lookup stays scalar (no NEON
        // gather). |valQ12 - mu| < 2^31 + 2^18 and |invScale| <= 6300
        // (< 2^13), so the Int64 lane product stays < 2^45 -- no wrap.
        // Clamping to Int32 first and then to +/-32768 equals the direct
        // 64-bit clamp because [-32768, 32767] lies inside the Int32 range.
        let lane = SIMD8<Int64>(0, 1, 2, 3, 4, 5, 6, 7)
        let muVec = SIMD8<Int64>(repeating: Int64(muQ12))
        let invVec = SIMD8<Int64>(repeating: Int64(invScaleQ12))
        let loBound = SIMD8<Int32>(repeating: -32768)
        let hiBound = SIMD8<Int32>(repeating: 32767)
        var s = 0
        while s + 8 <= 129 {
            let valQ12 = ((SIMD8<Int64>(repeating: Int64(s) - 64) &+ lane) &* 4096) &+ 2048
            let z64 = ((valQ12 &- muVec) &* invVec) &>> 12
            let zClamped = pointwiseMin(hiBound, pointwiseMax(loBound, SIMD8<Int32>(clamping: z64)))
            var k = 0
            while k < 8 {
                rawP[s + k + 1] = tables.fastSigmoidQ12(zClamped[k])
                k += 1
            }
            s += 8
        }
        while s < 129 {
            let val = Int32(s) - M
            let valQ12 = (Int64(val) * 4096) + 2048
            let diff64 = valQ12 - Int64(muQ12)
            let z64 = (diff64 * Int64(invScaleQ12)) >> 12
            let zClamped = Int32(max(-32768, min(32767, z64)))
            rawP[s + 1] = tables.fastSigmoidQ12(zClamped)
            s += 1
        }
        rawP[129] = 4096
        rawP[130] = 4096

        // freq entries are element-independent diffs; the sum of 130
        // frequencies is <= 130 * 4096 < 2^20, so lane totals cannot wrap.
        var totalFreq: UInt32 = 0
        var accT = SIMD8<UInt32>()
        let one = SIMD8<Int32>(repeating: 1)
        var sIdx = 0
        while sIdx + 8 <= totalSyms {
            let lo = UnsafeRawPointer(rawP + sIdx).loadUnaligned(as: SIMD8<Int32>.self)
            let hi = UnsafeRawPointer(rawP + sIdx + 1).loadUnaligned(as: SIMD8<Int32>.self)
            let f = SIMD8<UInt32>(truncatingIfNeeded: pointwiseMax(one, hi &- lo))
            UnsafeMutableRawPointer(freqP + sIdx).storeBytes(of: f, as: SIMD8<UInt32>.self)
            accT &+= f
            sIdx += 8
        }
        totalFreq = accT.wrappedSum()
        while sIdx < totalSyms {
            let f = UInt32(max(1, rawP[sIdx + 1] - rawP[sIdx]))
            freqP[sIdx] = f
            totalFreq += f
            sIdx += 1
        }

        // Normalize sum of frequencies to exactly 4096.
        // The over-total branch is a data-dependent argmax loop (each pass
        // reads the previous subtraction), so it stays scalar.
        if totalFreq != rANSContextScale {
            if totalFreq < rANSContextScale {
                let diff = rANSContextScale - totalFreq
                let centerSym = Int(max(0, min(128, (muQ12 >> 12) + M)))
                freqP[centerSym] += diff
            } else {
                var diff = totalFreq - rANSContextScale
                while 0 < diff {
                    var maxIdx = 0
                    var maxVal: UInt32 = 0
                    var i = 0
                    while i < totalSyms {
                        if maxVal < freqP[i] {
                            maxVal = freqP[i]
                            maxIdx = i
                        }
                        i += 1
                    }
                    if maxVal <= 1 {
                        break
                    }
                    let sub = min(diff, maxVal - 1)
                    freqP[maxIdx] -= sub
                    diff -= sub
                }
            }
        }

        // Cumulative CDF is a sequential prefix sum: stays scalar.
        var cum: UInt32 = 0
        cumP[0] = 0
        var cIdx = 0
        while cIdx < totalSyms {
            cum += freqP[cIdx]
            cumP[cIdx + 1] = cum
            cIdx += 1
        }
    }

    /// Neural Predictor: extracts features and runs MLP in Q12 fixed-point.
    /// featP/hiddenP are the workspace feat/hidden buffers and w1AllP/b1AllP/
    /// w2AllP the rANSContextFlatWeights arrays, all borrowed once by the
    /// caller outside its plane loop (rule 12'(j)).
    @inline(__always)
    func predict(
        pos: Int,
        blockCoeffs: UnsafePointer<Int16>,
        topCoeffs: UnsafePointer<Int16>?,
        leftCoeffs: UnsafePointer<Int16>?,
        tempCoeffs: UnsafePointer<Int16>?,
        isPFrame: Bool,
        plane: Int,
        qstep: Int32,
        featP: UnsafeMutablePointer<Int32>,
        hiddenP: UnsafeMutablePointer<Int32>,
        w1AllP: UnsafePointer<Int32>,
        b1AllP: UnsafePointer<Int32>,
        w2AllP: UnsafePointer<Int32>
    ) -> (mu: Int32, invScale: Int32) {
        var fCount = 0

        // 1. Bias
        featP[fCount] = 4096
        fCount += 1

        // 2. Causal intra-block coefficients (pos 0..pos-1) scaled by 1/16 (Q12 * 1/16 = * 256)
        var i = 0
        while i < pos {
            featP[fCount] = Int32(blockCoeffs[i]) << 8
            fCount += 1
            i += 1
        }

        // 3. Top block coefficients (16 elements)
        let topAvail = (topCoeffs != nil)
        if let top = topCoeffs {
            var k = 0
            while k < 16 {
                featP[fCount] = Int32(top[k]) << 8
                fCount += 1
                k += 1
            }
        } else {
            var k = 0
            while k < 16 {
                featP[fCount] = 0
                fCount += 1
                k += 1
            }
        }

        // 4. Left block coefficients (16 elements)
        let leftAvail = (leftCoeffs != nil)
        if let left = leftCoeffs {
            var k = 0
            while k < 16 {
                featP[fCount] = Int32(left[k]) << 8
                fCount += 1
                k += 1
            }
        } else {
            var k = 0
            while k < 16 {
                featP[fCount] = 0
                fCount += 1
                k += 1
            }
        }

        // 5. Temporal block coefficients (16 elements)
        let tempAvail = (tempCoeffs != nil)
        if let temp = tempCoeffs {
            var k = 0
            while k < 16 {
                featP[fCount] = Int32(temp[k]) << 8
                fCount += 1
                k += 1
            }
        } else {
            var k = 0
            while k < 16 {
                featP[fCount] = 0
                fCount += 1
                k += 1
            }
        }

        // 6. isP
        if isPFrame {
            featP[fCount] = 4096
        } else {
            featP[fCount] = 0
        }
        fCount += 1

        // 7. plane
        featP[fCount] = Int32(plane) * 2048
        fCount += 1

        // 8. logQ
        let logQDouble = log2(max(1.0, Double(qstep)))
        let logQQ12 = Int32(round(((logQDouble - 12.0) / 4.0) * 4096.0))
        featP[fCount] = logQQ12
        fCount += 1

        // 9. Flags
        if topAvail {
            featP[fCount] = 4096
        } else {
            featP[fCount] = 0
        }
        fCount += 1

        if leftAvail {
            featP[fCount] = 4096
        } else {
            featP[fCount] = 0
        }
        fCount += 1

        if tempAvail {
            featP[fCount] = 4096
        } else {
            featP[fCount] = 0
        }
        fCount += 1

        // 10. Position embeddings
        featP[fCount] = (Int32(pos / 4) * 4096) / 3
        fCount += 1
        featP[fCount] = (Int32(pos % 4) * 4096) / 3
        fCount += 1

        let inDim = fCount
        let w1P = w1AllP + flatWeights.w1Offsets[pos]
        let b1P = b1AllP + (pos * 32)
        let w2P = w2AllP + (pos * 32)
        let b2 = flatWeights.b2Q[pos]
        let invScale = flatWeights.invScalesQ[pos]

        // Int64 lane accumulators cannot wrap: |w1|,|w2| <= 6300 (< 2^13) and
        // |feat| <= 32767 << 8 (< 2^23), so a layer-1 product is < 2^36 and any
        // partial sum of <= 96 products stays < 2^43; |hidden| < 2^31, so a
        // layer-2 partial sum of 32 products stays < 2^49. The dot-product sum
        // is therefore order-independent and matches the scalar loop exactly.
        var h = 0
        while h < 32 {
            let rowP = w1P + (h * inDim)
            var acc = SIMD8<Int64>()
            var d = 0
            while d + 16 <= inDim {
                let wv = UnsafeRawPointer(rowP + d).loadUnaligned(as: SIMD16<Int32>.self)
                let fv = UnsafeRawPointer(featP + d).loadUnaligned(as: SIMD16<Int32>.self)
                acc &+= SIMD8<Int64>(truncatingIfNeeded: wv.lowHalf) &* SIMD8<Int64>(truncatingIfNeeded: fv.lowHalf)
                acc &+= SIMD8<Int64>(truncatingIfNeeded: wv.highHalf) &* SIMD8<Int64>(truncatingIfNeeded: fv.highHalf)
                d += 16
            }
            if d + 8 <= inDim {
                let wv = UnsafeRawPointer(rowP + d).loadUnaligned(as: SIMD8<Int32>.self)
                let fv = UnsafeRawPointer(featP + d).loadUnaligned(as: SIMD8<Int32>.self)
                acc &+= SIMD8<Int64>(truncatingIfNeeded: wv) &* SIMD8<Int64>(truncatingIfNeeded: fv)
                d += 8
            }
            var sum: Int64 = (Int64(b1P[h]) << 12) + acc.wrappedSum()
            while d < inDim {
                sum += Int64(rowP[d]) * Int64(featP[d])
                d += 1
            }
            let zQ = Int32(sum >> 12)
            hiddenP[h] = tables.fastGELUQ12(zQ)
            h += 1
        }

        let hLo = UnsafeRawPointer(hiddenP).loadUnaligned(as: SIMD16<Int32>.self)
        let hHi = UnsafeRawPointer(hiddenP + 16).loadUnaligned(as: SIMD16<Int32>.self)
        let wLo = UnsafeRawPointer(w2P).loadUnaligned(as: SIMD16<Int32>.self)
        let wHi = UnsafeRawPointer(w2P + 16).loadUnaligned(as: SIMD16<Int32>.self)
        var acc2 = SIMD8<Int64>(truncatingIfNeeded: hLo.lowHalf) &* SIMD8<Int64>(truncatingIfNeeded: wLo.lowHalf)
        acc2 &+= SIMD8<Int64>(truncatingIfNeeded: hLo.highHalf) &* SIMD8<Int64>(truncatingIfNeeded: wLo.highHalf)
        acc2 &+= SIMD8<Int64>(truncatingIfNeeded: hHi.lowHalf) &* SIMD8<Int64>(truncatingIfNeeded: wHi.lowHalf)
        acc2 &+= SIMD8<Int64>(truncatingIfNeeded: hHi.highHalf) &* SIMD8<Int64>(truncatingIfNeeded: wHi.highHalf)
        let muAcc = (Int64(b2) << 12) + acc2.wrappedSum()
        let mu = Int32(muAcc >> 12)

        return (mu, invScale)
    }
}

// MARK: - Rate-Distortion Selection Helper

@inline(__always)
func estimate4HTailBits(blockCoeffs: UnsafePointer<Int16>) -> Int {
    var nzCount = 0
    var bitCost = 0
    var pos = 4
    var lastNzPos = 3
    while pos < 16 {
        let val = blockCoeffs[pos]
        if val != 0 {
            nzCount += 1
            lastNzPos = pos
            let absVal: Int
            if val < 0 {
                absVal = -1 * Int(val)
            } else {
                absVal = Int(val)
            }
            let magBits = 31 - UInt32(absVal).leadingZeroBitCount
            bitCost += 9 + magBits
        }
        pos += 1
    }
    if nzCount < 2 {
        return 0
    }
    if lastNzPos < 15 {
        bitCost += 3
    }
    return bitCost
}
