// CtxRans.swift - Context-conditioned rANS with 12-bit per-symbol precision
// Target: Profile 2 L0 LL tail coefficients (pos 4..15)
// Zero-heap-allocation per block design

import Foundation

// MARK: - Global Gate Flag

/// Profile 2 + VEVC_CTXRANS=1 gate
nonisolated(unsafe) public var isCtxRansEnabled: Bool = {
    if let val = ProcessInfo.processInfo.environment["VEVC_CTXRANS"] {
        if val == "1" {
            return true
        }
    }
    if let val2 = ProcessInfo.processInfo.environment["VEVC_CTX_RANS"] {
        if val2 == "1" {
            return true
        }
    }
    return false
}()

// MARK: - 12-bit rANS Constants

public let ctxRansScaleBits: UInt32 = 12
public let ctxRansScale: UInt32 = 1 << 12 // 4096
public let ctxRansLBound: UInt32 = 1 << 15 // 32768

// MARK: - Lookup Tables

public final class CtxRansTables: @unchecked Sendable {
    public static let shared = CtxRansTables()

    public let sigmoidLUT: [Int32]
    public let geluLUT: [Int32]

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
    public func fastSigmoidQ12(_ xQ12: Int32) -> Int32 {
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
    public func fastGELUQ12(_ xQ12: Int32) -> Int32 {
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

// MARK: - CtxRans Workspace (Zero-Allocation per-block Buffer)

public final class CtxRansWorkspace: @unchecked Sendable {
    public let weights = CtxRansWeights.shared
    public let tables = CtxRansTables.shared

    // Buffers for rANS encoding/decoding
    public var encWords = [UInt16](repeating: 0, count: 128)
    public var encWordCount: Int = 0

    // Buffers for CDF
    public var rawCum = [Int32](repeating: 0, count: 132)
    public var freqs = [UInt32](repeating: 0, count: 130)
    public var cumFreqs = [UInt32](repeating: 0, count: 131)

    // Buffers for Neural Predictor
    public var feat = [Int32](repeating: 0, count: 96)
    public var hidden = [Int32](repeating: 0, count: 32)

    // Byte output buffer for block encoding
    public var outputBytes = [UInt8](repeating: 0, count: 256)
    public var outputByteCount: Int = 0

    // Reusable 16-coeff buffers
    public var blockCoeffs = [Int16](repeating: 0, count: 16)
    public var cArr = [Int16](repeating: 0, count: 16)
    public var topBuf = [Int16](repeating: 0, count: 16)
    public var leftBuf = [Int16](repeating: 0, count: 16)
    public var syms = [Int](repeating: 0, count: 16)
    public var escapes = [Int16](repeating: 0, count: 16)
    public var hasEscapes = [Bool](repeating: false, count: 16)

    // Plane-level rANS stream buffers
    public var planeWords = [UInt16](repeating: 0, count: 524288)
    public var planeWordCount: Int = 0
    public var planeEscapes = [UInt8](repeating: 0, count: 65536)
    public var planeEscapeCount: Int = 0
    public var planeState: UInt32 = ctxRansLBound

    // Plane-level rANS decoder state
    public var decPlaneState: UInt32 = 0
    public var decWordPtr: UnsafePointer<UInt8>!
    public var decWordEndPtr: UnsafePointer<UInt8>!
    public var decEscapePtr: UnsafePointer<UInt8>!
    public var decEscapeEndPtr: UnsafePointer<UInt8>!

    public init() {}

    @inline(__always)
    public func resetPlaneEncoder() {
        planeWordCount = 0
        planeEscapeCount = 0
        planeState = ctxRansLBound
    }

    @inline(__always)
    public func planeEncSymbol(sym: Int, freq: UInt32, cumFreq: UInt32) {
        let maxState = ((ctxRansLBound >> ctxRansScaleBits) << 16) * freq
        while maxState <= planeState {
            if planeWords.count <= planeWordCount {
                planeWords.append(contentsOf: [UInt16](repeating: 0, count: planeWords.count))
            }
            planeWords[planeWordCount] = UInt16(truncatingIfNeeded: planeState & 0xFFFF)
            planeWordCount += 1
            planeState = planeState >> 16
        }
        planeState = ((planeState / freq) << ctxRansScaleBits) + (planeState % freq) + cumFreq
    }

    @inline(__always)
    public func planeEncEscape(val: Int16) {
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
    public func finalizePlaneEncoder() -> [UInt8] {
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
    public func initPlaneDecoder(inputPtr: UnsafePointer<UInt8>, totalBytes: Int) throws {
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
        decPlaneState = (s0 << 24) | (s1 << 16) | (s2 << 8) | s3

        let wordsStart = inputPtr + 8
        let wordsBytes = totalBytes - 8 - escCount
        if wordsBytes < 0 {
            throw DecodeError.insufficientData
        }
        decWordPtr = wordsStart
        decWordEndPtr = wordsStart + wordsBytes
        decEscapePtr = decWordEndPtr
        decEscapeEndPtr = inputPtr + totalBytes
    }

    @inline(__always)
    public func planeDecSymbol() -> Int {
        let cum = decPlaneState & (ctxRansScale - 1)
        var low = 0
        var high = 130
        while low + 1 < high {
            let mid = (low + high) >> 1
            if cumFreqs[mid] <= cum {
                low = mid
            } else {
                high = mid
            }
        }
        let sym = low
        let freq = freqs[sym]
        let cumFreq = cumFreqs[sym]

        let mask = ctxRansScale - 1
        decPlaneState = (freq * (decPlaneState >> ctxRansScaleBits)) + (decPlaneState & mask) - cumFreq
        while decPlaneState < ctxRansLBound {
            if decWordPtr + 1 < decWordEndPtr {
                let w = (UInt32(decWordPtr[0]) << 8) | UInt32(decWordPtr[1])
                decWordPtr = decWordPtr + 2
                decPlaneState = (decPlaneState << 16) | w
            } else {
                decPlaneState = decPlaneState << 16
                decWordPtr = decWordPtr + 2
            }
        }
        return sym
    }

    @inline(__always)
    public func planeDecEscape() -> Int16 {
        if decEscapePtr + 1 < decEscapeEndPtr {
            let w = (UInt16(decEscapePtr[0]) << 8) | UInt16(decEscapePtr[1])
            decEscapePtr = decEscapePtr + 2
            return Int16(bitPattern: w)
        }
        return 0
    }

    /// Estimate model bits for tail coefficients of a block (pos 4..15)
    @inline(__always)
    public func estimateModelTailBits(
        blockCoeffs: UnsafePointer<Int16>,
        topCoeffs: UnsafePointer<Int16>?,
        leftCoeffs: UnsafePointer<Int16>?,
        tempCoeffs: UnsafePointer<Int16>?,
        isPFrame: Bool,
        plane: Int,
        qstep: Int32
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
                qstep: qstep
            )
            buildCDF(muQ12: mu, invScaleQ12: invScale)
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
            let freq = freqs[sym]
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
    public func resetEncoder() {
        encWordCount = 0
    }

    @inline(__always)
    public func encPutWord(_ word: UInt16) {
        encWords[encWordCount] = word
        encWordCount += 1
    }

    /// Build cumulative frequency table for logistic distribution with Q12 mu and invScale.
    @inline(__always)
    public func buildCDF(muQ12: Int32, invScaleQ12: Int32) {
        let totalSyms = 130
        let M: Int32 = 64
        rawCum[0] = 0

        var s = 0
        while s < 129 {
            let val = Int32(s) - M
            let valQ12 = (Int64(val) * 4096) + 2048
            let diff64 = valQ12 - Int64(muQ12)
            let z64 = (diff64 * Int64(invScaleQ12)) >> 12
            let zClamped = Int32(max(-32768, min(32767, z64)))
            rawCum[s + 1] = tables.fastSigmoidQ12(zClamped)
            s += 1
        }
        rawCum[129] = 4096
        rawCum[130] = 4096

        var totalFreq: UInt32 = 0
        var sIdx = 0
        while sIdx < totalSyms {
            let f = UInt32(max(1, rawCum[sIdx + 1] - rawCum[sIdx]))
            freqs[sIdx] = f
            totalFreq += f
            sIdx += 1
        }

        // Normalize sum of frequencies to exactly 4096
        if totalFreq != ctxRansScale {
            if totalFreq < ctxRansScale {
                let diff = ctxRansScale - totalFreq
                let centerSym = Int(max(0, min(128, (muQ12 >> 12) + M)))
                freqs[centerSym] += diff
            } else {
                var diff = totalFreq - ctxRansScale
                while 0 < diff {
                    var maxIdx = 0
                    var maxVal: UInt32 = 0
                    var i = 0
                    while i < totalSyms {
                        if maxVal < freqs[i] {
                            maxVal = freqs[i]
                            maxIdx = i
                        }
                        i += 1
                    }
                    if maxVal <= 1 {
                        break
                    }
                    let sub = min(diff, maxVal - 1)
                    freqs[maxIdx] -= sub
                    diff -= sub
                }
            }
        }

        var cum: UInt32 = 0
        cumFreqs[0] = 0
        var cIdx = 0
        while cIdx < totalSyms {
            cum += freqs[cIdx]
            cumFreqs[cIdx + 1] = cum
            cIdx += 1
        }
    }

    /// Neural Predictor: extracts features and runs MLP in Q12 fixed-point.
    @inline(__always)
    public func predict(
        pos: Int,
        blockCoeffs: UnsafePointer<Int16>,
        topCoeffs: UnsafePointer<Int16>?,
        leftCoeffs: UnsafePointer<Int16>?,
        tempCoeffs: UnsafePointer<Int16>?,
        isPFrame: Bool,
        plane: Int,
        qstep: Int32
    ) -> (mu: Int32, invScale: Int32) {
        var fCount = 0

        // 1. Bias
        feat[fCount] = 4096
        fCount += 1

        // 2. Causal intra-block coefficients (pos 0..pos-1) scaled by 1/16 (Q12 * 1/16 = * 256)
        var i = 0
        while i < pos {
            feat[fCount] = Int32(blockCoeffs[i]) << 8
            fCount += 1
            i += 1
        }

        // 3. Top block coefficients (16 elements)
        let topAvail = (topCoeffs != nil)
        if let top = topCoeffs {
            var k = 0
            while k < 16 {
                feat[fCount] = Int32(top[k]) << 8
                fCount += 1
                k += 1
            }
        } else {
            var k = 0
            while k < 16 {
                feat[fCount] = 0
                fCount += 1
                k += 1
            }
        }

        // 4. Left block coefficients (16 elements)
        let leftAvail = (leftCoeffs != nil)
        if let left = leftCoeffs {
            var k = 0
            while k < 16 {
                feat[fCount] = Int32(left[k]) << 8
                fCount += 1
                k += 1
            }
        } else {
            var k = 0
            while k < 16 {
                feat[fCount] = 0
                fCount += 1
                k += 1
            }
        }

        // 5. Temporal block coefficients (16 elements)
        let tempAvail = (tempCoeffs != nil)
        if let temp = tempCoeffs {
            var k = 0
            while k < 16 {
                feat[fCount] = Int32(temp[k]) << 8
                fCount += 1
                k += 1
            }
        } else {
            var k = 0
            while k < 16 {
                feat[fCount] = 0
                fCount += 1
                k += 1
            }
        }

        // 6. isP
        if isPFrame {
            feat[fCount] = 4096
        } else {
            feat[fCount] = 0
        }
        fCount += 1

        // 7. plane
        feat[fCount] = Int32(plane) * 2048
        fCount += 1

        // 8. logQ
        let logQDouble = log2(max(1.0, Double(qstep)))
        let logQQ12 = Int32(round(((logQDouble - 12.0) / 4.0) * 4096.0))
        feat[fCount] = logQQ12
        fCount += 1

        // 9. Flags
        if topAvail {
            feat[fCount] = 4096
        } else {
            feat[fCount] = 0
        }
        fCount += 1

        if leftAvail {
            feat[fCount] = 4096
        } else {
            feat[fCount] = 0
        }
        fCount += 1

        if tempAvail {
            feat[fCount] = 4096
        } else {
            feat[fCount] = 0
        }
        fCount += 1

        // 10. Position embeddings
        feat[fCount] = (Int32(pos / 4) * 4096) / 3
        fCount += 1
        feat[fCount] = (Int32(pos % 4) * 4096) / 3
        fCount += 1

        let inDim = fCount
        let w1Flat = weights.w1FlatQ[pos]
        let b1 = weights.b1Q[pos]
        let w2 = weights.w2Q[pos]
        let b2 = weights.b2Q[pos]
        let invScale = weights.invScalesQ[pos]

        var h = 0
        while h < 32 {
            var sum: Int64 = Int64(b1[h]) << 12
            let offset = h * inDim
            var d = 0
            while d < inDim {
                sum += Int64(w1Flat[offset + d]) * Int64(feat[d])
                d += 1
            }
            let zQ = Int32(sum >> 12)
            hidden[h] = tables.fastGELUQ12(zQ)
            h += 1
        }

        var muAcc: Int64 = Int64(b2) << 12
        var h2 = 0
        while h2 < 32 {
            muAcc += Int64(w2[h2]) * Int64(hidden[h2])
            h2 += 1
        }
        let mu = Int32(muAcc >> 12)

        return (mu, invScale)
    }

    /// Encode tail coefficients of a 4x4 block (pos 4..15) using 12-bit rANS.
    /// Returns byte count written to outputBytes.
    public func encodeBlockTail(
        blockCoeffs: UnsafePointer<Int16>,
        topCoeffs: UnsafePointer<Int16>?,
        leftCoeffs: UnsafePointer<Int16>?,
        tempCoeffs: UnsafePointer<Int16>?,
        isPFrame: Bool,
        plane: Int,
        qstep: Int32
    ) -> Int {
        resetEncoder()

        var pos = 4
        while pos < 16 {
            let val = blockCoeffs[pos]
            if val < -64 {
                syms[pos] = 129
                escapes[pos] = val
                hasEscapes[pos] = true
            } else {
                if 64 < val {
                    syms[pos] = 129
                    escapes[pos] = val
                    hasEscapes[pos] = true
                } else {
                    syms[pos] = Int(val + 64)
                    hasEscapes[pos] = false
                }
            }
            pos += 1
        }

        // rANS State initialization
        var state: UInt32 = ctxRansLBound

        // Encode symbols in reverse order: pos 15 down to 4
        var rPos = 15
        while 4 <= rPos {
            let (mu, invScale) = predict(
                pos: rPos,
                blockCoeffs: blockCoeffs,
                topCoeffs: topCoeffs,
                leftCoeffs: leftCoeffs,
                tempCoeffs: tempCoeffs,
                isPFrame: isPFrame,
                plane: plane,
                qstep: qstep
            )
            buildCDF(muQ12: mu, invScaleQ12: invScale)

            let sym = syms[rPos]
            let freq = freqs[sym]
            let cumFreq = cumFreqs[sym]

            // Renormalize
            let maxState = ((ctxRansLBound >> ctxRansScaleBits) << 16) * freq
            while maxState <= state {
                encPutWord(UInt16(truncatingIfNeeded: state & 0xFFFF))
                state = state >> 16
            }

            // Encode symbol
            state = ((state / freq) << ctxRansScaleBits) + (state % freq) + cumFreq

            rPos -= 1
        }

        // Flush rANS state (4 bytes, Big-Endian)
        outputByteCount = 0
        outputBytes[outputByteCount] = UInt8(truncatingIfNeeded: (state >> 24) & 0xFF)
        outputByteCount += 1
        outputBytes[outputByteCount] = UInt8(truncatingIfNeeded: (state >> 16) & 0xFF)
        outputByteCount += 1
        outputBytes[outputByteCount] = UInt8(truncatingIfNeeded: (state >> 8) & 0xFF)
        outputByteCount += 1
        outputBytes[outputByteCount] = UInt8(truncatingIfNeeded: state & 0xFF)
        outputByteCount += 1

        // Emit emitted words in reverse order
        var wIdx = encWordCount - 1
        while 0 <= wIdx {
            let w = encWords[wIdx]
            outputBytes[outputByteCount] = UInt8(truncatingIfNeeded: (w >> 8) & 0xFF)
            outputByteCount += 1
            outputBytes[outputByteCount] = UInt8(truncatingIfNeeded: w & 0xFF)
            outputByteCount += 1
            wIdx -= 1
        }

        // Emit escape values (2 bytes each, Big-Endian)
        var ePos = 4
        while ePos < 16 {
            if hasEscapes[ePos] {
                let uVal = UInt16(bitPattern: escapes[ePos])
                outputBytes[outputByteCount] = UInt8(truncatingIfNeeded: (uVal >> 8) & 0xFF)
                outputByteCount += 1
                outputBytes[outputByteCount] = UInt8(truncatingIfNeeded: uVal & 0xFF)
                outputByteCount += 1
            }
            ePos += 1
        }

        return outputByteCount
    }

    /// Decode tail coefficients of a 4x4 block using 12-bit rANS.
    /// Returns number of input bytes consumed.
    public func decodeBlockTail(
        inputPtr: UnsafePointer<UInt8>,
        inputCount: Int,
        blockCoeffs: UnsafeMutablePointer<Int16>,
        topCoeffs: UnsafePointer<Int16>?,
        leftCoeffs: UnsafePointer<Int16>?,
        tempCoeffs: UnsafePointer<Int16>?,
        isPFrame: Bool,
        plane: Int,
        qstep: Int32
    ) -> Int {
        var curPtr = inputPtr
        let endPtr = inputPtr + inputCount

        // Read initial state
        var state: UInt32 = 0
        if 4 <= inputCount {
            let b0 = UInt32(curPtr[0])
            let b1 = UInt32(curPtr[1])
            let b2 = UInt32(curPtr[2])
            let b3 = UInt32(curPtr[3])
            state = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
            curPtr = curPtr + 4
        }

        var decEscapes = [Bool](repeating: false, count: 16)

        // Decode forward: pos 4..15
        var pos = 4
        while pos < 16 {
            let (mu, invScale) = predict(
                pos: pos,
                blockCoeffs: UnsafePointer(blockCoeffs),
                topCoeffs: topCoeffs,
                leftCoeffs: leftCoeffs,
                tempCoeffs: tempCoeffs,
                isPFrame: isPFrame,
                plane: plane,
                qstep: qstep
            )
            buildCDF(muQ12: mu, invScaleQ12: invScale)

            let cum = state & (ctxRansScale - 1)

            // Binary search symbol
            var low = 0
            var high = 130
            while low + 1 < high {
                let mid = (low + high) >> 1
                if cumFreqs[mid] <= cum {
                    low = mid
                } else {
                    high = mid
                }
            }
            let sym = low
            let freq = freqs[sym]
            let cumFreq = cumFreqs[sym]

            // Advance state
            let mask = ctxRansScale - 1
            state = (freq * (state >> ctxRansScaleBits)) + (state & mask) - cumFreq
            while state < ctxRansLBound {
                if curPtr + 1 < endPtr {
                    let w = (UInt32(curPtr[0]) << 8) | UInt32(curPtr[1])
                    curPtr = curPtr + 2
                    state = (state << 16) | w
                } else {
                    state = state << 16
                    curPtr = curPtr + 2
                }
            }

            if sym == 129 {
                decEscapes[pos] = true
            } else {
                blockCoeffs[pos] = Int16(sym - 64)
                decEscapes[pos] = false
            }

            pos += 1
        }

        // Read escapes if any
        var ePos = 4
        while ePos < 16 {
            if decEscapes[ePos] {
                if curPtr + 1 < endPtr {
                    let w = (UInt16(curPtr[0]) << 8) | UInt16(curPtr[1])
                    curPtr = curPtr + 2
                    blockCoeffs[ePos] = Int16(bitPattern: w)
                }
            }
            ePos += 1
        }

        return curPtr - inputPtr
    }
}

// MARK: - Rate-Distortion Selection Helper

@inline(__always)
public func estimate4HTailBits(blockCoeffs: UnsafePointer<Int16>) -> Int {
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

@inline(__always)
public func shouldUseCtxRans(tailBytes: Int, estimated4HBits: Int) -> Bool {
    if estimated4HBits <= 0 {
        return false
    }
    let modelBits = tailBytes * 8
    if modelBits < estimated4HBits {
        return true
    }
    return false
}
