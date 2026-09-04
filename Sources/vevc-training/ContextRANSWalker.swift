import Foundation

// MARK: - rANS Context Fixed Tables

public final class RANSContextLookupTables: @unchecked Sendable {
    public static let shared = RANSContextLookupTables()

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

// MARK: - rANS Context Sample

public struct RANSContextSample {
    public let pos: Int
    public let feat: [Int32]
    public let val: Int16
    public let isEscape: Bool
    public let sym: Int
}

// MARK: - Context Weights Container (Q12)

public final class RANSContextWeightsContainer: @unchecked Sendable {
    public let invScalesQ: [Int32]
    public let b2Q: [Int32]
    public let inDims: [Int]
    public let w2Q: [[Int32]]
    public let b1Q: [[Int32]]
    public let w1FlatQ: [[Int32]]

    public init(blob: [Int32]) {
        var invS = [Int32](repeating: 0, count: 16)
        var b2 = [Int32](repeating: 0, count: 16)
        var dims = [Int](repeating: 0, count: 16)
        var w2 = [[Int32]](repeating: [], count: 16)
        var b1 = [[Int32]](repeating: [], count: 16)
        var w1 = [[Int32]](repeating: [], count: 16)

        var offset = 0
        for i in 0..<16 {
            invS[i] = blob[offset]
            offset += 1
        }
        for i in 0..<16 {
            b2[i] = blob[offset]
            offset += 1
        }
        for i in 0..<16 {
            dims[i] = Int(blob[offset])
            offset += 1
        }
        for pos in 4..<16 {
            var w2Row = [Int32](repeating: 0, count: 32)
            for h in 0..<32 {
                w2Row[h] = blob[offset]
                offset += 1
            }
            w2[pos] = w2Row

            var b1Row = [Int32](repeating: 0, count: 32)
            for h in 0..<32 {
                b1Row[h] = blob[offset]
                offset += 1
            }
            b1[pos] = b1Row

            let inDim = dims[pos]
            var w1Row = [Int32](repeating: 0, count: 32 * inDim)
            for i in 0..<(32 * inDim) {
                w1Row[i] = blob[offset]
                offset += 1
            }
            w1[pos] = w1Row
        }

        self.invScalesQ = invS
        self.b2Q = b2
        self.inDims = dims
        self.w2Q = w2
        self.b1Q = b1
        self.w1FlatQ = w1
    }

    public func predict(pos: Int, feat: [Int32]) -> (mu: Int32, invScale: Int32) {
        let tables = RANSContextLookupTables.shared
        let inDim = inDims[pos]
        let b1 = b1Q[pos]
        let w1 = w1FlatQ[pos]
        let w2 = w2Q[pos]
        let b2 = b2Q[pos]
        let invScale = invScalesQ[pos]

        var hidden = [Int32](repeating: 0, count: 32)
        for h in 0..<32 {
            var sum: Int64 = Int64(b1[h]) << 12
            let wOff = h * inDim
            for d in 0..<inDim {
                sum += Int64(w1[wOff + d]) * Int64(feat[d])
            }
            let zQ = Int32(sum >> 12)
            hidden[h] = tables.fastGELUQ12(zQ)
        }

        var muAcc: Int64 = Int64(b2) << 12
        for h in 0..<32 {
            muAcc += Int64(w2[h]) * Int64(hidden[h])
        }
        let mu = Int32(muAcc >> 12)
        return (mu, invScale)
    }

    public func buildCDF(muQ12: Int32, invScaleQ12: Int32) -> [UInt32] {
        let tables = RANSContextLookupTables.shared
        let totalSyms = 130
        let M: Int32 = 64
        var rawP = [Int32](repeating: 0, count: 131)
        rawP[0] = 0

        var s = 0
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

        var freqP = [UInt32](repeating: 0, count: totalSyms)
        var totalFreq: UInt32 = 0
        for sIdx in 0..<totalSyms {
            let f = UInt32(max(1, rawP[sIdx + 1] - rawP[sIdx]))
            freqP[sIdx] = f
            totalFreq += f
        }

        let rANSContextScale: UInt32 = 4096
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
                    for i in 0..<totalSyms {
                        if maxVal < freqP[i] {
                            maxVal = freqP[i]
                            maxIdx = i
                        }
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
        return freqP
    }
}

// MARK: - Walker Engine

public final class ContextRANSWalker {
    public init() {}

    /// L0Y 平面からコーディング対象ブロックを抽出し、pos 4..15 のコンテキスト特徴とシンボルを発行する。
    public func walkL0Y(frame: VSDFrame, onSample: (RANSContextSample) -> Void) {
        let subbands = frame.subbands
        let llPlane = subbands[0]
        let width = llPlane.width
        let height = llPlane.height
        let colCount = width / 4
        let rowCount = height / 4
        let blockCount = colCount * rowCount
        let qstep = frame.qstep

        var isZeroFlags = [Bool](repeating: false, count: blockCount)
        var nonZeroIndices: [Int] = []
        nonZeroIndices.reserveCapacity(blockCount)

        for bIdx in 0..<blockCount {
            let r = bIdx / colCount
            let c = bIdx % colCount
            var allZero = true

            for s in 0..<4 {
                let plane = subbands[s]
                let ox = c * 4
                let oy = r * 4
                for y in 0..<4 {
                    let rowOffset = (oy + y) * plane.width + ox
                    for x in 0..<4 {
                        if plane.data[rowOffset + x] != 0 {
                            allZero = false
                            break
                        }
                    }
                    if allZero != true {
                        break
                    }
                }
                if allZero != true {
                    break
                }
            }

            isZeroFlags[bIdx] = allZero
            if allZero != true {
                nonZeroIndices.append(bIdx)
            }
        }

        let logQDouble = log2(max(1.0, Double(qstep)))
        let logQQ12 = Int32(round(((logQDouble - 12.0) / 4.0) * 4096.0))

        for bIdx in nonZeroIndices {
            let r = bIdx / colCount
            let c = bIdx % colCount

            var blockCoeffs = [Int16](repeating: 0, count: 16)
            for y in 0..<4 {
                let rowOffset = (r * 4 + y) * width + (c * 4)
                for x in 0..<4 {
                    blockCoeffs[y * 4 + x] = llPlane.data[rowOffset + x]
                }
            }

            var topCoeffs: [Int16]? = nil
            if 0 < r {
                var top = [Int16](repeating: 0, count: 16)
                for y in 0..<4 {
                    let rowOffset = ((r - 1) * 4 + y) * width + (c * 4)
                    for x in 0..<4 {
                        top[y * 4 + x] = llPlane.data[rowOffset + x]
                    }
                }
                topCoeffs = top
            }

            var leftCoeffs: [Int16]? = nil
            if 0 < c {
                var left = [Int16](repeating: 0, count: 16)
                for y in 0..<4 {
                    let rowOffset = (r * 4 + y) * width + ((c - 1) * 4)
                    for x in 0..<4 {
                        left[y * 4 + x] = llPlane.data[rowOffset + x]
                    }
                }
                leftCoeffs = left
            }

            var topAvail: Int32 = 0
            if topCoeffs != nil {
                topAvail = 4096
            }
            var leftAvail: Int32 = 0
            if leftCoeffs != nil {
                leftAvail = 4096
            }

            for pos in 4..<16 {
                let inDim = 57 + pos
                var feat = [Int32](repeating: 0, count: inDim)

                feat[0] = 4096
                for i in 0..<pos {
                    feat[1 + i] = Int32(blockCoeffs[i]) << 8
                }

                let offTop = 1 + pos
                if let top = topCoeffs {
                    for k in 0..<16 {
                        feat[offTop + k] = Int32(top[k]) << 8
                    }
                }

                let offLeft = offTop + 16
                if let left = leftCoeffs {
                    for k in 0..<16 {
                        feat[offLeft + k] = Int32(left[k]) << 8
                    }
                }

                // tempCoeffs (16 zeros) at offLeft + 16
                let offMeta = offLeft + 32
                feat[offMeta + 0] = 4096 // isPFrame
                feat[offMeta + 1] = 0    // plane (luma = 0)
                feat[offMeta + 2] = logQQ12
                feat[offMeta + 3] = topAvail
                feat[offMeta + 4] = leftAvail
                feat[offMeta + 5] = 0    // tempAvail
                feat[offMeta + 6] = (Int32(pos / 4) * 4096) / 3
                feat[offMeta + 7] = (Int32(pos % 4) * 4096) / 3

                let val = blockCoeffs[pos]
                let isEscape = (val < -64) || (64 < val)
                let sym: Int
                if isEscape {
                    sym = 129
                } else {
                    sym = Int(val + 64)
                }

                let sample = RANSContextSample(
                    pos: pos,
                    feat: feat,
                    val: val,
                    isEscape: isEscape,
                    sym: sym
                )
                onSample(sample)
            }
        }
    }

    /// サンプル列を実際に rANSContext と全く同一の規則で符号化し、出力バイト列を生成する
    public func simulateRANSBytes(samples: [RANSContextSample], weights: RANSContextWeightsContainer) -> [UInt8] {
        let rANSContextScaleBits: UInt32 = 12
        let rANSContextLBound: UInt32 = 32768

        var planeWords: [UInt16] = []
        planeWords.reserveCapacity(samples.count)
        var planeEscapes: [UInt8] = []
        var planeState: UInt32 = rANSContextLBound

        // エンコード時はデコード時の LIFO のために逆順で push
        var idx = samples.count - 1
        while 0 <= idx {
            let s = samples[idx]
            if s.isEscape {
                let uVal = UInt16(bitPattern: s.val)
                planeEscapes.append(UInt8(truncatingIfNeeded: (uVal >> 8) & 0xFF))
                planeEscapes.append(UInt8(truncatingIfNeeded: uVal & 0xFF))
            }

            let (mu, invScale) = weights.predict(pos: s.pos, feat: s.feat)
            let freqP = weights.buildCDF(muQ12: mu, invScaleQ12: invScale)

            var cumFreq: UInt32 = 0
            var k = 0
            while k < s.sym {
                cumFreq += freqP[k]
                k += 1
            }

            let freq = freqP[s.sym]
            let maxState = ((rANSContextLBound >> rANSContextScaleBits) << 16) * freq
            while maxState <= planeState {
                planeWords.append(UInt16(truncatingIfNeeded: planeState & 0xFFFF))
                planeState = planeState >> 16
            }
            planeState = ((planeState / freq) << rANSContextScaleBits) + (planeState % freq) + cumFreq
            idx -= 1
        }

        // finalizePlaneEncoder 相当
        var out: [UInt8] = []
        let escCount32 = UInt32(planeEscapes.count)
        out.append(UInt8(truncatingIfNeeded: (escCount32 >> 24) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: (escCount32 >> 16) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: (escCount32 >> 8) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: escCount32 & 0xFF))

        out.append(UInt8(truncatingIfNeeded: (planeState >> 24) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: (planeState >> 16) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: (planeState >> 8) & 0xFF))
        out.append(UInt8(truncatingIfNeeded: planeState & 0xFF))

        var wIdx = planeWords.count - 1
        while 0 <= wIdx {
            let w = planeWords[wIdx]
            out.append(UInt8(truncatingIfNeeded: (w >> 8) & 0xFF))
            out.append(UInt8(truncatingIfNeeded: w & 0xFF))
            wIdx -= 1
        }

        out.append(contentsOf: planeEscapes)
        return out
    }

    /// 現行重みを用いてサンプル群の理論符号長（NLL bits）を計算する。
    public func evaluateNLLBits(samples: [RANSContextSample], weights: RANSContextWeightsContainer) -> Double {
        var totalBits: Double = 0.0
        for s in samples {
            let (mu, invScale) = weights.predict(pos: s.pos, feat: s.feat)
            let freqP = weights.buildCDF(muQ12: mu, invScaleQ12: invScale)
            let freq = freqP[s.sym]
            let nll = 12.0 - log2(Double(freq))
            totalBits += nll
            if s.isEscape {
                totalBits += 16.0
            }
        }
        return totalBits
    }
}
