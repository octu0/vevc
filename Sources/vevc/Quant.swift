// MARK: - Quantization

struct Quantizer: Sendable {
    let step: Int16
    let mul: Int32
    let bias: Int32
    let shift: Int16 = 16

    /// - Parameters:
    ///   - step: Quantization step size.
    ///   - roundToNearest: If true, sets bias to 1<<15 for round-to-nearest behavior.
    ///   - deadZoneBias: Pre-computed dead zone bias in Q16 fixed-point.
    ///     Positive values narrow the dead zone (towards round-to-nearest).
    ///     Negative values widen the dead zone (more values become 0).
    ///     Common values: -6554 (-0.10), -3277 (-0.05), 0 (none).
    init(step: Int, roundToNearest: Bool = false, deadZoneBias: Int32 = 0) {
        self.step = Int16(step)
        // reciprocal in Q16 fixed-point converts division to multiply+shift
        // Optimize by approximating division: val / step ≈ (val * mul) >> 16
        self.mul = Int32((1 << 16) / step)
        var b: Int32 = 0
        if roundToNearest {
            b = Int32(1 << 15)
        }
        if roundToNearest != true && deadZoneBias != 0 {
            b = deadZoneBias
        }
        self.bias = b
    }
}

struct QuantizationTable: Sendable {
    let step: Int16
    let isChroma: Bool
    public let qLow: Quantizer
    public let qMid: Quantizer
    public let qHigh: Quantizer

    init(baseStep: Int, isChroma: Bool = false, layerIndex: Int = 0) {
        let s = max(1, min(baseStep, 32767))
        self.step = Int16(s)
        self.isChroma = isChroma

        // CSF-based perceptual quantization
        // DWT subbands map to spatial frequency bands:
        //   Layer 0 (Base8)  = lowest freq  → high CSF sensitivity → fine quantization
        //   Layer 1 (L16)    = mid freq      → moderate sensitivity
        //   Layer 2 (L32)    = highest freq  → low sensitivity → coarse quantization
        //   HH (diagonal) = √2× higher freq than HL/LH → even less perceptible
        //
        // Scale factors are expressed as integer ratios (numerator, denominator)
        // to avoid floating-point computation entirely.
        var qMidNum = 4  // HL/LH scale numerator   (4/4 = 1.0)
        var qMidDen = 4  // HL/LH scale denominator
        var qHighNum = 6  // HH scale numerator      (6/4 = 1.5)
        var qHighDen = 4  // HH scale denominator
        var qLowDivisor = 6

        switch layerIndex {
        case 2:
            qLowDivisor = 1
            qMidNum = 1
            qMidDen = 1  // 1.0
            qHighNum = 5
            qHighDen = 4  // 1.25
        case 1:
            qMidNum = 1
            qMidDen = 2  // 0.50
            qHighNum = 1
            qHighDen = 1  // 1.00
        default:  // layerIndex == 0
            qMidNum = 1
            qMidDen = 2  // 0.5
            qHighNum = 3
            qHighDen = 4  // 0.75
            qLowDivisor = 8
        }

        var dzMidY: Int32 = -4000
        var dzHighY: Int32 = -8000
        
        if layerIndex == 2 {
            dzMidY = 0
            dzHighY = -8000
        } else if layerIndex == 1 {
            dzMidY = 8192
            dzHighY = 0
        } else if layerIndex == 0 {
            dzMidY = 16384  // +0.25 positive bias to boost Luma SSIM
            dzHighY = 8192
        }
        
        let dzMidC: Int32
        let dzHighC: Int32
        
        if layerIndex == 0 {
            dzMidC = -8000
            dzHighC = -16000
        } else if layerIndex == 1 {
            dzMidC = -16000
            dzHighC = -32000
        } else {
            dzMidC = -32000
            dzHighC = -64000
        }

        if isChroma {
            // qLow is the DC component: NEVER scale it to avoid destroying base color/brightness!
            let cLow = min(16, max(1, baseStep / 8))
            let cMid = min(24, max(1, (baseStep * qMidNum) / qMidDen))
            let cHigh = min(48, max(1, (baseStep * qHighNum) / qHighDen))

            self.qLow = Quantizer(step: Int(cLow), roundToNearest: true)
            self.qMid = Quantizer(step: Int(cMid), roundToNearest: false, deadZoneBias: dzMidC)
            self.qHigh = Quantizer(step: Int(cHigh), roundToNearest: false, deadZoneBias: dzHighC)
        } else {
            // qLow is the DC component: NEVER scale it!
            let lLow = min(16, max(1, baseStep / qLowDivisor))
            self.qLow = Quantizer(step: Int(lLow), roundToNearest: true)

            // Luma stepMult is 1: Never scale Luma steps because they ruin SSIM.
            let lMid = min(48, max(1, (baseStep * qMidNum) / qMidDen))
            self.qMid = Quantizer(step: Int(lMid), roundToNearest: false, deadZoneBias: dzMidY)

            let lHigh = min(64, max(1, (baseStep * qHighNum) / qHighDen))
            self.qHigh = Quantizer(step: Int(lHigh), roundToNearest: false, deadZoneBias: dzHighY)
        }
    }
}

// MARK: - Adaptive Quantization Table

/// Block-level adaptive quantization.
/// Pre-generates a discrete set of QuantizationTables with scaled qMid/qHigh
/// to redistribute bits from flat blocks to edge/texture blocks.
///
/// Selection logic:
///   - Measure each block's AC energy (sum of |HL| + |LH| + |HH| coefficients)
///   - Compare against the frame-wide average AC energy
///   - High-energy blocks (edges/textures) → higher qStep (coarser quantization, noise is masked)
///   - Low-energy blocks (flat regions, faces) → lower qStep (finer quantization, prevents visible blockiness)
///
/// qLow is NOT scaled — base frequency quality must remain constant.
struct AQTable: Sendable {
    /// 5 discrete quantization levels.
    /// Scale factors: [0.80, 0.90, 1.00, 1.10, 1.20]
    /// Level 2 = original (unmodified) quantization.
    /// Wider range prioritizes noise reduction in edge/texture regions at the cost of
    /// slightly larger file size. Applied to both I-frames and P-frames.
    let tables: (QuantizationTable, QuantizationTable, QuantizationTable, QuantizationTable, QuantizationTable)

    /// Unmodified table at level 2.
    var base: QuantizationTable { tables.2 }

    init(baseStep: Int, isChroma: Bool = false, layerIndex: Int = 0) {
        let scaleDen = 10

        @inline(__always)
        func makeScaled(_ num: Int) -> QuantizationTable {
            let scaledStep = max(1, (baseStep * num) / scaleDen)
            return QuantizationTable(baseStep: scaledStep, isChroma: isChroma, layerIndex: layerIndex)
        }

        self.tables = (
            makeScaled(7),  // 0.70 - faces/flat regions perfectly protected
            makeScaled(8),  // 0.80
            makeScaled(10), // 1.00 - average blocks
            makeScaled(11), // 1.10
            makeScaled(12)  // 1.20 - high energy textures slightly quantized
        )
    }

    @inline(__always)
    subscript(index: Int) -> QuantizationTable {
        switch index {
        case 0: return tables.0
        case 1: return tables.1
        case 2: return tables.2
        case 3: return tables.3
        default: return tables.4
        }
    }

    /// Select the appropriate quantization table index (0..4) based on block AC energy
    /// relative to the frame average, and block motion activity (SAD).
    @inline(__always)
    func selectIndex(energy: Int, avgEnergy: Int, sad: Int = -1, bx: Int = 0, by: Int = 0, colCount: Int = 1, rowCount: Int = 1) -> Int {
        // Psycho-visual AQ: ratio = energy / avgEnergy
        let safeAvg = max(1, avgEnergy)
        let ratioX10 = (energy * 10) / safeAvg

        var adjustedRatio = ratioX10

        // ROI (Region of Interest) Spatial Rate Allocation:
        let cx = colCount / 2
        let cy = rowCount / 2
        let dist = Int((bx - cx).magnitude) + Int((by - cy).magnitude)
        let maxDist = max(1, cx + cy)
        let distRatio = (dist * 10) / maxDist
        
        if 7 <= distRatio {
            adjustedRatio += 2
        } else if distRatio <= 4 {
            adjustedRatio -= 3
        }

        // Motion (SAD) and Spatial Frequency (Energy) Hybrid Adaptive Quantization:
        if sad != -1 {
            if sad < 128 {
                adjustedRatio += 2
            } else if 256 < sad {
                if ratioX10 <= 10 {
                    adjustedRatio -= 3
                } else {
                    adjustedRatio += 2
                }
            }
        }

        switch true {
        case adjustedRatio <= 6: return 0
        case adjustedRatio <= 8: return 1
        case adjustedRatio <= 12: return 2
        case adjustedRatio <= 15: return 3
        default: return 4
        }
    }

    /// Select the appropriate quantization table.
    @inline(__always)
    func select(energy: Int, avgEnergy: Int, sad: Int = -1, bx: Int = 0, by: Int = 0, colCount: Int = 1, rowCount: Int = 1) -> QuantizationTable {
        let idx = selectIndex(energy: energy, avgEnergy: avgEnergy, sad: sad, bx: bx, by: by, colCount: colCount, rowCount: rowCount)
        return self[idx]
    }
}

// MARK: - Quantization SIMD

@inline(__always)
internal func quantizeSIMD(_ block: BlockView, q: Quantizer) {
    switch block.width {
    case 8: quantizeSIMD8(block, q: q)
    case 16: quantizeSIMD16(block, q: q)
    case 32: quantizeSIMD32(block, q: q)
    default: quantizeSIMDGeneric(block, q: q)
    }
}

@inline(__always)
internal func quantizeSIMD8(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<8 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            ptr[i] = Int16(clamping: res)
        }
    }
}

@inline(__always)
internal func quantizeSIMD4(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<4 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            ptr[i] = Int16(clamping: res)
        }
    }
}

@inline(__always)
internal func quantizeSIMD16(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<16 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            ptr[i] = Int16(clamping: res)
        }
    }
}

@inline(__always)
internal func quantizeSIMD32(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<32 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            ptr[i] = Int16(clamping: res)
        }
    }
}

@inline(__always)
internal func quantizeSIMDGeneric(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    let width = block.width
    let height = block.height
    for y in 0..<height {
        let ptr = block.rowPointer(y: y)
        var x = 0
        while x + 16 <= width {
            for i in 0..<16 {
                let idx = x + i
                let val = Int32(ptr[idx])
                let signMask = val &>> 31
                let absVal = (val ^ signMask) &- signMask
                let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
                let res = (qVal ^ signMask) &- signMask
                ptr[idx] = Int16(clamping: res)
            }
            x += 16
        }
        while x + 8 <= width {
            for i in 0..<8 {
                let idx = x + i
                let val = Int32(ptr[idx])
                let signMask = val &>> 31
                let absVal = (val ^ signMask) &- signMask
                let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
                let res = (qVal ^ signMask) &- signMask
                ptr[idx] = Int16(clamping: res)
            }
            x += 8
        }
        while x + 4 <= width {
            for i in 0..<4 {
                let idx = x + i
                let val = Int32(ptr[idx])
                let signMask = val &>> 31
                let absVal = (val ^ signMask) &- signMask
                let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
                let res = (qVal ^ signMask) &- signMask
                ptr[idx] = Int16(clamping: res)
            }
            x += 4
        }
        while x < width {
            let val = Int32(ptr[x])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            ptr[x] = Int16(clamping: res)
            x += 1
        }
    }
}

@inline(__always)
internal func quantizeSIMDSignedMapping(_ block: BlockView, q: Quantizer) {
    switch block.width {
    case 8: quantizeSIMDSignedMapping8(block, q: q)
    case 16: quantizeSIMDSignedMapping16(block, q: q)
    case 32: quantizeSIMDSignedMapping32(block, q: q)
    default: quantizeSIMDSignedMappingGeneric(block, q: q)
    }
}

@inline(__always)
internal func quantizeSIMDSignedMapping8(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<8 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            let v = Int16(clamping: res)
            ptr[i] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
        }
    }
}

@inline(__always)
internal func quantizeSIMDSignedMapping4(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<4 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            let v = Int16(clamping: res)
            ptr[i] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
        }
    }
}

@inline(__always)
internal func quantizeSIMDSignedMapping16(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<16 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            let v = Int16(clamping: res)
            ptr[i] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
        }
    }
}

@inline(__always)
internal func quantizeSIMDSignedMapping32(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<32 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            let v = Int16(clamping: res)
            ptr[i] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
        }
    }
}

@inline(__always)
internal func quantizeSIMDSignedMappingGeneric(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    let width = block.width
    let height = block.height
    for y in 0..<height {
        let ptr = block.rowPointer(y: y)
        var x = 0
        while x + 16 <= width {
            for i in 0..<16 {
                let idx = x + i
                let val = Int32(ptr[idx])
                let signMask = val &>> 31
                let absVal = (val ^ signMask) &- signMask
                let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
                let res = (qVal ^ signMask) &- signMask
                let v = Int16(clamping: res)
                ptr[idx] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
            }
            x += 16
        }
        while x + 8 <= width {
            for i in 0..<8 {
                let idx = x + i
                let val = Int32(ptr[idx])
                let signMask = val &>> 31
                let absVal = (val ^ signMask) &- signMask
                let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
                let res = (qVal ^ signMask) &- signMask
                let v = Int16(clamping: res)
                ptr[idx] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
            }
            x += 8
        }
        while x + 4 <= width {
            for i in 0..<4 {
                let idx = x + i
                let val = Int32(ptr[idx])
                let signMask = val &>> 31
                let absVal = (val ^ signMask) &- signMask
                let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
                let res = (qVal ^ signMask) &- signMask
                let v = Int16(clamping: res)
                ptr[idx] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
            }
            x += 4
        }
        while x < width {
            let val = Int32(ptr[x])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            let v = Int16(clamping: res)
            ptr[x] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
            x += 1
        }
    }
}

// MARK: - Dequantization SIMD

@inline(__always)
internal func dequantizeSIMD(_ block: BlockView, q: Quantizer) {
    switch block.width {
    case 4: dequantizeSIMD4(block, q: q)
    case 8: dequantizeSIMD8(block, q: q)
    case 16: dequantizeSIMD16(block, q: q)
    case 32: dequantizeSIMD32(block, q: q)
    default: dequantizeSIMDGeneric(block, q: q)
    }
}

@inline(__always)
internal func dequantizeSIMD8(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<8 {
            ptr[i] = Int16(clamping: Int32(ptr[i]) &* step)
        }
    }
}

@inline(__always)
internal func dequantizeSIMD4(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<4 {
            ptr[i] = Int16(clamping: Int32(ptr[i]) &* step)
        }
    }
}

@inline(__always)
internal func dequantizeSIMD16(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<16 {
            ptr[i] = Int16(clamping: Int32(ptr[i]) &* step)
        }
    }
}

@inline(__always)
internal func dequantizeSIMD32(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<32 {
            ptr[i] = Int16(clamping: Int32(ptr[i]) &* step)
        }
    }
}

@inline(__always)
internal func dequantizeSIMDGeneric(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    let width = block.width
    let height = block.height
    for y in 0..<height {
        let ptr = block.rowPointer(y: y)
        var x = 0
        while x + 16 <= width {
            for i in 0..<16 {
                let idx = x + i
                let val = Int32(ptr[idx])
                let res = val &* step
                ptr[idx] = Int16(clamping: res)
            }
            x += 16
        }
        while x + 8 <= width {
            for i in 0..<8 {
                let idx = x + i
                let val = Int32(ptr[idx])
                let res = val &* step
                ptr[idx] = Int16(clamping: res)
            }
            x += 8
        }
        while x + 4 <= width {
            for i in 0..<4 {
                let idx = x + i
                let val = Int32(ptr[idx])
                let res = val &* step
                ptr[idx] = Int16(clamping: res)
            }
            x += 4
        }
        while x < width {
            let val = Int32(ptr[x])
            let res = val &* step
            ptr[x] = Int16(clamping: res)
            x += 1
        }
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMapping(_ block: BlockView, q: Quantizer) {
    switch block.width {
    case 4: dequantizeSIMDSignedMapping4(block, q: q)
    case 8: dequantizeSIMDSignedMapping8(block, q: q)
    case 16: dequantizeSIMDSignedMapping16(block, q: q)
    case 32: dequantizeSIMDSignedMapping32(block, q: q)
    default: dequantizeSIMDSignedMappingGeneric(block, q: q)
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMapping8(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<UInt16>.self)
        let decodedUInt = ((v &>> 1) ^ (.zero &- (v & 1)))
        let v16 = SIMD8<Int16>(truncatingIfNeeded: decodedUInt)
        let v0 = Int16(clamping: Int32(v16[0]) &* step)
        let v1 = Int16(clamping: Int32(v16[1]) &* step)
        let v2 = Int16(clamping: Int32(v16[2]) &* step)
        let v3 = Int16(clamping: Int32(v16[3]) &* step)
        let v4 = Int16(clamping: Int32(v16[4]) &* step)
        let v5 = Int16(clamping: Int32(v16[5]) &* step)
        let v6 = Int16(clamping: Int32(v16[6]) &* step)
        let v7 = Int16(clamping: Int32(v16[7]) &* step)
        let res16 = SIMD8<Int16>(v0, v1, v2, v3, v4, v5, v6, v7)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res16, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMapping4(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<UInt16>.self)
        let decodedUInt = ((v &>> 1) ^ (.zero &- (v & 1)))
        let v16 = SIMD4<Int16>(truncatingIfNeeded: decodedUInt)
        let v0 = Int16(clamping: Int32(v16[0]) &* step)
        let v1 = Int16(clamping: Int32(v16[1]) &* step)
        let v2 = Int16(clamping: Int32(v16[2]) &* step)
        let v3 = Int16(clamping: Int32(v16[3]) &* step)
        let res16 = SIMD4<Int16>(v0, v1, v2, v3)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res16, as: SIMD4<Int16>.self)
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMapping16(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        let v = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD16<UInt16>.self)
        let decodedUInt = ((v &>> 1) ^ (.zero &- (v & 1)))
        let v16 = SIMD16<Int16>(truncatingIfNeeded: decodedUInt)
        let v0 = Int16(clamping: Int32(v16[0]) &* step)
        let v1 = Int16(clamping: Int32(v16[1]) &* step)
        let v2 = Int16(clamping: Int32(v16[2]) &* step)
        let v3 = Int16(clamping: Int32(v16[3]) &* step)
        let v4 = Int16(clamping: Int32(v16[4]) &* step)
        let v5 = Int16(clamping: Int32(v16[5]) &* step)
        let v6 = Int16(clamping: Int32(v16[6]) &* step)
        let v7 = Int16(clamping: Int32(v16[7]) &* step)
        let v8 = Int16(clamping: Int32(v16[8]) &* step)
        let v9 = Int16(clamping: Int32(v16[9]) &* step)
        let v10 = Int16(clamping: Int32(v16[10]) &* step)
        let v11 = Int16(clamping: Int32(v16[11]) &* step)
        let v12 = Int16(clamping: Int32(v16[12]) &* step)
        let v13 = Int16(clamping: Int32(v16[13]) &* step)
        let v14 = Int16(clamping: Int32(v16[14]) &* step)
        let v15 = Int16(clamping: Int32(v16[15]) &* step)
        let res16 = SIMD16<Int16>(
            v0, v1, v2, v3, v4, v5, v6, v7,
            v8, v9, v10, v11, v12, v13, v14, v15
        )
        UnsafeMutableRawPointer(ptr).storeBytes(of: res16, as: SIMD16<Int16>.self)
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMapping32(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        let ptrRaw0 = UnsafeRawPointer(ptr)
        let ptrRaw1 = UnsafeRawPointer(ptr.advanced(by: 16))
        let v0 = ptrRaw0.loadUnaligned(as: SIMD16<UInt16>.self)
        let v1 = ptrRaw1.loadUnaligned(as: SIMD16<UInt16>.self)
        let decodedUInt0 = ((v0 &>> 1) ^ (.zero &- (v0 & 1)))
        let decodedUInt1 = ((v1 &>> 1) ^ (.zero &- (v1 & 1)))
        let v16_0 = SIMD16<Int16>(truncatingIfNeeded: decodedUInt0)
        let v16_1 = SIMD16<Int16>(truncatingIfNeeded: decodedUInt1)

        let a0 = Int16(clamping: Int32(v16_0[0]) &* step)
        let a1 = Int16(clamping: Int32(v16_0[1]) &* step)
        let a2 = Int16(clamping: Int32(v16_0[2]) &* step)
        let a3 = Int16(clamping: Int32(v16_0[3]) &* step)
        let a4 = Int16(clamping: Int32(v16_0[4]) &* step)
        let a5 = Int16(clamping: Int32(v16_0[5]) &* step)
        let a6 = Int16(clamping: Int32(v16_0[6]) &* step)
        let a7 = Int16(clamping: Int32(v16_0[7]) &* step)
        let a8 = Int16(clamping: Int32(v16_0[8]) &* step)
        let a9 = Int16(clamping: Int32(v16_0[9]) &* step)
        let a10 = Int16(clamping: Int32(v16_0[10]) &* step)
        let a11 = Int16(clamping: Int32(v16_0[11]) &* step)
        let a12 = Int16(clamping: Int32(v16_0[12]) &* step)
        let a13 = Int16(clamping: Int32(v16_0[13]) &* step)
        let a14 = Int16(clamping: Int32(v16_0[14]) &* step)
        let a15 = Int16(clamping: Int32(v16_0[15]) &* step)
        let res16_0 = SIMD16<Int16>(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15)

        let b0 = Int16(clamping: Int32(v16_1[0]) &* step)
        let b1 = Int16(clamping: Int32(v16_1[1]) &* step)
        let b2 = Int16(clamping: Int32(v16_1[2]) &* step)
        let b3 = Int16(clamping: Int32(v16_1[3]) &* step)
        let b4 = Int16(clamping: Int32(v16_1[4]) &* step)
        let b5 = Int16(clamping: Int32(v16_1[5]) &* step)
        let b6 = Int16(clamping: Int32(v16_1[6]) &* step)
        let b7 = Int16(clamping: Int32(v16_1[7]) &* step)
        let b8 = Int16(clamping: Int32(v16_1[8]) &* step)
        let b9 = Int16(clamping: Int32(v16_1[9]) &* step)
        let b10 = Int16(clamping: Int32(v16_1[10]) &* step)
        let b11 = Int16(clamping: Int32(v16_1[11]) &* step)
        let b12 = Int16(clamping: Int32(v16_1[12]) &* step)
        let b13 = Int16(clamping: Int32(v16_1[13]) &* step)
        let b14 = Int16(clamping: Int32(v16_1[14]) &* step)
        let b15 = Int16(clamping: Int32(v16_1[15]) &* step)
        let res16_1 = SIMD16<Int16>(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res16_0, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(ptr.advanced(by: 16)).storeBytes(of: res16_1, as: SIMD16<Int16>.self)
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMappingGeneric(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    let width = block.width
    let height = block.height
    for y in 0..<height {
        let ptr = block.rowPointer(y: y)
        var x = 0
        while x + 16 <= width {
            for i in 0..<16 {
                let idx = x + i
                let uVal = UInt16(bitPattern: ptr[idx])
                let decodedUInt = ((uVal &>> 1) ^ (0 &- (uVal & 1)))
                let val = Int32(Int16(bitPattern: decodedUInt))
                let res = val &* step
                ptr[idx] = Int16(clamping: res)
            }
            x += 16
        }
        while x + 8 <= width {
            for i in 0..<8 {
                let idx = x + i
                let uVal = UInt16(bitPattern: ptr[idx])
                let decodedUInt = ((uVal &>> 1) ^ (0 &- (uVal & 1)))
                let val = Int32(Int16(bitPattern: decodedUInt))
                let res = val &* step
                ptr[idx] = Int16(clamping: res)
            }
            x += 8
        }
        while x + 4 <= width {
            for i in 0..<4 {
                let idx = x + i
                let uVal = UInt16(bitPattern: ptr[idx])
                let decodedUInt = ((uVal &>> 1) ^ (0 &- (uVal & 1)))
                let val = Int32(Int16(bitPattern: decodedUInt))
                let res = val &* step
                ptr[idx] = Int16(clamping: res)
            }
            x += 4
        }
        while x < width {
            let uVal = UInt16(bitPattern: ptr[x])
            let decodedUInt = ((uVal &>> 1) ^ (0 &- (uVal & 1)))
            let val = Int32(Int16(bitPattern: decodedUInt))
            let res = val &* step
            ptr[x] = Int16(clamping: res)
            x += 1
        }
    }
}
