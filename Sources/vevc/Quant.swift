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
        // why: reciprocal in Q16 fixed-point converts division to multiply+shift
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

// why: dead-zone bias is skipped in favor of roundToNearest for better SSIM scaling at high bitrates

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
        var qMidNum = 4       // HL/LH scale numerator   (4/4 = 1.0)
        var qMidDen = 4       // HL/LH scale denominator
        var qHighNum = 6      // HH scale numerator      (6/4 = 1.5)
        var qHighDen = 4      // HH scale denominator
        var qLowDivisor = 6

        switch layerIndex {
        case 2:
            qLowDivisor = 1
            // Scale up quantization steps faster for higher frequencies to save bitrate
            qMidNum = 3; qMidDen = 2          // 1.5
            qHighNum = 2; qHighDen = 1        // 2.0
        case 1:
            qMidNum = 1; qMidDen = 1          // 1.0
            qHighNum = 3; qHighDen = 2        // 1.5
        default: // layerIndex == 0
            qMidNum = 1; qMidDen = 2          // 0.5
            qHighNum = 1; qHighDen = 1        // 1.0
            qLowDivisor = 8
        }

        if isChroma {
            // Prevent color loss in high-motion scenes by capping chroma quantization steps.
            let cLow = min(16, max(1, baseStep / 8))
            let cMid = min(24, max(1, (baseStep * qMidNum) / qMidDen))
            let cHigh = min(48, max(1, (baseStep * qHighNum) / qHighDen))
            
            self.qLow = Quantizer(step: Int(cLow), roundToNearest: true)
            self.qMid = Quantizer(step: Int(cMid), roundToNearest: true)
            self.qHigh = Quantizer(step: Int(cHigh), roundToNearest: true)
        } else {
            // qLow: Strictly cap at 16 to completely preserve face gradients and base brightness
            let lLow = min(16, max(1, baseStep / qLowDivisor))
            self.qLow = Quantizer(step: Int(lLow), roundToNearest: true)
            
            // qMid: Cap at 64 to preserve facial contours and important structural edges
            let lMid = min(64, max(1, (baseStep * qMidNum) / qMidDen))
            self.qMid = Quantizer(step: Int(lMid), roundToNearest: true)
            
            // qHigh: Cap at 128 to allow background/fine details to degrade but prevent severe blocking artifacts
            if layerIndex == 2 {
                let lHigh = min(128, max(1, (baseStep * qHighNum) / qHighDen))
                self.qHigh = Quantizer(step: Int(lHigh), roundToNearest: false, deadZoneBias: -1638)
            } else {
                let lHigh = min(128, max(1, (baseStep * qHighNum) / qHighDen))
                self.qHigh = Quantizer(step: Int(lHigh), roundToNearest: true)
            }
        }
    }
}

// MARK: - Quantization SIMD

@inline(__always)
internal func quantizeSIMD(_ block: BlockView, q: Quantizer) {
    switch block.width {
    case 8:  quantizeSIMD8(block, q: q)
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
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        for i in 0..<block.width {
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
internal func quantizeSIMDSignedMapping(_ block: BlockView, q: Quantizer) {
    switch block.width {
    case 8:  quantizeSIMDSignedMapping8(block, q: q)
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
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        for i in 0..<block.width {
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

// MARK: - Dequantization SIMD

@inline(__always)
internal func dequantizeSIMD(_ block: BlockView, q: Quantizer) {
    switch block.width {
    case 4:  dequantizeSIMD4(block, q: q)
    case 8:  dequantizeSIMD8(block, q: q)
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
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        for i in 0..<block.width {
            let val = Int32(ptr[i])
            let res = val &* step
            let offset: Int32 = if 0 < val { step / 2 } else if val < 0 { -step / 2 } else { 0 }
            ptr[i] = Int16(clamping: res + offset)
        }
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMapping(_ block: BlockView, q: Quantizer) {
    switch block.width {
    case 4:  dequantizeSIMDSignedMapping4(block, q: q)
    case 8:  dequantizeSIMDSignedMapping8(block, q: q)
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
        for i in 0..<8 {
            let uVal = UInt16(bitPattern: ptr[i])
            let decodedUInt = ((uVal &>> 1) ^ (0 &- (uVal & 1)))
            ptr[i] = Int16(clamping: Int32(Int16(bitPattern: decodedUInt)) &* step)
        }
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMapping4(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<4 {
            let uVal = UInt16(bitPattern: ptr[i])
            let decodedUInt = ((uVal &>> 1) ^ (0 &- (uVal & 1)))
            ptr[i] = Int16(clamping: Int32(Int16(bitPattern: decodedUInt)) &* step)
        }
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMapping16(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<16 {
            let uVal = UInt16(bitPattern: ptr[i])
            let decodedUInt = ((uVal &>> 1) ^ (0 &- (uVal & 1)))
            ptr[i] = Int16(clamping: Int32(Int16(bitPattern: decodedUInt)) &* step)
        }
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMapping32(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<32 {
            let uVal = UInt16(bitPattern: ptr[i])
            let decodedUInt = ((uVal &>> 1) ^ (0 &- (uVal & 1)))
            ptr[i] = Int16(clamping: Int32(Int16(bitPattern: decodedUInt)) &* step)
        }
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMappingGeneric(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        for i in 0..<block.width {
            let uVal = UInt16(bitPattern: ptr[i])
            let decodedUInt = ((uVal &>> 1) ^ (0 &- (uVal & 1)))
            let val = Int32(Int16(bitPattern: decodedUInt))
            let res = val &* step
            let offset: Int32 = if 0 < val { step / 2 } else if val < 0 { -step / 2 } else { 0 }
            ptr[i] = Int16(clamping: res + offset)
        }
    }
}