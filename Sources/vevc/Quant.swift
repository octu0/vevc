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
        // val / step ≈ (val * mul) >> 16 で高速化する
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

// why: dead-zone bias in Q16 fixed-point avoids rounding small coefficients
// up to 1, improving compression of near-zero subbands
private let kDeadZoneBiasNeg005: Int32 = -3277  // -0.05 * 65536
private let kDeadZoneBiasNeg010: Int32 = -6554  // -0.10 * 65536

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
        var deadZoneBiasMid: Int32 = 0   // HL/LH dead zone bias (Q16)
        var deadZoneBiasHigh: Int32 = 0  // HH dead zone bias (Q16)

        switch layerIndex {
        case 2:
            qMidNum = 6; qMidDen = 5          // 1.2
            qHighNum = 6; qHighDen = 4        // 1.5
            deadZoneBiasMid = kDeadZoneBiasNeg005
            deadZoneBiasHigh = kDeadZoneBiasNeg010
        case 1:
            qMidNum = 2; qMidDen = 4          // 0.5
            qHighNum = 4; qHighDen = 4        // 1.0
            deadZoneBiasMid = 0
            deadZoneBiasHigh = kDeadZoneBiasNeg005
        default: // layerIndex == 0
            qMidNum = 1; qMidDen = 4          // 0.25
            qHighNum = 2; qHighDen = 4        // 0.5
            qLowDivisor = 12
            deadZoneBiasMid = 0
            deadZoneBiasHigh = 0
        }

        if isChroma {
            self.qLow = Quantizer(step: Int(min(2048, max(1, baseStep / 8))), roundToNearest: true)
            self.qMid = Quantizer(step: Int(min(4096, max(1, (baseStep * qMidNum) / qMidDen))))
            self.qHigh = Quantizer(step: Int(min(8192, max(1, (baseStep * qHighNum) / qHighDen))))
        } else {
            self.qLow = Quantizer(step: Int(min(4096, max(1, baseStep / qLowDivisor))), roundToNearest: true)
            self.qMid = Quantizer(step: Int(min(8192, max(1, (baseStep * qMidNum) / qMidDen))), deadZoneBias: deadZoneBiasMid)
            self.qHigh = Quantizer(step: Int(min(16384, max(1, (baseStep * qHighNum) / qHighDen))), deadZoneBias: deadZoneBiasHigh)
        }
    }
}

// MARK: - Quantization SIMD

@inline(__always)
private func performQuantizeSIMD8(_ vec: SIMD8<Int16>, mul: Int32, shift: Int32, bias: Int32) -> SIMD8<Int16> {
    let mask = vec &>> 15
    let absVec = (vec ^ mask) &- mask
    
    let low32 = SIMD4<Int32>(
        Int32(absVec[0]), Int32(absVec[1]), Int32(absVec[2]), Int32(absVec[3])
    )
    let high32 = SIMD4<Int32>(
        Int32(absVec[4]), Int32(absVec[5]), Int32(absVec[6]), Int32(absVec[7])
    )
    
    let mulVec = SIMD4<Int32>(repeating: mul)
    let shiftVec = SIMD4<Int32>(repeating: shift)
    let biasVec = SIMD4<Int32>(repeating: bias)
    
    var resLow32 = (((low32 &* mulVec) &+ biasVec) &>> shiftVec)
    var resHigh32 = (((high32 &* mulVec) &+ biasVec) &>> shiftVec)
    
    resLow32.replace(with: SIMD4<Int32>.zero, where: resLow32 .< 0)
    resHigh32.replace(with: SIMD4<Int32>.zero, where: resHigh32 .< 0)
    
    let res = SIMD8<Int16>(
        Int16(resLow32[0]), Int16(resLow32[1]), Int16(resLow32[2]), Int16(resLow32[3]),
        Int16(resHigh32[0]), Int16(resHigh32[1]), Int16(resHigh32[2]), Int16(resHigh32[3])
    )
    
    return (res ^ mask) &- mask
}

@inline(__always)
private func performQuantizeSIMD4(_ vec: SIMD4<Int16>, mul: Int32, shift: Int32, bias: Int32) -> SIMD4<Int16> {
    let mask = vec &>> 15
    let absVec = (vec ^ mask) &- mask
    
    let vals = SIMD4<Int32>(
        Int32(absVec[0]), Int32(absVec[1]), Int32(absVec[2]), Int32(absVec[3])
    )
    
    let mulVec = SIMD4<Int32>(repeating: mul)
    let shiftVec = SIMD4<Int32>(repeating: shift)
    let biasVec = SIMD4<Int32>(repeating: bias)
    
    var res = (((vals &* mulVec) &+ biasVec) &>> shiftVec)
    res.replace(with: SIMD4<Int32>.zero, where: res .< 0)
    
    let res16 = SIMD4<Int16>(
        Int16(res[0]), Int16(res[1]), Int16(res[2]), Int16(res[3])
    )
    
    return (res16 ^ mask) &- mask
}

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
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        let res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func quantizeSIMD4(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
        let res = performQuantizeSIMD4(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD4<Int16>.self)
    }
}

@inline(__always)
internal func quantizeSIMD16(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        var vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        var res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD8<Int16>.self)

        vec = UnsafeRawPointer(ptr + 8).loadUnaligned(as: SIMD8<Int16>.self)
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr + 8).storeBytes(of: res, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func quantizeSIMD32(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        var vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        var res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD8<Int16>.self)

        vec = UnsafeRawPointer(ptr + 8).loadUnaligned(as: SIMD8<Int16>.self)
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr + 8).storeBytes(of: res, as: SIMD8<Int16>.self)

        vec = UnsafeRawPointer(ptr + 16).loadUnaligned(as: SIMD8<Int16>.self)
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr + 16).storeBytes(of: res, as: SIMD8<Int16>.self)

        vec = UnsafeRawPointer(ptr + 24).loadUnaligned(as: SIMD8<Int16>.self)
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr + 24).storeBytes(of: res, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func quantizeSIMDGeneric(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        var i = 0
        while (i + 8) <= block.width {
            let vec = UnsafeRawPointer(ptr + i).loadUnaligned(as: SIMD8<Int16>.self)
            let res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
            UnsafeMutableRawPointer(ptr + i).storeBytes(of: res, as: SIMD8<Int16>.self)
            i += 8
        }
        while i < block.width {
            let val = Int32(ptr[i])
            let absVal = abs(val)
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            var res: Int32 = qVal
            if val <= -1 {
                res = (-1 * qVal)
            }
            ptr[i] = Int16(res)
            i += 1
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
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        let res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        let mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer(ptr).storeBytes(of: mask, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func quantizeSIMDSignedMapping4(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
        let res = performQuantizeSIMD4(vec, mul: mul, shift: shift, bias: bias)
        let mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer(ptr).storeBytes(of: mask, as: SIMD4<Int16>.self)
    }
}

@inline(__always)
internal func quantizeSIMDSignedMapping16(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        var vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        var res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        var mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer(ptr).storeBytes(of: mask, as: SIMD8<Int16>.self)
        
        vec = UnsafeRawPointer(ptr + 8).loadUnaligned(as: SIMD8<Int16>.self)
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer(ptr + 8).storeBytes(of: mask, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func quantizeSIMDSignedMapping32(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        var vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        var res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        var mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer(ptr).storeBytes(of: mask, as: SIMD8<Int16>.self)
        
        vec = UnsafeRawPointer(ptr + 8).loadUnaligned(as: SIMD8<Int16>.self)
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer(ptr + 8).storeBytes(of: mask, as: SIMD8<Int16>.self)
        
        vec = UnsafeRawPointer(ptr + 16).loadUnaligned(as: SIMD8<Int16>.self)
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer(ptr + 16).storeBytes(of: mask, as: SIMD8<Int16>.self)
        
        vec = UnsafeRawPointer(ptr + 24).loadUnaligned(as: SIMD8<Int16>.self)
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer(ptr + 24).storeBytes(of: mask, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func quantizeSIMDSignedMappingGeneric(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        var i = 0
        while (i + 8) <= block.width {
            let vec = UnsafeRawPointer(ptr + i).loadUnaligned(as: SIMD8<Int16>.self)
            let res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
            let mask = ((res &<< 1) ^ (res &>> 15))
            UnsafeMutableRawPointer(ptr + i).storeBytes(of: mask, as: SIMD8<Int16>.self)
            i += 8
        }
        while i < block.width {
            let val = Int32(ptr[i])
            let absVal = abs(val)
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            var res: Int32 = qVal
            if val <= -1 {
                res = (-1 * qVal)
            }
            let v = Int16(res)
            ptr[i] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
            i += 1
        }
    }
}

// MARK: - Dequantization SIMD

@inline(__always)
private func performDequantizeSIMD8(_ vec: SIMD8<Int16>, step: Int32) -> SIMD8<Int16> {
    let vLow32 = SIMD4<Int32>(
        Int32(vec[0]), Int32(vec[1]), Int32(vec[2]), Int32(vec[3])
    )
    let vHigh32 = SIMD4<Int32>(
        Int32(vec[4]), Int32(vec[5]), Int32(vec[6]), Int32(vec[7])
    )
    
    let stepVec = SIMD4<Int32>(repeating: step)
    let rLow32 = vLow32 &* stepVec
    let rHigh32 = vHigh32 &* stepVec
    
    return SIMD8<Int16>(
        Int16(clamping: rLow32[0]), Int16(clamping: rLow32[1]), Int16(clamping: rLow32[2]), Int16(clamping: rLow32[3]),
        Int16(clamping: rHigh32[0]), Int16(clamping: rHigh32[1]), Int16(clamping: rHigh32[2]), Int16(clamping: rHigh32[3])
    )
}

@inline(__always)
private func performDequantizeSIMD4(_ vec: SIMD4<Int16>, step: Int32) -> SIMD4<Int16> {
    let vals = SIMD4<Int32>(
        Int32(vec[0]), Int32(vec[1]), Int32(vec[2]), Int32(vec[3])
    )
    let stepVec = SIMD4<Int32>(repeating: step)
    let r = vals &* stepVec
    
    return SIMD4<Int16>(
        Int16(clamping: r[0]), Int16(clamping: r[1]), Int16(clamping: r[2]), Int16(clamping: r[3])
    )
}

@inline(__always)
private func performDequantizeSIMDSignedMapping8(_ vec: SIMD8<Int16>, step: Int32) -> SIMD8<Int16> {
    let dVal = unsafeBitCast(vec, to: SIMD8<UInt16>.self)
    let one = SIMD8<UInt16>(repeating: 1)
    let signMask = ~(dVal & one) &+ one
    let sign = unsafeBitCast(signMask, to: SIMD8<Int16>.self)
    let orig = unsafeBitCast(dVal &>> 1, to: SIMD8<Int16>.self) ^ sign
    
    return performDequantizeSIMD8(orig, step: step)
}

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
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        let res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func dequantizeSIMD4(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
        let res = performDequantizeSIMD4(vec, step: step)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD4<Int16>.self)
    }
}

@inline(__always)
internal func dequantizeSIMD16(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        var vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        var res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD8<Int16>.self)
        
        vec = UnsafeRawPointer(ptr + 8).loadUnaligned(as: SIMD8<Int16>.self)
        res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr + 8).storeBytes(of: res, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func dequantizeSIMD32(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        var vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        var res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD8<Int16>.self)
        
        vec = UnsafeRawPointer(ptr + 8).loadUnaligned(as: SIMD8<Int16>.self)
        res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr + 8).storeBytes(of: res, as: SIMD8<Int16>.self)
        
        vec = UnsafeRawPointer(ptr + 16).loadUnaligned(as: SIMD8<Int16>.self)
        res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr + 16).storeBytes(of: res, as: SIMD8<Int16>.self)
        
        vec = UnsafeRawPointer(ptr + 24).loadUnaligned(as: SIMD8<Int16>.self)
        res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr + 24).storeBytes(of: res, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func dequantizeSIMDGeneric(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        var i = 0
        while (i + 8) <= block.width {
            let vec = UnsafeRawPointer(ptr + i).loadUnaligned(as: SIMD8<Int16>.self)
            let res = performDequantizeSIMD8(vec, step: step)
            UnsafeMutableRawPointer(ptr + i).storeBytes(of: res, as: SIMD8<Int16>.self)
            i += 8
        }
        while i < block.width {
            let val = Int32(ptr[i])
            let res = val &* step
            let offset: Int32 = (val > 0) ? (step / 2) : (val < 0 ? (-step / 2) : 0)
            ptr[i] = Int16(clamping: res + offset)
            i += 1
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
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        let mask = (0 &- (vec & 1))
        let logicalShift = ((vec &>> 1) & 0x7FFF)
        let decoded = (logicalShift ^ mask)
        let res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMapping4(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
        let mask = (0 &- (vec & 1))
        let logicalShift = ((vec &>> 1) & 0x7FFF)
        let decoded = (logicalShift ^ mask)
        let res = performDequantizeSIMD4(decoded, step: step)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD4<Int16>.self)
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMapping16(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        var vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        var mask = (0 &- (vec & 1))
        var logicalShift = ((vec &>> 1) & 0x7FFF)
        var decoded = (logicalShift ^ mask)
        var res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD8<Int16>.self)
        
        vec = UnsafeRawPointer(ptr + 8).loadUnaligned(as: SIMD8<Int16>.self)
        mask = (0 &- (vec & 1))
        logicalShift = ((vec &>> 1) & 0x7FFF)
        decoded = (logicalShift ^ mask)
        res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr + 8).storeBytes(of: res, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMapping32(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        var vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        var mask = (0 &- (vec & 1))
        var logicalShift = ((vec &>> 1) & 0x7FFF)
        var decoded = (logicalShift ^ mask)
        var res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD8<Int16>.self)

        vec = UnsafeRawPointer(ptr + 8).loadUnaligned(as: SIMD8<Int16>.self)
        mask = (0 &- (vec & 1))
        logicalShift = ((vec &>> 1) & 0x7FFF)
        decoded = (logicalShift ^ mask)
        res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr + 8).storeBytes(of: res, as: SIMD8<Int16>.self)

        vec = UnsafeRawPointer(ptr + 16).loadUnaligned(as: SIMD8<Int16>.self)
        mask = (0 &- (vec & 1))
        logicalShift = ((vec &>> 1) & 0x7FFF)
        decoded = (logicalShift ^ mask)
        res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr + 16).storeBytes(of: res, as: SIMD8<Int16>.self)
        
        vec = UnsafeRawPointer(ptr + 24).loadUnaligned(as: SIMD8<Int16>.self)
        mask = (0 &- (vec & 1))
        logicalShift = ((vec &>> 1) & 0x7FFF)
        decoded = (logicalShift ^ mask)
        res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr + 24).storeBytes(of: res, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
internal func dequantizeSIMDSignedMappingGeneric(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        var i = 0
        while (i + 8) <= block.width {
            let vec = UnsafeRawPointer(ptr + i).loadUnaligned(as: SIMD8<Int16>.self)
            let mask = (0 &- (vec & 1))
            let logicalShift = ((vec &>> 1) & 0x7FFF)
            let decoded = (logicalShift ^ mask)
            
            let res = performDequantizeSIMD8(decoded, step: step)
            UnsafeMutableRawPointer(ptr + i).storeBytes(of: res, as: SIMD8<Int16>.self)
            i += 8
        }
        while i < block.width {
            let uVal = UInt16(bitPattern: ptr[i])
            let decodedUInt = ((uVal &>> 1) ^ (0 &- (uVal & 1)))
            let decoded = Int16(bitPattern: decodedUInt)
            let val = Int32(decoded)
            let res = val &* step
            let offset: Int32 = (val > 0) ? (step / 2) : (val < 0 ? (-step / 2) : 0)
            ptr[i] = Int16(clamping: res + offset)
            i += 1
        }
    }
}

// MARK: - Cascaded Quantization

@inline(__always)
func quantizeCascaded32(block: inout BlockView, qt: QuantizationTable, isChroma: Bool) {
    let step = Int(qt.step)
    let sLL3 = isChroma ? max(1, step * 3 / 4) : max(1, step)
    let sHL3 = isChroma ? max(1, step * 4 / 4) : max(1, step * 5 / 4)
    let sHL2 = isChroma ? max(1, step * 5 / 4) : max(1, step * 6 / 4)
    let sHL1 = isChroma ? max(1, step * 6 / 4) : max(1, step * 8 / 4)

    let qLL3 = Quantizer(step: sLL3, roundToNearest: true)
    let qHL3 = Quantizer(step: sHL3, roundToNearest: true)
    let qHL2 = Quantizer(step: sHL2, roundToNearest: true)
    let qHL1 = Quantizer(step: sHL1, roundToNearest: true)

    let base = block.base
        // Cache parameters to avoid struct access overhead
        let qLL3mul = qLL3.mul; let qLL3bias = qLL3.bias; let qLL3shift = qLL3.shift
        let qHL3mul = qHL3.mul; let qHL3bias = qHL3.bias; let qHL3shift = qHL3.shift
        let qHL2mul = qHL2.mul; let qHL2bias = qHL2.bias; let qHL2shift = qHL2.shift
        let qHL1mul = qHL1.mul; let qHL1bias = qHL1.bias; let qHL1shift = qHL1.shift

        @inline(__always)
        func applyQLL3(_ v: Int16) -> Int16 {
            let absVal = abs(Int32(v))
            let qVal = max(0, (((absVal &* qLL3mul) &+ qLL3bias) &>> Int32(qLL3shift)))
            let res = v <= -1 ? (-1 * qVal) : qVal
            return Int16(res)
        }
        
        @inline(__always)
        func applyQSM(_ v: Int16, mul: Int32, bias: Int32, shift: Int16) -> Int16 {
            let absVal = abs(Int32(v))
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> Int32(shift)))
            let res = v <= -1 ? (-1 * qVal) : qVal
            let mask = ((res &<< 1) ^ (res &>> 15))
            return Int16(mask)
        }
        
        for y in 0..<4 {
            let rowOffset = y * 32
            for x in 0..<4   { base[rowOffset + x] = applyQLL3(base[rowOffset + x]) }
            for x in 4..<8   { base[rowOffset + x] = applyQSM(base[rowOffset + x], mul: qHL3mul, bias: qHL3bias, shift: qHL3shift) }
            for x in 8..<16  { base[rowOffset + x] = applyQSM(base[rowOffset + x], mul: qHL2mul, bias: qHL2bias, shift: qHL2shift) }
            for x in 16..<32 { base[rowOffset + x] = applyQSM(base[rowOffset + x], mul: qHL1mul, bias: qHL1bias, shift: qHL1shift) }
        }
        
        for y in 4..<8 {
            let rowOffset = y * 32
            for x in 0..<8   { base[rowOffset + x] = applyQSM(base[rowOffset + x], mul: qHL3mul, bias: qHL3bias, shift: qHL3shift) }
            for x in 8..<16  { base[rowOffset + x] = applyQSM(base[rowOffset + x], mul: qHL2mul, bias: qHL2bias, shift: qHL2shift) }
            for x in 16..<32 { base[rowOffset + x] = applyQSM(base[rowOffset + x], mul: qHL1mul, bias: qHL1bias, shift: qHL1shift) }
        }
        
        for y in 8..<16 {
            let rowOffset = y * 32
            for x in 0..<16  { base[rowOffset + x] = applyQSM(base[rowOffset + x], mul: qHL2mul, bias: qHL2bias, shift: qHL2shift) }
            for x in 16..<32 { base[rowOffset + x] = applyQSM(base[rowOffset + x], mul: qHL1mul, bias: qHL1bias, shift: qHL1shift) }
        }
        
        for y in 16..<32 {
            let rowOffset = y * 32
            for x in 0..<32  { base[rowOffset + x] = applyQSM(base[rowOffset + x], mul: qHL1mul, bias: qHL1bias, shift: qHL1shift) }
        }
    }

@inline(__always)
func dequantizeCascaded32(block: inout BlockView, qt: QuantizationTable, isChroma: Bool) {
    let step = Int(qt.step)
    let sLL3 = isChroma ? max(1, step * 3 / 4) : max(1, step)
    let sHL3 = isChroma ? max(1, step * 4 / 4) : max(1, step * 5 / 4)
    let sHL2 = isChroma ? max(1, step * 5 / 4) : max(1, step * 6 / 4)
    let sHL1 = isChroma ? max(1, step * 6 / 4) : max(1, step * 8 / 4)

    let qLL3 = Quantizer(step: sLL3, roundToNearest: true)
    let qHL3 = Quantizer(step: sHL3, roundToNearest: true)
    let qHL2 = Quantizer(step: sHL2, roundToNearest: true)
    let qHL1 = Quantizer(step: sHL1, roundToNearest: true)

    let base = block.base

    let stepLL3 = Int32(qLL3.step)
    let stepHL3 = Int32(qHL3.step)
    let stepHL2 = Int32(qHL2.step)
    let stepHL1 = Int32(qHL1.step)

    @inline(__always)
    func applyDQLL3(_ v: Int16) -> Int16 {
        return Int16(clamping: Int32(v) &* stepLL3)
    }
    
    @inline(__always)
    func applyDQSM(_ v: Int16, s: Int32) -> Int16 {
        let uVal = UInt16(bitPattern: v)
        let decodedUInt = ((uVal &>> 1) ^ (0 &- (uVal & 1)))
        return Int16(clamping: Int32(Int16(bitPattern: decodedUInt)) &* s)
    }

    for y in 0..<4 {
            let rowOffset = y * 32
            for x in 0..<4   { base[rowOffset + x] = applyDQLL3(base[rowOffset + x]) }
            for x in 4..<8   { base[rowOffset + x] = applyDQSM(base[rowOffset + x], s: stepHL3) }
            for x in 8..<16  { base[rowOffset + x] = applyDQSM(base[rowOffset + x], s: stepHL2) }
            for x in 16..<32 { base[rowOffset + x] = applyDQSM(base[rowOffset + x], s: stepHL1) }
        }
        
        for y in 4..<8 {
            let rowOffset = y * 32
            for x in 0..<8   { base[rowOffset + x] = applyDQSM(base[rowOffset + x], s: stepHL3) }
            for x in 8..<16  { base[rowOffset + x] = applyDQSM(base[rowOffset + x], s: stepHL2) }
            for x in 16..<32 { base[rowOffset + x] = applyDQSM(base[rowOffset + x], s: stepHL1) }
        }
        
        for y in 8..<16 {
            let rowOffset = y * 32
            for x in 0..<16  { base[rowOffset + x] = applyDQSM(base[rowOffset + x], s: stepHL2) }
            for x in 16..<32 { base[rowOffset + x] = applyDQSM(base[rowOffset + x], s: stepHL1) }
        }
        
        for y in 16..<32 {
            let rowOffset = y * 32
            for x in 0..<32  { base[rowOffset + x] = applyDQSM(base[rowOffset + x], s: stepHL1) }
        }
    }

