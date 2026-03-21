// MARK: - Quantization

struct Quantizer: Sendable {
    let step: Int16
    let mul: Int32
    let bias: Int32
    let shift: Int16 = 16
    
    init(step: Int, roundToNearest: Bool = false, deadZoneRatio: Double = 0.0) {
        self.step = Int16(step)
        self.mul = Int32((1 << 16) / step)
        var b: Int32 = 0
        if roundToNearest {
            b = Int32(1 << 15)
        } else if deadZoneRatio != 0.0 {
            // positive deadZoneRatio narrows the dead zone (towards round-to-nearest)
            // negative deadZoneRatio widens the dead zone (more values become 0)
            b = Int32(Double(1 << 16) * deadZoneRatio)
        }
        self.bias = b
    }
}

struct QuantizationTable: Sendable {
    let step: Int16
    let qLow: Quantizer
    let qMid: Quantizer
    let qHigh: Quantizer
    
    init(baseStep: Int, isChroma: Bool = false) {
        let s = max(1, min(baseStep, 32767))
        self.step = Int16(s)
        
        if isChroma {
            self.qLow = Quantizer(step: Int(min(4, max(1, baseStep / 8))), roundToNearest: true)
            // Widen deadzone slightly for mid/high chroma to drop barely-visible color variations
            self.qMid = Quantizer(step: Int(min(16, max(1, baseStep))), deadZoneRatio: -0.1)
            self.qHigh = Quantizer(step: Int(min(32, max(1, baseStep * 2))), deadZoneRatio: -0.2)
        } else {
            // Luma LL band holds structural integrity, keep it very high quality
            self.qLow = Quantizer(step: Int(min(8, max(1, baseStep / 6))), roundToNearest: true)
            // Widen deadzone for higher luma frequencies
            self.qMid = Quantizer(step: Int(min(256, max(1, baseStep))), deadZoneRatio: -0.1)
            self.qHigh = Quantizer(step: Int(min(512, max(1, Int(Double(baseStep) * 2.5)))), deadZoneRatio: -0.3)
        }
    }
}


@inline(__always)
internal func quantize(_ block: inout BlockView, q: Quantizer) {
    quantizeSIMD(&block, q: q)
}

@inline(__always)
internal func quantizeSignedMapping(_ block: inout BlockView, q: Quantizer) {
    quantizeSIMDSignedMapping(&block, q: q)
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
private func quantizeSIMD(_ block: inout BlockView, q: Quantizer) {
    switch block.width {
    case 8:  quantizeSIMD8(&block, q: q)
    case 16: quantizeSIMD16(&block, q: q)
    case 32: quantizeSIMD32(&block, q: q)
    default: quantizeSIMDGeneric(&block, q: q)
    }
}

@inline(__always)
private func quantizeSIMD8(_ block: inout BlockView, q: Quantizer) {
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
private func quantizeSIMD16(_ block: inout BlockView, q: Quantizer) {
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
private func quantizeSIMD32(_ block: inout BlockView, q: Quantizer) {
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
private func quantizeSIMDGeneric(_ block: inout BlockView, q: Quantizer) {
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
private func quantizeSIMDSignedMapping(_ block: inout BlockView, q: Quantizer) {
    switch block.width {
    case 8:  quantizeSIMDSignedMapping8(&block, q: q)
    case 16: quantizeSIMDSignedMapping16(&block, q: q)
    case 32: quantizeSIMDSignedMapping32(&block, q: q)
    default: quantizeSIMDSignedMappingGeneric(&block, q: q)
    }
}

@inline(__always)
private func quantizeSIMDSignedMapping8(_ block: inout BlockView, q: Quantizer) {
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
private func quantizeSIMDSignedMapping16(_ block: inout BlockView, q: Quantizer) {
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
private func quantizeSIMDSignedMapping32(_ block: inout BlockView, q: Quantizer) {
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
private func quantizeSIMDSignedMappingGeneric(_ block: inout BlockView, q: Quantizer) {
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


// MARK: - Dequantization


@inline(__always)
internal func dequantize(_ block: inout BlockView, q: Quantizer) {
    dequantizeSIMD(&block, q: q)
}

@inline(__always)
internal func dequantizeSignedMapping(_ block: inout BlockView, q: Quantizer) {
    dequantizeSIMDSignedMapping(&block, q: q)
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
private func performDequantizeSIMDSignedMapping8(_ vec: SIMD8<Int16>, step: Int32) -> SIMD8<Int16> {
    let dVal = unsafeBitCast(vec, to: SIMD8<UInt16>.self)
    let one = SIMD8<UInt16>(repeating: 1)
    let signMask = ~(dVal & one) &+ one
    let sign = unsafeBitCast(signMask, to: SIMD8<Int16>.self)
    let orig = unsafeBitCast(dVal &>> 1, to: SIMD8<Int16>.self) ^ sign
    
    return performDequantizeSIMD8(orig, step: step)
}

@inline(__always)
private func dequantizeSIMD(_ block: inout BlockView, q: Quantizer) {
    switch block.width {
    case 8:  dequantizeSIMD8(&block, q: q)
    case 16: dequantizeSIMD16(&block, q: q)
    case 32: dequantizeSIMD32(&block, q: q)
    default: dequantizeSIMDGeneric(&block, q: q)
    }
}

@inline(__always)
private func dequantizeSIMD8(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        let res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr).storeBytes(of: res, as: SIMD8<Int16>.self)
    }
}

@inline(__always)
private func dequantizeSIMD16(_ block: inout BlockView, q: Quantizer) {
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
private func dequantizeSIMD32(_ block: inout BlockView, q: Quantizer) {
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
private func dequantizeSIMDGeneric(_ block: inout BlockView, q: Quantizer) {
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
            ptr[i] = Int16(clamping: res)
            i += 1
        }
    }
}

@inline(__always)
private func dequantizeSIMDSignedMapping(_ block: inout BlockView, q: Quantizer) {
    switch block.width {
    case 8:  dequantizeSIMDSignedMapping8(&block, q: q)
    case 16: dequantizeSIMDSignedMapping16(&block, q: q)
    case 32: dequantizeSIMDSignedMapping32(&block, q: q)
    default: dequantizeSIMDSignedMappingGeneric(&block, q: q)
    }
}

@inline(__always)
private func dequantizeSIMDSignedMapping8(_ block: inout BlockView, q: Quantizer) {
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
private func dequantizeSIMDSignedMapping16(_ block: inout BlockView, q: Quantizer) {
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
private func dequantizeSIMDSignedMapping32(_ block: inout BlockView, q: Quantizer) {
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
private func dequantizeSIMDSignedMappingGeneric(_ block: inout BlockView, q: Quantizer) {
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
            ptr[i] = Int16(clamping: (Int32(decoded) &* step))
            i += 1
        }
    }
}


