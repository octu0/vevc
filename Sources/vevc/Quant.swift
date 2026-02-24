// MARK: - Quantization

struct Quantizer: Sendable {
    public let step: Int16
    public let mul: Int32
    public let bias: Int32
    public let shift: Int16 = 16
    
    public init(step: Int, roundToNearest: Bool = false) {
        self.step = Int16(step)
        self.mul = Int32((1 << 16) / step)
        self.bias = Int32(roundToNearest ? (1 << 15) : 0)
    }
}

struct QuantizationTable: Sendable {
    public let step: Int16
    public let qLow: Quantizer
    public let qMid: Quantizer
    public let qHigh: Quantizer
    
    public init(baseStep: Int) {
        let s = max(1, min(baseStep, 32767))
        self.step = Int16(s)
        self.qLow  = Quantizer(step: s, roundToNearest: true)
        self.qMid  = Quantizer(step: s * 2, roundToNearest: false)
        self.qHigh = Quantizer(step: s * 4, roundToNearest: false)
    }
}

@inline(__always)
func quantizeLow(_ block: inout BlockView, qt: QuantizationTable) {
    quantize(&block, q: qt.qLow)
}

@inline(__always)
func quantizeLowSignedMapping(_ block: inout BlockView, qt: QuantizationTable) {
    quantizeSignedMapping(&block, q: qt.qLow)
}

@inline(__always)
func quantizeMid(_ block: inout BlockView, qt: QuantizationTable) {
    quantize(&block, q: qt.qMid)
}

@inline(__always)
func quantizeMidSignedMapping(_ block: inout BlockView, qt: QuantizationTable) {
    quantizeSignedMapping(&block, q: qt.qMid)
}

@inline(__always)
func quantizeHigh(_ block: inout BlockView, qt: QuantizationTable) {
    quantize(&block, q: qt.qHigh)
}

@inline(__always)
func quantizeHighSignedMapping(_ block: inout BlockView, qt: QuantizationTable) {
    quantizeSignedMapping(&block, q: qt.qHigh)
}

@inline(__always)
internal func quantize(_ block: inout BlockView, q: Quantizer) {
    #if arch(arm64) || arch(x86_64) || arch(wasm32)
    quantizeSIMD(&block, q: q)
    #else
    quantizeScalar(&block, q: q)
    #endif
}

@inline(__always)
internal func quantizeSignedMapping(_ block: inout BlockView, q: Quantizer) {
    #if arch(arm64) || arch(x86_64) || arch(wasm32)
    quantizeSIMDSignedMapping(&block, q: q)
    #else
    quantizeScalarSignedMapping(&block, q: q)
    #endif
}

// MARK: - Quantization SIMD

#if arch(arm64) || arch(x86_64) || arch(wasm32)

@inline(__always)
private func performQuantizeSIMD8(_ vec: SIMD8<Int16>, mul: Int32, shift: Int32, bias: Int32) -> SIMD8<Int16> {
    let zero = SIMD8<Int16>.zero
    let isNeg = vec .< zero
    let absVec = vec.replacing(with: 0 &- vec, where: isNeg)
    
    let low32 = SIMD4<Int32>(
        Int32(absVec[0]), Int32(absVec[1]), Int32(absVec[2]), Int32(absVec[3])
    )
    let high32 = SIMD4<Int32>(
        Int32(absVec[4]), Int32(absVec[5]), Int32(absVec[6]), Int32(absVec[7])
    )
    
    let mulVec = SIMD4<Int32>(repeating: mul)
    let shiftVec = SIMD4<Int32>(repeating: shift)
    let biasVec = SIMD4<Int32>(repeating: bias)
    
    let resLow32 = ((low32 &* mulVec) &+ biasVec) &>> shiftVec
    let resHigh32 = ((high32 &* mulVec) &+ biasVec) &>> shiftVec
    
    let res = SIMD8<Int16>(
        Int16(resLow32[0]), Int16(resLow32[1]), Int16(resLow32[2]), Int16(resLow32[3]),
        Int16(resHigh32[0]), Int16(resHigh32[1]), Int16(resHigh32[2]), Int16(resHigh32[3])
    )
    
    return res.replacing(with: 0 &- res, where: isNeg)
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
        let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        let res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func quantizeSIMD16(_ block: inout BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        // 0
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        var res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        // 8
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 8, count: 8))
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr + 8).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func quantizeSIMD32(_ block: inout BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        // 0
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        var res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        // 8
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 8, count: 8))
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr + 8).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        // 16
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 16, count: 8))
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr + 16).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        // 24
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 24, count: 8))
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        UnsafeMutableRawPointer(ptr + 24).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
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
            let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + i, count: 8))
            let res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
            let rawPtr = UnsafeMutableRawPointer(ptr + i).assumingMemoryBound(to: SIMD8<Int16>.self)
            rawPtr.pointee = res
            i += 8
        }
        while i < block.width {
            let val = Int32(ptr[i])
            let absVal = abs(val)
            let qVal = ((absVal &* mul) &+ bias) &>> shift
            ptr[i] = Int16(val < 0 ? -qVal : qVal)
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
        let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        let res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        let mask = (res &<< 1) ^ (res &>> 15)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
    }
}

@inline(__always)
private func quantizeSIMDSignedMapping16(_ block: inout BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        // 0
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        var res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        var mask = (res &<< 1) ^ (res &>> 15)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
        // 8
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 8, count: 8))
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        mask = (res &<< 1) ^ (res &>> 15)
        UnsafeMutableRawPointer(ptr + 8).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
    }
}

@inline(__always)
private func quantizeSIMDSignedMapping32(_ block: inout BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        // 0
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        var res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        var mask = (res &<< 1) ^ (res &>> 15)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
        // 8
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 8, count: 8))
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        mask = (res &<< 1) ^ (res &>> 15)
        UnsafeMutableRawPointer(ptr + 8).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
        // 16
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 16, count: 8))
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        mask = (res &<< 1) ^ (res &>> 15)
        UnsafeMutableRawPointer(ptr + 16).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
        // 24
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 24, count: 8))
        res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
        mask = (res &<< 1) ^ (res &>> 15)
        UnsafeMutableRawPointer(ptr + 24).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
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
            let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + i, count: 8))
            let res = performQuantizeSIMD8(vec, mul: mul, shift: shift, bias: bias)
            let mask = (res &<< 1) ^ (res &>> 15)
            let rawPtr = UnsafeMutableRawPointer(ptr + i).assumingMemoryBound(to: SIMD8<Int16>.self)
            rawPtr.pointee = mask
            i += 8
        }
        while i < block.width {
            let val = Int32(ptr[i])
            let absVal = abs(val)
            let qVal = ((absVal &* mul) &+ bias) &>> shift
            let v = Int16(val < 0 ? -qVal : qVal)
            ptr[i] = Int16(bitPattern: UInt16(bitPattern: (v &<< 1) ^ (v >> 15)))
            i += 1
        }
    }
}

#endif  // arch(arm64) || arch(x86_64)

// MARK: - Quantization Scalar (fallback)

@inline(__always)
internal func quantizeScalar(_ block: inout BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        for x in 0..<block.width {
            let val = Int32(ptr[x])
            let absVal = abs(val)
            let qVal = ((absVal &* mul) &+ bias) &>> shift
            ptr[x] = Int16(val < 0 ? -qVal : qVal)
        }
    }
}

@inline(__always)
internal func quantizeScalarSignedMapping(_ block: inout BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        for x in 0..<block.width {
            let val = Int32(ptr[x])
            let absVal = abs(val)
            let qVal = ((absVal &* mul) &+ bias) &>> shift
            let v = Int16(val < 0 ? -qVal : qVal)
            ptr[x] = Int16(bitPattern: UInt16(bitPattern: (v &<< 1) ^ (v >> 15)))
        }
    }
}

// MARK: - Dequantization

@inline(__always)
func dequantizeLow(_ block: inout BlockView, qt: QuantizationTable) {
    dequantize(&block, q: qt.qLow)
}

@inline(__always)
func dequantizeLowSignedMapping(_ block: inout BlockView, qt: QuantizationTable) {
    dequantizeSignedMapping(&block, q: qt.qLow)
}

@inline(__always)
func dequantizeMid(_ block: inout BlockView, qt: QuantizationTable) {
    dequantize(&block, q: qt.qMid)
}

@inline(__always)
func dequantizeMidSignedMapping(_ block: inout BlockView, qt: QuantizationTable) {
    dequantizeSignedMapping(&block, q: qt.qMid)
}

@inline(__always)
func dequantizeHigh(_ block: inout BlockView, qt: QuantizationTable) {
    dequantize(&block, q: qt.qHigh)
}

@inline(__always)
func dequantizeHighSignedMapping(_ block: inout BlockView, qt: QuantizationTable) {
    dequantizeSignedMapping(&block, q: qt.qHigh)
}

@inline(__always)
internal func dequantize(_ block: inout BlockView, q: Quantizer) {
    #if arch(arm64) || arch(x86_64) || arch(wasm32)
    dequantizeSIMD(&block, q: q)
    #else
    dequantizeScalar(&block, q: q)
    #endif
}

@inline(__always)
internal func dequantizeSignedMapping(_ block: inout BlockView, q: Quantizer) {
    #if arch(arm64) || arch(x86_64) || arch(wasm32)
    dequantizeSIMDSignedMapping(&block, q: q)
    #else
    dequantizeScalarSignedMapping(&block, q: q)
    #endif
}

// MARK: - Dequantization SIMD

#if arch(arm64) || arch(x86_64) || arch(wasm32)

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
        let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        let res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func dequantizeSIMD16(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        // 0
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        var res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        // 8
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 8, count: 8))
        res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr + 8).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func dequantizeSIMD32(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        // 0
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        var res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        // 8
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 8, count: 8))
        res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr + 8).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        // 16
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 16, count: 8))
        res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr + 16).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        // 24
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 24, count: 8))
        res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer(ptr + 24).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func dequantizeSIMDGeneric(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        var i = 0
        while (i + 8) <= block.width {
            let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + i, count: 8))
            let res = performDequantizeSIMD8(vec, step: step)
            let rawPtr = UnsafeMutableRawPointer(ptr + i).assumingMemoryBound(to: SIMD8<Int16>.self)
            rawPtr.pointee = res
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
        let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        let mask = 0 &- (vec & 1)
        let logicalShift = (vec &>> 1) & 0x7FFF
        let decoded = logicalShift ^ mask
        let res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func dequantizeSIMDSignedMapping16(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        // 0
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        var mask = 0 &- (vec & 1)
        var logicalShift = (vec &>> 1) & 0x7FFF
        var decoded = logicalShift ^ mask
        var res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        // 8
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 8, count: 8))
        mask = 0 &- (vec & 1)
        logicalShift = (vec &>> 1) & 0x7FFF
        decoded = logicalShift ^ mask
        res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr + 8).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func dequantizeSIMDSignedMapping32(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        // 0
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        var mask = 0 &- (vec & 1)
        var logicalShift = (vec &>> 1) & 0x7FFF
        var decoded = logicalShift ^ mask
        var res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        // 8
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 8, count: 8))
        mask = 0 &- (vec & 1)
        logicalShift = (vec &>> 1) & 0x7FFF
        decoded = logicalShift ^ mask
        res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr + 8).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        // 16
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 16, count: 8))
        mask = 0 &- (vec & 1)
        logicalShift = (vec &>> 1) & 0x7FFF
        decoded = logicalShift ^ mask
        res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr + 16).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        // 24
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + 24, count: 8))
        mask = 0 &- (vec & 1)
        logicalShift = (vec &>> 1) & 0x7FFF
        decoded = logicalShift ^ mask
        res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr + 24).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func dequantizeSIMDSignedMappingGeneric(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        var i = 0
        while (i + 8) <= block.width {
            let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr + i, count: 8))
            let mask = 0 &- (vec & 1)
            let logicalShift = (vec &>> 1) & 0x7FFF
            let decoded = logicalShift ^ mask
            
            let res = performDequantizeSIMD8(decoded, step: step)
            let rawPtr = UnsafeMutableRawPointer(ptr + i).assumingMemoryBound(to: SIMD8<Int16>.self)
            rawPtr.pointee = res
            i += 8
        }
        while i < block.width {
            let uVal = UInt16(bitPattern: ptr[i])
            let decodedUInt = (uVal >> 1) ^ (0 &- (uVal & 1))
            let decoded = Int16(bitPattern: decodedUInt)
            ptr[i] = Int16(clamping: Int32(decoded) &* step)
            i += 1
        }
    }
}

#endif  // arch(arm64) || arch(x86_64)

// MARK: - Dequantization Scalar (fallback)

@inline(__always)
internal func dequantizeScalar(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        for x in 0..<block.width {
            let val = Int32(ptr[x])
            ptr[x] = Int16(clamping: (val &* step))
        }
    }
}

@inline(__always)
internal func dequantizeScalarSignedMapping(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        for x in 0..<block.width {
            let uVal = UInt16(bitPattern: ptr[x])
            let decodedUInt = (uVal >> 1) ^ (0 &- (uVal & 1))
            let decoded = Int16(bitPattern: decodedUInt)
            ptr[x] = Int16(clamping: Int32(decoded) &* step)
        }
    }
}
