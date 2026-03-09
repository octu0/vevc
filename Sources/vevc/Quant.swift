import Foundation

struct Quantizer: Sendable {
    public let step: Int16
    public let mul: Int32
    public let bias: Int32
    public let shift: Int32 = 16

    public init(step: Int, roundToNearest: Bool = false) {
        let s = max(1, step)
        self.step = Int16(s)
        // Use 16-bit fractional precision (1 << 16) to represent the multiplier (1 / step).
        // This allows quantization to be performed using integer multiplication and bit-shifting.
        self.mul = Int32((Double((1 * (1 << 16))) / Double(s)).rounded())
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
        self.qMid  = Quantizer(step: (s * 2), roundToNearest: false)
        self.qHigh = Quantizer(step: (s * 4), roundToNearest: false)
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

@inline(__always)
private func clampToI16(_ v: Int32) -> Int16 {
    if v < -32768 {
        return -32768
    }
    if 32767 < v {
        return 32767
    }
    return Int16(truncatingIfNeeded: v)
}

#if arch(arm64) || arch(x86_64) || arch(wasm32)

@inline(__always)
private func performQuantizeSIMD8(_ vec: SIMD8<Int16>, q: Quantizer) -> SIMD8<Int16> {
    // Perform quantization using Double precision to ensure accurate rounding as required by tests.
    // While slower than fixed-point, this guarantees correct behavior for all quantization steps.
    var res = SIMD8<Int16>()
    let s = Double(q.step)
    for i in 0..<8 {
        let v = Int32(vec[i+0])
        let absV = Double(abs(v))
        let qVal: Int32
        if q.bias != 0 {
            qVal = Int32((absV / s).rounded())
        } else {
            qVal = Int32(floor((absV / s)))
        }
        let signedQ = (v < 0 ? ((-1) * qVal) : qVal)
        res[i+0] = Int16(truncatingIfNeeded: signedQ)
    }
    return res
}

@inline(__always)
private func quantizeSIMD(_ block: inout BlockView, q: Quantizer) {
    let w = block.width
    if w == 8 {
        quantizeSIMD8(&block, q: q)
    } else {
        if w == 16 {
            quantizeSIMD16(&block, q: q)
        } else {
            if w == 32 {
                quantizeSIMD32(&block, q: q)
            } else {
                quantizeSIMDGeneric(&block, q: q)
            }
        }
    }
}

@inline(__always)
private func quantizeSIMD8(_ block: inout BlockView, q: Quantizer) {
    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        let res = performQuantizeSIMD8(vec, q: q)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func quantizeSIMD16(_ block: inout BlockView, q: Quantizer) {
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 0), count: 8))
        var res = performQuantizeSIMD8(vec, q: q)
        UnsafeMutableRawPointer((ptr + 0)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 8), count: 8))
        res = performQuantizeSIMD8(vec, q: q)
        UnsafeMutableRawPointer((ptr + 8)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func quantizeSIMD32(_ block: inout BlockView, q: Quantizer) {
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 0), count: 8))
        var res = performQuantizeSIMD8(vec, q: q)
        UnsafeMutableRawPointer((ptr + 0)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 8), count: 8))
        res = performQuantizeSIMD8(vec, q: q)
        UnsafeMutableRawPointer((ptr + 8)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 16), count: 8))
        res = performQuantizeSIMD8(vec, q: q)
        UnsafeMutableRawPointer((ptr + 16)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 24), count: 8))
        res = performQuantizeSIMD8(vec, q: q)
        UnsafeMutableRawPointer((ptr + 24)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func quantizeSIMDGeneric(_ block: inout BlockView, q: Quantizer) {
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        var i = 0
        while ((i + 8) <= block.width) {
            let vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + i), count: 8))
            let res = performQuantizeSIMD8(vec, q: q)
            let rawPtr = UnsafeMutableRawPointer((ptr + i)).assumingMemoryBound(to: SIMD8<Int16>.self)
            rawPtr.pointee = res
            i += 8
        }
        while i < block.width {
            let val = Int32(ptr[i+0])
            let absV = Double(abs(val))
            let s = Double(q.step)
            let qVal: Int32
            if q.bias != 0 {
                qVal = Int32((absV / s).rounded())
            } else {
                qVal = Int32(floor((absV / s)))
            }
            ptr[i+0] = Int16(truncatingIfNeeded: (val < 0 ? ((-1) * qVal) : qVal))
            i += 1
        }
    }
}

@inline(__always)
private func quantizeSIMDSignedMapping(_ block: inout BlockView, q: Quantizer) {
    let w = block.width
    if w == 8 {
        quantizeSIMDSignedMapping8(&block, q: q)
    } else {
        if w == 16 {
            quantizeSIMDSignedMapping16(&block, q: q)
        } else {
            if w == 32 {
                quantizeSIMDSignedMapping32(&block, q: q)
            } else {
                quantizeSIMDSignedMappingGeneric(&block, q: q)
            }
        }
    }
}

@inline(__always)
private func quantizeSIMDSignedMapping8(_ block: inout BlockView, q: Quantizer) {
    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        let res = performQuantizeSIMD8(vec, q: q)
        let mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
    }
}

@inline(__always)
private func quantizeSIMDSignedMapping16(_ block: inout BlockView, q: Quantizer) {
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 0), count: 8))
        var res = performQuantizeSIMD8(vec, q: q)
        var mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer((ptr + 0)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 8), count: 8))
        res = performQuantizeSIMD8(vec, q: q)
        mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer((ptr + 8)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
    }
}

@inline(__always)
private func quantizeSIMDSignedMapping32(_ block: inout BlockView, q: Quantizer) {
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 0), count: 8))
        var res = performQuantizeSIMD8(vec, q: q)
        var mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer((ptr + 0)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 8), count: 8))
        res = performQuantizeSIMD8(vec, q: q)
        mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer((ptr + 8)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 16), count: 8))
        res = performQuantizeSIMD8(vec, q: q)
        mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer((ptr + 16)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 24), count: 8))
        res = performQuantizeSIMD8(vec, q: q)
        mask = ((res &<< 1) ^ (res &>> 15))
        UnsafeMutableRawPointer((ptr + 24)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = mask
    }
}

@inline(__always)
private func quantizeSIMDSignedMappingGeneric(_ block: inout BlockView, q: Quantizer) {
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        var i = 0
        while ((i + 8) <= block.width) {
            let vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + i), count: 8))
            let res = performQuantizeSIMD8(vec, q: q)
            let mask = ((res &<< 1) ^ (res &>> 15))
            let rawPtr = UnsafeMutableRawPointer((ptr + i)).assumingMemoryBound(to: SIMD8<Int16>.self)
            rawPtr.pointee = mask
            i += 8
        }
        while i < block.width {
            let val = Int32(ptr[i+0])
            let absV = Double(abs(val))
            let s = Double(q.step)
            let qVal: Int32
            if q.bias != 0 {
                qVal = Int32((absV / s).rounded())
            } else {
                qVal = Int32(floor((absV / s)))
            }
            let v = Int16(truncatingIfNeeded: (val < 0 ? ((-1) * qVal) : qVal))
            ptr[i+0] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v >> 15))))
            i += 1
        }
    }
}

#endif

@inline(__always)
internal func quantizeScalar(_ block: inout BlockView, q: Quantizer) {
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        for x in 0..<block.width {
            let val = Int32(ptr[x+0])
            let absV = Double(abs(val))
            let s = Double(q.step)
            let qVal: Int32
            if q.bias != 0 {
                qVal = Int32((absV / s).rounded())
            } else {
                qVal = Int32(floor((absV / s)))
            }
            ptr[x+0] = Int16(truncatingIfNeeded: (val < 0 ? ((-1) * qVal) : qVal))
        }
    }
}

@inline(__always)
internal func quantizeScalarSignedMapping(_ block: inout BlockView, q: Quantizer) {
    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        for x in 0..<block.width {
            let val = Int32(ptr[x+0])
            let absV = Double(abs(val))
            let s = Double(q.step)
            let qVal: Int32
            if q.bias != 0 {
                qVal = Int32((absV / s).rounded())
            } else {
                qVal = Int32(floor((absV / s)))
            }
            let v = Int16(truncatingIfNeeded: (val < 0 ? ((-1) * qVal) : qVal))
            ptr[x+0] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v >> 15))))
        }
    }
}

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

#if arch(arm64) || arch(x86_64) || arch(wasm32)

@inline(__always)
private func performDequantizeSIMD8(_ vec: SIMD8<Int16>, step: Int32) -> SIMD8<Int16> {
    // Perform dequantization using scalar loops within SIMD-structured code paths.
    // This approach maintains the requested "Strict Explicit Style" while providing
    // predictable compiler optimization across all elements of the vector.
    var res = SIMD8<Int16>()
    for i in 0..<8 {
        let v = Int32(vec[i+0])
        let d = (v * step)
        res[i+0] = clampToI16(d)
    }
    return res
}

@inline(__always)
private func dequantizeSIMD(_ block: inout BlockView, q: Quantizer) {
    let w = block.width
    if w == 8 {
        dequantizeSIMD8(&block, q: q)
    } else {
        if w == 16 {
            dequantizeSIMD16(&block, q: q)
        } else {
            if w == 32 {
                dequantizeSIMD32(&block, q: q)
            } else {
                dequantizeSIMDGeneric(&block, q: q)
            }
        }
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
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 0), count: 8))
        var res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer((ptr + 0)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 8), count: 8))
        res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer((ptr + 8)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func dequantizeSIMD32(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 0), count: 8))
        var res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer((ptr + 0)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 8), count: 8))
        res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer((ptr + 8)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 16), count: 8))
        res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer((ptr + 16)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 24), count: 8))
        res = performDequantizeSIMD8(vec, step: step)
        UnsafeMutableRawPointer((ptr + 24)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func dequantizeSIMDGeneric(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)

    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        var i = 0
        while ((i + 8) <= block.width) {
            let vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + i), count: 8))
            let res = performDequantizeSIMD8(vec, step: step)
            let rawPtr = UnsafeMutableRawPointer((ptr + i)).assumingMemoryBound(to: SIMD8<Int16>.self)
            rawPtr.pointee = res
            i += 8
        }
        while i < block.width {
            let val = Int32(ptr[i+0])
            let res = (val * step)
            ptr[i+0] = clampToI16(res)
            i += 1
        }
    }
}

@inline(__always)
private func dequantizeSIMDSignedMapping(_ block: inout BlockView, q: Quantizer) {
    let w = block.width
    if w == 8 {
        dequantizeSIMDSignedMapping8(&block, q: q)
    } else {
        if w == 16 {
            dequantizeSIMDSignedMapping16(&block, q: q)
        } else {
            if w == 32 {
                dequantizeSIMDSignedMapping32(&block, q: q)
            } else {
                dequantizeSIMDSignedMappingGeneric(&block, q: q)
            }
        }
    }
}

@inline(__always)
private func dequantizeSIMDSignedMapping8(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        let vec = SIMD8<Int16>(UnsafeBufferPointer(start: ptr, count: 8))
        let mask = (0 &- (vec & 1))
        let logicalShift = ((vec &>> 1) & 0x7FFF)
        let decoded = (logicalShift ^ mask)
        let res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func dequantizeSIMDSignedMapping16(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 0), count: 8))
        var mask = (0 &- (vec & 1))
        var logicalShift = ((vec &>> 1) & 0x7FFF)
        var decoded = (logicalShift ^ mask)
        var res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer((ptr + 0)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 8), count: 8))
        mask = (0 &- (vec & 1))
        logicalShift = ((vec &>> 1) & 0x7FFF)
        decoded = (logicalShift ^ mask)
        res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer((ptr + 8)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func dequantizeSIMDSignedMapping32(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        var vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 0), count: 8))
        var mask = (0 &- (vec & 1))
        var logicalShift = ((vec &>> 1) & 0x7FFF)
        var decoded = (logicalShift ^ mask)
        var res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer((ptr + 0)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 8), count: 8))
        mask = (0 &- (vec & 1))
        logicalShift = ((vec &>> 1) & 0x7FFF)
        decoded = (logicalShift ^ mask)
        res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer((ptr + 8)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 16), count: 8))
        mask = (0 &- (vec & 1))
        logicalShift = ((vec &>> 1) & 0x7FFF)
        decoded = (logicalShift ^ mask)
        res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer((ptr + 16)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
        vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + 24), count: 8))
        mask = (0 &- (vec & 1))
        logicalShift = ((vec &>> 1) & 0x7FFF)
        decoded = (logicalShift ^ mask)
        res = performDequantizeSIMD8(decoded, step: step)
        UnsafeMutableRawPointer((ptr + 24)).assumingMemoryBound(to: SIMD8<Int16>.self).pointee = res
    }
}

@inline(__always)
private func dequantizeSIMDSignedMappingGeneric(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)

    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        var i = 0
        while ((i + 8) <= block.width) {
            let vec = SIMD8<Int16>(UnsafeBufferPointer(start: (ptr + i), count: 8))
            let mask = (0 &- (vec & 1))
            let logicalShift = ((vec &>> 1) & 0x7FFF)
            let decoded = (logicalShift ^ mask)

            let res = performDequantizeSIMD8(decoded, step: step)
            let rawPtr = UnsafeMutableRawPointer((ptr + i)).assumingMemoryBound(to: SIMD8<Int16>.self)
            rawPtr.pointee = res
            i += 8
        }
        while i < block.width {
            let uVal = UInt16(bitPattern: ptr[i+0])
            let decodedUInt = ((uVal >> 1) ^ (0 &- (uVal & 1)))
            let decoded = Int16(bitPattern: decodedUInt)
            let res = (Int32(decoded) * step)
            ptr[i+0] = clampToI16(res)
            i += 1
        }
    }
}

#endif

@inline(__always)
internal func dequantizeScalar(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)

    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        for x in 0..<block.width {
            let val = Int32(ptr[x+0])
            let res = (val * step)
            ptr[x+0] = clampToI16(res)
        }
    }
}

@inline(__always)
internal func dequantizeScalarSignedMapping(_ block: inout BlockView, q: Quantizer) {
    let step = Int32(q.step)

    for y in 0..<block.height {
        let ptr = block.rowPointer(y: y)
        for x in 0..<block.width {
            let uVal = UInt16(bitPattern: ptr[x+0])
            let decodedUInt = ((uVal >> 1) ^ (0 &- (uVal & 1)))
            let decoded = Int16(bitPattern: decodedUInt)
            let res = (Int32(decoded) * step)
            ptr[x+0] = clampToI16(res)
        }
    }
}
