import Foundation
import simd

private let cA: Int32 = -25987
private let cB: Int32 = -868
private let cG: Int32 = 14466
private let cD: Int32 = 7266

@inline(__always)
private func mulShift8(_ c: Int32, _ v: SIMD8<Int32>) -> SIMD8<Int32> {
    return (v &* c &+ 8192) &>> 14
}

@inline(__always)
private func mulShift4(_ c: Int32, _ v: SIMD4<Int32>) -> SIMD4<Int32> {
    return (v &* c &+ 8192) &>> 14
}

// MARK: - CDF 9/7 1D Core

@inline(__always)
private func applyCDF97Lift1D_8(low: UnsafeMutableBufferPointer<SIMD8<Int32>>, high: UnsafeMutableBufferPointer<SIMD8<Int32>>, half: Int) {
    for i in 0..<half {
        let l_i = low[i]; let l_next = (i + 1 < half) ? low[i+1] : low[i]
        high[i] &+= mulShift8(cA, l_i &+ l_next)
    }
    for i in 0..<half {
        let h_prev = (i > 0) ? high[i-1] : high[0]; let h_i = high[i]
        low[i] &+= mulShift8(cB, h_prev &+ h_i)
    }
    for i in 0..<half {
        let l_i = low[i]; let l_next = (i + 1 < half) ? low[i+1] : low[i]
        high[i] &+= mulShift8(cG, l_i &+ l_next)
    }
    for i in 0..<half {
        let h_prev = (i > 0) ? high[i-1] : high[0]; let h_i = high[i]
        low[i] &+= mulShift8(cD, h_prev &+ h_i)
    }
}

@inline(__always)
private func applyInverseCDF97Lift1D_8(low: UnsafeMutableBufferPointer<SIMD8<Int32>>, high: UnsafeMutableBufferPointer<SIMD8<Int32>>, half: Int) {
    for i in 0..<half {
        let h_prev = (i > 0) ? high[i-1] : high[0]; let h_i = high[i]
        low[i] &-= mulShift8(cD, h_prev &+ h_i)
    }
    for i in 0..<half {
        let l_i = low[i]; let l_next = (i + 1 < half) ? low[i+1] : low[i]
        high[i] &-= mulShift8(cG, l_i &+ l_next)
    }
    for i in 0..<half {
        let h_prev = (i > 0) ? high[i-1] : high[0]; let h_i = high[i]
        low[i] &-= mulShift8(cB, h_prev &+ h_i)
    }
    for i in 0..<half {
        let l_i = low[i]; let l_next = (i + 1 < half) ? low[i+1] : low[i]
        high[i] &-= mulShift8(cA, l_i &+ l_next)
    }
}

// MARK: - LeGall 5/3 1D Core

@inline(__always)
private func applyLeGall53Lift1D_8(low: UnsafeMutableBufferPointer<SIMD8<Int32>>, high: UnsafeMutableBufferPointer<SIMD8<Int32>>, half: Int) {
    for i in 0..<half {
        let l_i = low[i]; let l_next = (i + 1 < half) ? low[i+1] : low[i]
        high[i] &-= (l_i &+ l_next) &>> 1
    }
    for i in 0..<half {
        let h_prev = (i > 0) ? high[i-1] : high[0]; let h_i = high[i]
        low[i] &+= (h_prev &+ h_i &+ 2) &>> 2
    }
}

@inline(__always)
private func applyInverseLeGall53Lift1D_8(low: UnsafeMutableBufferPointer<SIMD8<Int32>>, high: UnsafeMutableBufferPointer<SIMD8<Int32>>, half: Int) {
    for i in 0..<half {
        let h_prev = (i > 0) ? high[i-1] : high[0]; let h_i = high[i]
        low[i] &-= (h_prev &+ h_i &+ 2) &>> 2
    }
    for i in 0..<half {
        let l_i = low[i]; let l_next = (i + 1 < half) ? low[i+1] : low[i]
        high[i] &+= (l_i &+ l_next) &>> 1
    }
}

@inline(__always)
private func applyLeGall53Lift1D_4(low: UnsafeMutableBufferPointer<SIMD4<Int32>>, high: UnsafeMutableBufferPointer<SIMD4<Int32>>, half: Int) {
    for i in 0..<half {
        let l_i = low[i]; let l_next = (i + 1 < half) ? low[i+1] : low[i]
        high[i] &-= (l_i &+ l_next) &>> 1
    }
    for i in 0..<half {
        let h_prev = (i > 0) ? high[i-1] : high[0]; let h_i = high[i]
        low[i] &+= (h_prev &+ h_i &+ 2) &>> 2
    }
}

@inline(__always)
private func applyInverseLeGall53Lift1D_4(low: UnsafeMutableBufferPointer<SIMD4<Int32>>, high: UnsafeMutableBufferPointer<SIMD4<Int32>>, half: Int) {
    for i in 0..<half {
        let h_prev = (i > 0) ? high[i-1] : high[0]; let h_i = high[i]
        low[i] &-= (h_prev &+ h_i &+ 2) &>> 2
    }
    for i in 0..<half {
        let l_i = low[i]; let l_next = (i + 1 < half) ? low[i+1] : low[i]
        high[i] &+= (l_i &+ l_next) &>> 1
    }
}

// MARK: - Load / Store Helpers

@inline(__always)
private func loadY8(_ base: UnsafePointer<Int16>, offset: Int, stride: Int) -> SIMD8<Int32> {
    let p = base + offset
    return SIMD8<Int32>(
        Int32(p[0]), Int32(p[stride]), Int32(p[2*stride]), Int32(p[3*stride]),
        Int32(p[4*stride]), Int32(p[5*stride]), Int32(p[6*stride]), Int32(p[7*stride])
    )
}

@inline(__always)
private func storeY8(_ base: UnsafeMutablePointer<Int16>, offset: Int, stride: Int, value: SIMD8<Int32>) {
    let p = base + offset
    let v = SIMD8<Int16>(truncatingIfNeeded: value)
    p[0] = v[0]; p[stride] = v[1]; p[2*stride] = v[2]; p[3*stride] = v[3]
    p[4*stride] = v[4]; p[5*stride] = v[5]; p[6*stride] = v[6]; p[7*stride] = v[7]
}

@inline(__always)
private func loadX8(_ base: UnsafePointer<Int16>, offset: Int) -> SIMD8<Int32> {
    let raw = UnsafeRawPointer(base + offset).loadUnaligned(as: SIMD8<Int16>.self)
    return SIMD8<Int32>(
        Int32(raw[0]), Int32(raw[1]), Int32(raw[2]), Int32(raw[3]),
        Int32(raw[4]), Int32(raw[5]), Int32(raw[6]), Int32(raw[7])
    )
}

@inline(__always)
private func storeX8(_ base: UnsafeMutablePointer<Int16>, offset: Int, value: SIMD8<Int32>) {
    let v = SIMD8<Int16>(truncatingIfNeeded: value)
    UnsafeMutableRawPointer(base + offset).storeBytes(of: v, as: SIMD8<Int16>.self)
}

@inline(__always)
private func loadY4(_ base: UnsafePointer<Int16>, offset: Int, stride: Int) -> SIMD4<Int32> {
    let p = base + offset
    return SIMD4<Int32>(
        Int32(p[0]), Int32(p[stride]), Int32(p[2*stride]), Int32(p[3*stride])
    )
}

@inline(__always)
private func storeY4(_ base: UnsafeMutablePointer<Int16>, offset: Int, stride: Int, value: SIMD4<Int32>) {
    let p = base + offset
    let v = SIMD4<Int16>(truncatingIfNeeded: value)
    p[0] = v[0]; p[stride] = v[1]; p[2*stride] = v[2]; p[3*stride] = v[3]
}

@inline(__always)
private func loadX4(_ base: UnsafePointer<Int16>, offset: Int) -> SIMD4<Int32> {
    let raw = UnsafeRawPointer(base + offset).loadUnaligned(as: SIMD4<Int16>.self)
    return SIMD4<Int32>(
        Int32(raw[0]), Int32(raw[1]), Int32(raw[2]), Int32(raw[3])
    )
}

@inline(__always)
private func storeX4(_ base: UnsafeMutablePointer<Int16>, offset: Int, value: SIMD4<Int32>) {
    let v = SIMD4<Int16>(truncatingIfNeeded: value)
    UnsafeMutableRawPointer(base + offset).storeBytes(of: v, as: SIMD4<Int16>.self)
}

// MARK: - Parametric Execution (X and Y passes)

@inline(__always)
private func runParametric8(_ base: UnsafeMutablePointer<Int16>, length: Int, count: Int, stride: Int,
                            isXPass: Bool, isInverse: Bool,
                            process: (UnsafeMutableBufferPointer<SIMD8<Int32>>, UnsafeMutableBufferPointer<SIMD8<Int32>>, Int) -> Void) {
    let half = length / 2
    withUnsafeTemporaryAllocation(of: SIMD8<Int32>.self, capacity: length) { buffer in
        let low = UnsafeMutableBufferPointer(start: buffer.baseAddress, count: half)
        let high = UnsafeMutableBufferPointer(start: buffer.baseAddress! + half, count: half)
        
        var c = 0
        while c < count {
            if isXPass {
                if isInverse {
                    for i in 0..<half {
                        low[i]  = loadY8(base, offset: i + c * stride, stride: stride)
                        high[i] = loadY8(base, offset: (half + i) + c * stride, stride: stride)
                    }
                    process(low, high, half)
                    for i in 0..<half {
                        storeY8(base, offset: (2*i) + c * stride, stride: stride, value: low[i])
                        storeY8(base, offset: (2*i + 1) + c * stride, stride: stride, value: high[i])
                    }
                } else {
                    for i in 0..<half {
                        low[i]  = loadY8(base, offset: (2*i) + c * stride, stride: stride)
                        high[i] = loadY8(base, offset: (2*i + 1) + c * stride, stride: stride)
                    }
                    process(low, high, half)
                    for i in 0..<half {
                        storeY8(base, offset: i + c * stride, stride: stride, value: low[i])
                        storeY8(base, offset: (half + i) + c * stride, stride: stride, value: high[i])
                    }
                }
            } else {
                if isInverse {
                    for i in 0..<half {
                        low[i]  = loadX8(base, offset: c + i * stride)
                        high[i] = loadX8(base, offset: c + (half + i) * stride)
                    }
                    process(low, high, half)
                    for i in 0..<half {
                        storeX8(base, offset: c + (2*i) * stride, value: low[i])
                        storeX8(base, offset: c + (2*i + 1) * stride, value: high[i])
                    }
                } else {
                    for i in 0..<half {
                        low[i]  = loadX8(base, offset: c + (2*i) * stride)
                        high[i] = loadX8(base, offset: c + (2*i + 1) * stride)
                    }
                    process(low, high, half)
                    for i in 0..<half {
                        storeX8(base, offset: c + i * stride, value: low[i])
                        storeX8(base, offset: c + (half + i) * stride, value: high[i])
                    }
                }
            }
            c += 8
        }
    }
}

@inline(__always)
private func runParametric4(_ base: UnsafeMutablePointer<Int16>, length: Int, count: Int, stride: Int,
                            isXPass: Bool, isInverse: Bool,
                            process: (UnsafeMutableBufferPointer<SIMD4<Int32>>, UnsafeMutableBufferPointer<SIMD4<Int32>>, Int) -> Void) {
    let half = length / 2
    withUnsafeTemporaryAllocation(of: SIMD4<Int32>.self, capacity: length) { buffer in
        let low = UnsafeMutableBufferPointer(start: buffer.baseAddress, count: half)
        let high = UnsafeMutableBufferPointer(start: buffer.baseAddress! + half, count: half)
        
        var c = 0
        while c < count {
            if isXPass {
                if isInverse {
                    for i in 0..<half {
                        low[i]  = loadY4(base, offset: i + c * stride, stride: stride)
                        high[i] = loadY4(base, offset: (half + i) + c * stride, stride: stride)
                    }
                    process(low, high, half)
                    for i in 0..<half {
                        storeY4(base, offset: (2*i) + c * stride, stride: stride, value: low[i])
                        storeY4(base, offset: (2*i + 1) + c * stride, stride: stride, value: high[i])
                    }
                } else {
                    for i in 0..<half {
                        low[i]  = loadY4(base, offset: (2*i) + c * stride, stride: stride)
                        high[i] = loadY4(base, offset: (2*i + 1) + c * stride, stride: stride)
                    }
                    process(low, high, half)
                    for i in 0..<half {
                        storeY4(base, offset: i + c * stride, stride: stride, value: low[i])
                        storeY4(base, offset: (half + i) + c * stride, stride: stride, value: high[i])
                    }
                }
            } else {
                if isInverse {
                    for i in 0..<half {
                        low[i]  = loadX4(base, offset: c + i * stride)
                        high[i] = loadX4(base, offset: c + (half + i) * stride)
                    }
                    process(low, high, half)
                    for i in 0..<half {
                        storeX4(base, offset: c + (2*i) * stride, value: low[i])
                        storeX4(base, offset: c + (2*i + 1) * stride, value: high[i])
                    }
                } else {
                    for i in 0..<half {
                        low[i]  = loadX4(base, offset: c + (2*i) * stride)
                        high[i] = loadX4(base, offset: c + (2*i + 1) * stride)
                    }
                    process(low, high, half)
                    for i in 0..<half {
                        storeX4(base, offset: c + i * stride, value: low[i])
                        storeX4(base, offset: c + (half + i) * stride, value: high[i])
                    }
                }
            }
            c += 4
        }
    }
}

// MARK: - Public API

public func cdf97LiftParametric(base: UnsafeMutablePointer<Int16>, length: Int, count: Int, stride: Int, isXPass: Bool, isInverse: Bool) {
    if isInverse {
        runParametric8(base, length: length, count: count, stride: stride, isXPass: isXPass, isInverse: true, process: applyInverseCDF97Lift1D_8)
    } else {
        runParametric8(base, length: length, count: count, stride: stride, isXPass: isXPass, isInverse: false, process: applyCDF97Lift1D_8)
    }
}

public func leGall53LiftParametric(base: UnsafeMutablePointer<Int16>, length: Int, count: Int, stride: Int, isXPass: Bool, isInverse: Bool) {
    if count % 8 == 0 {
        if isInverse {
            runParametric8(base, length: length, count: count, stride: stride, isXPass: isXPass, isInverse: true, process: applyInverseLeGall53Lift1D_8)
        } else {
            runParametric8(base, length: length, count: count, stride: stride, isXPass: isXPass, isInverse: false, process: applyLeGall53Lift1D_8)
        }
    } else {
        if isInverse {
            runParametric4(base, length: length, count: count, stride: stride, isXPass: isXPass, isInverse: true, process: applyInverseLeGall53Lift1D_4)
        } else {
            runParametric4(base, length: length, count: count, stride: stride, isXPass: isXPass, isInverse: false, process: applyLeGall53Lift1D_4)
        }
    }
}

public func intraDwt2D(base: UnsafeMutablePointer<Int16>, size: Int, stride: Int, levels: Int, filter: DWTFilterType) {
    for y in 0..<size {
        let p = base + y * stride
        for x in 0..<size {
            p[x] = p[x] << 1
        }
    }
    
    var currentSize = size
    for _ in 0..<levels {
        if filter == .cdf97 {
            cdf97LiftParametric(base: base, length: currentSize, count: currentSize, stride: stride, isXPass: true, isInverse: false)
            cdf97LiftParametric(base: base, length: currentSize, count: currentSize, stride: stride, isXPass: false, isInverse: false)
        } else {
            leGall53LiftParametric(base: base, length: currentSize, count: currentSize, stride: stride, isXPass: true, isInverse: false)
            leGall53LiftParametric(base: base, length: currentSize, count: currentSize, stride: stride, isXPass: false, isInverse: false)
        }
        currentSize /= 2
    }
}

public func inverseIntraDwt2D(base: UnsafeMutablePointer<Int16>, size: Int, stride: Int, levels: Int, filter: DWTFilterType) {
    var currentSize = size >> (levels - 1)
    for _ in 0..<levels {
        if filter == .cdf97 {
            cdf97LiftParametric(base: base, length: currentSize, count: currentSize, stride: stride, isXPass: false, isInverse: true)
            cdf97LiftParametric(base: base, length: currentSize, count: currentSize, stride: stride, isXPass: true, isInverse: true)
        } else {
            leGall53LiftParametric(base: base, length: currentSize, count: currentSize, stride: stride, isXPass: false, isInverse: true)
            leGall53LiftParametric(base: base, length: currentSize, count: currentSize, stride: stride, isXPass: true, isInverse: true)
        }
        currentSize *= 2
    }
}
