import Foundation

// Smooths block boundary discontinuities to suppress block noise.
// tc/beta parameters use non-linear scaling based on quantization step.
/// In-place applies deblocking filter to the reconstructed image.
@inline(__always)
func applyDeblockingFilter(plane: inout [Int16], width: Int, height: Int, blockSize: Int, qStep: Int) {
    plane.withUnsafeMutableBufferPointer { buffer in
        guard let base = buffer.baseAddress else { return }
        applyDeblockingFilterPtr(base: base, width: width, height: height, blockSize: blockSize, qStep: qStep)
    }
}

/// Pointer-based internal loop to avoid closures during execution.
@inline(__always)
func applyDeblockingFilterPtr(base: UnsafeMutablePointer<Int16>, width: Int, height: Int, blockSize: Int, qStep: Int) {
    // larger qStep (lower quality) needs more aggressive filtering
    // increase the minimum guaranteed value to eliminate fine block boundary noise even at high rates (small qStep)
    let tc = Int16(min(12, max(4, (qStep / 2) + 2)))
    // only apply filter when boundary step is below beta to preserve real edges
    let beta = Int32(min(45, max(16, qStep + 4)))
    
    // separate 16-row fast path from scalar remainder
    // to eliminate inner-loop branch prediction misses
    let hBlocks16 = height / 16
    let hFast = hBlocks16 * 16
    let wBlocks16 = width / 16
    let wFast = wBlocks16 * 16
    
    // Vertical Edges (x = blockSize, 2*blockSize, ...)
    for x in stride(from: blockSize, to: width, by: blockSize) {
        var y = 0
        while y < hFast {
            deblockFilterVerticalEdge16(base: base, width: width, x: x, y: y, tc: tc, beta: beta)
            y += 16
        }
        
        let remaining = height - y
        if 0 < remaining {
            deblockFilterVerticalEdgeScalar(base: base, width: width, x: x, y: y, count: remaining, tc: tc, beta: beta)
        }
    }
    
    // Horizontal Edges (y = blockSize, 2*blockSize, ...)
    for y in stride(from: blockSize, to: height, by: blockSize) {
        var x = 0
        while x < wFast {
            deblockFilterHorizontalEdgeSIMD16(base: base, width: width, x: x, y: y, tc: tc, beta: beta)
            x += 16
        }
        
        let remaining = width - x
        if 0 < remaining {
            deblockFilterHorizontalEdgeScalar(base: base, width: width, x: x, y: y, count: remaining, tc: tc, beta: beta)
        }
    }
}

@inline(__always)
private func deblockFilterVerticalEdge16(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, tc: Int16, beta: Int32) {
    let betah = Int16(beta >> 1)
    let beta16 = Int16(beta)
    
    var offset = (y * width) + x
    
    let betaV = SIMD4<Int16>(repeating: beta16)
    let betahV = SIMD4<Int16>(repeating: betah)
    let tcV = SIMD4<Int16>(repeating: tc)
    let ntcV = .zero &- tcV
    let v9 = SIMD4<Int16>(repeating: 9)
    let v3 = SIMD4<Int16>(repeating: 3)
    let v8 = SIMD4<Int16>(repeating: 8)
    
    // why: 4x4 matrix transposition (AoS to SoA) enables memory-contiguous SIMD vectorization
    // for vertical edges processing multiple rows at once without cache degradation
    let performDeblock4x4 = { (off: Int) in
        let off0 = off
        let off1 = off + width
        let off2 = off + width * 2
        let off3 = off + width * 3
        
        let r0 = UnsafeRawPointer(base.advanced(by: off0 - 2)).loadUnaligned(as: SIMD4<Int16>.self)
        let r1 = UnsafeRawPointer(base.advanced(by: off1 - 2)).loadUnaligned(as: SIMD4<Int16>.self)
        let r2 = UnsafeRawPointer(base.advanced(by: off2 - 2)).loadUnaligned(as: SIMD4<Int16>.self)
        let r3 = UnsafeRawPointer(base.advanced(by: off3 - 2)).loadUnaligned(as: SIMD4<Int16>.self)
        
        // Transpose 4x4 (AoS -> SoA)
        var p1 = SIMD4<Int16>(r0[0], r1[0], r2[0], r3[0])
        var p0 = SIMD4<Int16>(r0[1], r1[1], r2[1], r3[1])
        var q0 = SIMD4<Int16>(r0[2], r1[2], r2[2], r3[2])
        var q1 = SIMD4<Int16>(r0[3], r1[3], r2[3], r3[3])
        
        let delta = q0 &- p0
        let absDelta = delta.replacing(with: .zero &- delta, where: delta .< 0)
        
        let deltaP = p1 &- p0
        let deltaQ = q1 &- q0
        let absP = deltaP.replacing(with: .zero &- deltaP, where: deltaP .< 0)
        let absQ = deltaQ.replacing(with: .zero &- deltaQ, where: deltaQ .< 0)
        
        let mask = (absDelta .< betaV) .& (absP .< betahV) .& (absQ .< betahV)
        
        let t1 = q0 &- p0
        let t2 = q1 &- p1
        var d = (v9 &* t1 &- v3 &* t2 &+ v8) &>> 4
        
        d.replace(with: tcV, where: tcV .< d)
        d.replace(with: ntcV, where: d .< ntcV)
        
        let dHalf = d / 2
        
        p0.replace(with: p0 &+ d, where: mask)
        q0.replace(with: q0 &- d, where: mask)
        p1.replace(with: p1 &+ dHalf, where: mask)
        q1.replace(with: q1 &- dHalf, where: mask)
        
        // Transpose 4x4 back (SoA -> AoS)
        let out0 = SIMD4<Int16>(p1[0], p0[0], q0[0], q1[0])
        let out1 = SIMD4<Int16>(p1[1], p0[1], q0[1], q1[1])
        let out2 = SIMD4<Int16>(p1[2], p0[2], q0[2], q1[2])
        let out3 = SIMD4<Int16>(p1[3], p0[3], q0[3], q1[3])
        
        UnsafeMutableRawPointer(base.advanced(by: off0 - 2)).storeBytes(of: out0, as: SIMD4<Int16>.self)
        UnsafeMutableRawPointer(base.advanced(by: off1 - 2)).storeBytes(of: out1, as: SIMD4<Int16>.self)
        UnsafeMutableRawPointer(base.advanced(by: off2 - 2)).storeBytes(of: out2, as: SIMD4<Int16>.self)
        UnsafeMutableRawPointer(base.advanced(by: off3 - 2)).storeBytes(of: out3, as: SIMD4<Int16>.self)
    }
    
    // 16 iterations unrolled into 4 block iterations
    performDeblock4x4(offset); offset += width * 4
    performDeblock4x4(offset); offset += width * 4
    performDeblock4x4(offset); offset += width * 4
    performDeblock4x4(offset)
}

@inline(__always)
private func deblockFilterVerticalEdgeScalar(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, count: Int, tc: Int16, beta: Int32) {
    let betah = beta >> 1
    var offset = y * width + x
    for _ in 0..<count {
        var p1 = base[offset - 2]
        var p0 = base[offset - 1]
        var q0 = base[offset + 0]
        var q1 = base[offset + 1]
        
        let delta = Int32(q0) - Int32(p0)
        let absDelta = if delta < 0 { -delta } else { delta }
        if absDelta < beta {
            let deltaP = Int32(p1) - Int32(p0)
            let deltaQ = Int32(q1) - Int32(q0)
            let absP = if deltaP < 0 { -deltaP } else { deltaP }
            let absQ = if deltaQ < 0 { -deltaQ } else { deltaQ }
            if absP < betah && absQ < betah {
                var d = (9 * (Int32(q0) - Int32(p0)) - 3 * (Int32(q1) - Int32(p1)) + 8) >> 4
                let t = Int32(tc)
                if t < d { d = t }
                if d < (-1 * t) { d = (-1 * t) }
                
                let dHalf = d / 2
                let d16 = Int16(d)
                let dh16 = Int16(dHalf)
                
                p0 = p0 &+ d16
                q0 = q0 &- d16
                p1 = p1 &+ dh16
                q1 = q1 &- dh16
                
                base[offset - 2] = p1
                base[offset - 1] = p0
                base[offset + 0] = q0
                base[offset + 1] = q1
            }
        }
        offset += width
    }
}

@inline(__always)
private func deblockFilterHorizontalEdgeSIMD16(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, tc: Int16, beta: Int32) {
    let offP1 = (y - 2) * width + x
    let offP0 = (y - 1) * width + x
    let offQ0 = (y + 0) * width + x
    let offQ1 = (y + 1) * width + x
    
    let p1Ptr = UnsafeRawPointer(base.advanced(by: offP1))
    let p0Ptr = UnsafeRawPointer(base.advanced(by: offP0))
    let q0Ptr = UnsafeRawPointer(base.advanced(by: offQ0))
    let q1Ptr = UnsafeRawPointer(base.advanced(by: offQ1))
    
    let p1 = p1Ptr.loadUnaligned(fromByteOffset: 0, as: SIMD16<Int16>.self)
    let p0 = p0Ptr.loadUnaligned(fromByteOffset: 0, as: SIMD16<Int16>.self)
    let q0 = q0Ptr.loadUnaligned(fromByteOffset: 0, as: SIMD16<Int16>.self)
    let q1 = q1Ptr.loadUnaligned(fromByteOffset: 0, as: SIMD16<Int16>.self)
    
    let (newP1, newP0, newQ0, newQ1) = deblockComputeFilter(p1: p1, p0: p0, q0: q0, q1: q1, tc: tc, beta: beta)
    
    let p1MutPtr = UnsafeMutableRawPointer(base.advanced(by: offP1))
    let p0MutPtr = UnsafeMutableRawPointer(base.advanced(by: offP0))
    let q0MutPtr = UnsafeMutableRawPointer(base.advanced(by: offQ0))
    let q1MutPtr = UnsafeMutableRawPointer(base.advanced(by: offQ1))
    
    p1MutPtr.storeBytes(of: newP1, toByteOffset: 0, as: SIMD16<Int16>.self)
    p0MutPtr.storeBytes(of: newP0, toByteOffset: 0, as: SIMD16<Int16>.self)
    q0MutPtr.storeBytes(of: newQ0, toByteOffset: 0, as: SIMD16<Int16>.self)
    q1MutPtr.storeBytes(of: newQ1, toByteOffset: 0, as: SIMD16<Int16>.self)
}

@inline(__always)
private func deblockFilterHorizontalEdgeScalar(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, count: Int, tc: Int16, beta: Int32) {
    let betah = beta >> 1
    var offset = (y * width) + x
    for _ in 0..<count {
        var p1 = base[offset - 2 * width]
        var p0 = base[offset - 1 * width]
        var q0 = base[offset + 0 * width]
        var q1 = base[offset + 1 * width]
        
        let delta = Int32(q0) - Int32(p0)
        let absDelta = if delta < 0 { -1 * delta } else { delta }
        if absDelta < beta {
            let deltaP = Int32(p1) - Int32(p0)
            let deltaQ = Int32(q1) - Int32(q0)
            let absP = if deltaP < 0 { -1 * deltaP } else { deltaP }
            let absQ = if deltaQ < 0 { -1 * deltaQ } else { deltaQ }
            if absP < betah && absQ < betah {
                var d = (((9 * (Int32(q0) - Int32(p0))) - (3 * (Int32(q1) - Int32(p1)))) + 8) >> 4
                let t = Int32(tc)
                if t < d { d = t }
                if d < (-1 * t) { d = (-1 * t) }
                
                let dHalf = d / 2
                let d16 = Int16(d)
                let dh16 = Int16(dHalf)
                
                p0 = p0 &+ d16
                q0 = q0 &- d16
                p1 = p1 &+ dh16
                q1 = q1 &- dh16
                
                base[offset - 2 * width] = p1
                base[offset - 1 * width] = p0
                base[offset + 0 * width] = q0
                base[offset + 1 * width] = q1
            }
        }
        offset += 1
    }
}

@inline(__always)
private func deblockComputeFilter(p1: SIMD16<Int16>, p0: SIMD16<Int16>, q0: SIMD16<Int16>, q1: SIMD16<Int16>, tc: Int16, beta: Int32) -> (SIMD16<Int16>, SIMD16<Int16>, SIMD16<Int16>, SIMD16<Int16>) {
    // why: Int16 domain eliminates 4 widen + 2 narrow operations vs Int32
    // Safe because masked lanes satisfy |delta| < beta ≤ 45, so 9*delta ≤ 405 fits Int16
    // Unmasked lanes may overflow but are masked away before store
    let betah = Int16(beta >> 1)
    let beta16 = Int16(beta)
    
    let betaV = SIMD16<Int16>(repeating: beta16)
    let betahV = SIMD16<Int16>(repeating: betah)
    let tcV = SIMD16<Int16>(repeating: tc)
    let ntcV = .zero &- tcV
    let v9 = SIMD16<Int16>(repeating: 9)
    let v3 = SIMD16<Int16>(repeating: 3)
    let v8 = SIMD16<Int16>(repeating: 8)
    
    let delta = q0 &- p0
    let absDelta = delta.replacing(with: .zero &- delta, where: delta .< 0)
    
    let deltaP = p1 &- p0
    let deltaQ = q1 &- q0
    let absP = deltaP.replacing(with: .zero &- deltaP, where: deltaP .< 0)
    let absQ = deltaQ.replacing(with: .zero &- deltaQ, where: deltaQ .< 0)
    
    let mask = (absDelta .< betaV) .& (absP .< betahV) .& (absQ .< betahV)
    
    let t1 = q0 &- p0
    let t2 = q1 &- p1
    var d = (v9 &* t1 &- v3 &* t2 &+ v8) &>> 4
    
    d.replace(with: tcV, where: tcV .< d)
    d.replace(with: ntcV, where: d .< ntcV)
    
    let dHalf = d / 2
    
    var newP0 = p0
    var newQ0 = q0
    var newP1 = p1
    var newQ1 = q1
    
    newP0.replace(with: p0 &+ d, where: mask)
    newQ0.replace(with: q0 &- d, where: mask)
    newP1.replace(with: p1 &+ dHalf, where: mask)
    newQ1.replace(with: q1 &- dHalf, where: mask)
    
    return (newP1, newP0, newQ0, newQ1)
}

