// MARK: - Deblocking Filter

// Smooths block boundary discontinuities to suppress block noise.
// tc/beta parameters use non-linear scaling based on quantization step.
/// In-place applies deblocking filter to the reconstructed image (32x32 block resolution).
@inline(__always)
func applyDeblockingFilter32(plane: inout [Int16], width: Int, height: Int, qStep: Int) {
    plane.withUnsafeMutableBufferPointer { buffer in
        guard let base = buffer.baseAddress else { return }
        let tc = Int16(min(15, max(5, (qStep / 2) + 3)))
        let beta = Int32(min(50, max(18, qStep + 6)))
        
        let hFast = (height / 32) * 32
        let wFast = (width / 32) * 32
        let hRem = height - hFast
        let wRem = width - wFast
        
        // Vertical Edges
        for x in stride(from: 32, to: width, by: 32) {
            for y in stride(from: 0, to: hFast, by: 32) {
                deblockFilterVerticalEdge32SIMD(base: base, width: width, x: x, y: y, tc: tc, beta: beta)
            }
            deblockFilterVerticalEdgeScalar(base: base, width: width, x: x, y: hFast, count: hRem, tc: tc, beta: beta)
        }
        
        // Horizontal Edges
        for y in stride(from: 32, to: height, by: 32) {
            for x in stride(from: 0, to: wFast, by: 32) {
                deblockFilterHorizontalEdge32SIMD(base: base, width: width, x: x, y: y, tc: tc, beta: beta)
            }
            deblockFilterHorizontalEdgeScalar(base: base, width: width, x: wFast, y: y, count: wRem, tc: tc, beta: beta)
        }
    }
}

/// In-place applies deblocking filter to the reconstructed image (16x16 block resolution).
@inline(__always)
func applyDeblockingFilter16(plane: inout [Int16], width: Int, height: Int, qStep: Int) {
    plane.withUnsafeMutableBufferPointer { buffer in
        guard let base = buffer.baseAddress else { return }
        let tc = Int16(min(15, max(5, (qStep / 2) + 3)))
        let beta = Int32(min(50, max(18, qStep + 6)))
        
        let hFast = (height / 16) * 16
        let wFast = (width / 16) * 16
        let hRem = height - hFast
        let wRem = width - wFast
        
        // Vertical Edges
        for x in stride(from: 16, to: width, by: 16) {
            for y in stride(from: 0, to: hFast, by: 16) {
                deblockFilterVerticalEdge16SIMD(base: base, width: width, x: x, y: y, tc: tc, beta: beta)
            }
            deblockFilterVerticalEdgeScalar(base: base, width: width, x: x, y: hFast, count: hRem, tc: tc, beta: beta)
        }
        
        // Horizontal Edges
        for y in stride(from: 16, to: height, by: 16) {
            for x in stride(from: 0, to: wFast, by: 16) {
                deblockFilterHorizontalEdgeSIMD16(base: base, width: width, x: x, y: y, tc: tc, beta: beta)
            }
            deblockFilterHorizontalEdgeScalar(base: base, width: width, x: wFast, y: y, count: wRem, tc: tc, beta: beta)
        }
    }
}

@inline(__always)
private func deblockFilterVerticalEdge16SIMD(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, tc: Int16, beta: Int32) {
    var vP1 = SIMD16<Int16>()
    var vP0 = SIMD16<Int16>()
    var vQ0 = SIMD16<Int16>()
    var vQ1 = SIMD16<Int16>()
    
    var off = (y * width) + x
    for i in 0..<16 {
        vP1[i] = base[off - 2]
        vP0[i] = base[off - 1]
        vQ0[i] = base[off + 0]
        vQ1[i] = base[off + 1]
        off += width
    }
    
    let (nP1, nP0, nQ0, nQ1) = deblockComputeFilter(p1: vP1, p0: vP0, q0: vQ0, q1: vQ1, tc: tc, beta: beta)
    
    off = (y * width) + x
    for i in 0..<16 {
        base[off - 2] = nP1[i]
        base[off - 1] = nP0[i]
        base[off + 0] = nQ0[i]
        base[off + 1] = nQ1[i]
        off += width
    }
}

@inline(__always)
private func deblockFilterVerticalEdge32SIMD(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, tc: Int16, beta: Int32) {
    deblockFilterVerticalEdge16SIMD(base: base, width: width, x: x, y: y, tc: tc, beta: beta)
    deblockFilterVerticalEdge16SIMD(base: base, width: width, x: x, y: y + 16, tc: tc, beta: beta)
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
private func deblockFilterHorizontalEdge32SIMD(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, tc: Int16, beta: Int32) {
    deblockFilterHorizontalEdgeSIMD16(base: base, width: width, x: x, y: y, tc: tc, beta: beta)
    deblockFilterHorizontalEdgeSIMD16(base: base, width: width, x: x + 16, y: y, tc: tc, beta: beta)
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