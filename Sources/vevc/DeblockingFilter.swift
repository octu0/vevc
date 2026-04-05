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
    // why: larger qStep (lower quality) needs more aggressive filtering
    let tc = Int16(min(12, max(2, qStep / 2)))
    // why: only apply filter when boundary step is below beta to preserve real edges
    let beta = Int32(min(45, max(12, qStep)))
    
    // why: separate 16-row fast path from scalar remainder
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
    let betah = beta >> 1
    var offset = (y * width) + x
    
    // why: 16-row unrolling eliminates loop overhead and branch misprediction
    let performDeblock = { (off: Int) in
        var p1 = base[off - 2]
        var p0 = base[off - 1]
        var q0 = base[off + 0]
        var q1 = base[off + 1]
        
        let delta = Int32(q0) - Int32(p0)
        let absDelta = delta < 0 ? -delta : delta
        if absDelta < beta {
            let pDiff = Int32(p1) - Int32(p0)
            let qDiff = Int32(q1) - Int32(q0)
            let absP = pDiff < 0 ? -pDiff : pDiff
            let absQ = qDiff < 0 ? -qDiff : qDiff
            if absP < betah && absQ < betah {
            // why: weighted center difference suppresses ringing at block boundaries
            // d = (9*(q0-p0) - 3*(q1-p1) + 8) >> 4
                let d = (9 * (Int32(q0) - Int32(p0)) - 3 * (Int32(q1) - Int32(p1)) + 8) >> 4
                var dClipped = d
                let t = Int32(tc)
                if t < dClipped { dClipped = t }
                if dClipped < (-1 * t) { dClipped = (-1 * t) }
                
                let dHalf = dClipped / 2
                let d16 = Int16(dClipped)
                let dh16 = Int16(dHalf)
                
                p0 = p0 &+ d16
                q0 = q0 &- d16
                p1 = p1 &+ dh16
                q1 = q1 &- dh16
                
                base[off - 2] = p1
                base[off - 1] = p0
                base[off + 0] = q0
                base[off + 1] = q1
            }
        }
    }
    
    // 16 iterations unrolled
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset); offset += width
    performDeblock(offset)
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
        let absDelta = delta < 0 ? -delta : delta
        if absDelta < beta {
            let pDiff = Int32(p1) - Int32(p0)
            let qDiff = Int32(q1) - Int32(q0)
            let absP = pDiff < 0 ? -pDiff : pDiff
            let absQ = qDiff < 0 ? -qDiff : qDiff
            if absP < betah && absQ < betah {
                let d = (9 * (Int32(q0) - Int32(p0)) - 3 * (Int32(q1) - Int32(p1)) + 8) >> 4
                var dClipped = d
                let t = Int32(tc)
                if t < dClipped { dClipped = t }
                if dClipped < (-1 * t) { dClipped = (-1 * t) }
                
                let dHalf = dClipped / 2
                let d16 = Int16(dClipped)
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
    
    let p1 = p1Ptr.load(as: SIMD16<Int16>.self)
    let p0 = p0Ptr.load(as: SIMD16<Int16>.self)
    let q0 = q0Ptr.load(as: SIMD16<Int16>.self)
    let q1 = q1Ptr.load(as: SIMD16<Int16>.self)
    
    let (newP1, newP0, newQ0, newQ1) = deblockComputeFilter(p1: p1, p0: p0, q0: q0, q1: q1, tc: tc, beta: beta)
    
    let p1MutPtr = UnsafeMutableRawPointer(base.advanced(by: offP1))
    let p0MutPtr = UnsafeMutableRawPointer(base.advanced(by: offP0))
    let q0MutPtr = UnsafeMutableRawPointer(base.advanced(by: offQ0))
    let q1MutPtr = UnsafeMutableRawPointer(base.advanced(by: offQ1))
    
    p1MutPtr.storeBytes(of: newP1, as: SIMD16<Int16>.self)
    p0MutPtr.storeBytes(of: newP0, as: SIMD16<Int16>.self)
    q0MutPtr.storeBytes(of: newQ0, as: SIMD16<Int16>.self)
    q1MutPtr.storeBytes(of: newQ1, as: SIMD16<Int16>.self)
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
        let absDelta = delta < 0 ? -delta : delta
        if absDelta < beta {
            let pDiff = Int32(p1) - Int32(p0)
            let qDiff = Int32(q1) - Int32(q0)
            let absP = pDiff < 0 ? -pDiff : pDiff
            let absQ = qDiff < 0 ? -qDiff : qDiff
            if absP < betah && absQ < betah {
                let d = (9 * (Int32(q0) - Int32(p0)) - 3 * (Int32(q1) - Int32(p1)) + 8) >> 4
                var dClipped = d
                let t = Int32(tc)
                if t < dClipped { dClipped = t }
                if dClipped < (-1 * t) { dClipped = (-1 * t) }
                
                let dHalf = dClipped / 2
                let d16 = Int16(dClipped)
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
    let p1x = SIMD16<Int32>(truncatingIfNeeded: p1)
    let p0x = SIMD16<Int32>(truncatingIfNeeded: p0)
    let q0x = SIMD16<Int32>(truncatingIfNeeded: q0)
    let q1x = SIMD16<Int32>(truncatingIfNeeded: q1)
    
    let delta = q0x &- p0x
    let diffP = p1x &- p0x
    let diffQ = q1x &- q0x
    
    let absDelta = SIMD16<Int32>(
        abs(delta[0]), abs(delta[1]), abs(delta[2]), abs(delta[3]),
        abs(delta[4]), abs(delta[5]), abs(delta[6]), abs(delta[7]),
        abs(delta[8]), abs(delta[9]), abs(delta[10]), abs(delta[11]),
        abs(delta[12]), abs(delta[13]), abs(delta[14]), abs(delta[15])
    )
    let absP = SIMD16<Int32>(
        abs(diffP[0]), abs(diffP[1]), abs(diffP[2]), abs(diffP[3]),
        abs(diffP[4]), abs(diffP[5]), abs(diffP[6]), abs(diffP[7]),
        abs(diffP[8]), abs(diffP[9]), abs(diffP[10]), abs(diffP[11]),
        abs(diffP[12]), abs(diffP[13]), abs(diffP[14]), abs(diffP[15])
    )
    let absQ = SIMD16<Int32>(
        abs(diffQ[0]), abs(diffQ[1]), abs(diffQ[2]), abs(diffQ[3]),
        abs(diffQ[4]), abs(diffQ[5]), abs(diffQ[6]), abs(diffQ[7]),
        abs(diffQ[8]), abs(diffQ[9]), abs(diffQ[10]), abs(diffQ[11]),
        abs(diffQ[12]), abs(diffQ[13]), abs(diffQ[14]), abs(diffQ[15])
    )
    
    let betah = beta >> 1
    let maskDelta = absDelta .< beta
    let maskP = absP .< betah
    let maskQ = absQ .< betah
    let mask = maskDelta .& maskP .& maskQ
    
    let delta9 = delta &* 9
    let diffQ1P1 = q1x &- p1x
    let diffQ1P1_3 = diffQ1P1 &* 3
    let dSum = (delta9 &- diffQ1P1_3) &+ 8
    let dUnclipped = dSum &>> 4
    
    let threshold = Int32(tc)
    let lower = SIMD16<Int32>(repeating: -threshold)
    let upper = SIMD16<Int32>(repeating: threshold)
    var d = dUnclipped
    d.clamp(lowerBound: lower, upperBound: upper)
    
    let dMasked = SIMD16<Int32>(repeating: 0).replacing(with: d, where: mask)
    let d16 = SIMD16<Int16>(truncatingIfNeeded: dMasked)
    
    let newP0 = p0 &+ d16
    let newQ0 = q0 &- d16
    
    let dMaskedSign = SIMD16<Int32>(repeating: 0).replacing(with: 1, where: dMasked .< 0)
    let dHalfMasked = (dMasked &+ dMaskedSign) &>> 1
    let dHalf16 = SIMD16<Int16>(truncatingIfNeeded: dHalfMasked)
    
    let newP1 = p1 &+ dHalf16
    let newQ1 = q1 &- dHalf16
    
    return (newP1, newP0, newQ0, newQ1)
}
