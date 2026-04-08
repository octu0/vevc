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
        
        let pDiff = p1 &- p0
        let qDiff = q1 &- q0
        let absP = pDiff.replacing(with: .zero &- pDiff, where: pDiff .< 0)
        let absQ = qDiff.replacing(with: .zero &- qDiff, where: qDiff .< 0)
        
        let mask = (absDelta .< betaV) .& (absP .< betahV) .& (absQ .< betahV)
        
        let t1 = q0 &- p0
        let t2 = q1 &- p1
        let d = (v9 &* t1 &- v3 &* t2 &+ v8) &>> 4
        
        var dClipped = d
        dClipped.replace(with: tcV, where: dClipped .> tcV)
        dClipped.replace(with: ntcV, where: dClipped .< ntcV)
        
        let dHalf = dClipped / 2
        
        p0.replace(with: p0 &+ dClipped, where: mask)
        q0.replace(with: q0 &- dClipped, where: mask)
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

@inline(__always)
func applyDeringingFilter(plane: inout [Int16], width: Int, height: Int, qStep: Int) {
    let betaScalar = Int16(min(6, max(2, qStep / 6)))
    
    // Create a shadow buffer to completely decouple reads from writes.
    // By separating source (read-only) from destination (write-only), 
    // we eliminate loop-carried dependencies (RAW/WAR hazards).
    // This allows LLVM to heavily auto-vectorize the independent inner loops (SoA approach).
    let srcPlane = plane
    
    plane.withUnsafeMutableBufferPointer { dstBuf in
        srcPlane.withUnsafeBufferPointer { srcBuf in
            guard let dst = dstBuf.baseAddress, let src = srcBuf.baseAddress else { return }
            
            for y in 1..<(height - 1) {
                let rowOffset = y * width
                let srcY = src.advanced(by: rowOffset)
                let srcU = src.advanced(by: rowOffset - width)
                let srcD = src.advanced(by: rowOffset + width)
                let dstY = dst.advanced(by: rowOffset)
                
                // This inner loop only accesses perfectly contiguous memory arrays (srcY, srcU, srcD, dstY).
                // Without internal state modification, LLVM maps this directly to SIMD vectorizations.
                for x in 1..<(width - 1) {
                    let curr = srcY[x]
                    
                    let pL = srcY[x - 1]
                    let pR = srcY[x + 1]
                    let diffL = curr &- pL
                    let diffR = curr &- pR
                    let absL = diffL < 0 ? -diffL : diffL
                    let absR = diffR < 0 ? -diffR : diffR
                    let maskX: Int16 = (absL < betaScalar && absR < betaScalar) ? 1 : 0
                    let dX = (diffL &+ diffR &+ 2) &>> 2
                    
                    let pU = srcU[x]
                    let pD = srcD[x]
                    let diffU = curr &- pU
                    let diffD = curr &- pD
                    let absU = diffU < 0 ? -diffU : diffU
                    let absD = diffD < 0 ? -diffD : diffD
                    let maskY: Int16 = (absU < betaScalar && absD < betaScalar) ? 1 : 0
                    let dY = (diffU &+ diffD &+ 2) &>> 2
                    
                    dstY[x] = curr &- (dX &* maskX) &- (dY &* maskY)
                }
            }
        }
    }
}
