import Foundation

/// In-place applies deblocking filter to the reconstructed image.
@inline(__always)
func applyDeblockingFilter(plane: inout [Int16], width: Int, height: Int, blockSize: Int, qStep: Int) {
    // Post-processing: out-of-loop like filtering for the whole generated plane.
    // It smooths boundaries between `blockSize` regions.
    
    // tc controls maximum clipping: smaller tc -> less smoothing, prevents blurring of real edges.
    // beta controls edge detection: smaller beta -> preserves more detailed textures.
    // Use non-linear integer scaling to prevent over-smoothing at low qStep,
    // while allowing strong deblocking at high qStep where boundaries are severe.
    let qBase = max(0, qStep)
    let tcNonLinear = (qBase * qBase) / 400 + (qBase / 3)
    let tc = Int16(min(40, max(2, tcNonLinear)))
    
    let betaNonLinear = (qBase * qBase) / 200 + qBase
    let beta = Int32(min(128, max(12, betaNonLinear)))
    
    // Vertical Edges (x = blockSize, 2*blockSize, ...)
    plane.withUnsafeMutableBufferPointer { buffer in
        guard let base = buffer.baseAddress else { return }
        
        for x in stride(from: blockSize, to: width, by: blockSize) {
            for yStart in stride(from: 0, to: height, by: 16) {
                let rowsToProcess = min(16, height - yStart)
                if rowsToProcess == 16 {
                    deblockFilterVerticalEdgeSIMD16(base: base, width: width, x: x, y: yStart, tc: tc, beta: beta)
                } else {
                    // Fallback for non-multiple of 16 heights (rare since we pad to 32)
                    deblockFilterVerticalEdgeScalar(base: base, width: width, x: x, y: yStart, count: rowsToProcess, tc: tc, beta: beta)
                }
            }
        }
    }
    
    // Horizontal Edges (y = blockSize, 2*blockSize, ...)
    plane.withUnsafeMutableBufferPointer { buffer in
        guard let base = buffer.baseAddress else { return }
        
        for y in stride(from: blockSize, to: height, by: blockSize) {
            for xStart in stride(from: 0, to: width, by: 16) {
                let colsToProcess = min(16, width - xStart)
                if colsToProcess == 16 {
                    deblockFilterHorizontalEdgeSIMD16(base: base, width: width, x: xStart, y: y, tc: tc, beta: beta)
                } else {
                    deblockFilterHorizontalEdgeScalar(base: base, width: width, x: xStart, y: y, count: colsToProcess, tc: tc, beta: beta)
                }
            }
        }
    }
}

@inline(__always)
private func deblockFilterVerticalEdgeSIMD16(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, tc: Int16, beta: Int32) {
    let off0 = y * width + x
    let off1 = off0 + width
    let off2 = off1 + width
    let off3 = off2 + width
    let off4 = off3 + width
    let off5 = off4 + width
    let off6 = off5 + width
    let off7 = off6 + width
    let off8 = off7 + width
    let off9 = off8 + width
    let off10 = off9 + width
    let off11 = off10 + width
    let off12 = off11 + width
    let off13 = off12 + width
    let off14 = off13 + width
    let off15 = off14 + width

    let p1 = SIMD16<Int16>(
        base[off0 - 2], base[off1 - 2], base[off2 - 2], base[off3 - 2],
        base[off4 - 2], base[off5 - 2], base[off6 - 2], base[off7 - 2],
        base[off8 - 2], base[off9 - 2], base[off10 - 2], base[off11 - 2],
        base[off12 - 2], base[off13 - 2], base[off14 - 2], base[off15 - 2]
    )
    let p0 = SIMD16<Int16>(
        base[off0 - 1], base[off1 - 1], base[off2 - 1], base[off3 - 1],
        base[off4 - 1], base[off5 - 1], base[off6 - 1], base[off7 - 1],
        base[off8 - 1], base[off9 - 1], base[off10 - 1], base[off11 - 1],
        base[off12 - 1], base[off13 - 1], base[off14 - 1], base[off15 - 1]
    )
    let q0 = SIMD16<Int16>(
        base[off0 + 0], base[off1 + 0], base[off2 + 0], base[off3 + 0],
        base[off4 + 0], base[off5 + 0], base[off6 + 0], base[off7 + 0],
        base[off8 + 0], base[off9 + 0], base[off10 + 0], base[off11 + 0],
        base[off12 + 0], base[off13 + 0], base[off14 + 0], base[off15 + 0]
    )
    let q1 = SIMD16<Int16>(
        base[off0 + 1], base[off1 + 1], base[off2 + 1], base[off3 + 1],
        base[off4 + 1], base[off5 + 1], base[off6 + 1], base[off7 + 1],
        base[off8 + 1], base[off9 + 1], base[off10 + 1], base[off11 + 1],
        base[off12 + 1], base[off13 + 1], base[off14 + 1], base[off15 + 1]
    )
    
    let (newP1, newP0, newQ0, newQ1) = deblockComputeFilter(p1: p1, p0: p0, q0: q0, q1: q1, tc: tc, beta: beta)
    
    base[off0 - 2] = newP1[0]; base[off0 - 1] = newP0[0]; base[off0 + 0] = newQ0[0]; base[off0 + 1] = newQ1[0]
    base[off1 - 2] = newP1[1]; base[off1 - 1] = newP0[1]; base[off1 + 0] = newQ0[1]; base[off1 + 1] = newQ1[1]
    base[off2 - 2] = newP1[2]; base[off2 - 1] = newP0[2]; base[off2 + 0] = newQ0[2]; base[off2 + 1] = newQ1[2]
    base[off3 - 2] = newP1[3]; base[off3 - 1] = newP0[3]; base[off3 + 0] = newQ0[3]; base[off3 + 1] = newQ1[3]
    base[off4 - 2] = newP1[4]; base[off4 - 1] = newP0[4]; base[off4 + 0] = newQ0[4]; base[off4 + 1] = newQ1[4]
    base[off5 - 2] = newP1[5]; base[off5 - 1] = newP0[5]; base[off5 + 0] = newQ0[5]; base[off5 + 1] = newQ1[5]
    base[off6 - 2] = newP1[6]; base[off6 - 1] = newP0[6]; base[off6 + 0] = newQ0[6]; base[off6 + 1] = newQ1[6]
    base[off7 - 2] = newP1[7]; base[off7 - 1] = newP0[7]; base[off7 + 0] = newQ0[7]; base[off7 + 1] = newQ1[7]
    base[off8 - 2] = newP1[8]; base[off8 - 1] = newP0[8]; base[off8 + 0] = newQ0[8]; base[off8 + 1] = newQ1[8]
    base[off9 - 2] = newP1[9]; base[off9 - 1] = newP0[9]; base[off9 + 0] = newQ0[9]; base[off9 + 1] = newQ1[9]
    base[off10 - 2] = newP1[10]; base[off10 - 1] = newP0[10]; base[off10 + 0] = newQ0[10]; base[off10 + 1] = newQ1[10]
    base[off11 - 2] = newP1[11]; base[off11 - 1] = newP0[11]; base[off11 + 0] = newQ0[11]; base[off11 + 1] = newQ1[11]
    base[off12 - 2] = newP1[12]; base[off12 - 1] = newP0[12]; base[off12 + 0] = newQ0[12]; base[off12 + 1] = newQ1[12]
    base[off13 - 2] = newP1[13]; base[off13 - 1] = newP0[13]; base[off13 + 0] = newQ0[13]; base[off13 + 1] = newQ1[13]
    base[off14 - 2] = newP1[14]; base[off14 - 1] = newP0[14]; base[off14 + 0] = newQ0[14]; base[off14 + 1] = newQ1[14]
    base[off15 - 2] = newP1[15]; base[off15 - 1] = newP0[15]; base[off15 + 0] = newQ0[15]; base[off15 + 1] = newQ1[15]
}

@inline(__always)
private func deblockFilterVerticalEdgeScalar(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, count: Int, tc: Int16, beta: Int32) {
    let betah = beta >> 1
    for dy in 0..<count {
        let offset = (y + dy) * width + x
        var p1 = base[offset - 2]
        var p0 = base[offset - 1]
        var q0 = base[offset + 0]
        var q1 = base[offset + 1]
        
        let delta = Int32(q0) - Int32(p0)
        if abs(delta) < beta && abs(Int32(p1) - Int32(p0)) < betah && abs(Int32(q1) - Int32(q0)) < betah {
            let d = (9 * (Int32(q0) - Int32(p0)) - 3 * (Int32(q1) - Int32(p1)) + 8) >> 4
            let dClipped = Int16(max(-Int32(tc), min(Int32(tc), d)))
            
            p0 = p0 &+ dClipped
            q0 = q0 &- dClipped
            
            let dHalf = dClipped / 2
            p1 = p1 &+ dHalf
            q1 = q1 &- dHalf
            
            base[offset - 2] = p1
            base[offset - 1] = p0
            base[offset + 0] = q0
            base[offset + 1] = q1
        }
    }
}

@inline(__always)
private func deblockFilterHorizontalEdgeSIMD16(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, tc: Int16, beta: Int32) {
    // Horizontal edge: p1, p0 | q0, q1
    // Memory layout for 16 columns is continuous!
    let offP1 = (y - 2) * width + x
    let offP0 = (y - 1) * width + x
    let offQ0 = (y + 0) * width + x
    let offQ1 = (y + 1) * width + x
    
    let p1 = SIMD16<Int16>(UnsafeBufferPointer(start: base + offP1, count: 16))
    let p0 = SIMD16<Int16>(UnsafeBufferPointer(start: base + offP0, count: 16))
    let q0 = SIMD16<Int16>(UnsafeBufferPointer(start: base + offQ0, count: 16))
    let q1 = SIMD16<Int16>(UnsafeBufferPointer(start: base + offQ1, count: 16))
    
    let (newP1, newP0, newQ0, newQ1) = deblockComputeFilter(p1: p1, p0: p0, q0: q0, q1: q1, tc: tc, beta: beta)
    
    let p1Ptr = UnsafeMutableRawPointer(base + offP1).assumingMemoryBound(to: SIMD16<Int16>.self)
    let p0Ptr = UnsafeMutableRawPointer(base + offP0).assumingMemoryBound(to: SIMD16<Int16>.self)
    let q0Ptr = UnsafeMutableRawPointer(base + offQ0).assumingMemoryBound(to: SIMD16<Int16>.self)
    let q1Ptr = UnsafeMutableRawPointer(base + offQ1).assumingMemoryBound(to: SIMD16<Int16>.self)
    
    p1Ptr.pointee = newP1
    p0Ptr.pointee = newP0
    q0Ptr.pointee = newQ0
    q1Ptr.pointee = newQ1
}

@inline(__always)
private func deblockFilterHorizontalEdgeScalar(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, count: Int, tc: Int16, beta: Int32) {
    let betah = beta >> 1
    for dx in 0..<count {
        let offset = (y * width) + x + dx
        var p1 = base[offset - 2 * width]
        var p0 = base[offset - 1 * width]
        var q0 = base[offset + 0 * width]
        var q1 = base[offset + 1 * width]
        
        let delta = Int32(q0) - Int32(p0)
        if abs(delta) < beta && abs(Int32(p1) - Int32(p0)) < betah && abs(Int32(q1) - Int32(q0)) < betah {
            let d = (9 * (Int32(q0) - Int32(p0)) - 3 * (Int32(q1) - Int32(p1)) + 8) >> 4
            let dClipped = Int16(max(-Int32(tc), min(Int32(tc), d)))
            
            p0 = p0 &+ dClipped
            q0 = q0 &- dClipped
            
            let dHalf = dClipped / 2
            p1 = p1 &+ dHalf
            q1 = q1 &- dHalf
            
            base[offset - 2 * width] = p1
            base[offset - 1 * width] = p0
            base[offset + 0 * width] = q0
            base[offset + 1 * width] = q1
        }
    }
}

@inline(__always)
private func deblockComputeFilter(p1: SIMD16<Int16>, p0: SIMD16<Int16>, q0: SIMD16<Int16>, q1: SIMD16<Int16>, tc: Int16, beta: Int32) -> (SIMD16<Int16>, SIMD16<Int16>, SIMD16<Int16>, SIMD16<Int16>) {
    // Extracted integer upcast for overflow prevention
    let p1x = SIMD16<Int32>(truncatingIfNeeded: p1)
    let p0x = SIMD16<Int32>(truncatingIfNeeded: p0)
    let q0x = SIMD16<Int32>(truncatingIfNeeded: q0)
    let q1x = SIMD16<Int32>(truncatingIfNeeded: q1)
    
    // delta = q0 - p0
    let delta = q0x &- p0x
    
    @inline(__always)
    func getAbsVector(_ v: SIMD16<Int32>) -> SIMD16<Int32> {
        return SIMD16<Int32>(
            abs(v[0]), abs(v[1]), abs(v[2]), abs(v[3]),
            abs(v[4]), abs(v[5]), abs(v[6]), abs(v[7]),
            abs(v[8]), abs(v[9]), abs(v[10]), abs(v[11]),
            abs(v[12]), abs(v[13]), abs(v[14]), abs(v[15])
        )
    }
    
    let absDelta = getAbsVector(delta)
    let absP = getAbsVector(p1x &- p0x)
    let absQ = getAbsVector(q1x &- q0x)
    
    let betah = beta >> 1
    let maskDelta = absDelta .< beta
    let maskP = absP .< betah
    let maskQ = absQ .< betah
    let mask = maskDelta .& maskP .& maskQ // bool mask
    
    // d = (9*(q0-p0) - 3*(q1-p1) + 8) >> 4
    let delta9 = delta &* 9
    let diffQ1P1 = q1x &- p1x
    let diffQ1P1_3 = diffQ1P1 &* 3
    let dSum = (delta9 &- diffQ1P1_3) &+ 8
    let dUnclipped = dSum &>> 4
    
    // clamp: max(-tc, min(tc, d))
    let threshold = Int32(tc)
    let lower = SIMD16<Int32>(repeating: -threshold)
    let upper = SIMD16<Int32>(repeating: threshold)
    var d = dUnclipped
    d.clamp(lowerBound: lower, upperBound: upper)
    
    let dMasked = SIMD16<Int32>(repeating: 0).replacing(with: d, where: mask)
    let d16 = SIMD16<Int16>(truncatingIfNeeded: dMasked)
    
    let newP0 = p0 &+ d16
    let newQ0 = q0 &- d16
    
    // Simulate division by 2 (truncating toward zero) using SIMD arithmetic
    // scalar equivalent is `val / 2`, so `val < 0 ? (val + 1) >> 1 : val >> 1`
    let dMaskedSign = SIMD16<Int32>(repeating: 0).replacing(with: 1, where: dMasked .< 0)
    let dHalfMasked = (dMasked &+ dMaskedSign) &>> 1
    let dHalf16 = SIMD16<Int16>(truncatingIfNeeded: dHalfMasked)
    
    let newP1 = p1 &+ dHalf16
    let newQ1 = q1 &- dHalf16
    
    return (newP1, newP0, newQ0, newQ1)
}
