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

/// In-place applies deblocking filter to the reconstructed image (32x32 block resolution), with Intra/Inter boundary enhancement.
@inline(__always)
func applyDeblockingFilter32(plane: inout [Int16], width: Int, height: Int, qStep: Int, mvs: [MotionVector]) {
    plane.withUnsafeMutableBufferPointer { buffer in
        guard let base = buffer.baseAddress else { return }
        
        let defaultTc = Int16(min(15, max(5, (qStep / 2) + 3)))
        let defaultBeta = Int32(min(50, max(18, qStep + 6)))
        
        let enhancedTc = Int16(min(22, max(7, (qStep / 2) + 3) * 3 / 2))
        let enhancedBeta = Int32(min(100, max(36, (qStep + 6) * 2)))
        
        let colCount = (width + 31) / 32
        let rowCount = (height + 31) / 32
        
        let hFast = (height / 32) * 32
        let wFast = (width / 32) * 32
        let hRem = height - hFast
        let wRem = width - wFast
        
        // Vertical Edges
        for col in 1..<colCount {
            let x = col * 32
            for row in 0..<rowCount {
                let y = row * 32
                let idx = row * colCount + col
                
                // Enhance deblocking at motion boundaries:
                // - Intra/Inter boundary: strongest filtering
                // - Inter/Inter boundary (both blocks have motion): moderate enhancement
                // - Intra/Intra boundary: default (minimal)
                let leftIdx = idx - 1
                let hasMotionLeft = (leftIdx < mvs.count) && (mvs[leftIdx].isIntra != true)
                let hasMotionRight = (idx < mvs.count) && (mvs[idx].isIntra != true)
                let isIntraBoundary = (idx < mvs.count && leftIdx < mvs.count) && (mvs[idx].isIntra != mvs[leftIdx].isIntra)
                let isMotionBoundary = hasMotionLeft || hasMotionRight
                let tc = isIntraBoundary ? enhancedTc : (isMotionBoundary ? enhancedTc : defaultTc)
                let beta = isIntraBoundary ? enhancedBeta : (isMotionBoundary ? enhancedBeta : defaultBeta)
                
                if y < hFast {
                    deblockFilterVerticalEdge32SIMD(base: base, width: width, x: x, y: y, tc: tc, beta: beta)
                } else {
                    let safeH = min(hRem, height - y)
                    deblockFilterVerticalEdgeScalar(base: base, width: width, x: x, y: y, count: safeH, tc: tc, beta: beta)
                }
            }
        }
        
        // Horizontal Edges
        for row in 1..<rowCount {
            let y = row * 32
            for col in 0..<colCount {
                let x = col * 32
                let idx = row * colCount + col
                
                let topIdx = idx - colCount
                let hasMotionTop = (0 <= topIdx && topIdx < mvs.count) && (mvs[topIdx].isIntra != true)
                let hasMotionBottom = (idx < mvs.count) && (mvs[idx].isIntra != true)
                let isIntraBoundary = (idx < mvs.count && 0 <= topIdx) && (mvs[idx].isIntra != mvs[topIdx].isIntra)
                let isMotionBoundary = hasMotionTop || hasMotionBottom
                let tc = isIntraBoundary ? enhancedTc : (isMotionBoundary ? enhancedTc : defaultTc)
                let beta = isIntraBoundary ? enhancedBeta : (isMotionBoundary ? enhancedBeta : defaultBeta)
                
                if x < wFast {
                    deblockFilterHorizontalEdge32SIMD(base: base, width: width, x: x, y: y, tc: tc, beta: beta)
                } else {
                    let safeW = min(wRem, width - x)
                    deblockFilterHorizontalEdgeScalar(base: base, width: width, x: x, y: y, count: safeW, tc: tc, beta: beta)
                }
            }
        }
    }
}

/// In-place applies deblocking filter to the Chroma plane (16x16 blocks), with Intra/Inter boundary enhancement using Luma MVs.
@inline(__always)
func applyDeblockingFilterChroma16(plane: inout [Int16], width: Int, height: Int, qStep: Int, mvs: [MotionVector]) {
    plane.withUnsafeMutableBufferPointer { buffer in
        guard let base = buffer.baseAddress else { return }
        
        let defaultTc = Int16(min(15, max(5, (qStep / 2) + 3)))
        let defaultBeta = Int32(min(50, max(18, qStep + 6)))
        
        let enhancedTc = Int16(min(22, max(7, (qStep / 2) + 3) * 3 / 2))
        let enhancedBeta = Int32(min(100, max(36, (qStep + 6) * 2)))
        
        let colCountC = (width + 15) / 16
        let rowCountC = (height + 15) / 16
        
        // mvs array is based on 32x32 Luma blocks. 1 Chroma block (16x16) = 1 Luma block (32x32)
        // so mvs array has exactly colCountC columns.
        let mvColCount = colCountC 
        
        let hFast = (height / 16) * 16
        let wFast = (width / 16) * 16
        let hRem = height - hFast
        let wRem = width - wFast
        
        // Vertical Edges
        for col in 1..<colCountC {
            let x = col * 16
            for row in 0..<rowCountC {
                let y = row * 16
                let idx = row * mvColCount + col
                
                let leftIdx = idx - 1
                let hasMotionLeft = (leftIdx < mvs.count) && (mvs[leftIdx].isIntra != true)
                let hasMotionRight = (idx < mvs.count) && (mvs[idx].isIntra != true)
                let isIntraBoundary = (idx < mvs.count && leftIdx < mvs.count) && (mvs[idx].isIntra != mvs[leftIdx].isIntra)
                let isMotionBoundary = hasMotionLeft || hasMotionRight
                let tc = isIntraBoundary ? enhancedTc : (isMotionBoundary ? enhancedTc : defaultTc)
                let beta = isIntraBoundary ? enhancedBeta : (isMotionBoundary ? enhancedBeta : defaultBeta)
                
                if y < hFast {
                    deblockFilterVerticalEdge16SIMD(base: base, width: width, x: x, y: y, tc: tc, beta: beta)
                } else {
                    let safeH = min(hRem, height - y)
                    deblockFilterVerticalEdgeScalar(base: base, width: width, x: x, y: y, count: safeH, tc: tc, beta: beta)
                }
            }
        }
        
        // Horizontal Edges
        for row in 1..<rowCountC {
            let y = row * 16
            for col in 0..<colCountC {
                let x = col * 16
                let idx = row * mvColCount + col
                
                let topIdx = idx - mvColCount
                let hasMotionTop = (0 <= topIdx && topIdx < mvs.count) && (mvs[topIdx].isIntra != true)
                let hasMotionBottom = (idx < mvs.count) && (mvs[idx].isIntra != true)
                let isIntraBoundary = (idx < mvs.count && 0 <= topIdx) && (mvs[idx].isIntra != mvs[topIdx].isIntra)
                let isMotionBoundary = hasMotionTop || hasMotionBottom
                let tc = isIntraBoundary ? enhancedTc : (isMotionBoundary ? enhancedTc : defaultTc)
                let beta = isIntraBoundary ? enhancedBeta : (isMotionBoundary ? enhancedBeta : defaultBeta)
                
                if x < wFast {
                    deblockFilterHorizontalEdgeSIMD16(base: base, width: width, x: x, y: y, tc: tc, beta: beta)
                } else {
                    let safeW = min(wRem, width - x)
                    deblockFilterHorizontalEdgeScalar(base: base, width: width, x: x, y: y, count: safeW, tc: tc, beta: beta)
                }
            }
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
    var vP0 = SIMD16<Int16>()
    var vQ0 = SIMD16<Int16>()
    
    var off = (y * width) + x
    for i in 0..<16 {
        vP0[i] = base[off - 1]
        vQ0[i] = base[off + 0]
        off += width
    }
    
    let (nP0, nQ0) = deblockComputeFilter(p0: vP0, q0: vQ0, tc: tc, beta: beta)
    
    off = (y * width) + x
    for i in 0..<16 {
        base[off - 1] = nP0[i]
        base[off + 0] = nQ0[i]
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
    var offset = y * width + x
    for _ in 0..<count {
        var p0 = base[offset - 1]
        var q0 = base[offset + 0]
        
        let delta = Int32(q0) - Int32(p0)
        let absDelta = if delta < 0 { -1 * delta } else { delta }
        if absDelta < beta {
            var d = (delta + 1) >> 1
            let t = Int32(tc)
            if t < d { d = t }
            if d < (-1 * t) { d = (-1 * t) }
            
            let d16 = Int16(d)
            p0 = p0 &+ d16
            q0 = q0 &- d16
            
            base[offset - 1] = p0
            base[offset + 0] = q0
        }
        offset += width
    }
}

@inline(__always)
private func deblockFilterHorizontalEdgeSIMD16(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, tc: Int16, beta: Int32) {
    let offP0 = (y - 1) * width + x
    let offQ0 = (y + 0) * width + x
    
    let p0Ptr = UnsafeRawPointer(base.advanced(by: offP0))
    let q0Ptr = UnsafeRawPointer(base.advanced(by: offQ0))
    
    let p0 = p0Ptr.loadUnaligned(fromByteOffset: 0, as: SIMD16<Int16>.self)
    let q0 = q0Ptr.loadUnaligned(fromByteOffset: 0, as: SIMD16<Int16>.self)
    
    let (newP0, newQ0) = deblockComputeFilter(p0: p0, q0: q0, tc: tc, beta: beta)
    
    let p0MutPtr = UnsafeMutableRawPointer(base.advanced(by: offP0))
    let q0MutPtr = UnsafeMutableRawPointer(base.advanced(by: offQ0))
    
    p0MutPtr.storeBytes(of: newP0, toByteOffset: 0, as: SIMD16<Int16>.self)
    q0MutPtr.storeBytes(of: newQ0, toByteOffset: 0, as: SIMD16<Int16>.self)
}

@inline(__always)
private func deblockFilterHorizontalEdgeScalar(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, count: Int, tc: Int16, beta: Int32) {
    var offset = (y * width) + x
    for _ in 0..<count {
        var p0 = base[offset - 1 * width]
        var q0 = base[offset + 0 * width]
        
        let delta = Int32(q0) - Int32(p0)
        let absDelta = if delta < 0 { -1 * delta } else { delta }
        if absDelta < beta {
            var d = (delta + 1) >> 1
            let t = Int32(tc)
            if t < d { d = t }
            if d < (-1 * t) { d = (-1 * t) }
            
            let d16 = Int16(d)
            p0 = p0 &+ d16
            q0 = q0 &- d16
            
            base[offset - 1 * width] = p0
            base[offset + 0 * width] = q0
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
private func deblockComputeFilter(p0: SIMD16<Int16>, q0: SIMD16<Int16>, tc: Int16, beta: Int32) -> (SIMD16<Int16>, SIMD16<Int16>) {
    let beta16 = Int16(beta)
    
    let diff = q0 &- p0
    let absDiff = diff.replacing(with: p0 &- q0, where: q0 .< p0)
    
    let mask = absDiff .< beta16
    
    var d = (diff &+ 1) &>> 1
    
    let tc16 = SIMD16<Int16>(repeating: tc)
    let ntc16 = SIMD16<Int16>(repeating: -tc)
    d.replace(with: tc16, where: d .> tc16)
    d.replace(with: ntc16, where: d .< ntc16)
    
    var newP0 = p0
    var newQ0 = q0
    newP0.replace(with: p0 &+ d, where: mask)
    newQ0.replace(with: q0 &- d, where: mask)
    
    return (newP0, newQ0)
}

// MARK: - Intra/Inter Boundary Blend

/// Smooths the boundary between Intra and Inter blocks to reduce block noise.
@inline(__always)
func blendIntraInterBoundaryLuma32(plane: inout [Int16], mvs: [MotionVector], width: Int, height: Int) {
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32
    
    plane.withUnsafeMutableBufferPointer { buffer in
        guard let base = buffer.baseAddress else { return }
        for row in 0..<rowCount {
            for col in 0..<colCount {
                let idx = row * colCount + col
                if mvs[idx].isIntra {
                    let bx = col * 32
                    let by = row * 32
                    
                    // Left neighbor
                    if 0 < col && !mvs[idx - 1].isIntra {
                        let safeH = min(32, height - by)
                        if 4 <= bx && bx + 3 < width {
                            blendVerticalEdgeLuma32(base: base, width: width, x: bx, y: by, height: safeH)
                        }
                    }
                    // Right neighbor
                    if col < colCount - 1 && !mvs[idx + 1].isIntra {
                        let bxRight = bx + 32
                        if 4 <= bxRight && bxRight + 3 < width {
                            let safeH = min(32, height - by)
                            blendVerticalEdgeLuma32(base: base, width: width, x: bxRight, y: by, height: safeH)
                        }
                    }
                    // Top neighbor
                    if 0 < row && !mvs[idx - colCount].isIntra {
                        let safeW = min(32, width - bx)
                        if 4 <= by && by + 3 < height {
                            blendHorizontalEdgeLuma32(base: base, width: width, x: bx, y: by, widthBlock: safeW)
                        }
                    }
                    // Bottom neighbor
                    if row < rowCount - 1 && !mvs[idx + colCount].isIntra {
                        let byBottom = by + 32
                        let safeW = min(32, width - bx)
                        if 4 <= byBottom && byBottom + 3 < height {
                            blendHorizontalEdgeLuma32(base: base, width: width, x: bx, y: byBottom, widthBlock: safeW)
                        }
                    }
                }
            }
        }
    }
}

@inline(__always)
func blendIntraInterBoundaryChroma16(plane: inout [Int16], mvs: [MotionVector], width: Int, height: Int) {
    // For Chroma, the blocks are 16x16, but they correspond to the same mvs array.
    let colCount = (width + 15) / 16
    let rowCount = (height + 15) / 16
    // Note: The mvs array has dimensions based on Luma 32x32 blocks, which maps 1:1 to Chroma 16x16 blocks.
    
    plane.withUnsafeMutableBufferPointer { buffer in
        guard let base = buffer.baseAddress else { return }
        for row in 0..<rowCount {
            for col in 0..<colCount {
                let idx = row * colCount + col
                if idx < mvs.count && mvs[idx].isIntra {
                    let bx = col * 16
                    let by = row * 16
                    
                    // Left neighbor
                    if 0 < col && !mvs[idx - 1].isIntra {
                        let safeH = min(16, height - by)
                        if 2 <= bx && bx + 1 < width {
                            blendVerticalEdgeChroma16(base: base, width: width, x: bx, y: by, height: safeH)
                        }
                    }
                    // Right neighbor
                    if col < colCount - 1 && !mvs[idx + 1].isIntra {
                        let bxRight = bx + 16
                        if 2 <= bxRight && bxRight + 1 < width {
                            let safeH = min(16, height - by)
                            blendVerticalEdgeChroma16(base: base, width: width, x: bxRight, y: by, height: safeH)
                        }
                    }
                    // Top neighbor
                    if 0 < row && !mvs[idx - colCount].isIntra {
                        let safeW = min(16, width - bx)
                        if 2 <= by && by + 1 < height {
                            blendHorizontalEdgeChroma16(base: base, width: width, x: bx, y: by, widthBlock: safeW)
                        }
                    }
                    // Bottom neighbor
                    if row < rowCount - 1 && !mvs[idx + colCount].isIntra {
                        let byBottom = by + 16
                        let safeW = min(16, width - bx)
                        if 2 <= byBottom && byBottom + 1 < height {
                            blendHorizontalEdgeChroma16(base: base, width: width, x: bx, y: byBottom, widthBlock: safeW)
                        }
                    }
                }
            }
        }
    }
}

@inline(__always)
private func blendVerticalEdgeLuma32(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, height: Int) {
    var offset = y * width + x
    for _ in 0..<height {
        let p3 = Int32(base[offset - 4])
        let p2 = Int32(base[offset - 3])
        let p1 = Int32(base[offset - 2])
        let p0 = Int32(base[offset - 1])
        let q0 = Int32(base[offset + 0])
        let q1 = Int32(base[offset + 1])
        let q2 = Int32(base[offset + 2])
        let q3 = Int32(base[offset + 3])
        
        base[offset - 4] = Int16((p3 * 7 + q0 * 1) >> 3)
        base[offset - 3] = Int16((p2 * 6 + q0 * 2) >> 3)
        base[offset - 2] = Int16((p1 * 5 + q0 * 3) >> 3)
        base[offset - 1] = Int16((p0 * 4 + q0 * 4) >> 3)
        
        base[offset + 0] = Int16((q0 * 4 + p0 * 4) >> 3)
        base[offset + 1] = Int16((q1 * 5 + p0 * 3) >> 3)
        base[offset + 2] = Int16((q2 * 6 + p0 * 2) >> 3)
        base[offset + 3] = Int16((q3 * 7 + p0 * 1) >> 3)
        
        offset += width
    }
}

@inline(__always)
private func blendHorizontalEdgeLuma32(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, widthBlock: Int) {
    var offset = y * width + x
    for _ in 0..<widthBlock {
        let p3 = Int32(base[offset - 4 * width])
        let p2 = Int32(base[offset - 3 * width])
        let p1 = Int32(base[offset - 2 * width])
        let p0 = Int32(base[offset - 1 * width])
        let q0 = Int32(base[offset + 0 * width])
        let q1 = Int32(base[offset + 1 * width])
        let q2 = Int32(base[offset + 2 * width])
        let q3 = Int32(base[offset + 3 * width])
        
        base[offset - 4 * width] = Int16((p3 * 7 + q0 * 1) >> 3)
        base[offset - 3 * width] = Int16((p2 * 6 + q0 * 2) >> 3)
        base[offset - 2 * width] = Int16((p1 * 5 + q0 * 3) >> 3)
        base[offset - 1 * width] = Int16((p0 * 4 + q0 * 4) >> 3)
        
        base[offset + 0 * width] = Int16((q0 * 4 + p0 * 4) >> 3)
        base[offset + 1 * width] = Int16((q1 * 5 + p0 * 3) >> 3)
        base[offset + 2 * width] = Int16((q2 * 6 + p0 * 2) >> 3)
        base[offset + 3 * width] = Int16((q3 * 7 + p0 * 1) >> 3)
        
        offset += 1
    }
}

@inline(__always)
private func blendVerticalEdgeChroma16(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, height: Int) {
    var offset = y * width + x
    for _ in 0..<height {
        let p1 = Int32(base[offset - 2])
        let p0 = Int32(base[offset - 1])
        let q0 = Int32(base[offset + 0])
        let q1 = Int32(base[offset + 1])
        
        base[offset - 2] = Int16((p1 * 3 + q0 * 1) >> 2)
        base[offset - 1] = Int16((p0 * 2 + q0 * 2) >> 2)
        base[offset + 0] = Int16((q0 * 2 + p0 * 2) >> 2)
        base[offset + 1] = Int16((q1 * 3 + p0 * 1) >> 2)
        
        offset += width
    }
}

@inline(__always)
private func blendHorizontalEdgeChroma16(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, widthBlock: Int) {
    var offset = y * width + x
    for _ in 0..<widthBlock {
        let p1 = Int32(base[offset - 2 * width])
        let p0 = Int32(base[offset - 1 * width])
        let q0 = Int32(base[offset + 0 * width])
        let q1 = Int32(base[offset + 1 * width])
        
        base[offset - 2 * width] = Int16((p1 * 3 + q0 * 1) >> 2)
        base[offset - 1 * width] = Int16((p0 * 2 + q0 * 2) >> 2)
        base[offset + 0 * width] = Int16((q0 * 2 + p0 * 2) >> 2)
        base[offset + 1 * width] = Int16((q1 * 3 + p0 * 1) >> 2)
        
        offset += 1
    }
}
