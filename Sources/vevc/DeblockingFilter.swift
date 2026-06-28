// MARK: - Deblocking Filter

// Smooths block boundary discontinuities to suppress block noise.
// tc/beta parameters use non-linear scaling based on quantization step.
/// In-place applies deblocking filter to the reconstructed image (32x32 block resolution).
@inline(__always)
func applyDeblockingFilter32(plane: inout [Int16], width: Int, height: Int, qStep: Int) {
    withUnsafePointers(mut: &plane) { base in
        let rawTc = (qStep / 2) + 3
        let tc: Int16 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int16((rawTc * (qStep - 3)) / 12)
            default: Int16(min(15, rawTc))
        }
        let rawBeta = qStep + 6
        let beta: Int32 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int32((rawBeta * (qStep - 3)) / 12)
            default: Int32(min(50, rawBeta))
        }
        
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
func applyDeblockingFilter32(plane: inout [Int16], width: Int, height: Int, qStep: Int, mvs: MotionVectors) {
    guard mvs.isEmpty != true else {
        applyDeblockingFilter32(plane: &plane, width: width, height: height, qStep: qStep)
        return
    }
    
    withUnsafePointers(mut: &plane) { base in
        let rawTc = (qStep / 2) + 3
        let defaultTc: Int16 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int16((rawTc * (qStep - 3)) / 12)
            default: Int16(min(15, rawTc))
        }
        let rawBeta = qStep + 6
        let defaultBeta: Int32 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int32((rawBeta * (qStep - 3)) / 12)
            default: Int32(min(50, rawBeta))
        }
        
        let rawETc = ((qStep / 2) + 3) * 3 / 2
        let enhancedTc: Int16 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int16((rawETc * (qStep - 3)) / 12)
            default: Int16(min(22, rawETc))
        }
        let rawEBeta = (qStep + 6) * 2
        let enhancedBeta: Int32 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int32((rawEBeta * (qStep - 3)) / 12)
            default: Int32(min(100, rawEBeta))
        }
        
        let colCount = (width + 31) / 32
        let rowCount = (height + 31) / 32
        
        let hFast = (height / 32) * 32
        let wFast = (width / 32) * 32
        let hRem = height - hFast
        let wRem = width - wFast
        
        withUnsafePointers(mvs.dx) { mvDxBase in
            let mvCount = mvs.dx.count
            
            // Vertical Edges
            for col in 1..<colCount {
                let x = col * 32
                for row in 0..<rowCount {
                    let y = row * 32
                    let idx = row * colCount + col
                    
                    let leftIdx = idx - 1
                    let leftDx: Int16 = if leftIdx < mvCount { mvDxBase[leftIdx] } else { 0 }
                    let rightDx: Int16 = if idx < mvCount { mvDxBase[idx] } else { 0 }
                    
                    let leftIsIntra = leftDx == 32767
                    let rightIsIntra = rightDx == 32767
                    
                    let isIntraBoundary = leftIsIntra || rightIsIntra
                    let tc = if isIntraBoundary { enhancedTc } else { defaultTc }
                    let beta = if isIntraBoundary { enhancedBeta } else { defaultBeta }
                    
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
                    let topDx: Int16 = if topIdx < mvCount { mvDxBase[topIdx] } else { 0 }
                    let bottomDx: Int16 = if idx < mvCount { mvDxBase[idx] } else { 0 }
                    
                    let topIsIntra = topDx == 32767
                    let bottomIsIntra = bottomDx == 32767
                    
                    let isIntraBoundary = topIsIntra || bottomIsIntra
                    let tc = if isIntraBoundary { enhancedTc } else { defaultTc }
                    let beta = if isIntraBoundary { enhancedBeta } else { defaultBeta }
                    
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
}

@inline(__always)
func applyDeblockingFilterChroma16(plane: inout [Int16], width: Int, height: Int, qStep: Int, mvs: MotionVectors) {
    guard mvs.isEmpty != true else {
        applyDeblockingFilter16(plane: &plane, width: width, height: height, qStep: qStep)
        return
    }
    
    withUnsafePointers(mut: &plane) { base in
        let rawTc = (qStep / 2) + 3
        let defaultTc: Int16 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int16((rawTc * (qStep - 3)) / 12)
            default: Int16(min(15, rawTc))
        }
        let rawBeta = qStep + 6
        let defaultBeta: Int32 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int32((rawBeta * (qStep - 3)) / 12)
            default: Int32(min(50, rawBeta))
        }
        
        let rawETc = ((qStep / 2) + 3) * 3 / 2
        let enhancedTc: Int16 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int16((rawETc * (qStep - 3)) / 12)
            default: Int16(min(22, rawETc))
        }
        let rawEBeta = (qStep + 6) * 2
        let enhancedBeta: Int32 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int32((rawEBeta * (qStep - 3)) / 12)
            default: Int32(min(100, rawEBeta))
        }
        
        let colCountC = (width + 15) / 16
        let rowCountC = (height + 15) / 16
        let mvColCount = colCountC 
        
        let hFast = (height / 16) * 16
        let wFast = (width / 16) * 16
        let hRem = height - hFast
        let wRem = width - wFast
        
        withUnsafePointers(mvs.dx) { mvDxBase in
            let mvCount = mvs.dx.count
            
            // Vertical Edges
            for col in 1..<colCountC {
                let x = col * 16
                for row in 0..<rowCountC {
                    let y = row * 16
                    let idx = row * mvColCount + col
                    
                    let leftIdx = idx - 1
                    let leftDx: Int16 = if leftIdx < mvCount { mvDxBase[leftIdx] } else { 0 }
                    let rightDx: Int16 = if idx < mvCount { mvDxBase[idx] } else { 0 }
                    
                    let leftIsIntra = leftDx == 32767
                    let rightIsIntra = rightDx == 32767
                    
                    let isIntraBoundary = leftIsIntra || rightIsIntra
                    let tc = if isIntraBoundary { enhancedTc } else { defaultTc }
                    let beta = if isIntraBoundary { enhancedBeta } else { defaultBeta }
                    
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
                    let topDx: Int16 = if topIdx < mvCount { mvDxBase[topIdx] } else { 0 }
                    let bottomDx: Int16 = if idx < mvCount { mvDxBase[idx] } else { 0 }
                    
                    let topIsIntra = topDx == 32767
                    let bottomIsIntra = bottomDx == 32767
                    
                    let isIntraBoundary = topIsIntra || bottomIsIntra
                    let tc = if isIntraBoundary { enhancedTc } else { defaultTc }
                    let beta = if isIntraBoundary { enhancedBeta } else { defaultBeta }
                    
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
}


/// In-place applies deblocking filter to the reconstructed image (16x16 block resolution).
@inline(__always)
func applyDeblockingFilter16(plane: inout [Int16], width: Int, height: Int, qStep: Int) {
    withUnsafePointers(mut: &plane) { base in
        let rawTc = (qStep / 2) + 3
        let tc: Int16 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int16((rawTc * (qStep - 3)) / 12)
            default: Int16(min(15, rawTc))
        }
        let rawBeta = qStep + 6
        let beta: Int32 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int32((rawBeta * (qStep - 3)) / 12)
            default: Int32(min(50, rawBeta))
        }
        
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
    let off0 = (y * width) + x
    
    var vP1 = SIMD16<Int16>()
    var vP0 = SIMD16<Int16>()
    var vQ0 = SIMD16<Int16>()
    var vQ1 = SIMD16<Int16>()
    
    var off = off0
    for i in 0..<16 {
        vP1[i] = base[off - 2]
        vP0[i] = base[off - 1]
        vQ0[i] = base[off + 0]
        vQ1[i] = base[off + 1]
        off += width
    }
    
    let (nP0, nQ0) = deblockComputeFilter(p1: vP1, p0: vP0, q0: vQ0, q1: vQ1, tc: tc, beta: beta)
    
    off = off0
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
    let betah = beta >> 1
    var offset = y * width + x
    for _ in 0..<count {
        let p1 = base[offset - 2]
        var p0 = base[offset - 1]
        var q0 = base[offset + 0]
        let q1 = base[offset + 1]
        
        let delta = Int32(q0) - Int32(p0)
        let absDelta = if delta < 0 { -delta } else { delta }
        if absDelta < beta {
            let deltaP = Int32(p1) - Int32(p0)
            let deltaQ = Int32(q1) - Int32(q0)
            let absP = if deltaP < 0 { -deltaP } else { deltaP }
            let absQ = if deltaQ < 0 { -deltaQ } else { deltaQ }
            if absP < betah && absQ < betah {
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
        }
        offset += width
    }
}

@inline(__always)
private func deblockFilterHorizontalEdgeSIMD16(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, tc: Int16, beta: Int32) {
    let offP0 = (y - 1) * width + x
    let offQ0 = (y + 0) * width + x
    
    let offP1 = (y - 2) * width + x
    let offQ1 = (y + 1) * width + x
    
    let p1Ptr = UnsafeRawPointer(base.advanced(by: offP1))
    let p0Ptr = UnsafeRawPointer(base.advanced(by: offP0))
    let q0Ptr = UnsafeRawPointer(base.advanced(by: offQ0))
    let q1Ptr = UnsafeRawPointer(base.advanced(by: offQ1))
    
    let p1 = p1Ptr.loadUnaligned(fromByteOffset: 0, as: SIMD16<Int16>.self)
    let p0 = p0Ptr.loadUnaligned(fromByteOffset: 0, as: SIMD16<Int16>.self)
    let q0 = q0Ptr.loadUnaligned(fromByteOffset: 0, as: SIMD16<Int16>.self)
    let q1 = q1Ptr.loadUnaligned(fromByteOffset: 0, as: SIMD16<Int16>.self)
    
    let (newP0, newQ0) = deblockComputeFilter(p1: p1, p0: p0, q0: q0, q1: q1, tc: tc, beta: beta)
    
    let p0MutPtr = UnsafeMutableRawPointer(base.advanced(by: offP0))
    let q0MutPtr = UnsafeMutableRawPointer(base.advanced(by: offQ0))
    
    p0MutPtr.storeBytes(of: newP0, toByteOffset: 0, as: SIMD16<Int16>.self)
    q0MutPtr.storeBytes(of: newQ0, toByteOffset: 0, as: SIMD16<Int16>.self)
}

@inline(__always)
private func deblockFilterHorizontalEdgeScalar(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, count: Int, tc: Int16, beta: Int32) {
    let betah = beta >> 1
    var offset = (y * width) + x
    for _ in 0..<count {
        let p1 = base[offset - 2 * width]
        var p0 = base[offset - 1 * width]
        var q0 = base[offset + 0 * width]
        let q1 = base[offset + 1 * width]
        
        let delta = Int32(q0) - Int32(p0)
        let absDelta = if delta < 0 { -1 * delta } else { delta }
        if absDelta < beta {
            let deltaP = Int32(p1) - Int32(p0)
            let deltaQ = Int32(q1) - Int32(q0)
            let absP = if deltaP < 0 { -1 * deltaP } else { deltaP }
            let absQ = if deltaQ < 0 { -1 * deltaQ } else { deltaQ }
            if absP < betah && absQ < betah {
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
private func deblockComputeFilter(p1: SIMD16<Int16>, p0: SIMD16<Int16>, q0: SIMD16<Int16>, q1: SIMD16<Int16>, tc: Int16, beta: Int32) -> (SIMD16<Int16>, SIMD16<Int16>) {
    let betah = Int16(beta >> 1)
    let beta16 = Int16(beta)
    
    let betaV = SIMD16<Int16>(repeating: beta16)
    let betahV = SIMD16<Int16>(repeating: betah)
    let tcV = SIMD16<Int16>(repeating: tc)
    let ntcV = .zero &- tcV
    
    let delta = q0 &- p0
    let absDelta = delta.replacing(with: .zero &- delta, where: delta .< 0)
    
    let deltaP = p1 &- p0
    let deltaQ = q1 &- q0
    let absP = deltaP.replacing(with: .zero &- deltaP, where: deltaP .< 0)
    let absQ = deltaQ.replacing(with: .zero &- deltaQ, where: deltaQ .< 0)
    
    let mask = (absDelta .< betaV) .& (absP .< betahV) .& (absQ .< betahV)
    
    var d = (delta &+ 1) &>> 1
    
    d.replace(with: tcV, where: tcV .< d)
    d.replace(with: ntcV, where: d .< ntcV)
    
    var newP0 = p0
    var newQ0 = q0
    
    newP0.replace(with: p0 &+ d, where: mask)
    newQ0.replace(with: q0 &- d, where: mask)
    
    return (newP0, newQ0)
}

// MARK: - Intra/Inter Boundary Blend

@inline(__always)
func blendIntraInterBoundaryLuma32(plane: inout [Int16], mvs: MotionVectors, width: Int, height: Int) {
    let colCount = (width + 31) / 32
    let rowCount = (height + 31) / 32
    
    withUnsafePointers(mut: &plane) { base in
        for row in 0..<rowCount {
            for col in 0..<colCount {
                let idx = row * colCount + col
                let isIntra = mvs.dx[idx] == 32767 && mvs.dy[idx] == 32767
                if isIntra {
                    let bx = col * 32
                    let by = row * 32
                    
                    // Left neighbor
                    let leftNotIntra = if 0 < col { mvs.dx[idx - 1] != 32767 || mvs.dy[idx - 1] != 32767 } else { false }
                    if leftNotIntra {
                        let safeH = min(32, height - by)
                        if 4 <= bx && bx + 3 < width {
                            blendVerticalEdgeLuma32(base: base, width: width, x: bx, y: by, height: safeH)
                        }
                    }
                    // Right neighbor
                    let rightNotIntra = if col < colCount - 1 { mvs.dx[idx + 1] != 32767 || mvs.dy[idx + 1] != 32767 } else { false }
                    if rightNotIntra {
                        let bxRight = bx + 32
                        if 4 <= bxRight && bxRight + 3 < width {
                            let safeH = min(32, height - by)
                            blendVerticalEdgeLuma32(base: base, width: width, x: bxRight, y: by, height: safeH)
                        }
                    }
                    // Top neighbor
                    let topNotIntra = if 0 < row { mvs.dx[idx - colCount] != 32767 || mvs.dy[idx - colCount] != 32767 } else { false }
                    if topNotIntra {
                        let safeW = min(32, width - bx)
                        if 4 <= by && by + 3 < height {
                            blendHorizontalEdgeLuma32(base: base, width: width, x: bx, y: by, widthBlock: safeW)
                        }
                    }
                    // Bottom neighbor
                    let bottomNotIntra = if row < rowCount - 1 { mvs.dx[idx + colCount] != 32767 || mvs.dy[idx + colCount] != 32767 } else { false }
                    if bottomNotIntra {
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
