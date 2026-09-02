// MARK: - Deblocking Filter

// Smooths block boundary discontinuities to suppress block noise.
// tc/beta parameters use non-linear scaling based on quantization step.
/// In-place applies deblocking filter to the reconstructed image (32x32 block resolution).
@inline(__always)
func applyDeblockingFilter32(plane: inout [Int16], width: Int, height: Int, qStep: Int) {
    if qStep <= 3 { return }
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
        if tc == 0 && beta == 0 { return }
        
        let hFast = (height / 32) * 32
        let wFast = (width / 32) * 32
        let hRem = height - hFast
        let wRem = width - wFast
        
        // Vertical Edges
        for x in stride(from: 32, to: width, by: 32) {
            if width <= x + 1 { continue }  // skip boundary columns where reading the right neighbor q1 would run out of range
            for y in stride(from: 0, to: hFast, by: 32) {
                deblockFilterVerticalEdge(base: base, width: width, x: x, y: y, count: 32, tc: tc, beta: beta)
            }
            deblockFilterVerticalEdge(base: base, width: width, x: x, y: hFast, count: hRem, tc: tc, beta: beta)
        }
        
        // Horizontal Edges
        for y in stride(from: 32, to: height, by: 32) {
            if height <= y + 1 { continue }  // skip boundary rows where reading the lower neighbor q1's row would run out of range
            for x in stride(from: 0, to: wFast, by: 32) {
                deblockFilterHorizontalEdge(base: base, width: width, x: x, y: y, count: 32, tc: tc, beta: beta)
            }
            deblockFilterHorizontalEdge(base: base, width: width, x: wFast, y: y, count: wRem, tc: tc, beta: beta)
        }
    }
}

/// In-place applies deblocking filter to the reconstructed image (32x32 block resolution), with Intra/Inter boundary enhancement.
@inline(__always)
func applyDeblockingFilter32(plane: inout [Int16], width: Int, height: Int, qStep: Int, mvs: MotionVectors, skipMap: [BlockMode]? = nil) {
    if qStep <= 3 { return }
    guard mvs.isEmpty != true else {
        applyDeblockingFilter32(plane: &plane, width: width, height: height, qStep: qStep)
        return
    }
    
    withUnsafePointers(mut: &plane, mvs.dx) { base, mvDxBase in
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
        
        let rawETc = ((qStep / 2) + 3) * 2
        let enhancedTc: Int16 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int16((rawETc * (qStep - 3)) / 12)
            default: Int16(min(30, rawETc))
        }
        let rawEBeta = (qStep + 6) * 2
        let enhancedBeta: Int32 = switch true {
            case qStep <= 3: 0
            case qStep <= 15: Int32((rawEBeta * (qStep - 3)) / 12)
            default: Int32(min(100, rawEBeta))
        }
        
        if defaultTc == 0 && defaultBeta == 0 && enhancedTc == 0 && enhancedBeta == 0 { return }
        
        let colCount = (width + 31) / 32
        let rowCount = (height + 31) / 32
        
        let hFast = (height / 32) * 32
        let wFast = (width / 32) * 32
        let hRem = height - hFast
        let wRem = width - wFast
        let mvCount = mvs.dx.count
            
        // Vertical Edges
        for col in 1..<colCount {
            let x = col * 32
            if width <= x + 1 { continue }  // skip boundary columns where reading the right neighbor q1 would run out of range
            for row in 0..<rowCount {
                let y = row * 32
                let idx = row * colCount + col
                let leftIdx = idx - 1
                
                if let sm = skipMap, idx < sm.count, leftIdx < sm.count {
                    let leftMode = sm[leftIdx]
                    let rightMode = sm[idx]
                    if leftMode != .inter && rightMode != .inter && leftMode == rightMode {
                        continue
                    }
                }
                
                let leftDx: Int16 = if leftIdx < mvCount { mvDxBase[leftIdx] } else { 0 }
                let rightDx: Int16 = if idx < mvCount { mvDxBase[idx] } else { 0 }
                
                let leftIsIntra = leftDx == 32767
                let rightIsIntra = rightDx == 32767
                
                let isIntraBoundary = leftIsIntra || rightIsIntra
                let tc = if isIntraBoundary { enhancedTc } else { defaultTc }
                let beta = if isIntraBoundary { enhancedBeta } else { defaultBeta }
                
                if tc != 0 || beta != 0 {
                    if y < hFast {
                        deblockFilterVerticalEdge(base: base, width: width, x: x, y: y, count: 32, tc: tc, beta: beta)
                    } else {
                        let safeH = min(hRem, height - y)
                        deblockFilterVerticalEdge(base: base, width: width, x: x, y: y, count: safeH, tc: tc, beta: beta)
                    }
                }
            }
        }
        
        // Horizontal Edges
        for row in 1..<rowCount {
            let y = row * 32
            if height <= y + 1 { continue }  // skip boundary rows where reading the lower neighbor q1's row would run out of range
            for col in 0..<colCount {
                let x = col * 32
                let idx = row * colCount + col
                let topIdx = idx - colCount
                
                if let sm = skipMap, idx < sm.count, topIdx < sm.count {
                    let topMode = sm[topIdx]
                    let bottomMode = sm[idx]
                    if topMode != .inter && bottomMode != .inter && topMode == bottomMode {
                        continue
                    }
                }
                
                let topDx: Int16 = if topIdx < mvCount { mvDxBase[topIdx] } else { 0 }
                let bottomDx: Int16 = if idx < mvCount { mvDxBase[idx] } else { 0 }
                
                let topIsIntra = topDx == 32767
                let bottomIsIntra = bottomDx == 32767
                
                let isIntraBoundary = topIsIntra || bottomIsIntra
                let tc = if isIntraBoundary { enhancedTc } else { defaultTc }
                let beta = if isIntraBoundary { enhancedBeta } else { defaultBeta }
                
                if tc != 0 || beta != 0 {
                    if x < wFast {
                        deblockFilterHorizontalEdge(base: base, width: width, x: x, y: y, count: 32, tc: tc, beta: beta)
                    } else {
                        let safeW = min(wRem, width - x)
                        deblockFilterHorizontalEdge(base: base, width: width, x: x, y: y, count: safeW, tc: tc, beta: beta)
                    }
                }
            }
        }
    }
}

@inline(__always)
func applyDeblockingFilterChroma16(plane: inout [Int16], width: Int, height: Int, qStep: Int, mvs: MotionVectors, skipMap: [BlockMode]? = nil) {
    if qStep <= 3 { return }
    guard mvs.isEmpty != true else {
        applyDeblockingFilter16(plane: &plane, width: width, height: height, qStep: qStep)
        return
    }
    
    withUnsafePointers(mut: &plane, mvs.dx) { base, mvDxBase in
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
        
        if defaultTc == 0 && defaultBeta == 0 && enhancedTc == 0 && enhancedBeta == 0 { return }
        
        let colCountC = (width + 15) / 16
        let rowCountC = (height + 15) / 16
        let mvColCount = colCountC 
        
        let hFast = (height / 16) * 16
        let wFast = (width / 16) * 16
        let hRem = height - hFast
        let wRem = width - wFast
        let mvCount = mvs.dx.count
            
        // Vertical Edges
        for col in 1..<colCountC {
            let x = col * 16
            if width <= x + 1 { continue }  // skip boundary columns where reading the right neighbor q1 would run out of range
            for row in 0..<rowCountC {
                let y = row * 16
                let idx = row * mvColCount + col
                let leftIdx = idx - 1
                
                if let sm = skipMap, idx < sm.count, leftIdx < sm.count {
                    let leftMode = sm[leftIdx]
                    let rightMode = sm[idx]
                    if leftMode != .inter && rightMode != .inter && leftMode == rightMode {
                        continue
                    }
                }
                
                let leftDx: Int16 = if leftIdx < mvCount { mvDxBase[leftIdx] } else { 0 }
                let rightDx: Int16 = if idx < mvCount { mvDxBase[idx] } else { 0 }
                
                let leftIsIntra = leftDx == 32767
                let rightIsIntra = rightDx == 32767
                
                let isIntraBoundary = leftIsIntra || rightIsIntra
                let tc = if isIntraBoundary { enhancedTc } else { defaultTc }
                let beta = if isIntraBoundary { enhancedBeta } else { defaultBeta }
                
                if tc != 0 || beta != 0 {
                    if y < hFast {
                        deblockFilterVerticalEdge(base: base, width: width, x: x, y: y, count: 16, tc: tc, beta: beta)
                    } else {
                        let safeH = min(hRem, height - y)
                        deblockFilterVerticalEdge(base: base, width: width, x: x, y: y, count: safeH, tc: tc, beta: beta)
                    }
                }
            }
        }
        
        // Horizontal Edges
        for row in 1..<rowCountC {
            let y = row * 16
            if height <= y + 1 { continue }  // skip boundary rows where reading the lower neighbor q1's row would run out of range
            for col in 0..<colCountC {
                let x = col * 16
                let idx = row * mvColCount + col
                let topIdx = idx - mvColCount
                
                if let sm = skipMap, idx < sm.count, topIdx < sm.count {
                    let topMode = sm[topIdx]
                    let bottomMode = sm[idx]
                    if topMode != .inter && bottomMode != .inter && topMode == bottomMode {
                        continue
                    }
                }
                
                let topDx: Int16 = if topIdx < mvCount { mvDxBase[topIdx] } else { 0 }
                let bottomDx: Int16 = if idx < mvCount { mvDxBase[idx] } else { 0 }
                
                let topIsIntra = topDx == 32767
                let bottomIsIntra = bottomDx == 32767
                
                let isIntraBoundary = topIsIntra || bottomIsIntra
                let tc = if isIntraBoundary { enhancedTc } else { defaultTc }
                let beta = if isIntraBoundary { enhancedBeta } else { defaultBeta }
                
                if tc != 0 || beta != 0 {
                    if x < wFast {
                        deblockFilterHorizontalEdge(base: base, width: width, x: x, y: y, count: 16, tc: tc, beta: beta)
                    } else {
                        let safeW = min(wRem, width - x)
                        deblockFilterHorizontalEdge(base: base, width: width, x: x, y: y, count: safeW, tc: tc, beta: beta)
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
            if width <= x + 1 { continue }  // skip boundary columns where reading the right neighbor q1 would run out of range
            for y in stride(from: 0, to: hFast, by: 16) {
                deblockFilterVerticalEdge(base: base, width: width, x: x, y: y, count: 16, tc: tc, beta: beta)
            }
            deblockFilterVerticalEdge(base: base, width: width, x: x, y: hFast, count: hRem, tc: tc, beta: beta)
        }
        
        // Horizontal Edges
        for y in stride(from: 16, to: height, by: 16) {
            if height <= y + 1 { continue }  // skip boundary rows where reading the lower neighbor q1's row would run out of range
            for x in stride(from: 0, to: wFast, by: 16) {
                deblockFilterHorizontalEdge(base: base, width: width, x: x, y: y, count: 16, tc: tc, beta: beta)
            }
            deblockFilterHorizontalEdge(base: base, width: width, x: wFast, y: y, count: wRem, tc: tc, beta: beta)
        }
    }
}

/// In-place applies deblocking filter to the reconstructed image with a customizable block size boundary.
@inline(__always)
func applyDeblockingFilterN(plane: inout [Int16], width: Int, height: Int, qStep: Int, blockSize: Int) {
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
        
        // Vertical Edges
        for x in stride(from: blockSize, to: width, by: blockSize) {
            if 2 <= x && x + 1 < width {
                deblockFilterVerticalEdge(base: base, width: width, x: x, y: 0, count: height, tc: tc, beta: beta)
            }
        }
        
        // Horizontal Edges
        for y in stride(from: blockSize, to: height, by: blockSize) {
            if 2 <= y && y + 1 < height {
                deblockFilterHorizontalEdge(base: base, width: width, x: 0, y: y, count: width, tc: tc, beta: beta)
            }
        }
    }
}

@inline(__always)
private func deblockFilterVerticalEdge(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, count: Int, tc: Int16, beta: Int32) {
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
private func deblockFilterHorizontalEdge(base: UnsafeMutablePointer<Int16>, width: Int, x: Int, y: Int, count: Int, tc: Int16, beta: Int32) {
    if tc == 0 && beta == 0 { return }
    var curX = x
    let endX = x + count
    let p1Row = base.advanced(by: (y - 2) * width)
    let p0Row = base.advanced(by: (y - 1) * width)
    let q0Row = base.advanced(by: y * width)
    let q1Row = base.advanced(by: (y + 1) * width)
    
    let betah = Int16(beta >> 1)
    let beta16 = Int16(beta)
    let betaV = SIMD16<Int16>(repeating: beta16)
    let betahV = SIMD16<Int16>(repeating: betah)
    let tcV = SIMD16<Int16>(repeating: tc)
    let ntcV = .zero &- tcV
    
    while curX &+ 16 <= endX {
        let p1 = UnsafeRawPointer(p1Row.advanced(by: curX)).loadUnaligned(as: SIMD16<Int16>.self)
        let p0 = UnsafeRawPointer(p0Row.advanced(by: curX)).loadUnaligned(as: SIMD16<Int16>.self)
        let q0 = UnsafeRawPointer(q0Row.advanced(by: curX)).loadUnaligned(as: SIMD16<Int16>.self)
        let q1 = UnsafeRawPointer(q1Row.advanced(by: curX)).loadUnaligned(as: SIMD16<Int16>.self)
        
        let (newP0, newQ0) = deblockComputeFilter(p1: p1, p0: p0, q0: q0, q1: q1, betaV: betaV, betahV: betahV, tcV: tcV, ntcV: ntcV)
        
        UnsafeMutableRawPointer(p0Row.advanced(by: curX)).storeBytes(of: newP0, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(q0Row.advanced(by: curX)).storeBytes(of: newQ0, as: SIMD16<Int16>.self)
        curX &+= 16
    }
    
    if curX == endX { return }
    
    while curX < endX {
        let p1 = p1Row[curX]
        var p0 = p0Row[curX]
        var q0 = q0Row[curX]
        let q1 = q1Row[curX]
        
        let delta = Int32(q0) - Int32(p0)
        let absDelta = if delta < 0 { -1 * delta } else { delta }
        if absDelta < beta {
            let deltaP = Int32(p1) - Int32(p0)
            let deltaQ = Int32(q1) - Int32(q0)
            let absP = if deltaP < 0 { -1 * deltaP } else { deltaP }
            let absQ = if deltaQ < 0 { -1 * deltaQ } else { deltaQ }
            if absP < Int32(betah) && absQ < Int32(betah) {
                var d = (delta + 1) >> 1
                let t = Int32(tc)
                if t < d { d = t }
                if d < (-1 * t) { d = (-1 * t) }
                
                let d16 = Int16(d)
                p0 = p0 &+ d16
                q0 = q0 &- d16
                
                p0Row[curX] = p0
                q0Row[curX] = q0
            }
        }
        curX += 1
    }
}

@inline(__always)
private func deblockComputeFilter(
    p1: SIMD16<Int16>, p0: SIMD16<Int16>, q0: SIMD16<Int16>, q1: SIMD16<Int16>,
    betaV: SIMD16<Int16>, betahV: SIMD16<Int16>, tcV: SIMD16<Int16>, ntcV: SIMD16<Int16>
) -> (SIMD16<Int16>, SIMD16<Int16>) {
    let delta = q0 &- p0
    let absDelta = pointwiseMax(q0, p0) &- pointwiseMin(q0, p0)
    let absP = pointwiseMax(p1, p0) &- pointwiseMin(p1, p0)
    let absQ = pointwiseMax(q1, q0) &- pointwiseMin(q1, q0)
    
    let mask = (absDelta .< betaV) .& (absP .< betahV) .& (absQ .< betahV)
    
    let rawD = (delta &+ 1) &>> 1
    let d = pointwiseMax(ntcV, pointwiseMin(tcV, rawD))
    
    let newP0 = p0.replacing(with: p0 &+ d, where: mask)
    let newQ0 = q0.replacing(with: q0 &- d, where: mask)
    
    return (newP0, newQ0)
}
