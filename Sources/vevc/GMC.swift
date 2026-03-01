import Foundation

@inline(__always)
func calculateSAD(p1: UnsafePointer<Int16>, p2: UnsafePointer<Int16>, count: Int) -> Int {
    var sad: UInt = 0
    let step = 8
    let end = count - (count % step)
    var i = 0
    
    // Explicit unrolling to help auto-vectorizer
    while i < end {
        let diff0 = Int(p1[i]) - Int(p2[i])
        let diff1 = Int(p1[i+1]) - Int(p2[i+1])
        let diff2 = Int(p1[i+2]) - Int(p2[i+2])
        let diff3 = Int(p1[i+3]) - Int(p2[i+3])
        let diff4 = Int(p1[i+4]) - Int(p2[i+4])
        let diff5 = Int(p1[i+5]) - Int(p2[i+5])
        let diff6 = Int(p1[i+6]) - Int(p2[i+6])
        let diff7 = Int(p1[i+7]) - Int(p2[i+7])

        sad &+= UInt(diff0 > 0 ? diff0 : -diff0)
        sad &+= UInt(diff1 > 0 ? diff1 : -diff1)
        sad &+= UInt(diff2 > 0 ? diff2 : -diff2)
        sad &+= UInt(diff3 > 0 ? diff3 : -diff3)
        sad &+= UInt(diff4 > 0 ? diff4 : -diff4)
        sad &+= UInt(diff5 > 0 ? diff5 : -diff5)
        sad &+= UInt(diff6 > 0 ? diff6 : -diff6)
        sad &+= UInt(diff7 > 0 ? diff7 : -diff7)

        i += step
    }
    
    while i < count {
        let diff = Int(p1[i]) - Int(p2[i])
        sad &+= UInt(diff > 0 ? diff : -diff)
        i += 1
    }
    
    return Int(sad)
}

@inline(__always)
func downscale4x(pd: PlaneData420) -> (data: [Int16], w: Int, h: Int) {
    let w = pd.width / 4
    let h = pd.height / 4
    var out = [Int16](repeating: 0, count: w * h)
    
    pd.y.withUnsafeBufferPointer { yPtr in
        guard let pY = yPtr.baseAddress else { return }
        out.withUnsafeMutableBufferPointer { outPtr in
            guard let pOut = outPtr.baseAddress else { return }
            
            let pdWidth = pd.width
            for y in 0..<h {
                let py = y * 4
                let outRow = y * w
                for x in 0..<w {
                    let px = x * 4
                    var sum: Int = 0
                    
                    // Simple average of 4x4 block using auto-vectorized loop
                    for dy in 0..<4 {
                        let off = (py + dy) * pdWidth + px
                        sum &+= Int(pY[off]) &+ Int(pY[off+1]) &+ Int(pY[off+2]) &+ Int(pY[off+3])
                    }
                    pOut[outRow + x] = Int16(sum / 16)
                }
            }
        }
    }
    return (out, w, h)
}

public struct MotionVector {
    public let dx: Int
    public let dy: Int
}

@inline(__always)
func estimateGMV_old(curr: PlaneData420, prev: PlaneData420) -> (dx: Int, dy: Int) {
    let dsCurr = downscale4x(pd: curr)
    let dsPrev = downscale4x(pd: prev)

    var bestSAD = Int.max
    var bestDX = 0
    var bestDY = 0
    
    // Coarse search range: +- 32 pixels in full res (+- 8 in 1/4 scale)
    let range = 4
    
    dsCurr.data.withUnsafeBufferPointer { currPtr in
        guard let pCurr = currPtr.baseAddress else { return }
        dsPrev.data.withUnsafeBufferPointer { prevPtr in
            guard let pPrev = prevPtr.baseAddress else { return }
            
            for dy in -range...range {
                for dx in -range...range {
                    var sad = 0
                    
                    // Evaluate overlapping area in 1/4 scale sparsely for speed
                    // (stride by 8 in 1/4 scale means stride by 32 in full res)
                    for y in stride(from: 0, to: dsCurr.h, by: 8) {
                        let srcY = y - dy
                        if srcY < 0 || srcY >= dsPrev.h { continue }
                        
                        let dstRow = y * dsCurr.w
                        let srcRow = srcY * dsPrev.w
                        
                        let safeStartX = max(0, dx)
                        let safeEndX = min(dsCurr.w, dsPrev.w + dx)
                        
                        if safeStartX < safeEndX {
                            let count = safeEndX - safeStartX
                            sad &+= calculateSAD(
                                p1: pCurr.advanced(by: dstRow + safeStartX), 
                                p2: pPrev.advanced(by: srcRow + safeStartX - dx), 
                                count: count
                            )
                        }
                    }
                    
                    if sad < bestSAD {
                        bestSAD = sad
                        bestDX = dx
                        bestDY = dy
                    }
                }
            }
        }
    }

    let cDX = bestDX * 4
    let cDY = bestDY * 4
    
    var fineBestSAD = Int.max
    var fineBestDX = cDX
    var fineBestDY = cDY
    
    // Fine search in full resolution with sparse sampling (16x16 grid effectively)
    // Coarse search at 1/4 scale got us within 4 pixels, so range 2 is sufficient
    // to find the best pixel-aligned match, saving massive computation time.
    curr.y.withUnsafeBufferPointer { currYPtr in
        guard let pCurrY = currYPtr.baseAddress else { return }
        prev.y.withUnsafeBufferPointer { prevYPtr in
            guard let pPrevY = prevYPtr.baseAddress else { return }
            
            let marginY = 16
            let marginX = 16
            let stepY = 128 // Evaluate only 1 in 128 lines for extreme speed
            
            for dy in (cDY - 2)...(cDY + 2) {
                for dx in (cDX - 2)...(cDX + 2) {
                    var sad = 0
                    
                    let startY = max(marginY, marginY + dy)
                    let endY = min(curr.height - marginY, prev.height - marginY + dy)
                    
                    if startY >= endY { continue }
                    
                    for y in stride(from: startY, to: endY, by: stepY) {
                        let srcY = y - dy
                        let dstRow = y * curr.width
                        let srcRow = srcY * prev.width
                        
                        let safeStartX = max(marginX, marginX + dx)
                        let safeEndX = min(curr.width - marginX, prev.width - marginX + dx)
                        
                        if safeStartX < safeEndX {
                            let count = safeEndX - safeStartX
                            sad &+= calculateSAD(
                                p1: pCurrY.advanced(by: dstRow + safeStartX), 
                                p2: pPrevY.advanced(by: srcRow + safeStartX - dx), 
                                count: count
                            )
                        }
                    }
                    
                    if sad < fineBestSAD {
                        fineBestSAD = sad
                        fineBestDX = dx
                        fineBestDY = dy
                    }
                }
            }
        }
    }
    
    return (dx: fineBestDX, dy: fineBestDY)
}

@inline(__always)
func estimateGMV(curr: PlaneData420, prev: PlaneData420, blockSize: Int) -> [MotionVector] {
    let dsCurr = downscale4x(pd: curr)
    let dsPrev = downscale4x(pd: prev)

    // Number of blocks in horizontal and vertical directions
    let blocksX = (curr.width + blockSize - 1) / blockSize
    let blocksY = (curr.height + blockSize - 1) / blockSize
    var mvs = [MotionVector](repeating: MotionVector(dx: 0, dy: 0), count: blocksX * blocksY)
    
    // Coarse search range for Global MV: +- 32 pixels in full res (+- 8 in 1/4 scale)
    let coarseRange = 8
    
    // Fine search range in full res: +- 2 pixels around global coarse result
    let fineRange = 2

    dsCurr.data.withUnsafeBufferPointer { currPtr in
    guard let pCurrDs = currPtr.baseAddress else { return }
    dsPrev.data.withUnsafeBufferPointer { prevPtr in
    guard let pPrevDs = prevPtr.baseAddress else { return }
    curr.y.withUnsafeBufferPointer { currYPtr in
    guard let pCurrY = currYPtr.baseAddress else { return }
    prev.y.withUnsafeBufferPointer { prevYPtr in
    guard let pPrevY = prevYPtr.baseAddress else { return }
        
        // ----------------------------------------------------
        // 1. Global Coarse Search on 1/4 scale image
        // ----------------------------------------------------
        var globalSAD = Int.max
        var globalDX = 0
        var globalDY = 0
        
        for dy in -coarseRange...coarseRange {
            for dx in -coarseRange...coarseRange {
                var sad = 0
                
                // Evaluate overlapping area in 1/4 scale for the entire image
                let dsStartY = 0
                let dsEndY = dsCurr.h
                let dsStartX = 0
                let dsEndX = dsCurr.w
                
                // Step by 8 (same as 1.04% version) for speed and avoiding high-frequency noise in global coarse
                for y in stride(from: dsStartY, to: dsEndY, by: 8) {
                    let srcY = y - dy
                    if srcY < 0 || srcY >= dsPrev.h { continue }
                    
                    let dstRow = y * dsCurr.w
                    let srcRow = srcY * dsPrev.w
                    
                    let safeStartX = max(0, dx)
                    let safeEndX = min(dsCurr.w, dsPrev.w + dx)
                    
                    if safeStartX < safeEndX {
                        let count = safeEndX - safeStartX
                        sad &+= calculateSAD(
                            p1: pCurrDs.advanced(by: dstRow + safeStartX),
                            p2: pPrevDs.advanced(by: srcRow + safeStartX - dx),
                            count: count
                        )
                    }
                }
                
                if sad < globalSAD {
                    globalSAD = sad
                    globalDX = dx
                    globalDY = dy
                }
            }
        }
        
        let baseDX = globalDX * 4
        let baseDY = globalDY * 4

        // ----------------------------------------------------
        // 2. Local Fine Search (Refinement) per MacroBlock
        // ----------------------------------------------------
        for by in 0..<blocksY {
            for bx in 0..<blocksX {
                let blockId = by * blocksX + bx
                
                // Full res block boundaries
                let startY = by * blockSize
                let endY = min(startY + blockSize, curr.height)
                let startX = bx * blockSize
                let endX = min(startX + blockSize, curr.width)
                
                var fineBestSAD = Int.max
                var fineBestDX = baseDX
                var fineBestDY = baseDY
                
                for fineDy in (baseDY - fineRange)...(baseDY + fineRange) {
                    for fineDx in (baseDX - fineRange)...(baseDX + fineRange) {
                        var sad = 0
                        
                        // Exact match inside the block
                        for y in startY..<endY {
                            let srcY = y - fineDy
                            if srcY < 0 || srcY >= prev.height { continue }
                            
                            let dstRow = y * curr.width
                            let srcRow = srcY * prev.width
                            
                            let scanStartX = max(startX, startX + fineDx)
                            let scanEndX = min(endX, prev.width + fineDx, curr.width)
                            
                            if scanStartX < scanEndX {
                                let count = scanEndX - scanStartX
                                sad &+= calculateSAD(
                                    p1: pCurrY.advanced(by: dstRow + scanStartX),
                                    p2: pPrevY.advanced(by: srcRow + scanStartX - fineDx),
                                    count: count
                                )
                            }
                        }
                        
                        // Slight penalty to prefer keeping the global motion vector
                        let diffGlobalX = abs(fineDx - baseDX)
                        let diffGlobalY = abs(fineDy - baseDY)
                        sad &+= (diffGlobalX + diffGlobalY) * 2 // Mild smoothing
                        
                        if sad < fineBestSAD {
                            fineBestSAD = sad
                            fineBestDX = fineDx
                            fineBestDY = fineDy
                        }
                    }
                }
                
                // Test: Force Global MV to everything
                // mvs[blockId] = MotionVector(dx: fineBestDX, dy: fineBestDY)
                mvs[blockId] = MotionVector(dx: baseDX, dy: baseDY)
            }
        }
    }}}}

    
    return mvs
}

// ShiftPlane is replaced by HBMA implementation in Encode.swift/Decode.swift
