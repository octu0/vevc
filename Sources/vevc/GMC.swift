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
func estimateGMV(curr: PlaneData420, prev: PlaneData420, blockSize: Int) -> [MotionVector] {
    let dsCurr = downscale4x(pd: curr)
    let dsPrev = downscale4x(pd: prev)

    // Number of blocks in horizontal and vertical directions
    let blocksX = (curr.width + blockSize - 1) / blockSize
    let blocksY = (curr.height + blockSize - 1) / blockSize
    var mvs = [MotionVector](repeating: MotionVector(dx: 0, dy: 0), count: blocksX * blocksY)
    
    // Coarse search range: +- 32 pixels in full res (+- 8 in 1/4 scale)
    let range = 8
    
    // Fine search range in full res: +- 2 pixels around coarse result
    let fineRange = 2

    dsCurr.data.withUnsafeBufferPointer { currPtr in
    guard let pCurrDs = currPtr.baseAddress else { return }
    dsPrev.data.withUnsafeBufferPointer { prevPtr in
    guard let pPrevDs = prevPtr.baseAddress else { return }
    curr.y.withUnsafeBufferPointer { currYPtr in
    guard let pCurrY = currYPtr.baseAddress else { return }
    prev.y.withUnsafeBufferPointer { prevYPtr in
    guard let pPrevY = prevYPtr.baseAddress else { return }
        
        for by in 0..<blocksY {
            for bx in 0..<blocksX {
                let blockId = by * blocksX + bx
                
                // Full res block boundaries
                let startY = by * blockSize
                let endY = min(startY + blockSize, curr.height)
                let startX = bx * blockSize
                let endX = min(startX + blockSize, curr.width)
                
                // 1/4 scale block boundaries
                let dsStartY = startY / 4
                let dsEndY = endY / 4
                let dsStartX = startX / 4
                let dsEndX = endX / 4
                let dsBlockH = dsEndY - dsStartY
                let dsBlockW = dsEndX - dsStartX
                
                if dsBlockW <= 0 || dsBlockH <= 0 { continue }
                
                // ----------------------------------------------------
                // 1. Coarse search in 1/4 scale
                // ----------------------------------------------------
                var bestSAD = Int.max
                var bestDX = 0
                var bestDY = 0
                
                for dy in -range...range {
                    for dx in -range...range {
                        var sad = 0
                        
                        // Evaluate overlapping area in 1/4 scale
                        // We step by 2 in 1/4 scale to keep it fast but accurate
                        for y in stride(from: dsStartY, to: dsEndY, by: 2) {
                            let srcY = y - dy
                            if srcY < 0 || srcY >= dsPrev.h { continue }
                            
                            let dstRow = y * dsCurr.w
                            let srcRow = srcY * dsPrev.w
                            
                            // Calculate safe X boundaries for SAD
                            let scanStartX = max(dsStartX, dsStartX + dx)
                            let scanEndX = min(dsEndX, dsPrev.w + dx, dsCurr.w)
                            
                            if scanStartX < scanEndX {
                                let count = scanEndX - scanStartX
                                sad &+= calculateSAD(
                                    p1: pCurrDs.advanced(by: dstRow + scanStartX),
                                    p2: pPrevDs.advanced(by: srcRow + scanStartX - dx),
                                    count: count
                                )
                            }
                        }
                        
                        // Small penalty to prefer static (0,0) or small motions
                        sad &+= (abs(dx) + abs(dy)) * 2
                        
                        if sad < bestSAD {
                            bestSAD = sad
                            bestDX = dx
                            bestDY = dy
                        }
                    }
                }
                
                // ----------------------------------------------------
                // 2. Fine search in full resolution (Refinement)
                // ----------------------------------------------------
                let cDX = bestDX * 4
                let cDY = bestDY * 4
                
                var fineBestSAD = Int.max
                var fineBestDX = cDX
                var fineBestDY = cDY
                
                let stepY = 2 // Extremely dense search but only +-2 range for this tiny MB
                for fineDy in (cDY - fineRange)...(cDY + fineRange) {
                    for fineDx in (cDX - fineRange)...(cDX + fineRange) {
                        var sad = 0
                        
                        for y in stride(from: startY, to: endY, by: stepY) {
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
                        
                        sad &+= (abs(fineDx) + abs(fineDy)) * 2
                        
                        if sad < fineBestSAD {
                            fineBestSAD = sad
                            fineBestDX = fineDx
                            fineBestDY = fineDy
                        }
                    }
                }
                
                mvs[blockId] = MotionVector(dx: fineBestDX, dy: fineBestDY)
            }
        }
    }}}}
    
    return mvs
}

// ShiftPlane is replaced by HBMA implementation in Encode.swift/Decode.swift
