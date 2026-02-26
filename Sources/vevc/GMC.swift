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
func downscale8x(pd: PlaneData420) -> (data: [Int16], w: Int, h: Int) {
    let w = pd.width / 8
    let h = pd.height / 8
    var out = [Int16](repeating: 0, count: w * h)
    
    pd.y.withUnsafeBufferPointer { yPtr in
        guard let pY = yPtr.baseAddress else { return }
        out.withUnsafeMutableBufferPointer { outPtr in
            guard let pOut = outPtr.baseAddress else { return }
            
            let pdWidth = pd.width
            for y in 0..<h {
                let py = y * 8
                let outRow = y * w
                for x in 0..<w {
                    let px = x * 8
                    var sum: Int = 0
                    
                    // Simple average of 8x8 block using auto-vectorized loop
                    for dy in 0..<8 {
                        let off = (py + dy) * pdWidth + px
                        sum &+= Int(pY[off]) &+ Int(pY[off+1]) &+ Int(pY[off+2]) &+ Int(pY[off+3]) &+
                                Int(pY[off+4]) &+ Int(pY[off+5]) &+ Int(pY[off+6]) &+ Int(pY[off+7])
                    }
                    pOut[outRow + x] = Int16(sum / 64)
                }
            }
        }
    }
    return (out, w, h)
}

@inline(__always)
func estimateGMV(curr: PlaneData420, prev: PlaneData420) -> (dx: Int, dy: Int) {
    let dsCurr = downscale8x(pd: curr)
    let dsPrev = downscale8x(pd: prev)

    var bestSAD = Int.max
    var bestDX = 0
    var bestDY = 0
    
    // Coarse search range: +- 32 pixels in full res (+- 4 in 1/8 scale)
    let range = 4
    
    dsCurr.data.withUnsafeBufferPointer { currPtr in
        guard let pCurr = currPtr.baseAddress else { return }
        dsPrev.data.withUnsafeBufferPointer { prevPtr in
            guard let pPrev = prevPtr.baseAddress else { return }
            
            for dy in -range...range {
                for dx in -range...range {
                    var sad = 0
                    
                    // Evaluate overlapping area in 1/8 scale sparsely for speed
                    for y in stride(from: 0, to: dsCurr.h, by: 32) {
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
                    
                    // Penalty for motion to prefer static background
                    sad &+= (abs(dx) + abs(dy)) * 8
                    
                    if sad < bestSAD {
                        bestSAD = sad
                        bestDX = dx
                        bestDY = dy
                    }
                }
            }
        }
    }

    let cDX = bestDX * 8
    let cDY = bestDY * 8
    
    var fineBestSAD = Int.max
    var fineBestDX = cDX
    var fineBestDY = cDY
    
    // Fine search in full resolution with sparse sampling (16x16 grid effectively)
    // Coarse search at 1/8 scale got us within 8 pixels, so range 4 is sufficient
    // to find the best pixel-aligned match.
    curr.y.withUnsafeBufferPointer { currYPtr in
        guard let pCurrY = currYPtr.baseAddress else { return }
        prev.y.withUnsafeBufferPointer { prevYPtr in
            guard let pPrevY = prevYPtr.baseAddress else { return }
            
            let marginY = 16
            let marginX = 16
            let stepY = 64 // Evaluate only 1 in 64 lines for speed
            
            for dy in (cDY - 4)...(cDY + 4) {
                for dx in (cDX - 4)...(cDX + 4) {
                    var sad = 0
                    
                    let startY = max(marginY, marginY + dy)
                    let endY = min(curr.height - marginY, prev.height - marginY + dy)
                    
                    let startX = max(marginX, marginX + dx)
                    let endX = min(curr.width - marginX, prev.width - marginX + dx)
                    let countX = endX - startX
                    
                    if startY < endY && countX > 0 {
                        for y in stride(from: startY, to: endY, by: stepY) {
                            let dstRow = y * curr.width
                            let srcRow = (y - dy) * prev.width
                            
                            sad &+= calculateSAD(
                                p1: pCurrY.advanced(by: dstRow + startX),
                                p2: pPrevY.advanced(by: srcRow + startX - dx),
                                count: countX
                            )
                        }
                    }
                    
                    sad &+= (abs(dx) + abs(dy)) * 2
                    
                    if sad < fineBestSAD {
                        fineBestSAD = sad
                        fineBestDX = dx
                        fineBestDY = dy
                    }
                }
            }
        }
    }
    
    return (fineBestDX, fineBestDY)
}

@inline(__always)
func shiftPlane(_ plane: PlaneData420, dx: Int, dy: Int) -> PlaneData420 {
    if dx == 0 && dy == 0 { return plane }
    
    @Sendable
    func shift(data: [Int16], w: Int, h: Int, sX: Int, sY: Int) -> [Int16] {
        if w == 0 || h == 0 { return data }
        
        var out = [Int16](repeating: 0, count: w * h)
        
        data.withUnsafeBufferPointer { dPtr in
            guard let pData = dPtr.baseAddress else { return }
            out.withUnsafeMutableBufferPointer { oPtr in
                guard let pOut = oPtr.baseAddress else { return }
                
                let eX = ((sX % w) + w) % w
                let eY = ((sY % h) + h) % h
                
                for dstY in 0..<h {
                    let dstRow = dstY * w
                    let srcY = (dstY - eY + h) % h
                    let srcRow = srcY * w
                    
                    if eX == 0 {
                        pOut.advanced(by: dstRow).update(from: pData.advanced(by: srcRow), count: w)
                    } else {
                        let part1Len = eX
                        let part2Len = w - eX
                        
                        pOut.advanced(by: dstRow).update(from: pData.advanced(by: srcRow + w - eX), count: part1Len)
                        pOut.advanced(by: dstRow + eX).update(from: pData.advanced(by: srcRow), count: part2Len)
                    }
                }
            }
        }
        
        return out
    }
    
    let results = ConcurrentBox([[Int16]](repeating: [], count: 3))
    
    DispatchQueue.concurrentPerform(iterations: 3) { index in
        switch index {
        case 0: results.value[0] = shift(data: plane.y, w: plane.width, h: plane.height, sX: dx, sY: dy)
        case 1: results.value[1] = shift(data: plane.cb, w: (plane.width + 1) / 2, h: (plane.height + 1) / 2, sX: dx / 2, sY: dy / 2)
        case 2: results.value[2] = shift(data: plane.cr, w: (plane.width + 1) / 2, h: (plane.height + 1) / 2, sX: dx / 2, sY: dy / 2)
        default: break
        }
    }
    
    return PlaneData420(width: plane.width, height: plane.height, y: results.value[0], cb: results.value[1], cr: results.value[2])
}
