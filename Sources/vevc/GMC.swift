@inline(__always)
func calculateSAD(p1: UnsafePointer<Int16>, p2: UnsafePointer<Int16>, count: Int) -> Int {
    var sad: Int32 = 0
    var i = 0
    let step = 16
    let end = count - (count % step)
    
    var sum16 = SIMD16<Int32>(repeating: 0)
    let zero = SIMD16<Int16>(repeating: 0)
    
    while i < end {
        let v1 = UnsafeRawPointer(p1.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        let v2 = UnsafeRawPointer(p2.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        
        let d = v1 &- v2
        let absD = d.replacing(with: zero &- d, where: d .< 0)
        
        sum16 &+= SIMD16<Int32>(clamping: absD)
        
        i += step
    }
    
    sad &+= sum16[0] &+ sum16[1] &+ sum16[2] &+ sum16[3]
    sad &+= sum16[4] &+ sum16[5] &+ sum16[6] &+ sum16[7]
    sad &+= sum16[8] &+ sum16[9] &+ sum16[10] &+ sum16[11]
    sad &+= sum16[12] &+ sum16[13] &+ sum16[14] &+ sum16[15]
    
    while i < count {
        let d = Int32(p1[i]) - Int32(p2[i])
        sad &+= abs(d)
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
                    var sum: Int32 = 0
                    
                    // Simple average of 8x8 block using SIMD
                    for dy in 0..<8 {
                        let off = (py + dy) * pdWidth + px
                        let v = UnsafeRawPointer(pY.advanced(by: off)).loadUnaligned(as: SIMD8<Int16>.self)
                        let v32 = SIMD8<Int32>(clamping: v)
                        sum &+= v32[0] &+ v32[1] &+ v32[2] &+ v32[3] &+ v32[4] &+ v32[5] &+ v32[6] &+ v32[7]
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
func shiftPlane(_ plane: PlaneData420, dx: Int, dy: Int) async -> PlaneData420 {
    if dx == 0 && dy == 0 { return plane }
    
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
    
    return await withTaskGroup(of: (Int, [Int16]).self) { group in
        group.addTask { (0, shift(data: plane.y, w: plane.width, h: plane.height, sX: dx, sY: dy)) }
        group.addTask { (1, shift(data: plane.cb, w: (plane.width + 1) / 2, h: (plane.height + 1) / 2, sX: dx / 2, sY: dy / 2)) }
        group.addTask { (2, shift(data: plane.cr, w: (plane.width + 1) / 2, h: (plane.height + 1) / 2, sX: dx / 2, sY: dy / 2)) }
        
        var yOut = [Int16]()
        var cbOut = [Int16]()
        var crOut = [Int16]()
        
        for await (index, out) in group {
            switch index {
            case 0: yOut = out
            case 1: cbOut = out
            case 2: crOut = out
            default: break
            }
        }
        
        return PlaneData420(width: plane.width, height: plane.height, y: yOut, cb: cbOut, cr: crOut)
    }
}
