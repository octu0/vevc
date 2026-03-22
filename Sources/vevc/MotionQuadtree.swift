import Foundation

@inline(__always)
func calculateSAD16x16(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, currStride: Int, prevStride: Int) -> Int {
    var sumVec0 = SIMD16<UInt16>()
    for y in 0..<16 {
        let currRow = pCurr.advanced(by: y * currStride)
        let prevRow = pPrev.advanced(by: y * prevStride)

        let c0 = UnsafeRawPointer(currRow).loadUnaligned(as: SIMD16<Int16>.self)
        let p0 = UnsafeRawPointer(prevRow).loadUnaligned(as: SIMD16<Int16>.self)

        let diff0 = c0 &- p0
        let mask0 = diff0 &>> 15
        let abs0 = (diff0 ^ mask0) &- mask0

        sumVec0 &+= SIMD16<UInt16>(truncatingIfNeeded: abs0)
    }
    return Int(SIMD16<UInt32>(truncatingIfNeeded: sumVec0).wrappedSum())
}

@inline(__always)
func calculateSAD8x8(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, currStride: Int, prevStride: Int) -> Int {
    var sumVec0 = SIMD8<UInt16>()
    for y in 0..<8 {
        let currRow = pCurr.advanced(by: y * currStride)
        let prevRow = pPrev.advanced(by: y * prevStride)

        let cv = UnsafeRawPointer(currRow).loadUnaligned(as: SIMD8<Int16>.self)
        let pv = UnsafeRawPointer(prevRow).loadUnaligned(as: SIMD8<Int16>.self)
        
        let diff0 = cv &- pv
        let mask0 = diff0 &>> 15
        let abs0 = (diff0 ^ mask0) &- mask0
        sumVec0 &+= SIMD8<UInt16>(truncatingIfNeeded: abs0)
    }
    return Int(SIMD8<UInt32>(truncatingIfNeeded: sumVec0).wrappedSum())
}

@inline(__always)
func evaluateMotionQuadtreeNode(
    pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int,
    startX: Int, startY: Int, size: Int, searchRange: Int,
    coarseDX: Int, coarseDY: Int,
    grid: inout MVGrid,
    fracRefBuf: UnsafeMutablePointer<Int16>,
    fracExtBuf: UnsafeMutablePointer<Int16>
) -> (node: MotionNode, sad: Int) {
    if startX >= w || startY >= h {
        return (.leaf(mv: .zero), 0)
    }

    let actW = min(size, w - startX)
    let actH = min(size, h - startY)

    let minX = -startX
    let minY = -startY
    let maxX = w - startX - actW
    let maxY = h - startY - actH

    let pmv = grid.getPMV(x: startX, y: startY, w: size)
    
    // Layer0 Coarse MV serves as the search center at the root (64x64) level
    var centerDX = Int(pmv.x)
    var centerDY = Int(pmv.y)
    if size == 64 {
        centerDX = coarseDX
        centerDY = coarseDY
    }
    
    // Extremely tight fine search around the center
    // E.g., for ±3 range, we only check 49 points.
    var bestDX = Int(pmv.x) >> 2
    var bestDY = Int(pmv.y) >> 2
    
    bestDX = max(minX, min(maxX, bestDX))
    bestDY = max(minY, min(maxY, bestDY))

    var bestSAD: Int
    if size == 64 && actW == 64 && actH == 64 {
        bestSAD = calculateSAD64x64(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: (startY + bestDY) * w + startX + bestDX),
            currStride: w, prevStride: w
        )
        let zeroSAD = calculateSAD64x64(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: startY * w + startX),
            currStride: w, prevStride: w
        )
        if zeroSAD < bestSAD { bestSAD = zeroSAD; bestDX = 0; bestDY = 0 }
    } else if size == 32 && actW == 32 && actH == 32 {
        bestSAD = calculateSAD32x32(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: (startY + bestDY) * w + startX + bestDX),
            currStride: w, prevStride: w
        )
        let zeroSAD = calculateSAD32x32(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: startY * w + startX),
            currStride: w, prevStride: w
        )
        if zeroSAD < bestSAD { bestSAD = zeroSAD; bestDX = 0; bestDY = 0 }
    } else if size == 16 && actW == 16 && actH == 16 {
        bestSAD = calculateSAD16x16(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: (startY + bestDY) * w + startX + bestDX),
            currStride: w, prevStride: w
        )
        let zeroSAD = calculateSAD16x16(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: startY * w + startX),
            currStride: w, prevStride: w
        )
        if zeroSAD < bestSAD { bestSAD = zeroSAD; bestDX = 0; bestDY = 0 }
    } else if size == 8 && actW == 8 && actH == 8 {
        bestSAD = calculateSAD8x8(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: (startY + bestDY) * w + startX + bestDX),
            currStride: w, prevStride: w
        )
        let zeroSAD = calculateSAD8x8(
            pCurr: pCurr.advanced(by: startY * w + startX),
            pPrev: pPrev.advanced(by: startY * w + startX),
            currStride: w, prevStride: w
        )
        if zeroSAD < bestSAD { bestSAD = zeroSAD; bestDX = 0; bestDY = 0 }
    } else {
        bestSAD = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: centerDX, dy: centerDY)
        let zeroSAD = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: 0, dy: 0)
        
        // Zero-bias (0,0 is generally preferred if close)
        if zeroSAD <= bestSAD + (actW * actH / 16) {
            bestDX = 0
            bestDY = 0
            bestSAD = zeroSAD
        } else {
            bestDX = centerDX
            bestDY = centerDY
        }
    }
    // Fast Mode: If PMV or ZeroMV already yields an excellent match.
    // Given that typical quantization noise adds SAD=3~5 per pixel, we set the threshold to SAD=8 per pixel.
    // If the error is under this, it's mostly quantization noise and the MV is correct.
    let earlyExitThreshold = size * size * 8
    if earlyExitThreshold < bestSAD {
        var step = max(1, searchRange / 2)
        while 1 <= step {
            var currentBestDX = bestDX
            var currentBestDY = bestDY
            var currentBestSAD = bestSAD

            if step == 1 && size == 64 && actW == 64 && actH == 64 {
                currentBestSAD = calculateSAD64x64(
                    pCurr: pCurr.advanced(by: startY * w + startX),
                    pPrev: pPrev.advanced(by: (startY + bestDY) * w + startX + bestDX),
                    currStride: w, prevStride: w
                )
            }

            for j in -1...1 {
                for i in -1...1 {
                    if i == 0 && j == 0 { continue }
                    let dx = bestDX + i * step
                    let dy = bestDY + j * step
                    
                    let maxAbsoluteMV = 64
                    if dx < -maxAbsoluteMV || maxAbsoluteMV < dx || dy < -maxAbsoluteMV || maxAbsoluteMV < dy { continue }
                    if dx < minX || maxX < dx || dy < minY || maxY < dy { continue }
                    
                    let penalty = abs(dx) + abs(dy)
                    if currentBestSAD <= penalty { continue }
                    
                    let sad: Int
                    if size == 64 && actW == 64 && actH == 64 {
                        sad = step == 1 ? calculateSAD64x64(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: (startY + dy) * w + startX + dx),
                            currStride: w, prevStride: w
                        ) : calculateSAD64x64_Subsample(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: (startY + dy) * w + startX + dx),
                            currStride: w, prevStride: w
                        )
                    } else if size == 32 && actW == 32 && actH == 32 {
                        sad = calculateSAD32x32(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: (startY + dy) * w + startX + dx),
                            currStride: w, prevStride: w
                        )
                    } else if size == 16 && actW == 16 && actH == 16 {
                        sad = calculateSAD16x16(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: (startY + dy) * w + startX + dx),
                            currStride: w, prevStride: w
                        )
                    } else if size == 8 && actW == 8 && actH == 8 {
                        sad = calculateSAD8x8(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: (startY + dy) * w + startX + dx),
                            currStride: w, prevStride: w
                        )
                    } else {
                        sad = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: dx, dy: dy)
                    }
                    
                    let totalSAD = sad &+ penalty
                    if totalSAD < currentBestSAD {
                        currentBestSAD = totalSAD
                        currentBestDX = dx
                        currentBestDY = dy
                    }
                }
            }
            bestDX = currentBestDX
            bestDY = currentBestDY
            bestSAD = currentBestSAD
            step /= 2
        }
    }

    // Fractional ME should ONLY run if we decide NOT to split (to save massive CPU time)
    let penaltyValues = [64: 0, 32: 200, 16: 400, 8: 600] // RDO penalty based on size
    let baseLeafSAD = bestSAD + (penaltyValues[size] ?? 0)

    let minSize = 8
    let splitEarlyExitThreshold = actW * actH * 8 // SAD=8 per pixel (tolerates quantization noise)
    var performSplit = false
    var splitNode: MotionNode? = nil
    var splitSAD = Int.max

    // Evaluate Split
    if size > minSize && actW == size && actH == size && bestSAD > splitEarlyExitThreshold {
        let half = size / 2
        // Save grid state in case we don't pick split
        let savedGrid = grid
        
        // Provide integer PMV for children during their search to keep it accurate
        let intPMV = SIMD2<Int16>(Int16(bestDX * 4), Int16(bestDY * 4))
        grid.fill(x: startX, y: startY, w: size, h: size, mv: intPMV)
        
        // Hierarchical ME (HME): Reduce the search range exponentially for child nodes.
        // Parent already found the bulk motion vector. Children only need to slightly refine it.
        let childSearchRange = max(1, searchRange / 2) // Drastically shrink for children
        let tl = evaluateMotionQuadtreeNode(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, size: half, searchRange: childSearchRange, coarseDX: 0, coarseDY: 0, grid: &grid, fracRefBuf: fracRefBuf, fracExtBuf: fracExtBuf)
        let tr = evaluateMotionQuadtreeNode(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX + half, startY: startY, size: half, searchRange: childSearchRange, coarseDX: 0, coarseDY: 0, grid: &grid, fracRefBuf: fracRefBuf, fracExtBuf: fracExtBuf)
        let bl = evaluateMotionQuadtreeNode(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY + half, size: half, searchRange: childSearchRange, coarseDX: 0, coarseDY: 0, grid: &grid, fracRefBuf: fracRefBuf, fracExtBuf: fracExtBuf)
        let br = evaluateMotionQuadtreeNode(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX + half, startY: startY + half, size: half, searchRange: childSearchRange, coarseDX: 0, coarseDY: 0, grid: &grid, fracRefBuf: fracRefBuf, fracExtBuf: fracExtBuf)
        // Penalty for sending 4 MVs instead of 1 MV.
        // It must scale with area to prevent overfitting to quantization noise!
        let splitPenalty = (actW * actH) * 1 // Requires at least an average of SAD=1 reduction per pixel to justify a split
        splitSAD = tl.sad + tr.sad + bl.sad + br.sad + splitPenalty
        
        if splitSAD < baseLeafSAD {
            // Split preferred
            performSplit = true
            splitNode = .split(tl: tl.node, tr: tr.node, bl: bl.node, br: br.node)
        } else {
            // No Split preferred: Revert grid
            grid = savedGrid
        }
    }

    if performSplit {
        return (splitNode!, splitSAD)
    } else {
        // Run heavy subpixel refinement ONLY on the finalized leaf node!
        let refinedQMV = refineFractionalMBME(
            pCurr: pCurr, pPrev: pPrev, w: w, h: h,
            startX: startX, startY: startY, actW: actW, actH: actH,
            bestIntDX: bestDX, bestIntDY: bestDY, bestIntSAD: bestSAD,
            fracRefBuffer: fracRefBuf, fracExtBuffer: fracExtBuf
        )
        grid.fill(x: startX, y: startY, w: actW, h: actH, mv: refinedQMV)
        return (.leaf(mv: refinedQMV), baseLeafSAD)
    }
}

@inline(__always)
func evaluateLayer0Motion(
    pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int,
    startX: Int, startY: Int, size: Int, searchRange: Int
) -> (dx: Int, dy: Int) {
    if startX >= w || startY >= h { return (0, 0) }
    
    let actW = min(size, w - startX)
    let actH = min(size, h - startY)
    
    var bestSAD = Int.max
    var bestDX = 0
    var bestDY = 0

    // Evaluate Center (from Coarse MV or PMV depending on size)
    let zeroSAD = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: 0, dy: 0)
    bestSAD = zeroSAD
    
    for dy in -searchRange...searchRange {
        for dx in -searchRange...searchRange {
            if dx == 0 && dy == 0 { continue }
            
            let sad = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: dx, dy: dy)
            let penalty = (abs(dx) + abs(dy)) * 2
            
            if sad + penalty < bestSAD {
                bestSAD = sad + penalty
                bestDX = dx
                bestDY = dy
            }
        }
    }
    
    return (bestDX, bestDY)
}

func estimateMotionQuadtree(curr: PlaneData420, prev: PlaneData420, layer0Curr: (data: [Int16], w: Int, h: Int), layer0Prev: (data: [Int16], w: Int, h: Int)) -> MotionTree {
    let mbSize = 64
    let w = curr.width
    let h = curr.height
    let mbCols = (w + mbSize - 1) / mbSize
    let mbRows = (h + mbSize - 1) / mbSize

    var ctuNodes = [MotionNode]()
    var grid = MVGrid(width: w, height: h, minSize: 8)

    var fracRefBuffer = [Int16](repeating: 0, count: 64 * 64)
    var fracExtBuffer = [Int16](repeating: 0, count: 71 * 71)

    curr.y.withUnsafeBufferPointer { currPtr in
        guard let pCurr = currPtr.baseAddress else { return }
        prev.y.withUnsafeBufferPointer { prevPtr in
            guard let pPrev = prevPtr.baseAddress else { return }
            fracRefBuffer.withUnsafeMutableBufferPointer { fracRefPtr in
                guard let pFracRef = fracRefPtr.baseAddress else { return }
                fracExtBuffer.withUnsafeMutableBufferPointer { fracExtPtr in
                    guard let pFracExt = fracExtPtr.baseAddress else { return }

            layer0Curr.data.withUnsafeBufferPointer { layer0CPtr in
                guard let pLayer0C = layer0CPtr.baseAddress else { return }
                layer0Prev.data.withUnsafeBufferPointer { layer0PPtr in
                    guard let pLayer0P = layer0PPtr.baseAddress else { return }
                    
                    for mbY in 0..<mbRows {
                        let startY = mbY * mbSize
                        for mbX in 0..<mbCols {
                            let startX = mbX * mbSize
                            
                            // 1. DWT/Layer0 Coarse ME: Search in the 1/8 downscaled plane
                            // The 64x64 block maps to an 8x8 block in Layer0. Search range ±3 means ±24 in full res!
                            let cDXDY = evaluateLayer0Motion(
                                pCurr: pLayer0C, pPrev: pLayer0P, w: layer0Curr.w, h: layer0Curr.h,
                                startX: mbX * 8, startY: mbY * 8, size: 8, searchRange: 3
                            )
                            
                            // 2. Full-res Fine ME: Refine around the scaled Coarse MV
                            // Fine search range of ±3 is extremely light (49 evaluations) instead of ±16 (1089 evaluations)
                            let nodeEval = evaluateMotionQuadtreeNode(
                                pCurr: pCurr, pPrev: pPrev, w: w, h: h,
                                startX: startX, startY: startY, size: mbSize, searchRange: 3,
                                coarseDX: cDXDY.dx * 8, coarseDY: cDXDY.dy * 8,
                                grid: &grid, fracRefBuf: pFracRef, fracExtBuf: pFracExt
                            )
                            ctuNodes.append(nodeEval.node)
                        }
                    }
                }
            }
                }
            }
        }
    }
    return MotionTree(ctuNodes: ctuNodes, width: w, height: h)
}

func applyMotionQuadtreeNode(
    node: MotionNode,
    pDstY: UnsafeMutablePointer<Int16>,
    pDstCb: UnsafeMutablePointer<Int16>,
    pDstCr: UnsafeMutablePointer<Int16>,
    pPrevY: UnsafePointer<Int16>,
    pPrevCb: UnsafePointer<Int16>,
    pPrevCr: UnsafePointer<Int16>,
    w: Int, h: Int,
    startX: Int, startY: Int, size: Int,
    extBuffer: UnsafeMutablePointer<Int16>
) {
    if startX >= w || startY >= h { return }
    let actW = min(size, w - startX)
    let actH = min(size, h - startY)

    switch node {
    case .split(let tl, let tr, let bl, let br):
        let half = size / 2
        applyMotionQuadtreeNode(node: tl, pDstY: pDstY, pDstCb: pDstCb, pDstCr: pDstCr, pPrevY: pPrevY, pPrevCb: pPrevCb, pPrevCr: pPrevCr, w: w, h: h, startX: startX, startY: startY, size: half, extBuffer: extBuffer)
        applyMotionQuadtreeNode(node: tr, pDstY: pDstY, pDstCb: pDstCb, pDstCr: pDstCr, pPrevY: pPrevY, pPrevCb: pPrevCb, pPrevCr: pPrevCr, w: w, h: h, startX: startX + half, startY: startY, size: half, extBuffer: extBuffer)
        applyMotionQuadtreeNode(node: bl, pDstY: pDstY, pDstCb: pDstCb, pDstCr: pDstCr, pPrevY: pPrevY, pPrevCb: pPrevCb, pPrevCr: pPrevCr, w: w, h: h, startX: startX, startY: startY + half, size: half, extBuffer: extBuffer)
        applyMotionQuadtreeNode(node: br, pDstY: pDstY, pDstCb: pDstCb, pDstCr: pDstCr, pPrevY: pPrevY, pPrevCb: pPrevCb, pPrevCr: pPrevCr, w: w, h: h, startX: startX + half, startY: startY + half, size: half, extBuffer: extBuffer)
    case .leaf(let mv):
        let dx = Int(mv.x)
        let dy = Int(mv.y)
        let intDX = dx >> 2
        let intDY = dy >> 2
        let fracX = dx & 3
        let fracY = dy & 3

        let refX = startX + intDX
        let refY = startY + intDY

        let isYSafe = (refX - 3 >= 0) && (refY - 3 >= 0) && (refX + actW + 4 <= w) && (refY + actH + 4 <= h)

        let dstPtrY = pDstY.advanced(by: startY * w + startX)

        if isYSafe {
            SubpixelInterpolator.interpolateBlock(
                src: pPrevY, srcStride: w, dst: dstPtrY, dstStride: w,
                width: actW, height: actH, fracX: fracX, fracY: fracY, startX: refX, startY: refY
            )
        } else {
            let extW = actW + 7
            let extH = actH + 7
            let extStartX = refX - 3
            let extStartY = refY - 3
            let minSafeX = max(0, min(extW, -extStartX))
            let maxSafeX = max(0, min(extW, w - extStartX))
            
            for y in 0..<extH {
                let srcY = max(0, min(h - 1, extStartY + y))
                let srcRow = srcY * w
                let dstRow = y * extW
                let pDstBase = extBuffer.advanced(by: dstRow)

                if 0 < minSafeX {
                    let leftEdgeVal = pPrevY[srcRow]
                    for x in 0..<minSafeX { pDstBase[x] = leftEdgeVal }
                }

                let copyCount = maxSafeX - minSafeX
                if 0 < copyCount {
                    pDstBase.advanced(by: minSafeX).update(from: pPrevY.advanced(by: srcRow + extStartX + minSafeX), count: copyCount)
                }

                if maxSafeX < extW {
                    let rightEdgeVal = pPrevY[srcRow + w - 1]
                    for x in maxSafeX..<extW { pDstBase[x] = rightEdgeVal }
                }
            }
            
            SubpixelInterpolator.interpolateBlock(
                src: extBuffer, srcStride: extW, dst: dstPtrY, dstStride: w,
                width: actW, height: actH, fracX: fracX, fracY: fracY, startX: 3, startY: 3
            )
        }

        // Chroma
        let cw = w / 2
        let ch = h / 2
        let actCW = actW / 2
        let actCH = actH / 2
        let startCX = startX / 2
        let startCY = startY / 2

        let chromaDX = dx >> 1 // Since luma is 1/4 pel, chroma resolution makes it 1/8 pel, but we map to 1/4 pel filter precision
        let chromaDY = dy >> 1
        let cIntDX = chromaDX >> 2
        let cIntDY = chromaDY >> 2
        let cFracX = chromaDX & 3
        let cFracY = chromaDY & 3

        let crefX = startCX + cIntDX
        let crefY = startCY + cIntDY

        let isCSafe = (crefX - 3 >= 0) && (crefY - 3 >= 0) && (crefX + actCW + 4 <= cw) && (crefY + actCH + 4 <= ch)

        let dstPtrCb = pDstCb.advanced(by: startCY * cw + startCX)
        let dstPtrCr = pDstCr.advanced(by: startCY * cw + startCX)

        if isCSafe {
            SubpixelInterpolator.interpolateBlock(
                src: pPrevCb, srcStride: cw, dst: dstPtrCb, dstStride: cw,
                width: actCW, height: actCH, fracX: cFracX, fracY: cFracY, startX: crefX, startY: crefY
            )
            SubpixelInterpolator.interpolateBlock(
                src: pPrevCr, srcStride: cw, dst: dstPtrCr, dstStride: cw,
                width: actCW, height: actCH, fracX: cFracX, fracY: cFracY, startX: crefX, startY: crefY
            )
        } else {
            let extCW = actCW + 7
            let extCH = actCH + 7
            let cExtStartX = crefX - 3
            let cExtStartY = crefY - 3
            let cMinSafeX = max(0, min(extCW, -cExtStartX))
            let cMaxSafeX = max(0, min(extCW, cw - cExtStartX))

            func copyChromaEdge(pPrevC: UnsafePointer<Int16>, dstC: UnsafeMutablePointer<Int16>) {
                for y in 0..<extCH {
                    let srcY = max(0, min(ch - 1, cExtStartY + y))
                    let srcRow = srcY * cw
                    let dstRow = y * extCW
                    let pDstBase = extBuffer.advanced(by: dstRow)

                    if 0 < cMinSafeX {
                        let leftEdgeVal = pPrevC[srcRow]
                        for x in 0..<cMinSafeX { pDstBase[x] = leftEdgeVal }
                    }
                    let copyCount = cMaxSafeX - cMinSafeX
                    if 0 < copyCount {
                        pDstBase.advanced(by: cMinSafeX).update(from: pPrevC.advanced(by: srcRow + cExtStartX + cMinSafeX), count: copyCount)
                    }
                    if cMaxSafeX < extCW {
                        let rightEdgeVal = pPrevC[srcRow + cw - 1]
                        for x in cMaxSafeX..<extCW { pDstBase[x] = rightEdgeVal }
                    }
                }
                SubpixelInterpolator.interpolateBlock(
                    src: extBuffer, srcStride: extCW, dst: dstC, dstStride: cw,
                    width: actCW, height: actCH, fracX: cFracX, fracY: cFracY, startX: 3, startY: 3
                )
            }
            
            copyChromaEdge(pPrevC: pPrevCb, dstC: dstPtrCb)
            copyChromaEdge(pPrevC: pPrevCr, dstC: dstPtrCr)
        }
    }
}

func applyMotionQuadtree(prev: PlaneData420, tree: MotionTree) async -> PlaneData420 {
    let w = prev.width
    let h = prev.height
    var result = PlaneData420(
        width: w, height: h,
        y: [Int16](repeating: 0, count: w * h),
        cb: [Int16](repeating: 0, count: (w * h) / 4),
        cr: [Int16](repeating: 0, count: (w * h) / 4)
    )

    let mbSize = 64
    let mbCols = (w + mbSize - 1) / mbSize
    let mbRows = (h + mbSize - 1) / mbSize

    result.y.withUnsafeMutableBufferPointer { resYPtr in
        guard let pResY = resYPtr.baseAddress else { return }
        result.cb.withUnsafeMutableBufferPointer { resCbPtr in
            guard let pResCb = resCbPtr.baseAddress else { return }
            result.cr.withUnsafeMutableBufferPointer { resCrPtr in
                guard let pResCr = resCrPtr.baseAddress else { return }
                prev.y.withUnsafeBufferPointer { prevYPtr in
                    guard let pPrevY = prevYPtr.baseAddress else { return }
                    prev.cb.withUnsafeBufferPointer { prevCbPtr in
                        guard let pPrevCb = prevCbPtr.baseAddress else { return }
                        prev.cr.withUnsafeBufferPointer { prevCrPtr in
                            guard let pPrevCr = prevCrPtr.baseAddress else { return }
                            
                            var extBuffer = [Int16](repeating: 0, count: 71 * 71)
                            extBuffer.withUnsafeMutableBufferPointer { extPtr in
                                guard let pExt = extPtr.baseAddress else { return }

                                for mbY in 0..<mbRows {
                                    let startY = mbY * mbSize
                                    for mbX in 0..<mbCols {
                                        let startX = mbX * mbSize
                                        let nodeIndex = mbY * mbCols + mbX
                                        if nodeIndex < tree.ctuNodes.count {
                                            let node = tree.ctuNodes[nodeIndex]
                                            applyMotionQuadtreeNode(
                                                node: node,
                                                pDstY: pResY, pDstCb: pResCb, pDstCr: pResCr,
                                                pPrevY: pPrevY, pPrevCb: pPrevCb, pPrevCr: pPrevCr,
                                                w: w, h: h,
                                                startX: startX, startY: startY, size: mbSize,
                                                extBuffer: pExt
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return result
}

func encodeMotionQuadtreeNode(
    node: MotionNode,
    w: Int, h: Int,
    startX: Int, startY: Int, size: Int,
    grid: inout MVGrid,
    bw: inout EntropyEncoder
) {
    if startX >= w || startY >= h { return }
    let minSize = 8

    switch node {
    case .split(let tl, let tr, let bl, let br):
        if size > minSize {
            bw.encodeBypass(binVal: 1)
        }
        let half = size / 2
        encodeMotionQuadtreeNode(node: tl, w: w, h: h, startX: startX, startY: startY, size: half, grid: &grid, bw: &bw)
        encodeMotionQuadtreeNode(node: tr, w: w, h: h, startX: startX + half, startY: startY, size: half, grid: &grid, bw: &bw)
        encodeMotionQuadtreeNode(node: bl, w: w, h: h, startX: startX, startY: startY + half, size: half, grid: &grid, bw: &bw)
        encodeMotionQuadtreeNode(node: br, w: w, h: h, startX: startX + half, startY: startY + half, size: half, grid: &grid, bw: &bw)
        
    case .leaf(let mv):
        if size > minSize {
            bw.encodeBypass(binVal: 0)
        }
        let actW = min(size, w - startX)
        let actH = min(size, h - startY)

        let pmv = grid.getPMV(x: startX, y: startY, w: actW)
        grid.fill(x: startX, y: startY, w: actW, h: actH, mv: mv)

        let mvdX = Int(mv.x) - Int(pmv.x)
        let mvdY = Int(mv.y) - Int(pmv.y)

        if mvdX == 0 && mvdY == 0 {
            bw.encodeBypass(binVal: 0)
        } else {
            bw.encodeBypass(binVal: 1)
            let sx: UInt8 = mvdX <= -1 ? 1 : 0
            bw.encodeBypass(binVal: sx)
            let mx = UInt32(abs(mvdX))
            encodeExpGolomb(val: mx, encoder: &bw)

            let sy: UInt8 = mvdY <= -1 ? 1 : 0
            bw.encodeBypass(binVal: sy)
            let my = UInt32(abs(mvdY))
            encodeExpGolomb(val: my, encoder: &bw)
        }
    }
}

func decodeMotionQuadtreeNode(
    w: Int, h: Int,
    startX: Int, startY: Int, size: Int,
    grid: inout MVGrid,
    br: inout EntropyDecoder
) throws -> MotionNode {
    if startX >= w || startY >= h { return .leaf(mv: .zero) }
    let minSize = 8

    var isSplit = false
    if size > minSize {
        isSplit = try br.decodeBypass() == 1
    }

    if isSplit {
        let half = size / 2
        let tl = try decodeMotionQuadtreeNode(w: w, h: h, startX: startX, startY: startY, size: half, grid: &grid, br: &br)
        let tr = try decodeMotionQuadtreeNode(w: w, h: h, startX: startX + half, startY: startY, size: half, grid: &grid, br: &br)
        let bl = try decodeMotionQuadtreeNode(w: w, h: h, startX: startX, startY: startY + half, size: half, grid: &grid, br: &br)
        let brNode = try decodeMotionQuadtreeNode(w: w, h: h, startX: startX + half, startY: startY + half, size: half, grid: &grid, br: &br)
        return .split(tl: tl, tr: tr, bl: bl, br: brNode)
    } else {
        let actW = min(size, w - startX)
        let actH = min(size, h - startY)

        let pmv = grid.getPMV(x: startX, y: startY, w: actW)

        let hasMVD = try br.decodeBypass() == 1
        var mvdX: Int = 0
        var mvdY: Int = 0

        if hasMVD {
            let sx = try br.decodeBypass() == 1
            let mx = Int(try decodeExpGolomb(decoder: &br))
            mvdX = sx ? -mx : mx

            let sy = try br.decodeBypass() == 1
            let my = Int(try decodeExpGolomb(decoder: &br))
            mvdY = sy ? -my : my
        }

        let mv = SIMD2<Int16>(Int16(clamping: mvdX + Int(pmv.x)), Int16(clamping: mvdY + Int(pmv.y)))
        grid.fill(x: startX, y: startY, w: actW, h: actH, mv: mv)
        
        return .leaf(mv: mv)
    }
}
