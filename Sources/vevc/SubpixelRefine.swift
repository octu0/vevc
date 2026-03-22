// refineFractionalMBME definition
@inline(__always)
func refineFractionalMBME(
    pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int,
    startX: Int, startY: Int, actW: Int, actH: Int,
    bestIntDX: Int, bestIntDY: Int, bestIntSAD: Int,
    fracRefBuffer: UnsafeMutablePointer<Int16>,
    fracExtBuffer: UnsafeMutablePointer<Int16>
) -> SIMD2<Int16> {
    
    var currentBestQDX = bestIntDX * 4
    var currentBestQDY = bestIntDY * 4
    var currentBestSAD = bestIntSAD

    @inline(__always)
    func evaluatePosition(qdx: Int, qdy: Int) {
        let dx = qdx >> 2
        let dy = qdy >> 2
        let fracX = qdx & 3
        let fracY = qdy & 3

        let refX = startX + dx
        let refY = startY + dy
        
        // Edge check
        let isSafe = (refX - 3 >= 0) && (refY - 3 >= 0) && (refX + actW + 4 <= w) && (refY + actH + 4 <= h)

        if isSafe {
            subpixelInterpolateBlock(
                src: pPrev, srcStride: w,
                dst: fracRefBuffer, dstStride: actW,
                width: actW, height: actH,
                fracX: fracX, fracY: fracY,
                startX: refX, startY: refY
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
                let pDstBase = fracExtBuffer.advanced(by: dstRow)
                
                if 0 < minSafeX {
                    let leftEdgeVal = pPrev[srcRow]
                    for x in 0..<minSafeX {
                        pDstBase[x] = leftEdgeVal
                    }
                }
                
                let copyCount = maxSafeX - minSafeX
                if 0 < copyCount {
                    pDstBase.advanced(by: minSafeX).update(from: pPrev.advanced(by: srcRow + extStartX + minSafeX), count: copyCount)
                }
                
                if maxSafeX < extW {
                    let rightEdgeVal = pPrev[srcRow + w - 1]
                    for x in maxSafeX..<extW {
                        pDstBase[x] = rightEdgeVal
                    }
                }
            }
            
            subpixelInterpolateBlock(
                src: fracExtBuffer, srcStride: extW,
                dst: fracRefBuffer, dstStride: actW,
                width: actW, height: actH,
                fracX: fracX, fracY: fracY,
                startX: 3, startY: 3
            )
        }

        // Calculate SAD between pCurr and fracRefBuffer
        var sad: UInt = 0
        for y in 0..<actH {
            let pCurrRow = pCurr.advanced(by: (startY + y) * w + startX)
            let pRefRow = fracRefBuffer.advanced(by: y * actW)
            
            var x = 0
            if 16 <= actW {
                var sumVec = SIMD16<UInt16>()
                while x <= actW - 16 {
                    let c0 = UnsafeRawPointer(pCurrRow.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                    let p0 = UnsafeRawPointer(pRefRow.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                    let diff0 = c0 &- p0
                    let mask0 = diff0 &>> 15
                    let abs0 = (diff0 ^ mask0) &- mask0
                    sumVec &+= SIMD16<UInt16>(truncatingIfNeeded: abs0)
                    x += 16
                }
                sad &+= UInt(SIMD16<UInt32>(truncatingIfNeeded: sumVec).wrappedSum())
            }
            
            while x < actW {
                let diff = Int(pCurrRow[x]) - Int(pRefRow[x])
                let mask = diff >> 31
                sad &+= UInt((diff ^ mask) - mask)
                x += 1
            }
        }
        
        let diffX = qdx >= 0 ? qdx : -qdx
        let diffY = qdy >= 0 ? qdy : -qdy
        let penalty = (diffX + diffY) / 2 // Small penalty for fractional vectors
        let totalSad = Int(sad) + penalty
        
        if totalSad < currentBestSAD {
            currentBestSAD = totalSad
            currentBestQDX = qdx
            currentBestQDY = qdy
        }
    }

    // 1. Half-pel Refinement (8 positions around best integer MV)
    let halfPelOffsets = [
        (2, 0), (2, 2), (0, 2), (-2, 2),
        (-2, 0), (-2, -2), (0, -2), (2, -2)
    ]
    let centerQDX = bestIntDX * 4
    let centerQDY = bestIntDY * 4

    for offset in halfPelOffsets {
        evaluatePosition(qdx: centerQDX + offset.0, qdy: centerQDY + offset.1)
    }

    // 2. Quarter-pel Refinement (8 positions around best half/int MV)
    let bestHalfQDX = currentBestQDX
    let bestHalfQDY = currentBestQDY
    
    let quarterPelOffsets = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ]
    
    for offset in quarterPelOffsets {
        evaluatePosition(qdx: bestHalfQDX + offset.0, qdy: bestHalfQDY + offset.1)
    }

    return SIMD2<Int16>(Int16(currentBestQDX), Int16(currentBestQDY))
}
