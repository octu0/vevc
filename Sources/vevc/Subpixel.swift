import Foundation

// Quarter-pel: alpha = 1/4
@inline(__always)
func subpixelInterpolateQuarterX(ptr: UnsafePointer<Int16>, offset: Int) -> Int16 {
    let p = ptr.advanced(by: offset - 3)
    // [-1, 4, -10, 58, 17, -5, 1, 0]
    let sum = -1 * Int(p[0])
            +  4 * Int(p[1])
            - 10 * Int(p[2])
            + 58 * Int(p[3])
            + 17 * Int(p[4])
            -  5 * Int(p[5])
            +  1 * Int(p[6])
            +  0 * Int(p[7])
    
    // Add 32 for rounding before shift right by 6
    let val = (sum + 32) >> 6
    return Int16(clamping: val)
}

// Half-pel: alpha = 2/4 = 1/2
@inline(__always)
func subpixelInterpolateHalfX(ptr: UnsafePointer<Int16>, offset: Int) -> Int16 {
    let p = ptr.advanced(by: offset - 3)
    // [-1, 4, -11, 40, 40, -11, 4, -1]
    let sum = -1 * Int(p[0])
            +  4 * Int(p[1])
            - 11 * Int(p[2])
            + 40 * Int(p[3])
            + 40 * Int(p[4])
            - 11 * Int(p[5])
            +  4 * Int(p[6])
            -  1 * Int(p[7])
    
    let val = (sum + 32) >> 6
    return Int16(clamping: val)
}

// Three Quarter-pel: alpha = 3/4
@inline(__always)
func subpixelInterpolateThreeQuarterX(ptr: UnsafePointer<Int16>, offset: Int) -> Int16 {
    let p = ptr.advanced(by: offset - 3)
    // [0, 1, -5, 17, 58, -10, 4, -1]
    let sum =  0 * Int(p[0])
            +  1 * Int(p[1])
            -  5 * Int(p[2])
            + 17 * Int(p[3])
            + 58 * Int(p[4])
            - 10 * Int(p[5])
            +  4 * Int(p[6])
            -  1 * Int(p[7])
    
    let val = (sum + 32) >> 6
    return Int16(clamping: val)
}

// Y-direction interpolation
@inline(__always)
func subpixelInterpolateY(ptr: UnsafePointer<Int16>, offset: Int, stride: Int, fracY: Int) -> Int16 {
    let p0 = ptr.advanced(by: offset - 3 * stride)
    let p1 = ptr.advanced(by: offset - 2 * stride)
    let p2 = ptr.advanced(by: offset - 1 * stride)
    let p3 = ptr.advanced(by: offset)
    let p4 = ptr.advanced(by: offset + 1 * stride)
    let p5 = ptr.advanced(by: offset + 2 * stride)
    let p6 = ptr.advanced(by: offset + 3 * stride)
    let p7 = ptr.advanced(by: offset + 4 * stride)
    
    var sum = 0
    switch fracY {
    case 1:
        sum = -1 * Int(p0.pointee) + 4 * Int(p1.pointee) - 10 * Int(p2.pointee) + 58 * Int(p3.pointee) + 17 * Int(p4.pointee) - 5 * Int(p5.pointee) + 1 * Int(p6.pointee)
    case 2:
        sum = -1 * Int(p0.pointee) + 4 * Int(p1.pointee) - 11 * Int(p2.pointee) + 40 * Int(p3.pointee) + 40 * Int(p4.pointee) - 11 * Int(p5.pointee) + 4 * Int(p6.pointee) - 1 * Int(p7.pointee)
    case 3:
        sum = 1 * Int(p1.pointee) - 5 * Int(p2.pointee) + 17 * Int(p3.pointee) + 58 * Int(p4.pointee) - 10 * Int(p5.pointee) + 4 * Int(p6.pointee) - 1 * Int(p7.pointee)
    default:
        return ptr.advanced(by: offset).pointee
    }
    
    let val = (sum + 32) >> 6
    return Int16(clamping: val)
}

// 2D Interpolation routine
// fracX, fracY = [0, 1, 2, 3] representing 0, 1/4, 2/4, 3/4 pel offset
@inline(__always)
func subpixelInterpolateBlock(
    src: UnsafePointer<Int16>, srcStride: Int,
    dst: UnsafeMutablePointer<Int16>, dstStride: Int,
    width: Int, height: Int,
    fracX: Int, fracY: Int,
    startX: Int, startY: Int
) {
    if fracX == 0 && fracY == 0 {
        for y in 0..<height {
            let srcPtr = src.advanced(by: (startY + y) * srcStride + startX)
            let dstPtr = dst.advanced(by: y * dstStride)
            dstPtr.update(from: srcPtr, count: width)
        }
        return
    }
    
    if fracY == 0 {
        for y in 0..<height {
            let srcYOffset = (startY + y) * srcStride
            let dstYOffset = y * dstStride
            for x in 0..<width {
                let rx = startX + x
                switch fracX {
                case 1: dst[dstYOffset + x] = subpixelInterpolateQuarterX(ptr: src, offset: srcYOffset + rx)
                case 2: dst[dstYOffset + x] = subpixelInterpolateHalfX(ptr: src, offset: srcYOffset + rx)
                case 3: dst[dstYOffset + x] = subpixelInterpolateThreeQuarterX(ptr: src, offset: srcYOffset + rx)
                default: break
                }
            }
        }
        return
    }
    
    if fracX == 0 {
        for y in 0..<height {
            let dstYOffset = y * dstStride
            for x in 0..<width {
                let offset = (startY + y) * srcStride + startX + x
                dst[dstYOffset + x] = subpixelInterpolateY(ptr: src, offset: offset, stride: srcStride, fracY: fracY)
            }
        }
        return
    }
    
    let extraLines = 7
    let intermediateCount = width * (height + extraLines)
    
    withUnsafeTemporaryAllocation(of: Int16.self, capacity: intermediateCount) { interPtr in
        guard let intBase = interPtr.baseAddress else { return }
        
        // Performance Guideline: "2. Eliminate conditional branches from inner loops" -> hoisting branches outside
        switch fracX {
        case 1:
            for y in 0..<(height + extraLines) {
                let srcYOffset = (startY + y - 3) * srcStride
                let intYOffset = y * width
                for x in 0..<width {
                    intBase[intYOffset + x] = subpixelInterpolateQuarterX(ptr: src, offset: srcYOffset + startX + x)
                }
            }
        case 2:
            for y in 0..<(height + extraLines) {
                let srcYOffset = (startY + y - 3) * srcStride
                let intYOffset = y * width
                for x in 0..<width {
                    intBase[intYOffset + x] = subpixelInterpolateHalfX(ptr: src, offset: srcYOffset + startX + x)
                }
            }
        case 3:
            for y in 0..<(height + extraLines) {
                let srcYOffset = (startY + y - 3) * srcStride
                let intYOffset = y * width
                for x in 0..<width {
                    intBase[intYOffset + x] = subpixelInterpolateThreeQuarterX(ptr: src, offset: srcYOffset + startX + x)
                }
            }
        default:
            break
        }
        
        for y in 0..<height {
            let dstYOffset = y * dstStride
            let intYOffset = (y + 3) * width
            for x in 0..<width {
                dst[dstYOffset + x] = subpixelInterpolateY(ptr: intBase, offset: intYOffset + x, stride: width, fracY: fracY)
            }
        }
    }
}

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
