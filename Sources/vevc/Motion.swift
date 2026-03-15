import Foundation

@inline(__always)
func calculateSAD32x32(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, currStride: Int, prevStride: Int) -> Int {
    var sumVec0 = SIMD16<UInt16>()
    var sumVec1 = SIMD16<UInt16>()
    
    for y in 0..<32 {
        let currRow = pCurr.advanced(by: y * currStride)
        let prevRow = pPrev.advanced(by: y * prevStride)

        let c0 = UnsafeRawPointer(currRow).loadUnaligned(as: SIMD16<Int16>.self)
        let p0 = UnsafeRawPointer(prevRow).loadUnaligned(as: SIMD16<Int16>.self)
        let c1 = UnsafeRawPointer(currRow.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        let p1 = UnsafeRawPointer(prevRow.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)

        let diff0 = c0 &- p0
        let mask0 = diff0 &>> 15
        let abs0 = (diff0 ^ mask0) &- mask0

        let diff1 = c1 &- p1
        let mask1 = diff1 &>> 15
        let abs1 = (diff1 ^ mask1) &- mask1

        sumVec0 &+= SIMD16<UInt16>(truncatingIfNeeded: abs0)
        sumVec1 &+= SIMD16<UInt16>(truncatingIfNeeded: abs1)
    }
    
    let total0 = SIMD16<UInt32>(truncatingIfNeeded: sumVec0).wrappedSum()
    let total1 = SIMD16<UInt32>(truncatingIfNeeded: sumVec1).wrappedSum()
    return Int(total0 &+ total1)
}

@inline(__always)
func calculateSAD64x64(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, currStride: Int, prevStride: Int) -> Int {
    var sumVec0 = SIMD16<UInt16>()
    var sumVec1 = SIMD16<UInt16>()
    var sumVec2 = SIMD16<UInt16>()
    var sumVec3 = SIMD16<UInt16>()
    
    for y in 0..<64 {
        let currRow = pCurr.advanced(by: y * currStride)
        let prevRow = pPrev.advanced(by: y * prevStride)

        let c0 = UnsafeRawPointer(currRow).loadUnaligned(as: SIMD16<Int16>.self)
        let p0 = UnsafeRawPointer(prevRow).loadUnaligned(as: SIMD16<Int16>.self)
        let c1 = UnsafeRawPointer(currRow.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        let p1 = UnsafeRawPointer(prevRow.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        let c2 = UnsafeRawPointer(currRow.advanced(by: 32)).loadUnaligned(as: SIMD16<Int16>.self)
        let p2 = UnsafeRawPointer(prevRow.advanced(by: 32)).loadUnaligned(as: SIMD16<Int16>.self)
        let c3 = UnsafeRawPointer(currRow.advanced(by: 48)).loadUnaligned(as: SIMD16<Int16>.self)
        let p3 = UnsafeRawPointer(prevRow.advanced(by: 48)).loadUnaligned(as: SIMD16<Int16>.self)

        let diff0 = c0 &- p0
        let mask0 = diff0 &>> 15
        let abs0 = (diff0 ^ mask0) &- mask0

        let diff1 = c1 &- p1
        let mask1 = diff1 &>> 15
        let abs1 = (diff1 ^ mask1) &- mask1

        let diff2 = c2 &- p2
        let mask2 = diff2 &>> 15
        let abs2 = (diff2 ^ mask2) &- mask2

        let diff3 = c3 &- p3
        let mask3 = diff3 &>> 15
        let abs3 = (diff3 ^ mask3) &- mask3

        sumVec0 &+= SIMD16<UInt16>(truncatingIfNeeded: abs0)
        sumVec1 &+= SIMD16<UInt16>(truncatingIfNeeded: abs1)
        sumVec2 &+= SIMD16<UInt16>(truncatingIfNeeded: abs2)
        sumVec3 &+= SIMD16<UInt16>(truncatingIfNeeded: abs3)
    }
    
    let total0 = SIMD16<UInt32>(truncatingIfNeeded: sumVec0).wrappedSum()
    let total1 = SIMD16<UInt32>(truncatingIfNeeded: sumVec1).wrappedSum()
    let total2 = SIMD16<UInt32>(truncatingIfNeeded: sumVec2).wrappedSum()
    let total3 = SIMD16<UInt32>(truncatingIfNeeded: sumVec3).wrappedSum()
    
    return Int(total0 &+ total1 &+ total2 &+ total3)
}

struct MotionVector: Sendable {
    let dx: Int
    let dy: Int
}

struct MotionVectors: Sendable {
    var vectors: [SIMD2<Int16>]

    init(count: Int) {
        self.vectors = [SIMD2<Int16>](repeating: .zero, count: count)
    }
}

@inline(__always)
func estimateMBMEBlock64x64(
    pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int,
    startX: Int, startY: Int, searchRange: Int,
    mvs: inout MotionVectors, mvIdx: Int
) {
    var bestSAD = calculateSAD64x64(
        pCurr: pCurr.advanced(by: startY * w + startX),
        pPrev: pPrev.advanced(by: startY * w + startX),
        currStride: w, prevStride: w
    )
    var bestDX = 0
    var bestDY = 0

    let earlyExitThreshold = 64 * 64 * 2
    if earlyExitThreshold < bestSAD {
        let minSafeDX = max(-1 * searchRange, -startX)
        let maxSafeDX = min(searchRange, w - 64 - startX)
        let minSafeDY = max(-1 * searchRange, -startY)
        let maxSafeDY = min(searchRange, h - 64 - startY)

        let negSearchRange = -1 * searchRange
        let posSearchRange = searchRange

        var step = searchRange / 2
        while step >= 1 {
            var currentBestDX = bestDX
            var currentBestDY = bestDY
            var currentBestSAD = bestSAD

            for j in -1...1 {
                for i in -1...1 {
                    if i == 0 && j == 0 { continue }
                    
                    let dx = bestDX + i * step
                    let dy = bestDY + j * step

                    if dx < negSearchRange || posSearchRange < dx || dy < negSearchRange || posSearchRange < dy { continue }

                    let diffX = dx >= 0 ? dx : -1 * dx
                    let diffY = dy >= 0 ? dy : -1 * dy
                    let penalty = diffX + diffY
                    if currentBestSAD <= penalty { continue }

                    let sad: Int
                    let isDYSafe = dy >= minSafeDY && dy <= maxSafeDY
                    if isDYSafe && dx >= minSafeDX && dx <= maxSafeDX {
                        let refYRowOffset = (startY + dy) * w
                        sad = calculateSAD64x64(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: refYRowOffset + startX + dx),
                            currStride: w, prevStride: w
                        )
                    } else {
                        sad = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: 64, actH: 64, dx: dx, dy: dy)
                    }

                    let totalSad = sad &+ penalty
                    if totalSad < currentBestSAD {
                        currentBestSAD = totalSad
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

    mvs.vectors[mvIdx] = SIMD2(Int16(bestDX), Int16(bestDY))
}

@inline(__always)
func estimateMBMEBlock32x32(
    pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int,
    startX: Int, startY: Int, searchRange: Int,
    mvs: inout MotionVectors, mvIdx: Int
) {
    var bestSAD = calculateSAD32x32(
        pCurr: pCurr.advanced(by: startY * w + startX),
        pPrev: pPrev.advanced(by: startY * w + startX),
        currStride: w, prevStride: w
    )
    var bestDX = 0
    var bestDY = 0

    let earlyExitThreshold = 32 * 32 * 2
    if earlyExitThreshold < bestSAD {
        let minSafeDX = max(-1 * searchRange, -startX)
        let maxSafeDX = min(searchRange, w - 32 - startX)
        let minSafeDY = max(-1 * searchRange, -startY)
        let maxSafeDY = min(searchRange, h - 32 - startY)

        let negSearchRange = -1 * searchRange
        let posSearchRange = searchRange

        var step = searchRange / 2
        while 1 <= step {
            var currentBestDX = bestDX
            var currentBestDY = bestDY
            var currentBestSAD = bestSAD

            for j in -1...1 {
                for i in -1...1 {
                    if i == 0 && j == 0 { continue }
                    
                    let dx = bestDX + i * step
                    let dy = bestDY + j * step

                    if dx < negSearchRange || posSearchRange < dx || dy < negSearchRange || posSearchRange < dy { continue }

                    let diffX = dx >= 0 ? dx : -1 * dx
                    let diffY = dy >= 0 ? dy : -1 * dy
                    let penalty = diffX + diffY
                    if currentBestSAD <= penalty { continue }

                    let sad: Int
                    let isDYSafe = dy >= minSafeDY && dy <= maxSafeDY
                    if isDYSafe && dx >= minSafeDX && dx <= maxSafeDX {
                        let refYRowOffset = (startY + dy) * w
                        sad = calculateSAD32x32(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: refYRowOffset + startX + dx),
                            currStride: w, prevStride: w
                        )
                    } else {
                        sad = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: 32, actH: 32, dx: dx, dy: dy)
                    }

                    let totalSad = sad &+ penalty
                    if totalSad < currentBestSAD {
                        currentBestSAD = totalSad
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

    mvs.vectors[mvIdx] = SIMD2(Int16(bestDX), Int16(bestDY))
}

@inline(__always)
func estimateMBMEBlockEdge(
    pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int,
    startX: Int, startY: Int, actW: Int, actH: Int, searchRange: Int,
    mvs: inout MotionVectors, mvIdx: Int
) {
    var bestSAD = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: 0, dy: 0)
    var bestDX = 0
    var bestDY = 0

    let earlyExitThreshold = actW * actH * 2
    if earlyExitThreshold < bestSAD {
        let negSearchRange = -1 * searchRange
        let posSearchRange = searchRange

        var step = searchRange / 2
        while 1 <= step {
            var currentBestDX = bestDX
            var currentBestDY = bestDY
            var currentBestSAD = bestSAD

            for j in -1...1 {
                for i in -1...1 {
                    if i == 0 && j == 0 { continue }

                    let dx = bestDX + i * step
                    let dy = bestDY + j * step

                    if dx < negSearchRange || posSearchRange < dx || dy < negSearchRange || posSearchRange < dy { continue }

                    let diffX = dx >= 0 ? dx : -1 * dx
                    let diffY = dy >= 0 ? dy : -1 * dy
                    let penalty = diffX + diffY
                    if currentBestSAD <= penalty { continue }

                    let refX = startX + dx
                    let refY = startY + dy
                    let sad: Int

                    if 0 <= refX && 0 <= refY && refX + actW <= w && refY + actH <= h {
                        var s: UInt = 0
                        for y in 0..<actH {
                            let currRow = (startY + y) * w + startX
                            let prevRow = (refY + y) * w + refX
                            let pCurrRow = pCurr.advanced(by: currRow)
                            let pPrevRow = pPrev.advanced(by: prevRow)
                            for x in 0..<actW {
                                let diff = Int(pCurrRow[x]) - Int(pPrevRow[x])
                                let mask = diff >> 31
                                s &+= UInt((diff ^ mask) - mask)
                            }
                        }
                        sad = Int(s)
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

    mvs.vectors[mvIdx] = SIMD2(Int16(bestDX), Int16(bestDY))
}

@inline(__always)
func estimateMBME(curr: PlaneData420, prev: PlaneData420) -> MotionVectors {
    let mbSize = 64
    let w = curr.width
    let h = curr.height
    let mbCols = (w + mbSize - 1) / mbSize
    let mbRows = (h + mbSize - 1) / mbSize

    var mvs = MotionVectors(count: mbCols * mbRows)

    let searchRange = 16

    curr.y.withUnsafeBufferPointer { currPtr in
        guard let pCurr = currPtr.baseAddress else { return }
        prev.y.withUnsafeBufferPointer { prevPtr in
            guard let pPrev = prevPtr.baseAddress else { return }

            let fullMbCols = w / mbSize
            let fullMbRows = h / mbSize
            let remW = w % mbSize
            let remH = h % mbSize

            for mbY in 0..<fullMbRows {
                let startY = mbY * mbSize
                for mbX in 0..<fullMbCols {
                    let startX = mbX * mbSize
                    let idx = mbY * mbCols + mbX
                    estimateMBMEBlock64x64(
                        pCurr: pCurr, pPrev: pPrev, w: w, h: h,
                        startX: startX, startY: startY, searchRange: searchRange,
                        mvs: &mvs, mvIdx: idx
                    )
                }
            }

            if 0 < remW {
                let mbX = fullMbCols
                let startX = mbX * mbSize
                for mbY in 0..<fullMbRows {
                    let startY = mbY * mbSize
                    let idx = mbY * mbCols + mbX
                    estimateMBMEBlockEdge(
                        pCurr: pCurr, pPrev: pPrev, w: w, h: h,
                        startX: startX, startY: startY, actW: remW, actH: mbSize, searchRange: searchRange,
                        mvs: &mvs, mvIdx: idx
                    )
                }
            }

            if 0 < remH {
                let mbY = fullMbRows
                let startY = mbY * mbSize
                for mbX in 0..<fullMbCols {
                    let startX = mbX * mbSize
                    let idx = mbY * mbCols + mbX
                    estimateMBMEBlockEdge(
                        pCurr: pCurr, pPrev: pPrev, w: w, h: h,
                        startX: startX, startY: startY, actW: mbSize, actH: remH, searchRange: searchRange,
                        mvs: &mvs, mvIdx: idx
                    )
                }
            }

            if 0 < remW && 0 < remH {
                let mbX = fullMbCols
                let mbY = fullMbRows
                let startX = mbX * mbSize
                let startY = mbY * mbSize
                let idx = mbY * mbCols + mbX
                estimateMBMEBlockEdge(
                    pCurr: pCurr, pPrev: pPrev, w: w, h: h,
                    startX: startX, startY: startY, actW: remW, actH: remH, searchRange: searchRange,
                    mvs: &mvs, mvIdx: idx
                )
            }
        }
    }

    return mvs
}

@inline(__always)
func calculateSADEdge(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int, startX: Int, startY: Int, actW: Int, actH: Int, dx: Int, dy: Int) -> Int {
    var sad: UInt = 0
    let refX = startX + dx
    let minSafeX = max(0, min(actW, -refX))
    let maxSafeX = max(0, min(actW, w - refX))

    for y in 0..<actH {
        let cy = startY + y
        let currRow = cy * w

        let py = max(0, min(h - 1, cy + dy))
        let prevRow = py * w

        let pCurrRow = pCurr.advanced(by: currRow + startX)
        let pPrevRow = pPrev.advanced(by: prevRow)

        if 0 < minSafeX {
            let leftEdgeVal = pPrevRow[0]
            for x in 0..<minSafeX {
                let diff = Int(pCurrRow[x]) - Int(leftEdgeVal)
                let mask = diff >> 31
                sad &+= UInt((diff ^ mask) - mask)
            }
        }
        
        let copyCount = maxSafeX - minSafeX
        if 0 < copyCount {
            let pPrevSafe = pPrevRow.advanced(by: refX + minSafeX)
            let pCurrSafe = pCurrRow.advanced(by: minSafeX)
            
            var x = 0
            if 16 <= copyCount {
                var sumVec = SIMD16<UInt16>()
                while x <= copyCount - 16 {
                    let c0 = UnsafeRawPointer(pCurrSafe.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                    let p0 = UnsafeRawPointer(pPrevSafe.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
                    let diff0 = c0 &- p0
                    let mask0 = diff0 &>> 15
                    let abs0 = (diff0 ^ mask0) &- mask0
                    sumVec &+= SIMD16<UInt16>(truncatingIfNeeded: abs0)
                    x += 16
                }
                sad &+= UInt(SIMD16<UInt32>(truncatingIfNeeded: sumVec).wrappedSum())
            }
            
            while x < copyCount {
                let diff = Int(pCurrSafe[x]) - Int(pPrevSafe[x])
                let mask = diff >> 31
                sad &+= UInt((diff ^ mask) - mask)
                x += 1
            }
        }

        if maxSafeX < actW {
            let rightEdgeVal = pPrevRow[w - 1]
            for x in maxSafeX..<actW {
                let diff = Int(pCurrRow[x]) - Int(rightEdgeVal)
                let mask = diff >> 31
                sad &+= UInt((diff ^ mask) - mask)
            }
        }
    }
    return Int(sad)
}

@inline(__always)
func applyMBME(prev: PlaneData420, mvs: MotionVectors) async -> PlaneData420 {
    let mbSize = 64
    let w = prev.width
    let h = prev.height
    let mbCols = (w + mbSize - 1) / mbSize

    @Sendable @inline(__always)
    func apply(data: [Int16], pW: Int, pH: Int, div: Int) async -> [Int16] {
        if pW == 0 || pH == 0 { return data }
        var out = [Int16](repeating: 0, count: pW * pH)

        let localMbCols = mbCols
        let pMbSize = mbSize / div

        let fullMbCols = pW / pMbSize
        let fullMbRows = pH / pMbSize
        let remW = pW % pMbSize
        let remH = pH % pMbSize

        data.withUnsafeBufferPointer { pPtr in
            guard let pData = pPtr.baseAddress else { return }
            out.withUnsafeMutableBufferPointer { oPtr in
                guard let pOut = oPtr.baseAddress else { return }

                @inline(__always)
                func applyBlock(mbX: Int, mbY: Int, actW: Int, actH: Int) {
                    let startX = mbX * pMbSize
                    let startY = mbY * pMbSize
                    let idx = mbY * localMbCols + mbX
                    let vec = mvs.vectors[idx]
                    let dx = Int(vec.x) / div
                    let dy = Int(vec.y) / div

                    let refX = startX + dx
                    let refY = startY + dy
                    
                    if 0 <= refX && 0 <= refY && (refX + actW) <= pW && (refY + actH) <= pH {
                        switch actW {
                        case 64:
                            for y in 0..<actH {
                                let dstRow = (startY + y) * pW
                                let srcRow = (refY + y) * pW
                                let c0 = UnsafeRawPointer(pData.advanced(by: srcRow + refX)).loadUnaligned(as: SIMD16<Int16>.self)
                                let c1 = UnsafeRawPointer(pData.advanced(by: srcRow + refX + 16)).loadUnaligned(as: SIMD16<Int16>.self)
                                let c2 = UnsafeRawPointer(pData.advanced(by: srcRow + refX + 32)).loadUnaligned(as: SIMD16<Int16>.self)
                                let c3 = UnsafeRawPointer(pData.advanced(by: srcRow + refX + 48)).loadUnaligned(as: SIMD16<Int16>.self)
                                let pDst = UnsafeMutableRawPointer(pOut.advanced(by: dstRow + startX))
                                pDst.storeBytes(of: c0, as: SIMD16<Int16>.self)
                                pDst.advanced(by: 32).storeBytes(of: c1, as: SIMD16<Int16>.self)
                                pDst.advanced(by: 64).storeBytes(of: c2, as: SIMD16<Int16>.self)
                                pDst.advanced(by: 96).storeBytes(of: c3, as: SIMD16<Int16>.self)
                            }
                        case 32:
                            for y in 0..<actH {
                                let dstRow = (startY + y) * pW
                                let srcRow = (refY + y) * pW
                                let c0 = UnsafeRawPointer(pData.advanced(by: srcRow + refX)).loadUnaligned(as: SIMD16<Int16>.self)
                                let c1 = UnsafeRawPointer(pData.advanced(by: srcRow + refX + 16)).loadUnaligned(as: SIMD16<Int16>.self)
                                let pDst = UnsafeMutableRawPointer(pOut.advanced(by: dstRow + startX))
                                pDst.storeBytes(of: c0, as: SIMD16<Int16>.self)
                                pDst.advanced(by: 32).storeBytes(of: c1, as: SIMD16<Int16>.self)
                            }
                        default:
                            for y in 0..<actH {
                                let dstRow = (startY + y) * pW
                                let srcRow = (refY + y) * pW
                                pOut.advanced(by: dstRow + startX).update(from: pData.advanced(by: srcRow + refX), count: actW)
                            }
                        }
                    } else {
                        let minSafeX = max(0, min(actW, -refX))
                        let maxSafeX = max(0, min(actW, pW - refX))

                        for y in 0..<actH {
                            let dstY = startY + y
                            let srcY = max(0, min(pH - 1, dstY + dy))
                            let dstRow = dstY * pW
                            let srcRow = srcY * pW
                            
                            let pDstBase = pOut.advanced(by: dstRow + startX)

                            if 0 < minSafeX {
                                let leftEdgeVal = pData[srcRow]
                                for x in 0..<minSafeX {
                                    pDstBase[x] = leftEdgeVal
                                }
                            }
                            
                            let copyCount = maxSafeX - minSafeX
                            if 0 < copyCount {
                                pDstBase.advanced(by: minSafeX).update(from: pData.advanced(by: srcRow + refX + minSafeX), count: copyCount)
                            }
                            
                            if maxSafeX < actW {
                                let rightEdgeVal = pData[srcRow + pW - 1]
                                for x in maxSafeX..<actW {
                                    pDstBase[x] = rightEdgeVal
                                }
                            }
                        }
                    }
                }

                for mbY in 0..<fullMbRows {
                    for mbX in 0..<fullMbCols {
                        applyBlock(mbX: mbX, mbY: mbY, actW: pMbSize, actH: pMbSize)
                    }
                }

                if 0 < remW {
                    let mbX = fullMbCols
                    for mbY in 0..<fullMbRows {
                        applyBlock(mbX: mbX, mbY: mbY, actW: remW, actH: pMbSize)
                    }
                }

                if 0 < remH {
                    let mbY = fullMbRows
                    for mbX in 0..<fullMbCols {
                        applyBlock(mbX: mbX, mbY: mbY, actW: pMbSize, actH: remH)
                    }
                }

                if 0 < remW && 0 < remH {
                    let mbX = fullMbCols
                    let mbY = fullMbRows
                    applyBlock(mbX: mbX, mbY: mbY, actW: remW, actH: remH)
                }
            }
        }
        return out
    }

    async let yTask = apply(data: prev.y, pW: w, pH: h, div: 1)
    async let cbTask = apply(data: prev.cb, pW: (w + 1) / 2, pH: (h + 1) / 2, div: 2)
    async let crTask = apply(data: prev.cr, pW: (w + 1) / 2, pH: (h + 1) / 2, div: 2)

    return PlaneData420(width: w, height: h, y: await yTask, cb: await cbTask, cr: await crTask)
}

@inline(__always)
func calculatePMV(mvs: MotionVectors, mbX: Int, mbY: Int, mbCols: Int) -> (dx: Int, dy: Int) {
    let hasLeft = mbX > 0
    let hasTop = mbY > 0
    let hasTopRight = mbY > 0 && mbX < mbCols - 1

    let idxLeft = hasLeft ? (mbY * mbCols + (mbX - 1)) : -1
    let idxTop = hasTop ? ((mbY - 1) * mbCols + mbX) : -1
    let idxTopRight = hasTopRight ? ((mbY - 1) * mbCols + (mbX + 1)) : -1

    var count = 0
    if hasLeft { count += 1 }
    if hasTop { count += 1 }
    if hasTopRight { count += 1 }

    if count == 0 {
        return (0, 0)
    }
    if count == 1 {
        let vec = mvs.vectors[hasLeft ? idxLeft : (hasTop ? idxTop : idxTopRight)]
        return (Int(vec.x), Int(vec.y))
    }
    if count == 2 {
        var dxSum = 0
        var dySum = 0
        if hasLeft { let v = mvs.vectors[idxLeft]; dxSum += Int(v.x); dySum += Int(v.y) }
        if hasTop { let v = mvs.vectors[idxTop]; dxSum += Int(v.x); dySum += Int(v.y) }
        if hasTopRight { let v = mvs.vectors[idxTopRight]; dxSum += Int(v.x); dySum += Int(v.y) }
        return (dxSum / 2, dySum / 2)
    }
    
    let lVec = mvs.vectors[idxLeft]; let lx = Int(lVec.x); let ly = Int(lVec.y)
    let tVec = mvs.vectors[idxTop]; let tx = Int(tVec.x); let ty = Int(tVec.y)
    let rVec = mvs.vectors[idxTopRight]; let rx = Int(rVec.x); let ry = Int(rVec.y)

    let minX = min(lx, min(tx, rx))
    let maxX = max(lx, max(tx, rx))
    let pmvX = lx + tx + rx - minX - maxX

    let minY = min(ly, min(ty, ry))
    let maxY = max(ly, max(ty, ry))
    let pmvY = ly + ty + ry - minY - maxY

    return (pmvX, pmvY)
}
