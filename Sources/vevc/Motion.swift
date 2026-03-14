import Foundation

@inline(__always)
func calculateSAD32x32(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, currStride: Int, prevStride: Int) -> Int {
    var sad: Int32 = 0
    for y in 0..<32 {
        let currRow = pCurr.advanced(by: y * currStride)
        let prevRow = pPrev.advanced(by: y * prevStride)

        // SIMD16を展開して高速化
        let c0 = UnsafeRawPointer(currRow).loadUnaligned(as: SIMD16<Int16>.self)
        let p0 = UnsafeRawPointer(prevRow).loadUnaligned(as: SIMD16<Int16>.self)
        let c1 = UnsafeRawPointer(currRow.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)
        let p1 = UnsafeRawPointer(prevRow.advanced(by: 16)).loadUnaligned(as: SIMD16<Int16>.self)

        let diff0 = c0 &- p0
        let diff1 = c1 &- p1

        // 負の数を絶対値に変換 (ビット演算によるabs)
        // mask = diff >> 15 (正: 0, 負: -1)
        // (diff ^ mask) - mask
        let mask0 = diff0 &>> 15
        let abs0 = (diff0 ^ mask0) &- mask0

        let mask1 = diff1 &>> 15
        let abs1 = (diff1 ^ mask1) &- mask1

        // オーバーフローを避けるためInt32に拡張してから合算
        let sum0 = SIMD16<Int32>(clamping: abs0).wrappedSum()
        let sum1 = SIMD16<Int32>(clamping: abs1).wrappedSum()

        sad &+= sum0 &+ sum1
    }
    return Int(sad)
}

struct MotionVector: Sendable {
    let dx: Int
    let dy: Int
}

struct MotionVectors: Sendable {
    var dx: [Int]
    var dy: [Int]

    init(count: Int) {
        self.dx = [Int](repeating: 0, count: count)
        self.dy = [Int](repeating: 0, count: count)
    }
}

@inline(__always)
func estimateMBME(curr: PlaneData420, prev: PlaneData420) -> MotionVectors {
    let mbSize = 32
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

            for mbY in 0..<mbRows {
                let startY = mbY * mbSize
                let actH = min(mbSize, h - startY)

                for mbX in 0..<mbCols {
                    let startX = mbX * mbSize
                    let actW = min(mbSize, w - startX)

                    var bestSAD = Int.max
                    var bestDX = 0
                    var bestDY = 0

                    if actW == 32 && actH == 32 {
                        bestSAD = calculateSAD32x32(
                            pCurr: pCurr.advanced(by: startY * w + startX),
                            pPrev: pPrev.advanced(by: startY * w + startX),
                            currStride: w, prevStride: w
                        )
                    } else {
                        bestSAD = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: 0, dy: 0)
                    }

                    for dy in -searchRange...searchRange {
                        for dx in -searchRange...searchRange {
                            if dx == 0 && dy == 0 { continue }

                            let refX = startX + dx
                            let refY = startY + dy

                            let penalty = (dx >= 0 ? dx : -1 * dx) + (dy >= 0 ? dy : -1 * dy)
                            if bestSAD <= penalty { continue }

                            var sad = 0
                            if refX >= 0 && refY >= 0 && refX + actW <= w && refY + actH <= h {
                                if actW == 32 && actH == 32 {
                                    sad = calculateSAD32x32(
                                        pCurr: pCurr.advanced(by: startY * w + startX),
                                        pPrev: pPrev.advanced(by: refY * w + refX),
                                        currStride: w, prevStride: w
                                    )
                                } else {
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
                                }
                            } else {
                                sad = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: dx, dy: dy)
                            }


                            sad &+= penalty

                            if sad < bestSAD {
                                bestSAD = sad
                                bestDX = dx
                                bestDY = dy
                            }
                        }
                    }

                    let idx = mbY * mbCols + mbX
                    mvs.dx[idx] = bestDX
                    mvs.dy[idx] = bestDY
                }
            }
        }
    }

    return mvs
}

@inline(__always)
func calculateSADEdge(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, w: Int, h: Int, startX: Int, startY: Int, actW: Int, actH: Int, dx: Int, dy: Int) -> Int {
    var sad: UInt = 0
    for y in 0..<actH {
        let cy = startY + y
        let currRow = cy * w

        let py = max(0, min(h - 1, cy + dy))
        let prevRow = py * w

        let pCurrRow = pCurr.advanced(by: currRow + startX)
        let pPrevRow = pPrev.advanced(by: prevRow)

        for x in 0..<actW {
            let cx = startX + x
            let px = max(0, min(w - 1, cx + dx))

            let diff = Int(pCurrRow[x]) - Int(pPrevRow[px])

            let mask = diff >> 31
            let absDiff = (diff ^ mask) - mask
            sad &+= UInt(absDiff)
        }
    }
    return Int(sad)
}

@inline(__always)
func applyMBME(prev: PlaneData420, mvs: MotionVectors) async -> PlaneData420 {
    let mbSize = 32
    let w = prev.width
    let h = prev.height
    let mbCols = (w + mbSize - 1) / mbSize

    @Sendable
    func apply(data: [Int16], pW: Int, pH: Int, div: Int) async -> [Int16] {
        if pW == 0 || pH == 0 { return data }
        var out = [Int16](repeating: 0, count: pW * pH)

        let localMbCols = mbCols
        let pMbSize = mbSize / div

        data.withUnsafeBufferPointer { pPtr in
            guard let pData = pPtr.baseAddress else { return }
            out.withUnsafeMutableBufferPointer { oPtr in
                guard let pOut = oPtr.baseAddress else { return }

                for mbY in 0..<((pH + pMbSize - 1) / pMbSize) {
                    let startY = mbY * pMbSize
                    let actH = min(pMbSize, pH - startY)

                    for mbX in 0..<((pW + pMbSize - 1) / pMbSize) {
                        let startX = mbX * pMbSize
                        let actW = min(pMbSize, pW - startX)

                        let idx = mbY * localMbCols + mbX
                        let dx = mvs.dx[idx] / div
                        let dy = mvs.dy[idx] / div

                        
                        let refX = startX + dx
                        let refY = startY + dy
                        
                        if refX >= 0 && refY >= 0 && refX + actW <= pW && refY + actH <= pH {
                            for y in 0..<actH {
                                let dstRow = (startY + y) * pW
                                let srcRow = (refY + y) * pW
                                for x in 0..<actW {
                                    pOut[dstRow + startX + x] = pData[srcRow + refX + x]
                                }
                            }
                        } else {
                            for y in 0..<actH {
                                let dstY = startY + y
                                let srcY = max(0, min(pH - 1, dstY + dy))
                                let dstRow = dstY * pW
                                let srcRow = srcY * pW

                                for x in 0..<actW {
                                    let dstX = startX + x
                                    let srcX = max(0, min(pW - 1, dstX + dx))
                                    pOut[dstRow + dstX] = pData[srcRow + srcX]
                                }
                            }
                        }

                    }
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
    } else if count == 1 {
        let idx = hasLeft ? idxLeft : (hasTop ? idxTop : idxTopRight)
        return (mvs.dx[idx], mvs.dy[idx])
    } else if count == 2 {
        var dxSum = 0
        var dySum = 0
        if hasLeft { dxSum += mvs.dx[idxLeft]; dySum += mvs.dy[idxLeft] }
        if hasTop { dxSum += mvs.dx[idxTop]; dySum += mvs.dy[idxTop] }
        if hasTopRight { dxSum += mvs.dx[idxTopRight]; dySum += mvs.dy[idxTopRight] }
        return (dxSum / 2, dySum / 2)
    } else {
        let lx = mvs.dx[idxLeft]; let ly = mvs.dy[idxLeft]
        let tx = mvs.dx[idxTop]; let ty = mvs.dy[idxTop]
        let rx = mvs.dx[idxTopRight]; let ry = mvs.dy[idxTopRight]

        let minX = min(lx, min(tx, rx))
        let maxX = max(lx, max(tx, rx))
        let pmvX = lx + tx + rx - minX - maxX

        let minY = min(ly, min(ty, ry))
        let maxY = max(ly, max(ty, ry))
        let pmvY = ly + ty + ry - minY - maxY

        return (pmvX, pmvY)
    }
}
