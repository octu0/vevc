import Foundation

@inline(__always)
func calculateSAD32x32(pCurr: UnsafePointer<Int16>, pPrev: UnsafePointer<Int16>, currStride: Int, prevStride: Int) -> Int {
    var sad: UInt = 0
    for y in 0..<32 {
        let currRow = pCurr.advanced(by: y * currStride)
        let prevRow = pPrev.advanced(by: y * prevStride)

        for x in 0..<32 {
            let diff = Int(currRow[x]) - Int(prevRow[x])
            sad &+= UInt(diff > 0 ? diff : -diff)
        }
    }
    return Int(sad)
}

public struct MotionVector: Sendable {
    public let dx: Int
    public let dy: Int
}

func estimateMBME(curr: PlaneData420, prev: PlaneData420) -> [MotionVector] {
    let mbSize = 32
    let w = curr.width
    let h = curr.height
    let mbCols = (w + mbSize - 1) / mbSize
    let mbRows = (h + mbSize - 1) / mbSize

    var mvs = [MotionVector](repeating: MotionVector(dx: 0, dy: 0), count: mbCols * mbRows)

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
                                        let currRow = (startY + y) * w
                                        let prevRow = (refY + y) * w
                                        for x in 0..<actW {
                                            let diff = Int(pCurr[currRow + startX + x]) - Int(pPrev[prevRow + refX + x])
                                            s &+= UInt(diff > 0 ? diff : -diff)
                                        }
                                    }
                                    sad = Int(s)
                                }
                            } else {
                                sad = calculateSADEdge(pCurr: pCurr, pPrev: pPrev, w: w, h: h, startX: startX, startY: startY, actW: actW, actH: actH, dx: dx, dy: dy)
                            }


                            sad += (abs(dx) + abs(dy))

                            if sad < bestSAD {
                                bestSAD = sad
                                bestDX = dx
                                bestDY = dy
                            }
                        }
                    }

                    mvs[mbY * mbCols + mbX] = MotionVector(dx: bestDX, dy: bestDY)
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

        for x in 0..<actW {
            let cx = startX + x
            let px = max(0, min(w - 1, cx + dx))

            let diff = Int(pCurr[currRow + cx]) - Int(pPrev[prevRow + px])
            sad &+= UInt(diff > 0 ? diff : -diff)
        }
    }
    return Int(sad)
}

func applyMBME(prev: PlaneData420, mvs: [MotionVector]) async -> PlaneData420 {
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

                        let mv = mvs[mbY * localMbCols + mbX]
                        let dx = mv.dx / div
                        let dy = mv.dy / div

                        
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
