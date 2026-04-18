import Foundation

struct MotionVector: Sendable {
    let dx: Int16
    let dy: Int16
    
    init(dx: Int16, dy: Int16) {
        self.dx = dx
        self.dy = dy
    }
    
    @inline(__always)
    var isIntra: Bool {
        return dx == 32767 && dy == 32767
    }
    
    static let intraBlock = MotionVector(dx: 32767, dy: 32767)
}

struct MotionEstimation {

    // small blocks benefit more from compiler auto-vectorization than manual SIMD
    @inline(__always)
    static func absDiff(_ a: Int16, _ b: Int16) -> Int {
        return abs(Int(a) - Int(b))
    }

    // why: Lagrange multiplier for rate-distortion optimization:
    // penalizes motion vectors with large magnitude to favor zero-MV
    static func getPenalty(dx: Int, dy: Int, pmv: MotionVector, lambda: Int) -> Int {
        return (abs(dx) + abs(dy)) * lambda
    }

    @inline(__always)
    static func fetchPixelsBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, x: Int, y: Int, dest: UnsafeMutablePointer<Int16>) {
        if x >= 0 && y >= 0 && x + 8 <= width && y + 8 <= height {
            // Fast path: block is fully within image bounds
            for ry in 0..<8 {
                let offset = (y + ry) * width + x
                let srcPtr = plane.advanced(by: offset)
                let dstPtr = dest.advanced(by: ry * 8)
                for rx in 0..<8 {
                    dstPtr[rx] = srcPtr[rx]
                }
            }
        } else {
            // Slow path: clamp to bounds
            for i in 0..<64 {
                let ry = i / 8
                let rx = i % 8
                let srcY = min(max(0, y + ry), height - 1)
                let srcX = min(max(0, x + rx), width - 1)
                dest[i] = plane[srcY * width + srcX]
            }
        }
    }

    @inline(__always)
    static func fetchHalfPixelBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, intX: Int, intY: Int, fractX: Int, fractY: Int, dest: UnsafeMutablePointer<Int16>, roundOffset: Int) {
        if fractX == 0 && fractY == 0 {
            fetchPixelsBlock8(plane: plane, width: width, height: height, x: intX, y: intY, dest: dest)
            return
        }
        
        if intX >= 0 && intY >= 0 && intX + 8 + fractX <= width && intY + 8 + fractY <= height {
            if fractY == 0 {
                for ry in 0..<8 {
                    let row = plane.advanced(by: (intY + ry) * width + intX)
                    let dst = dest.advanced(by: ry * 8)
                    for rx in 0..<8 { dst[rx] = Int16((Int(row[rx]) + Int(row[rx + 1]) + roundOffset) >> 1) }
                }
            } else if fractX == 0 {
                for ry in 0..<8 {
                    let row0 = plane.advanced(by: (intY + ry) * width + intX)
                    let row1 = plane.advanced(by: (intY + ry + 1) * width + intX)
                    let dst = dest.advanced(by: ry * 8)
                    for rx in 0..<8 { dst[rx] = Int16((Int(row0[rx]) + Int(row1[rx]) + roundOffset) >> 1) }
                }
            } else {
                for ry in 0..<8 {
                    let row0 = plane.advanced(by: (intY + ry) * width + intX)
                    let row1 = plane.advanced(by: (intY + ry + 1) * width + intX)
                    let dst = dest.advanced(by: ry * 8)
                    for rx in 0..<8 { dst[rx] = Int16((Int(row0[rx]) + Int(row0[rx+1]) + Int(row1[rx]) + Int(row1[rx+1]) + 1 + roundOffset) >> 2) }
                }
            }
            return
        }
        
        for ry in 0..<8 {
            let sy0 = max(0, min(intY + ry, height - 1))
            let sy1 = max(0, min(intY + ry + fractY, height - 1))
            let row0 = plane.advanced(by: sy0 * width)
            let row1 = plane.advanced(by: sy1 * width)
            let dstPtr = dest.advanced(by: ry * 8)
            for rx in 0..<8 {
                let sx0 = max(0, min(intX + rx, width - 1))
                let sx1 = max(0, min(intX + rx + fractX, width - 1))
                if fractY == 0 {
                    dstPtr[rx] = Int16((Int(row0[sx0]) + Int(row0[sx1]) + roundOffset) >> 1)
                } else if fractX == 0 {
                    dstPtr[rx] = Int16((Int(row0[sx0]) + Int(row1[sx0]) + roundOffset) >> 1)
                } else {
                    dstPtr[rx] = Int16((Int(row0[sx0]) + Int(row0[sx1]) + Int(row1[sx0]) + Int(row1[sx1]) + 1 + roundOffset) >> 2)
                }
            }
        }
    }

    @inline(__always)
    static func fetchQuarterPixelBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, intX: Int, intY: Int, remX: Int, remY: Int, dest: UnsafeMutablePointer<Int16>, roundOffset: Int) {
        if remX == 0 && remY == 0 {
            fetchPixelsBlock8(plane: plane, width: width, height: height, x: intX, y: intY, dest: dest)
            return
        }
        let nextX = remX == 0 ? 0 : 1
        let nextY = remY == 0 ? 0 : 1
        let wA = 4 - remX
        let wB = remX
        let wC = 4 - remY
        let wD = remY
        
        if intX >= 0 && intY >= 0 && intX + 8 + nextX <= width && intY + 8 + nextY <= height {
            for ry in 0..<8 {
                let row0 = plane.advanced(by: (intY + ry) * width + intX)
                let row1 = plane.advanced(by: (intY + ry + nextY) * width + intX)
                let dst = dest.advanced(by: ry * 8)
                for rx in 0..<8 {
                    let v = wA * wC * Int(row0[rx]) + wB * wC * Int(row0[rx + nextX]) + wA * wD * Int(row1[rx]) + wB * wD * Int(row1[rx + nextX])
                    dst[rx] = Int16((v + 7 + roundOffset) >> 4)
                }
            }
            return
        }
        for ry in 0..<8 {
            let sy0 = max(0, min(intY + ry, height - 1))
            let sy1 = max(0, min(intY + ry + nextY, height - 1))
            let row0 = plane.advanced(by: sy0 * width)
            let row1 = plane.advanced(by: sy1 * width)
            let dst = dest.advanced(by: ry * 8)
            for rx in 0..<8 {
                let sx0 = max(0, min(intX + rx, width - 1))
                let sx1 = max(0, min(intX + rx + nextX, width - 1))
                let v = wA * wC * Int(row0[sx0]) + wB * wC * Int(row0[sx1]) + wA * wD * Int(row1[sx0]) + wB * wD * Int(row1[sx1])
                dst[rx] = Int16((v + 7 + roundOffset) >> 4)
            }
        }
    }

    @inline(__always)
    static func fetchEighthPixelBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, intX: Int, intY: Int, remX: Int, remY: Int, dest: UnsafeMutablePointer<Int16>, roundOffset: Int) {
        if remX == 0 && remY == 0 {
            fetchPixelsBlock8(plane: plane, width: width, height: height, x: intX, y: intY, dest: dest)
            return
        }
        let nextX = remX == 0 ? 0 : 1
        let nextY = remY == 0 ? 0 : 1
        let wA = 8 - remX
        let wB = remX
        let wC = 8 - remY
        let wD = remY
        
        if intX >= 0 && intY >= 0 && intX + 8 + nextX <= width && intY + 8 + nextY <= height {
            for ry in 0..<8 {
                let row0 = plane.advanced(by: (intY + ry) * width + intX)
                let row1 = plane.advanced(by: (intY + ry + nextY) * width + intX)
                let dst = dest.advanced(by: ry * 8)
                for rx in 0..<8 {
                    let v = wA * wC * Int(row0[rx]) + wB * wC * Int(row0[rx + nextX]) + wA * wD * Int(row1[rx]) + wB * wD * Int(row1[rx + nextX])
                    dst[rx] = Int16((v + 31 + roundOffset) >> 6)
                }
            }
            return
        }
        for ry in 0..<8 {
            let sy0 = max(0, min(intY + ry, height - 1))
            let sy1 = max(0, min(intY + ry + nextY, height - 1))
            let row0 = plane.advanced(by: sy0 * width)
            let row1 = plane.advanced(by: sy1 * width)
            let dst = dest.advanced(by: ry * 8)
            for rx in 0..<8 {
                let sx0 = max(0, min(intX + rx, width - 1))
                let sx1 = max(0, min(intX + rx + nextX, width - 1))
                let v = wA * wC * Int(row0[sx0]) + wB * wC * Int(row0[sx1]) + wA * wD * Int(row1[sx0]) + wB * wD * Int(row1[sx1])
                dst[rx] = Int16((v + 31 + roundOffset) >> 6)
            }
        }
    }

    @inline(__always)
    static func compute64PointSAD_Blocks(cBase: UnsafePointer<Int16>, pBase: UnsafePointer<Int16>) -> Int {
        var sad: Int32 = 0
        for i in 0..<64 {
            let diff = Int32(cBase[i]) - Int32(pBase[i])
            sad &+= diff < 0 ? -diff : diff
        }
        return Int(sad)
    }
    @inline(__always)
    static func compute32PointSAD_EvenRows(cBase: UnsafePointer<Int16>, pBase: UnsafePointer<Int16>) -> Int {
        var sad: Int32 = 0
        for row in 0..<4 {
            let offset = row * 16
            let cRow = cBase.advanced(by: offset)
            let pRow = pBase.advanced(by: offset)
            for x in 0..<8 {
                let diff = Int32(cRow[x]) - Int32(pRow[x])
                sad &+= diff < 0 ? -diff : diff
            }
        }
        return Int(sad) * 2
    }
    
    @inline(__always)
    static func median(_ a: Int, _ b: Int, _ c: Int) -> Int {
        return max(min(a, b), min(max(a, b), c))
    }
    @inline(__always)
    private static func evaluateSearch(
        cPtr: UnsafePointer<Int16>, 
        pBase: UnsafePointer<Int16>, 
        oPtr: UnsafeMutablePointer<Int16>,
        tPtr: UnsafeMutablePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int, range: Int, pmv: MotionVector, roundOffset: Int
    ) -> (Int, Int, Int) {
        fetchPixelsBlock8(plane: pBase, width: width, height: height, x: bx, y: by, dest: oPtr)
        let zeroSad: Int = compute64PointSAD_Blocks(cBase: cPtr, pBase: oPtr)
        
        if zeroSad < 64 {
            return (0, 0, zeroSad)
        }
        
        var bestCoarseSad: Int = zeroSad
        var bestCoarseDx: Int = 0
        var bestCoarseDy: Int = 0
        
        let minDy = max(-range, -by)
        let maxDy = min(range, height - by - 8)
        let minDx = max(-range, -bx)
        let maxDx = min(range, width - bx - 8)
        
        if minDy <= maxDy && minDx <= maxDx {
            for dy in minDy...maxDy {
                for dx in minDx...maxDx {
                    if dx == 0 && dy == 0 { continue }
                    
                    let penalty: Int = getPenalty(dx: dx, dy: dy, pmv: pmv, lambda: 8)
                    let maxSad = bestCoarseSad - penalty
                    if maxSad < 0 { continue }
                    
                    fetchPixelsBlock8(plane: pBase, width: width, height: height, x: bx + dx, y: by + dy, dest: tPtr)
                    let sad: Int = compute64PointSAD_Blocks(cBase: cPtr, pBase: tPtr)
                    
                    let totalSad: Int = sad + penalty
                    if totalSad < bestCoarseSad {
                        bestCoarseSad = totalSad
                        bestCoarseDx = dx
                        bestCoarseDy = dy
                    }
                }
            }
        }
        
        var bestFineSad: Int = bestCoarseSad
        var bestFineDx: Int = bestCoarseDx
        var bestFineDy: Int = bestCoarseDy
        
        let fineOffsets: [(Int, Int)] = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1)
        ]
        
        for offset in fineOffsets {
            let fx: Int = offset.0
            let fy: Int = offset.1
            let fineDx: Int = bestCoarseDx + fx
            let fineDy: Int = bestCoarseDy + fy
            
            if fineDx < -4 || 4 < fineDx || fineDy < -4 || 4 < fineDy { continue }
            
            let penalty: Int = getPenalty(dx: fineDx, dy: fineDy, pmv: pmv, lambda: 8)
            let maxSad = bestFineSad - penalty
            if maxSad < 0 { continue }
            
            fetchPixelsBlock8(plane: pBase, width: width, height: height, x: bx + fineDx, y: by + fineDy, dest: tPtr)
            let sad: Int = compute64PointSAD_Blocks(cBase: cPtr, pBase: tPtr)
            
            let totalSad: Int = sad + penalty
            if totalSad < bestFineSad {
                bestFineSad = totalSad
                bestFineDx = fineDx
                bestFineDy = fineDy
            }
        }
        
        var bestHpDx: Int = bestFineDx * 2
        var bestHpDy: Int = bestFineDy * 2
        var bestHpSad: Int = bestFineSad
        
        if 256 < bestFineSad {
            for oi in 0..<8 {
                let hx = oi == 0 ? -1 : (oi == 1 ? 0 : (oi == 2 ? 1 : (oi == 3 ? -1 : (oi == 4 ? 1 : (oi == 5 ? -1 : (oi == 6 ? 0 : 1))))))
                let hy = oi < 3 ? -1 : (oi < 5 ? 0 : 1)
                let hpDx: Int = bestFineDx * 2 + hx
                let hpDy: Int = bestFineDy * 2 + hy
                
                let intDx: Int = hpDx >> 1
                let intDy: Int = hpDy >> 1
                let fractX: Int = hpDx & 1
                let fractY: Int = hpDy & 1
                
                let penalty: Int = getPenalty(dx: hpDx, dy: hpDy, pmv: pmv, lambda: 4)
                let maxSad = bestHpSad - penalty
                if maxSad < 0 { continue }
                
                fetchHalfPixelBlock8(plane: pBase, width: width, height: height,
                                     intX: bx + intDx, intY: by + intDy,
                                     fractX: fractX, fractY: fractY, dest: tPtr, roundOffset: roundOffset)
                let sad: Int = compute64PointSAD_Blocks(cBase: cPtr, pBase: tPtr)
                
                let totalSad: Int = sad + penalty
                if totalSad < bestHpSad {
                    bestHpSad = totalSad
                    bestHpDx = hpDx
                    bestHpDy = hpDy
                }
            }
        }
        
        var bestEpDx: Int = bestHpDx * 4
        var bestEpDy: Int = bestHpDy * 4
        var bestEpSad: Int = bestHpSad
        
        if 128 < bestHpSad {
            for oi in 0..<8 {
                let hx = oi == 0 ? -1 : (oi == 1 ? 0 : (oi == 2 ? 1 : (oi == 3 ? -1 : (oi == 4 ? 1 : (oi == 5 ? -1 : (oi == 6 ? 0 : 1))))))
                let hy = oi < 3 ? -1 : (oi < 5 ? 0 : 1)
                let epDx: Int = bestHpDx * 4 + hx
                let epDy: Int = bestHpDy * 4 + hy
                
                let intDx: Int = epDx >> 3
                let intDy: Int = epDy >> 3
                let remX: Int = epDx & 7
                let remY: Int = epDy & 7
                
                let penalty: Int = getPenalty(dx: epDx, dy: epDy, pmv: pmv, lambda: 2)
                let maxSad = bestEpSad - penalty
                if maxSad < 0 { continue }
                
                fetchEighthPixelBlock8(plane: pBase, width: width, height: height,
                                       intX: bx + intDx, intY: by + intDy,
                                       remX: remX, remY: remY, dest: tPtr, roundOffset: roundOffset)
                let sad: Int = compute64PointSAD_Blocks(cBase: cPtr, pBase: tPtr)
                
                let totalSad: Int = sad + penalty
                if totalSad < bestEpSad {
                    bestEpSad = totalSad
                    bestEpDx = epDx
                    bestEpDy = epDy
                }
            }
        }
        
        return (bestEpDx, bestEpDy, bestEpSad)
    }

    @inline(__always)
    static func searchPixels(
        currPlane: [Int16], 
        prevPlane: [Int16], 
        cPtr: UnsafeMutablePointer<Int16>,
        oPtr: UnsafeMutablePointer<Int16>,
        tPtr: UnsafeMutablePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int, range: Int = 4, pmv: MotionVector, roundOffset: Int
    ) -> (MotionVector, Int) {
        return currPlane.withUnsafeBufferPointer { cBuf in
            prevPlane.withUnsafeBufferPointer { pBuf in
                guard let cBase = cBuf.baseAddress, let pBase = pBuf.baseAddress else {
                    return (MotionVector(dx: 0, dy: 0), 0)
                }

                fetchPixelsBlock8(plane: cBase, width: width, height: height, x: bx, y: by, dest: cPtr)
                let (dx, dy, sad) = evaluateSearch(cPtr: cPtr, pBase: pBase, oPtr: oPtr, tPtr: tPtr, width: width, height: height, bx: bx, by: by, range: range, pmv: pmv, roundOffset: roundOffset)
                return (MotionVector(dx: Int16(dx), dy: Int16(dy)), sad)
            }
        }
    }

    /// Extract approximate structure contrast (max - min) from 8x8 block
    /// Zero-cost feature extraction without additional SIMD loop overheads.
    @inline(__always)
    static func extractContrast8x8(plane: [Int16], width: Int, height: Int, bx: Int, by: Int) -> Int {
        var minVal: Int32 = 32767
        var maxVal: Int32 = -32768
        
        plane.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            
            if bx >= 0 && by >= 0 && bx + 8 <= width && by + 8 <= height {
                for y in 0..<8 {
                    let row = base.advanced(by: (by + y) * width + bx)
                    for x in 0..<8 {
                        let val = Int32(row[x])
                        if val < minVal { minVal = val }
                        if maxVal < val { maxVal = val }
                    }
                }
            } else {
                for y in 0..<8 {
                    let sy = max(0, min(by + y, height - 1))
                    let row = base.advanced(by: sy * width)
                    for x in 0..<8 {
                        let sx = max(0, min(bx + x, width - 1))
                        let val = Int32(row[sx])
                        if val < minVal { minVal = val }
                        if maxVal < val { maxVal = val }
                    }
                }
            }
        }
        
        return Int(maxVal - minVal)
    }

    @inline(__always)
    static func computeChromaSAD(
        curr: PlaneData420, ref: PlaneData420,
        bx: Int, by: Int, refDx: Int, refDy: Int
    ) -> Int {
        let cbw = (curr.width + 1) / 2
        let cbh = (curr.height + 1) / 2
        let bx2 = bx * 2
        let by2 = by * 2
        let refX2 = bx2 + (refDx * 2)
        let refY2 = by2 + (refDy * 2)
        
        var chromaSAD = 1000
        if bx2 >= 0 && by2 >= 0 && bx2 + 16 <= cbw && by2 + 16 <= cbh &&
           refX2 >= 0 && refY2 >= 0 && refX2 + 16 <= cbw && refY2 + 16 <= cbh {
            curr.cb.withUnsafeBufferPointer { cbCurrBuf in
            curr.cr.withUnsafeBufferPointer { crCurrBuf in
            ref.cb.withUnsafeBufferPointer { cbRefBuf in
            ref.cr.withUnsafeBufferPointer { crRefBuf in
                let cCb = cbCurrBuf.baseAddress!
                let cCr = crCurrBuf.baseAddress!
                let rCb = cbRefBuf.baseAddress!
                let rCr = crRefBuf.baseAddress!
                
                var sad: Int32 = 0
                for cy in stride(from: 0, to: 16, by: 2) {
                    let currOffset = (by2 + cy) * cbw + bx2
                    let refOffset = (refY2 + cy) * cbw + refX2
                    for cx in stride(from: 0, to: 16, by: 2) {
                        let diffCb = Int32(cCb[currOffset + cx]) - Int32(rCb[refOffset + cx])
                        sad &+= diffCb < 0 ? -diffCb : diffCb
                        let diffCr = Int32(cCr[currOffset + cx]) - Int32(rCr[refOffset + cx])
                        sad &+= diffCr < 0 ? -diffCr : diffCr
                    }
                    if 2000 < sad { break } // Early Termination
                }
                chromaSAD = Int(sad) * 4 // 間引いた分をスケールアップ
            }}}}
        }
        return chromaSAD
    }

    @inline(__always)
    static func computeQuarterPixelSAD_Subsampled32(
        curr: UnsafePointer<Int16>, 
        prev: UnsafePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int,
        qDx: Int, qDy: Int
    ) -> Int {
        let intDx = qDx >> 2
        let intDy = qDy >> 2
        let fractX = qDx & 3
        let fractY = qDy & 3
        
        let fX = FIRLUMACoeffs[fractX]
        let fY = FIRLUMACoeffs[fractY]
        
        var sad: Int32 = 0
        let safe = (bx + intDx - 1 >= 0) && (by + intDy - 1 >= 0) && (bx + intDx + 32 + 2 < width) && (by + intDy + 32 + 2 < height)
        let useFIR = (fractX != 0 || fractY != 0)
        
        let cX0 = Int32(fX[0]), cX1 = Int32(fX[1]), cX2 = Int32(fX[2]), cX3 = Int32(fX[3])
        let cY0 = Int32(fY[0]), cY1 = Int32(fY[1]), cY2 = Int32(fY[2]), cY3 = Int32(fY[3])
        
        if safe {
            for ry in stride(from: 0, to: 32, by: 4) {
                let cy = by + ry
                let rowC = curr.advanced(by: cy * width + bx)
                
                let py = by + intDy + ry
                if !useFIR {
                    let r = prev.advanced(by: py * width + bx + intDx)
                    for rx in stride(from: 0, to: 32, by: 2) {
                        let diff = Int32(rowC[rx]) - Int32(r[rx])
                        sad &+= diff < 0 ? -diff : diff
                    }
                    continue
                }
                
                let rM1 = prev.advanced(by: (py - 1) * width + bx + intDx)
                let r0 = prev.advanced(by: py * width + bx + intDx)
                let rP1 = prev.advanced(by: (py + 1) * width + bx + intDx)
                let rP2 = prev.advanced(by: (py + 2) * width + bx + intDx)
                
                var rx = 0
                while rx < 32 {
                    let vM1 = cX0 &* Int32(rM1[rx - 1]) &+ cX1 &* Int32(rM1[rx]) &+ cX2 &* Int32(rM1[rx + 1]) &+ cX3 &* Int32(rM1[rx + 2])
                    let v0  = cX0 &* Int32(r0[rx - 1])  &+ cX1 &* Int32(r0[rx])  &+ cX2 &* Int32(r0[rx + 1])  &+ cX3 &* Int32(r0[rx + 2])
                    let vP1 = cX0 &* Int32(rP1[rx - 1]) &+ cX1 &* Int32(rP1[rx]) &+ cX2 &* Int32(rP1[rx + 1]) &+ cX3 &* Int32(rP1[rx + 2])
                    let vP2 = cX0 &* Int32(rP2[rx - 1]) &+ cX1 &* Int32(rP2[rx]) &+ cX2 &* Int32(rP2[rx + 1]) &+ cX3 &* Int32(rP2[rx + 2])
                    
                    let refVal = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                    let pVal = (refVal &+ 31) >> 6
                    let diff = Int32(rowC[rx]) &- pVal
                    sad &+= diff < 0 ? -diff : diff
                    rx &+= 2
                }
            }
            return Int(sad)
        }
        
        for ry in stride(from: 0, to: 32, by: 4) {
            let cy = min(by + ry, height - 1)
            let rowC = curr.advanced(by: cy * width)
            
            let py = by + intDy + ry
            if !useFIR {
                let sy0 = max(0, min(py, height - 1))
                let r = prev.advanced(by: sy0 * width)
                for rx in stride(from: 0, to: 32, by: 2) {
                    let px = bx + intDx + rx
                    let sx = max(0, min(px, width - 1))
                    let cx = min(bx + rx, width - 1)
                    let diff = Int32(rowC[cx]) - Int32(r[sx])
                    sad &+= diff < 0 ? -diff : diff
                }
                continue
            }
            
            let syM1 = max(0, min(py - 1, height - 1))
            let sy0  = max(0, min(py, height - 1))
            let syP1 = max(0, min(py + 1, height - 1))
            let syP2 = max(0, min(py + 2, height - 1))
            let rM1 = prev.advanced(by: syM1 * width)
            let r0  = prev.advanced(by: sy0 * width)
            let rP1 = prev.advanced(by: syP1 * width)
            let rP2 = prev.advanced(by: syP2 * width)
            
            var rx = 0
            while rx < 32 {
                let px = bx &+ intDx &+ rx
                let cx = min(bx &+ rx, width - 1)
                
                let sxM1 = max(0, min(px - 1, width - 1))
                let sx0  = max(0, min(px, width - 1))
                let sxP1 = max(0, min(px + 1, width - 1))
                let sxP2 = max(0, min(px + 2, width - 1))
                
                let vM1 = cX0 &* Int32(rM1[sxM1]) &+ cX1 &* Int32(rM1[sx0]) &+ cX2 &* Int32(rM1[sxP1]) &+ cX3 &* Int32(rM1[sxP2])
                let v0  = cX0 &* Int32(r0[sxM1])  &+ cX1 &* Int32(r0[sx0])  &+ cX2 &* Int32(r0[sxP1])  &+ cX3 &* Int32(r0[sxP2])
                let vP1 = cX0 &* Int32(rP1[sxM1]) &+ cX1 &* Int32(rP1[sx0]) &+ cX2 &* Int32(rP1[sxP1]) &+ cX3 &* Int32(rP1[sxP2])
                let vP2 = cX0 &* Int32(rP2[sxM1]) &+ cX1 &* Int32(rP2[sx0]) &+ cX2 &* Int32(rP2[sxP1]) &+ cX3 &* Int32(rP2[sxP2])
                
                let refVal = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let pVal = (refVal &+ 31) >> 6
                let diff = Int32(rowC[cx]) &- pVal
                sad &+= diff < 0 ? -diff : diff
                rx &+= 2
            }
        }
        return Int(sad)
    }

    @inline(__always)
    static func searchPixelsQuarterRefinement32(
        currPlane: [Int16],
        prevPlane: [Int16],
        width: Int, height: Int, bx: Int, by: Int, pmv: MotionVector
    ) -> (MotionVector, Int) {
        return currPlane.withUnsafeBufferPointer { cBuf in
            prevPlane.withUnsafeBufferPointer { pBuf in
                guard let cBase = cBuf.baseAddress, let pBase = pBuf.baseAddress else {
                    return (MotionVector(dx: 0, dy: 0), 0)
                }
                
                // pmv is in 1/8 units of dx/4 == 1/2 units of Luma dx
                // We convert it to 1/4 units of Luma dx by multiplying by 2
                let baseQx = Int(pmv.dx) * 2
                let baseQy = Int(pmv.dy) * 2
                
                var bestQx = baseQx
                var bestQy = baseQy
                
                var bestSad = computeQuarterPixelSAD_Subsampled32(curr: cBase, prev: pBase, width: width, height: height, bx: bx, by: by, qDx: bestQx, qDy: bestQy)
                
                if bestSad < 128 { return (MotionVector(dx: Int16(bestQx), dy: Int16(bestQy)), bestSad) }
                
                let offsets: [(Int, Int)] = [
                    (-1, -1), (0, -1), (1, -1),
                    (-1,  0),          (1,  0),
                    (-1,  1), (0,  1), (1,  1)
                ]
                
                for (ox, oy) in offsets {
                    let qx = baseQx + ox
                    let qy = baseQy + oy
                    
                    let intDx = qx >> 2
                    let intDy = qy >> 2
                    if bx + intDx < -32 || bx + intDx + 32 > width + 32 { continue }
                    if by + intDy < -32 || by + intDy + 32 > height + 32 { continue }
                    
                    let penalty = (abs(ox) + abs(oy)) * 6
                    let maxSad = bestSad - penalty
                    if maxSad <= 0 { continue }
                    
                    let sad = computeQuarterPixelSAD_Subsampled32(curr: cBase, prev: pBase, width: width, height: height, bx: bx, by: by, qDx: qx, qDy: qy)
                    let totalSad = sad + penalty
                    if totalSad < bestSad {
                        bestSad = totalSad
                        bestQx = qx
                        bestQy = qy
                    }
                }
                
                return (MotionVector(dx: Int16(bestQx), dy: Int16(bestQy)), bestSad)
            }
        }
    }
}
