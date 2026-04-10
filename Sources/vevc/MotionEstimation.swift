import Foundation

struct MotionVector: Sendable {
    let dx: Int16
    let dy: Int16
    
    init(dx: Int16, dy: Int16) {
        self.dx = dx
        self.dy = dy
    }
}

struct MotionEstimation {

    // small blocks benefit more from compiler auto-vectorization than manual SIMD
    @inline(__always)
    static func absDiff(_ a: Int16, _ b: Int16) -> Int {
        return abs(Int(a) - Int(b))
    }

    // why: Lagrange multiplier for rate-distortion optimization:
    // penalizes motion vectors with large magnitude to favor zero-MV
    @inline(__always)
    static func getPenalty(dx: Int, dy: Int, lambda: Int) -> Int {
        let absX = dx < 0 ? (-1 * dx) : dx
        let absY = dy < 0 ? (-1 * dy) : dy
        return (absX + absY) * lambda
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
    static func fetchHalfPixelBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, intX: Int, intY: Int, fractX: Int, fractY: Int, dest: UnsafeMutablePointer<Int16>) {
        if fractX == 0 && fractY == 0 {
            fetchPixelsBlock8(plane: plane, width: width, height: height, x: intX, y: intY, dest: dest)
            return
        }
        
        if intX >= 0 && intY >= 0 && intX + 8 + fractX <= width && intY + 8 + fractY <= height {
            if fractY == 0 {
                for ry in 0..<8 {
                    let row = plane.advanced(by: (intY + ry) * width + intX)
                    let dst = dest.advanced(by: ry * 8)
                    for rx in 0..<8 { dst[rx] = Int16((Int(row[rx]) + Int(row[rx + 1]) + 1) >> 1) }
                }
            } else if fractX == 0 {
                for ry in 0..<8 {
                    let row0 = plane.advanced(by: (intY + ry) * width + intX)
                    let row1 = plane.advanced(by: (intY + ry + 1) * width + intX)
                    let dst = dest.advanced(by: ry * 8)
                    for rx in 0..<8 { dst[rx] = Int16((Int(row0[rx]) + Int(row1[rx]) + 1) >> 1) }
                }
            } else {
                for ry in 0..<8 {
                    let row0 = plane.advanced(by: (intY + ry) * width + intX)
                    let row1 = plane.advanced(by: (intY + ry + 1) * width + intX)
                    let dst = dest.advanced(by: ry * 8)
                    for rx in 0..<8 { dst[rx] = Int16((Int(row0[rx]) + Int(row0[rx+1]) + Int(row1[rx]) + Int(row1[rx+1]) + 2) >> 2) }
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
                    dstPtr[rx] = Int16((Int(row0[sx0]) + Int(row0[sx1]) + 1) >> 1)
                } else if fractX == 0 {
                    dstPtr[rx] = Int16((Int(row0[sx0]) + Int(row1[sx0]) + 1) >> 1)
                } else {
                    dstPtr[rx] = Int16((Int(row0[sx0]) + Int(row0[sx1]) + Int(row1[sx0]) + Int(row1[sx1]) + 2) >> 2)
                }
            }
        }
    }

    @inline(__always)
    static func fetchQuarterPixelBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, intX: Int, intY: Int, remX: Int, remY: Int, dest: UnsafeMutablePointer<Int16>) {
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
                    dst[rx] = Int16((v + 8) >> 4)
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
                dst[rx] = Int16((v + 8) >> 4)
            }
        }
    }

    @inline(__always)
    static func fetchEighthPixelBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, intX: Int, intY: Int, remX: Int, remY: Int, dest: UnsafeMutablePointer<Int16>) {
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
                    dst[rx] = Int16((v + 32) >> 6)
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
                dst[rx] = Int16((v + 32) >> 6)
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
    private static func evaluateSearch(
        cPtr: UnsafePointer<Int16>, 
        pBase: UnsafePointer<Int16>, 
        oPtr: UnsafeMutablePointer<Int16>,
        tPtr: UnsafeMutablePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int
    ) -> (Int, Int, Int) {
        fetchPixelsBlock8(plane: pBase, width: width, height: height, x: bx, y: by, dest: oPtr)
        let zeroSad: Int = compute64PointSAD_Blocks(cBase: cPtr, pBase: oPtr)
        
        if zeroSad < 64 {
            return (0, 0, zeroSad)
        }
        
        var bestCoarseSad: Int = zeroSad
        var bestCoarseDx: Int = 0
        var bestCoarseDy: Int = 0
        
        let coarseOffsets: [(Int, Int)] = [
            (-2, -2), (0, -2), (2, -2),
            (-2,  0),          (2,  0),
            (-2,  2), (0,  2), (2,  2)
        ]
        
        for offset in coarseOffsets {
            let dx: Int = offset.0
            let dy: Int = offset.1
            
            fetchPixelsBlock8(plane: pBase, width: width, height: height, x: bx + dx, y: by + dy, dest: tPtr)
            let sad: Int = compute64PointSAD_Blocks(cBase: cPtr, pBase: tPtr)
            
            let penalty: Int = getPenalty(dx: dx, dy: dy, lambda: 40)
            let totalSad: Int = sad + penalty
            
            if totalSad < bestCoarseSad {
                bestCoarseSad = totalSad
                bestCoarseDx = dx
                bestCoarseDy = dy
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
            
            fetchPixelsBlock8(plane: pBase, width: width, height: height, x: bx + fineDx, y: by + fineDy, dest: tPtr)
            let sad: Int = compute64PointSAD_Blocks(cBase: cPtr, pBase: tPtr)
            
            let penalty: Int = getPenalty(dx: fineDx, dy: fineDy, lambda: 40)
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
                
                fetchHalfPixelBlock8(plane: pBase, width: width, height: height,
                                     intX: bx + intDx, intY: by + intDy,
                                     fractX: fractX, fractY: fractY, dest: tPtr)
                let sad: Int = compute64PointSAD_Blocks(cBase: cPtr, pBase: tPtr)
                let penalty: Int = getPenalty(dx: hpDx, dy: hpDy, lambda: 20)
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
                
                fetchEighthPixelBlock8(plane: pBase, width: width, height: height,
                                       intX: bx + intDx, intY: by + intDy,
                                       remX: remX, remY: remY, dest: tPtr)
                let sad: Int = compute64PointSAD_Blocks(cBase: cPtr, pBase: tPtr)
                let penalty: Int = getPenalty(dx: epDx, dy: epDy, lambda: 5)
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
        width: Int, height: Int, bx: Int, by: Int, range: Int = 4
    ) -> (MotionVector, Int) {
        return currPlane.withUnsafeBufferPointer { cBuf in
            prevPlane.withUnsafeBufferPointer { pBuf in
                guard let cBase = cBuf.baseAddress, let pBase = pBuf.baseAddress else {
                    return (MotionVector(dx: 0, dy: 0), 0)
                }

                fetchPixelsBlock8(plane: cBase, width: width, height: height, x: bx, y: by, dest: cPtr)
                let (dx, dy, sad) = evaluateSearch(cPtr: cPtr, pBase: pBase, oPtr: oPtr, tPtr: tPtr, width: width, height: height, bx: bx, by: by)
                return (MotionVector(dx: Int16(dx), dy: Int16(dy)), sad)
            }
        }
    }
}
