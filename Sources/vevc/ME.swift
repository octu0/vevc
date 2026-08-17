// MARK: - MotionVector

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

/// SoA (Structure of Arrays) layout for motion vectors.
/// dx and dy are stored in separate contiguous arrays for cache-friendly access.
struct MotionVectors: Sendable {
    var dx: [Int16]
    var dy: [Int16]
    
    var count: Int { dx.count }
    var isEmpty: Bool { dx.isEmpty }
    
    init(dx: [Int16], dy: [Int16]) {
        self.dx = dx
        self.dy = dy
    }
    
    init(count: Int) {
        self.dx = [Int16](repeating: 0, count: count)
        self.dy = [Int16](repeating: 0, count: count)
    }
    
    static let empty = MotionVectors(dx: [], dy: [])
}

private let meFineOffsets: [(Int, Int)] = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1)
]

private let meSearchOffsetX: [Int] = [-1, 0, 1, -1, 1, -1, 0, 1]
private let meSearchOffsetY: [Int] = [-1, -1, -1, 0, 0, 1, 1, 1]

// MARK: - MotionEstimation

struct MotionEstimation {

    @inline(__always)
    fileprivate static func median(_ a: Int, _ b: Int, _ c: Int) -> Int {
        return max(min(a, b), min(max(a, b), c))
    }

    // Lagrange multiplier for rate-distortion optimization:
    // penalizes Motion Vector Difference (MVD) to favor pmv
    @inline(__always)
    static func getMVDPenalty(dx: Int, dy: Int, pmvDx: Int, pmvDy: Int) -> Int {
        return Int((dx - pmvDx).magnitude) + Int((dy - pmvDy).magnitude)
    }

    @inline(__always)
    static func fetchPixelsBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, x: Int, y: Int, dest: UnsafeMutablePointer<Int16>) {
        if 0 <= x && 0 <= y && x + 8 <= width && y + 8 <= height {
            for ry in 0..<8 {
                let offset = (y + ry) * width + x
                let sPtr = plane.advanced(by: offset)
                let dPtr = dest.advanced(by: ry * 8)
                dPtr[0] = sPtr[0]; dPtr[1] = sPtr[1]; dPtr[2] = sPtr[2]; dPtr[3] = sPtr[3]
                dPtr[4] = sPtr[4]; dPtr[5] = sPtr[5]; dPtr[6] = sPtr[6]; dPtr[7] = sPtr[7]
            }
        } else {
            for i in 0..<64 {
                let ry = i >> 3
                let rx = i & 7
                let srcY = max(0, min(y + ry, height - 1))
                let srcX = max(0, min(x + rx, width - 1))
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
        
        if 0 <= intX && 0 <= intY && ((intX + 8) + fractX) <= width && ((intY + 8) + fractY) <= height {
            if fractY == 0 {
                for ry in 0..<8 {
                    let row = plane.advanced(by: (intY + ry) * width + intX)
                    let dst = dest.advanced(by: ry * 8)
                    for rx in 0..<8 {
                        dst[rx] = Int16((Int(row[rx]) + Int(row[rx + 1]) + roundOffset) >> 1)
                    }
                }
                return
            }
            if fractX == 0 {
                for ry in 0..<8 {
                    let row0 = plane.advanced(by: (intY + ry) * width + intX)
                    let row1 = plane.advanced(by: (intY + ry + 1) * width + intX)
                    let dst = dest.advanced(by: ry * 8)
                    for rx in 0..<8 {
                        dst[rx] = Int16((Int(row0[rx]) + Int(row1[rx]) + roundOffset) >> 1)
                    }
                }
                return
            }
            
            for ry in 0..<8 {
                let row0 = plane.advanced(by: (intY + ry) * width + intX)
                let row1 = plane.advanced(by: (intY + ry + 1) * width + intX)
                let dst = dest.advanced(by: ry * 8)
                for rx in 0..<8 {
                    dst[rx] = Int16((Int(row0[rx]) + Int(row0[rx+1]) + Int(row1[rx]) + Int(row1[rx+1]) + 1 + roundOffset) >> 2)
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
                switch true {
                case fractY == 0:
                    dstPtr[rx] = Int16((Int(row0[sx0]) + Int(row0[sx1]) + roundOffset) >> 1)
                case fractX == 0:
                    dstPtr[rx] = Int16((Int(row0[sx0]) + Int(row1[sx0]) + roundOffset) >> 1)
                default:
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
        let nextX = if remX == 0 { 0 } else { 1 }
        let nextY = if remY == 0 { 0 } else { 1 }
        let wA = 4 - remX
        let wB = remX
        let wC = 4 - remY
        let wD = remY
        
        if 0 <= intX && 0 <= intY && ((intX + 8) + nextX) <= width && ((intY + 8) + nextY) <= height {
            for ry in 0..<8 {
                let row0 = plane.advanced(by: (intY + ry) * width + intX)
                let row1 = plane.advanced(by: (intY + ry + nextY) * width + intX)
                let dst = dest.advanced(by: ry * 8)
                for rx in 0..<8 {
                    let v0 = (wA * wC) * Int(row0[rx])
                    let v1 = (wB * wC) * Int(row0[rx + nextX])
                    let v2 = (wA * wD) * Int(row1[rx])
                    let v3 = (wB * wD) * Int(row1[rx + nextX])
                    let v = (v0 + v1) + (v2 + v3)
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
                let v0 = (wA * wC) * Int(row0[sx0])
                let v1 = (wB * wC) * Int(row0[sx1])
                let v2 = (wA * wD) * Int(row1[sx0])
                let v3 = (wB * wD) * Int(row1[sx1])
                let v = (v0 + v1) + (v2 + v3)
                dst[rx] = Int16((v + 7 + roundOffset) >> 4)
            }
        }
    }

    @inline(__always)
    static func compute64PointSADBlocksWithStride(cBase: UnsafePointer<Int16>, pBase: UnsafePointer<Int16>, pStride: Int) -> Int {
        var sad: Int32 = 0
        for ry in 0..<8 {
            let cRow = cBase.advanced(by: ry * 8)
            let pRow = pBase.advanced(by: ry * pStride)
            
            sad &+= Int32((Int32(cRow[0]) - Int32(pRow[0])).magnitude)
            sad &+= Int32((Int32(cRow[1]) - Int32(pRow[1])).magnitude)
            sad &+= Int32((Int32(cRow[2]) - Int32(pRow[2])).magnitude)
            sad &+= Int32((Int32(cRow[3]) - Int32(pRow[3])).magnitude)
            sad &+= Int32((Int32(cRow[4]) - Int32(pRow[4])).magnitude)
            sad &+= Int32((Int32(cRow[5]) - Int32(pRow[5])).magnitude)
            sad &+= Int32((Int32(cRow[6]) - Int32(pRow[6])).magnitude)
            sad &+= Int32((Int32(cRow[7]) - Int32(pRow[7])).magnitude)
        }
        return Int(sad)
    }

    @inline(__always)
    static func compute64PointSADBlocks(cBase: UnsafePointer<Int16>, pBase: UnsafePointer<Int16>) -> Int {
        var sad: Int32 = 0
        sad &+= Int32((Int32(cBase[0]) - Int32(pBase[0])).magnitude)
        sad &+= Int32((Int32(cBase[1]) - Int32(pBase[1])).magnitude)
        sad &+= Int32((Int32(cBase[2]) - Int32(pBase[2])).magnitude)
        sad &+= Int32((Int32(cBase[3]) - Int32(pBase[3])).magnitude)
        sad &+= Int32((Int32(cBase[4]) - Int32(pBase[4])).magnitude)
        sad &+= Int32((Int32(cBase[5]) - Int32(pBase[5])).magnitude)
        sad &+= Int32((Int32(cBase[6]) - Int32(pBase[6])).magnitude)
        sad &+= Int32((Int32(cBase[7]) - Int32(pBase[7])).magnitude)
        sad &+= Int32((Int32(cBase[8]) - Int32(pBase[8])).magnitude)
        sad &+= Int32((Int32(cBase[9]) - Int32(pBase[9])).magnitude)
        sad &+= Int32((Int32(cBase[10]) - Int32(pBase[10])).magnitude)
        sad &+= Int32((Int32(cBase[11]) - Int32(pBase[11])).magnitude)
        sad &+= Int32((Int32(cBase[12]) - Int32(pBase[12])).magnitude)
        sad &+= Int32((Int32(cBase[13]) - Int32(pBase[13])).magnitude)
        sad &+= Int32((Int32(cBase[14]) - Int32(pBase[14])).magnitude)
        sad &+= Int32((Int32(cBase[15]) - Int32(pBase[15])).magnitude)
        sad &+= Int32((Int32(cBase[16]) - Int32(pBase[16])).magnitude)
        sad &+= Int32((Int32(cBase[17]) - Int32(pBase[17])).magnitude)
        sad &+= Int32((Int32(cBase[18]) - Int32(pBase[18])).magnitude)
        sad &+= Int32((Int32(cBase[19]) - Int32(pBase[19])).magnitude)
        sad &+= Int32((Int32(cBase[20]) - Int32(pBase[20])).magnitude)
        sad &+= Int32((Int32(cBase[21]) - Int32(pBase[21])).magnitude)
        sad &+= Int32((Int32(cBase[22]) - Int32(pBase[22])).magnitude)
        sad &+= Int32((Int32(cBase[23]) - Int32(pBase[23])).magnitude)
        sad &+= Int32((Int32(cBase[24]) - Int32(pBase[24])).magnitude)
        sad &+= Int32((Int32(cBase[25]) - Int32(pBase[25])).magnitude)
        sad &+= Int32((Int32(cBase[26]) - Int32(pBase[26])).magnitude)
        sad &+= Int32((Int32(cBase[27]) - Int32(pBase[27])).magnitude)
        sad &+= Int32((Int32(cBase[28]) - Int32(pBase[28])).magnitude)
        sad &+= Int32((Int32(cBase[29]) - Int32(pBase[29])).magnitude)
        sad &+= Int32((Int32(cBase[30]) - Int32(pBase[30])).magnitude)
        sad &+= Int32((Int32(cBase[31]) - Int32(pBase[31])).magnitude)
        sad &+= Int32((Int32(cBase[32]) - Int32(pBase[32])).magnitude)
        sad &+= Int32((Int32(cBase[33]) - Int32(pBase[33])).magnitude)
        sad &+= Int32((Int32(cBase[34]) - Int32(pBase[34])).magnitude)
        sad &+= Int32((Int32(cBase[35]) - Int32(pBase[35])).magnitude)
        sad &+= Int32((Int32(cBase[36]) - Int32(pBase[36])).magnitude)
        sad &+= Int32((Int32(cBase[37]) - Int32(pBase[37])).magnitude)
        sad &+= Int32((Int32(cBase[38]) - Int32(pBase[38])).magnitude)
        sad &+= Int32((Int32(cBase[39]) - Int32(pBase[39])).magnitude)
        sad &+= Int32((Int32(cBase[40]) - Int32(pBase[40])).magnitude)
        sad &+= Int32((Int32(cBase[41]) - Int32(pBase[41])).magnitude)
        sad &+= Int32((Int32(cBase[42]) - Int32(pBase[42])).magnitude)
        sad &+= Int32((Int32(cBase[43]) - Int32(pBase[43])).magnitude)
        sad &+= Int32((Int32(cBase[44]) - Int32(pBase[44])).magnitude)
        sad &+= Int32((Int32(cBase[45]) - Int32(pBase[45])).magnitude)
        sad &+= Int32((Int32(cBase[46]) - Int32(pBase[46])).magnitude)
        sad &+= Int32((Int32(cBase[47]) - Int32(pBase[47])).magnitude)
        sad &+= Int32((Int32(cBase[48]) - Int32(pBase[48])).magnitude)
        sad &+= Int32((Int32(cBase[49]) - Int32(pBase[49])).magnitude)
        sad &+= Int32((Int32(cBase[50]) - Int32(pBase[50])).magnitude)
        sad &+= Int32((Int32(cBase[51]) - Int32(pBase[51])).magnitude)
        sad &+= Int32((Int32(cBase[52]) - Int32(pBase[52])).magnitude)
        sad &+= Int32((Int32(cBase[53]) - Int32(pBase[53])).magnitude)
        sad &+= Int32((Int32(cBase[54]) - Int32(pBase[54])).magnitude)
        sad &+= Int32((Int32(cBase[55]) - Int32(pBase[55])).magnitude)
        sad &+= Int32((Int32(cBase[56]) - Int32(pBase[56])).magnitude)
        sad &+= Int32((Int32(cBase[57]) - Int32(pBase[57])).magnitude)
        sad &+= Int32((Int32(cBase[58]) - Int32(pBase[58])).magnitude)
        sad &+= Int32((Int32(cBase[59]) - Int32(pBase[59])).magnitude)
        sad &+= Int32((Int32(cBase[60]) - Int32(pBase[60])).magnitude)
        sad &+= Int32((Int32(cBase[61]) - Int32(pBase[61])).magnitude)
        sad &+= Int32((Int32(cBase[62]) - Int32(pBase[62])).magnitude)
        sad &+= Int32((Int32(cBase[63]) - Int32(pBase[63])).magnitude)
        return Int(sad)
    }

    private static let dsLdspX: [Int] = [0, 1, 2, 1, 0, -1, -2, -1]
    private static let dsLdspY: [Int] = [-2, -1, 0, 1, 2, 1, 0, -1]
    private static let dsSdspX: [Int] = [0, 1, 0, -1]
    private static let dsSdspY: [Int] = [-1, 0, 1, 0]

    @inline(__always)
    private static func evaluateSearch(
        cPtr: UnsafePointer<Int16>, 
        pBase: UnsafePointer<Int16>, 
        oPtr: UnsafeMutablePointer<Int16>,
        tPtr: UnsafeMutablePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int, range: Int, pmv: MotionVector, roundOffset: Int
    ) -> (Int, Int, Int) {
        fetchPixelsBlock8(plane: pBase, width: width, height: height, x: bx, y: by, dest: oPtr)
        let zeroSAD: Int = compute64PointSADBlocks(cBase: cPtr, pBase: oPtr)
        
        if zeroSAD < 32 {
            return (0, 0, zeroSAD)
        }
        
        let pmvDx4 = Int(pmv.dx) / 4
        let pmvDy4 = Int(pmv.dy) / 4
        let zeroPenalty = getMVDPenalty(dx: 0, dy: 0, pmvDx: pmvDx4, pmvDy: pmvDy4)
        var bestCoarseSAD = zeroSAD + zeroPenalty
        var bestCoarseDx = 0
        var bestCoarseDy = 0
        
        let minDy = max(-1 * range, -1 * by)
        let maxDy = min(range, height - by - 8)
        let minDx = max(-1 * range, -1 * bx)
        let maxDx = min(range, width - bx - 8)
        
        if minDy <= maxDy && minDx <= maxDx {
            var centerX = Int(pmv.dx) / 4
            var centerY = Int(pmv.dy) / 4
            centerX = max(minDx, min(maxDx, centerX))
            centerY = max(minDy, min(maxDy, centerY))
            
            let pPtr = pBase.advanced(by: (by + centerY) * width + (bx + centerX))
            let pmvSAD = compute64PointSADBlocksWithStride(cBase: cPtr, pBase: pPtr, pStride: width)
            let pmvPenalty = getMVDPenalty(dx: centerX, dy: centerY, pmvDx: pmvDx4, pmvDy: pmvDy4)
            
            if pmvSAD + pmvPenalty < bestCoarseSAD {
                bestCoarseSAD = pmvSAD + pmvPenalty
                bestCoarseDx = centerX
                bestCoarseDy = centerY
            } else {
                centerX = 0
                centerY = 0
            }
            
            // Limit LDSP iterations to 2 to avoid excessive search in difficult blocks
            var ldspIter = 0
            while ldspIter < 2 {
                ldspIter += 1
                var minSAD = bestCoarseSAD
                var minDxPos = centerX
                var minDyPos = centerY
                var foundSmaller = false
                
                for i in 0..<8 {
                    let dx = centerX + dsLdspX[i]
                    let dy = centerY + dsLdspY[i]
                    
                    if dx < minDx { continue }
                    if maxDx < dx { continue }
                    if dy < minDy { continue }
                    if maxDy < dy { continue }
                    
                    let penalty = getMVDPenalty(dx: dx, dy: dy, pmvDx: pmvDx4, pmvDy: pmvDy4)
                    let maxSAD = bestCoarseSAD - penalty
                    if maxSAD < 0 { continue }
                    
                    let pPtr = pBase.advanced(by: (by + dy) * width + (bx + dx))
                    let sad = compute64PointSADBlocksWithStride(cBase: cPtr, pBase: pPtr, pStride: width)
                    
                    let totalSAD = sad + penalty
                    if totalSAD < minSAD {
                        minSAD = totalSAD
                        minDxPos = dx
                        minDyPos = dy
                        foundSmaller = true
                    }
                }
                
                if foundSmaller != true {
                    break
                }
                
                bestCoarseSAD = minSAD
                bestCoarseDx = minDxPos
                bestCoarseDy = minDyPos
                centerX = minDxPos
                centerY = minDyPos
                
                if bestCoarseSAD < 64 {
                    break
                }
            }
            
            var finalMinSAD = bestCoarseSAD
            var finalMinDx = centerX
            var finalMinDy = centerY
            
            if 64 <= finalMinSAD {
                for i in 0..<4 {
                    let dx = centerX + dsSdspX[i]
                    let dy = centerY + dsSdspY[i]
                    
                    if dx < minDx { continue }
                    if maxDx < dx { continue }
                    if dy < minDy { continue }
                    if maxDy < dy { continue }
                    
                    let penalty = getMVDPenalty(dx: dx, dy: dy, pmvDx: pmvDx4, pmvDy: pmvDy4)
                    let maxSAD = bestCoarseSAD - penalty
                    if maxSAD < 0 { continue }
                    
                    let pPtr = pBase.advanced(by: (by + dy) * width + (bx + dx))
                    let sad = compute64PointSADBlocksWithStride(cBase: cPtr, pBase: pPtr, pStride: width)
                    
                    let totalSAD = sad + penalty
                    if totalSAD < finalMinSAD {
                        finalMinSAD = totalSAD
                        finalMinDx = dx
                        finalMinDy = dy
                    }
                }
            }
            
            bestCoarseSAD = finalMinSAD
            bestCoarseDx = finalMinDx
            bestCoarseDy = finalMinDy
        }
        
        var bestFineSAD: Int = bestCoarseSAD
        var bestFineDx: Int = bestCoarseDx
        var bestFineDy: Int = bestCoarseDy
        
        // Early exit: skip fine search when coarse SAD is already good enough
        if 128 <= bestCoarseSAD {
        let fineOffsets = meFineOffsets
        
        for offset in fineOffsets {
            let fx: Int = offset.0
            let fy: Int = offset.1
            let fineDx: Int = bestCoarseDx + fx
            let fineDy: Int = bestCoarseDy + fy
            
            if fineDx < -4 || 4 < fineDx || fineDy < -4 || 4 < fineDy { continue }
            
            let penalty = getMVDPenalty(dx: fineDx, dy: fineDy, pmvDx: pmvDx4, pmvDy: pmvDy4)
            let maxSAD = bestFineSAD - penalty
            if maxSAD < 0 { continue }
            
            fetchPixelsBlock8(plane: pBase, width: width, height: height, x: bx + fineDx, y: by + fineDy, dest: tPtr)
            let sad = compute64PointSADBlocks(cBase: cPtr, pBase: tPtr)
            
            let totalSAD = sad + penalty
            if totalSAD < bestFineSAD {
                bestFineSAD = totalSAD
                bestFineDx = fineDx
                bestFineDy = fineDy
            }
        }
        } // end early exit guard
        
        var bestHpDx: Int = bestFineDx * 2
        var bestHpDy: Int = bestFineDy * 2
        var bestHpSAD: Int = bestFineSAD
        
        // Half-pixel refinement: threshold lowered 224→160 for more aggressive early exit
        if 160 < bestFineSAD {
            for oi in 0..<8 {
                let hx = meSearchOffsetX[oi]
                let hy = meSearchOffsetY[oi]
                let hpDx: Int = bestFineDx * 2 + hx
                let hpDy: Int = bestFineDy * 2 + hy
                
                let intDx: Int = hpDx >> 1
                let intDy: Int = hpDy >> 1
                let fractX: Int = hpDx & 1
                let fractY: Int = hpDy & 1
                
                let blurPenalty = (fractX + fractY) * 16
                let penalty = getMVDPenalty(dx: hpDx, dy: hpDy, pmvDx: Int(pmv.dx) / 2, pmvDy: Int(pmv.dy) / 2) + blurPenalty
                let maxSAD = bestHpSAD - penalty
                if maxSAD < 0 { continue }
                
                fetchHalfPixelBlock8(
                    plane: pBase, width: width, height: height,
                    intX: bx + intDx, intY: by + intDy,
                    fractX: fractX, fractY: fractY, dest: tPtr, roundOffset: roundOffset
                )
                let sad = compute64PointSADBlocks(cBase: cPtr, pBase: tPtr)
                
                let totalSAD = sad + penalty
                if totalSAD < bestHpSAD {
                    bestHpSAD = totalSAD
                    bestHpDx = hpDx
                    bestHpDy = hpDy
                }
            }
        }
        
        // Quarter-pixel refinement around best half-pixel position
        // bestHpDx is in half-pixel units -> multiply by 2 to convert to quarter-pixel units
        // This search yields odd-valued MVs, ensuring non-zero fractX/Y values even after Layer2 scaling,
        // which makes FIR interpolation effective.
        var bestQpDx: Int = bestHpDx * 2
        var bestQpDy: Int = bestHpDy * 2
        var bestQpSAD: Int = bestHpSAD
        
        // Quarter-pixel refinement: threshold lowered 96→64 for more aggressive early exit
        if 64 < bestHpSAD {
            for oi in 0..<8 {
                let qpDx: Int = bestHpDx * 2 + meSearchOffsetX[oi]
                let qpDy: Int = bestHpDy * 2 + meSearchOffsetY[oi]
                
                let intDx: Int = qpDx >> 2
                let intDy: Int = qpDy >> 2
                let remX: Int = qpDx & 3
                let remY: Int = qpDy & 3
                
                // Quarter-pixel blur penalty is lighter than half-pixel
                let blurPenalty = (remX + remY) * 4
                let penalty = getMVDPenalty(dx: qpDx, dy: qpDy, pmvDx: Int(pmv.dx), pmvDy: Int(pmv.dy)) + blurPenalty
                let maxSAD = bestQpSAD - penalty
                if maxSAD < 0 { continue }
                
                fetchQuarterPixelBlock8(
                    plane: pBase, width: width, height: height,
                    intX: bx + intDx, intY: by + intDy,
                    remX: remX, remY: remY, dest: tPtr, roundOffset: roundOffset
                )
                let sad = compute64PointSADBlocks(cBase: cPtr, pBase: tPtr)
                
                let totalSAD = sad + penalty
                if totalSAD < bestQpSAD {
                    bestQpSAD = totalSAD
                    bestQpDx = qpDx
                    bestQpDy = qpDy
                }
            }
        }
        
        let bestMVIsZero = (bestQpDx == 0 && bestQpDy == 0)
        if bestMVIsZero != true {
            if zeroSAD < 32 && zeroSAD < bestQpSAD + 2 {
                return (0, 0, zeroSAD)
            }
        }
        
        return (bestQpDx, bestQpDy, bestQpSAD)
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
        return withUnsafePointers(currPlane, prevPlane) { cBase, pBase in
            fetchPixelsBlock8(plane: cBase, width: width, height: height, x: bx, y: by, dest: cPtr)
            return searchPixels(cBase: cBase, pBase: pBase, cPtr: cPtr, oPtr: oPtr, tPtr: tPtr, width: width, height: height, bx: bx, by: by, range: range, pmv: pmv, roundOffset: roundOffset)
        }
    }
    
    @inline(__always)
    static func searchPixels(
        cBase: UnsafePointer<Int16>, 
        pBase: UnsafePointer<Int16>, 
        cPtr: UnsafeMutablePointer<Int16>,
        oPtr: UnsafeMutablePointer<Int16>,
        tPtr: UnsafeMutablePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int, range: Int = 4, pmv: MotionVector, roundOffset: Int
    ) -> (MotionVector, Int) {
        let (dx, dy, sad) = evaluateSearch(cPtr: cPtr, pBase: pBase, oPtr: oPtr, tPtr: tPtr, width: width, height: height, bx: bx, by: by, range: range, pmv: pmv, roundOffset: roundOffset)
        return (MotionVector(dx: Int16(dx), dy: Int16(dy)), sad)
    }

    /// Extract approximate structure contrast (max - min) from 8x8 block
    /// Zero-cost feature extraction without additional SIMD loop overheads.
    @inline(__always)
    static func extractContrast8x8(plane: [Int16], width: Int, height: Int, bx: Int, by: Int) -> Int {
        return plane.withUnsafeBufferPointer { base in
            extractContrast8x8(base: base.baseAddress!, width: width, height: height, bx: bx, by: by)
        }
    }
    
    @inline(__always)
    static func extractContrast8x8(base: UnsafePointer<Int16>, width: Int, height: Int, bx: Int, by: Int) -> Int {
        var minVal: Int32 = 32767
        var maxVal: Int32 = -32768
        
        let isSafeX = (0 <= bx) && (bx + 8 <= width)
        let isSafeY = (0 <= by) && (by + 8 <= height)
        if isSafeX && isSafeY {
            for y in 0..<8 {
                let row = base.advanced(by: (by + y) * width + bx)
                for x in 0..<8 {
                    let val = Int32(row[x])
                    minVal = min(minVal, val)
                    maxVal = max(maxVal, val)
                }
            }
            return Int(maxVal - minVal)
        }

        for y in 0..<8 {
            let sy = max(0, min(by + y, height - 1))
            let row = base.advanced(by: sy * width)
            for x in 0..<8 {
                let sx = max(0, min(bx + x, width - 1))
                let val = Int32(row[sx])
                minVal = min(minVal, val)
                maxVal = max(maxVal, val)
            }
        }
        
        return Int(maxVal - minVal)
    }

    @inline(__always)
    static func computeChromaSAD(
        currCb: UnsafePointer<Int16>, currCr: UnsafePointer<Int16>,
        refCb: UnsafePointer<Int16>, refCr: UnsafePointer<Int16>,
        cbw: Int, cbh: Int,
        bx: Int, by: Int, refDx: Int, refDy: Int
    ) -> Int {
        // bx/by are quarter-resolution block coordinates; the chroma planes
        // are half resolution, so the block's chroma region is 16×16 at
        // (2bx, 2by). refDx/refDy are the full-resolution-pixel motion vector
        // (ChromaSADCoordinateTests pins this contract), so the chroma
        // displacement is refDx>>1. Callers must NOT pre-scale the MV — the
        // previous call sites passed mv*2, sampling the reference at DOUBLE
        // the true motion displacement (correct only for zero MVs, wrong for
        // exactly the moving blocks the penalty exists for).
        let cx = bx << 1
        let cy = by << 1
        let crx = cx + (refDx >> 1)
        let cry = cy + (refDy >> 1)

        // 4×4 samples strided by 4 across the whole 16×16 chroma region (the
        // corner-only version read the top-left 4×4 — 1/16 coverage, which
        // let chroma mismatches in the remaining area go unseen).
        let isCurrSafe = (0 <= cx) && (0 <= cy) && (cx + 13 <= cbw) && (cy + 13 <= cbh)
        let isRefSafe = (0 <= crx) && (0 <= cry) && (crx + 13 <= cbw) && (cry + 13 <= cbh)

        if isCurrSafe && isRefSafe {
            var sad: Int32 = 0
            for y in 0..<4 {
                let currOffset = (cy + y * 4) * cbw + cx
                let refOffset = (cry + y * 4) * cbw + crx

                for x in 0..<4 {
                    sad &+= Int32((Int32(currCb[currOffset + x * 4]) - Int32(refCb[refOffset + x * 4])).magnitude)
                    sad &+= Int32((Int32(currCr[currOffset + x * 4]) - Int32(refCr[refOffset + x * 4])).magnitude)
                }
            }
            return Int(sad) * 4 // Luma SAD scale matching (16 sample positions × 2 planes vs 64 luma points)
        }
        return 1000
    }
    
    @inline(__always)
    static func computeChromaSAD(
        curr: PlaneData420, ref: PlaneData420,
        bx: Int, by: Int, refDx: Int, refDy: Int
    ) -> Int {
        let cbw = (curr.width + 1) / 2
        let cbh = (curr.height + 1) / 2
        return withUnsafePointers(curr.cb, curr.cr, ref.cb, ref.cr) { cCb, cCr, rCb, rCr in
            computeChromaSAD(currCb: cCb, currCr: cCr, refCb: rCb, refCr: rCr, cbw: cbw, cbh: cbh, bx: bx, by: by, refDx: refDx, refDy: refDy)
        }
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32_Safe_NoFIR(
        curr: UnsafePointer<Int16>, prev: UnsafePointer<Int16>,
        width: Int, bx: Int, by: Int, intDx: Int, intDy: Int
    ) -> Int {
        var sad: Int32 = 0
        for ry in stride(from: 0, to: 32, by: 8) {
            let cy = by + ry
            let rowC = curr.advanced(by: cy * width + bx)
            let py = by + intDy + ry
            let r = prev.advanced(by: py * width + bx + intDx)
            for rx in stride(from: 0, to: 32, by: 2) {
                sad &+= Int32((Int32(rowC[rx]) - Int32(r[rx])).magnitude)
            }
        }
        return Int(sad)
    }

@inline(__always)
    static func computeQuarterPixelSADSubsampled32_Safe_FIR_Y0(
        curr: UnsafePointer<Int16>, prev: UnsafePointer<Int16>,
        width: Int, bx: Int, by: Int, intDx: Int, intDy: Int,
        cX0: Int32, cX1: Int32, cX2: Int32, cX3: Int32
    ) -> Int {
        var sad: Int32 = 0
        for ry in stride(from: 0, to: 32, by: 8) {
            let cy = by + ry
            let rowC = curr.advanced(by: cy * width + bx)
            let py = by + intDy + ry
            let r0 = prev.advanced(by: py * width + bx + intDx)
            for rx in stride(from: 0, to: 32, by: 2) {
                let h0 = cX0 &* Int32(r0[rx - 1]) &+ cX1 &* Int32(r0[rx]) &+ cX2 &* Int32(r0[rx + 1]) &+ cX3 &* Int32(r0[rx + 2])
                let pVal = (h0 &+ 3) >> 3
                sad &+= Int32((Int32(rowC[rx]) &- pVal).magnitude)
            }
        }
        return Int(sad)
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32_Safe_FIR_X0(
        curr: UnsafePointer<Int16>, prev: UnsafePointer<Int16>,
        width: Int, bx: Int, by: Int, intDx: Int, intDy: Int,
        cY0: Int32, cY1: Int32, cY2: Int32, cY3: Int32
    ) -> Int {
        var sad: Int32 = 0
        for ry in stride(from: 0, to: 32, by: 8) {
            let cy = by + ry
            let rowC = curr.advanced(by: cy * width + bx)
            let py = by + intDy + ry
            let rM1 = prev.advanced(by: (py - 1) * width + bx + intDx)
            let r0 = prev.advanced(by: py * width + bx + intDx)
            let rP1 = prev.advanced(by: (py + 1) * width + bx + intDx)
            let rP2 = prev.advanced(by: (py + 2) * width + bx + intDx)
            for rx in stride(from: 0, to: 32, by: 2) {
                let vertSum = cY0 &* Int32(rM1[rx]) &+ cY1 &* Int32(r0[rx]) &+ cY2 &* Int32(rP1[rx]) &+ cY3 &* Int32(rP2[rx])
                let pVal = (vertSum &+ 3) >> 3
                sad &+= Int32((Int32(rowC[rx]) &- pVal).magnitude)
            }
        }
        return Int(sad)
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32_Safe_FIR_XY(
        curr: UnsafePointer<Int16>, prev: UnsafePointer<Int16>,
        width: Int, bx: Int, by: Int, intDx: Int, intDy: Int,
        cX0: Int32, cX1: Int32, cX2: Int32, cX3: Int32,
        cY0: Int32, cY1: Int32, cY2: Int32, cY3: Int32
    ) -> Int {
        var sad: Int32 = 0
        for ry in stride(from: 0, to: 32, by: 8) {
            let cy = by + ry
            let rowC = curr.advanced(by: cy * width + bx)
            let py = by + intDy + ry
            let rM1 = prev.advanced(by: (py - 1) * width + bx + intDx)
            let r0 = prev.advanced(by: py * width + bx + intDx)
            let rP1 = prev.advanced(by: (py + 1) * width + bx + intDx)
            let rP2 = prev.advanced(by: (py + 2) * width + bx + intDx)
            for rx in stride(from: 0, to: 32, by: 2) {
                let vM1 = cX0 &* Int32(rM1[rx - 1]) &+ cX1 &* Int32(rM1[rx]) &+ cX2 &* Int32(rM1[rx + 1]) &+ cX3 &* Int32(rM1[rx + 2])
                let v0  = cX0 &* Int32(r0[rx - 1])  &+ cX1 &* Int32(r0[rx])  &+ cX2 &* Int32(r0[rx + 1])  &+ cX3 &* Int32(r0[rx + 2])
                let vP1 = cX0 &* Int32(rP1[rx - 1]) &+ cX1 &* Int32(rP1[rx]) &+ cX2 &* Int32(rP1[rx + 1]) &+ cX3 &* Int32(rP1[rx + 2])
                let vP2 = cX0 &* Int32(rP2[rx - 1]) &+ cX1 &* Int32(rP2[rx]) &+ cX2 &* Int32(rP2[rx + 1]) &+ cX3 &* Int32(rP2[rx + 2])
                
                let refVal = cY0 &* vM1 &+ cY1 &* v0 &+ cY2 &* vP1 &+ cY3 &* vP2
                let pVal = (refVal &+ 31) >> 6
                
                sad &+= Int32((Int32(rowC[rx]) &- pVal).magnitude)
            }
        }
        return Int(sad)
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32_Unsafe_NoFIR(
        curr: UnsafePointer<Int16>, prev: UnsafePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int, intDx: Int, intDy: Int
    ) -> Int {
        var sad: Int32 = 0
        for ry in stride(from: 0, to: 32, by: 8) {
            let cy = min(by + ry, height - 1)
            let rowC = curr.advanced(by: cy * width)
            let py = by + intDy + ry
            let sy0 = max(0, min(py, height - 1))
            let r = prev.advanced(by: sy0 * width)
            for rx in stride(from: 0, to: 32, by: 2) {
                let sx = max(0, min(bx + intDx + rx, width - 1))
                sad &+= Int32((Int32(rowC[min(bx + rx, width - 1)]) - Int32(r[sx])).magnitude)
            }
        }
        return Int(sad)
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32_Unsafe_FIR(
        curr: UnsafePointer<Int16>, prev: UnsafePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int, intDx: Int, intDy: Int,
        cX0: Int32, cX1: Int32, cX2: Int32, cX3: Int32,
        cY0: Int32, cY1: Int32, cY2: Int32, cY3: Int32
    ) -> Int {
        var sad: Int32 = 0
        for ry in stride(from: 0, to: 32, by: 8) {
            let cy = min(by + ry, height - 1)
            let rowC = curr.advanced(by: cy * width)
            let py = by + intDy + ry
            let syM1 = max(0, min(py - 1, height - 1))
            let sy0  = max(0, min(py, height - 1))
            let syP1 = max(0, min(py + 1, height - 1))
            let syP2 = max(0, min(py + 2, height - 1))
            let rM1 = prev.advanced(by: syM1 * width)
            let r0  = prev.advanced(by: sy0 * width)
            let rP1 = prev.advanced(by: syP1 * width)
            let rP2 = prev.advanced(by: syP2 * width)
            for rx in stride(from: 0, to: 32, by: 2) {
                let px = bx &+ intDx &+ rx
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
                
                sad &+= Int32((Int32(rowC[min(bx &+ rx, width - 1)]) &- pVal).magnitude)
            }
        }
        return Int(sad)
    }

    @inline(__always)
    static func computeQuarterPixelSADSubsampled32(
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
        
        let safe = (0 <= ((bx + intDx) - 1)) && (0 <= ((by + intDy) - 1)) && ((((bx + intDx) + 32) + 2) < width) && ((((by + intDy) + 32) + 2) < height) && ((bx + 32) <= width) && ((by + 32) <= height)
        let useFIR = (fractX != 0 || fractY != 0)
        
        let cX0 = Int32(fX[0]), cX1 = Int32(fX[1]), cX2 = Int32(fX[2]), cX3 = Int32(fX[3])
        let cY0 = Int32(fY[0]), cY1 = Int32(fY[1]), cY2 = Int32(fY[2]), cY3 = Int32(fY[3])
        
        if safe {
            if useFIR {
                switch true {
                case fractY == 0:
                    return computeQuarterPixelSADSubsampled32_Safe_FIR_Y0(curr: curr, prev: prev, width: width, bx: bx, by: by, intDx: intDx, intDy: intDy, cX0: cX0, cX1: cX1, cX2: cX2, cX3: cX3)
                case fractX == 0:
                    return computeQuarterPixelSADSubsampled32_Safe_FIR_X0(curr: curr, prev: prev, width: width, bx: bx, by: by, intDx: intDx, intDy: intDy, cY0: cY0, cY1: cY1, cY2: cY2, cY3: cY3)
                default:
                    return computeQuarterPixelSADSubsampled32_Safe_FIR_XY(curr: curr, prev: prev, width: width, bx: bx, by: by, intDx: intDx, intDy: intDy, cX0: cX0, cX1: cX1, cX2: cX2, cX3: cX3, cY0: cY0, cY1: cY1, cY2: cY2, cY3: cY3)
                }
            }
            return computeQuarterPixelSADSubsampled32_Safe_NoFIR(curr: curr, prev: prev, width: width, bx: bx, by: by, intDx: intDx, intDy: intDy)
        }
        
        if useFIR {
            return computeQuarterPixelSADSubsampled32_Unsafe_FIR(curr: curr, prev: prev, width: width, height: height, bx: bx, by: by, intDx: intDx, intDy: intDy, cX0: cX0, cX1: cX1, cX2: cX2, cX3: cX3, cY0: cY0, cY1: cY1, cY2: cY2, cY3: cY3)
        }
        return computeQuarterPixelSADSubsampled32_Unsafe_NoFIR(curr: curr, prev: prev, width: width, height: height, bx: bx, by: by, intDx: intDx, intDy: intDy)
    }

    static let searchOffsets = [
        (0, -1), (0, 1), (-1, 0), (1, 0),
        (-1, -1), (1, -1), (-1, 1), (1, 1)
    ]

    @inline(__always)
    static func searchPixelsSubpixelRefinement32(
        currPlane: [Int16],
        prevPlane: [Int16],
        width: Int, height: Int, bx: Int, by: Int, pmv: MotionVector
    ) -> (MotionVector, Int) {
        return withUnsafePointers(currPlane, prevPlane) { cBase, pBase in
            // pmv is in full units of Luma dx
            // We convert it to 1/4 units of Luma dx by multiplying by 4
            let baseQx = Int(pmv.dx) * 4
            let baseQy = Int(pmv.dy) * 4
            
            var bestSAD = computeQuarterPixelSADSubsampled32(curr: cBase, prev: pBase, width: width, height: height, bx: bx, by: by, qDx: baseQx, qDy: baseQy)
            if bestSAD < 256 { return (MotionVector(dx: Int16(baseQx), dy: Int16(baseQy)), bestSAD) }
            
            // 1. Half-pixel search (step = 2 quarter pixels)
            var hpBestQx = baseQx
            var hpBestQy = baseQy
            for (ox, oy) in searchOffsets {
                let qx = baseQx + ox * 2
                let qy = baseQy + oy * 2
                
                let intDx = qx >> 2
                let intDy = qy >> 2
                if (bx + intDx) < -32 || (width + 32) < (bx + intDx + 32) { continue }
                if (by + intDy) < -32 || (height + 32) < (by + intDy + 32) { continue }
                
                let penalty = (Int((ox).magnitude) + Int((oy).magnitude)) * 6
                let maxSAD = bestSAD - penalty
                if maxSAD <= 0 { continue }
                
                let sad = computeQuarterPixelSADSubsampled32(curr: cBase, prev: pBase, width: width, height: height, bx: bx, by: by, qDx: qx, qDy: qy)
                let totalSAD = sad + penalty
                if totalSAD < bestSAD {
                    bestSAD = totalSAD
                    hpBestQx = qx
                    hpBestQy = qy
                }
            }
            
            if bestSAD < 384 { return (MotionVector(dx: Int16(hpBestQx), dy: Int16(hpBestQy)), bestSAD) }
            
            // 2. Quarter-pixel search (step = 1 quarter pixel)
            var qpBestQx = hpBestQx
            var qpBestQy = hpBestQy
            for (ox, oy) in searchOffsets {
                let qx = hpBestQx + ox
                let qy = hpBestQy + oy
                
                let intDx = qx >> 2
                let intDy = qy >> 2
                if (bx + intDx) < -32 || (width + 32) < (bx + intDx + 32) { continue }
                if (by + intDy) < -32 || (height + 32) < (by + intDy + 32) { continue }
                
                let penalty = (Int((ox).magnitude) + Int((oy).magnitude)) * 4
                let maxSAD = bestSAD - penalty
                if maxSAD <= 0 { continue }
                
                let sad = computeQuarterPixelSADSubsampled32(curr: cBase, prev: pBase, width: width, height: height, bx: bx, by: by, qDx: qx, qDy: qy)
                let totalSAD = sad + penalty
                if totalSAD < bestSAD {
                    bestSAD = totalSAD
                    qpBestQx = qx
                    qpBestQy = qy
                }
            }
            
            return (MotionVector(dx: Int16(qpBestQx), dy: Int16(qpBestQy)), bestSAD)
        }
    }

    @inline(__always)
    static func computeOcclusionScores(
        currPlane: [Int16],
        prevPlane: [Int16],
        width: Int,
        height: Int,
        globalPrior: MotionVector
    ) -> [Int] {
        let colCount = (width + 7) / 8
        let rowCount = (height + 7) / 8
        var scores = [Int](repeating: 0, count: rowCount * colCount)
        let gdy = Int(globalPrior.dy) / 4
        let gdx = Int(globalPrior.dx) / 4
        
        withUnsafePointers(currPlane, prevPlane, mut: &scores) { cBase, pBase, sBase in
            for row in 0..<rowCount {
                let by = row * 8
                let bHeight = min(8, height - by)
                if bHeight <= 0 { continue }
                
                let pBy = min(max(0, by + gdy), height - bHeight)
                for col in 0..<colCount {
                    let bx = col * 8
                    let bWidth = min(8, width - bx)
                    if bWidth <= 0 { continue }
                    
                    var cProfile0: Int32 = 0, cProfile1: Int32 = 0, cProfile2: Int32 = 0, cProfile3: Int32 = 0
                    var cProfile4: Int32 = 0, cProfile5: Int32 = 0, cProfile6: Int32 = 0, cProfile7: Int32 = 0
                    
                    var pProfile0: Int32 = 0, pProfile1: Int32 = 0, pProfile2: Int32 = 0, pProfile3: Int32 = 0
                    var pProfile4: Int32 = 0, pProfile5: Int32 = 0, pProfile6: Int32 = 0, pProfile7: Int32 = 0
                    
                    // apply gdx
                    let px0 = max(0, min(width - 1, bx + 0 + gdx)) - bx
                    let px1 = max(0, min(width - 1, bx + 1 + gdx)) - bx
                    let px2 = max(0, min(width - 1, bx + 2 + gdx)) - bx
                    let px3 = max(0, min(width - 1, bx + 3 + gdx)) - bx
                    let px4 = max(0, min(width - 1, bx + 4 + gdx)) - bx
                    let px5 = max(0, min(width - 1, bx + 5 + gdx)) - bx
                    let px6 = max(0, min(width - 1, bx + 6 + gdx)) - bx
                    let px7 = max(0, min(width - 1, bx + 7 + gdx)) - bx
                    
                    for y in 0..<bHeight {
                        let cOffset = (by + y) * width + bx
                        let cRow = cBase.advanced(by: cOffset)
                        if 0 < bWidth { cProfile0 &+= Int32(cRow[0]) }
                        if 1 < bWidth { cProfile1 &+= Int32(cRow[1]) }
                        if 2 < bWidth { cProfile2 &+= Int32(cRow[2]) }
                        if 3 < bWidth { cProfile3 &+= Int32(cRow[3]) }
                        if 4 < bWidth { cProfile4 &+= Int32(cRow[4]) }
                        if 5 < bWidth { cProfile5 &+= Int32(cRow[5]) }
                        if 6 < bWidth { cProfile6 &+= Int32(cRow[6]) }
                        if 7 < bWidth { cProfile7 &+= Int32(cRow[7]) }
                        
                        let pOffset = (pBy + y) * width + bx
                        let pRow = pBase.advanced(by: pOffset)
                        
                        if 0 < bWidth { pProfile0 &+= Int32(pRow[px0]) }
                        if 1 < bWidth { pProfile1 &+= Int32(pRow[px1]) }
                        if 2 < bWidth { pProfile2 &+= Int32(pRow[px2]) }
                        if 3 < bWidth { pProfile3 &+= Int32(pRow[px3]) }
                        if 4 < bWidth { pProfile4 &+= Int32(pRow[px4]) }
                        if 5 < bWidth { pProfile5 &+= Int32(pRow[px5]) }
                        if 6 < bWidth { pProfile6 &+= Int32(pRow[px6]) }
                        if 7 < bWidth { pProfile7 &+= Int32(pRow[px7]) }
                    }
                    
                    var score = 0
                    if 0 < bWidth { score &+= Int(Int32((cProfile0 - pProfile0).magnitude)) }
                    if 1 < bWidth { score &+= Int(Int32((cProfile1 - pProfile1).magnitude)) }
                    if 2 < bWidth { score &+= Int(Int32((cProfile2 - pProfile2).magnitude)) }
                    if 3 < bWidth { score &+= Int(Int32((cProfile3 - pProfile3).magnitude)) }
                    if 4 < bWidth { score &+= Int(Int32((cProfile4 - pProfile4).magnitude)) }
                    if 5 < bWidth { score &+= Int(Int32((cProfile5 - pProfile5).magnitude)) }
                    if 6 < bWidth { score &+= Int(Int32((cProfile6 - pProfile6).magnitude)) }
                    if 7 < bWidth { score &+= Int(Int32((cProfile7 - pProfile7).magnitude)) }
                    
                    sBase[row * colCount + col] = score / (bHeight * bWidth)
                }
            }
        }
        return scores
    }
}



/// Bidirectional MV calculation: searches MV in both forward (prev) and backward (next) frames, 
struct UnsafePointerWrapper<T>: @unchecked Sendable {
    let base: UnsafePointer<T>
}

@inline(__always)
func computeBidirectionalMotionVectors(curr: PlaneData420, prev: PlaneData420, next: PlaneData420, prevMVs: MotionVectors, pool: BlockViewPool, roundOffset: Int, gopPosition: Int, skipMap: [BlockMode], cachedNextSub2: [Int16]? = nil, cachedNextSub1: [Int16]? = nil) async -> (MotionVectors, [Int], [Bool], [Int], [Int16], [Int16]) {
    let dx = curr.width
    let dy = curr.height
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    
    // Compute DWT LL band (Base8 resolution) for current frame
    async let (currSub2, rCurrSub2) = extractSingleTransformSubband32(r: curr.rY, width: dx, height: dy, pool: pool)
    
    // Forward reference DWT LL band
    async let (prevSub2, rPrevSub2) = extractSingleTransformSubband32(r: prev.rY, width: dx, height: dy, pool: pool)
    
    let cS2 = await currSub2
    let pS2 = await prevSub2
    
    let nS2: [Int16]
    let nR2: @Sendable () -> Void
    if let cached = cachedNextSub2 {
        nS2 = cached
        nR2 = {}
    } else {
        let (sub2, releaseFn) = await extractSingleTransformSubband32(r: next.rY, width: dx, height: dy, pool: pool)
        nS2 = sub2
        nR2 = releaseFn
    }
    
    async let (currSub1, rCurrSub1) = extractSingleTransformSubband16(r: Int16Reader(data: cS2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    async let (prevSub1, rPrevSub1) = extractSingleTransformSubband16(r: Int16Reader(data: pS2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    
    let (cS1, cR1) = await (currSub1, rCurrSub1)
    let (pS1, pR1) = await (prevSub1, rPrevSub1)
    
    let nS1: [Int16]
    let nR1: @Sendable () -> Void
    if let cached = cachedNextSub1 {
        nS1 = cached
        nR1 = {}
    } else {
        let (sub1, releaseFn) = await extractSingleTransformSubband16(r: Int16Reader(data: nS2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
        nS1 = sub1
        nR1 = releaseFn
    }
    
    let cR2 = await rCurrSub2
    let pR2 = await rPrevSub2
    
    defer {
        cR2()
        cR1()
        pR2()
        pR1()
        nR2()
        nR1()
    }
    
    let targetWidth = l0dx
    let targetHeight = l0dy
    let colCount = (targetWidth + 7) / 8
    let rowCount = (targetHeight + 7) / 8
    let blocks8Count = colCount * rowCount
    
    var mvs = MotionVectors(count: blocks8Count)
    var sads = [Int](repeating: 0, count: blocks8Count)
    var refDirs = [Bool](repeating: false, count: blocks8Count)
    
    struct SliceResult {
        let startIdx: Int
        let dx: [Int16]
        let dy: [Int16]
        let sads: [Int]
        let refDirs: [Bool]
    }
    
    let cS1BaseWrapper = UnsafePointerWrapper(base: cS1.withUnsafeBufferPointer { $0.baseAddress! })
    let pS1BaseWrapper = UnsafePointerWrapper(base: pS1.withUnsafeBufferPointer { $0.baseAddress! })
    let nS1BaseWrapper = UnsafePointerWrapper(base: nS1.withUnsafeBufferPointer { $0.baseAddress! })
    
    let hasSkipMap = skipMap.count > 0
    let skipMapConst = skipMap
    let prevMVsConst = prevMVs
    let currConst = curr
    let prevConst = prev
    let nextConst = next
    
    let numSlices = 4
    let rowsPerSlice = (rowCount + numSlices - 1) / numSlices
    
    let results: [SliceResult] = await withTaskGroup(of: SliceResult.self) { group in
        for sliceIdx in 0..<numSlices {
            let startRow = sliceIdx * rowsPerSlice
            let endRow = min(rowCount, startRow + rowsPerSlice)
            if endRow <= startRow { continue }
            
            let startIdx = startRow * colCount
            let sliceBlocksCount = (endRow - startRow) * colCount
            
            group.addTask {
                let tmpC = pool.get64()
                let tmpO = pool.get64()
                let tmpT = pool.get64()
                defer {
                    pool.put(tmpC)
                    pool.put(tmpO)
                    pool.put(tmpT)
                }
                let cPtr = tmpC.base
                let oPtr = tmpO.base
                let tPtr = tmpT.base
                
                let cBase = cS1BaseWrapper.base
                let pBase = pS1BaseWrapper.base
                let nBase = nS1BaseWrapper.base
                
                var dx = [Int16](repeating: 0, count: sliceBlocksCount)
                var dy = [Int16](repeating: 0, count: sliceBlocksCount)
                var sads = [Int](repeating: 0, count: sliceBlocksCount)
                var refDirs = [Bool](repeating: false, count: sliceBlocksCount)
                
                for i in 0..<sliceBlocksCount {
                    let idx = startIdx + i
                    let col = idx % colCount
                    let row = idx / colCount
                    let bx = col * 8
                    let by = row * 8
                    
                    if hasSkipMap && skipMapConst[idx] != .inter {
                        dx[i] = 0
                        dy[i] = 0
                        sads[i] = 0
                        refDirs[i] = false
                        continue
                    }
                    
                    MotionEstimation.fetchPixelsBlock8(plane: cBase, width: targetWidth, height: targetHeight, x: bx, y: by, dest: cPtr)
                    
                    let mvADx = if 0 < col { dx[i - 1] } else { Int16(0) }
                    let mvADy = if 0 < col { dy[i - 1] } else { Int16(0) }
                    let mvBDx = if colCount <= i { dx[i - colCount] } else { Int16(0) }
                    let mvBDy = if colCount <= i { dy[i - colCount] } else { Int16(0) }
                    let mvCDx = if colCount <= (i - 1) && col < (colCount - 1) { dx[i - colCount + 1] } else { Int16(0) }
                    let mvCDy = if colCount <= (i - 1) && col < (colCount - 1) { dy[i - colCount + 1] } else { Int16(0) }
                    
                    let pmvDx = MotionEstimation.median(Int(mvADx) >> 2, Int(mvBDx) >> 2, Int(mvCDx) >> 2)
                    let pmvDy = MotionEstimation.median(Int(mvADy) >> 2, Int(mvBDy) >> 2, Int(mvCDy) >> 2)
                    let pmv = MotionVector(dx: Int16(pmvDx), dy: Int16(pmvDy))
                    
                    var (mvPrev, mutSADPrev) = MotionEstimation.searchPixels(
                        cBase: cBase, pBase: pBase,
                        cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
                        width: targetWidth, height: targetHeight, bx: bx, by: by, range: 8, pmv: pmv, roundOffset: roundOffset
                    )
                    
                    if 512 < mutSADPrev && idx < prevMVsConst.count {
                        let tmv = MotionVector(dx: Int16(Int(prevMVsConst.dx[idx]) >> 2), dy: Int16(Int(prevMVsConst.dy[idx]) >> 2))
                        let (tmvMv, tmvSad) = MotionEstimation.searchPixels(
                            cBase: cBase, pBase: pBase,
                            cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
                            width: targetWidth, height: targetHeight, bx: bx, by: by, range: 4, pmv: tmv, roundOffset: roundOffset
                        )
                        if tmvSad < mutSADPrev {
                            mvPrev = tmvMv
                            mutSADPrev = tmvSad
                        }
                    }
                    
                    let currContrast = MotionEstimation.extractContrast8x8(base: cBase, width: targetWidth, height: targetHeight, bx: bx, by: by)
                    
                    var dynamicThreshold = 1024
                    if 1024 < mutSADPrev {
                        let dynT = max(1024, currContrast * 48)
                        if dynT < mutSADPrev {
                            var mvVariance = 0
                            if 0 < col && colCount <= i {
                                let dxDiff = Int((Int(dx[i - 1]) - Int(dx[i - colCount])).magnitude)
                                let dyDiff = Int((Int(dy[i - 1]) - Int(dy[i - colCount])).magnitude)
                                mvVariance = dxDiff + dyDiff
                            }
                            if 32 < mvVariance && (dynT * 2) < mutSADPrev {
                                let (expMv, expSad) = MotionEstimation.searchPixels(
                                    cBase: cBase, pBase: pBase,
                                    cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
                                    width: targetWidth, height: targetHeight, bx: bx, by: by, range: 12, pmv: mvPrev, roundOffset: roundOffset
                                )
                                if expSad < mutSADPrev {
                                    mvPrev = expMv
                                    mutSADPrev = expSad
                                }
                            }
                        }
                    }
                    
                    let (mvNext, mutSADNext) = MotionEstimation.searchPixels(
                        cBase: cBase, pBase: nBase,
                        cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
                        width: targetWidth, height: targetHeight, bx: bx, by: by, range: 8, pmv: pmv, roundOffset: roundOffset
                    )
                    
                    let prevChromaSad = MotionEstimation.computeChromaSAD(curr: currConst, ref: prevConst, bx: bx, by: by, refDx: Int(mvPrev.dx), refDy: Int(mvPrev.dy))
                    mutSADPrev += prevChromaSad / 4
                    let prevSAD = mutSADPrev
                    
                    var bestMV = mvPrev
                    var dir = false
                    
                    let earlyExitThreshold = min(1536, 512 + (gopPosition * 16))
                    let gopPenalty = min(1024, gopPosition * 16)
                    
                    if earlyExitThreshold <= prevSAD {
                        let mvEnergyNext = Int((Int(mvNext.dx)).magnitude) + Int((Int(mvNext.dy)).magnitude)
                        let effectiveGopPenalty = if mutSADNext < 384 { 0 } else { gopPenalty }
                        let baselinePenalty = (mvEnergyNext * 8) + 32 + effectiveGopPenalty
                        
                        if mutSADNext + baselinePenalty < prevSAD {
                            // Quarter-resolution displacement is mv/4 (the mv
                            // is in full-resolution pixels; the previous
                            // intNextDx>>2 = mv/2 probed double the true
                            // displacement).
                            let nextContrast = MotionEstimation.extractContrast8x8(base: nBase, width: targetWidth, height: targetHeight, bx: bx + (Int(mvNext.dx) >> 2), by: by + (Int(mvNext.dy) >> 2))
                            
                            let contrastDiff = Int((currContrast - nextContrast).magnitude)
                            let structurePenalty = (contrastDiff * contrastDiff) / 4
                            let chromaSAD = MotionEstimation.computeChromaSAD(curr: currConst, ref: nextConst, bx: bx, by: by, refDx: Int(mvNext.dx), refDy: Int(mvNext.dy))
                            let chromaPenalty = chromaSAD / 4
                            
                            let totalNextPenalty = ((mutSADNext + baselinePenalty) + (structurePenalty + chromaPenalty))
                            let energyNext = (mvNext.dy * mvNext.dy) + (mvNext.dx * mvNext.dx)
                            let energyPrev = (mvPrev.dy * mvPrev.dy) + (mvPrev.dx * mvPrev.dx)
                            
                            switch true {
                            case totalNextPenalty < prevSAD:
                                bestMV = mvNext
                                dir = true
                            case (totalNextPenalty == prevSAD) && (energyNext < energyPrev):
                                bestMV = mvNext
                                dir = true
                            default:
                                break
                            }
                        }
                    }
                    dynamicThreshold = max(1024, currContrast * 48)
                    let finalSAD = dir ? (mutSADNext + (MotionEstimation.computeChromaSAD(curr: currConst, ref: nextConst, bx: bx, by: by, refDx: Int(mvNext.dx), refDy: Int(mvNext.dy)) / 4)) : prevSAD
                    switch true {
                    case dynamicThreshold < finalSAD:
                        dx[i] = 0
                        dy[i] = 0
                    case finalSAD < 256:
                        dx[i] = bestMV.dx * 4
                        dy[i] = bestMV.dy * 4
                    default:
                        let refPlane = if dir { nextConst.y } else { prevConst.y }
                        let (refinedMV, _) = MotionEstimation.searchPixelsSubpixelRefinement32(
                            currPlane: currConst.y, prevPlane: refPlane,
                            width: targetWidth, height: targetHeight, bx: col * 32, by: row * 32, pmv: bestMV
                        )
                        dx[i] = refinedMV.dx
                        dy[i] = refinedMV.dy
                    }
                    sads[i] = finalSAD
                    refDirs[i] = dir
                }
                return SliceResult(startIdx: startIdx, dx: dx, dy: dy, sads: sads, refDirs: refDirs)
            }
        }
        
        var collected = [SliceResult]()
        for await res in group {
            collected.append(res)
        }
        return collected
    }
    
    for res in results {
        let endIdx = res.startIdx + res.dx.count
        mvs.dx.replaceSubrange(res.startIdx..<endIdx, with: res.dx)
        mvs.dy.replaceSubrange(res.startIdx..<endIdx, with: res.dy)
        sads.replaceSubrange(res.startIdx..<endIdx, with: res.sads)
        refDirs.replaceSubrange(res.startIdx..<endIdx, with: res.refDirs)
    }
    
    let occlusionScores = MotionEstimation.computeOcclusionScores(currPlane: cS1, prevPlane: pS1, width: targetWidth, height: targetHeight, globalPrior: MotionVector(dx: 0, dy: 0))
    return (mvs, sads, refDirs, occlusionScores, nS2, nS1)
}
