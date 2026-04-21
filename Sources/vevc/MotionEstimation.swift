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

private let meFineOffsets: [(Int, Int)] = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1)
]

private let meSearchOffsetX: [Int] = [-1, 0, 1, -1, 1, -1, 0, 1]
private let meSearchOffsetY: [Int] = [-1, -1, -1, 0, 0, 1, 1, 1]

struct MotionEstimation {

    // small blocks benefit more from compiler auto-vectorization than manual SIMD
    @inline(__always)
    static func absDiff(_ a: Int16, _ b: Int16) -> Int {
        return abs(Int(a) - Int(b))
    }

    // Lagrange multiplier for rate-distortion optimization:
    // penalizes motion vectors with large magnitude to favor zero-MV
    @inline(__always)
    static func getPenalty(dx: Int, dy: Int, pmv: MotionVector, lambda: Int) -> Int {
        return (abs(dx) + abs(dy)) * lambda
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
    static func fetchEighthPixelBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, intX: Int, intY: Int, remX: Int, remY: Int, dest: UnsafeMutablePointer<Int16>, roundOffset: Int) {
        if remX == 0 && remY == 0 {
            fetchPixelsBlock8(plane: plane, width: width, height: height, x: intX, y: intY, dest: dest)
            return
        }
        let nextX = if remX == 0 { 0 } else { 1 }
        let nextY = if remY == 0 { 0 } else { 1 }
        let wA = 8 - remX
        let wB = remX
        let wC = 8 - remY
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
                let v0 = (wA * wC) * Int(row0[sx0])
                let v1 = (wB * wC) * Int(row0[sx1])
                let v2 = (wA * wD) * Int(row1[sx0])
                let v3 = (wB * wD) * Int(row1[sx1])
                let v = (v0 + v1) + (v2 + v3)
                dst[rx] = Int16((v + 31 + roundOffset) >> 6)
            }
        }
    }

    @inline(__always)
    static func compute64PointSADBlocksWithStride(cBase: UnsafePointer<Int16>, pBase: UnsafePointer<Int16>, pStride: Int) -> Int {
        var sad: Int32 = 0
        for ry in 0..<8 {
            let cRow = cBase.advanced(by: ry * 8)
            let pRow = pBase.advanced(by: ry * pStride)
            
            for rx in 0..<8 {
                let diff = Int32(cRow[rx]) - Int32(pRow[rx])
                let absDiff = if diff < 0 { -1 * diff } else { diff }
                sad &+= absDiff
            }
        }
        return Int(sad)
    }

    @inline(__always)
    static func compute64PointSADBlocks(cBase: UnsafePointer<Int16>, pBase: UnsafePointer<Int16>) -> Int {
        var sad: Int32 = 0
        for i in 0..<64 {
            let diff = Int32(cBase[i]) - Int32(pBase[i])
            let absDiff = if diff < 0 { -1 * diff } else { diff }
            sad &+= absDiff
        }
        return Int(sad)
    }

    @inline(__always)
    static func compute32PointSADEvenRows(cBase: UnsafePointer<Int16>, pBase: UnsafePointer<Int16>) -> Int {
        var sad: Int32 = 0
        for row in 0..<4 {
            let offset = row * 16
            let cRow = cBase.advanced(by: offset)
            let pRow = pBase.advanced(by: offset)
            for x in 0..<8 {
                let diff = Int32(cRow[x]) - Int32(pRow[x])
                let absDiff = if diff < 0 { -1 * diff } else { diff }
                sad &+= absDiff
            }
        }
        return Int(sad) * 2
    }
    
    @inline(__always)
    static func median(_ a: Int, _ b: Int, _ c: Int) -> Int {
        return max(min(a, b), min(max(a, b), c))
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
        let zeroSad: Int = compute64PointSADBlocks(cBase: cPtr, pBase: oPtr)
        
        if zeroSad < 64 {
            return (0, 0, zeroSad)
        }
        
        var bestCoarseSad = zeroSad
        var bestCoarseDx = 0
        var bestCoarseDy = 0
        
        let minDy = max(-1 * range, -1 * by)
        let maxDy = min(range, height - by - 8)
        let minDx = max(-1 * range, -1 * bx)
        let maxDx = min(range, width - bx - 8)
        
        if minDy <= maxDy && minDx <= maxDx {
            var centerX = 0
            var centerY = 0
            
            while true {
                var minSAD = bestCoarseSad
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
                    
                    let penalty = getPenalty(dx: dx, dy: dy, pmv: pmv, lambda: 8)
                    let maxSad = bestCoarseSad - penalty
                    if maxSad < 0 { continue }
                    
                    let pPtr = pBase.advanced(by: (by + dy) * width + (bx + dx))
                    let sad = compute64PointSADBlocksWithStride(cBase: cPtr, pBase: pPtr, pStride: width)
                    
                    let totalSad = sad + penalty
                    if totalSad < minSAD {
                        minSAD = totalSad
                        minDxPos = dx
                        minDyPos = dy
                        foundSmaller = true
                    }
                }
                
                if !foundSmaller {
                    break
                }
                
                bestCoarseSad = minSAD
                bestCoarseDx = minDxPos
                bestCoarseDy = minDyPos
                centerX = minDxPos
                centerY = minDyPos
                
                if bestCoarseSad < 64 {
                    break
                }
            }
            
            var finalMinSAD = bestCoarseSad
            var finalMinDx = centerX
            var finalMinDy = centerY
            
            if finalMinSAD >= 64 {
                for i in 0..<4 {
                    let dx = centerX + dsSdspX[i]
                    let dy = centerY + dsSdspY[i]
                    
                    if dx < minDx { continue }
                    if maxDx < dx { continue }
                    if dy < minDy { continue }
                    if maxDy < dy { continue }
                    
                    let penalty = getPenalty(dx: dx, dy: dy, pmv: pmv, lambda: 8)
                    let maxSad = bestCoarseSad - penalty
                    if maxSad < 0 { continue }
                    
                    let pPtr = pBase.advanced(by: (by + dy) * width + (bx + dx))
                    let sad = compute64PointSADBlocksWithStride(cBase: cPtr, pBase: pPtr, pStride: width)
                    
                    let totalSad = sad + penalty
                    if totalSad < finalMinSAD {
                        finalMinSAD = totalSad
                        finalMinDx = dx
                        finalMinDy = dy
                    }
                }
            }
            
            bestCoarseSad = finalMinSAD
            bestCoarseDx = finalMinDx
            bestCoarseDy = finalMinDy
        }
        
        var bestFineSad: Int = bestCoarseSad
        var bestFineDx: Int = bestCoarseDx
        var bestFineDy: Int = bestCoarseDy
        
        let fineOffsets = meFineOffsets
        
        for offset in fineOffsets {
            let fx: Int = offset.0
            let fy: Int = offset.1
            let fineDx: Int = bestCoarseDx + fx
            let fineDy: Int = bestCoarseDy + fy
            
            if fineDx < -4 || 4 < fineDx || fineDy < -4 || 4 < fineDy { continue }
            
            let penalty = getPenalty(dx: fineDx, dy: fineDy, pmv: pmv, lambda: 8)
            let maxSad = bestFineSad - penalty
            if maxSad < 0 { continue }
            
            fetchPixelsBlock8(plane: pBase, width: width, height: height, x: bx + fineDx, y: by + fineDy, dest: tPtr)
            let sad = compute64PointSADBlocks(cBase: cPtr, pBase: tPtr)
            
            let totalSad = sad + penalty
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
                let hx = meSearchOffsetX[oi]
                let hy = meSearchOffsetY[oi]
                let hpDx: Int = bestFineDx * 2 + hx
                let hpDy: Int = bestFineDy * 2 + hy
                
                let intDx: Int = hpDx >> 1
                let intDy: Int = hpDy >> 1
                let fractX: Int = hpDx & 1
                let fractY: Int = hpDy & 1
                
                let penalty = getPenalty(dx: hpDx, dy: hpDy, pmv: pmv, lambda: 4)
                let maxSad = bestHpSad - penalty
                if maxSad < 0 { continue }
                
                fetchHalfPixelBlock8(
                    plane: pBase, width: width, height: height,
                    intX: bx + intDx, intY: by + intDy,
                    fractX: fractX, fractY: fractY, dest: tPtr, roundOffset: roundOffset
                )
                let sad = compute64PointSADBlocks(cBase: cPtr, pBase: tPtr)
                
                let totalSad = sad + penalty
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
                let hx = meSearchOffsetX[oi]
                let hy = meSearchOffsetY[oi]
                let epDx: Int = bestHpDx * 4 + hx
                let epDy: Int = bestHpDy * 4 + hy
                
                let intDx: Int = epDx >> 3
                let intDy: Int = epDy >> 3
                let remX: Int = epDx & 7
                let remY: Int = epDy & 7
                
                let penalty = getPenalty(dx: epDx, dy: epDy, pmv: pmv, lambda: 2)
                let maxSad = bestEpSad - penalty
                if maxSad < 0 { continue }
                
                fetchEighthPixelBlock8(
                    plane: pBase, width: width, height: height,
                    intX: bx + intDx, intY: by + intDy,
                    remX: remX, remY: remY, dest: tPtr, roundOffset: roundOffset
                )
                let sad = compute64PointSADBlocks(cBase: cPtr, pBase: tPtr)
                
                let totalSad = sad + penalty
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
            
            let isSafeX = (0 <= bx) && (bx + 8 <= width)
            let isSafeY = (0 <= by) && (by + 8 <= height)
            if isSafeX && isSafeY {
                for y in 0..<8 {
                    let row = base.advanced(by: (by + y) * width + bx)
                    for x in 0..<8 {
                        let val = Int32(row[x])
                        if val < minVal { minVal = val }
                        if maxVal < val { maxVal = val }
                    }
                }
                return
            }

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
        
        let isCurrSafe = (0 <= bx2) && (0 <= by2) && (bx2 + 16 <= cbw) && (by2 + 16 <= cbh)
        let isRefSafe = (0 <= refX2) && (0 <= refY2) && (refX2 + 16 <= cbw) && (refY2 + 16 <= cbh)
        
        if isCurrSafe && isRefSafe {
            let sad = withUnsafePointers(curr.cb, curr.cr, ref.cb, ref.cr) { cCb, cCr, rCb, rCr in
                var sad: Int32 = 0
                for cy in stride(from: 0, to: 16, by: 2) {
                    let currOffset = (by2 + cy) * cbw + bx2
                    let refOffset = (refY2 + cy) * cbw + refX2
                    for cx in stride(from: 0, to: 16, by: 2) {
                        let diffCb = Int32(cCb[currOffset + cx]) - Int32(rCb[refOffset + cx])
                        let absDiffCb = if diffCb < 0 { -1 * diffCb } else { diffCb }
                        sad &+= absDiffCb
                        let diffCr = Int32(cCr[currOffset + cx]) - Int32(rCr[refOffset + cx])
                        let absDiffCr = if diffCr < 0 { -1 * diffCr } else { diffCr }
                        sad &+= absDiffCr
                    }
                    if 2000 < sad { break } // Early Termination
                }
                return Int(sad) * 4 // 間引いた分をスケールアップ
            }
            return sad
        }
        return 1000
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
        
        var sad: Int32 = 0
        let safe = (bx + intDx - 1 >= 0) && (by + intDy - 1 >= 0) && (bx + intDx + 32 + 2 < width) && (by + intDy + 32 + 2 < height)
        let useFIR = (fractX != 0 || fractY != 0)
        
        let cX0 = Int32(fX[0]), cX1 = Int32(fX[1]), cX2 = Int32(fX[2]), cX3 = Int32(fX[3])
        let cY0 = Int32(fY[0]), cY1 = Int32(fY[1]), cY2 = Int32(fY[2]), cY3 = Int32(fY[3])
        
        if safe {
            if useFIR != true {
                for ry in stride(from: 0, to: 32, by: 4) {
                    let cy = by + ry
                    let rowC = curr.advanced(by: cy * width + bx)
                    
                    let py = by + intDy + ry
                    let r = prev.advanced(by: py * width + bx + intDx)
                    for rx in stride(from: 0, to: 32, by: 2) {
                        let diff = Int32(rowC[rx]) - Int32(r[rx])
                        let absDiff = if diff < 0 { -1 * diff } else { diff }
                        sad &+= absDiff
                    }
                }
            } else {
                for ry in stride(from: 0, to: 32, by: 4) {
                    let cy = by + ry
                    let rowC = curr.advanced(by: cy * width + bx)
                    
                    let py = by + intDy + ry
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
                        let absDiff = if diff < 0 { -1 * diff } else { diff }
                        sad &+= absDiff
                        rx &+= 2
                    }
                }
            }
            return Int(sad)
        }
        
        if useFIR != true {
            for ry in stride(from: 0, to: 32, by: 4) {
                let cy = min(by + ry, height - 1)
                let rowC = curr.advanced(by: cy * width)
                
                let py = by + intDy + ry
                let sy0 = max(0, min(py, height - 1))
                let r = prev.advanced(by: sy0 * width)
                for rx in stride(from: 0, to: 32, by: 2) {
                    let px = bx + intDx + rx
                    let sx = max(0, min(px, width - 1))
                    let cx = min(bx + rx, width - 1)
                    let diff = Int32(rowC[cx]) - Int32(r[sx])
                    let absDiff = if diff < 0 { -1 * diff } else { diff }
                    sad &+= absDiff
                }
            }
        } else {
            for ry in stride(from: 0, to: 32, by: 4) {
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
                    let absDiff = if diff < 0 { -1 * diff } else { diff }
                    sad &+= absDiff
                    rx &+= 2
                }
            }
        }
        return Int(sad)
    }

    static let searchOffsets = [
        (0, -1), (0, 1), (-1, 0), (1, 0),
        (-1, -1), (1, -1), (-1, 1), (1, 1)
    ]

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
                
                var bestSad = computeQuarterPixelSADSubsampled32(curr: cBase, prev: pBase, width: width, height: height, bx: bx, by: by, qDx: bestQx, qDy: bestQy)
                
                if bestSad < 128 { return (MotionVector(dx: Int16(bestQx), dy: Int16(bestQy)), bestSad) }
                

                
                for (ox, oy) in searchOffsets {
                    let qx = baseQx + ox
                    let qy = baseQy + oy
                    
                    let intDx = qx >> 2
                    let intDy = qy >> 2
                    if bx + intDx < -32 || bx + intDx + 32 > width + 32 { continue }
                    if by + intDy < -32 || by + intDy + 32 > height + 32 { continue }
                    
                    let penalty = (abs(ox) + abs(oy)) * 6
                    let maxSad = bestSad - penalty
                    if maxSad <= 0 { continue }
                    
                    let sad = computeQuarterPixelSADSubsampled32(curr: cBase, prev: pBase, width: width, height: height, bx: bx, by: by, qDx: qx, qDy: qy)
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

@inline(__always)
func computeMotionVectors(curr: PlaneData420, prev: PlaneData420, pool: BlockViewPool, roundOffset: Int) async -> ([MotionVector], [Int]) {
    let dx = curr.width
    let dy = curr.height
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    
    let (currSub2, rCurrSub2) = await extractSingleTransformSubband32(r: curr.rY, width: dx, height: dy, pool: pool)
    let (currSub1, rCurrSub1) = await extractSingleTransformSubband16(r: Int16Reader(data: currSub2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    let (currBlocks8, rCurrBlocks8) = await extractSingleTransformBlocksBase8(r: Int16Reader(data: currSub1, width: l0dx, height: l0dy), width: l0dx, height: l0dy, pool: pool)
    defer {
        rCurrSub2()
        rCurrSub1()
        rCurrBlocks8()
    }

    let (prevSub2, rPrevSub2) = await extractSingleTransformSubband32(r: prev.rY, width: dx, height: dy, pool: pool)
    let (prevSub1, rPrevSub1) = await extractSingleTransformSubband16(r: Int16Reader(data: prevSub2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    defer {
        rPrevSub2()
        rPrevSub1()
    }
    
    let targetWidth = l0dx
    let targetHeight = l0dy
    let colCount = (targetWidth + 7) / 8
    
    var mvs = [MotionVector](repeating: MotionVector(dx: 0, dy: 0), count: currBlocks8.count)
    var sads = [Int](repeating: 0, count: currBlocks8.count)
    
    let tmpC = pool.get(width: 8, height: 8)
    let tmpO = pool.get(width: 8, height: 8)
    let tmpT = pool.get(width: 8, height: 8)
    defer {
        pool.put(tmpC)
        pool.put(tmpO)
        pool.put(tmpT)
    }
    let cPtr = tmpC.base
    let oPtr = tmpO.base
    let tPtr = tmpT.base

    mvs.withUnsafeMutableBufferPointer { mvsPtr in
        sads.withUnsafeMutableBufferPointer { sadsPtr in
            for idx in currBlocks8.indices {
                let col = idx % colCount
                let row = idx / colCount
                let bx = col * 8
                let by = row * 8
                let pmv = if 0 < col { mvsPtr[idx - 1] } else { MotionVector(dx: 0, dy: 0) }
                let (mv, sad) = MotionEstimation.searchPixels(
                    currPlane: currSub1, prevPlane: prevSub1, 
                    cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
                    width: targetWidth, height: targetHeight, bx: bx, by: by, range: 2, pmv: pmv,
                    roundOffset: roundOffset
                )
                
                let currContrast = MotionEstimation.extractContrast8x8(plane: currSub1, width: targetWidth, height: targetHeight, bx: bx, by: by)
                
                // Dynamic Threshold: For flat blocks use 2048 (lenient), for high contrast (< 1000) use a lower value to promote Intra prediction.
                // Ghosting occurs when high contrast edges are dragged in the wrong direction.
                let dynamicThreshold = max(256, 2048 - (currContrast * 2))
                
                if dynamicThreshold < sad {
                    mvsPtr[idx] = MotionVector.intraBlock
                    sadsPtr[idx] = sad
                } else {
                    mvsPtr[idx] = mv
                    sadsPtr[idx] = sad
                }
            }
        }
    }
    return (mvs, sads)
}

/// Bidirectional MV calculation: searches MV in both forward (prev) and backward (next) frames, 
/// and selects the one with smaller SAD per block.
/// - Returns: (mvs, sads, refDirs) where refDirs is the reference direction flag per block (false=forward, true=backward)
@inline(__always)
func computeBidirectionalMotionVectors(curr: PlaneData420, prev: PlaneData420, next: PlaneData420, pool: BlockViewPool, roundOffset: Int, gopPosition: Int) async -> ([MotionVector], [Int], [Bool]) {
    let dx = curr.width
    let dy = curr.height
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    
    // Compute DWT LL band (Base8 resolution) for current frame
    let (currSub2, rCurrSub2) = await extractSingleTransformSubband32(r: curr.rY, width: dx, height: dy, pool: pool)
    let (currSub1, rCurrSub1) = await extractSingleTransformSubband16(r: Int16Reader(data: currSub2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    let (currBlocks8, rCurrBlocks8) = await extractSingleTransformBlocksBase8(r: Int16Reader(data: currSub1, width: l0dx, height: l0dy), width: l0dx, height: l0dy, pool: pool)
    defer {
        rCurrSub2()
        rCurrSub1()
        rCurrBlocks8()
    }

    // Forward reference DWT LL band
    let (prevSub2, rPrevSub2) = await extractSingleTransformSubband32(r: prev.rY, width: dx, height: dy, pool: pool)
    let (prevSub1, rPrevSub1) = await extractSingleTransformSubband16(r: Int16Reader(data: prevSub2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    defer {
        rPrevSub2()
        rPrevSub1()
    }
    
    // Backward reference DWT LL band
    let (nextSub2, rNextSub2) = await extractSingleTransformSubband32(r: next.rY, width: dx, height: dy, pool: pool)
    let (nextSub1, rNextSub1) = await extractSingleTransformSubband16(r: Int16Reader(data: nextSub2, width: l1dx, height: l1dy), width: l1dx, height: l1dy, pool: pool)
    defer {
        rNextSub2()
        rNextSub1()
    }
    
    let targetWidth = l0dx
    let targetHeight = l0dy
    let colCount = (targetWidth + 7) / 8
    
    var mvs = [MotionVector](repeating: MotionVector(dx: 0, dy: 0), count: currBlocks8.count)
    var sads = [Int](repeating: 0, count: currBlocks8.count)
    var refDirs = [Bool](repeating: false, count: currBlocks8.count)
    
    let tmpC = pool.get(width: 8, height: 8)
    let tmpO = pool.get(width: 8, height: 8)
    let tmpT = pool.get(width: 8, height: 8)
    defer {
        pool.put(tmpC)
        pool.put(tmpO)
        pool.put(tmpT)
    }
    let cPtr = tmpC.base
    let oPtr = tmpO.base
    let tPtr = tmpT.base

    withUnsafePointers(mut: &mvs, mut: &sads, mut: &refDirs) { mvsPtr, sadsPtr, refDirsPtr in
        for idx in currBlocks8.indices {
            let col = idx % colCount
            let row = idx / colCount
            let bx = col * 8
            let by = row * 8
            
            let mvA = if 0 < col { mvsPtr[idx - 1] } else { MotionVector(dx: 0, dy: 0) }
            let mvB = if 0 < row { mvsPtr[idx - colCount] } else { MotionVector(dx: 0, dy: 0) }
            let mvC = if 0 < row && col < (colCount - 1) { mvsPtr[idx - colCount + 1] } else { MotionVector(dx: 0, dy: 0) }
            
            let pmvDx = MotionEstimation.median(Int(mvA.dx), Int(mvB.dx), Int(mvC.dx))
            let pmvDy = MotionEstimation.median(Int(mvA.dy), Int(mvB.dy), Int(mvC.dy))
            let pmv = MotionVector(dx: Int16(pmvDx), dy: Int16(pmvDy))
            
            let (mvPrev, mutSadPrev) = MotionEstimation.searchPixels(
                currPlane: currSub1, prevPlane: prevSub1,
                cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
                width: targetWidth, height: targetHeight, bx: bx, by: by, range: 4, pmv: pmv, roundOffset: roundOffset
            )
            
            let prevChromaPenalty: Int
            if mutSadPrev <= 512 {
                let intPrevDx = Int(mvPrev.dx) >> 3
                let intPrevDy = Int(mvPrev.dy) >> 3
                let prevChromaSad = MotionEstimation.computeChromaSAD(curr: curr, ref: prev, bx: bx, by: by, refDx: intPrevDx, refDy: intPrevDy)
                prevChromaPenalty = prevChromaSad / 4
            } else {
                prevChromaPenalty = 0
            }
            let sadPrev = mutSadPrev + prevChromaPenalty
            
            var bestMv = mvPrev
            var dir = false
            
            // Early Exit: If the forward prediction is extremely good, skip backward prediction.
            // A SAD of 256 means an average error of 4 per pixel in an 8x8 block.
            // Always check backward prediction if sadPrev >= 256, so trailing ghosts can be erased.
            let gopPenalty = gopPosition * 64
            if 256 <= sadPrev {
                let (mvNext, sadNext) = MotionEstimation.searchPixels(
                    currPlane: currSub1, prevPlane: nextSub1,
                    cPtr: cPtr, oPtr: oPtr, tPtr: tPtr,
                    width: targetWidth, height: targetHeight, bx: bx, by: by, range: 4, pmv: pmv, roundOffset: roundOffset
                )
                
                let mvEnergyNext = abs(Int(mvNext.dx)) + abs(Int(mvNext.dy))
                
                // If I-frame (next) predicts the block extremely well, it's overwhelmingly likely
                // a static background. Waive the GOP penalty to allow instantaneous ghost erasure.
                let effectiveGopPenalty = if sadNext < 256 { 0 } else { gopPenalty }
                let baselinePenalty = (mvEnergyNext * 8) + 32 + effectiveGopPenalty
                
                if sadNext + baselinePenalty < sadPrev {
                    // Structural Validation
                    let currContrast = MotionEstimation.extractContrast8x8(plane: currSub1, width: targetWidth, height: targetHeight, bx: bx, by: by)
                    let intNextDx = Int(mvNext.dx) >> 3
                    let intNextDy = Int(mvNext.dy) >> 3
                    let nextContrast = MotionEstimation.extractContrast8x8(plane: nextSub1, width: targetWidth, height: targetHeight, bx: bx + intNextDx, by: by + intNextDy)
                    
                    let contrastDiff = abs(currContrast - nextContrast)
                    let structurePenalty = contrastDiff * contrastDiff
                    
                    // Chroma SAD penalty to completely block mismatching colors (e.g. blue background vs red hair)
                    let chromaSAD = MotionEstimation.computeChromaSAD(curr: curr, ref: next, bx: bx, by: by, refDx: intNextDx, refDy: intNextDy)
                    
                    let chromaPenalty = chromaSAD / 4
                    
                    let totalNextPenalty = ((sadNext + baselinePenalty) + (structurePenalty + chromaPenalty))
                    let energyNext = (mvNext.dy * mvNext.dy) + (mvNext.dx * mvNext.dx)
                    let energyPrev = (mvPrev.dy * mvPrev.dy) + (mvPrev.dx * mvPrev.dx)
                    
                    switch true {
                    case totalNextPenalty < sadPrev:
                        bestMv = mvNext
                        dir = true
                    case (totalNextPenalty == sadPrev) && (energyNext < energyPrev):
                        bestMv = mvNext
                        dir = true
                    default:
                        break
                    }
                }
            }
            
            // Refine ME on full resolution Luma (1/4 pixel precision)
            let actPrev = if dir { next } else { prev }
            let (rv, rsad) = MotionEstimation.searchPixelsQuarterRefinement32(
                currPlane: curr.y, prevPlane: actPrev.y,
                width: curr.width, height: curr.height,
                bx: bx * 4, by: by * 4, pmv: bestMv
            )
            
            // Full res 32x32 block (1024 pixels).
            // Average allowed pixel error.
            let currContrast = MotionEstimation.extractContrast8x8(plane: currSub1, width: targetWidth, height: targetHeight, bx: bx, by: by)
            // If there is strong contrast (edges), aggressively lower the SAD tolerance.
            // 8192 corresponds to an average error of 8, causing white blurring and ghosting globally.
            // Lower the SAD tolerance based on contrast (max 8192, min ~1024).
            let dynamicThreshold = max(1024, 8192 - (currContrast * 8))
            
            if dynamicThreshold < rsad {
                mvsPtr[idx] = MotionVector.intraBlock
                sadsPtr[idx] = rsad
                refDirsPtr[idx] = false
            } else {
                mvsPtr[idx] = rv
                sadsPtr[idx] = rsad
                refDirsPtr[idx] = dir
            }
        }
    }
    return (mvs, sads, refDirs)
}
