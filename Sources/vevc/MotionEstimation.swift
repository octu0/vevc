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
    @inline(__always)
    static func absDiff(_ a: Int16, _ b: Int16) -> Int {
        return abs(Int(a) - Int(b))
    }

    @inline(__always)
    static func getPenalty(dx: Int, dy: Int, lambda: Int) -> Int {
        let absX = dx < 0 ? -dx : dx
        let absY = dy < 0 ? -dy : dy
        return (absX + absY) * lambda
    }

    @inline(__always)
    static func fetchPixelsBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, x: Int, y: Int, dest: UnsafeMutablePointer<Int16>) {
        for i in 0..<64 {
            let ry = i / 8
            let rx = i % 8
            let srcY = min(max(0, y + ry), height - 1)
            let srcX = min(max(0, x + rx), width - 1)
            dest[i] = plane[srcY * width + srcX]
        }
    }

    @inline(__always)
    static func compute64PointSAD_Blocks(cBase: UnsafePointer<Int16>, pBase: UnsafePointer<Int16>) -> Int {
        var sad = 0
        for i in 0..<64 {
            sad += absDiff(cBase[i], pBase[i])
        }
        return sad
    }

    @inline(__always)
    private static func evaluateSearch(
        cPtr: UnsafePointer<Int16>, 
        pBase: UnsafePointer<Int16>, 
        width: Int, height: Int, bx: Int, by: Int
    ) -> (Int, Int, Int) {
        let oPtr = UnsafeMutablePointer<Int16>.allocate(capacity: 64)
        let tPtr = UnsafeMutablePointer<Int16>.allocate(capacity: 64)
        defer {
            oPtr.deallocate()
            tPtr.deallocate()
        }

        // 1. Evaluate (0,0) first as baseline
        fetchPixelsBlock8(plane: pBase, width: width, height: height, x: bx, y: by, dest: oPtr)
        let zeroSad: Int = compute64PointSAD_Blocks(cBase: cPtr, pBase: oPtr)
        
        if zeroSad < 64 {
            return (0, 0, zeroSad)
        }
        
        // 2. Coarse Search (step=2, range=4 implies ±2, ±4 but in EncodePlane we pass range=2. To be safe, test 8 surrounding Coarse points)
        var bestCoarseSad: Int = zeroSad
        var bestCoarseDx: Int = 0
        var bestCoarseDy: Int = 0
        
        // Hardcoded tuple array for coarse search (±2, step=2)
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
        
        // 3. Fine Search (step=1) around bestCoarse
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
            
            if fineDx < -4 || fineDx > 4 || fineDy < -4 || fineDy > 4 { continue }
            
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
        
        return (bestFineDx, bestFineDy, bestFineSad)
    }

    @inline(__always)
    static func searchPixels(currPlane: [Int16], prevPlane: [Int16], width: Int, height: Int, bx: Int, by: Int, range: Int = 4) -> (MotionVector, Int) {
        return currPlane.withUnsafeBufferPointer { cBuf in
            prevPlane.withUnsafeBufferPointer { pBuf in
                guard let cBase = cBuf.baseAddress, let pBase = pBuf.baseAddress else {
                    return (MotionVector(dx: 0, dy: 0), 0)
                }

                let cPtr = UnsafeMutablePointer<Int16>.allocate(capacity: 64)
                defer { cPtr.deallocate() }
                
                fetchPixelsBlock8(plane: cBase, width: width, height: height, x: bx, y: by, dest: cPtr)
                let (dx, dy, sad) = evaluateSearch(cPtr: cPtr, pBase: pBase, width: width, height: height, bx: bx, by: by)
                return (MotionVector(dx: Int16(dx), dy: Int16(dy)), sad)
            }
        }
    }
}



