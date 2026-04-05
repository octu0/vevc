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

    // why: coarse-to-fine two-stage search reduces computation from O(N^2) to O(16)
    // 1. evaluate SAD at zero vector (0,0) as baseline
    // 2. coarse search: step=2 diamond, pick lowest-cost among 8 neighbors
    // 3. fine search: step=1 around coarse best, re-evaluate 8 neighbors
    // achieves near-optimal result at fixed O(16) cost vs full search O(N^2)
    @inline(__always)
    private static func evaluateSearch(
        cPtr: UnsafePointer<Int16>, 
        pBase: UnsafePointer<Int16>, 
        oPtr: UnsafeMutablePointer<Int16>,
        tPtr: UnsafeMutablePointer<Int16>,
        width: Int, height: Int, bx: Int, by: Int
    ) -> (Int, Int, Int) {
        // why: skip search when zero-MV SAD is below threshold
        // (static scene or very similar blocks)
        fetchPixelsBlock8(plane: pBase, width: width, height: height, x: bx, y: by, dest: oPtr)
        let zeroSad: Int = compute64PointSAD_Blocks(cBase: cPtr, pBase: oPtr)
        
        if zeroSad < 64 {
            return (0, 0, zeroSad)
        }
        
        // why: coarse step evaluates ±2 pixel 8-neighbors
        var bestCoarseSad: Int = zeroSad
        var bestCoarseDx: Int = 0
        var bestCoarseDy: Int = 0
        
        // why: fixed 8-direction pattern (not full search) keeps cost at O(8)
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
        
        // why: fine step re-evaluates ±1 pixel neighbors around the coarse best
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
            
            // why: clip to [-4, 4] to stay within reference image valid region
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
        
        return (bestFineDx, bestFineDy, bestFineSad)
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
