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
    /// 12-point (4 corners of HL, LH, HH) SAD computation for Coarse Search & Early Exit.
    /// Fully unrolled to avoid loops and branches in the critical path.
    @inline(__always)
    static func compute12PointSAD(cView: BlockView, pView: BlockView) -> Int {
        var sad = 0
        let cBase = cView.base
        let pBase = pView.base
        
        // HL Corners
        sad += abs(Int(cBase[4]) - Int(pBase[4]))
        sad += abs(Int(cBase[7]) - Int(pBase[7]))
        sad += abs(Int(cBase[28]) - Int(pBase[28]))
        sad += abs(Int(cBase[31]) - Int(pBase[31]))
        
        // LH Corners
        sad += abs(Int(cBase[32]) - Int(pBase[32]))
        sad += abs(Int(cBase[35]) - Int(pBase[35]))
        sad += abs(Int(cBase[56]) - Int(pBase[56]))
        sad += abs(Int(cBase[59]) - Int(pBase[59]))
        
        // HH Corners
        sad += abs(Int(cBase[36]) - Int(pBase[36]))
        sad += abs(Int(cBase[39]) - Int(pBase[39]))
        sad += abs(Int(cBase[60]) - Int(pBase[60]))
        sad += abs(Int(cBase[63]) - Int(pBase[63]))
        
        return sad
    }

    /// 48-point (full HL, LH, HH subbands) SAD computation for Fine Search.
    /// Fully unrolled to avoid loops and branches in the critical path.
    @inline(__always)
    static func compute48PointSAD(cView: BlockView, pView: BlockView) -> Int {
        var sad = 0
        let cBase = cView.base
        let pBase = pView.base
        
        // HL (右上)
        sad += abs(Int(cBase[4]) - Int(pBase[4]))
        sad += abs(Int(cBase[5]) - Int(pBase[5]))
        sad += abs(Int(cBase[6]) - Int(pBase[6]))
        sad += abs(Int(cBase[7]) - Int(pBase[7]))
        sad += abs(Int(cBase[12]) - Int(pBase[12]))
        sad += abs(Int(cBase[13]) - Int(pBase[13]))
        sad += abs(Int(cBase[14]) - Int(pBase[14]))
        sad += abs(Int(cBase[15]) - Int(pBase[15]))
        sad += abs(Int(cBase[20]) - Int(pBase[20]))
        sad += abs(Int(cBase[21]) - Int(pBase[21]))
        sad += abs(Int(cBase[22]) - Int(pBase[22]))
        sad += abs(Int(cBase[23]) - Int(pBase[23]))
        sad += abs(Int(cBase[28]) - Int(pBase[28]))
        sad += abs(Int(cBase[29]) - Int(pBase[29]))
        sad += abs(Int(cBase[30]) - Int(pBase[30]))
        sad += abs(Int(cBase[31]) - Int(pBase[31]))
        
        // LH (左下)
        sad += abs(Int(cBase[32]) - Int(pBase[32]))
        sad += abs(Int(cBase[33]) - Int(pBase[33]))
        sad += abs(Int(cBase[34]) - Int(pBase[34]))
        sad += abs(Int(cBase[35]) - Int(pBase[35]))
        sad += abs(Int(cBase[40]) - Int(pBase[40]))
        sad += abs(Int(cBase[41]) - Int(pBase[41]))
        sad += abs(Int(cBase[42]) - Int(pBase[42]))
        sad += abs(Int(cBase[43]) - Int(pBase[43]))
        sad += abs(Int(cBase[48]) - Int(pBase[48]))
        sad += abs(Int(cBase[49]) - Int(pBase[49]))
        sad += abs(Int(cBase[50]) - Int(pBase[50]))
        sad += abs(Int(cBase[51]) - Int(pBase[51]))
        sad += abs(Int(cBase[56]) - Int(pBase[56]))
        sad += abs(Int(cBase[57]) - Int(pBase[57]))
        sad += abs(Int(cBase[58]) - Int(pBase[58]))
        sad += abs(Int(cBase[59]) - Int(pBase[59]))
        
        // HH (右下)
        sad += abs(Int(cBase[36]) - Int(pBase[36]))
        sad += abs(Int(cBase[37]) - Int(pBase[37]))
        sad += abs(Int(cBase[38]) - Int(pBase[38]))
        sad += abs(Int(cBase[39]) - Int(pBase[39]))
        sad += abs(Int(cBase[44]) - Int(pBase[44]))
        sad += abs(Int(cBase[45]) - Int(pBase[45]))
        sad += abs(Int(cBase[46]) - Int(pBase[46]))
        sad += abs(Int(cBase[47]) - Int(pBase[47]))
        sad += abs(Int(cBase[52]) - Int(pBase[52]))
        sad += abs(Int(cBase[53]) - Int(pBase[53]))
        sad += abs(Int(cBase[54]) - Int(pBase[54]))
        sad += abs(Int(cBase[55]) - Int(pBase[55]))
        sad += abs(Int(cBase[60]) - Int(pBase[60]))
        sad += abs(Int(cBase[61]) - Int(pBase[61]))
        sad += abs(Int(cBase[62]) - Int(pBase[62]))
        sad += abs(Int(cBase[63]) - Int(pBase[63]))
        
        return sad
    }
    
    @inline(__always)
    static func fetchAndTransformInnerBlock8(plane: UnsafePointer<Int16>, width: Int, x: Int, y: Int, view: inout BlockView) {
        for i in 0..<8 {
            let srcRow = plane.advanced(by: (y + i) * width + x)
            let dstRow = view.rowPointer(y: i)
            dstRow.update(from: srcRow, count: 8)
        }
        dwt2d_8(&view)
    }
    
    @inline(__always)
    static func fetchAndTransformEdgeBlock8(plane: UnsafePointer<Int16>, width: Int, height: Int, x: Int, y: Int, view: inout BlockView) {
        for i in 0..<8 {
            let clampedY = min(max(0, y + i), height - 1)
            let dstRow = view.rowPointer(y: i)
            for j in 0..<8 {
                let clampedX = min(max(0, x + j), width - 1)
                dstRow[j] = plane[(clampedY * width) + clampedX]
            }
        }
        dwt2d_8(&view)
    }

    static func search(currBlock: inout Block2D, prevPlane: [Int16], width: Int, height: Int, bx: Int, by: Int, range: Int = 4) -> (MotionVector, Block2D, Int) {
        let isInner = (bx - range >= 0) && (bx + range + 8 <= width) && (by - range >= 0) && (by + range + 8 <= height)
        if isInner {
            return searchInner(currBlock: &currBlock, prevPlane: prevPlane, width: width, height: height, bx: bx, by: by, range: range)
        } else {
            return searchEdge(currBlock: &currBlock, prevPlane: prevPlane, width: width, height: height, bx: bx, by: by, range: range)
        }
    }

    private static func searchInner(currBlock: inout Block2D, prevPlane: [Int16], width: Int, height: Int, bx: Int, by: Int, range: Int = 4) -> (MotionVector, Block2D, Int) {
        var origShiftedBlock = Block2D(width: 8, height: 8)
        var tempBlock = Block2D(width: 8, height: 8)
        
        return prevPlane.withUnsafeBufferPointer { prevBuf in
            guard let prevBase = prevBuf.baseAddress else { return (MotionVector(dx: 0, dy: 0), currBlock, 0) }
            
            // 0. Extract Original Block at (0,0) shift
            origShiftedBlock.withView { oView in
                fetchAndTransformInnerBlock8(plane: prevBase, width: width, x: bx, y: by, view: &oView)
            }
            
            let fullSad = currBlock.withView { cView -> Int in
                var finalSad = 0
                origShiftedBlock.withView { oView in
                    // 1. Early Exit Check using 12-point feature SAD
                    let zeroSad12 = compute12PointSAD(cView: cView, pView: oView)
                    if zeroSad12 < 80 {
                        finalSad = compute48PointSAD(cView: cView, pView: oView)
                    } else {
                        finalSad = -1 // flag to continue search
                    }
                }
                return finalSad
            }
            if fullSad != -1 {
                return (MotionVector(dx: 0, dy: 0), origShiftedBlock, fullSad)
            }
            
            // 2. Coarse Search using 12-point SAD
            let step = 2
            var bestCoarseSad = Int.max
            var bestCoarseDx: Int = 0
            var bestCoarseDy: Int = 0
            var bestCoarseBlock = origShiftedBlock // initialize with 0,0 block
            
            currBlock.withView { cView in
                origShiftedBlock.withView { oView in
                    bestCoarseSad = compute12PointSAD(cView: cView, pView: oView)
                }
                for dy in stride(from: -range, through: range, by: step) {
                    for dx in stride(from: -range, through: range, by: step) {
                        if dx == 0 && dy == 0 { continue }
                        let targetX = bx + dx
                        let targetY = by + dy
                        
                        var sad12 = Int.max
                        tempBlock.withView { tView in
                            fetchAndTransformInnerBlock8(plane: prevBase, width: width, x: targetX, y: targetY, view: &tView)
                            sad12 = compute12PointSAD(cView: cView, pView: tView)
                        }
                        if sad12 < bestCoarseSad {
                            bestCoarseSad = sad12
                            bestCoarseDx = dx
                            bestCoarseDy = dy
                            bestCoarseBlock.data = tempBlock.data
                        }
                    }
                }
            }
            
            // 3. Fine Search using full 48-point SAD
            var bestFineSad = Int.max
            var bestFineDx = bestCoarseDx
            var bestFineDy = bestCoarseDy
            var bestFineBlock = bestCoarseBlock // initialize with coarse best
            
            currBlock.withView { cView in
                bestCoarseBlock.withView { bcView in
                    bestFineSad = compute48PointSAD(cView: cView, pView: bcView)
                }
                
                for fy in -1...1 {
                    for fx in -1...1 {
                        if fx == 0 && fy == 0 { continue }
                        let fineDx = bestCoarseDx + fx
                        let fineDy = bestCoarseDy + fy
                        
                        if fineDx < -range || fineDx > range || fineDy < -range || fineDy > range { continue }
                        
                        let targetX = bx + fineDx
                        let targetY = by + fineDy
                        
                        var sad48 = Int.max
                        tempBlock.withView { tView in
                            fetchAndTransformInnerBlock8(plane: prevBase, width: width, x: targetX, y: targetY, view: &tView)
                            sad48 = compute48PointSAD(cView: cView, pView: tView)
                        }
                        if sad48 < bestFineSad {
                            bestFineSad = sad48
                            bestFineDx = fineDx
                            bestFineDy = fineDy
                            bestFineBlock.data = tempBlock.data
                        }
                    }
                }
            }
            
            return (MotionVector(dx: Int16(bestFineDx), dy: Int16(bestFineDy)), bestFineBlock, bestFineSad)
        }
    }

    private static func searchEdge(currBlock: inout Block2D, prevPlane: [Int16], width: Int, height: Int, bx: Int, by: Int, range: Int = 4) -> (MotionVector, Block2D, Int) {
        var origShiftedBlock = Block2D(width: 8, height: 8)
        var tempBlock = Block2D(width: 8, height: 8)
        
        return prevPlane.withUnsafeBufferPointer { prevBuf in
            guard let prevBase = prevBuf.baseAddress else { return (MotionVector(dx: 0, dy: 0), currBlock, 0) }
            
            // 0. Extract Original Block at (0,0) shift
            origShiftedBlock.withView { oView in
                fetchAndTransformEdgeBlock8(plane: prevBase, width: width, height: height, x: bx, y: by, view: &oView)
            }
            
            let fullSad = currBlock.withView { cView -> Int in
                var finalSad = 0
                origShiftedBlock.withView { oView in
                    // 1. Early Exit Check using 12-point feature SAD
                    let zeroSad12 = compute12PointSAD(cView: cView, pView: oView)
                    if zeroSad12 < 80 {
                        finalSad = compute48PointSAD(cView: cView, pView: oView)
                    } else {
                        finalSad = -1 // flag to continue search
                    }
                }
                return finalSad
            }
            if fullSad != -1 {
                return (MotionVector(dx: 0, dy: 0), origShiftedBlock, fullSad)
            }
            
            // 2. Coarse Search using 12-point SAD
            let step = 2
            var bestCoarseSad = Int.max
            var bestCoarseDx: Int = 0
            var bestCoarseDy: Int = 0
            var bestCoarseBlock = origShiftedBlock // initialize with 0,0 block
            
            currBlock.withView { cView in
                origShiftedBlock.withView { oView in
                    bestCoarseSad = compute12PointSAD(cView: cView, pView: oView)
                }
                for dy in stride(from: -range, through: range, by: step) {
                    for dx in stride(from: -range, through: range, by: step) {
                        if dx == 0 && dy == 0 { continue }
                        let targetX = bx + dx
                        let targetY = by + dy
                        
                        var sad12 = Int.max
                        tempBlock.withView { tView in
                            fetchAndTransformEdgeBlock8(plane: prevBase, width: width, height: height, x: targetX, y: targetY, view: &tView)
                            sad12 = compute12PointSAD(cView: cView, pView: tView)
                        }
                        if sad12 < bestCoarseSad {
                            bestCoarseSad = sad12
                            bestCoarseDx = dx
                            bestCoarseDy = dy
                            bestCoarseBlock.data = tempBlock.data
                        }
                    }
                }
            }
            
            // 3. Fine Search using full 48-point SAD
            var bestFineSad = Int.max
            var bestFineDx = bestCoarseDx
            var bestFineDy = bestCoarseDy
            var bestFineBlock = bestCoarseBlock // initialize with coarse best
            
            currBlock.withView { cView in
                bestCoarseBlock.withView { bcView in
                    bestFineSad = compute48PointSAD(cView: cView, pView: bcView)
                }
                
                for fy in -1...1 {
                    for fx in -1...1 {
                        if fx == 0 && fy == 0 { continue }
                        let fineDx = bestCoarseDx + fx
                        let fineDy = bestCoarseDy + fy
                        
                        if fineDx < -range || fineDx > range || fineDy < -range || fineDy > range { continue }
                        
                        let targetX = bx + fineDx
                        let targetY = by + fineDy
                        
                        var sad48 = Int.max
                        tempBlock.withView { tView in
                            fetchAndTransformEdgeBlock8(plane: prevBase, width: width, height: height, x: targetX, y: targetY, view: &tView)
                            sad48 = compute48PointSAD(cView: cView, pView: tView)
                        }
                        if sad48 < bestFineSad {
                            bestFineSad = sad48
                            bestFineDx = fineDx
                            bestFineDy = fineDy
                            bestFineBlock.data = tempBlock.data
                        }
                    }
                }
            }
            
            return (MotionVector(dx: Int16(bestFineDx), dy: Int16(bestFineDy)), bestFineBlock, bestFineSad)
        }
    }

    @inline(__always)
    static func fetchAndTransformInnerBlock32(plane: UnsafePointer<Int16>, width: Int, x: Int, y: Int) -> Block2D {
        var block = Block2D(width: 32, height: 32)
        block.withView { view in
            for i in 0..<32 {
                let srcRow = plane.advanced(by: (y + i) * width + x)
                let dstRow = view.rowPointer(y: i)
                dstRow.update(from: srcRow, count: 32)
            }
            dwt2d_32(&view)
        }
        return block
    }

    @inline(__always)
    static func fetchAndTransformEdgeBlock32(plane: UnsafePointer<Int16>, width: Int, height: Int, x: Int, y: Int) -> Block2D {
        var block = Block2D(width: 32, height: 32)
        block.withView { view in
            for i in 0..<32 {
                let clampedY = min(max(0, y + i), height - 1)
                let dstRow = view.rowPointer(y: i)
                for j in 0..<32 {
                    let clampedX = min(max(0, x + j), width - 1)
                    dstRow[j] = plane[(clampedY * width) + clampedX]
                }
            }
            dwt2d_32(&view)
        }
        return block
    }

    @inline(__always)
    static func fetchAndTransformInnerBlock16(plane: UnsafePointer<Int16>, width: Int, x: Int, y: Int) -> Block2D {
        var block = Block2D(width: 16, height: 16)
        block.withView { view in
            for i in 0..<16 {
                let srcRow = plane.advanced(by: (y + i) * width + x)
                let dstRow = view.rowPointer(y: i)
                dstRow.update(from: srcRow, count: 16)
            }
            dwt2d_16(&view)
        }
        return block
    }

    @inline(__always)
    static func fetchAndTransformEdgeBlock16(plane: UnsafePointer<Int16>, width: Int, height: Int, x: Int, y: Int) -> Block2D {
        var block = Block2D(width: 16, height: 16)
        block.withView { view in
            for i in 0..<16 {
                let clampedY = min(max(0, y + i), height - 1)
                let dstRow = view.rowPointer(y: i)
                for j in 0..<16 {
                    let clampedX = min(max(0, x + j), width - 1)
                    dstRow[j] = plane[(clampedY * width) + clampedX]
                }
            }
            dwt2d_16(&view)
        }
        return block
    }
}
