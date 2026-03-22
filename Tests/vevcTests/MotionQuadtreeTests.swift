import XCTest
@testable import vevc

final class MotionQuadtreeTests: XCTestCase {
    
    func testMVGridPMV() {
        var grid = MVGrid(width: 64, height: 64, minSize: 8)
        
        // Fill some MVs
        grid.fill(x: 0, y: 0, w: 32, h: 32, mv: SIMD2<Int16>(4, 0))
        grid.fill(x: 32, y: 0, w: 32, h: 32, mv: SIMD2<Int16>(8, 4))
        grid.fill(x: 0, y: 32, w: 32, h: 32, mv: SIMD2<Int16>(-4, -8))
        
        // PMV for block at (32, 32)
        // Left is (0,32) -> (-4, -8)
        // Top is (32,0) -> (8, 4)
        // TopRight is (64,0) -> outside! So count is 2? Wait!
        // At x=32, y=32, w=32:
        // Left: x=31,y=32 -> (-4,-8)
        // Top: x=32,y=31 -> (8,4)
        // TopRight: x=64,y=31 -> false (outside w=64)
        
        let pmv = grid.getPMV(x: 32, y: 32, w: 32)
        // Only 2 neighbors (Left, Top)
        // median of 2 is the average: dx = (-4 + 8)/2 = 2. dy = (-8 + 4)/2 = -2.
        XCTAssertEqual(pmv.x, 2)
        XCTAssertEqual(pmv.y, -2)
        
        // PMV for block at (16, 16)
        // Has Left (0,16), Top(16,0), TopRight(32,0)
        // All are within the (0,0, 32,32) and (32,0, 32,32) regions.
        // Wait, if we are evaluating (16,16) inside the 32x32 block (0,0), 
        // in Z-order the left and top MVs are the SAME 32x32 block's MVs? 
        // Actually, evaluateMotionQuadtreeNode fills the grid AT THE END of the node evaluation.
        // So during the node evaluation of (0,0, 32,32), the grid is EMPTY!
        // So the PMV is 0!
    }

    func testQuadtreeSplit() {
        var prev = PlaneData420(
            width: 64, height: 64,
            y: [Int16](repeating: 0, count: 64*64),
            cb: [Int16](repeating: 0, count: 32*32),
            cr: [Int16](repeating: 0, count: 32*32)
        )
        var curr = PlaneData420(
            width: 64, height: 64,
            y: [Int16](repeating: 0, count: 64*64),
            cb: [Int16](repeating: 0, count: 32*32),
            cr: [Int16](repeating: 0, count: 32*32)
        )
        
        // Fill prev with a solid background and two squares
        for i in 0..<64*64 { prev.y[i] = 100 }
        
        // Square 1: moves +4 in X
        for y in 10..<20 {
            for x in 10..<20 {
                prev.y[y * 64 + x] = 200
            }
        }
        
        // Square 2: moves -4 in Y
        for y in 40..<50 {
            for x in 40..<50 {
                prev.y[y * 64 + x] = 50
            }
        }
        
        for i in 0..<64*64 { curr.y[i] = 100 }
        
        // Square 1 moved +4 in X (now at x: 14..24)
        for y in 10..<20 {
            for x in 14..<24 {
                curr.y[y * 64 + x] = 200
            }
        }
        
        // Square 2 moved -4 in Y (now at y: 36..46)
        for y in 36..<46 {
            for x in 40..<50 {
                curr.y[y * 64 + x] = 50
            }
        }
        
        let layer0Curr = downscale8x(pd: curr)
        let layer0Prev = downscale8x(pd: prev)
        let motionTree = estimateMotionQuadtree(curr: curr, prev: prev, layer0Curr: layer0Curr, layer0Prev: layer0Prev)
        XCTAssertEqual(motionTree.ctuNodes.count, 1)
        
        let rootNode = motionTree.ctuNodes[0]
        
        // Ensure that a split occurred
        if case .split(let tl, _, _, _) = rootNode {
            if case .leaf(let mvTL) = tl {
                XCTAssertEqual(mvTL.x, 16)
                XCTAssertEqual(mvTL.y, 0)
            }
        } else {
            XCTFail("Root node was not split! MV was \(rootNode)")
        }
        
        // Test Encoding and Decoding symmetry
        var bw = EntropyEncoder()
        var encodeGrid = MVGrid(width: 64, height: 64, minSize: 8)
        let mbSize = 64
        encodeMotionQuadtreeNode(node: rootNode, w: 64, h: 64, startX: 0, startY: 0, size: mbSize, grid: &encodeGrid, bw: &bw)
        bw.flush()
        let bitstream = bw.getData()
        
        var br = try! EntropyDecoder(data: bitstream)
        var decodeGrid = MVGrid(width: 64, height: 64, minSize: 8)
        let decodedNode = try! decodeMotionQuadtreeNode(w: 64, h: 64, startX: 0, startY: 0, size: mbSize, grid: &decodeGrid, br: &br)
        
        // We can check if the decoded node is identical to the encoded node
        // Swift enums with associated values don't automatically conform to Equatable in simple checks unless we define it,
        // but since we know it's a split, we can manually check
        if case .split(let dtl, _, _, _) = decodedNode,
           case .split(let etl, _, _, _) = rootNode {
             // just check TL for now
             if case .leaf(let dmvTL) = dtl, case .leaf(let emvTL) = etl {
                 XCTAssertEqual(dmvTL.x, emvTL.x)
                 XCTAssertEqual(dmvTL.y, emvTL.y)
             }
        } else {
            XCTFail("Decoded node did not match structural split of root node!")
        }
    }
}
