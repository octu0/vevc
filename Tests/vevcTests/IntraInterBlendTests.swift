import XCTest
@testable import vevc

final class IntraInterBlendTests: XCTestCase {
    
    func testBlendSmoothsIntraInterBoundary() {
        let width = 64
        let height = 32
        var plane = [Int16](repeating: 100, count: width * height)
        // Left block (Inter) is 100, Right block (Intra) is 150
        for y in 0..<height {
            for x in 32..<width {
                plane[y * width + x] = 150
            }
        }
        
        var mvsDx: [Int16] = [1, 32767]
        var mvsDy: [Int16] = [1, 32767]
        let mvs = MotionVectors(dx: mvsDx, dy: mvsDy)
        
        blendIntraInterBoundaryLuma32(plane: &plane, mvs: mvs, width: width, height: height)
        
        // Check boundary pixels
        let p3 = plane[0 * width + 28]
        let p2 = plane[0 * width + 29]
        let p1 = plane[0 * width + 30]
        let p0 = plane[0 * width + 31]
        let q0 = plane[0 * width + 32]
        let q1 = plane[0 * width + 33]
        let q2 = plane[0 * width + 34]
        let q3 = plane[0 * width + 35]
        
        print("Blend Luma: p3:\(p3) p2:\(p2) p1:\(p1) p0:\(p0) | q0:\(q0) q1:\(q1) q2:\(q2) q3:\(q3)")
        
        XCTAssertTrue(100 < p0 && p0 < 150, "p0 should be blended")
        XCTAssertTrue(100 < q0 && q0 < 150, "q0 should be blended")
        XCTAssertTrue(p0 <= q0, "p0 should be smaller or equal to q0")
        XCTAssertTrue(p3 < p2 && p2 < p1 && p1 < p0, "Smooth gradient on Inter side")
        XCTAssertTrue(q0 < q1 && q1 < q2 && q2 < q3, "Smooth gradient on Intra side")
    }

    func testBlendSkipsInterInterBoundary() {
        let width = 64
        let height = 32
        var plane = [Int16](repeating: 100, count: width * height)
        for y in 0..<height {
            for x in 32..<width {
                plane[y * width + x] = 150
            }
        }
        
        let mvs = MotionVectors(dx: [1, 2], dy: [1, 2])
        
        blendIntraInterBoundaryLuma32(plane: &plane, mvs: mvs, width: width, height: height)
        
        let p0 = plane[0 * width + 31]
        let q0 = plane[0 * width + 32]
        XCTAssertEqual(p0, 100, "Should not be blended")
        XCTAssertEqual(q0, 150, "Should not be blended")
    }

    func testBlendSkipsIntraIntraBoundary() {
        let width = 64
        let height = 32
        var plane = [Int16](repeating: 100, count: width * height)
        for y in 0..<height {
            for x in 32..<width {
                plane[y * width + x] = 150
            }
        }
        
        let mvs = MotionVectors(dx: [32767, 32767], dy: [32767, 32767])
        
        blendIntraInterBoundaryLuma32(plane: &plane, mvs: mvs, width: width, height: height)
        
        let p0 = plane[0 * width + 31]
        let q0 = plane[0 * width + 32]
        XCTAssertEqual(p0, 100, "Should not be blended")
        XCTAssertEqual(q0, 150, "Should not be blended")
    }
}
