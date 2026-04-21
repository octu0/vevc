import XCTest
@testable import vevc

final class DeblockingFilterTests: XCTestCase {
    
    // Helper to print a 1D slice of 1D array for visualization
    private func printRow(plane: [Int16], width: Int, row: Int, colStart: Int, colEnd: Int) {
        var str = ""
        for x in colStart...colEnd {
            str += "\(plane[row * width + x]) "
        }
        print("Row \(row): [\(str)]")
    }
    
    func testVerticalEdgeSmoothing() {
        let width = 64
        let height = 32
        var plane = [Int16](repeating: 100, count: width * height) // Left Block is 100
        for y in 0..<height {
            for x in 32..<width {
                plane[y * width + x] = 110 // Right Block is 110
            }
        }
        
        let qStep = 48 // Threshold `tc=qStep/4=12` easily catches delta of 10
        applyDeblockingFilter32(plane: &plane, width: width, height: height, qStep: qStep)
        
        // After filtering, the boundary pixels (x=31 and x=32) should be smoothed
        // p0 is x=31, q0 is x=32
        let p0 = plane[0 * width + 31]
        let q0 = plane[0 * width + 32]
        
        print("Smoothed Vertical Edge p0: \(p0), q0: \(q0)")
        
        XCTAssertTrue(100 < p0, "Left boundary pixel should be pulled towards right")
        XCTAssertTrue(q0 < 110, "Right boundary pixel should be pulled towards left")
        XCTAssertTrue(abs(p0 - q0) < 10, "Edge contrast should be reduced")
    }
    
    func testVerticalEdgePreservation() {
        let width = 64
        let height = 32
        var plane = [Int16](repeating: 100, count: width * height) // Left Block is 100
        for y in 0..<height {
            for x in 32..<width {
                plane[y * width + x] = 200 // Right Block is 200 (True Image Edge)
            }
        }
        
        let qStep = 48
        applyDeblockingFilter32(plane: &plane, width: width, height: height, qStep: qStep)
        
        // After filtering, the boundary pixels should be untouched because delta 100 > threshold
        let p0 = plane[0 * width + 31]
        let q0 = plane[0 * width + 32]
        
        print("Preserved Vertical Edge p0: \(p0), q0: \(q0)")
        
        XCTAssertEqual(p0, 100, "True edge left pixel should be untouched")
        XCTAssertEqual(q0, 200, "True edge right pixel should be untouched")
    }
    
    func testHorizontalEdgeSmoothing() {
        let width = 32
        let height = 64
        var plane = [Int16](repeating: 100, count: width * height) // Top Block is 100
        for y in 32..<height {
            for x in 0..<width {
                plane[y * width + x] = 110 // Bottom Block is 110
            }
        }
        
        let qStep = 48
        applyDeblockingFilter32(plane: &plane, width: width, height: height, qStep: qStep)
        
        // After filtering, the boundary pixels (y=31 and y=32) should be smoothed
        // p0 is y=31, q0 is y=32
        let p0 = plane[31 * width + 0]
        let q0 = plane[32 * width + 0]
        
        print("Smoothed Horizontal Edge p0: \(p0), q0: \(q0)")
        
        XCTAssertTrue(100 < p0, "Top boundary pixel should be pulled towards bottom")
        XCTAssertTrue(q0 < 110, "Bottom boundary pixel should be pulled towards top")
        XCTAssertTrue(abs(p0 - q0) < 10, "Edge contrast should be reduced")
    }
}
