import XCTest
@testable import vevc

final class IntraPredictorTests: XCTestCase {

    func testModeDC() {
        let width = 4
        let height = 4
        var block = [Int16](repeating: 0, count: width * height)
        let top: [Int16] = [10, 20, 30, 40]
        let left: [Int16] = [5, 15, 25, 35]

        // DC should be average of top and left = (10+20+30+40 + 5+15+25+35) / 8 = 180 / 8 = 22.5 => 22
        intraPredictorPredict(mode: .dc, block: &block, width: width, height: height, top: top, left: left)

        for i in 0..<16 {
            XCTAssertEqual(block[i], 22, "DC mode should fill block with average 22")
        }
    }

    func testModeVertical() {
        let width = 4
        let height = 4
        var block = [Int16](repeating: 0, count: width * height)
        let top: [Int16] = [10, 20, 30, 40]
        let left: [Int16] = [5, 15, 25, 35]

        // Vertical should copy top row down
        intraPredictorPredict(mode: .vertical, block: &block, width: width, height: height, top: top, left: left)

        for y in 0..<height {
            for x in 0..<width {
                XCTAssertEqual(block[y * width + x], top[x], "Vertical mode should copy top pixel")
            }
        }
    }

    func testModeHorizontal() {
        let width = 4
        let height = 4
        var block = [Int16](repeating: 0, count: width * height)
        let top: [Int16] = [10, 20, 30, 40]
        let left: [Int16] = [5, 15, 25, 35]

        // Horizontal should copy left column right
        intraPredictorPredict(mode: .horizontal, block: &block, width: width, height: height, top: top, left: left)

        for y in 0..<height {
            for x in 0..<width {
                XCTAssertEqual(block[y * width + x], left[y], "Horizontal mode should copy left pixel")
            }
        }
    }

    func testModePlanar() {
        let width = 4
        let height = 4
        var block = [Int16](repeating: 0, count: width * height)
        let top: [Int16] = [10, 20, 30, 40]
        let left: [Int16] = [5, 15, 25, 35]

        // Planar (TrueMotion) simplified prediction
        // p[x,y] = top[x] + left[y] - top_left
        // Let's assume top_left is implicitly derived or provided.
        // For simplicity, let top_left be top[0] + left[0] / 2 or we can pass it.
        // Let's pass topLeft explicitly to the predictor.
        let topLeft: Int16 = 8
        intraPredictorPredict(mode: .planar, block: &block, width: width, height: height, top: top, left: left, topLeft: topLeft)

        XCTAssertEqual(block[0 * width + 0], 10 + 5 - 8)
        XCTAssertEqual(block[3 * width + 3], 40 + 35 - 8)
    }
}
