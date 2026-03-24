import XCTest
@testable import vevc

final class ImageTests: XCTestCase {

    func testBoundaryRepeat() {
        let width = 4
        let height = 4

        // 1. Within bounds
        // (0, 0) -> (0, 0)
        XCTAssertEqual(boundaryRepeat(width, height, 0, 0).0, 0)
        XCTAssertEqual(boundaryRepeat(width, height, 0, 0).1, 0)
        // (3, 3) -> (3, 3)
        XCTAssertEqual(boundaryRepeat(width, height, 3, 3).0, 3)
        XCTAssertEqual(boundaryRepeat(width, height, 3, 3).1, 3)

        // 2. Positive boundary (Width)
        // x = 4 (width) -> (4 - 1 - (4 - 4)) = 3
        XCTAssertEqual(boundaryRepeat(width, height, 4, 0).0, 3)
        // x = 5 (width + 1) -> (4 - 1 - (5 - 4)) = 2
        XCTAssertEqual(boundaryRepeat(width, height, 5, 0).0, 2)
        // x = 7 (2 * width - 1) -> (4 - 1 - (7 - 4)) = 0
        XCTAssertEqual(boundaryRepeat(width, height, 7, 0).0, 0)
        // x = 8 (2 * width) -> (4 - 1 - (8 - 4)) = -1 -> Clamp 0
        XCTAssertEqual(boundaryRepeat(width, height, 8, 0).0, 0)
        // x = 100 (Large) -> Clamp 0
        XCTAssertEqual(boundaryRepeat(width, height, 100, 0).0, 0)

        // 3. Negative boundary (Width)
        // x = -1 -> abs(-1) = 1
        XCTAssertEqual(boundaryRepeat(width, height, -1, 0).0, 1)
        // x = -3 -> abs(-3) = 3
        XCTAssertEqual(boundaryRepeat(width, height, -3, 0).0, 3)
        // x = -4 -> abs(-4) = 4 -> Clamp 3
        XCTAssertEqual(boundaryRepeat(width, height, -4, 0).0, 3)
        // x = -100 (Large negative) -> Clamp 3
        XCTAssertEqual(boundaryRepeat(width, height, -100, 0).0, 3)

        // 4. Height boundary (similar to width)
        XCTAssertEqual(boundaryRepeat(width, height, 0, 4).1, 3)
        XCTAssertEqual(boundaryRepeat(width, height, 0, 5).1, 2)
        XCTAssertEqual(boundaryRepeat(width, height, 0, -1).1, 1)
        XCTAssertEqual(boundaryRepeat(width, height, 0, -4).1, 3)

        // 5. Edge case: 1x1 image
        let w1 = 1
        let h1 = 1
        XCTAssertEqual(boundaryRepeat(w1, h1, 0, 0).0, 0)
        XCTAssertEqual(boundaryRepeat(w1, h1, 1, 1).0, 0)
        XCTAssertEqual(boundaryRepeat(w1, h1, -1, -1).0, 0)
    }

}
