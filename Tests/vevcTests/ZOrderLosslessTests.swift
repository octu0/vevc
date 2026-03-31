import XCTest
@testable import vevc

final class ZOrderLosslessTests: XCTestCase {
    func testZOrderBijection() {
        func checkBijection(coords: [(x: Int, y: Int)], size: Int) {
            XCTAssertEqual(coords.count, size * size)
            var seen = Set<String>()
            for (x, y) in coords {
                XCTAssertTrue(x >= 0 && x < size)
                XCTAssertTrue(y >= 0 && y < size)
                let key = "\(x),\(y)"
                XCTAssertFalse(seen.contains(key))
                seen.insert(key)
            }
            XCTAssertEqual(seen.count, size * size)
        }
        
        checkBijection(coords: ZOrder.coords4, size: 4)
        checkBijection(coords: ZOrder.coords8, size: 8)
        checkBijection(coords: ZOrder.coords16, size: 16)
        checkBijection(coords: ZOrder.coords32, size: 32)
    }
}
