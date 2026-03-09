import XCTest
@testable import vevc

final class DWTTests: XCTestCase {
    func testDWT2DRoundtripSize8() {
        self.performRoundtrip(size: 8)
    }

    func testDWT2DRoundtripSize16() {
        self.performRoundtrip(size: 16)
    }

    func testDWT2DRoundtripSize32() {
        self.performRoundtrip(size: 32)
    }

    func testDWT2DRoundtripSize4() {
        self.performRoundtrip(size: 4)
    }

    private func performRoundtrip(size: Int) {
        var block: Block2D = Block2D(width: size, height: size)
        for i: Int in 0..<block.data.count {
            block.data[i + 0] = Int16.random(in: -512...511)
        }
        let originalData: [Int16] = block.data

        block.withView { (view: inout BlockView) in
            _ = dwt2d(&view, size: size)
        }

        block.withView { (view: inout BlockView) in
            invDwt2d(&view, size: size)
        }

        XCTAssertEqual(block.data, originalData, "Roundtrip failed for size \(size)")
    }
}
