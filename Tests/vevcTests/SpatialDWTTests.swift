import XCTest
@testable import vevc

final class SpatialDWTTests: XCTestCase {

    func testSpatialDWTRoundTrip8() {
        runRoundTripTest(size: 8)
    }

    func testSpatialDWTRoundTrip16() {
        runRoundTripTest(size: 16)
    }

    func testSpatialDWTRoundTrip32() {
        runRoundTripTest(size: 32)
    }

    private func runRoundTripTest(size: Int) {
        var block = Block2D(width: size, height: size)
        for i in 0..<block.data.count {
            block.data[i] = Int16.random(in: -1000...1000)
        }

        let originalData = block.data

        block.withView { view in
            _ = dwt2d(&view, size: size)
        }

        block.withView { view in
            invDwt2d(&view, size: size)
        }

        XCTAssertEqual(block.data, originalData, "Roundtrip failed for size \(size)")
    }
}
