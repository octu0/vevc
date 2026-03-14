import XCTest
@testable import vevc

final class DWT2DTests: XCTestCase {

    func testDWT2DRoundtripSize8() {
        verifyRoundtrip(size: 8)
    }

    func testDWT2DRoundtripSize16() {
        verifyRoundtrip(size: 16)
    }

    func testDWT2DRoundtripSize32() {
        verifyRoundtrip(size: 32)
    }


    private func verifyRoundtrip(size: Int) {
        var block = Block2D(width: size, height: size)

        // Fill with random values
        for y in 0..<size {
            for x in 0..<size {
                block.data[y * size + x] = Int16.random(in: -1000...1000)
            }
        }

        let originalData = block.data

        block.withView { view in
            _ = dwt2d(&view, size: size)
        }

        // Ensure data has changed (transform actually happened)
        XCTAssertNotEqual(originalData, block.data, "DWT should have modified the data for size \(size)")

        block.withView { view in
            invDwt2d(&view, size: size)
        }

        XCTAssertEqual(originalData, block.data, "Inverse DWT should perfectly reconstruct the original data for size \(size)")
    }
}
