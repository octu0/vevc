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

    func testLift53Lossless() {
        let sizes: [Int] = [8, 16, 32] // SIMD and scalar sizes

        for size in sizes {
            var data = [Int16](repeating: 0, count: size)
            for i in 0..<size {
                data[i] = Int16.random(in: -1000...1000)
            }

            let original = data

            data.withUnsafeMutableBufferPointer { ptr in
                lift53(ptr, count: size, stride: 1)
                invLift53(ptr, count: size, stride: 1)
            }

            XCTAssertEqual(data, original, "Failed for size \(size)")
        }
    }

    func testDWT2DLossless() {
        let sizes: [Int] = [8, 16, 32] // SIMD and scalar sizes

        for size in sizes {
            var block = Block2D(width: size, height: size)
            for y in 0..<size {
                for x in 0..<size {
                    block.data[(y * size) + x] = Int16.random(in: -1000...1000)
                }
            }

            let originalData = block.data

            block.withView { view in
                _ = dwt2d(&view, size: size)
                invDwt2d(&view, size: size)
            }

            XCTAssertEqual(block.data, originalData, "Failed for block size \(size)")
        }
    }
}
