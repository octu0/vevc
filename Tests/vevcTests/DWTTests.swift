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

    func testDWT2DRoundtrip() {
        let sizes = [8, 16, 32]

        for size in sizes {
            var block = Block2D(width: size, height: size)
            for y in 0..<size {
                for x in 0..<size {
                    let index = (y * size) + x
                    block.data[index] = Int16.random(in: (-1 * 1000)...1000)
                }
            }

            let originalData = block.data

            block.withView { view in
                _ = dwt2d(&view, size: size)
                invDwt2d(&view, size: size)
            }

            XCTAssertEqual(block.data, originalData, "Roundtrip failed for size \(size)")
        }
    }

    func testDWT2DSIMDMatchesScalar() {
        #if arch(arm64) || arch(x86_64) || arch(wasm32)
        let sizes = [8, 16, 32]

        for size in sizes {
            var blockScalar = Block2D(width: size, height: size)
            for y in 0..<size {
                for x in 0..<size {
                    let index = (y * size) + x
                    blockScalar.data[index] = Int16.random(in: (-1 * 1000)...1000)
                }
            }

            var blockSIMD = blockScalar

            blockScalar.withView { view in
                _ = dwt2dScalar(&view, size: size)
            }

            blockSIMD.withView { view in
                // dwt2d will pick SIMD path for 8, 16, 32
                _ = dwt2d(&view, size: size)
            }

            XCTAssertEqual(blockSIMD.data, blockScalar.data, "SIMD vs Scalar mismatch for dwt2d size \(size)")

            blockScalar.withView { view in
                invDwt2dScalar(&view, size: size)
            }

            blockSIMD.withView { view in
                invDwt2d(&view, size: size)
            }

            XCTAssertEqual(blockSIMD.data, blockScalar.data, "SIMD vs Scalar mismatch for invDwt2d size \(size)")
        }
        #endif
    }
}
