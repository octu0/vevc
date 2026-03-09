import XCTest
@testable import vevc

final class DWTTests: XCTestCase {

    func testDWT2DRoundtrip() {
        let sizes = [4, 8, 16, 32]

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
