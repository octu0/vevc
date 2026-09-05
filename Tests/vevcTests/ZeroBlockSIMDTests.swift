import XCTest
@testable import vevc

final class ZeroBlockSIMDTests: XCTestCase {
    private func isZeroBlockScalarOracle(view: BlockView) -> Bool {
        let ptr = view.base
        let w = view.width
        let h = view.height
        let s = view.stride
        for y in 0..<h {
            let row = ptr.advanced(by: y * s)
            for x in 0..<w {
                if row[x] != 0 { return false }
            }
        }
        return true
    }

    func testZeroBlockExactZeroAndPatterns() {
        let sizes = [8, 16, 32]

        for size in sizes {
            var buffer = [Int16](repeating: 0, count: size * size)
            buffer.withUnsafeMutableBufferPointer { ptr in
                let view = BlockView(base: ptr.baseAddress!, width: size, height: size, stride: size)

                // 1. All zero
                XCTAssertTrue(isZeroBlock(view: view))
                XCTAssertEqual(isZeroBlock(view: view), isZeroBlockScalarOracle(view: view))

                // 2. First pixel non-zero
                ptr[0] = 1
                XCTAssertFalse(isZeroBlock(view: view))
                XCTAssertEqual(isZeroBlock(view: view), isZeroBlockScalarOracle(view: view))
                ptr[0] = 0

                // 3. Middle pixel non-zero
                let mid = (size / 2) * size + (size / 2)
                ptr[mid] = -5
                XCTAssertFalse(isZeroBlock(view: view))
                XCTAssertEqual(isZeroBlock(view: view), isZeroBlockScalarOracle(view: view))
                ptr[mid] = 0

                // 4. Last pixel non-zero
                let last = size * size - 1
                ptr[last] = 100
                XCTAssertFalse(isZeroBlock(view: view))
                XCTAssertEqual(isZeroBlock(view: view), isZeroBlockScalarOracle(view: view))
                ptr[last] = 0

                // 5. Sweep every single element
                for i in 0..<(size * size) {
                    ptr[i] = 42
                    XCTAssertFalse(isZeroBlock(view: view))
                    XCTAssertEqual(isZeroBlock(view: view), isZeroBlockScalarOracle(view: view))
                    ptr[i] = 0
                }
            }
        }
    }

    func testZeroBlockRandomTiles() {
        var rng = UInt64(123456789)
        func nextRand() -> UInt64 {
            rng ^= rng &<< 13
            rng ^= rng &>> 7
            rng ^= rng &<< 17
            return rng
        }

        let sizes = [8, 16, 32]
        for size in sizes {
            let count = size * size
            var buffer = [Int16](repeating: 0, count: count)
            buffer.withUnsafeMutableBufferPointer { ptr in
                let view = BlockView(base: ptr.baseAddress!, width: size, height: size, stride: size)

                for _ in 0..<200 {
                    // Generate sparse or dense random patterns
                    let nonZeroCount = Int(nextRand() % UInt64(count + 1))
                    for i in 0..<count { ptr[i] = 0 }
                    for _ in 0..<nonZeroCount {
                        let idx = Int(nextRand() % UInt64(count))
                        var val = Int16(truncatingIfNeeded: Int64(nextRand() % 1000) - 500)
                        if val == 0 {
                            val = 1
                        }
                        ptr[idx] = val
                    }

                    let actual = isZeroBlock(view: view)
                    let expected = isZeroBlockScalarOracle(view: view)
                    XCTAssertEqual(actual, expected)
                }
            }
        }
    }

    func testZeroBlockNonContiguousFallback() {
        let width = 16
        let height = 16
        let stride = 32
        var buffer = [Int16](repeating: 0, count: stride * height)

        buffer.withUnsafeMutableBufferPointer { ptr in
            let view = BlockView(base: ptr.baseAddress!, width: width, height: height, stride: stride)

            XCTAssertTrue(isZeroBlock(view: view))
            XCTAssertEqual(isZeroBlock(view: view), isZeroBlockScalarOracle(view: view))

            // Non-zero inside active block
            ptr[5 * stride + 7] = 99
            XCTAssertFalse(isZeroBlock(view: view))
            XCTAssertEqual(isZeroBlock(view: view), isZeroBlockScalarOracle(view: view))
            ptr[5 * stride + 7] = 0

            // Non-zero in padding (outside width, inside stride) -> should still be considered zero
            ptr[5 * stride + 20] = 99
            XCTAssertTrue(isZeroBlock(view: view))
            XCTAssertEqual(isZeroBlock(view: view), isZeroBlockScalarOracle(view: view))
        }
    }
}
