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
            block.base[i + 0] = Int16.random(in: -512...511)
        }
        let originalData: [Int16] = Array(block.data)

        block.withView { (view: inout BlockView) in
            switch size {
            case 8:
                dwt2d_8(&view)
                invDwt2d_8(&view)
            case 16:
                dwt2d_16(&view)
                invDwt2d_16(&view)
            case 32:
                dwt2d_32(&view)
                invDwt2d_32(&view)
            default:
                XCTFail("Unsupported size: \(size)")
            }
        }

        XCTAssertEqual(Array(block.data), originalData, "Roundtrip failed for size \(size)")
    }

    func testDWT2DRoundtrip() {
        let sizes = [8, 16, 32]

        for size in sizes {
            var block = Block2D(width: size, height: size)
            for y in 0..<size {
                for x in 0..<size {
                    let index = (y * size) + x
                    block.base[index] = Int16.random(in: (-1 * 1000)...1000)
                }
            }

            let originalData = Array(block.data)

            block.withView { view in
                _ = dwt2d(&view, size: size)
                invDwt2d(&view, size: size)
            }

            XCTAssertEqual(Array(block.data), originalData, "Roundtrip failed for size \(size)")
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
                    blockScalar.base[index] = Int16.random(in: (-1 * 1000)...1000)
                }
            }

            var blockSIMD = Block2D(width: size, height: size)
            blockSIMD.setData(Array(blockScalar.data))

            blockScalar.withView { view in
                _ = dwt2dScalar(&view, size: size)
            }

            blockSIMD.withView { view in
                // dwt2d will pick SIMD path for 8, 16, 32
                _ = dwt2d(&view, size: size)
            }

            XCTAssertEqual(Array(blockSIMD.data), Array(blockScalar.data), "SIMD vs Scalar mismatch for dwt2d size \(size)")

            blockScalar.withView { view in
                invDwt2dScalar(&view, size: size)
            }

            blockSIMD.withView { view in
                invDwt2d(&view, size: size)
            }

            XCTAssertEqual(Array(blockSIMD.data), Array(blockScalar.data), "SIMD vs Scalar mismatch for invDwt2d size \(size)")
        }
        #endif
    }

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
            block.base[i] = Int16.random(in: (-1 * 1000)...1000)
        }

        let originalData = Array(block.data)

        block.withView { view in
            switch size {
            case 8:
                dwt2d_8(&view)
            case 16:
                dwt2d_16(&view)
            case 32:
                dwt2d_32(&view)
            default:
                XCTFail("Unsupported size: \(size)")
            }
        }

        block.withView { view in
            switch size {
            case 8:
                invDwt2d_8(&view)
            case 16:
                invDwt2d_16(&view)
            case 32:
                invDwt2d_32(&view)
            default:
                XCTFail("Unsupported size: \(size)")
            }
        }

        XCTAssertEqual(Array(block.data), originalData, "Roundtrip failed for size \(size)")
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
                switch size {
                case 8:
                    lift53_8(ptr, stride: 1)
                    invLift53_8(ptr, stride: 1)
                case 16:
                    lift53_16(ptr, stride: 1)
                    invLift53_16(ptr, stride: 1)
                case 32:
                    lift53_32(ptr, stride: 1)
                    invLift53_32(ptr, stride: 1)
                default:
                    XCTFail("Unsupported size: \(size)")
                }
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
                    block.base[(y * size) + x] = Int16.random(in: -1000...1000)
                }
            }

            let originalData = Array(block.data)

            block.withView { view in
                switch size {
                case 8:
                    dwt2d_8(&view)
                    invDwt2d_8(&view)
                case 16:
                    dwt2d_16(&view)
                    invDwt2d_16(&view)
                case 32:
                    dwt2d_32(&view)
                    invDwt2d_32(&view)
                default:
                    XCTFail("Unsupported size: \(size)")
                }
            }

            XCTAssertEqual(Array(block.data), originalData, "Failed for block size \(size)")
        }
    }
    
    #if arch(arm64) || arch(x86_64) || arch(wasm32)
    func testLift53SIMDLossless() {
        let sizes: [Int] = [8, 16, 32]

        for size in sizes {
            var data = [Int16](repeating: 0, count: size)
            for i in 0..<size {
                data[i + 0] = Int16.random(in: (-1 * 1000)...1000)
            }

            let original = data

            data.withUnsafeMutableBufferPointer { ptr in
                switch size {
                case 8:
                    lift53_8(ptr, stride: 1)
                    invLift53_8(ptr, stride: 1)
                case 16:
                    lift53_16(ptr, stride: 1)
                    invLift53_16(ptr, stride: 1)
                case 32:
                    lift53_32(ptr, stride: 1)
                    invLift53_32(ptr, stride: 1)
                default:
                    XCTFail("Unsupported size: \(size)")
                }
            }

            XCTAssertEqual(data, original, "SIMD failed for size \(size)")
        }
    }
    #endif
}
