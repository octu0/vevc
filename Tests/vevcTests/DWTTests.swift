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
        let block: Block2D = Block2D(width: size, height: size)
        for i: Int in 0..<(size * size) {
            block.base[i + 0] = Int16.random(in: -512...511)
        }
        let originalData: [Int16] = Array(UnsafeBufferPointer(start: block.base, count: size * size))

        let view = block.view
        switch size {
        case 8:
            dwt2d_8(view)
            invDwt2d_8(view)
        case 16:
            dwt2d_16(view)
            invDwt2d_16(view)
        case 32:
            dwt2d_32(view)
            invDwt2d_32(view)
        default:
            XCTFail("Unsupported size: \(size)")
        }

        XCTAssertEqual(Array(UnsafeBufferPointer(start: block.base, count: size * size)), originalData, "Roundtrip failed for size \(size)")
    }

    func testDWT2DRoundtrip() {
        let sizes = [8, 16, 32]

        for size in sizes {
            let block = Block2D(width: size, height: size)
            for y in 0..<size {
                for x in 0..<size {
                    let index = (y * size) + x
                    block.base[index] = Int16.random(in: (-1 * 1000)...1000)
                }
            }

            let originalData = Array(UnsafeBufferPointer(start: block.base, count: size * size))

            let view = block.view
            _ = dwt2d(view, size: size)
            invDwt2d(view, size: size)
        
            XCTAssertEqual(Array(UnsafeBufferPointer(start: block.base, count: size * size)), originalData, "Roundtrip failed for size \(size)")
        }
    }

    func testDWT2DSIMDMatchesScalar() {
        #if arch(arm64) || arch(x86_64) || arch(wasm32)
        let sizes = [8, 16, 32]

        for size in sizes {
            let blockScalar = Block2D(width: size, height: size)
            for y in 0..<size {
                for x in 0..<size {
                    let index = (y * size) + x
                    blockScalar.base[index] = Int16.random(in: (-1 * 1000)...1000)
                }
            }

            let blockSIMD = Block2D(width: size, height: size)
            let scalarData = Array(UnsafeBufferPointer(start: blockScalar.base, count: size * size))
            scalarData.withUnsafeBufferPointer { ptr in
                blockSIMD.base.update(from: ptr.baseAddress!, count: size * size)
            }

            var view = blockScalar.view
            _ = dwt2dScalar(view, size: size)
        
            view = blockSIMD.view
            // dwt2d will pick SIMD path for 8, 16, 32
            _ = dwt2d(view, size: size)
        
            let arrSIMD = Array(UnsafeBufferPointer(start: blockSIMD.base, count: size * size))
            let arrScalar = Array(UnsafeBufferPointer(start: blockScalar.base, count: size * size))
            XCTAssertEqual(arrSIMD, arrScalar, "SIMD vs Scalar mismatch for dwt2d size \(size)")

            view = blockScalar.view
            invDwt2dScalar(view, size: size)
        
            view = blockSIMD.view
            invDwt2d(view, size: size)
        
            let arrSIMD2 = Array(UnsafeBufferPointer(start: blockSIMD.base, count: size * size))
            let arrScalar2 = Array(UnsafeBufferPointer(start: blockScalar.base, count: size * size))
            XCTAssertEqual(arrSIMD2, arrScalar2, "SIMD vs Scalar mismatch for invDwt2d size \(size)")
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
        for i in 0..<(size * size) {
            block.base[i] = Int16.random(in: (-1 * 1000)...1000)
        }

        let originalData = Array(UnsafeBufferPointer(start: block.base, count: size * size))

        var view = block.view
        switch size {
        case 8:
            dwt2d_8(view)
        case 16:
            dwt2d_16(view)
        case 32:
            dwt2d_32(view)
        default:
            XCTFail("Unsupported size: \(size)")
        }
    
        view = block.view
        switch size {
        case 8:
            invDwt2d_8(view)
        case 16:
            invDwt2d_16(view)
        case 32:
            invDwt2d_32(view)
        default:
            XCTFail("Unsupported size: \(size)")
        }
    
        XCTAssertEqual(Array(UnsafeBufferPointer(start: block.base, count: size * size)), originalData, "Roundtrip failed for size \(size)")
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

            let originalData = Array(UnsafeBufferPointer(start: block.base, count: size * size))

            let view = block.view
            switch size {
            case 8:
                dwt2d_8(view)
                invDwt2d_8(view)
            case 16:
                dwt2d_16(view)
                invDwt2d_16(view)
            case 32:
                dwt2d_32(view)
                invDwt2d_32(view)
            default:
                XCTFail("Unsupported size: \(size)")
            }
        
            XCTAssertEqual(Array(UnsafeBufferPointer(start: block.base, count: size * size)), originalData, "Failed for block size \(size)")
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
