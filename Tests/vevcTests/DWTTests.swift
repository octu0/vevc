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
        let block = BlockView.allocate(width: size, height: size)
        defer { block.deallocate() }
        for i: Int in 0..<(size * size) {
            block.base[i + 0] = Int16.random(in: -512...511)
        }
        let originalData: [Int16] = Array(UnsafeBufferPointer(start: block.base, count: size * size))

        switch size {
        case 8:
            dwt2d_8(block)
            invDwt2d_8(block)
        case 16:
            dwt2d_16(block)
            invDwt2d_16(block)
        case 32:
            dwt2d_32(block)
            invDwt2d_32(block)
        default:
            XCTFail("Unsupported size: \(size)")
        }

        XCTAssertEqual(Array(UnsafeBufferPointer(start: block.base, count: size * size)), originalData, "Roundtrip failed for size \(size)")
    }

    func testDWT2DRoundtrip() {
        let sizes = [8, 16, 32]

        for size in sizes {
            let block = BlockView.allocate(width: size, height: size)
            defer { block.deallocate() }
            for y in 0..<size {
                for x in 0..<size {
                    let index = (y * size) + x
                    block.base[index] = Int16.random(in: (-1 * 1000)...1000)
                }
            }

            let originalData = Array(UnsafeBufferPointer(start: block.base, count: size * size))

            _ = dwt2d(block, size: size)
            invDwt2d(block, size: size)
        
            XCTAssertEqual(Array(UnsafeBufferPointer(start: block.base, count: size * size)), originalData, "Roundtrip failed for size \(size)")
        }
    }

    func testDWT2DSIMDMatchesScalar() {
        #if arch(arm64) || arch(x86_64) || arch(wasm32)
        let sizes = [8, 16, 32]

        for size in sizes {
            let blockScalar = BlockView.allocate(width: size, height: size)
            defer { blockScalar.deallocate() }
            for y in 0..<size {
                for x in 0..<size {
                    let index = (y * size) + x
                    blockScalar.base[index] = Int16.random(in: (-1 * 1000)...1000)
                }
            }

            let blockSIMD = BlockView.allocate(width: size, height: size)
            defer { blockSIMD.deallocate() }
            let scalarData = Array(UnsafeBufferPointer(start: blockScalar.base, count: size * size))
            scalarData.withUnsafeBufferPointer { ptr in
                blockSIMD.base.update(from: ptr.baseAddress!, count: size * size)
            }

            _ = dwt2dScalar(blockScalar, size: size)
        
            // dwt2d will pick SIMD path for 8, 16, 32
            _ = dwt2d(blockSIMD, size: size)
        
            let arrSIMD = Array(UnsafeBufferPointer(start: blockSIMD.base, count: size * size))
            let arrScalar = Array(UnsafeBufferPointer(start: blockScalar.base, count: size * size))
            XCTAssertEqual(arrSIMD, arrScalar, "SIMD vs Scalar mismatch for dwt2d size \(size)")

            invDwt2dScalar(blockScalar, size: size)
        
            invDwt2d(blockSIMD, size: size)
        
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
        let block = BlockView.allocate(width: size, height: size)
        defer { block.deallocate() }
        for i in 0..<(size * size) {
            block.base[i] = Int16.random(in: (-1 * 1000)...1000)
        }

        let originalData = Array(UnsafeBufferPointer(start: block.base, count: size * size))

        switch size {
        case 8:
            dwt2d_8(block)
        case 16:
            dwt2d_16(block)
        case 32:
            dwt2d_32(block)
        default:
            XCTFail("Unsupported size: \(size)")
        }
    
        switch size {
        case 8:
            invDwt2d_8(block)
        case 16:
            invDwt2d_16(block)
        case 32:
            invDwt2d_32(block)
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
            let block = BlockView.allocate(width: size, height: size)
            defer { block.deallocate() }
            for y in 0..<size {
                for x in 0..<size {
                    block.base[(y * size) + x] = Int16.random(in: -1000...1000)
                }
            }

            let originalData = Array(UnsafeBufferPointer(start: block.base, count: size * size))

            switch size {
            case 8:
                dwt2d_8(block)
                invDwt2d_8(block)
            case 16:
                dwt2d_16(block)
                invDwt2d_16(block)
            case 32:
                dwt2d_32(block)
                invDwt2d_32(block)
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
