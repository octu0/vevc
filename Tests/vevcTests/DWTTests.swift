import XCTest
@testable import vevc

final class DWTTests: XCTestCase {


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
                    lift53Block8(ptr, stride: 1)
                    inverseLift53Block8(ptr, stride: 1)
                case 16:
                    lift53Block16(ptr, stride: 1)
                    inverseLift53Block16(ptr, stride: 1)
                case 32:
                    lift53Block32(ptr, stride: 1)
                    inverseLift53Block32(ptr, stride: 1)
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
                dwt2DBlock8(block)
                inverseDWT2DBlock8(block)
            case 16:
                dwt2DBlock16(block)
                inverseDWT2DBlock16(block)
            case 32:
                dwt2DBlock32(block)
                inverseDWT2DBlock32(block)
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
                    lift53Block8(ptr, stride: 1)
                    inverseLift53Block8(ptr, stride: 1)
                case 16:
                    lift53Block16(ptr, stride: 1)
                    inverseLift53Block16(ptr, stride: 1)
                case 32:
                    lift53Block32(ptr, stride: 1)
                    inverseLift53Block32(ptr, stride: 1)
                default:
                    XCTFail("Unsupported size: \(size)")
                }
            }

            XCTAssertEqual(data, original, "SIMD failed for size \(size)")
        }
    }
    #endif
}
