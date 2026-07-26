import XCTest
@testable import vevc

final class QuantTests: XCTestCase {
    func testQuantizerInit() {
        let q1 = Quantizer(step: 64, roundToNearest: false)
        XCTAssertEqual(q1.step, 64)
        XCTAssertEqual(q1.mul, 16384)  // (1<<20)/64 = 16384
        XCTAssertEqual(q1.bias, 0)

        let q2 = Quantizer(step: 64, roundToNearest: true)
        XCTAssertEqual(q2.bias, 32768)  // 1<<15
    }

    func testQuantizationTableInit() {
        let qt = QuantizationTable(baseStep: 1600) // Q4 rep of 100
        XCTAssertEqual(qt.step, 1600)
        XCTAssertEqual(qt.qLow.step, 200) // 1600 / 8
        XCTAssertEqual(qt.qMid.step, 768) // 1600 clipped at 768
        XCTAssertEqual(qt.qHigh.step, 1024) // 1600 * 4 / 5 -> 1280 but clipped at 1024
    }

    func performRoundTripTest(width: Int, height: Int, step: Int, roundToNearest: Bool, signedMapping: Bool) {
        var block = BlockView.allocate(width: width, height: height)
        defer { block.deallocate() }
        let q = Quantizer(step: step, roundToNearest: roundToNearest)

        let originalValues: [Int16] = (0..<(width * height)).map { i in
            Int16.random(in: -32767...32767)
        }

        for y in 0..<height {
            for x in 0..<width {
                block[y, x] = originalValues[y * width + x]
            }
        }

        if signedMapping {
            switch width {
            case 8:
                quantizeSIMDSignedMapping8(block, q: q)
                dequantizeSIMDSignedMapping8(block, q: q)
            case 16:
                quantizeSIMDSignedMapping16(block, q: q)
                dequantizeSIMDSignedMapping16(block, q: q)
            case 32:
                quantizeSIMDSignedMapping32(block, q: q)
                dequantizeSIMDSignedMapping32(block, q: q)
            default:
                quantizeSIMDSignedMappingGeneric(block, q: q)
                dequantizeSIMDSignedMappingGeneric(block, q: q)
            }
        } else {
            switch width {
            case 4:
                quantizeSIMD4(block, q: q)
                dequantizeSIMD4(block, q: q)
            case 8:
                quantizeSIMD8(block, q: q)
                dequantizeSIMD8(block, q: q)
            case 16:
                quantizeSIMD16(block, q: q)
                dequantizeSIMD16(block, q: q)
            case 32:
                quantizeSIMD32(block, q: q)
                dequantizeSIMD32(block, q: q)
            default:
                fatalError("Unsupported size: \(width)")
            }
        }

        for y in 0..<height {
            for x in 0..<width {
                let original = Int32(originalValues[y * width + x])
                let reconstructed = Int32(block[y, x])
                let diff = abs(original - reconstructed)

                // The error should be at most real_step.
                // Due to fixed-point precision with 16-bit shift, it might be real_step + 1 in some cases.
                let realStep = Int32(step) / 16
                let limit = roundToNearest ? (realStep / 2 + 1) : (realStep + 1)
                XCTAssertLessThanOrEqual(
                    diff, limit,
                    "Error too large at (\(x), \(y)) for step \(step), roundToNearest: \(roundToNearest), signedMapping: \(signedMapping), original: \(original), recon: \(reconstructed), size: \(width)x\(height)"
                )
            }
        }
    }

    func testQuantizeRoundTrip() {
        let sizes = [8, 16, 32, 4]
        let steps = [16, 64, 208, 2048] // Q4 representations of 1, 4, 13, 128

        for size in sizes {
            for step in steps {
                performRoundTripTest(width: size, height: size, step: step, roundToNearest: false, signedMapping: false)
                performRoundTripTest(width: size, height: size, step: step, roundToNearest: true, signedMapping: false)
            }
        }
    }

    func testQuantizeSignedMappingRoundTrip() {
        let sizes = [8, 16, 32, 4]
        let steps = [16, 64, 208, 2048] // Q4 representations of 1, 4, 13, 128

        for size in sizes {
            for step in steps {
                performRoundTripTest(width: size, height: size, step: step, roundToNearest: false, signedMapping: true)
                performRoundTripTest(width: size, height: size, step: step, roundToNearest: true, signedMapping: true)
            }
        }
    }
}
