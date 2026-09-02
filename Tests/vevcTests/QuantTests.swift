import XCTest
@testable import vevc

final class QuantTests: XCTestCase {
    func testQuantizerInit() {
        let q1 = Quantizer(step: 64, roundToNearest: false, deadZoneBias: 0, centroidOffset: false)
        XCTAssertEqual(q1.step, 64)
        XCTAssertEqual(q1.mul, 16384)  // (1<<20)/64 = 16384
        XCTAssertEqual(q1.bias, 0)

        let q2 = Quantizer(step: 64, roundToNearest: true, deadZoneBias: 0, centroidOffset: false)
        XCTAssertEqual(q2.bias, 32768)  // 1<<15
    }

    func testQuantizationTableInit() {
        let qt = QuantizationTable(baseStep: 1600, isChroma: false, layerIndex: 0) // Q4 rep of 100
        XCTAssertEqual(qt.step, 1600)
        XCTAssertEqual(qt.qLow.step, 192) // 1600 / 8 = 200, capped at qLowCapQ4 (real step 12, DC banding threshold)
        XCTAssertEqual(qt.qMid.step, 768) // 1600 clipped at 768
        XCTAssertEqual(qt.qHigh.step, 1024) // 1600 * 4 / 5 -> 1280 but clipped at 1024
    }

    func performRoundTripTest(width: Int, height: Int, step: Int, roundToNearest: Bool, signedMapping: Bool) {
        var block = BlockView.allocate(width: width, height: height)
        defer { block.deallocate() }
        let q = Quantizer(step: step, roundToNearest: roundToNearest, deadZoneBias: 0, centroidOffset: false)

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
            case 4:
                quantize4(block, q: q)
                dequantize4(ptr: block.base, stride: block.stride, q: q)
            case 8:
                quantize8(block, q: q)
                dequantize8(ptr: block.base, stride: block.stride, q: q)
            case 16:
                quantize16(block, q: q)
                dequantize16(ptr: block.base, stride: block.stride, q: q)
            case 32:
                quantize32(block, q: q)
                dequantize32(block, q: q)
            default:
                fatalError("Unsupported size: \(width)")
            }
        } else {
            switch width {
            case 4:
                quantizeDPCM(block, q: q)
                dequantizeDPCM(ptr: block.base, stride: block.stride, q: q)
            default:
                return
            }
        }

        for y in 0..<height {
            for x in 0..<width {
                let original = Int32(originalValues[y * width + x])
                let reconstructed = Int32(block[y, x])
                let diff = abs(original - reconstructed)

                // The error should be at most real_step.
                // Due to fixed-point precision with 16-bit shift, it might be real_step + 1 in some cases.
                // The AC (signed-mapping) dequantizers additionally apply the
                // +3/16-step centroid offset away from zero (DataLayout §4),
                // which raises the worst-case error bound by 3·Δ/16 while
                // lowering the mean error.
                let realStep = Int32(step) / 16
                let centroidMargin: Int32 = signedMapping ? (realStep * 3) / 16 + 1 : 0
                let limit = (roundToNearest ? (realStep / 2 + 1) : (realStep + 1)) + centroidMargin
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
