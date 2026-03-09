import XCTest
@testable import vevc

final class QuantTests: XCTestCase {

    func testQuantizerInit() {
        let q1 = Quantizer(step: 4, roundToNearest: false)
        XCTAssertEqual(q1.step, 4)
        XCTAssertEqual(q1.mul, ((1 << 16) / 4))
        XCTAssertEqual(q1.bias, 0)

        let q2 = Quantizer(step: 4, roundToNearest: true)
        XCTAssertEqual(q2.step, 4)
        XCTAssertEqual(q2.mul, ((1 << 16) / 4))
        XCTAssertEqual(q2.bias, (1 << 15))
    }

    func testQuantizationTableInit() {
        let baseStep = 8
        let qt = QuantizationTable(baseStep: baseStep)
        XCTAssertEqual(qt.step, 8)
        XCTAssertEqual(qt.qLow.step, 1)
        XCTAssertEqual(qt.qMid.step, 16)
        XCTAssertEqual(qt.qHigh.step, 32)

        XCTAssertEqual(qt.qLow.bias, (1 << 15)) // roundToNearest: true
        XCTAssertEqual(qt.qMid.bias, 0)       // roundToNearest: false
        XCTAssertEqual(qt.qHigh.bias, 0)      // roundToNearest: false
    }

    func testQuantizeDequantizeRoundTrip() {
        let sizes = [8, 16, 32, 10] // Including a generic size (10)
        let steps = [1, 2, 4, 8, 16]

        for size in sizes {
            for step in steps {
                var block = Block2D(width: size, height: size)
                let q = Quantizer(step: step, roundToNearest: true)

                // Fill block with some values
                block.withView { view in
                    for y in 0..<size {
                        for x in 0..<size {
                            view[y, x] = Int16((x * y) - ((size * size) / 2))
                        }
                    }
                }

                let originalData = block.data

                block.withView { view in
                    quantize(&view, q: q)
                }

                block.withView { view in
                    dequantize(&view, q: q)
                }

                // Verify results
                for i in 0..<block.data.count {
                    let original = Double(originalData[i])
                    let reconstructed = Double(block.data[i])
                    let error = abs(original - reconstructed)

                    // Reconstruction error should be at most step / 2 when rounding to nearest,
                    // but integer division/multiplication might have slight deviations.
                    // Max error for quantization step 'S' is S/2.
                    XCTAssertLessThanOrEqual(error, ((Double(step) / 2.0) + 0.5), "Error too large for step \(step) at index \(i), size \(size)")
                }
            }
        }
    }

    func testSignedMappingRoundTrip() {
        let sizes = [8, 16, 32, 7]
        let steps = [2, 4, 8]

        for size in sizes {
            for step in steps {
                var block = Block2D(width: size, height: size)
                let q = Quantizer(step: step, roundToNearest: false)

                block.withView { view in
                    for y in 0..<size {
                        for x in 0..<size {
                            view[y, x] = Int16((x - (size / 2)) * (y - (size / 2)))
                        }
                    }
                }

                let originalData = block.data

                block.withView { view in
                    quantizeSignedMapping(&view, q: q)
                }

                // After quantizeSignedMapping, values should be non-negative (mapped)
                for val in block.data {
                    XCTAssertLessThanOrEqual(0, val)
                }

                block.withView { view in
                    dequantizeSignedMapping(&view, q: q)
                }

                // Verify results
                for i in 0..<block.data.count {
                    let original = Double(originalData[i])
                    let reconstructed = Double(block.data[i])
                    let error = abs(original - reconstructed)

                    // Rounding is floor here, so error can be up to step - 1
                    XCTAssertLessThanOrEqual(error, Double(step), "Error too large for step \(step) at index \(i), size \(size)")
                }
            }
        }
    }

    func testClamping() {
        // Test that dequantization clamps to Int16 range
        var block = Block2D(width: 8, height: 8)
        let q = Quantizer(step: 1000)

        block.withView { view in
            view[0, 0] = 50 // 50 * 1000 = 50000 -> should clamp to 32767
            view[0, 1] = (-1 * 50) // -50 * 1000 = -50000 -> should clamp to -32768
        }

        block.withView { view in
            dequantize(&view, q: q)
        }

        XCTAssertEqual(block.data[0], 32767)
        XCTAssertEqual(block.data[1], Int16.min)
    }
}
