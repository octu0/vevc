import XCTest
@testable import vevc

final class QuantTests: XCTestCase {
    
    func testQuantizerInitialization() {
        let q1 = Quantizer(step: 1, roundToNearest: true)
        XCTAssertEqual(q1.step, 1)
        XCTAssertEqual(q1.mul, 65536)
        XCTAssertEqual(q1.bias, 32768)
        
        let q2 = Quantizer(step: 4, roundToNearest: false)
        XCTAssertEqual(q2.step, 4)
        XCTAssertEqual(q2.mul, 16384)
        XCTAssertEqual(q2.bias, 0)
    }
    
    func testQuantizationTableInitialization() {
        let qt = QuantizationTable(baseStep: 8)
        XCTAssertEqual(qt.step, 8)
        XCTAssertEqual(qt.qLow.step, 1) // LOSSLESS FOR DC
        XCTAssertEqual(qt.qLow.bias, 32768) // roundToNearest: true
        XCTAssertEqual(qt.qMid.step, 16)
        XCTAssertEqual(qt.qMid.bias, 0) // roundToNearest: false
        XCTAssertEqual(qt.qHigh.step, 32)
        XCTAssertEqual(qt.qHigh.bias, 0) // roundToNearest: false
    }
    
    private func runRoundTripTest(width: Int, height: Int, step: Int, roundToNearest: Bool, signedMapping: Bool) {
        var block = Block2D(width: width, height: height)
        let q = Quantizer(step: step, roundToNearest: roundToNearest)
        
        let originalValues = (0..<height).map { y in
            (0..<width).map { x in
                Int16.random(in: -2000...2000)
            }
        }
        
        block.withView { view in
            for y in 0..<height {
                for x in 0..<width {
                    view[y, x] = originalValues[y][x]
                }
            }
            
            if signedMapping {
                quantizeSignedMapping(&view, q: q)
                dequantizeSignedMapping(&view, q: q)
            } else {
                quantize(&view, q: q)
                dequantize(&view, q: q)
            }
            
            for y in 0..<height {
                for x in 0..<width {
                    let original = Double(originalValues[y][x])
                    let reconstructed = Double(view[y, x])
                    let diff = abs(original - reconstructed)
                    
                    // The maximum error should be around step/2 for rounded, 
                    // or < step for truncated.
                    if roundToNearest {
                        XCTAssertLessThanOrEqual(diff, Double(step) / 2.0 + 0.5, "Failed at (\(x),\(y)) with step \(step)")
                    } else {
                        XCTAssertLessThanOrEqual(diff, Double(step), "Failed at (\(x),\(y)) with step \(step)")
                    }
                }
            }
        }
    }
    
    func testQuantizationRoundTrip8x8() {
        runRoundTripTest(width: 8, height: 8, step: 1, roundToNearest: true, signedMapping: false)
        runRoundTripTest(width: 8, height: 8, step: 8, roundToNearest: true, signedMapping: false)
        runRoundTripTest(width: 8, height: 8, step: 8, roundToNearest: false, signedMapping: false)
    }
    
    func testQuantizationRoundTrip16x16() {
        runRoundTripTest(width: 16, height: 16, step: 4, roundToNearest: true, signedMapping: false)
    }
    
    func testQuantizationRoundTrip32x32() {
        runRoundTripTest(width: 32, height: 32, step: 16, roundToNearest: true, signedMapping: false)
    }
    
    func testQuantizationRoundTripGeneric() {
        runRoundTripTest(width: 12, height: 10, step: 5, roundToNearest: true, signedMapping: false)
    }
    
    func testQuantizationSignedMappingRoundTrip8x8() {
        runRoundTripTest(width: 8, height: 8, step: 4, roundToNearest: true, signedMapping: true)
    }
    
    func testQuantizationSignedMappingRoundTripGeneric() {
        runRoundTripTest(width: 10, height: 10, step: 8, roundToNearest: false, signedMapping: true)
    }
    
    func testQuantizationEdgeCases() {
        var block = Block2D(width: 8, height: 8)
        let q = Quantizer(step: 10, roundToNearest: true)
        
        block.withView { view in
            view[0, 0] = 0
            view[0, 1] = Int16.max
            view[0, 2] = Int16.min
            
            quantize(&view, q: q)
            
            XCTAssertEqual(view[0, 0], 0)
            // Int16.max = 32767. (32767 * 6553 + 32768) >> 16 = 3276
            XCTAssertEqual(view[0, 1], 3276)
            // Int16.min = -32768. abs is 32768. (32768 * 6553 + 32768) >> 16 = 3276
            XCTAssertEqual(view[0, 2], -3276)
            
            dequantize(&view, q: q)
            XCTAssertEqual(view[0, 0], 0)
            XCTAssertEqual(view[0, 1], 32760)
            XCTAssertEqual(view[0, 2], -32760)
        }
    }
}
