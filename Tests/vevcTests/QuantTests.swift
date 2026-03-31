import XCTest
@testable import vevc

final class QuantTests: XCTestCase {
    func testQuantizerInit() {
        let q1 = Quantizer(step: 4, roundToNearest: false)
        XCTAssertEqual(q1.step, 4)
        XCTAssertEqual(q1.mul, 16384) // (1<<16)/4
        XCTAssertEqual(q1.bias, 0)
        
        let q2 = Quantizer(step: 4, roundToNearest: true)
        XCTAssertEqual(q2.bias, 32768) // 1<<15
    }
    
    func testQuantizationTableInit() {
        let qt = QuantizationTable(baseStep: 100)
        XCTAssertEqual(qt.step, 100)
        XCTAssertEqual(qt.qLow.step, 8)
        XCTAssertEqual(qt.qMid.step, 25)
        XCTAssertEqual(qt.qHigh.step, 50)
    }
    
    func performRoundTripTest(width: Int, height: Int, step: Int, roundToNearest: Bool, signedMapping: Bool) {
        var block = Block2D(width: width, height: height)
        let q = Quantizer(step: step, roundToNearest: roundToNearest)
        
        let originalValues: [Int16] = (0..<(width * height)).map { i in
            Int16.random(in: -16382...16382)
        }
        
        block.withView { view in
            for y in 0..<height {
                for x in 0..<width {
                    view[y, x] = originalValues[y * width + x]
                }
            }
            
            if signedMapping {
                quantizeSIMDSignedMapping(&view, q: q)
                dequantizeSIMDSignedMapping(&view, q: q)
            } else {
                quantizeSIMD(&view, q: q)
                dequantizeSIMD(&view, q: q)
            }
            
            for y in 0..<height {
                for x in 0..<width {
                    let original = Int32(originalValues[y * width + x])
                    let reconstructed = Int32(view[y, x])
                    let diff = abs(original - reconstructed)
                    
                    // The error should be at most step.
                    // Due to fixed-point precision with 16-bit shift, it might be step + 1 in some cases.
                    let limit = roundToNearest ? (Int32(step) / 2 + 1) : (Int32(step) + 1)
                    XCTAssertLessThanOrEqual(diff, limit, "Error too large at (\(x), \(y)) for step \(step), roundToNearest: \(roundToNearest), signedMapping: \(signedMapping), original: \(original), recon: \(reconstructed), size: \(width)x\(height)")
                }
            }
        }
    }
    
    func testQuantizeRoundTrip() {
        let sizes = [8, 16, 32, 4]
        let steps = [1, 4, 13, 128]
        
        for size in sizes {
            for step in steps {
                performRoundTripTest(width: size, height: size, step: step, roundToNearest: false, signedMapping: false)
                performRoundTripTest(width: size, height: size, step: step, roundToNearest: true, signedMapping: false)
            }
        }
    }
    
    func testQuantizeSignedMappingRoundTrip() {
        let sizes = [8, 16, 32, 4]
        let steps = [1, 4, 13, 128]
        
        for size in sizes {
            for step in steps {
                performRoundTripTest(width: size, height: size, step: step, roundToNearest: false, signedMapping: true)
                performRoundTripTest(width: size, height: size, step: step, roundToNearest: true, signedMapping: true)
            }
        }
    }
    
    // Add strict test for negative drift off-by-one error in dequantizeSIMDSignedMapping
    func testSignedMappingNegativeDrift() {
        var block = Block2D(width: 8, height: 8)
        let q = Quantizer(step: 1, roundToNearest: false)
        
        block.withView { view in
            for y in 0..<8 {
                for x in 0..<8 {
                    // Fill with simple negative values
                    view[y, x] = Int16(-1 - (y * 8 + x))
                }
            }
            // Create a copy of the original values
            var original = [Int16](repeating: 0, count: 64)
            for y in 0..<8 {
                for x in 0..<8 {
                    original[y * 8 + x] = view[y, x]
                }
            }
            
            // Encode -> Decode
            quantizeSIMDSignedMapping(&view, q: q)
            dequantizeSIMDSignedMapping(&view, q: q)
            
            // With step=1 and no rounding, the reconstructed values MUST be exactly identical
            for y in 0..<8 {
                for x in 0..<8 {
                    let orig = original[y * 8 + x]
                    let recon = view[y, x]
                    XCTAssertEqual(orig, recon, "Signed mapping exhibited drift! Original: \(orig), Recon: \(recon)")
                }
            }
        }
    }
}
