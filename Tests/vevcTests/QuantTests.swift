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
            quantizeSIMDSignedMapping(block, q: q)
            dequantizeSIMDSignedMapping(block, q: q)
        } else {
            quantizeSIMD(block, q: q)
            dequantizeSIMD(block, q: q)
        }
        
        for y in 0..<height {
            for x in 0..<width {
                let original = Int32(originalValues[y * width + x])
                let reconstructed = Int32(block[y, x])
                let diff = abs(original - reconstructed)
                
                // The error should be at most step.
                // Due to fixed-point precision with 16-bit shift, it might be step + 1 in some cases.
                let limit = roundToNearest ? (Int32(step) / 2 + 1) : (Int32(step) + 1)
                XCTAssertLessThanOrEqual(diff, limit, "Error too large at (\(x), \(y)) for step \(step), roundToNearest: \(roundToNearest), signedMapping: \(signedMapping), original: \(original), recon: \(reconstructed), size: \(width)x\(height)")
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
    func testCascadedQuantizeRoundTrip() {
        let size = 32
        let step = 100
        
        var blockLuma = BlockView.allocate(width: size, height: size)
        defer { blockLuma.deallocate() }
        var blockChroma = BlockView.allocate(width: size, height: size)
        defer { blockChroma.deallocate() }
        
        let originalValues: [Int16] = (0..<(size * size)).map { _ in
            Int16.random(in: -200...200)
        }
        
        for y in 0..<size {
            for x in 0..<size {
                blockLuma[y, x] = originalValues[y * size + x]
            }
        }
            
        for y in 0..<size {
            for x in 0..<size {
                blockChroma[y, x] = originalValues[y * size + x]
            }
        }
            
        let qt = QuantizationTable(baseStep: step, isChroma: false, layerIndex: 2)
        let qtChroma = QuantizationTable(baseStep: step, isChroma: true, layerIndex: 2)
        
        quantizeCascaded32(block: &blockLuma, qt: qt, isChroma: false)
        dequantizeCascaded32(block: &blockLuma, qt: qt, isChroma: false)
        
        quantizeCascaded32(block: &blockChroma, qt: qtChroma, isChroma: true)
        dequantizeCascaded32(block: &blockChroma, qt: qtChroma, isChroma: true)
        
        var lumaErrorDiff: Int64 = 0
        var chromaErrorDiff: Int64 = 0
        
        for y in 0..<size {
            for x in 0..<size {
                let diff = abs(Int32(blockLuma[y, x]) - Int32(originalValues[y * size + x]))
                lumaErrorDiff += Int64(diff)
            }
        }
        for y in 0..<size {
            for x in 0..<size {
                let diff = abs(Int32(blockChroma[y, x]) - Int32(originalValues[y * size + x]))
                chromaErrorDiff += Int64(diff)
            }
        }
            
        // 修正前はLumaとChromaが同様のアルゴリズムのため誤差が同じかそれに近くなる。
        // TDDにおける修正後であれば、Chroma専用のより細かい（小さな）ステップを用いるため、誤差がより小さくなるはず。
        XCTAssertLessThan(chromaErrorDiff, lumaErrorDiff, "Chroma should have smaller quantization error due to finer quantization steps.")
    }
}
