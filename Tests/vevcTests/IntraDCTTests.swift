import XCTest
import Foundation
@testable import vevc

final class IntraDCTTests: XCTestCase {
    
    /// (1) Transform Accuracy (Forward / Inverse)
    func testTransformAccuracy() throws {
        var src = [Int16](repeating: 0, count: 64)
        for i in 0..<64 {
            src[i] = Int16((i * 13) % 256 - 128)
        }
        
        let bytes = encodeL0PlaneDCT(plane: src, width: 8, height: 8, stride: 8, step: 1).bytes
        let dec = try decodeL0PlaneDCT(bytes: bytes, width: 8, height: 8, step: 1)
        
        // 最大誤差は小さいはず (quantization error 程度)
        var maxDiff: Int = 0
        for i in 0..<64 {
            let diff = abs(Int(src[i]) - Int(dec[i]))
            if diff > maxDiff {
                maxDiff = diff
            }
        }
        XCTAssertLessThanOrEqual(maxDiff, 5, "Reconstruction error too large")
    }
    
    /// (2) Roundtrip Consistency 
    func testRoundtripConsistency() throws {
        let width = 64
        let height = 64
        var src = [Int16](repeating: 0, count: width * height)
        for i in 0..<src.count {
            src[i] = Int16((i * 17) % 256 - 128)
        }
        
        let enc1 = encodeL0PlaneDCT(plane: src, width: width, height: height, stride: width, step: 10)
        let dec1 = try decodeL0PlaneDCT(bytes: enc1.bytes, width: width, height: height, step: 10)
        
        let enc2 = encodeL0PlaneDCT(plane: dec1, width: width, height: height, stride: width, step: 10)
        let dec2 = try decodeL0PlaneDCT(bytes: enc2.bytes, width: width, height: height, step: 10)
        
        XCTAssertEqual(enc1.bytes, enc2.bytes, "Encoded bytes should match on second pass")
        XCTAssertEqual(dec1, dec2, "Decoded pixels should be identical on second pass")
    }
    
    /// (3) PSNR Constraints
    func testPSNRConstraints() throws {
        let width = 128
        let height = 128
        var src = [Int16](repeating: 0, count: width * height)
        for i in 0..<src.count {
            src[i] = Int16((i * 7) % 256 - 128)
        }
        
        let step = 8
        let enc = encodeL0PlaneDCT(plane: src, width: width, height: height, stride: width, step: step)
        let dec = try decodeL0PlaneDCT(bytes: enc.bytes, width: width, height: height, step: step)
        
        var mseSum: Double = 0
        for i in 0..<src.count {
            let diff = Double(src[i]) - Double(dec[i])
            mseSum += diff * diff
        }
        let mse = mseSum / Double(src.count)
        let psnr = 10 * log10((255 * 255) / mse)
        
        XCTAssertGreaterThanOrEqual(psnr, 25.0, "PSNR too low for step=8")
    }
    
    /// (4) Determinism
    func testDeterminism() throws {
        let width = 32
        let height = 32
        var src = [Int16](repeating: 0, count: width * height)
        for i in 0..<src.count {
            src[i] = Int16((i * i) % 256 - 128)
        }
        
        let enc1 = encodeL0PlaneDCT(plane: src, width: width, height: height, stride: width, step: 5)
        let enc2 = encodeL0PlaneDCT(plane: src, width: width, height: height, stride: width, step: 5)
        
        XCTAssertEqual(enc1.bytes, enc2.bytes, "Multiple encode calls must produce exact same byte stream")
    }
}
