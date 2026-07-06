import XCTest
@testable import vevc

final class EncoderTests: XCTestCase {

    func testComputeMaskedReconDistortion() {
        let width = 64
        let height = 64
        let yCount = width * height
        let uvCount = (width / 2) * (height / 2)
        
        let pool = BlockViewPool()
        var originalPd = PlaneData420(width: width, height: height, y: pool.getInt16(count: yCount), cb: pool.getInt16(count: uvCount), cr: pool.getInt16(count: uvCount))
        var reconPd = PlaneData420(width: width, height: height, y: pool.getInt16(count: yCount), cb: pool.getInt16(count: uvCount), cr: pool.getInt16(count: uvCount))
        
        originalPd.y.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count { ptr[i] = 0 }
        }
        reconPd.y.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count { ptr[i] = 0 }
        }
        
        // 64x64 grid -> 2x2 32x32 blocks
        // Block 0 (top-left): difference = 10
        // Block 1 (top-right): difference = 0 (static black)
        // Block 2 (bottom-left): difference = 0 (static black)
        // Block 3 (bottom-right): difference = 20
        
        originalPd.y.withUnsafeMutableBufferPointer { oPtr in
            reconPd.y.withUnsafeMutableBufferPointer { rPtr in
                for y in 0..<32 {
                    for x in 0..<32 {
                        let idx = y * width + x
                        oPtr[idx] = 10 // Block 0 SAD will be 10 * 1024
                    }
                }
                for y in 32..<64 {
                    for x in 32..<64 {
                        let idx = y * width + x
                        oPtr[idx] = 20 // Block 3 SAD will be 20 * 1024
                    }
                }
            }
        }
        
        // sads mapping to the 4 blocks
        // Block 0: Active
        // Block 1: Inactive
        // Block 2: Inactive
        // Block 3: Active
        let sads = [500, 0, 0, 800]
        
        let maskedSAD = computeMaskedReconDistortion(original: originalPd, reconstructed: reconPd, sads: sads)
        
        // Total SAD for active blocks = (10 * 1024) + (20 * 1024) = 30 * 1024
        // Active pixels = 2 * 1024 = 2048
        // maskedSAD = 30 * 1024 / 2048 = 15
        XCTAssertEqual(maskedSAD, 15)
        
        // Fallback test: no sads provided
        let fallbackSAD = computeMaskedReconDistortion(original: originalPd, reconstructed: reconPd, sads: nil)
        // Total SAD = 30 * 1024
        // Total pixels = 4096
        // fallbackSAD = 30 * 1024 / 4096 = 7.5 -> 7
        XCTAssertEqual(fallbackSAD, 7)
    }
}
