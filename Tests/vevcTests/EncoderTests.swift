import XCTest
@testable import vevc

final class EncoderTests: XCTestCase {

    func testComputeMaskedReconDistortion() {
        let width = 64
        let height = 64
        let yCount = width * height
        let uvCount = (width / 2) * (height / 2)

        let pool = BlockViewPool()
        var originalPd = PlaneData420(
            width: width, height: height, y: pool.getInt16(count: yCount), cb: pool.getInt16(count: uvCount), cr: pool.getInt16(count: uvCount))
        var reconPd = PlaneData420(
            width: width, height: height, y: pool.getInt16(count: yCount), cb: pool.getInt16(count: uvCount), cr: pool.getInt16(count: uvCount))

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
                        oPtr[idx] = 10  // Block 0 SAD will be 10 * 1024
                    }
                }
                for y in 32..<64 {
                    for x in 32..<64 {
                        let idx = y * width + x
                        oPtr[idx] = 20  // Block 3 SAD will be 20 * 1024
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
        // maskedSAD = (30 * 1024 * 256) / 2048 = 3840
        XCTAssertEqual(maskedSAD, 3840)

        // Fallback test: no sads provided
        let fallbackSAD = computeMaskedReconDistortion(original: originalPd, reconstructed: reconPd, sads: nil)
        // Total SAD = 30 * 1024
        // Total pixels = 4096
        // fallbackSAD = (30 * 1024 * 256) / 4096 = 1920
        XCTAssertEqual(fallbackSAD, 1920)
    }
    func testCQPDeterminism() async throws {
        let width = 64
        let height = 64

        var img1 = YCbCrImage(width: width, height: height, ratio: .ratio420)
        var img2 = YCbCrImage(width: width, height: height, ratio: .ratio420)

        // Fill some predictable data
        img1.yPlane.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count { ptr[i] = UInt8(i % 256) }
        }
        img1.cbPlane.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count { ptr[i] = 128 }
        }
        img1.crPlane.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count { ptr[i] = 128 }
        }

        // Same data for img2
        img2.yPlane.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count { ptr[i] = UInt8(i % 256) }
        }
        img2.cbPlane.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count { ptr[i] = 128 }
        }
        img2.crPlane.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count { ptr[i] = 128 }
        }

        let encoder1 = VEVCEncoder(width: width, height: height, profile: 0x01)
        encoder1.qstep = 100
        let encoder2 = VEVCEncoder(width: width, height: height, profile: 0x01)
        encoder2.qstep = 100

        let bytes1 = try await encoder1.encode(image: img1)
        let bytes2 = try await encoder2.encode(image: img2)

        XCTAssertEqual(bytes1, bytes2)

        // Test a second frame
        let bytes1_p = try await encoder1.encode(image: img1)
        let bytes2_p = try await encoder2.encode(image: img2)

        XCTAssertEqual(bytes1_p, bytes2_p)
    }
}
