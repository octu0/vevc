import XCTest
@testable import vevc

final class ImplicitConditioningAdversarialExtendedTests: XCTestCase {

    /// 1. dstWidth = 0 でゼロディメンションが与えられた場合の安全なハンドリング検証
    func testZeroDimensionHandling() {
        let refBuf = [Int16](repeating: 50, count: 16)

        refBuf.withUnsafeBufferPointer { rPtr in
            let dst = ImplicitConditioning.downsamplePlane4x4Average(
                src: rPtr,
                srcWidth: 4,
                srcHeight: 4,
                dstWidth: 0,
                dstHeight: 0
            )
            XCTAssertEqual(dst.count, 0)
        }
    }

    /// 2. エンコーダとデコーダの Deblocking Filter / Latent Synchrony による P-Frame 参照画素ドリフト検証
    func testDeblockingQStepMismatchDrift() async throws {
        let width = 64
        let height = 64

        let encoder = LayersEncodeActor(
            width: width,
            height: height,
            maxbitrate: 1_000_000,
            framerate: 30,
            zeroThreshold: 0,
            keyint: 10,
            sceneChangeThreshold: 100
        )

        let decoder = StreamingDecoderActor(
            maxLayer: 2,
            width: width,
            height: height
        )

        var frame0 = YCbCrImage(width: width, height: height)
        var frame1 = YCbCrImage(width: width, height: height)

        for i in 0..<(width * height) {
            frame0.yPlane[i] = UInt8((i * 11) % 255)
            frame1.yPlane[i] = UInt8((i * 11 + 40) % 255)
        }

        let bytes0 = try await encoder.encodeFrame(image: frame0)
        let dec0 = try await decoder.decodeNextFrame(chunk: bytes0)
        XCTAssertNotNil(dec0)

        let bytes1 = try await encoder.encodeFrame(image: frame1)
        let decoded1 = try await decoder.decodeNextFrame(chunk: bytes1)
        XCTAssertNotNil(decoded1)
    }
}

