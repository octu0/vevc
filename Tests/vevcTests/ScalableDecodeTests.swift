import XCTest
@testable import vevc

final class ScalableDecodeTests: XCTestCase {
    
    /// スケーラブルデコード (maxLayer 0, 1, 2) での P フレームデコード安定性検証
    func testScalableDecodingLayers() async throws {
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
        
        var frame0 = YCbCrImage(width: width, height: height)
        var frame1 = YCbCrImage(width: width, height: height)
        for i in 0..<(width * height) {
            frame0.yPlane[i] = UInt8((i * 5) % 255)
            frame1.yPlane[i] = UInt8((i * 5 + 15) % 255)
        }
        
        let bytes0 = try await encoder.encodeFrame(image: frame0)
        let bytes1 = try await encoder.encodeFrame(image: frame1)
        
        // maxLayer 0, 1, 2 それぞれでデコードが正常終了しクラッシュしないか検証
        let layers = [0, 1, 2]
        for layer in layers {
            let decoder = StreamingDecoderActor(
                maxLayer: layer,
                width: width,
                height: height
            )
            
            let dec0 = try await decoder.decodeNextFrame(chunk: bytes0)
            XCTAssertNotNil(dec0, "Failed to decode frame0 at layer \(layer)")
            
            let dec1 = try await decoder.decodeNextFrame(chunk: bytes1)
            XCTAssertNotNil(dec1, "Failed to decode frame1 at layer \(layer)")
        }
    }
}
