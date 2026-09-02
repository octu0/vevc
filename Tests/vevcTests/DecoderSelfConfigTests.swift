import XCTest
@testable import vevc

/// StreamingDecoderActor の公開 init(maxLayer:) は寸法・profile・GOP 形状を
/// ストリームの file header から自己設定する。player/libvevc が使う経路。
final class DecoderSelfConfigTests: XCTestCase {

    private func makeFrame(width: Int, height: Int, seed: Int) -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
        for y in 0..<height {
            for x in 0..<width {
                img.yPlane[y * width + x] = UInt8(clamping: (x * 2 + y + seed * 5) % 256)
            }
        }
        let cw = (width + 1) / 2
        let ch = (height + 1) / 2
        for cy in 0..<ch {
            for cx in 0..<cw {
                img.cbPlane[cy * cw + cx] = UInt8(clamping: 100 + cx + seed)
                img.crPlane[cy * cw + cx] = UInt8(clamping: 140 + cy)
            }
        }
        return img
    }

    /// VEVCEncoder の先頭チャンク(file header + I フレーム)を食わせると
    /// 自己設定され、以降のフレームも復号できる。profile 0x02 の状態確保
    /// (entropy history / MV state / rANSContext workspace)もヘッダ適用時に
    /// 行われることを roundtrip で確認する。
    func testSelfConfigureFromFileHeaderChunk() async throws {
        let width = 128
        let height = 96
        let encoder = VEVCEncoder(width: width, height: height, profile: 0x02)
        encoder.qstep = 16
        encoder.keyint = 10
        encoder.iqFloor = 0

        let decoder = StreamingDecoderActor(maxLayer: 2)
        for f in 0..<5 {
            let img = makeFrame(width: width, height: height, seed: f)
            let chunk = try await encoder.encode(image: img)
            let out = try await decoder.decodeNextFrame(chunk: chunk)
            XCTAssertNotNil(out, "frame \(f) decoded nil")
            XCTAssertEqual(out?.width, width)
            XCTAssertEqual(out?.height, height)
        }
    }

    /// ヘッダ前にフレームチャンクが来たらエラー(黙って壊れない)。
    func testFrameChunkBeforeHeaderThrows() async throws {
        let decoder = StreamingDecoderActor(maxLayer: 2)
        do {
            _ = try await decoder.decodeNextFrame(chunk: [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            XCTFail("frame chunk before header must throw")
        } catch {
        }
    }
}
