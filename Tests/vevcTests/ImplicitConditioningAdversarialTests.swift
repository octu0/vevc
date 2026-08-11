import XCTest
@testable import vevc

final class ImplicitConditioningAdversarialTests: XCTestCase {
    
    /// テスト1: 連続Pフレームでのエンコード・デコード ラウンドトリップ検証
    func testConsecutivePFramesRoundtrip() async throws {
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
        var frame2 = YCbCrImage(width: width, height: height)
        
        for i in 0..<(width * height) {
            frame0.yPlane[i] = UInt8((i * 7) % 255)
            frame1.yPlane[i] = UInt8((i * 7 + 10) % 255)
            frame2.yPlane[i] = UInt8((i * 7 + 20) % 255)
        }
        
        let bytes0 = try await encoder.encodeFrame(image: frame0)
        let decoded0 = try await decoder.decodeNextFrame(chunk: bytes0)
        XCTAssertNotNil(decoded0)
        
        let bytes1 = try await encoder.encodeFrame(image: frame1)
        let decoded1 = try await decoder.decodeNextFrame(chunk: bytes1)
        XCTAssertNotNil(decoded1)
        
        let bytes2 = try await encoder.encodeFrame(image: frame2)
        let decoded2 = try await decoder.decodeNextFrame(chunk: bytes2)
        XCTAssertNotNil(decoded2)
        
        if let dec2 = decoded2 {
            var totalDiff = 0
            for i in 0..<(width * height) {
                let orig = Int(frame2.yPlane[i])
                let dec = Int(dec2.yPlane[i])
                totalDiff += abs(orig - dec)
            }
            let avgDiff = Double(totalDiff) / Double(width * height)
            XCTAssertTrue(avgDiff <= 90.0, "P-Frame #2 pixel difference too large (\(avgDiff))")
        }
    }
    
    /// テスト2: 大振幅ラテントにおける残差変調・復元の境界値検証
    func testExtremeLatentResidualModulation() {
        let blockSize = 16
        let pool = BlockViewPool()
        let blockX = pool.get(width: blockSize, height: blockSize)
        let blockMu = pool.get(width: blockSize, height: blockSize)
        defer {
            pool.put(blockX)
            pool.put(blockMu)
        }
        
        for y in 0..<blockSize {
            let rX = blockX.rowPointer(y: y)
            let rMu = blockMu.rowPointer(y: y)
            for x in 0..<blockSize {
                rX[x] = 20000
                rMu[x] = -5000
            }
        }
        
        var blocks = [blockX]
        let mcBlocks = [blockMu]
        
        ImplicitConditioning.applyResidualModulation(blocks: &blocks, mcBlocks: mcBlocks, blockSize: blockSize)
        ImplicitConditioning.applyResidualDemodulation(blocks: &blocks, mcBlocks: mcBlocks, blockSize: blockSize)
        
        for y in 0..<blockSize {
            let rRestored = blocks[0].rowPointer(y: y)
            for x in 0..<blockSize {
                XCTAssertEqual(rRestored[x], 20000)
            }
        }
    }
}


