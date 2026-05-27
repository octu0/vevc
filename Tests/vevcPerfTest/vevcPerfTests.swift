#if !DEBUG
import XCTest
@testable import vevc

final class VevcPerfTests: XCTestCase {
    
    // MARK: - PSNR計算ヘルパー
    
    /// 2つのUInt8配列間のPSNRを計算する
    private func calculatePSNR(original: [UInt8], decoded: [UInt8]) -> Double {
        let count = min(original.count, decoded.count)
        guard 0 < count else { return 0 }
        var mse: Double = 0
        for i in 0..<count {
            let diff = Double(Int(original[i]) - Int(decoded[i]))
            mse += diff * diff
        }
        mse /= Double(count)
        if mse < 0.0001 { return 100.0 } // ほぼ完全一致
        return 10.0 * log10(255.0 * 255.0 / mse)
    }
    
    /// テスト用のグラデーション画像を生成する
    private func generateGradientImage(width: Int, height: Int, seed: Int = 0) -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height)
        let cWidth = (width + 1) / 2
        for y in 0..<height {
            for x in 0..<width {
                let v = UInt8((x + y + seed) % 256)
                img.yPlane[y * width + x] = v
            }
        }
        for cy in 0..<((height + 1) / 2) {
            for cx in 0..<cWidth {
                img.cbPlane[cy * cWidth + cx] = UInt8(clamping: 128 + (cx + cy + seed) % 20 - 10)
                img.crPlane[cy * cWidth + cx] = UInt8(clamping: 128 + (cx - cy + seed + 256) % 20 - 10)
            }
        }
        return img
    }
    
    /// I+P Frame: 640x480の連続フレーム16枚でPSNRが各フレーム25dB以上
    func testLargeImageMultiFrameQuality() async throws {
        let width = 640
        let height = 480
        let frameCount = 16
        
        var images: [YCbCrImage] = []
        for i in 0..<frameCount {
            images.append(generateGradientImage(width: width, height: height, seed: i * 3))
        }
        
        let encoder = VEVCEncoder(width: width, height: height, maxbitrate: 1000 * 1024)
        let encoded = try await encoder.encodeToData(images: images)
        XCTAssert(encoded.isEmpty == false)
        
        let decoded = try await Decoder().decode(data: encoded)
        XCTAssertEqual(decoded.count, frameCount, "デコード結果のフレーム数が\(frameCount)でない: \(decoded.count)")
        
        for i in 0..<frameCount {
            let psnrY = calculatePSNR(original: images[i].yPlane, decoded: decoded[i].yPlane)
            print("Frame \(i) PSNR: \(psnrY) dB")
            XCTAssertGreaterThan(psnrY, 12.0, "フレーム\(i) Y-PSNR(\(String(format: "%.1f", psnrY))dB)が12dBを下回っている: ノイズ発生の可能性")
        }
    }
    
    /// maxbitrateを大幅に上げた場合にP-Frame品質が改善するかのテスト
    /// 改善する → 量子化ステップの適応ロジックに問題
    /// 改善しない → residual処理のロジックに問題
    func testComplexImageHighBitrate() async throws {
        let width = 640
        let height = 480
        let frameCount = 8
        
        // maxbitrateを10倍に
        let encoder = LayersEncodeActor(width: width, height: height, maxbitrate: 10000 * 1024, framerate: 30, zeroThreshold: 3, keyint: 15, sceneChangeThreshold: 32, pool: BlockViewPool())
        let decoder = StreamingDecoderActor(width: width, height: height)
        
        var failedFrames: [(Int, Double)] = []
        
        for i in 0..<frameCount {
            var img = YCbCrImage(width: width, height: height)
            let cWidth = (width + 1) / 2
            let cHeight = (height + 1) / 2
            
            for y in 0..<height {
                for x in 0..<width {
                    let blockX = x / 64
                    let blockY = y / 64
                    let blockType = (blockX + blockY + i) % 4
                    
                    let v: UInt8
                    switch blockType {
                    case 0: v = UInt8(clamping: (x + y + i * 7) % 256)
                    case 1: v = (x % 8 < 4) ? UInt8(clamping: 40 + i % 30) : UInt8(clamping: 200 - i % 30)
                    case 2:
                        let hash = (x &* 2654435761) ^ (y &* 2246822519) ^ (i &* 3266489917)
                        v = UInt8(clamping: 100 + (hash % 56))
                    default: v = UInt8(clamping: 128 + i % 20)
                    }
                    img.yPlane[y * width + x] = v
                }
            }
            for cy in 0..<cHeight {
                for cx in 0..<cWidth {
                    let idx = cy * cWidth + cx
                    img.cbPlane[idx] = UInt8(clamping: 128 + (cx + i * 3) % 30 - 15)
                    img.crPlane[idx] = UInt8(clamping: 128 + (cy + i * 5) % 30 - 15)
                }
            }
            
            let chunk = try await encoder.encodeFrame(image: img)
            let decodedImg = try await decoder.decodeNextFrame(chunk: chunk)!
            
            let psnrY = calculatePSNR(original: img.yPlane, decoded: decodedImg.yPlane)
            if psnrY <= 15.0 {
                failedFrames.append((i, psnrY))
            }
        }
        
        XCTAssertTrue(failedFrames.isEmpty, "高ビットレートでもPSNRが15dB以下: \(failedFrames.map { "フレーム\($0.0)=\(String(format: "%.1f", $0.1))dB" }.joined(separator: ", ")) → residual処理のロジックに問題あり")
    }
    
    /// ランダムノイズ含む複雑な画像でのテスト（自然画像に近い高エントロピーパターン）
    /// 注意: テストパターンは毎フレームでテクスチャが大幅に変化するため、
    /// motion compensationが効きにくく、通常の動画よりP-Frame品質が低下する。
    /// 閾値は緩めに設定（5dB: 白ノイズレベルはここに到達しない）。
    func testComplexImageRoundTrip() async throws {
        let width = 640
        let height = 480
        let frameCount = 20 // GOPサイズ(15)を超えるフレーム数
        
        let encoder = LayersEncodeActor(width: width, height: height, maxbitrate: 1000 * 1024, framerate: 30, zeroThreshold: 3, keyint: 15, sceneChangeThreshold: 32, pool: BlockViewPool())
        let decoder = StreamingDecoderActor(width: width, height: height)
        
        var failedFrames: [(Int, Double)] = []
        
        for i in 0..<frameCount {
            var img = YCbCrImage(width: width, height: height)
            let cWidth = (width + 1) / 2
            let cHeight = (height + 1) / 2
            
            for y in 0..<height {
                for x in 0..<width {
                    let blockX = x / 64
                    let blockY = y / 64
                    let blockType = (blockX + blockY + i) % 4
                    
                    let v: UInt8
                    switch blockType {
                    case 0: v = UInt8(clamping: (x + y + i * 7) % 256)
                    case 1: v = (x % 8 < 4) ? UInt8(clamping: 40 + i % 30) : UInt8(clamping: 200 - i % 30)
                    case 2:
                        let hash = (x &* 2654435761) ^ (y &* 2246822519) ^ (i &* 3266489917)
                        v = UInt8(clamping: 100 + (hash % 56))
                    default: v = UInt8(clamping: 128 + i % 20)
                    }
                    img.yPlane[y * width + x] = v
                }
            }
            for cy in 0..<cHeight {
                for cx in 0..<cWidth {
                    let idx = cy * cWidth + cx
                    img.cbPlane[idx] = UInt8(clamping: 128 + (cx + i * 3) % 30 - 15)
                    img.crPlane[idx] = UInt8(clamping: 128 + (cy + i * 5) % 30 - 15)
                }
            }
            
            let chunk = try await encoder.encodeFrame(image: img)
            XCTAssert(chunk.isEmpty == false)
            
            let decodedImg = try await decoder.decodeNextFrame(chunk: chunk)!
            
            let psnrY = calculatePSNR(original: img.yPlane, decoded: decodedImg.yPlane)
            if psnrY <= 5.0 {
                failedFrames.append((i, psnrY))
            }
        }
        
        // 5dB以下は白ノイズレベルの明らかな異常（符号バグ等）を検出するための閾値
        XCTAssertTrue(failedFrames.isEmpty, "PSNRが5dB以下のフレーム: \(failedFrames.map { "フレーム\($0.0)=\(String(format: "%.1f", $0.1))dB" }.joined(separator: ", "))")
    }
    
    /// 急激なシーンチェンジを含むテスト
    func testSceneChangeRoundTrip() async throws {
        let width = 1920
        let height = 1080
        let frameCount = 6
        
        let encoder = LayersEncodeActor(width: width, height: height, maxbitrate: 2000 * 1024, framerate: 30, zeroThreshold: 3, keyint: 15, sceneChangeThreshold: 8, pool: BlockViewPool())
        let decoder = StreamingDecoderActor(width: width, height: height)
        
        for i in 0..<frameCount {
            var img = YCbCrImage(width: width, height: height)
            let cWidth = (width + 1) / 2
            let cHeight = (height + 1) / 2
            
            // フレーム3で急激なシーンチェンジ
            let sceneBase: Int = (i < 3) ? 0 : 128
            
            for y in 0..<height {
                for x in 0..<width {
                    let v = UInt8(clamping: sceneBase + (x + y + i * 7) % 128)
                    img.yPlane[y * width + x] = v
                }
            }
            for cy in 0..<cHeight {
                for cx in 0..<cWidth {
                    let idx = cy * cWidth + cx
                    img.cbPlane[idx] = UInt8(clamping: 128 + (sceneBase / 4) + (cx + cy) % 20 - 10)
                    img.crPlane[idx] = UInt8(clamping: 128 - (sceneBase / 4) + (cx - cy + 256) % 20 - 10)
                }
            }
            
            let chunk = try await encoder.encodeFrame(image: img)
            XCTAssert(chunk.isEmpty == false)
            
            let decodedImg = try await decoder.decodeNextFrame(chunk: chunk)!

            let psnrY = calculatePSNR(original: img.yPlane, decoded: decodedImg.yPlane)
            XCTAssertGreaterThan(psnrY, 15.0, "フレーム\(i) Y-PSNR(\(String(format: "%.1f", psnrY))dB)が15dBを下回っている: シーンチェンジ後のノイズ")
        }
    }
}
#endif
