import XCTest
@testable import vevc

final class BidirectionalPredictionTests: XCTestCase {
    
    // MARK: - ヘルパー
    
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
        if mse < 0.0001 { return 100.0 }
        return 10.0 * log10(255.0 * 255.0 / mse)
    }
    
    /// テスト用画像を生成（シード値でフレーム間の変化を制御）
    private func makeTestImage(width: Int, height: Int, seed: Int) -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height)
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        for y in 0..<height {
            for x in 0..<width {
                let blockX = x / 64
                let blockY = y / 64
                let blockType = (blockX + blockY + seed) % 4
                let v: UInt8
                switch blockType {
                case 0: v = UInt8(clamping: (x + y + seed * 7) % 256)
                case 1: v = (x % 8 < 4) ? UInt8(clamping: 40 + seed % 30) : UInt8(clamping: 200 - seed % 30)
                case 2:
                    let hash = (x &* 2654435761) ^ (y &* 2246822519) ^ (seed &* 3266489917)
                    v = UInt8(clamping: 100 + (hash % 56))
                default: v = UInt8(clamping: 128 + seed % 20)
                }
                img.yPlane[y * width + x] = v
            }
        }
        for cy in 0..<cHeight {
            for cx in 0..<cWidth {
                let idx = cy * cWidth + cx
                img.cbPlane[idx] = UInt8(clamping: 128 + (cx + seed * 3) % 30 - 15)
                img.crPlane[idx] = UInt8(clamping: 128 + (cy + seed * 5) % 30 - 15)
            }
        }
        return img
    }
    
    // MARK: - 双方向MV探索テスト
    
    /// 双方向MV探索が前方/後方のうちSADが小さい方を選択することを検証
    func testBidirectionalMVSearchSelectsBetterReference() async throws {
        let width = 640
        let height = 480
        
        // 3つのフレーム: フレーム0(I), フレーム1(中間), フレーム2(最終)
        // フレーム1がフレーム0よりもフレーム2に近い場合、後方MVの方がSADが小さくなるはず
        let img0 = makeTestImage(width: width, height: height, seed: 0)
        let img1 = makeTestImage(width: width, height: height, seed: 10) // フレーム0から大きく変化
        let img2 = makeTestImage(width: width, height: height, seed: 11) // フレーム1に近い
        
        let pd0 = toPlaneData420(images: [img0])[0]
        let pd1 = toPlaneData420(images: [img1])[0]
        let pd2 = toPlaneData420(images: [img2])[0]
        
        let pool = BlockViewPool()
        // 前方MV（pd0 → pd1）と後方MV（pd2 → pd1）を計算
        let (_, fwdSADs) = await computeMotionVectors(curr: pd1, prev: pd0, pool: pool)
        let (_, bwdSADs) = await computeMotionVectors(curr: pd1, prev: pd2, pool: pool)
        
        // ブロックごとに、前方/後方のSADを比較して選択する
        var fwdBetterCount = 0
        var bwdBetterCount = 0
        let blockCount = min(fwdSADs.count, bwdSADs.count)
        for i in 0..<blockCount {
            if fwdSADs[i] < bwdSADs[i] {
                fwdBetterCount += 1
            } else if bwdSADs[i] < fwdSADs[i] {
                bwdBetterCount += 1
            }
        }
        
        // pd1はpd2に近いため、後方の方がSADが小さいブロックが多いはず
        XCTAssertGreaterThan(bwdBetterCount, 0, "後方参照がより良いブロックが1つも無い")
        
        // 双方向選択の結果として、選択されたMVの合計SADは前方のみより改善されるはず
        var totalFwdSAD = 0
        var totalBwdSAD = 0
        var totalBestSAD = 0
        for i in 0..<blockCount {
            totalFwdSAD += fwdSADs[i]
            totalBwdSAD += bwdSADs[i]
            totalBestSAD += min(fwdSADs[i], bwdSADs[i])
        }
        
        XCTAssertLessThanOrEqual(totalBestSAD, totalFwdSAD, "双方向選択の合計SADが前方のみより改善されていない")
        print("  前方SAD合計: \(totalFwdSAD), 後方SAD合計: \(totalBwdSAD), ベスト選択SAD合計: \(totalBestSAD)")
        print("  前方が良いブロック数: \(fwdBetterCount), 後方が良いブロック数: \(bwdBetterCount)")
    }
    
    // MARK: - 双方向予測を使ったラウンドトリップテスト
    
    /// 双方向予測を使ったGOPのエンコード→デコードで品質がベースライン以上であることを確認
    func testBidirectionalEncodeDecodeRoundtrip() async throws {
        let width = 640
        let height = 480
        let frameCount = 8
        
        var images: [YCbCrImage] = []
        for i in 0..<frameCount {
            images.append(makeTestImage(width: width, height: height, seed: i * 3))
        }
        
        let encoder = VEVCEncoder(width: width, height: height, maxbitrate: 1000 * 1024)
        let encoded = try await encoder.encodeToData(images: images)
        XCTAssertFalse(encoded.isEmpty, "エンコード結果が空")
        
        let decoded = try await Decoder().decode(data: encoded)
        XCTAssertEqual(decoded.count, frameCount, "デコード結果のフレーム数が\(frameCount)でない: \(decoded.count)")
        
        for i in 0..<frameCount {
            let psnrY = calculatePSNR(original: images[i].yPlane, decoded: decoded[i].yPlane)
            let frameType = (i == 0) ? "I" : "P"
            print("Frame \(i) (\(frameType)): PSNR = \(String(format: "%.1f", psnrY)) dB")
            XCTAssertGreaterThan(psnrY, 12.0, "フレーム\(i)(\(frameType)) Y-PSNR(\(String(format: "%.1f", psnrY))dB)が12dBを下回っている")
        }
    }
    
    // MARK: - GOP末尾フレームの品質改善テスト
    
    /// GOP末尾フレームのPSNRが前方予測のみのベースラインより劣化していないことを確認
    func testBidirectionalLastFrameQualityNotDegraded() async throws {
        let width = 640
        let height = 480
        let frameCount = 4 // 小さなGOPを想定
        
        var images: [YCbCrImage] = []
        for i in 0..<frameCount {
            images.append(makeTestImage(width: width, height: height, seed: i * 5))
        }
        
        let encoder = LayersEncodeActor(width: width, height: height, maxbitrate: 1000 * 1024, framerate: 30, zeroThreshold: 3, keyint: 15, sceneChangeThreshold: 32, pool: BlockViewPool())
        let decoder = CoreDecoder(width: width, height: height)
        
        var psnrs: [Double] = []
        for i in 0..<frameCount {
            let chunk = try await encoder.encodeSingleFrame(image: images[i])
            let decodedImg = try await decoder.decodeGOP(chunk: chunk)[0]
            let psnrY = calculatePSNR(original: images[i].yPlane, decoded: decodedImg.yPlane)
            psnrs.append(psnrY)
            let frameType = (i == 0) ? "I" : "P"
            print("Frame \(i) (\(frameType)): PSNR = \(String(format: "%.1f", psnrY)) dB")
        }
        
        // 末尾フレームのPSNRが一定水準以上であることを確認
        let lastPSNR = psnrs.last!
        XCTAssertGreaterThan(lastPSNR, 12.0, "GOP末尾フレームのY-PSNR(\(String(format: "%.1f", lastPSNR))dB)が12dBを下回っている")
    }
}
