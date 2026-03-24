import XCTest
import Foundation
@testable import vevc

/// P-frameの連続エンコード・デコードにおけるフリッカー（フレーム間品質変動）を検出するテスト
final class FlickerDetectionTests: XCTestCase {

    // MARK: - SSIM計算（QualityDropTestsと同一ロジック）

    private func calcPlaneSSIM(p1: [UInt8], p2: [UInt8], w: Int, h: Int, stride1: Int, stride2: Int) -> Double {
        var ssimSum: Double = 0
        var blocks = 0
        let C1: Double = 6.5025
        let C2: Double = 58.5225

        p1.withUnsafeBufferPointer { ptr1 in
            p2.withUnsafeBufferPointer { ptr2 in
                guard let b1 = ptr1.baseAddress, let b2 = ptr2.baseAddress else { return }
                for y in stride(from: 0, to: h - 7, by: 8) {
                    for x in stride(from: 0, to: w - 7, by: 8) {
                        var sum1 = 0, sum2 = 0, sum1sq = 0, sum2sq = 0, sum12 = 0
                        for dy in 0..<8 {
                            let r1 = b1.advanced(by: (y + dy) * stride1 + x)
                            let r2 = b2.advanced(by: (y + dy) * stride2 + x)
                            for dx in 0..<8 {
                                let v1 = Int(r1[dx])
                                let v2 = Int(r2[dx])
                                sum1 += v1
                                sum2 += v2
                                sum1sq += v1 * v1
                                sum2sq += v2 * v2
                                sum12 += v1 * v2
                            }
                        }
                        let n = 64.0
                        let mu1 = Double(sum1) / n
                        let mu2 = Double(sum2) / n
                        let mu1sq = mu1 * mu1
                        let mu2sq = mu2 * mu2
                        let mu12 = mu1 * mu2
                        let sigma1sq = (Double(sum1sq) / n) - mu1sq
                        let sigma2sq = (Double(sum2sq) / n) - mu2sq
                        let sigma12 = (Double(sum12) / n) - mu12
                        let num = (2.0 * mu12 + C1) * (2.0 * sigma12 + C2)
                        let den = (mu1sq + mu2sq + C1) * (sigma1sq + sigma2sq + C2)
                        ssimSum += num / den
                        blocks += 1
                    }
                }
            }
        }
        return blocks == 0 ? 1.0 : ssimSum / Double(blocks)
    }

    private func calculateSSIMAll(img1: YCbCrImage, img2: YCbCrImage) -> Double {
        let w = min(img1.width, img2.width)
        let h = min(img1.height, img2.height)
        let ssimY = calcPlaneSSIM(p1: img1.yPlane, p2: img2.yPlane, w: w, h: h, stride1: img1.width, stride2: img2.width)
        let cw = min((img1.width + 1) / 2, (img2.width + 1) / 2)
        let ch = min((img1.height + 1) / 2, (img2.height + 1) / 2)
        let ssimU = calcPlaneSSIM(p1: img1.cbPlane, p2: img2.cbPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
        let ssimV = calcPlaneSSIM(p1: img1.crPlane, p2: img2.crPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
        return (4.0 * ssimY + ssimU + ssimV) / 6.0
    }

    // MARK: - テストパターン生成

    /// ゆっくり変化するグラデーション画像を生成
    /// フレームインデックスによって微小な動きをシミュレート
    private func generateGradientFrame(width: Int, height: Int, frameIndex: Int) -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
        let shift = frameIndex * 2  // フレームごとに2ピクセル右にシフト

        // Y: 水平グラデーション（シフトにより微小な動き）
        for y in 0..<height {
            for x in 0..<width {
                let srcX = (x + shift) % width
                let val = UInt8((srcX * 200) / width + 28) // 28..228の範囲
                img.yPlane[y * width + x] = val
            }
        }

        // Cb/Cr: 垂直方向のカラーグラデーション
        let cw = (width + 1) / 2
        let ch = (height + 1) / 2
        for cy in 0..<ch {
            for cx in 0..<cw {
                let cbVal = UInt8((cy * 180) / ch + 38)
                let crVal = UInt8((cx * 160) / cw + 48)
                img.cbPlane[cy * cw + cx] = cbVal
                img.crPlane[cy * cw + cx] = crVal
            }
        }

        return img
    }

    // MARK: - フリッカー検出テスト

    /// 連続フレームをエンコード・デコードし、フレーム間のSSIM変動（標準偏差）を計測する。
    /// フリッカーが存在する場合、連続フレーム間でSSIMの変動が大きくなる。
    func testFlickerDetectionOnSyntheticSequence() async throws {
        let width = 320
        let height = 240
        let frameCount = 20  // I-frame + P-frames（gopSize=15以内）

        let encoder = Encoder(
            width: width,
            height: height,
            maxbitrate: 500 * 1000,
            zeroThreshold: 3,
            keyint: 30,  // 全フレームが1GOP内に入るようにする
            sceneChangeThreshold: 8
        )
        let decoder = CoreDecoder()

        var originals: [YCbCrImage] = []
        var decoded: [YCbCrImage] = []
        var frameSizes: [Int] = []

        for i in 0..<frameCount {
            let img = generateGradientFrame(width: width, height: height, frameIndex: i)
            originals.append(img)

            let chunk = try await encoder.encode(image: img)
            frameSizes.append(chunk.count)

            let dec = try await decoder.decode(chunk: chunk)
            decoded.append(dec)
        }

        // 各フレームのSSIMを計算
        var ssimValues: [Double] = []
        for i in 0..<frameCount {
            let ssim = calculateSSIMAll(img1: originals[i], img2: decoded[i])
            ssimValues.append(ssim)
        }

        // フレーム間SSIM変動の統計
        let meanSSIM = ssimValues.reduce(0.0, +) / Double(ssimValues.count)
        var varianceSum: Double = 0
        for ssim in ssimValues {
            let diff = ssim - meanSSIM
            varianceSum += diff * diff
        }
        let stddevSSIM = (varianceSum / Double(ssimValues.count)).squareRoot()

        // 最低SSIMフレームの特定
        let minSSIM = ssimValues.min() ?? 0
        let maxSSIM = ssimValues.max() ?? 0
        let ssimRange = maxSSIM - minSSIM

        // P-frameのサイズ変動
        let pFrameSizes = Array(frameSizes.dropFirst()) // 最初のI-frameを除く
        let meanSize = Double(pFrameSizes.reduce(0, +)) / Double(pFrameSizes.count)
        var sizeVariance: Double = 0
        for size in pFrameSizes {
            let diff = Double(size) - meanSize
            sizeVariance += diff * diff
        }
        let stddevSize = (sizeVariance / Double(pFrameSizes.count)).squareRoot()

        // 結果の記録
        print("[Flicker] SSIM統計: 平均=\(String(format: "%.4f", meanSSIM)) 標準偏差=\(String(format: "%.6f", stddevSSIM)) 範囲=\(String(format: "%.4f", ssimRange))")
        print("[Flicker] 最低SSIM=\(String(format: "%.4f", minSSIM)) 最高SSIM=\(String(format: "%.4f", maxSSIM))")
        print("[Flicker] P-frameサイズ: 平均=\(String(format: "%.0f", meanSize)) 標準偏差=\(String(format: "%.0f", stddevSize))")

        // フレーム間の品質変動：SSIMの標準偏差が小さいほどフリッカーが少ない
        // 現状の測定値を記録するため、アサーションは緩めに設定
        XCTAssertLessThan(stddevSSIM, 0.05, "SSIM標準偏差が大きすぎる（フリッカーの兆候）")
        XCTAssertLessThan(ssimRange, 0.15, "SSIM範囲が広すぎる（フレーム間の品質差が大きい）")
    }

    /// 量子化ステップの変動を直接テストする。
    /// 同一の画像パターンに対して、異なるmeanSAD条件で量子化ステップがどう変わるかを検証。
    func testQuantizationStepStability() async throws {
        let width = 320
        let height = 240

        // 同じフレームを連続で送った場合、動きが無いのでmeanSADは低くなるはず
        // → 粗い量子化ケースに入る
        let encoder = Encoder(
            width: width,
            height: height,
            maxbitrate: 500 * 1024,
            zeroThreshold: 20,
            keyint: 15,
            sceneChangeThreshold: 10
        )
        let decoder = CoreDecoder()
        let img = generateGradientFrame(width: width, height: height, frameIndex: 0)
        let imgSlightlyDifferent = generateGradientFrame(width: width, height: height, frameIndex: 1)

        // I-frame
        let iFrameBytes = try await encoder.encode(image: img)

        // 動きがほぼないP-frame（同じ画像）
        let pFrame1Bytes = try await encoder.encode(image: img)

        // 微小な動きのあるP-frame
        let pFrame2Bytes = try await encoder.encode(image: imgSlightlyDifferent)

        // 再度動きなし
        let pFrame3Bytes = try await encoder.encode(image: img)

        // サイズの変動を記録
        print("[Flicker] I-frame: \(iFrameBytes.count) bytes")
        print("[Flicker] P-frame1(同一): \(pFrame1Bytes.count) bytes")
        print("[Flicker] P-frame2(微小動き): \(pFrame2Bytes.count) bytes")
        print("[Flicker] P-frame3(同一に戻る): \(pFrame3Bytes.count) bytes")

        // P-frameサイズが急変していないか確認（フレームサイズが急変=量子化ステップが急変）
        let maxPFrameSize = max(pFrame1Bytes.count, pFrame2Bytes.count, pFrame3Bytes.count)
        let minPFrameSize = min(pFrame1Bytes.count, pFrame2Bytes.count, pFrame3Bytes.count)
        let sizeRatio = Double(maxPFrameSize) / Double(max(1, minPFrameSize))

        print("[Flicker] P-frameサイズ比: \(String(format: "%.2f", sizeRatio))x")

        // サイズの急変はフリッカーの間接的指標
        // 10倍以上の変動は問題（現状の閾値は緩め）
        XCTAssertLessThan(sizeRatio, 10.0, "P-frameサイズの比率が大きすぎる")
    }
}
