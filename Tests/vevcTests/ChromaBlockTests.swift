import XCTest
import Foundation
@testable import vevc

/// クロマ（Cb/Cr）の量子化品質をテストし、色調ブロック問題を検出する
final class ChromaBlockTests: XCTestCase {

    // MARK: - SSIM計算

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

    // MARK: - テストパターン生成

    /// カラーバーパターン: 色の変化が激しい境界を含むテストパターン
    private func generateColorBarFrame(width: Int, height: Int) -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)

        // Y: 水平方向にバーパターン（8色セクション）
        let barCount = 8
        let barWidth = width / barCount
        let yLevels: [UInt8] = [235, 210, 170, 145, 110, 85, 45, 16]

        for y in 0..<height {
            for x in 0..<width {
                let barIdx = min(x / max(1, barWidth), barCount - 1)
                img.yPlane[y * width + x] = yLevels[barIdx]
            }
        }

        // Cb/Cr: 色差信号でカラーバー表現
        let cw = (width + 1) / 2
        let ch = (height + 1) / 2
        let cbLevels: [UInt8] = [128, 44, 156, 72, 184, 100, 212, 128]
        let crLevels: [UInt8] = [128, 156, 44, 72, 212, 184, 100, 128]

        for cy in 0..<ch {
            for cx in 0..<cw {
                let srcX = cx * 2
                let barIdx = min(srcX / max(1, barWidth), barCount - 1)
                img.cbPlane[cy * cw + cx] = cbLevels[barIdx]
                img.crPlane[cy * cw + cx] = crLevels[barIdx]
            }
        }

        return img
    }

    /// 斜めカラーグラデーション: ブロック境界で色が不連続になりやすいパターン
    private func generateDiagonalGradientFrame(width: Int, height: Int) -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)

        for y in 0..<height {
            for x in 0..<width {
                img.yPlane[y * width + x] = UInt8(clamping: (x + y) * 200 / (width + height) + 28)
            }
        }

        let cw = (width + 1) / 2
        let ch = (height + 1) / 2
        for cy in 0..<ch {
            for cx in 0..<cw {
                // 斜めの色変化: Cbは左上→右下、Crは右上→左下
                let diagVal1 = (cx + cy) * 200 / (cw + ch)
                let diagVal2 = (cx + (ch - cy)) * 200 / (cw + ch)
                img.cbPlane[cy * cw + cx] = UInt8(clamping: diagVal1 + 28)
                img.crPlane[cy * cw + cx] = UInt8(clamping: diagVal2 + 28)
            }
        }

        return img
    }

    // MARK: - クロマ量子化の品質テスト

    /// I-frameでのクロマSSIMを個別に計測する。
    /// 色調ブロック問題はCb/CrのSSIMが低いことで検出される。
    func testChromaQualityOnColorBars() async throws {
        let width = 320
        let height = 240

        let encoder = Encoder(
            width: width,
            height: height,
            maxbitrate: 500 * 1000,
            zeroThreshold: 3,
            keyint: 15,
            sceneChangeThreshold: 8
        )
        let decoder = Decoder()

        let original = generateColorBarFrame(width: width, height: height)
        let chunk = try await encoder.encode(image: original)
        let dec = try await decoder.decode(chunk: chunk)

        // プレーンごとのSSIM
        let ssimY = calcPlaneSSIM(
            p1: original.yPlane, p2: dec.yPlane,
            w: width, h: height,
            stride1: width, stride2: width
        )
        let cw = (width + 1) / 2
        let ch = (height + 1) / 2
        let ssimCb = calcPlaneSSIM(
            p1: original.cbPlane, p2: dec.cbPlane,
            w: cw, h: ch,
            stride1: cw, stride2: cw
        )
        let ssimCr = calcPlaneSSIM(
            p1: original.crPlane, p2: dec.crPlane,
            w: cw, h: ch,
            stride1: cw, stride2: cw
        )

        // 結果ログ
        print("[ChromaBlock] カラーバーI-frame: Y=\(String(format: "%.4f", ssimY)) Cb=\(String(format: "%.4f", ssimCb)) Cr=\(String(format: "%.4f", ssimCr))")

        // Cb/CrのSSIMは少なくともYの80%以上あるべき
        // 色調ブロック問題があるとCb/CrのSSIMがYより大幅に低下する
        let chromaToLumaRatio = min(ssimCb, ssimCr) / max(ssimY, 0.001)
        print("[ChromaBlock] クロマ/輝度SSIM比: \(String(format: "%.4f", chromaToLumaRatio))")
    }

    /// 斜めグラデーションでのクロマ品質をP-frameで計測
    func testChromaQualityOnDiagonalGradient() async throws {
        let width = 320
        let height = 240

        let encoder = Encoder(
            width: width,
            height: height,
            maxbitrate: 500 * 1000,
            zeroThreshold: 3,
            keyint: 15,
            sceneChangeThreshold: 8
        )
        let decoder = Decoder()

        // I-frame
        let frame0 = generateDiagonalGradientFrame(width: width, height: height)
        let chunk0 = try await encoder.encode(image: frame0)
        _ = try await decoder.decode(chunk: chunk0)

        // P-frame: 同一フレーム（動きなし → 粗い量子化ケースへ）
        let frame1 = generateDiagonalGradientFrame(width: width, height: height)
        let chunk1 = try await encoder.encode(image: frame1)
        let dec1 = try await decoder.decode(chunk: chunk1)

        let cw = (width + 1) / 2
        let ch = (height + 1) / 2
        let ssimCb = calcPlaneSSIM(
            p1: frame1.cbPlane, p2: dec1.cbPlane,
            w: cw, h: ch,
            stride1: cw, stride2: cw
        )
        let ssimCr = calcPlaneSSIM(
            p1: frame1.crPlane, p2: dec1.crPlane,
            w: cw, h: ch,
            stride1: cw, stride2: cw
        )
        let ssimY = calcPlaneSSIM(
            p1: frame1.yPlane, p2: dec1.yPlane,
            w: width, h: height,
            stride1: width, stride2: width
        )

        print("[ChromaBlock] 斜めグラデーションP-frame: Y=\(String(format: "%.4f", ssimY)) Cb=\(String(format: "%.4f", ssimCb)) Cr=\(String(format: "%.4f", ssimCr))")

        // P-frameの動きなしケースではqtCが特に粗い(x3.0)ため、クロマSSIMが低くなるはず
        let chromaToLumaRatio = min(ssimCb, ssimCr) / max(ssimY, 0.001)
        print("[ChromaBlock] P-frame クロマ/輝度SSIM比: \(String(format: "%.4f", chromaToLumaRatio))")
    }

    /// 量子化テーブルの異なるステップでのクロマ歪みを直接比較する。
    /// encodeSpatialLayersを使わず、QuantizationTableレベルで誤差を測定する。
    func testChromaQuantizationDistortionByStep() async throws {
        // テスト対象: 量子化→逆量子化のラウンドトリップでの情報損失
        let blockSize = 8

        // 色差パターン: 高周波成分を含むグラデーション（量子化ステップの差異が出やすいようにする）
        var originalValues = [Int16](repeating: 0, count: blockSize * blockSize)
        var rng = LCG(seed: 12345)
        for y in 0..<blockSize {
            for x in 0..<blockSize {
                originalValues[y * blockSize + x] = Int16(Int(rng.next() % 100) - 50)
            }
        }

        // 異なるステップでの歪みを計測
        let steps = [2, 4, 6, 8, 12, 16]
        var distortions: [(step: Int, mse: Double)] = []

        for step in steps {
            let qt = QuantizationTable(baseStep: step)

            // ブロックを作成して量子化→逆量子化
            var block = Block2D(width: blockSize, height: blockSize)
            block.withView { view in
                for y in 0..<blockSize {
                    let ptr = view.rowPointer(y: y)
                    for x in 0..<blockSize {
                        ptr[x] = originalValues[y * blockSize + x]
                    }
                }
                dwt2d_8(&view)
            }

            // 量子化
            evaluateQuantizeBase8(block: &block, qt: qt)

            // 逆量子化
            block.withView { view in
                let half = blockSize / 2
                let base = view.base
                var llView = BlockView(base: base, width: half, height: half, stride: blockSize)
                var hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: blockSize)
                var lhView = BlockView(base: base.advanced(by: half * blockSize), width: half, height: half, stride: blockSize)
                var hhView = BlockView(base: base.advanced(by: half * blockSize + half), width: half, height: half, stride: blockSize)
                dequantizeSIMD(&llView, q: qt.qLow)
                dequantizeSIMDSignedMapping(&hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping(&hhView, q: qt.qHigh)
                invDwt2d_8(&view)
            }

            // MSE計算
            var mse: Double = 0
            block.withView { view in
                for y in 0..<blockSize {
                    let ptr = view.rowPointer(y: y)
                    for x in 0..<blockSize {
                        let diff = Double(ptr[x]) - Double(originalValues[y * blockSize + x])
                        mse += diff * diff
                    }
                }
            }
            mse /= Double(blockSize * blockSize)
            distortions.append((step: step, mse: mse))
        }

        // クロマ用ステップ(step*3等)の歪みが輝度用(step)に比べてどの程度大きいか確認
        // step=2でのMSEとstep=6(=2*3)でのMSE比較
        let mseLuma = distortions.first(where: { $0.step == 2 })?.mse ?? 0
        let mseChrCurrent = distortions.first(where: { $0.step == 6 })?.mse ?? 0 // 現在: step*3
        let mseChrProposed = distortions.first(where: { $0.step == 4 })?.mse ?? 0 // 提案: step*2

        // 提案版（step=4）は現行（step=6）より歪みが小さいことを確認
        XCTAssertLessThan(mseChrProposed, mseChrCurrent, "提案クロマステップ(step=4)は現行(step=6)より歪みが小さいべき")

        // 提案版のクロマMSEが輝度MSEの4倍以内であることを確認
        let proposedRatio = mseChrProposed / max(mseLuma, 0.001)
        XCTAssertLessThan(proposedRatio, 4.0, "提案クロマ/輝度MSE比が4倍以内であるべき")
    }
}

// 簡易的な決定論的乱数生成器
private struct LCG {
    var state: UInt32
    init(seed: UInt32) { self.state = seed }
    mutating func next() -> UInt32 {
        state = state &* 1664525 &+ 1013904223
        return state
    }
}

