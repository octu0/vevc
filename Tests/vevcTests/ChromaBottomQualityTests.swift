import XCTest
@testable import vevc

/// デコード画像の下部カラーノイズ問題を診断するためのテスト。
/// 
/// 症状: 画面下部に赤・青・緑の帯状ノイズが発生
/// 仮説: クロマプレーン(Cb/Cr)の下部ブロック行でデータが破損している
///
/// テスト戦略:
/// 1. I-Frameのみでクロマ下部品質を確認 → エンコーダの境界処理チェック
/// 2. P-Frameでクロマ下部品質の推移確認 → 累積劣化チェック
/// 3. エンコーダ再構築 vs デコーダ出力の差分確認 → enc/dec非対称性チェック
/// 4. Y4MReaderのcSize計算確認 → 潜在的バグチェック
final class ChromaBottomQualityTests: XCTestCase {
    
    // MARK: - ヘルパー
    
    /// 指定領域のPSNRを計算する
    private func calculateRegionPSNR(
        original: [UInt8], decoded: [UInt8],
        width: Int, height: Int,
        startY: Int, endY: Int
    ) -> Double {
        var mse: Double = 0
        var count = 0
        for y in startY..<endY {
            for x in 0..<width {
                let idx = y * width + x
                guard idx < original.count, idx < decoded.count else { continue }
                let diff = Double(Int(original[idx]) - Int(decoded[idx]))
                mse += diff * diff
                count += 1
            }
        }
        guard 0 < count else { return 0 }
        mse /= Double(count)
        if mse < 0.0001 { return 100.0 }
        return 10.0 * log10(255.0 * 255.0 / mse)
    }
    
    /// 全体のPSNRを計算する
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
    
    /// クロマプレーンの上半分と下半分のPSNR差を計算し、下半分が極端に劣化していないか確認
    private func assertChromaBottomQuality(
        originalCb: [UInt8], decodedCb: [UInt8],
        originalCr: [UInt8], decodedCr: [UInt8],
        cWidth: Int, cHeight: Int,
        frameLabel: String,
        maxPsnrDifference: Double = 10.0,  // 上半分と下半分のPSNR差の上限
        minBottomPsnr: Double = 15.0       // 下半分の最低PSNR
    ) {
        let midY = cHeight / 2
        
        let cbTopPsnr = calculateRegionPSNR(
            original: originalCb, decoded: decodedCb,
            width: cWidth, height: cHeight,
            startY: 0, endY: midY
        )
        let cbBottomPsnr = calculateRegionPSNR(
            original: originalCb, decoded: decodedCb,
            width: cWidth, height: cHeight,
            startY: midY, endY: cHeight
        )
        
        let crTopPsnr = calculateRegionPSNR(
            original: originalCr, decoded: decodedCr,
            width: cWidth, height: cHeight,
            startY: 0, endY: midY
        )
        let crBottomPsnr = calculateRegionPSNR(
            original: originalCr, decoded: decodedCr,
            width: cWidth, height: cHeight,
            startY: midY, endY: cHeight
        )
        
        // 下半分のPSNRが最低ラインを下回っていないか
        XCTAssertGreaterThan(cbBottomPsnr, minBottomPsnr,
            "\(frameLabel) Cb下半分PSNR(\(String(format: "%.1f", cbBottomPsnr))dB)が\(minBottomPsnr)dBを下回っている")
        XCTAssertGreaterThan(crBottomPsnr, minBottomPsnr,
            "\(frameLabel) Cr下半分PSNR(\(String(format: "%.1f", crBottomPsnr))dB)が\(minBottomPsnr)dBを下回っている")
        
        // 上半分と下半分のPSNR差が許容範囲内か
        let cbDiff = cbTopPsnr - cbBottomPsnr
        let crDiff = crTopPsnr - crBottomPsnr
        
        if maxPsnrDifference < cbDiff {
            XCTFail("\(frameLabel) Cb上下PSNR差(\(String(format: "%.1f", cbDiff))dB)が\(maxPsnrDifference)dBを超えている (上=\(String(format: "%.1f", cbTopPsnr))dB, 下=\(String(format: "%.1f", cbBottomPsnr))dB) → 下部に集中的なカラーノイズ")
        }
        if maxPsnrDifference < crDiff {
            XCTFail("\(frameLabel) Cr上下PSNR差(\(String(format: "%.1f", crDiff))dB)が\(maxPsnrDifference)dBを超えている (上=\(String(format: "%.1f", crTopPsnr))dB, 下=\(String(format: "%.1f", crBottomPsnr))dB) → 下部に集中的なカラーノイズ")
        }
    }
    
    /// テスト用の自然画像に近いパターンを生成
    private func generateNaturalImage(width: Int, height: Int, seed: Int) -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height)
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        
        for y in 0..<height {
            for x in 0..<width {
                // 自然画像に近いグラデーション＋テクスチャ
                let grad = Double(y) / Double(height) * 128.0
                let tex = sin(Double(x + seed * 7) * 0.05) * 30.0
                let noise = Double((x &* 2654435761 + y &* 2246822519 + seed &* 3266489917) % 256) * 0.1
                let v = Int(grad + tex + noise + 64.0)
                img.yPlane[y * width + x] = UInt8(clamping: v)
            }
        }
        for cy in 0..<cHeight {
            for cx in 0..<cWidth {
                let idx = cy * cWidth + cx
                // クロマは緩やかな変化 (下部でも有意なデータがあること)
                let cbVal = 128.0 + sin(Double(cx + seed * 3) * 0.1) * 20.0 + Double(cy) / Double(cHeight) * 10.0
                let crVal = 128.0 + cos(Double(cy + seed * 5) * 0.1) * 20.0 - Double(cx) / Double(cWidth) * 10.0
                img.cbPlane[idx] = UInt8(clamping: Int(cbVal))
                img.crPlane[idx] = UInt8(clamping: Int(crVal))
            }
        }
        return img
    }
    
    // MARK: - テスト1: I-Frameのみ・クロマ下部品質
    
    /// I-Frameのみで下部クロマの品質が上部と同等であることを確認。
    /// 失敗する場合: エンコーダの境界ブロック処理に問題がある。
    func testIFrameChromaBottomQuality() async throws {
        let width = 1920
        let height = 1080
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        
        let img = generateNaturalImage(width: width, height: height, seed: 42)
        
        let encoder = LayersEncodeActor(width: width, height: height, maxbitrate: 2000 * 1024, framerate: 30, zeroThreshold: 3, keyint: 15, sceneChangeThreshold: 32)
        let decoder = CoreDecoder(width: width, height: height)
        
        let chunk = try await encoder.encodeSingleFrame(image: img)
        let decoded = try await decoder.decodeGOP(chunk: chunk)[0]
        
        assertChromaBottomQuality(
            originalCb: img.cbPlane, decodedCb: decoded.cbPlane,
            originalCr: img.crPlane, decodedCr: decoded.crPlane,
            cWidth: cWidth, cHeight: cHeight,
            frameLabel: "I-Frame",
            maxPsnrDifference: 10.0,
            minBottomPsnr: 15.0
        )
        
        // 全体のクロマPSNRも確認
        let cbPsnr = calculatePSNR(original: img.cbPlane, decoded: decoded.cbPlane)
        let crPsnr = calculatePSNR(original: img.crPlane, decoded: decoded.crPlane)
        XCTAssertGreaterThan(cbPsnr, 20.0, "I-Frame Cb全体PSNR(\(String(format: "%.1f", cbPsnr))dB)が低い")
        XCTAssertGreaterThan(crPsnr, 20.0, "I-Frame Cr全体PSNR(\(String(format: "%.1f", crPsnr))dB)が低い")
    }
    
    // MARK: - テスト2: P-Frame・クロマ下部品質の推移
    
    /// 複数フレームでP-Frame品質推移を確認。
    /// 特定フレーム以降で下部クロマが急激に劣化していないか確認。
    func testPFrameChromaBottomProgression() async throws {
        let width = 1920
        let height = 1080
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        let frameCount = 8
        
        let encoder = LayersEncodeActor(width: width, height: height, maxbitrate: 2000 * 1024, framerate: 30, zeroThreshold: 3, keyint: 15, sceneChangeThreshold: 32)
        let decoder = CoreDecoder(width: width, height: height)
        
        var bottomCbPsnrs: [Double] = []
        var bottomCrPsnrs: [Double] = []
        
        for i in 0..<frameCount {
            let img = generateNaturalImage(width: width, height: height, seed: i * 3)
            let chunk = try await encoder.encodeSingleFrame(image: img)
            let decoded = try await decoder.decodeGOP(chunk: chunk)[0]
            
            let frameType = (i == 0) ? "I" : "P"
            
            // 下部クロマ品質チェック
            assertChromaBottomQuality(
                originalCb: img.cbPlane, decodedCb: decoded.cbPlane,
                originalCr: img.crPlane, decodedCr: decoded.crPlane,
                cWidth: cWidth, cHeight: cHeight,
                frameLabel: "Frame\(i)(\(frameType))",
                maxPsnrDifference: 15.0,
                minBottomPsnr: 10.0
            )
            
            let midY = cHeight / 2
            let cbBottom = calculateRegionPSNR(
                original: img.cbPlane, decoded: decoded.cbPlane,
                width: cWidth, height: cHeight,
                startY: midY, endY: cHeight
            )
            let crBottom = calculateRegionPSNR(
                original: img.crPlane, decoded: decoded.crPlane,
                width: cWidth, height: cHeight,
                startY: midY, endY: cHeight
            )
            bottomCbPsnrs.append(cbBottom)
            bottomCrPsnrs.append(crBottom)
        }
        
        // P-Frame間で下部品質が急激に劣化していないか
        if 3 <= bottomCbPsnrs.count {
            let firstPsnr = bottomCbPsnrs[1]  // 最初のP-Frame
            let lastPsnr = bottomCbPsnrs.last!
            if 15.0 < firstPsnr - lastPsnr {
                XCTFail("Cb下部PSNRが急激に劣化: first P-Frame=\(String(format: "%.1f", firstPsnr))dB → last=\(String(format: "%.1f", lastPsnr))dB → P-Frame累積劣化")
            }
        }
    }
    
    // MARK: - テスト3: エンコーダ再構築 vs デコーダ出力の差分
    
    /// エンコーダ再構築結果とデコーダ出力のクロマプレーンが一致することを確認。
    /// 不一致の場合: エンコーダ/デコーダ間の処理に非対称性がある。
    func testEncoderDecoderChromaSymmetry() async throws {
        let width = 640
        let height = 480
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        
        let img0 = generateNaturalImage(width: width, height: height, seed: 0)
        let img1 = generateNaturalImage(width: width, height: height, seed: 3)
        
        let pd0 = toPlaneData420(images: [img0])[0]
        let pd1 = toPlaneData420(images: [img1])[0]
        
        let qtY = QuantizationTable(baseStep: 1) // 量子化を最小にして処理フローのみ検証
        let qtC = QuantizationTable(baseStep: 1)
        
        // I-Frame: encode → reconstruct
        let (iBytes, iRecon) = try await encodeSpatialLayers(pd: pd0, predictedPd: nil, maxbitrate: 10000 * 1024, qtY: qtY, qtC: qtC, zeroThreshold: 0)
        
        // I-Frame: decode
        let iDecoded = try await decodeSpatialLayers(r: iBytes, maxLayer: 2, dx: width, dy: height)
        let iDecodedPd = PlaneData420(img16: iDecoded)
        
        // I-Frameのクロマ差分
        let iCbDiff = calculatePlaneMAE(a: iRecon.cb, b: iDecodedPd.cb)
        let iCrDiff = calculatePlaneMAE(a: iRecon.cr, b: iDecodedPd.cr)
        
        XCTAssertLessThan(iCbDiff, 1.0,
            "I-Frame エンコーダ再構築とデコーダ出力のCb差異(MAE=\(String(format: "%.2f", iCbDiff)))が大きい → enc/dec非対称性")
        XCTAssertLessThan(iCrDiff, 1.0,
            "I-Frame エンコーダ再構築とデコーダ出力のCr差異(MAE=\(String(format: "%.2f", iCrDiff)))が大きい → enc/dec非対称性")
        
        // P-Frame: encode (without motion compensation since it's removed)
        let (pBytes, pRecon) = try await encodeSpatialLayers(pd: pd1, predictedPd: iRecon, maxbitrate: 10000 * 1024, qtY: qtY, qtC: qtC, zeroThreshold: 0)
        
        // P-Frame: decode
        let pDecoded = try await decodeSpatialLayers(r: pBytes, maxLayer: 2, dx: width, dy: height)
        let pDecodedPd = PlaneData420(img16: pDecoded)
        
        // P-Frameの残差のクロマ差分
        let pCbDiff = calculatePlaneMAE(a: pRecon.cb, b: pDecodedPd.cb)
        let pCrDiff = calculatePlaneMAE(a: pRecon.cr, b: pDecodedPd.cr)
        
        XCTAssertLessThan(pCbDiff, 1.0,
            "P-Frame エンコーダ再構築とデコーダ出力のCb差異(MAE=\(String(format: "%.2f", pCbDiff)))が大きい → enc/dec非対称性")
        XCTAssertLessThan(pCrDiff, 1.0,
            "P-Frame エンコーダ再構築とデコーダ出力のCr差異(MAE=\(String(format: "%.2f", pCrDiff)))が大きい → enc/dec非対称性")
        
        // 下半分限定のMAEも確認
        let halfCount = (cWidth * cHeight) / 2
        let pCbBottomDiff = calculatePlaneMAE(
            a: Array(pRecon.cb[halfCount...]),
            b: Array(pDecodedPd.cb[halfCount...])
        )
        let pCrBottomDiff = calculatePlaneMAE(
            a: Array(pRecon.cr[halfCount...]),
            b: Array(pDecodedPd.cr[halfCount...])
        )
        
        XCTAssertLessThan(pCbBottomDiff, 1.0,
            "P-Frame Cb下半分のenc/dec差異(MAE=\(String(format: "%.2f", pCbBottomDiff)))が大きい → 下部でのみ非対称性")
        XCTAssertLessThan(pCrBottomDiff, 1.0,
            "P-Frame Cr下半分のenc/dec差異(MAE=\(String(format: "%.2f", pCrBottomDiff)))が大きい → 下部でのみ非対称性")
    }
    
    // MARK: - テスト4: Y4MReaderのcSize計算確認（潜在的バグ）
    
    /// Y4MReaderで使用されている cSize = (width/2) * (height/2) と
    /// YCbCrImageで使用されている cSize = (width+1)/2 * (height+1)/2 の
    /// 差異が奇数サイズで問題になることを確認する。
    func testY4mChromaSizeConsistency() {
        // 偶数サイズ: 差異なし
        let w1 = 1920, h1 = 1080
        let y4mSize1 = (w1 / 2) * (h1 / 2)        // 518400
        let imgSize1 = ((w1 + 1) / 2) * ((h1 + 1) / 2)  // 518400
        XCTAssertEqual(y4mSize1, imgSize1, "偶数サイズでは一致するべき")
        
        // 奇数サイズ: 差異あり（潜在的バグ）
        let w2 = 1921, h2 = 1081
        let y4mSize2 = (w2 / 2) * (h2 / 2)         // 960 * 540 = 518400
        let imgSize2 = ((w2 + 1) / 2) * ((h2 + 1) / 2) // 961 * 541 = 519901
        
        // この差異は潜在的バグ: Y4Mから読み込む際にバッファが足りなくなる
        if y4mSize2 != imgSize2 {
            // Y4Mから読み込むサイズ(y4mSize2)がバッファサイズ(imgSize2)より少ない
            // → バッファ末尾にゴミデータが残る
            XCTAssertLessThan(y4mSize2, imgSize2,
                "奇数サイズでY4M cSize(\(y4mSize2)) < YCbCrImage cSize(\(imgSize2)) → 未初期化データの可能性")
        }
    }
    
    // MARK: - テスト5: 最下行ブロックのクロマ品質を詳細に分析
    
    /// 画像の最下部8行分（最後のブロック行）のクロマPSNRと、それ以外の部分のクロマPSNRを比較。
    /// 最下行ブロックに集中的な劣化があるかを確認する。
    func testLastBlockRowChromaQuality() async throws {
        let width = 1920
        let height = 1080
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        
        let img = generateNaturalImage(width: width, height: height, seed: 42)
        
        let encoder = LayersEncodeActor(width: width, height: height, maxbitrate: 2000 * 1024, framerate: 30, zeroThreshold: 3, keyint: 15, sceneChangeThreshold: 32)
        let decoder = CoreDecoder(width: width, height: height)
        
        let chunk = try await encoder.encodeSingleFrame(image: img)
        let decoded = try await decoder.decodeGOP(chunk: chunk)[0]
        
        // 最後のブロック行（8ピクセル行）のクロマPSNR
        let lastBlockStartY = max(0, cHeight - 8)
        
        let cbLastBlock = calculateRegionPSNR(
            original: img.cbPlane, decoded: decoded.cbPlane,
            width: cWidth, height: cHeight,
            startY: lastBlockStartY, endY: cHeight
        )
        let crLastBlock = calculateRegionPSNR(
            original: img.crPlane, decoded: decoded.crPlane,
            width: cWidth, height: cHeight,
            startY: lastBlockStartY, endY: cHeight
        )
        
        // 上の部分のPSNR
        let cbUpper = calculateRegionPSNR(
            original: img.cbPlane, decoded: decoded.cbPlane,
            width: cWidth, height: cHeight,
            startY: 0, endY: lastBlockStartY
        )
        let crUpper = calculateRegionPSNR(
            original: img.crPlane, decoded: decoded.crPlane,
            width: cWidth, height: cHeight,
            startY: 0, endY: lastBlockStartY
        )
        
        // 報告
        let cbDiff = cbUpper - cbLastBlock
        let crDiff = crUpper - crLastBlock
        
        XCTAssertGreaterThan(cbLastBlock, 10.0,
            "Cb最下ブロック行PSNR(\(String(format: "%.1f", cbLastBlock))dB)が非常に低い (上部=\(String(format: "%.1f", cbUpper))dB, 差=\(String(format: "%.1f", cbDiff))dB)")
        XCTAssertGreaterThan(crLastBlock, 10.0,
            "Cr最下ブロック行PSNR(\(String(format: "%.1f", crLastBlock))dB)が非常に低い (上部=\(String(format: "%.1f", crUpper))dB, 差=\(String(format: "%.1f", crDiff))dB)")
        
        // 上部との差が大きすぎないか
        XCTAssertLessThan(cbDiff, 15.0,
            "Cb最下ブロック行と上部のPSNR差(\(String(format: "%.1f", cbDiff))dB)が大きい → 最下行に集中的な問題")
        XCTAssertLessThan(crDiff, 15.0,
            "Cr最下ブロック行と上部のPSNR差(\(String(format: "%.1f", crDiff))dB)が大きい → 最下行に集中的な問題")
    }
    
    // MARK: - Int16配列のMAEヘルパー
    
    private func calculatePlaneMAE(a: [Int16], b: [Int16]) -> Double {
        let count = min(a.count, b.count)
        guard 0 < count else { return 0 }
        var sum: Double = 0
        for i in 0..<count {
            sum += abs(Double(a[i]) - Double(b[i]))
        }
        return sum / Double(count)
    }
}
