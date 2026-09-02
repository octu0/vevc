import XCTest
@testable import vevc

final class Profile0x02Tests: XCTestCase {
    
    func testProfile0x02StaticSequence() async throws {
        let width = 1920
        let height = 1080
        var y = [UInt8](repeating: 128, count: width * height)
        let cb = [UInt8](repeating: 128, count: width * height / 4)
        let cr = [UInt8](repeating: 128, count: width * height / 4)
        
        // 複雑すぎない静止画を生成（量子化誤差が 1画素あたり2 以下に収まるようにフラットにする）
        for i in 0..<height {
            for j in 0..<width {
                y[i * width + j] = UInt8((i / 32 * 32) % 256) // 32x32のブロック単位で一定の色にする
            }
        }
        
        var staticImage = YCbCrImage(width: width, height: height, ratio: .ratio420)
        staticImage.yPlane = y
        staticImage.cbPlane = cb
        staticImage.crPlane = cr
        var frames = [YCbCrImage]()
        
        // 100フレーム。CopyFrameにならないように1画素だけごくわずかにいじる（微小ノイズ）
        for f in 0..<100 {
            var mY = y
            mY[0] = UInt8(((Int(y[0]) + f) % 5) + 128) // PSNR > 99になる程度の超微小変更
            var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
            img.yPlane = mY
            img.cbPlane = cb
            img.crPlane = cr
            frames.append(img)
        }
        
        // qstep16でエンコード
        let encoder = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 30, profile: 0x02)
        let bitstream = try await encoder.encodeToData(images: frames)
        
        // デコード
        let decoder = Decoder(maxLayer: 2)
        let outFrames = try await decoder.decode(data: bitstream)
        
        XCTAssertEqual(outFrames.count, 100)
        
        // 解析
        var offset = 0
        var skipPrev = 0
        var skipLtr = 0
        var inter = 0
        var totalBlocks = 0
        
        let bw = (width + 31) / 32
        let bh = (height + 31) / 32
        let blockCount = bw * bh
        // #36: the skipMap is context-coded with per-GOP adaptive models, so a
        // standalone bitstream walker has to carry the same state and reset it
        // at every I frame, exactly as the decoder does.
        let walkerContext = SyntaxContextModels()

        while offset < bitstream.count {
            if offset + 4 <= bitstream.count && bitstream[offset] == 0x56 && bitstream[offset+1] == 0x45 && bitstream[offset+2] == 0x56 && bitstream[offset+3] == 0x43 {
                let _ = try VEVCFileHeader.deserialize(from: bitstream, offset: &offset)
                continue
            }
            
            let start = offset
            let fh = try VEVCFrameHeader.deserialize(from: bitstream, offset: &offset, profile: 0x02)
            if !fh.isCopyFrame {
                if fh.isIFrame {
                    walkerContext.reset()
                }
                if fh.skipMapSize > 0 {
                    let smData = Array(bitstream[offset..<(offset + fh.skipMapSize)])
                    let map = try decodeSkipMapContext(data: smData, count: blockCount, cols: bw, state: walkerContext)
                    for m in map {
                        if m == .skip_prev { skipPrev += 1 }
                        else if m == .skip_ltr { skipLtr += 1 }
                        else { inter += 1 }
                        totalBlocks += 1
                    }
                }
                
                let headerSize = offset - start
                offset = start + headerSize + fh.skipMapSize + fh.mvsSize + fh.refDirSize + fh.treeMapSize + fh.layer0Size + fh.layer1Size + fh.layer2Size
            } else {
                let headerSize = offset - start
                skipPrev += blockCount
                totalBlocks += blockCount
                offset = start + headerSize
            }
        }
        
        print("--- Profile0x02 Static Test ---")
        print("Total Blocks: \(totalBlocks)")
        print("Skip Prev   : \(skipPrev)")
        print("Skip LTR    : \(skipLtr)")
        print("Inter       : \(inter)")
        
        // skip (prev + LTR) がほぼ全発火していること (微小ノイズの1ブロック以外)
        let skipRatio = Double(skipPrev + skipLtr) / Double(totalBlocks)
        XCTAssertGreaterThan(skipRatio, 0.99)
        
        // 全フレーム一致 (PSNR/SSIM)
        var minSsim: Double = 1.0
        for i in 0..<100 {
            let ssim = calculateSSIM(img1: staticImage, img2: outFrames[i])
            if ssim < minSsim { minSsim = ssim }
        }
        print("Min SSIM    : \(minSsim)")
        XCTAssertGreaterThan(minSsim, 0.99)
    }
    
    func calculateSSIM(img1: YCbCrImage, img2: YCbCrImage) -> Double {
        // SSIM計算は重いので中央だけ
        let w = min(64, img1.width)
        let h = min(64, img1.height)
        
        var sum1: Double = 0
        var sum2: Double = 0
        var sum1Sq: Double = 0
        var sum2Sq: Double = 0
        var pdt: Double = 0
        let N = Double(w * h)
        
        img1.yPlane.withUnsafeBufferPointer { p1 in
            img2.yPlane.withUnsafeBufferPointer { p2 in
                for y in 0..<h {
                    for x in 0..<w {
                        let idx = y * img1.width + x
                        let val1 = Double(p1[idx])
                        let val2 = Double(p2[idx])
                        sum1 += val1
                        sum2 += val2
                        sum1Sq += val1 * val1
                        sum2Sq += val2 * val2
                        pdt += val1 * val2
                    }
                }
            }
        }
        
        let mu1 = sum1 / N
        let mu2 = sum2 / N
        let sigma1Sq = (sum1Sq / N) - (mu1 * mu1)
        let sigma2Sq = (sum2Sq / N) - (mu2 * mu2)
        let sigma12 = (pdt / N) - (mu1 * mu2)
        
        let C1 = 6.5025
        let C2 = 58.5225
        
        let ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1Sq + sigma2Sq + C2))
        return ssim
    }
    
    // 4. エンコーダ内部の再構成画像とデコーダ出力画像が全フレームで完全にバイナリ一致すること。
    func testProfile0x02PFrameReconstructionMatch() async throws {
        let width = 640
        let height = 480
        let pool = BlockViewPool()
        let qtY = QuantizationTable(baseStep: 16, isChroma: false, layerIndex: 0)
        let qtC = QuantizationTable(baseStep: 16, isChroma: true, layerIndex: 0)
        
        var img0 = YCbCrImage(width: width, height: height, ratio: .ratio420)
        var img1 = YCbCrImage(width: width, height: height, ratio: .ratio420)
        for i in 0..<(width * height) {
            let v = UInt8.random(in: 0...255)
            img0.yPlane[i] = v
            img1.yPlane[i] = UInt8(clamping: Int(v) + Int.random(in: -5...5))
        }
        for i in 0..<(width * height / 4) {
            img0.cbPlane[i] = 128
            img0.crPlane[i] = 128
            img1.cbPlane[i] = 128
            img1.crPlane[i] = 128
        }
        
        let (pd0, rel0) = toPlaneData420(image: img0, pool: pool)
        defer { rel0() }
        let (pd1, rel1) = toPlaneData420(image: img1, pool: pool)
        defer { rel1() }
        
        // I-frame
        let l0StateEnc = L0RefState()
        let l0StateDec = L0RefState()
        let (bytesI, encReconI, _, _, relEncI) = try await encodeSpatialLayersIntraForProfile2(
            pd: pd0, pool: pool, qtY: qtY, qtC: qtC, zeroThreshold: 5, l0State: l0StateEnc)
        defer { relEncI() }
        let decImg16I = try await decodeSpatialLayersForProfile2(r: bytesI, pool: pool, maxLayer: 2, dx: width, dy: height, predictedPd: nil, nextPd: nil, roundOffset: 0, entropyHistories: nil, l0State: l0StateDec, mvState: nil, parallelEntropy: true, updateL0Prev: true, ransContextWorkspace: nil)
        let decReconI = PlaneData420(img16: decImg16I)
        
        // P-frame
        let bw = (width + 31) / 32
        let bh = (height + 31) / 32
        
        var counters = [Int](repeating: 0, count: bw * bh)
        let syntaxContext = SyntaxContextModels()
        let (bytesP, encReconP, _, _, relEncP, _, _, _) = try await encodeSpatialLayersForProfile2(
            pd: pd1, pool: pool, predictedPd: encReconI, nextPd: encReconI, prevInput: pd1, ltrInput: encReconI, prevMVs: nil, maxbitrate: 500*1024, qtY: qtY, qtC: qtC, zeroThreshold: 5, roundOffset: 0, gopPosition: 2, ltrAge: 1, skipThreshold: 2, reconThresholdScale: 1, staticCounters: &counters, cachedNextSub2: nil, cachedNextSub1: nil, entropyHistories: nil, syntaxContext: syntaxContext, l0State: l0StateEnc, l2Cadence: 1, l1Cadence: 1, l0Cadence: 1, framerate: 30, motionMaskingPx: 0, adjustedStep: Int(qtY.step), smooth: 0, updateL0Prev: true, skipModel: nil, ransContextWorkspace: nil, dumpWriter: nil)
        defer { relEncP() }
        
        let decImg16P = try await decodeSpatialLayersForProfile2(r: bytesP, pool: pool, maxLayer: 2, dx: width, dy: height, predictedPd: decReconI, nextPd: nil, roundOffset: 0, entropyHistories: nil, l0State: l0StateDec, mvState: nil, parallelEntropy: true, updateL0Prev: true, ransContextWorkspace: nil)
        let decReconP = PlaneData420(img16: decImg16P)
        
        XCTAssertEqual(encReconP.y, decReconP.y)
        XCTAssertEqual(encReconP.cb, decReconP.cb)
        XCTAssertEqual(encReconP.cr, decReconP.cr)
    }

    // 5. Skip ゼロ素材（ランダムノイズなど）での中立性（Profile 0x01 の出力とサイズ・品質が同等。SkipMap ヘッダ等による数バイト〜数十バイトの微増は適正）。
    func testProfile0x02Neutrality() async throws {
        let width = 320
        let height = 240
        var frames = [YCbCrImage]()
        for _ in 0..<5 {
            var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
            for i in 0..<(width * height) {
                img.yPlane[i] = UInt8.random(in: 0...255)
            }
            for i in 0..<(width * height / 4) {
                img.cbPlane[i] = UInt8.random(in: 0...255)
                img.crPlane[i] = UInt8.random(in: 0...255)
            }
            frames.append(img)
        }
        
        // エントロピー段とSkip機構のオーバーヘッド中立性を計測するため、
        // sceneChangeThreshold に maxEstimateFastSAD を超える値を指定してシーン検出を無効化し、
        // cadence を 1 にして間引きなしの同等レイヤー条件で P1/P2 のフレームタイプ・構造を揃える。
        let encoder01 = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 10, sceneChangeThreshold: 1_000_000_000, profile: 0x01)
        let bitstream01 = try await encoder01.encodeToData(images: frames)
        
        let encoder02 = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 10, sceneChangeThreshold: 1_000_000_000, profile: 0x02, l2Cadence: 1, l1Cadence: 1)
        let bitstream02 = try await encoder02.encodeToData(images: frames)
        
        let diff = abs(bitstream01.count - bitstream02.count)
        XCTAssertLessThan(diff, bitstream01.count / 50)
    }

    // 6. 決定論（同じ素材なら完全に同じバイトストリームが出力されること）。
    func testProfile0x02Determinism() async throws {
        let width = 320
        let height = 240
        var frames = [YCbCrImage]()
        for _ in 0..<3 {
            var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
            for i in 0..<(width * height) { img.yPlane[i] = UInt8(i % 256) }
            for i in 0..<(width * height / 4) {
                img.cbPlane[i] = 128
                img.crPlane[i] = 128
            }
            frames.append(img)
        }
        
        let encoder1 = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 10, profile: 0x02)
        let bitstream1 = try await encoder1.encodeToData(images: frames)
        
        let encoder2 = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 10, profile: 0x02)
        let bitstream2 = try await encoder2.encodeToData(images: frames)
        
        XCTAssertEqual(bitstream1, bitstream2)
    }

    // 7. rANSContext の可逆性検証（有効時と無効時でデコード画素が 100% 完全一致すること）
    func testRANSContextBitExactReconstruction() async throws {
        let width = 320
        let height = 240
        var frames = [YCbCrImage]()
        for f in 0..<5 {
            var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
            for i in 0..<(width * height) {
                img.yPlane[i] = UInt8(((i * 7) + (f * 13)) % 256)
            }
            for i in 0..<(width * height / 4) {
                img.cbPlane[i] = UInt8(((i * 3) + (f * 5)) % 256)
                img.crPlane[i] = UInt8(((i * 5) + (f * 7)) % 256)
            }
            frames.append(img)
        }

        // 1. rANSContext 無効時
        let encoderBase = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 10, profile: 0x02)
        let bitstreamBase = try await encoderBase.encodeToData(images: frames)
        let decoderBase = Decoder(maxLayer: 2)
        let outBase = try await decoderBase.decode(data: bitstreamBase)

        // 2. rANSContext 有効時
        let encoderModel = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 10, profile: 0x02, ransContext: 1)
        let bitstreamModel = try await encoderModel.encodeToData(images: frames)
        let decoderModel = Decoder(maxLayer: 2)
        let outModel = try await decoderModel.decode(data: bitstreamModel)

        // リセット

        XCTAssertEqual(outBase.count, outModel.count)
        for f in 0..<outBase.count {
            for i in 0..<outBase[f].yPlane.count {
                if outBase[f].yPlane[i] != outModel[f].yPlane[i] {
                    XCTFail("Frame \(f) yPlane mismatch at pixel \(i): base=\(outBase[f].yPlane[i]), model=\(outModel[f].yPlane[i])")
                    return
                }
            }
            for i in 0..<outBase[f].cbPlane.count {
                if outBase[f].cbPlane[i] != outModel[f].cbPlane[i] {
                    XCTFail("Frame \(f) cbPlane mismatch at pixel \(i): base=\(outBase[f].cbPlane[i]), model=\(outModel[f].cbPlane[i])")
                    return
                }
            }
            for i in 0..<outBase[f].crPlane.count {
                if outBase[f].crPlane[i] != outModel[f].crPlane[i] {
                    XCTFail("Frame \(f) crPlane mismatch at pixel \(i): base=\(outBase[f].crPlane[i]), model=\(outModel[f].crPlane[i])")
                    return
                }
            }
        }
    }

    func testRDDiagnostic() async throws {
        let ws = rANSContextWorkspace()
        let coeffs: [Int16] = [10, 0, 0, 0,  2, 1, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0]
        let c4h = coeffs.withUnsafeBufferPointer { estimate4HTailBits(blockCoeffs: $0.baseAddress!) }
        let cModel = coeffs.withUnsafeBufferPointer {
            ws.estimateModelTailBits(
                blockCoeffs: $0.baseAddress!,
                topCoeffs: nil,
                leftCoeffs: nil,
                tempCoeffs: nil,
                isPFrame: true,
                plane: 0,
                qstep: 4096
            )
        }
        print("=== Diagnostic: 4H=\(c4h) bits, Model=\(cModel) bits ===")
        XCTAssertLessThan(0, c4h)
        XCTAssertLessThan(0, cModel)
    }

    func testRealVideoLossless() async throws {
        let sequences: [(String, String)] = [
            ("tos", "/Users/yusuke.hata/Downloads/tos.y4m"),
            ("miko700", "/Users/yusuke.hata/workspace/vevc/.tmp/miko_700.y4m"),
            ("miko1", "/Users/yusuke.hata/Downloads/miko1.y4m"),
            ("src_1", "/Users/yusuke.hata/Downloads/src_1.y4m")
        ]

        var testedAny = false
        for (name, path) in sequences {
            guard let handle = FileHandle(forReadingAtPath: path) else {
                print("Skipping \(name), path not found: \(path)")
                continue
            }
            defer { try? handle.close() }

            let reader = try Y4MReader(fileHandle: handle)
            var frames: [YCbCrImage] = []
            for _ in 0..<10 {
                if let img = try reader.readFrame() {
                    frames.append(img)
                }
            }
            if frames.isEmpty { continue }
            testedAny = true

            // 1. Base
            let encBase = VEVCEncoder(width: frames[0].width, height: frames[0].height, qstep: 16, keyint: 10, profile: 0x02)
            let dataBase = try await encBase.encodeToData(images: frames)
            let decBase = Decoder(maxLayer: 2)
            let outBase = try await decBase.decode(data: dataBase)

            // 2. rANSContext
            let encModel = VEVCEncoder(width: frames[0].width, height: frames[0].height, qstep: 16, keyint: 10, profile: 0x02, ransContext: 1)
            let dataModel = try await encModel.encodeToData(images: frames)
            let decModel = Decoder(maxLayer: 2)
            let outModel = try await decModel.decode(data: dataModel)


            print("[\(name)] Base: \(dataBase.count) B, Model: \(dataModel.count) B")
            XCTAssertEqual(outBase.count, outModel.count)
            for f in 0..<outBase.count {
                for i in 0..<outBase[f].yPlane.count {
                    if outBase[f].yPlane[i] != outModel[f].yPlane[i] {
                        XCTFail("[\(name)] Frame \(f) Y mismatch at pixel \(i): base=\(outBase[f].yPlane[i]), model=\(outModel[f].yPlane[i])")
                        return
                    }
                }
                for i in 0..<outBase[f].cbPlane.count {
                    if outBase[f].cbPlane[i] != outModel[f].cbPlane[i] {
                        XCTFail("[\(name)] Frame \(f) Cb mismatch at pixel \(i): base=\(outBase[f].cbPlane[i]), model=\(outModel[f].cbPlane[i])")
                        return
                    }
                }
                for i in 0..<outBase[f].crPlane.count {
                    if outBase[f].crPlane[i] != outModel[f].crPlane[i] {
                        XCTFail("[\(name)] Frame \(f) Cr mismatch at pixel \(i): base=\(outBase[f].crPlane[i]), model=\(outModel[f].crPlane[i])")
                        return
                    }
                }
            }
        }
        if testedAny != true {
            throw XCTSkip("No test video sequences found on disk")
        }
    }

    func testBitrateAllSequences() async throws {
        let sequences: [(String, String, Int)] = [
            ("src_1", "/Users/yusuke.hata/Downloads/src_1.y4m", 100),
            ("tos", "/Users/yusuke.hata/Downloads/tos.y4m", 100),
            ("miko700", "/Users/yusuke.hata/workspace/vevc/.tmp/miko_700.y4m", 100),
            ("miko1", "/Users/yusuke.hata/Downloads/miko1.y4m", 100)
        ]

        var testedAny = false
        for (name, path, limit) in sequences {
            guard let handle = FileHandle(forReadingAtPath: path) else {
                print("Skipping \(name), path not found: \(path)")
                continue
            }
            defer { try? handle.close() }

            let reader = try Y4MReader(fileHandle: handle)
            var frames: [YCbCrImage] = []
            for _ in 0..<limit {
                if let img = try reader.readFrame() {
                    frames.append(img)
                }
            }
            if frames.isEmpty { continue }
            testedAny = true

            // 1. Base
            let encBase = VEVCEncoder(width: frames[0].width, height: frames[0].height, maxbitrate: 500_000, framerate: 30, profile: 0x02)
            let dataBase = try await encBase.encodeToData(images: frames)

            // 2. rANSContext
            let encModel = VEVCEncoder(width: frames[0].width, height: frames[0].height, maxbitrate: 500_000, framerate: 30, profile: 0x02, ransContext: 1)
            let dataModel = try await encModel.encodeToData(images: frames)


            XCTAssertLessThan(0, dataBase.count)
            XCTAssertLessThan(0, dataModel.count)

            let diff = dataModel.count - dataBase.count
            let pct = Double(diff) / Double(dataBase.count) * 100.0
            print("=== [\(name)] \(limit) frames Base: \(dataBase.count) B, Model: \(dataModel.count) B (diff: \(diff) B, \(String(format: "%.3f", pct))%) ===")
        }
        if testedAny != true {
            throw XCTSkip("No test video sequences found on disk")
        }
    }
}

