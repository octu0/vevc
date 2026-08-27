import XCTest
@testable import vevc

/// Milestone 3 閉ループ SNN 推論および DPCM 後方切り詰めに対する敵対的・実験的ストレステスト
final class DPCMTruncationEmpiricalStressTests: XCTestCase {

    // MARK: - 1. SNNPureIntegerPredictor の極端値・オーバーフロー耐性および決定性検証

    /// 全ゼロ入力における SNN 純整数推論の決定性とゼロ出力完全性
    func testSNNPredictorAllZeroBoundary() {
        let predictor = SNNPureIntegerPredictor()
        let scanOrders = [DPCMScanOrder.raster, DPCMScanOrder.serpentine]
        let kValues = [4, 6, 8, 12]

        for scan in scanOrders {
            for k in kValues {
                let transmitted = [Int16](repeating: 0, count: k)
                let context = DPCMPredictContext(
                    k: k,
                    scanOrder: scan,
                    qLow: 0,
                    lastVal: 0,
                    topBoundary: (0, 0, 0, 0),
                    leftBoundary: (0, 0, 0, 0),
                    topLeftBoundary: 0,
                    mcPred: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                )

                var output1 = [Int16](repeating: -1, count: 16)
                var output2 = [Int16](repeating: -2, count: 16)

                transmitted.withUnsafeBufferPointer { kPtr in
                    output1.withUnsafeMutableBufferPointer { oPtr in
                        if let kBase = kPtr.baseAddress, let oBase = oPtr.baseAddress {
                            predictor.predictBlock(transmittedK: kBase, context: context, output: oBase)
                        }
                    }
                    output2.withUnsafeMutableBufferPointer { oPtr in
                        if let kBase = kPtr.baseAddress, let oBase = oPtr.baseAddress {
                            predictor.predictBlock(transmittedK: kBase, context: context, output: oBase)
                        }
                    }
                }

                // 全 16 要素が完全に 0 かつ 2 回実行で完全一致
                for i in 0..<16 {
                    XCTAssertEqual(output1[i], 0, "All-zero 出力ゼロ検証: scan=\(scan), k=\(k), idx=\(i)")
                    XCTAssertEqual(output1[i], output2[i], "All-zero 決定性検証: scan=\(scan), k=\(k), idx=\(i)")
                }
            }
        }
    }

    /// Int16.max および Int16.min 極大・極小振幅におけるオーバーフロー耐性と決定性検証
    func testSNNPredictorExtremeAmplitudeBoundaries() {
        let predictor = SNNPureIntegerPredictor()
        let extremeValues: [Int16] = [Int16.max, Int16.min, 16384, -16384, 30000, -30000]
        let scanOrders = [DPCMScanOrder.raster, DPCMScanOrder.serpentine]
        let kValues = [4, 6, 8, 12]

        for extVal in extremeValues {
            for scan in scanOrders {
                for k in kValues {
                    let transmitted = [Int16](repeating: extVal, count: k)
                    let context = DPCMPredictContext(
                        k: k,
                        scanOrder: scan,
                        qLow: 255,
                        lastVal: extVal,
                        topBoundary: (extVal, extVal, extVal, extVal),
                        leftBoundary: (extVal, extVal, extVal, extVal),
                        topLeftBoundary: extVal,
                        mcPred: (
                            extVal, extVal, extVal, extVal,
                            extVal, extVal, extVal, extVal,
                            extVal, extVal, extVal, extVal,
                            extVal, extVal, extVal, extVal
                        )
                    )

                    var output1 = [Int16](repeating: 0, count: 16)
                    var output2 = [Int16](repeating: 0, count: 16)

                    transmitted.withUnsafeBufferPointer { kPtr in
                        output1.withUnsafeMutableBufferPointer { oPtr in
                            if let kBase = kPtr.baseAddress, let oBase = oPtr.baseAddress {
                                predictor.predictBlock(transmittedK: kBase, context: context, output: oBase)
                            }
                        }
                        output2.withUnsafeMutableBufferPointer { oPtr in
                            if let kBase = kPtr.baseAddress, let oBase = oPtr.baseAddress {
                                predictor.predictBlock(transmittedK: kBase, context: context, output: oBase)
                            }
                        }
                    }

                    // 1. 送信済み K 要素の完全一致
                    for i in 0..<k {
                        let pt = scan.toXY(index: i)
                        let rawIdx = pt.y * 4 + pt.x
                        XCTAssertEqual(output1[rawIdx], extVal, "極端値 K 要素保持検証: val=\(extVal), scan=\(scan), k=\(k), idx=\(i)")
                    }

                    // 2. 2回実行の完全一致（決定論的ビット同一性）
                    for i in 0..<16 {
                        XCTAssertEqual(output1[i], output2[i], "極端値 2回実行決定性検証: val=\(extVal), scan=\(scan), k=\(k), idx=\(i)")
                    }
                }
            }
        }
    }

    /// チェッカーボードパターン（市松模様）および高周波急峻勾配における推論検証
    func testSNNPredictorCheckerboardAndHighFrequency() {
        let predictor = SNNPureIntegerPredictor()
        let scanOrders = [DPCMScanOrder.raster, DPCMScanOrder.serpentine]

        for scan in scanOrders {
            let k = 6
            var transmitted = [Int16](repeating: 0, count: k)
            for i in 0..<k {
                let pt = scan.toXY(index: i)
                let isEven = ((pt.x + pt.y) & 1) == 0
                if isEven {
                    transmitted[i] = 1000
                } else {
                    transmitted[i] = -1000
                }
            }

            let top = (Int16(1000), Int16(-1000), Int16(1000), Int16(-1000))
            let left = (Int16(1000), Int16(-1000), Int16(1000), Int16(-1000))
            let mc = (
                Int16(1000), Int16(-1000), Int16(1000), Int16(-1000),
                Int16(-1000), Int16(1000), Int16(-1000), Int16(1000),
                Int16(1000), Int16(-1000), Int16(1000), Int16(-1000),
                Int16(-1000), Int16(1000), Int16(-1000), Int16(1000)
            )

            let context = DPCMPredictContext(
                k: k,
                scanOrder: scan,
                qLow: 32,
                lastVal: 1000,
                topBoundary: top,
                leftBoundary: left,
                topLeftBoundary: -1000,
                mcPred: mc
            )

            var output1 = [Int16](repeating: 0, count: 16)
            var output2 = [Int16](repeating: 0, count: 16)

            transmitted.withUnsafeBufferPointer { kPtr in
                output1.withUnsafeMutableBufferPointer { oPtr in
                    if let kBase = kPtr.baseAddress, let oBase = oPtr.baseAddress {
                        predictor.predictBlock(transmittedK: kBase, context: context, output: oBase)
                    }
                }
                output2.withUnsafeMutableBufferPointer { oPtr in
                    if let kBase = kPtr.baseAddress, let oBase = oPtr.baseAddress {
                        predictor.predictBlock(transmittedK: kBase, context: context, output: oBase)
                    }
                }
            }

            for i in 0..<k {
                let pt = scan.toXY(index: i)
                let rawIdx = pt.y * 4 + pt.x
                XCTAssertEqual(output1[rawIdx], transmitted[i], "チェッカーボード K 要素一致検証")
            }

            for i in 0..<16 {
                XCTAssertEqual(output1[i], output2[i], "チェッカーボード 2回実行決定性検証")
            }
        }
    }

    /// 疑似乱数ストレスファジングによる 200 パターンの SNN 純整数推論の決定性検証
    func testSNNPredictorFuzzingDeterminism() {
        let predictor = SNNPureIntegerPredictor()
        var rngState: UInt64 = 0xDEADBEEFCAFE1234

        func nextRandomInt16(minVal: Int32 = -2000, maxVal: Int32 = 2000) -> Int16 {
            rngState = rngState &* 6364136223846793005 &+ 1442695040888963407
            let span = maxVal - minVal + 1
            let raw = Int32(truncatingIfNeeded: (rngState >> 32) & 0x7FFFFFFF)
            return Int16(clamping: minVal + (raw % span))
        }

        let kOptions = [4, 6, 8, 12]
        let scanOptions = [DPCMScanOrder.raster, DPCMScanOrder.serpentine]

        for _ in 0..<200 {
            let k = kOptions[Int(nextRandomInt16(minVal: 0, maxVal: 3))]
            let scan = scanOptions[Int(nextRandomInt16(minVal: 0, maxVal: 1))]

            var transmitted = [Int16](repeating: 0, count: k)
            for i in 0..<k {
                transmitted[i] = nextRandomInt16()
            }

            let top = (nextRandomInt16(), nextRandomInt16(), nextRandomInt16(), nextRandomInt16())
            let left = (nextRandomInt16(), nextRandomInt16(), nextRandomInt16(), nextRandomInt16())
            let tl = nextRandomInt16()
            let mc = (
                nextRandomInt16(), nextRandomInt16(), nextRandomInt16(), nextRandomInt16(),
                nextRandomInt16(), nextRandomInt16(), nextRandomInt16(), nextRandomInt16(),
                nextRandomInt16(), nextRandomInt16(), nextRandomInt16(), nextRandomInt16(),
                nextRandomInt16(), nextRandomInt16(), nextRandomInt16(), nextRandomInt16()
            )

            let context = DPCMPredictContext(
                k: k,
                scanOrder: scan,
                qLow: Int16(abs(Int32(nextRandomInt16(minVal: 1, maxVal: 128)))),
                lastVal: nextRandomInt16(),
                topBoundary: top,
                leftBoundary: left,
                topLeftBoundary: tl,
                mcPred: mc
            )

            var out1 = [Int16](repeating: 0, count: 16)
            var out2 = [Int16](repeating: 0, count: 16)

            transmitted.withUnsafeBufferPointer { kPtr in
                out1.withUnsafeMutableBufferPointer { oPtr in
                    if let kBase = kPtr.baseAddress, let oBase = oPtr.baseAddress {
                        predictor.predictBlock(transmittedK: kBase, context: context, output: oBase)
                    }
                }
                out2.withUnsafeMutableBufferPointer { oPtr in
                    if let kBase = kPtr.baseAddress, let oBase = oPtr.baseAddress {
                        predictor.predictBlock(transmittedK: kBase, context: context, output: oBase)
                    }
                }
            }

            for i in 0..<k {
                let pt = scan.toXY(index: i)
                XCTAssertEqual(out1[pt.y * 4 + pt.x], transmitted[i])
            }
            for i in 0..<16 {
                XCTAssertEqual(out1[i], out2[i])
            }
        }
    }

    // MARK: - 2. 1bit モードフラグ構文の境界テスト (全フォールバック / 全省略 / 混在)

    /// 全ブロックフォールバック (Force Mode 0) における連続ブロックのエンコード・デコード完全一致
    func testAllFallbackBlocksRoundtrip() throws {
        // 50 ブロックの急峻なランダム残差を生成し、epsilon=0 かつ複雑なパターンで確実にフォールバックを強制
        let blockCount = 50
        var sourceBlocks = [[Int16]]()
        var rng: UInt64 = 0x5432167890ABCDEF

        for _ in 0..<blockCount {
            var blk = [Int16](repeating: 0, count: 16)
            for i in 0..<16 {
                rng = rng &* 6364136223846793005 &+ 1
                let v = Int32(truncatingIfNeeded: rng >> 32) % 300
                blk[i] = Int16(clamping: v)
            }
            sourceBlocks.append(blk)
        }

        var encBlocks = sourceBlocks
        var encLastVal: Int16 = 0
        var encoder = EntropyEncoder()

        for b in 0..<blockCount {
            encBlocks[b].withUnsafeMutableBufferPointer { ptr in
                let view = BlockView(base: ptr.baseAddress!, width: 4, height: 4, stride: 4)
                // Mode 0 (Fallback) を直接呼び出し、全ブロック Mode 0 のビットストリームをシミュレート
                encoder.encodeBypass(binVal: 0)
                blockEncodeDPCM4Baseline(
                    encoder: &encoder,
                    block: view,
                    lastVal: &encLastVal
                )
            }
        }
        encoder.flush()

        let encodedData = encoder.getData(selectModel: unifiedSelectModel)

        // 復号
        var decBlocks = [[Int16]](repeating: [Int16](repeating: 0, count: 16), count: blockCount)
        var decLastVal: Int16 = 0

        try encodedData.withUnsafeBufferPointer { bufPtr in
            guard let base = bufPtr.baseAddress else { return }
            var decoder = try EntropyDecoder(base: base, count: bufPtr.count, startOffset: 0, history: nil, parentFreeStatics: false)

            for b in 0..<blockCount {
                let modeFlag = try decoder.decodeBypass()
                XCTAssertEqual(modeFlag, 0, "ブロック \(b) の modeFlag は 0 (Fallback) であるべき")
                try decBlocks[b].withUnsafeMutableBufferPointer { decPtr in
                    try blockDecodeDPCM4Baseline(
                        decoder: &decoder,
                        ptr: decPtr.baseAddress!,
                        stride: 4,
                        lastVal: &decLastVal
                    )
                }
            }
        }

        // エンコード後の再構成面とデコード結果が 100% 一致し、lastVal も同期
        for b in 0..<blockCount {
            for i in 0..<16 {
                XCTAssertEqual(decBlocks[b][i], encBlocks[b][i], "全フォールバック Block \(b), Idx \(i) 一致検証")
            }
        }
        XCTAssertEqual(decLastVal, encLastVal, "全フォールバック 最終 lastVal 同期検証")
    }

    /// 全ブロック後方省略 (Force Mode 1) における SNN 再構成面のエンコード・デコード完全一致
    func testAllTruncatedBlocksRoundtrip() throws {
        let k = 6
        let scanOrder = DPCMScanOrder.serpentine
        let predictor = SNNPureIntegerPredictor()
        let blockCount = 50

        var sourceBlocks = [[Int16]]()
        var rng: UInt64 = 0xFEEDFACE12345678

        for _ in 0..<blockCount {
            var blk = [Int16](repeating: 0, count: 16)
            for i in 0..<16 {
                rng = rng &* 6364136223846793005 &+ 1
                let v = Int32(truncatingIfNeeded: rng >> 32) % 200
                blk[i] = Int16(clamping: v)
            }
            sourceBlocks.append(blk)
        }

        var encReconBlocks = [[Int16]](repeating: [Int16](repeating: 0, count: 16), count: blockCount)
        var encLastVal: Int16 = 0
        var encoder = EntropyEncoder()

        for b in 0..<blockCount {
            var transmittedK = [Int16](repeating: 0, count: k)
            for i in 0..<k {
                let pt = scanOrder.toXY(index: i)
                transmittedK[i] = sourceBlocks[b][pt.y * 4 + pt.x]
            }

            let context = DPCMPredictContext(
                k: k,
                scanOrder: scanOrder,
                qLow: 16,
                lastVal: encLastVal,
                topBoundary: (0, 0, 0, 0),
                leftBoundary: (0, 0, 0, 0),
                topLeftBoundary: 0,
                mcPred: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            )

            var snnRecon = [Int16](repeating: 0, count: 16)
            transmittedK.withUnsafeBufferPointer { kPtr in
                snnRecon.withUnsafeMutableBufferPointer { outPtr in
                    if let kBase = kPtr.baseAddress, let outBase = outPtr.baseAddress {
                        predictor.predictBlock(transmittedK: kBase, context: context, output: outBase)
                    }
                }
            }

            // Mode 1 エンコード
            encoder.encodeBypass(binVal: 1)
            encodeTruncatedKSubblock(
                encoder: &encoder,
                transmittedK: transmittedK,
                k: k,
                scanOrder: scanOrder,
                lastVal: encLastVal
            )

            encReconBlocks[b] = snnRecon
            encLastVal = snnRecon[15]
        }
        encoder.flush()

        let encodedData = encoder.getData(selectModel: unifiedSelectModel)

        // 復号
        var decReconBlocks = [[Int16]](repeating: [Int16](repeating: 0, count: 16), count: blockCount)
        var decLastVal: Int16 = 0

        try encodedData.withUnsafeBufferPointer { bufPtr in
            guard let base = bufPtr.baseAddress else { return }
            var decoder = try EntropyDecoder(base: base, count: bufPtr.count, startOffset: 0, history: nil, parentFreeStatics: false)

            for b in 0..<blockCount {
                let modeFlag = try decoder.decodeBypass()
                XCTAssertEqual(modeFlag, 1, "ブロック \(b) の modeFlag は 1 (Truncated) であるべき")

                var decodedK = [Int16](repeating: 0, count: k)
                try decodeTruncatedKSubblock(
                    decoder: &decoder,
                    transmittedK: &decodedK,
                    k: k,
                    scanOrder: scanOrder,
                    lastVal: decLastVal
                )

                let context = DPCMPredictContext(
                    k: k,
                    scanOrder: scanOrder,
                    qLow: 16,
                    lastVal: decLastVal,
                    topBoundary: (0, 0, 0, 0),
                    leftBoundary: (0, 0, 0, 0),
                    topLeftBoundary: 0,
                    mcPred: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                )

                var snnRecon = [Int16](repeating: 0, count: 16)
                decodedK.withUnsafeBufferPointer { kPtr in
                    snnRecon.withUnsafeMutableBufferPointer { outPtr in
                        if let kBase = kPtr.baseAddress, let outBase = outPtr.baseAddress {
                            predictor.predictBlock(transmittedK: kBase, context: context, output: outBase)
                        }
                    }
                }

                decReconBlocks[b] = snnRecon
                decLastVal = snnRecon[15]
            }
        }

        // エンコーダ側 SNN 再構成面とデコーダ側 SNN 再構成面の 100% 完全一致
        for b in 0..<blockCount {
            for i in 0..<16 {
                XCTAssertEqual(decReconBlocks[b][i], encReconBlocks[b][i], "全省略 Block \(b), Idx \(i) SNN 再構成面一致検証")
            }
        }
        XCTAssertEqual(decLastVal, encLastVal, "全省略 最終 lastVal 同期検証")
    }

    /// Mode 0 (Fallback) と Mode 1 (Truncated) が混在するストリームの境界・連鎖同期検証
    func testMixedModeAlternatingRoundtrip() throws {
        let k = 6
        let scanOrder = DPCMScanOrder.serpentine
        let predictor = SNNPureIntegerPredictor()
        let blockCount = 60

        var sourceBlocks = [[Int16]]()
        var rng: UInt64 = 0xABCDEF0123456789

        for _ in 0..<blockCount {
            var blk = [Int16](repeating: 0, count: 16)
            for i in 0..<16 {
                rng = rng &* 6364136223846793005 &+ 1
                let v = Int32(truncatingIfNeeded: rng >> 32) % 150
                blk[i] = Int16(clamping: v)
            }
            sourceBlocks.append(blk)
        }

        var encReconBlocks = sourceBlocks
        var encLastVal: Int16 = 50
        var encoder = EntropyEncoder()

        var modeSequence = [UInt8]()
        for b in 0..<blockCount {
            let useTrunc = (b % 3) != 0 // 2/3 Truncated, 1/3 Fallback
            if useTrunc {
                modeSequence.append(1)
                var transmittedK = [Int16](repeating: 0, count: k)
                for i in 0..<k {
                    let pt = scanOrder.toXY(index: i)
                    transmittedK[i] = sourceBlocks[b][pt.y * 4 + pt.x]
                }

                let context = DPCMPredictContext(
                    k: k,
                    scanOrder: scanOrder,
                    qLow: 8,
                    lastVal: encLastVal,
                    topBoundary: (0, 0, 0, 0),
                    leftBoundary: (0, 0, 0, 0),
                    topLeftBoundary: 0,
                    mcPred: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                )

                var snnRecon = [Int16](repeating: 0, count: 16)
                transmittedK.withUnsafeBufferPointer { kPtr in
                    snnRecon.withUnsafeMutableBufferPointer { outPtr in
                        if let kBase = kPtr.baseAddress, let outBase = outPtr.baseAddress {
                            predictor.predictBlock(transmittedK: kBase, context: context, output: outBase)
                        }
                    }
                }

                encoder.encodeBypass(binVal: 1)
                encodeTruncatedKSubblock(
                    encoder: &encoder,
                    transmittedK: transmittedK,
                    k: k,
                    scanOrder: scanOrder,
                    lastVal: encLastVal
                )

                for y in 0..<4 {
                    for x in 0..<4 {
                        encReconBlocks[b][y * 4 + x] = snnRecon[y * 4 + x]
                    }
                }
                encLastVal = snnRecon[15]
            } else {
                modeSequence.append(0)
                encoder.encodeBypass(binVal: 0)
                encReconBlocks[b].withUnsafeMutableBufferPointer { ptr in
                    let view = BlockView(base: ptr.baseAddress!, width: 4, height: 4, stride: 4)
                    blockEncodeDPCM4Baseline(
                        encoder: &encoder,
                        block: view,
                        lastVal: &encLastVal
                    )
                }
            }
        }
        encoder.flush()

        let encodedData = encoder.getData(selectModel: unifiedSelectModel)

        // 復号
        var decReconBlocks = [[Int16]](repeating: [Int16](repeating: 0, count: 16), count: blockCount)
        var decLastVal: Int16 = 50

        try encodedData.withUnsafeBufferPointer { bufPtr in
            guard let base = bufPtr.baseAddress else { return }
            var decoder = try EntropyDecoder(base: base, count: bufPtr.count, startOffset: 0, history: nil, parentFreeStatics: false)

            for b in 0..<blockCount {
                let modeFlag = try decoder.decodeBypass()
                XCTAssertEqual(modeFlag, modeSequence[b], "混在モード Block \(b) のフラグ一致検証")

                if modeFlag == 1 {
                    var decodedK = [Int16](repeating: 0, count: k)
                    try decodeTruncatedKSubblock(
                        decoder: &decoder,
                        transmittedK: &decodedK,
                        k: k,
                        scanOrder: scanOrder,
                        lastVal: decLastVal
                    )

                    let context = DPCMPredictContext(
                        k: k,
                        scanOrder: scanOrder,
                        qLow: 8,
                        lastVal: decLastVal,
                        topBoundary: (0, 0, 0, 0),
                        leftBoundary: (0, 0, 0, 0),
                        topLeftBoundary: 0,
                        mcPred: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                    )

                    var snnRecon = [Int16](repeating: 0, count: 16)
                    decodedK.withUnsafeBufferPointer { kPtr in
                        snnRecon.withUnsafeMutableBufferPointer { outPtr in
                            if let kBase = kPtr.baseAddress, let outBase = outPtr.baseAddress {
                                predictor.predictBlock(transmittedK: kBase, context: context, output: outBase)
                            }
                        }
                    }

                    decReconBlocks[b] = snnRecon
                    decLastVal = snnRecon[15]
                } else {
                    try decReconBlocks[b].withUnsafeMutableBufferPointer { decPtr in
                        try blockDecodeDPCM4Baseline(
                            decoder: &decoder,
                            ptr: decPtr.baseAddress!,
                            stride: 4,
                            lastVal: &decLastVal
                        )
                    }
                }
            }
        }

        // エンコーダ・デコーダ再構成面の完全ビット一致
        for b in 0..<blockCount {
            for i in 0..<16 {
                XCTAssertEqual(decReconBlocks[b][i], encReconBlocks[b][i], "混在モード Block \(b), Idx \(i) 一致検証")
            }
        }
        XCTAssertEqual(decLastVal, encLastVal, "混在モード 最終 lastVal 同期検証")
    }
}
