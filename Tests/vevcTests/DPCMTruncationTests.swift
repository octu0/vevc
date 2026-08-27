import XCTest
@testable import vevc

final class DPCMTruncationTests: XCTestCase {

    /// 走査順序および座標変換の決定性と一貫性を検証
    func testScanOrderCoordinates() {
        let raster = DPCMScanOrder.raster
        let serpentine = DPCMScanOrder.serpentine

        for i in 0..<16 {
            let rPt = raster.toXY(index: i)
            let rIdx = raster.toIndex(x: rPt.x, y: rPt.y)
            XCTAssertEqual(rIdx, i)

            let sPt = serpentine.toXY(index: i)
            let sIdx = serpentine.toIndex(x: sPt.x, y: sPt.y)
            XCTAssertEqual(sIdx, i)
        }
    }

    /// 純整数 SNN 予測器エンジンの決定論的完全一致および K 要素一致を検証
    func testSNNPureIntegerPredictorDeterminism() {
        let predictor = SNNPureIntegerPredictor()
        let k = 6
        let scanOrder = DPCMScanOrder.serpentine

        var transmitted = [Int16](repeating: 0, count: k)
        for i in 0..<k {
            transmitted[i] = Int16((i * 17) + 50)
        }

        let context = DPCMPredictContext(
            k: k,
            scanOrder: scanOrder,
            qLow: 64,
            lastVal: 48,
            topBoundary: (52, 54, 56, 58),
            leftBoundary: (51, 53, 55, 57),
            topLeftBoundary: 50,
            mcPred: (50, 51, 52, 53,
                    54, 55, 56, 57,
                    58, 59, 60, 61,
                    62, 63, 64, 65)
        )

        var output1 = [Int16](repeating: 0, count: 16)
        var output2 = [Int16](repeating: 0, count: 16)

        transmitted.withUnsafeBufferPointer { kPtr in
            output1.withUnsafeMutableBufferPointer { outPtr in
                if let kBase = kPtr.baseAddress, let outBase = outPtr.baseAddress {
                    predictor.predictBlock(transmittedK: kBase, context: context, output: outBase)
                }
            }
            output2.withUnsafeMutableBufferPointer { outPtr in
                if let kBase = kPtr.baseAddress, let outBase = outPtr.baseAddress {
                    predictor.predictBlock(transmittedK: kBase, context: context, output: outBase)
                }
            }
        }

        // 1. 同一入力に対する 2 回実行の bit-identical 完全一致検証
        for i in 0..<16 {
            XCTAssertEqual(output1[i], output2[i], "SNN 推論の決定論的 2 回実行完全一致テスト (index \(i))")
        }

        // 2. 送信済み K 要素が誤差ゼロで完全一致していることを検証
        for i in 0..<k {
            let pt = scanOrder.toXY(index: i)
            let rawIdx = pt.y * 4 + pt.x
            XCTAssertEqual(output1[rawIdx], transmitted[i], "送信 K 要素の誤差ゼロ強制一致検証 (index \(i))")
        }
    }

    /// 前方 K 要素の DPCM 差分符号化・復号の可逆性を検証
    func testTruncatedKSubblockRoundtrip() throws {
        let k = 6
        let scanOrder = DPCMScanOrder.serpentine
        let lastValInit: Int16 = 32

        var origK = [Int16](repeating: 0, count: k)
        for i in 0..<k {
            origK[i] = Int16(32 + i * 4)
        }

        // 符号化
        var encoder = EntropyEncoder()
        encodeTruncatedKSubblock(
            encoder: &encoder,
            transmittedK: origK,
            k: k,
            scanOrder: scanOrder,
            lastVal: lastValInit
        )
        encoder.flush()

        let encodedData = encoder.getData(selectModel: unifiedSelectModel)

        // 復号
        try encodedData.withUnsafeBufferPointer { bufPtr in
            guard let base = bufPtr.baseAddress else { return }
            var decoder = try EntropyDecoder(base: base, count: bufPtr.count, startOffset: 0, history: nil, parentFreeStatics: false)
            var decodedK = [Int16](repeating: 0, count: k)
            try decodeTruncatedKSubblock(
                decoder: &decoder,
                transmittedK: &decodedK,
                k: k,
                scanOrder: scanOrder,
                lastVal: lastValInit
            )

            for i in 0..<k {
                XCTAssertEqual(decodedK[i], origK[i], "K 要素 DPCM ラウンドトリップ検証 (index \(i))")
            }
        }
    }

    /// ベースライン DPCM エンコーダ・デコーダの閉ループ再構成一致および lastVal 同期を検証
    func testBaselineEncoderDecoderSynchronization() throws {
        var rawPixels: [Int16] = [
            120, 122, 124, 126,
            121, 123, 125, 127,
            122, 124, 126, 128,
            123, 125, 127, 129
        ]

        var encLastVal: Int16 = 118
        var decLastVal: Int16 = 118

        var encoder = EntropyEncoder()
        rawPixels.withUnsafeMutableBufferPointer { ptr in
            let view = BlockView(base: ptr.baseAddress!, width: 4, height: 4, stride: 4)
            blockEncodeDPCM4Baseline(
                encoder: &encoder,
                block: view,
                lastVal: &encLastVal,
                qLow: 16
            )
        }
        encoder.flush()

        let encodedData = encoder.getData(selectModel: unifiedSelectModel)

        var decPixels = [Int16](repeating: 0, count: 16)
        try encodedData.withUnsafeBufferPointer { bufPtr in
            guard let base = bufPtr.baseAddress else { return }
            var decoder = try EntropyDecoder(base: base, count: bufPtr.count, startOffset: 0, history: nil, parentFreeStatics: false)
            try decPixels.withUnsafeMutableBufferPointer { decPtr in
                try blockDecodeDPCM4Baseline(
                    decoder: &decoder,
                    ptr: decPtr.baseAddress!,
                    stride: 4,
                    lastVal: &decLastVal
                )
            }
        }

        // エンコーダ再構成面とデコーダ再構成面の完全ビット一致を検証
        for i in 0..<16 {
            XCTAssertEqual(decPixels[i], rawPixels[i], "DPCM Baseline 再構成画素の一致検証 (index \(i))")
        }
        // lastVal の完全同期を検証
        XCTAssertEqual(decLastVal, encLastVal, "DPCM Baseline lastVal 同期検証")
    }

    /// 全ゼロ残差ブロックにおけるエンコーダ・デコーダ閉ループ同期を検証
    func testAllZeroBlockSynchronization() throws {
        var rawPixels = [Int16](repeating: 100, count: 16)
        var encLastVal: Int16 = 100
        var decLastVal: Int16 = 100

        var encoder = EntropyEncoder()
        rawPixels.withUnsafeMutableBufferPointer { ptr in
            let view = BlockView(base: ptr.baseAddress!, width: 4, height: 4, stride: 4)
            blockEncodeDPCM4Baseline(
                encoder: &encoder,
                block: view,
                lastVal: &encLastVal,
                qLow: 16
            )
        }
        encoder.flush()

        let encodedData = encoder.getData(selectModel: unifiedSelectModel)

        var decPixels = [Int16](repeating: 0, count: 16)
        try encodedData.withUnsafeBufferPointer { bufPtr in
            guard let base = bufPtr.baseAddress else { return }
            var decoder = try EntropyDecoder(base: base, count: bufPtr.count, startOffset: 0, history: nil, parentFreeStatics: false)
            try decPixels.withUnsafeMutableBufferPointer { decPtr in
                try blockDecodeDPCM4Baseline(
                    decoder: &decoder,
                    ptr: decPtr.baseAddress!,
                    stride: 4,
                    lastVal: &decLastVal
                )
            }
        }

        for i in 0..<16 {
            XCTAssertEqual(decPixels[i], 100, "All-Zero ブロック再構成画素検証 (index \(i))")
        }
        XCTAssertEqual(decLastVal, encLastVal, "All-Zero ブロック lastVal 同期検証")
    }
}
