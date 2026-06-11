import XCTest
@testable import vevc

/// rANSModel のnormalize vs readback テスト
final class RansModelTests: XCTestCase {

    /// normalize後のtokenFreqsと、writeCompressedFreqTable→readCompressedFreqTableで復元したtokenFreqsを比較
    func testFreqTableRoundtrip() async throws {
        let pool = BlockViewPool()
        // 実DWTデータのtoken分布を再現
        var runTokenCounts = [Int](repeating: 0, count: 64)
        var valTokenCounts = [Int](repeating: 0, count: 64)

        // 実際のDWTデータからblock encodeしたpairsのtoken分布
        let width = 128
        let height = 128
        var img = YCbCrImage(width: width, height: height)
        for y in 0..<height {
            for x in 0..<width {
                let base = (x + y * 2) % 256
                let noise = (x &* 2654435761 ^ y &* 2246822519) % 20
                img.yPlane[y * width + x] = UInt8(clamping: base + noise - 10)
            }
        }
        let cW = (width + 1) / 2
        let cH = (height + 1) / 2
        for cy in 0..<cH {
            for cx in 0..<cW {
                img.cbPlane[cy * cW + cx] = 128
                img.crPlane[cy * cW + cx] = 128
            }
        }
        let pd = toPlaneData420(image: img, pool: BlockViewPool()).0
        let qtY = QuantizationTable(baseStep: 2)
        let (blocks, _, rel) = await extractSingleTransformBlocks32(r: pd.rY, width: width, height: height, pool: pool, qt: qtY)
        defer { rel() }
        for i in blocks.indices { evaluateQuantizeLayer32(view: blocks[i], qt: qtY) }

        let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
        var encoder = EntropyEncoder()
        for i in blocks.indices {
            let isZero = isEffectivelyZero32(data: blocks[i].base, threshold: safeThreshold)
            if isZero { continue }
            let view = blocks[i]
            let subs = getSubbands32(view: view)
            blockEncode16V(encoder: &encoder, block: subs.hl)
            blockEncode16H(encoder: &encoder, block: subs.lh)
            blockEncode16H(encoder: &encoder, block: subs.hh)
        }

        // pairs からrunTokenCountsとvalTokenCountsを計算
        for pair in encoder.pairs {
            let rt = valueTokenizeUnsigned(pair.run)
            runTokenCounts[Int(rt.token)] += 1
            let vt = valueTokenize(pair.val)
            valTokenCounts[Int(vt.token)] += 1
        }

        // normalize
        var runModel = rANSModel()
        runModel.normalize(tokenCounts: runTokenCounts)
        var valModel = rANSModel()
        valModel.normalize(tokenCounts: valTokenCounts)

        // freq table serialize→deserialize
        var runOut = [UInt8]()
        writeCompressedFreqTable(&runOut, freqs: runModel.tokenFreqs)
        var runOffset = 0
        let runFreqsRestored = try runOut.withUnsafeBufferPointer { buf -> [UInt32] in
            try EntropyDecoder.readCompressedFreqTable(buf.baseAddress!, at: &runOffset, count: runOut.count)
        }

        var valOut = [UInt8]()
        writeCompressedFreqTable(&valOut, freqs: valModel.tokenFreqs)
        var valOffset = 0
        let valFreqsRestored = try valOut.withUnsafeBufferPointer { buf -> [UInt32] in
            try EntropyDecoder.readCompressedFreqTable(buf.baseAddress!, at: &valOffset, count: valOut.count)
        }

        // 比較
        print("=== Run token counts: \(runTokenCounts) ===")
        print("=== Run tokenFreqs: \(runModel.tokenFreqs) ===")
        print("=== Run freqsRestored: \(runFreqsRestored) ===")

        var runSum: UInt32 = 0
        var resRunSum: UInt32 = 0
        for i in 0..<64 {
            XCTAssertEqual(runModel.tokenFreqs[i], runFreqsRestored[i], "Run freq[\(i)] mismatch")
            runSum += runModel.tokenFreqs[i]
            resRunSum += runFreqsRestored[i]
        }
        print("=== Run freq sum: \(runSum) restored sum: \(resRunSum) rANSScale=\(rANSScale) ===")
        XCTAssertEqual(runSum, rANSScale, "Run freq sum != rANSScale")
        XCTAssertEqual(resRunSum, rANSScale, "Restored run freq sum != rANSScale")

        print("=== Val token counts: \(valTokenCounts) ===")
        print("=== Val tokenFreqs: \(valModel.tokenFreqs) ===")
        var valSum: UInt32 = 0
        var resValSum: UInt32 = 0
        for i in 0..<64 {
            XCTAssertEqual(valModel.tokenFreqs[i], valFreqsRestored[i], "Val freq[\(i)] mismatch")
            valSum += valModel.tokenFreqs[i]
            resValSum += valFreqsRestored[i]
        }
        print("=== Val freq sum: \(valSum) restored sum: \(resValSum) ===")
        XCTAssertEqual(valSum, rANSScale, "Val freq sum != rANSScale")

        // LUT 検証
        let restoredRunModel = rANSModel(tokenFreqs: runFreqsRestored)
        let restoredValModel = rANSModel(tokenFreqs: valFreqsRestored)

        // cumFreqs 比較
        for i in 0..<64 {
            XCTAssertEqual(runModel.tokenCumFreqs[i], restoredRunModel.tokenCumFreqs[i], "Run cumFreq[\(i)]")
            XCTAssertEqual(valModel.tokenCumFreqs[i], restoredValModel.tokenCumFreqs[i], "Val cumFreq[\(i)]")
        }
    }

}
