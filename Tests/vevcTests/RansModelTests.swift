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
        let pd = toPlaneData420(images: [img])[0]
        let qtY = QuantizationTable(baseStep: 2)
        var (blocks, _) = await extractSingleTransformBlocks32(r: pd.rY, width: width, height: height, pool: pool)
        for i in blocks.indices { evaluateQuantizeLayer32(block: &blocks[i], qt: qtY) }
        
        let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        for i in blocks.indices {
            let isZero = isEffectivelyZero32(data: blocks[i].base, threshold: safeThreshold)
            if isZero { continue }
            let view = blocks[i]
            let subs = getSubbands32(view: view)
            blockEncode16(encoder: &encoder, block: subs.hl, parentBlock: nil)
            blockEncode16(encoder: &encoder, block: subs.lh, parentBlock: nil)
            blockEncode16(encoder: &encoder, block: subs.hh, parentBlock: nil)
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
        runModel.normalize(sigCounts: [0, 1], tokenCounts: runTokenCounts)
        var valModel = rANSModel()
        valModel.normalize(sigCounts: [0, 1], tokenCounts: valTokenCounts)
        
        // freq table serialize→deserialize
        var runOut = [UInt8]()
        writeCompressedFreqTableTest(&runOut, freqs: runModel.tokenFreqs)
        var runOffset = 0
        let runFreqsRestored = try readCompressedFreqTableTest(runOut, at: &runOffset)
        
        var valOut = [UInt8]()
        writeCompressedFreqTableTest(&valOut, freqs: valModel.tokenFreqs)
        var valOffset = 0
        let valFreqsRestored = try readCompressedFreqTableTest(valOut, at: &valOffset)
        
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
        print("=== Run freq sum: \(runSum) restored sum: \(resRunSum) RANS_SCALE=\(RANS_SCALE) ===")
        XCTAssertEqual(runSum, RANS_SCALE, "Run freq sum != RANS_SCALE")
        XCTAssertEqual(resRunSum, RANS_SCALE, "Restored run freq sum != RANS_SCALE")
        
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
        XCTAssertEqual(valSum, RANS_SCALE, "Val freq sum != RANS_SCALE")
        
        // LUT 検証
        let restoredRunModel = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: runFreqsRestored)
        let restoredValModel = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: valFreqsRestored)
        
        // cumFreqs 比較
        for i in 0..<64 {
            XCTAssertEqual(runModel.tokenCumFreqs[i], restoredRunModel.tokenCumFreqs[i], "Run cumFreq[\(i)]")
            XCTAssertEqual(valModel.tokenCumFreqs[i], restoredValModel.tokenCumFreqs[i], "Val cumFreq[\(i)]")
        }
    }
    
    private func writeCompressedFreqTableTest(_ out: inout [UInt8], freqs: [UInt32]) {
        var bitmap: UInt64 = 0
        for i in 0..<64 {
            if 1 < freqs[i] { bitmap |= UInt64(1) << i }
        }
        appendBE64(&out, bitmap)
        for i in 0..<64 {
            if (bitmap & (UInt64(1) << i)) != 0 {
                out.append(UInt8(truncatingIfNeeded: freqs[i] >> 8))
                out.append(UInt8(truncatingIfNeeded: freqs[i] & 0xFF))
            }
        }
    }
    
    private func readCompressedFreqTableTest(_ data: [UInt8], at offset: inout Int) throws -> [UInt32] {
        guard offset + 8 <= data.count else { throw DecodeError.insufficientData }
        let bitmap = (UInt64(data[offset]) << 56) | (UInt64(data[offset+1]) << 48) | (UInt64(data[offset+2]) << 40) | (UInt64(data[offset+3]) << 32) |
                     (UInt64(data[offset+4]) << 24) | (UInt64(data[offset+5]) << 16) | (UInt64(data[offset+6]) << 8) | UInt64(data[offset+7])
        offset += 8
        var freqs = [UInt32](repeating: 1, count: 64)
        for i in 0..<64 {
            if (bitmap & (UInt64(1) << i)) != 0 {
                guard offset + 2 <= data.count else { throw DecodeError.insufficientData }
                freqs[i] = (UInt32(data[offset]) << 8) | UInt32(data[offset + 1])
                offset += 2
            }
        }
        return freqs
    }
    
    private func appendBE64(_ out: inout [UInt8], _ val: UInt64) {
        out.append(UInt8((val >> 56) & 0xFF))
        out.append(UInt8((val >> 48) & 0xFF))
        out.append(UInt8((val >> 40) & 0xFF))
        out.append(UInt8((val >> 32) & 0xFF))
        out.append(UInt8((val >> 24) & 0xFF))
        out.append(UInt8((val >> 16) & 0xFF))
        out.append(UInt8((val >> 8) & 0xFF))
        out.append(UInt8(val & 0xFF))
    }
}
