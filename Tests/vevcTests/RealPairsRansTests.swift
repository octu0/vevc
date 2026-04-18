import XCTest
@testable import vevc

/// 実DWTデータのpairs配列を直接EntropyEncoderに渡してrANSラウンドトリップテスト
final class RealPairsRansTests: XCTestCase {
    
    private func generateRealPairs() async -> (pairs: [(run: UInt32, val: Int16, isParentZero: Bool)], bypass: [UInt8]) {
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
        let pool = BlockViewPool()
        
        var (blocks, _) = await extractSingleTransformBlocks32(r: pd.rY, width: width, height: height, pool: pool)
        for i in blocks.indices {
            evaluateQuantizeLayer32(block: blocks[i], qt: qtY)
        }
        
        let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
        
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        for i in blocks.indices {
            let isZero = isEffectivelyZero32(data: blocks[i].base, threshold: safeThreshold)
            if isZero { continue }
            
            let view = blocks[i]
            let subs = getSubbands32(view: view)
            blockEncode16V(encoder: &encoder, block: subs.hl, parentBlock: nil)
            blockEncode16H(encoder: &encoder, block: subs.lh, parentBlock: nil)
            blockEncode16H(encoder: &encoder, block: subs.hh, parentBlock: nil)
        }
        
        encoder.bypassWriter.flush()
        return (pairs: encoder.pairs, bypass: encoder.bypassWriter.bytes)
    }
    
    /// 実DWTデータからpairsを抽出し、新しいEntropyEncoderで再エンコード→デコード
    func testReEncodePairs() async throws {
        let (realPairs, _) = await generateRealPairs()
        
        print("=== Real pairs count: \(realPairs.count) ===")
        
        // 新しいエンコーダに同じpairsを追加
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        for pair in realPairs {
            encoder.addPair(run: pair.run, val: pair.val, isParentZero: pair.isParentZero)
        }
        
        let data = encoder.getData()
        var decPairs: [(run: Int, val: Int16)] = []
        try data.withUnsafeBufferPointer { ptr in
            var decoder = try EntropyDecoder(base: ptr.baseAddress!, count: ptr.count)
            for i in 0..<encoder.pairs.count {
                let pair = decoder.readPair(isParentZero: encoder.pairs[i].isParentZero)
                decPairs.append(pair)
            }
            
            XCTAssertEqual(encoder.pairs.count, decPairs.count, "pairs count")
            for i in 0..<encoder.pairs.count {
                XCTAssertEqual(encoder.pairs[i].run, UInt32(decPairs[i].run), "run at \(i)")
                XCTAssertEqual(encoder.pairs[i].val, decPairs[i].val, "val at \(i)")
            }
        }
        
        var firstDiff = -1
        var diffCount = 0
        for i in 0..<min(realPairs.count, decPairs.count) {
            if realPairs[i].run != decPairs[i].run || realPairs[i].val != decPairs[i].val {
                if firstDiff < 0 {
                    firstDiff = i
                    print("DIFF[\(i)]: enc=(\(realPairs[i].run), \(realPairs[i].val)) dec=(\(decPairs[i].run), \(decPairs[i].val))")
                }
                diffCount += 1
            }
        }
        
        if 0 < diffCount {
            // チャンク分析
            let pairCount = realPairs.count
            let chunkBase = pairCount / 4
            let chunkRemainder = pairCount % 4
            var chunkStarts = [Int](repeating: 0, count: 5)
            for i in 0..<4 {
                chunkStarts[i + 1] = chunkStarts[i] + chunkBase + (i < chunkRemainder ? 1 : 0)
            }
            print("=== Chunk boundaries: \(chunkStarts) ===")
            print("=== First diff at \(firstDiff), chunk=\(firstDiff < chunkStarts[1] ? 0 : firstDiff < chunkStarts[2] ? 1 : firstDiff < chunkStarts[3] ? 2 : 3) ===")
            
            // token分析：最初の差異前後のpairのtoken
            for d in max(0, firstDiff - 3)..<min(realPairs.count, firstDiff + 5) {
                let encT = valueTokenize(realPairs[d].val)
                let encR = valueTokenizeUnsigned(realPairs[d].run)
                let decT = d < decPairs.count ? valueTokenize(decPairs[d].val) : (token: UInt8(255), bypassBits: UInt32(0), bypassLen: 0)
                let decR = d < decPairs.count ? valueTokenizeUnsigned(UInt32(decPairs[d].run)) : (token: UInt8(255), bypassBits: UInt32(0), bypassLen: 0)
                print("  [\(d)] enc.run=\(realPairs[d].run)(t\(encR.token)/bp\(encR.bypassLen)) val=\(realPairs[d].val)(t\(encT.token)/bp\(encT.bypassLen)) | dec.run=\(d < decPairs.count ? Int(decPairs[d].run) : -1)(t\(decR.token)/bp\(decR.bypassLen)) val=\(d < decPairs.count ? Int(decPairs[d].val) : -1)(t\(decT.token)/bp\(decT.bypassLen))")
            }
        }
        
        XCTAssertEqual(diffCount, 0, "Pairs diff: \(diffCount) total, first at \(firstDiff)")
    }
}
