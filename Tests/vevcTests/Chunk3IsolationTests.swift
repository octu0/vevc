import XCTest
@testable import vevc

/// chunk[3]のpairsだけを独立にエンコード/デコードして問題を分離
final class Chunk3IsolationTests: XCTestCase {
    
    private func generateRealPairs() async -> [(run: UInt32, val: Int16, isParentZero: Bool)] {
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
        let pool = BlockViewPool()
        
        let (blocks, _, rel) = await extractSingleTransformBlocks32(r: pd.rY, width: width, height: height, pool: pool, qt: qtY)
        defer { rel() }
        for i in blocks.indices {
            evaluateQuantizeLayer32(view: blocks[i], qt: qtY)
        }
        
        let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
        
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        for i in blocks.indices {
            let isZero = isEffectivelyZero32(data: blocks[i].base, threshold: safeThreshold)
            if isZero { continue }
            let view = blocks[i]
            let subs = getSubbands32(view: view)
            blockEncode16V(encoder: &encoder, block: subs.hl)
            blockEncode16H(encoder: &encoder, block: subs.lh)
            blockEncode16H(encoder: &encoder, block: subs.hh)
                }
        return encoder.pairs
    }
    
    /// chunk[3]のpairsだけをrANSラウンドトリップ
    func testChunk3Only() async throws {
        let allPairs = await generateRealPairs()
        let pairCount = allPairs.count
        let chunkBase = pairCount / 4
        let chunkRemainder = pairCount % 4
        var chunkStarts = [Int](repeating: 0, count: 5)
        for i in 0..<4 {
            chunkStarts[i + 1] = chunkStarts[i] + chunkBase + (i < chunkRemainder ? 1 : 0)
        }
        
        // chunk[3] のpairs
        let chunk3Pairs = Array(allPairs[chunkStarts[3]..<chunkStarts[4]])
        print("=== Chunk[3]: \(chunk3Pairs.count) pairs (indices \(chunkStarts[3])..\(chunkStarts[4]-1)) ===")
        
        // chunk[3]のpairsだけを新しいEntropyEncoderに渡す
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        for pair in chunk3Pairs {
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
        }
        
        XCTAssertEqual(chunk3Pairs.count, decPairs.count, "pairs count: \(chunk3Pairs.count) vs \(decPairs.count)")
        
        var diffCount = 0
        var firstDiff = -1
        for i in 0..<min(chunk3Pairs.count, decPairs.count) {
            if chunk3Pairs[i].run != decPairs[i].run || chunk3Pairs[i].val != decPairs[i].val {
                if firstDiff < 0 {
                    firstDiff = i
                    print("DIFF[\(i)]: enc=(\(chunk3Pairs[i].run), \(chunk3Pairs[i].val)) dec=(\(decPairs[i].run), \(decPairs[i].val))")
                }
                diffCount += 1
            }
        }
        
        XCTAssertEqual(diffCount, 0, "Chunk[3] pairs diff: \(diffCount), first at \(firstDiff)")
    }
    
    /// 各chunkを独立にテスト
    func testEachChunkIndependently() async throws {
        let allPairs = await generateRealPairs()
        let pairCount = allPairs.count
        let chunkBase = pairCount / 4
        let chunkRemainder = pairCount % 4
        var chunkStarts = [Int](repeating: 0, count: 5)
        for i in 0..<4 {
            chunkStarts[i + 1] = chunkStarts[i] + chunkBase + (i < chunkRemainder ? 1 : 0)
        }
        
        for chunk in 0..<4 {
            let chunkPairs = Array(allPairs[chunkStarts[chunk]..<chunkStarts[chunk + 1]])
            
            var encoder = EntropyEncoder<DynamicEntropyModel>()
            for pair in chunkPairs {
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
            }
            
            var diffCount = 0
            for i in 0..<min(chunkPairs.count, decPairs.count) {
                if chunkPairs[i].run != decPairs[i].run || chunkPairs[i].val != decPairs[i].val {
                    diffCount += 1
                }
            }
            
            if 0 < diffCount {
                XCTFail("Chunk[\(chunk)] (\(chunkPairs.count) pairs): \(diffCount) diffs")
            }
        }
    }
}
