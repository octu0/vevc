import XCTest
@testable import vevc

/// chunk[3]のpairsだけを独立にエンコード/デコードして問題を分離
final class Chunk3IsolationTests: XCTestCase {
    
    private func generateRealPairs() -> [(run: UInt32, val: Int16, isParentZero: Bool)] {
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
        
        var (blocks, _) = extractSingleTransformBlocks32(r: pd.rY, width: width, height: height)
        for i in blocks.indices {
            evaluateQuantizeLayer32(block: &blocks[i], qt: qtY)
        }
        
        let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
        
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        for i in blocks.indices {
            let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
                return isEffectivelyZero32(data: ptr, threshold: safeThreshold)
            }
            if isZero { continue }
            blocks[i].withView { view in
                let subs = getSubbands32(view: view)
                blockEncode16(encoder: &encoder, block: subs.hl, parentBlock: nil)
                blockEncode16(encoder: &encoder, block: subs.lh, parentBlock: nil)
                blockEncode16(encoder: &encoder, block: subs.hh, parentBlock: nil)
            }
        }
        return encoder.pairs
    }
    
    /// chunk[3]のpairsだけをrANSラウンドトリップ
    func testChunk3Only() throws {
        let allPairs = generateRealPairs()
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
        var decoder = try EntropyDecoder(data: data)
        var decPairs: [(run: Int, val: Int16)] = []
        for i in 0..<encoder.pairs.count {
            let pair = decoder.readPair(isParentZero: encoder.pairs[i].isParentZero)
            decPairs.append(pair)
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
    func testEachChunkIndependently() throws {
        let allPairs = generateRealPairs()
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
            var decoder = try EntropyDecoder(data: data)
            var decPairs: [(run: Int, val: Int16)] = []
        for i in 0..<encoder.pairs.count {
            let pair = decoder.readPair(isParentZero: encoder.pairs[i].isParentZero)
            decPairs.append(pair)
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
