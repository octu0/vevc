import XCTest
@testable import vevc

/// EntropyEncoder/Decoderの4-way rANSモードの直接テスト
final class EntropyCodecTests: XCTestCase {
    
    /// 50個以上のpairsでrANSモードのラウンドトリップ
    func testRansRoundtrip() throws {
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        
        // 50個のpairを追加（nonZeroCount > 32でrANSモード）
        var expectedPairs: [(run: UInt32, val: Int16)] = []
        for i in 0..<50 {
            let run = UInt32(i % 5)
            let val = Int16(clamping: (i * 7 - 175) % 100)
            if val == 0 { continue }
            encoder.addPair(run: run, val: val, isParentZero: false)
            expectedPairs.append((run: run, val: val))
        }
        
        let data = encoder.getData()
        
        try data.withUnsafeBufferPointer { ptr in
            var decoder = try EntropyDecoder(base: ptr.baseAddress!, count: ptr.count)
            
            for (i, expected) in expectedPairs.enumerated() {
                let pair = decoder.readPair(isParentZero: false)
                XCTAssertEqual(pair.run, Int(expected.run), "Pair[\(i)] run: expected=\(expected.run) got=\(pair.run)")
                XCTAssertEqual(pair.val, expected.val, "Pair[\(i)] val: expected=\(expected.val) got=\(pair.val)")
            }
        }
    }

    
    /// 実際のblockEncode16データでEntropyEncoderのrANSラウンドトリップ
    func testBlockEncode16MultipleBlocks() throws {
        // 16ブロック分のデータを作成
        let blocks = (0..<16).map { _ in BlockView.allocate(width: 16, height: 16) }
        defer { for b in blocks { b.deallocate() } }
        for i in 0..<16 {
            let values = (0..<256).map { idx in
                Int16(clamping: (i * 256 + idx) &* 7 % 41 - 20)
            }
            values.withUnsafeBufferPointer { ptr in
                blocks[i].base.update(from: ptr.baseAddress!, count: 256)
            }
        }
        
        // 全ブロックをエンコード
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        for i in 0..<16 {
            blockEncode16V(encoder: &encoder, block: blocks[i], parentBlock: nil)
        }
        
        let data = encoder.getData()
        
        // デコード
        let decBlocks = (0..<16).map { _ in BlockView.allocate(width: 16, height: 16) }
        defer { for b in decBlocks { b.deallocate() } }
        try data.withUnsafeBufferPointer { ptr in
            var decoder = try EntropyDecoder(base: ptr.baseAddress!, count: ptr.count)
            for i in 0..<16 {
                try! blockDecode16V(decoder: &decoder, block: decBlocks[i])
            }
        }
        
        // 比較 (blockEncode16後のデータ vs デコード後)
        for i in 0..<16 {
            for idx in 0..<256 {
                XCTAssertEqual(blocks[i].base[idx], decBlocks[i].base[idx], "Block[\(i)] idx=\(idx) enc=\(blocks[i].base[idx]) dec=\(decBlocks[i].base[idx])")
            }
        }
    }
}
