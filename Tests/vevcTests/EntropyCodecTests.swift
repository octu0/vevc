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
        
        var decoder = try EntropyDecoder(data: data)
        
        for (i, expected) in expectedPairs.enumerated() {
            let pair = decoder.readPair(isParentZero: false)
            XCTAssertEqual(pair.run, Int(expected.run), "Pair[\(i)] run: expected=\(expected.run) got=\(pair.run)")
            XCTAssertEqual(pair.val, expected.val, "Pair[\(i)] val: expected=\(expected.val) got=\(pair.val)")
        }
    }

    
    /// 実際のblockEncode16データでEntropyEncoderのrANSラウンドトリップ
    func testBlockEncode16MultipleBlocks() throws {
        // 16ブロック分のデータを作成
        var blocks = (0..<16).map { _ in Block2D(width: 16, height: 16) }
        for i in 0..<16 {
            blocks[i].data = (0..<256).map { idx in
                Int16(clamping: (i * 256 + idx) &* 7 % 41 - 20)
            }
        }
        
        // 全ブロックをエンコード
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        for i in 0..<16 {
            blocks[i].withView { view in
                blockEncode16(encoder: &encoder, block: view, parentBlock: nil)
            }
        }
        
        let data = encoder.getData()
        
        // デコード
        var decBlocks = (0..<16).map { _ in Block2D(width: 16, height: 16) }
        var decoder = try EntropyDecoder(data: data)
        for i in 0..<16 {
            decBlocks[i].withView { view in
                try! blockDecode16(decoder: &decoder, block: &view, parentBlock: nil)
            }
        }
        
        // 比較 (blockEncode16後のデータ vs デコード後)
        for i in 0..<16 {
            for idx in 0..<256 {
                XCTAssertEqual(blocks[i].data[idx], decBlocks[i].data[idx], "Block[\(i)] idx=\(idx) enc=\(blocks[i].data[idx]) dec=\(decBlocks[i].data[idx])")
            }
        }
    }
}
