import XCTest
@testable import vevc

/// EntropyEncoder/Decoderの4-way rANSモードの直接テスト
final class EntropyCodecTests: XCTestCase {
    
    /// 50個以上のpairsでrANSモードのラウンドトリップ
    func testRansRoundtrip() throws {
        var encoder = EntropyEncoder()
        
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
            let pair = decoder.readPair()
            XCTAssertEqual(pair.run, Int(expected.run), "Pair[\(i)] run: expected=\(expected.run) got=\(pair.run)")
            XCTAssertEqual(pair.val, expected.val, "Pair[\(i)] val: expected=\(expected.val) got=\(pair.val)")
        }
    }

    /// bypassWriter + pairs の混合テスト (blockEncode16と同じパターン)
    func testBypassAndPairsMixed() throws {
        var encoder = EntropyEncoder()
        
        // blockEncode16が16ブロック×3サブバンド=48回呼ばれるシミュレーション
        var expectedBits: [UInt8] = []
        var expectedPairs: [(run: UInt32, val: Int16)] = []
        
        for block in 0..<48 {
            // hasNonZero = 1 (全てのブロックが非ゼロ)
            encoder.encodeBypass(binVal: 1)
            expectedBits.append(1)
            
            // lscpX と lscpY (Exp-Golomb)
            let lscpX = (block % 16)
            let lscpY = (block / 3) % 16
            encodeExpGolomb(val: UInt32(lscpX), encoder: &encoder)
            encodeExpGolomb(val: UInt32(lscpY), encoder: &encoder)
            
            // 係数 (3個のpair per block)
            for j in 0..<3 {
                let run = UInt32(j)
                let val = Int16(block * 3 + j + 1)
                encoder.addPair(run: run, val: val, isParentZero: false)
                expectedPairs.append((run: run, val: val))
            }
        }
        
        let data = encoder.getData()
        
        var decoder = try EntropyDecoder(data: data)
        
        // bypass bits の確認
        for block in 0..<48 {
            let hasNonZero = try decoder.decodeBypass()
            XCTAssertEqual(hasNonZero, expectedBits[block], "Block[\(block)] hasNonZero")
            
            let lscpX = try decodeExpGolomb(decoder: &decoder)
            let expectedLscpX = UInt32(block % 16)
            XCTAssertEqual(lscpX, expectedLscpX, "Block[\(block)] lscpX: expected=\(expectedLscpX) got=\(lscpX)")
            
            let lscpY = try decodeExpGolomb(decoder: &decoder)
            let expectedLscpY = UInt32((block / 3) % 16)
            XCTAssertEqual(lscpY, expectedLscpY, "Block[\(block)] lscpY: expected=\(expectedLscpY) got=\(lscpY)")
        }
        
        // pairs の確認
        for (i, expected) in expectedPairs.enumerated() {
            let pair = decoder.readPair()
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
        var encoder = EntropyEncoder()
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
