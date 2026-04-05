import XCTest
@testable import vevc

/// blockEncode→blockDecodeのラウンドトリップを直接テスト
final class BlockRoundtripTests: XCTestCase {
    
    /// blockEncode16→blockDecode16 ラウンドトリップ（量子化済みデータ）
    func testBlockEncode16Roundtrip() throws {
        // 量子化後のSignedMappingデータを模擬（-50〜50の範囲のランダムな値）
        let block = BlockView.allocate(width: 16, height: 16)
        defer { block.deallocate() }
        let temp256 = (0..<256).map { i -> Int16 in
            Int16(clamping: (i &* 7 + 13) % 101 - 50)
        }
        temp256.withUnsafeBufferPointer { src in
            block.base.update(from: src.baseAddress!, count: 256)
        }
        
        // エンコード前のデータを保存
        let originalData = Array(UnsafeBufferPointer(start: block.base, count: 256))
        
        // blockEncode16
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        blockEncode16(encoder: &encoder, block: block, parentBlock: nil)
        encoder.flush()
        let encoded = encoder.getData()
        
        // blockEncode16のゼロクリア修正後のデータ
        let afterEncodeData = Array(UnsafeBufferPointer(start: block.base, count: 256))
        
        // blockDecode16
        let decBlock = BlockView.allocate(width: 16, height: 16)
        defer { decBlock.deallocate() }
        try encoded.withUnsafeBufferPointer { ptr in
            var decoder = try EntropyDecoder(base: ptr.baseAddress!, count: ptr.count)
            try! blockDecode16(decoder: &decoder, block: decBlock, parentBlock: nil)
        }
        // エンコード後のデータとデコード後のデータを比較
        for i in 0..<256 {
            XCTAssertEqual(afterEncodeData[i], decBlock.base[i], "idx=\(i) y:\(i/16) x:\(i%16): encAfter=\(afterEncodeData[i]) dec=\(decBlock.base[i]) original=\(originalData[i])")
        }
    }
    
    /// blockEncode8→blockDecode8 ラウンドトリップ
    func testBlockEncode8Roundtrip() throws {
        let block = BlockView.allocate(width: 8, height: 8)
        defer { block.deallocate() }
        let temp64 = (0..<64).map { i -> Int16 in
            Int16(clamping: (i &* 11 + 3) % 61 - 30)
        }
        temp64.withUnsafeBufferPointer { src in
            block.base.update(from: src.baseAddress!, count: 64)
        }
        
        let originalData = Array(UnsafeBufferPointer(start: block.base, count: 64))
        
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        blockEncode8(encoder: &encoder, block: block, parentBlock: nil)
        encoder.flush()
        let encoded = encoder.getData()
        
        let afterEncodeData = Array(UnsafeBufferPointer(start: block.base, count: 64))
        
        let decBlock = BlockView.allocate(width: 8, height: 8)
        defer { decBlock.deallocate() }
        try encoded.withUnsafeBufferPointer { ptr in
            var decoder = try EntropyDecoder(base: ptr.baseAddress!, count: ptr.count)
            try! blockDecode8(decoder: &decoder, block: decBlock, parentBlock: nil)
        }
        for i in 0..<64 {
            XCTAssertEqual(afterEncodeData[i], decBlock.base[i], "idx=\(i) y:\(i/8) x:\(i%8): encAfter=\(afterEncodeData[i]) dec=\(decBlock.base[i]) original=\(originalData[i])")
        }
    }
    
    /// blockEncode16 ゼロクリアデバッグ: lscp超がクリアされるか確認
    func testBlockEncode16ZeroClear() throws {
        let block = BlockView.allocate(width: 16, height: 16)
        defer { block.deallocate() }
        // lscp = (5, 5) になるように設計
        // (5,5)以降に非ゼロデータを入れる
        for i in 0..<256 {
            block.base[i] = Int16(i + 1)  // 全て非ゼロ
        }
        // (5,5)より後を全てゼロにしてlscpを(5,5)にする
        for y in 0..<16 {
            for x in 0..<16 {
                let idx = y * 16 + x
                if 5 < y || (y == 5 && 5 < x) {
                    block.base[idx] = 0
                }
            }
        }
        
        let beforeData = Array(UnsafeBufferPointer(start: block.base, count: 256))
        
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        blockEncode16(encoder: &encoder, block: block, parentBlock: nil)
            
        // afterEncodeでは(5,5)以降のデータがゼロにクリアされているはず
        let afterData = Array(UnsafeBufferPointer(start: block.base, count: 256))
        
        // (5,5)以降のデータがゼロかチェック
        for y in 0..<16 {
            for x in 0..<16 {
                let idx = y * 16 + x
                if 5 < y || (y == 5 && 5 < x) {
                    XCTAssertEqual(afterData[idx], 0, "lscp超のデータが残留: y=\(y) x=\(x) before=\(beforeData[idx]) after=\(afterData[idx])")
                }
            }
        }
    }
    
    /// stride=32のBlockView（実際の32x32ブロック内サブバンド）でblockEncode16→blockDecode16ラウンドトリップ
    func testBlockEncode16RoundtripStride32() throws {
        let block32 = BlockView.allocate(width: 32, height: 32)
        defer { block32.deallocate() }
        // HLサブバンド (offset=16, stride=32) にデータをセット
        var hlView = BlockView(base: block32.base.advanced(by: 16), width: 16, height: 16, stride: 32)
        for y in 0..<16 {
            let ptr = hlView.rowPointer(y: y)
            for x in 0..<16 {
                ptr[x] = Int16(clamping: (y * 16 + x) &* 7 % 41 - 20)
            }
        }
            
        // エンコード (stride=32)
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        hlView = BlockView(base: block32.base.advanced(by: 16), width: 16, height: 16, stride: 32)
        blockEncode16(encoder: &encoder, block: hlView, parentBlock: nil)
        encoder.flush()
        let encoded = encoder.getData()
        
        // エンコード後のHLサブバンドデータ
        var encAfterHL = [Int16](repeating: 0, count: 256)
        hlView = BlockView(base: block32.base.advanced(by: 16), width: 16, height: 16, stride: 32)
        for y in 0..<16 {
            let ptr = hlView.rowPointer(y: y)
            for x in 0..<16 {
                encAfterHL[y * 16 + x] = ptr[x]
            }
        }
            
        // デコード (stride=32)  
        let decBlock32 = BlockView.allocate(width: 32, height: 32)
        defer { decBlock32.deallocate() }
        try encoded.withUnsafeBufferPointer { ptr in
            var decoder = try EntropyDecoder(base: ptr.baseAddress!, count: ptr.count)
            hlView = BlockView(base: decBlock32.base.advanced(by: 16), width: 16, height: 16, stride: 32)
            try! blockDecode16(decoder: &decoder, block: hlView, parentBlock: nil)
        }
            
        // デコード後のHLデータ
        var decHL = [Int16](repeating: 0, count: 256)
        hlView = BlockView(base: decBlock32.base.advanced(by: 16), width: 16, height: 16, stride: 32)
        for y in 0..<16 {
            let ptr = hlView.rowPointer(y: y)
            for x in 0..<16 {
                decHL[y * 16 + x] = ptr[x]
            }
        }
            
        // 比較
        var diffCount = 0
        for i in 0..<256 {
            if encAfterHL[i] != decHL[i] {
                diffCount += 1
                if diffCount <= 5 {
                    XCTFail("stride32 HL不一致: y=\(i/16) x=\(i%16) encAfter=\(encAfterHL[i]) dec=\(decHL[i])")
                }
            }
        }
        XCTAssertEqual(diffCount, 0, "stride=32 blockEncode16→blockDecode16 不一致ピクセル: \(diffCount)/256")
    }
}
