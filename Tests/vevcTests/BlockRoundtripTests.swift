import XCTest
@testable import vevc

/// blockEncode→blockDecodeのラウンドトリップを直接テスト
final class BlockRoundtripTests: XCTestCase {
    
    /// blockEncode16→blockDecode16 ラウンドトリップ（量子化済みデータ）
    func testBlockEncode16Roundtrip() throws {
        // 量子化後のSignedMappingデータを模擬（-50〜50の範囲のランダムな値）
        var block = Block2D(width: 16, height: 16)
        block.setData((0..<256).map { i in
            let v = Int16(clamping: (i &* 7 + 13) % 101 - 50)
            return v
        })
        
        // エンコード前のデータを保存
        let originalData = Array(block.data)
        
        // blockEncode16
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        block.withView { view in
            blockEncode16(encoder: &encoder, block: view, parentBlock: nil)
        }
        encoder.flush()
        let encoded = encoder.getData()
        
        // blockEncode16のゼロクリア修正後のデータ
        let afterEncodeData = Array(block.data)
        
        // blockDecode16
        var decBlock = Block2D(width: 16, height: 16)
        var decoder = try EntropyDecoder(data: encoded)
        decBlock.withView { view in
            try! blockDecode16(decoder: &decoder, block: &view, parentBlock: nil)
        }
        
        // エンコード後のデータとデコード後のデータを比較
        for i in 0..<256 {
            XCTAssertEqual(afterEncodeData[i], decBlock.base[i], "idx=\(i) y:\(i/16) x:\(i%16): encAfter=\(afterEncodeData[i]) dec=\(decBlock.base[i]) original=\(originalData[i])")
        }
    }
    
    /// blockEncode8→blockDecode8 ラウンドトリップ
    func testBlockEncode8Roundtrip() throws {
        var block = Block2D(width: 8, height: 8)
        block.setData((0..<64).map { i in
            Int16(clamping: (i &* 11 + 3) % 61 - 30)
        })
        
        let originalData = Array(block.data)
        
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        block.withView { view in
            blockEncode8(encoder: &encoder, block: view, parentBlock: nil)
        }
        encoder.flush()
        let encoded = encoder.getData()
        
        let afterEncodeData = Array(block.data)
        
        var decBlock = Block2D(width: 8, height: 8)
        var decoder = try EntropyDecoder(data: encoded)
        decBlock.withView { view in
            try! blockDecode8(decoder: &decoder, block: &view, parentBlock: nil)
        }
        
        for i in 0..<64 {
            XCTAssertEqual(afterEncodeData[i], decBlock.base[i], "idx=\(i) y:\(i/8) x:\(i%8): encAfter=\(afterEncodeData[i]) dec=\(decBlock.base[i]) original=\(originalData[i])")
        }
    }
    
    /// blockEncode16 ゼロクリアデバッグ: lscp超がクリアされるか確認
    func testBlockEncode16ZeroClear() throws {
        var block = Block2D(width: 16, height: 16)
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
        
        let beforeData = Array(block.data)
        
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        block.withView { view in
            blockEncode16(encoder: &encoder, block: view, parentBlock: nil)
        }
        
        // afterEncodeでは(5,5)以降のデータがゼロにクリアされているはず
        // ただしblockEncode16はstride=16のBlockViewで動作
        // Block2D.withViewではstride=widthなので stride=16
        let afterData = Array(block.data)
        
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
        var block32 = Block2D(width: 32, height: 32)
        // HLサブバンド (offset=16, stride=32) にデータをセット
        block32.withView { view in
            let hlView = BlockView(base: view.base.advanced(by: 16), width: 16, height: 16, stride: 32)
            for y in 0..<16 {
                let ptr = hlView.rowPointer(y: y)
                for x in 0..<16 {
                    ptr[x] = Int16(clamping: (y * 16 + x) &* 7 % 41 - 20)
                }
            }
        }
        
        // エンコード (stride=32)
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        block32.withView { view in
            let hlView = BlockView(base: view.base.advanced(by: 16), width: 16, height: 16, stride: 32)
            blockEncode16(encoder: &encoder, block: hlView, parentBlock: nil)
        }
        encoder.flush()
        let encoded = encoder.getData()
        
        // エンコード後のHLサブバンドデータ
        var encAfterHL = [Int16](repeating: 0, count: 256)
        block32.withView { view in
            let hlView = BlockView(base: view.base.advanced(by: 16), width: 16, height: 16, stride: 32)
            for y in 0..<16 {
                let ptr = hlView.rowPointer(y: y)
                for x in 0..<16 {
                    encAfterHL[y * 16 + x] = ptr[x]
                }
            }
        }
        
        // デコード (stride=32)  
        var decBlock32 = Block2D(width: 32, height: 32)
        var decoder = try EntropyDecoder(data: encoded)
        decBlock32.withView { view in
            var hlView = BlockView(base: view.base.advanced(by: 16), width: 16, height: 16, stride: 32)
            try! blockDecode16(decoder: &decoder, block: &hlView, parentBlock: nil)
        }
        
        // デコード後のHLデータ
        var decHL = [Int16](repeating: 0, count: 256)
        decBlock32.withView { view in
            let hlView = BlockView(base: view.base.advanced(by: 16), width: 16, height: 16, stride: 32)
            for y in 0..<16 {
                let ptr = hlView.rowPointer(y: y)
                for x in 0..<16 {
                    decHL[y * 16 + x] = ptr[x]
                }
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
