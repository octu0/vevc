import XCTest
@testable import vevc

/// encodePlaneSubbands32→decodePlaneSubbands32の直接ラウンドトリップテスト
final class SubbandsRoundtripTests: XCTestCase {
    
    /// 少数ブロックでencodePlaneSubbands32→decodePlaneSubbands32
    func testSubbands32Roundtrip_3blocks() throws {
        // 3ブロック（1行3列）のテスト
        var blocks = (0..<3).map { _ in Block2D(width: 32, height: 32) }
        
        // 各ブロックのHL/LH/HHにデータをセット（LLは0のまま）
        for bi in 0..<3 {
            blocks[bi].withView { view in
                let half = 16
                // HL (right-top quadrant)
                for y in 0..<half {
                    let ptr = view.base.advanced(by: y * 32 + half)
                    for x in 0..<half {
                        ptr[x] = Int16(clamping: (bi * 100 + y * 16 + x) &* 7 % 41 - 20)
                    }
                }
                // LH (left-bottom quadrant) 
                for y in 0..<half {
                    let ptr = view.base.advanced(by: (y + half) * 32)
                    for x in 0..<half {
                        ptr[x] = Int16(clamping: (bi * 200 + y * 16 + x) &* 11 % 37 - 18)
                    }
                }
                // HH (right-bottom quadrant)
                for y in 0..<half {
                    let ptr = view.base.advanced(by: (y + half) * 32 + half)
                    for x in 0..<half {
                        ptr[x] = Int16(clamping: (bi * 300 + y * 16 + x) &* 13 % 31 - 15)
                    }
                }
            }
        }
        
        // エンコード前のHL/LH/HHを保存
        var origHL: [[Int16]] = []
        for bi in 0..<3 {
            var hl = [Int16](repeating: 0, count: 256)
            blocks[bi].withView { view in
                for y in 0..<16 {
                    let ptr = view.base.advanced(by: y * 32 + 16)
                    for x in 0..<16 {
                        hl[y * 16 + x] = ptr[x]
                    }
                }
            }
            origHL.append(hl)
        }
        
        // encodePlaneSubbands32
        let data = encodePlaneSubbands32(blocks: &blocks, zeroThreshold: 3)
        
        // エンコード後のHL
        var encAfterHL: [[Int16]] = []
        for bi in 0..<3 {
            var hl = [Int16](repeating: 0, count: 256)
            blocks[bi].withView { view in
                for y in 0..<16 {
                    let ptr = view.base.advanced(by: y * 32 + 16)
                    for x in 0..<16 {
                        hl[y * 16 + x] = ptr[x]
                    }
                }
            }
            encAfterHL.append(hl)
        }
        
        // decodePlaneSubbands32
        let decBlocks = try decodePlaneSubbands32(data: data, blockCount: 3)
        
        // デコード後のHL
        var decHL: [[Int16]] = []
        for bi in 0..<3 {
            var hl = [Int16](repeating: 0, count: 256)
            var b = decBlocks[bi]
            b.withView { view in
                for y in 0..<16 {
                    let ptr = view.base.advanced(by: y * 32 + 16)
                    for x in 0..<16 {
                        hl[y * 16 + x] = ptr[x]
                    }
                }
            }
            decHL.append(hl)
        }
        
        // 比較: encAfterHL vs decHL
        for bi in 0..<3 {
            var diffCount = 0
            var maxD = 0
            for i in 0..<256 {
                let d = abs(Int(encAfterHL[bi][i]) - Int(decHL[bi][i]))
                if 0 < d { diffCount += 1 }
                if maxD < d { maxD = d }
            }
            
            // 差異があった場合、最初の5ピクセルの詳細を出力
            if 0 < diffCount {
                var details = ""
                var shown = 0
                for i in 0..<256 {
                    let d = abs(Int(encAfterHL[bi][i]) - Int(decHL[bi][i]))
                    if 0 < d && shown < 5 {
                        details += " [y:\(i/16) x:\(i%16)]: orig=\(origHL[bi][i]) encAfter=\(encAfterHL[bi][i]) dec=\(decHL[bi][i])"
                        shown += 1
                    }
                }
                XCTFail("Block[\(bi)] HL不一致: \(diffCount)/256 pixels, maxDiff=\(maxD)\(details)")
            }
        }
    }
    
    /// splitが発生するデータでencodePlaneSubbands32→decodePlaneSubbands32
    func testSubbands32Roundtrip_withSplit() throws {
        var blocks = (0..<3).map { _ in Block2D(width: 32, height: 32) }
        
        // Block0: 全象限にデータ → splitしない
        blocks[0].withView { view in
            for y in 0..<16 {
                for x in 0..<16 {
                    view.base.advanced(by: y * 32 + 16)[x] = Int16(y * 16 + x + 1)  // HL
                    view.base.advanced(by: (y + 16) * 32)[x] = Int16(y * 16 + x + 1) // LH
                    view.base.advanced(by: (y + 16) * 32 + 16)[x] = Int16(y * 16 + x + 1) // HH
                }
            }
        }
        
        // Block1: TL象限のみデータ、他はゼロ → shouldSplit=true
        blocks[1].withView { view in
            // HL TL (8x8)
            for y in 0..<8 {
                for x in 0..<8 {
                    view.base.advanced(by: y * 32 + 16)[x] = Int16(y * 8 + x + 1)
                }
            }
            // LH TL (8x8)
            for y in 0..<8 {
                for x in 0..<8 {
                    view.base.advanced(by: (y + 16) * 32)[x] = Int16(y * 8 + x + 1)
                }
            }
            // HH TL (8x8)
            for y in 0..<8 {
                for x in 0..<8 {
                    view.base.advanced(by: (y + 16) * 32 + 16)[x] = Int16(y * 8 + x + 1)
                }
            }
        }
        
        // Block2: 全象限にデータ
        blocks[2].withView { view in
            for y in 0..<16 {
                for x in 0..<16 {
                    view.base.advanced(by: y * 32 + 16)[x] = Int16((y * 16 + x) % 30 - 15)
                    view.base.advanced(by: (y + 16) * 32)[x] = Int16((y * 16 + x) % 25 - 12)
                    view.base.advanced(by: (y + 16) * 32 + 16)[x] = Int16((y * 16 + x) % 20 - 10)
                }
            }
        }
        
        // encodePlaneSubbands32
        let data = encodePlaneSubbands32(blocks: &blocks, zeroThreshold: 3)
        
        // decodePlaneSubbands32
        let decBlocks = try decodePlaneSubbands32(data: data, blockCount: 3)
        
        // 各ブロックのHL/LH/HH比較
        for bi in 0..<3 {
            var encBlk = blocks[bi]
            var decBlk = decBlocks[bi]
            
            var diffHL = 0, diffLH = 0, diffHH = 0
            encBlk.withView { encView in
                decBlk.withView { decView in
                    for y in 0..<16 {
                        for x in 0..<16 {
                            if encView.base.advanced(by: y * 32 + 16)[x] != decView.base.advanced(by: y * 32 + 16)[x] { diffHL += 1 }
                            if encView.base.advanced(by: (y + 16) * 32)[x] != decView.base.advanced(by: (y + 16) * 32)[x] { diffLH += 1 }
                            if encView.base.advanced(by: (y + 16) * 32 + 16)[x] != decView.base.advanced(by: (y + 16) * 32 + 16)[x] { diffHH += 1 }
                        }
                    }
                }
            }
            XCTAssertEqual(diffHL + diffLH + diffHH, 0, "Block[\(bi)] split不一致: HL=\(diffHL) LH=\(diffLH) HH=\(diffHH)")
        }
    }
    
    /// 実際のencodePlaneLayer32が生成するblocks（量子化後）でencodePlaneSubbands32→decodePlaneSubbands32テスト
    func testSubbands32Roundtrip_realDWTBlocks() async throws {
        let width = 640
        let height = 480
        
        var img = YCbCrImage(width: width, height: height)
        for y in 0..<height {
            for x in 0..<width {
                let base = (x + y * 2) % 256
                let noise = (x &* 2654435761 ^ y &* 2246822519) % 20
                img.yPlane[y * width + x] = UInt8(clamping: base + noise - 10)
            }
        }
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        for cy in 0..<cHeight {
            for cx in 0..<cWidth {
                img.cbPlane[cy * cWidth + cx] = UInt8(clamping: 128 + (cx + cy) % 20 - 10)
                img.crPlane[cy * cWidth + cx] = UInt8(clamping: 128 + (cx - cy + 256) % 20 - 10)
            }
        }
        
        let pd = toPlaneData420(images: [img])[0]
        let qtY = QuantizationTable(baseStep: 2)
        
        // DWT + 量子化のみ実行して、encodePlaneSubbands32へ渡すブロック配列を取得
        var (blocks, _) = extractSingleTransformBlocks32(r: pd.rY, width: width, height: height)
        for i in blocks.indices {
            evaluateQuantizeLayer32(block: &blocks[i], qt: qtY)
        }
        
        // encodePlaneSubbands32
        let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
        let data = encodePlaneSubbands32(blocks: &blocks, zeroThreshold: safeThreshold)
        
        // decodePlaneSubbands32
        let decBlocks = try decodePlaneSubbands32(data: data, blockCount: blocks.count)
        
        // 比較
        XCTAssertEqual(blocks.count, decBlocks.count, "blocks count mismatch")
        
        var totalDiff = 0
        var maxDiff = 0
        var firstDiffBlock = -1
        
        for bi in 0..<blocks.count {
            var encBlk = blocks[bi]
            var decBlk = decBlocks[bi]
            
            encBlk.withView { encView in
                decBlk.withView { decView in
                    for y in 0..<16 {
                        for x in 0..<16 {
                            // HL
                            let ev = encView.base.advanced(by: y * 32 + 16)[x]
                            let dv = decView.base.advanced(by: y * 32 + 16)[x]
                            let d = abs(Int(ev) - Int(dv))
                            if 0 < d {
                                totalDiff += 1
                                if maxDiff < d { maxDiff = d }
                                if firstDiffBlock < 0 { firstDiffBlock = bi }
                            }
                            // LH
                            let ev2 = encView.base.advanced(by: (y + 16) * 32)[x]
                            let dv2 = decView.base.advanced(by: (y + 16) * 32)[x]
                            if ev2 != dv2 { totalDiff += 1 }
                            // HH
                            let ev3 = encView.base.advanced(by: (y + 16) * 32 + 16)[x]
                            let dv3 = decView.base.advanced(by: (y + 16) * 32 + 16)[x]
                            if ev3 != dv3 { totalDiff += 1 }
                        }
                    }
                }
            }
        }
        
        XCTAssertEqual(totalDiff, 0, "Real DWT blocks不一致: \(totalDiff) pixels, maxDiff=\(maxDiff), firstBlock=\(firstDiffBlock), totalBlocks=\(blocks.count)")
    }
    
    private func subbands32RoundtripForSize(width: Int, height: Int) throws -> (totalDiff: Int, firstBlock: Int, totalBlocks: Int) {
        var rY = [Int16](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let base = (x + y * 2) % 256
                let noise = (x &* 2654435761 ^ y &* 2246822519) % 20
                rY[y * width + x] = Int16(clamping: base + noise - 138)
            }
        }
        
        let qtY = QuantizationTable(baseStep: 2)
        let reader = Int16Reader(data: rY, width: width, height: height)
        var (blocks, _) = extractSingleTransformBlocks32(r: reader, width: width, height: height)
        for i in blocks.indices {
            evaluateQuantizeLayer32(block: &blocks[i], qt: qtY)
        }
        
        let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
        let data = encodePlaneSubbands32(blocks: &blocks, zeroThreshold: safeThreshold)
        let decBlocks = try decodePlaneSubbands32(data: data, blockCount: blocks.count)
        
        var totalDiff = 0
        var firstDiffBlock = -1
        for bi in 0..<blocks.count {
            var encBlk = blocks[bi]
            var decBlk = decBlocks[bi]
            encBlk.withView { encView in
                decBlk.withView { decView in
                    for y in 0..<16 {
                        for x in 0..<16 {
                            if encView.base.advanced(by: y * 32 + 16)[x] != decView.base.advanced(by: y * 32 + 16)[x] { totalDiff += 1; if firstDiffBlock < 0 { firstDiffBlock = bi } }
                            if encView.base.advanced(by: (y + 16) * 32)[x] != decView.base.advanced(by: (y + 16) * 32)[x] { totalDiff += 1 }
                            if encView.base.advanced(by: (y + 16) * 32 + 16)[x] != decView.base.advanced(by: (y + 16) * 32 + 16)[x] { totalDiff += 1 }
                        }
                    }
                }
            }
        }
        return (totalDiff, firstDiffBlock, blocks.count)
    }
    
    func testSubbands32Roundtrip_sizeSearch() throws {
        let sizes: [(Int, Int)] = [(64, 64), (128, 128), (192, 128), (192, 192), (256, 192), (320, 240)]
        for (w, h) in sizes {
            let (totalDiff, firstBlock, totalBlocks) = try subbands32RoundtripForSize(width: w, height: h)
            if 0 < totalDiff {
                XCTFail("\(w)x\(h) (\(totalBlocks)blocks): totalDiff=\(totalDiff) firstBlock=\(firstBlock)")
            }
        }
    }
    
    /// toPlaneData420経由のデータで各サイズテスト
    func testSubbands32Roundtrip_viaPD420() async throws {
        let sizes: [(Int, Int)] = [(128, 128), (256, 192), (320, 240), (640, 480)]
        for (w, h) in sizes {
            var img = YCbCrImage(width: w, height: h)
            for y in 0..<h {
                for x in 0..<w {
                    let base = (x + y * 2) % 256
                    let noise = (x &* 2654435761 ^ y &* 2246822519) % 20
                    img.yPlane[y * w + x] = UInt8(clamping: base + noise - 10)
                }
            }
            let cW = (w + 1) / 2
            let cH = (h + 1) / 2
            for cy in 0..<cH {
                for cx in 0..<cW {
                    img.cbPlane[cy * cW + cx] = 128
                    img.crPlane[cy * cW + cx] = 128
                }
            }
            
            let pd = toPlaneData420(images: [img])[0]
            let qtY = QuantizationTable(baseStep: 2)
            
            var (blocks, _) = extractSingleTransformBlocks32(r: pd.rY, width: w, height: h)
            for i in blocks.indices {
                evaluateQuantizeLayer32(block: &blocks[i], qt: qtY)
            }
            
            let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
            let data = encodePlaneSubbands32(blocks: &blocks, zeroThreshold: safeThreshold)
            let decBlocks = try decodePlaneSubbands32(data: data, blockCount: blocks.count)
            
            var totalDiff = 0
            var firstDiffBlock = -1
            for bi in 0..<blocks.count {
                var encBlk = blocks[bi]
                var decBlk = decBlocks[bi]
                encBlk.withView { encView in
                    decBlk.withView { decView in
                        for y in 0..<16 {
                            for x in 0..<16 {
                                if encView.base.advanced(by: y * 32 + 16)[x] != decView.base.advanced(by: y * 32 + 16)[x] { totalDiff += 1; if firstDiffBlock < 0 { firstDiffBlock = bi } }
                                if encView.base.advanced(by: (y + 16) * 32)[x] != decView.base.advanced(by: (y + 16) * 32)[x] { totalDiff += 1 }
                                if encView.base.advanced(by: (y + 16) * 32 + 16)[x] != decView.base.advanced(by: (y + 16) * 32 + 16)[x] { totalDiff += 1 }
                            }
                        }
                    }
                }
            }
            if 0 < totalDiff {
                XCTFail("\(w)x\(h) (\(blocks.count)blocks) PD420: totalDiff=\(totalDiff) firstBlock=\(firstDiffBlock)")
            }
        }
    }
}
