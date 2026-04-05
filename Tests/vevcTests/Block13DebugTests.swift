import XCTest
@testable import vevc

/// 128x128 block13 の不一致を詳細分析するテスト
final class Block13DebugTests: XCTestCase {
    
    func testBlock13Detail() async throws {
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
            evaluateQuantizeLayer32(block: &blocks[i], qt: qtY)
        }
        
        // エンコード前の各ブロックのisZero/split判定を確認
        let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
        
        var blockInfos: [(isZero: Bool, forceSplit: Bool)] = []
        for i in blocks.indices {
            let isZero = isEffectivelyZero32(data: blocks[i].base, threshold: safeThreshold)
            var forceSplit = false
            if isZero != true {
                forceSplit = shouldSplit32(data: blocks[i].base, skipLL: true)
            }
            blockInfos.append((isZero: isZero, forceSplit: forceSplit))
        }
        
        print("=== Block Info ===")
        for (i, info) in blockInfos.enumerated() {
            print("  block[\(i)]: isZero=\(info.isZero) forceSplit=\(info.forceSplit)")
        }
        
        // isEffectivelyZeroチェックがblockのデータを変更するため、元のブロックを再作成
        var (blocks2, _) = await extractSingleTransformBlocks32(r: pd.rY, width: width, height: height, pool: pool)
        for i in blocks2.indices {
            evaluateQuantizeLayer32(block: &blocks2[i], qt: qtY)
        }
        
        // encodePlaneSubbands32
        let data = encodePlaneSubbands32(blocks: &blocks2, zeroThreshold: safeThreshold, parentBlocks: nil)
        
        // decodePlaneSubbands32  
        let decBlocks = try decodePlaneSubbands32(data: data, pool: pool, blockCount: blocks2.count, parentBlocks: nil)
        
        // block 13 の詳細比較
        for bi in [12, 13, 14] {
            let encBlk = blocks2[bi]
            let decBlk = decBlocks[bi]
            
            var diffHL = 0, diffLH = 0, diffHH = 0
            var firstDiffDetail = ""
            
            let encView = encBlk
            let decView = decBlk
            // HL
            for y in 0..<16 {
                for x in 0..<16 {
                    let ev = encView.base.advanced(by: y * 32 + 16)[x]
                    let dv = decView.base.advanced(by: y * 32 + 16)[x]
                    if ev != dv {
                        diffHL += 1
                        if firstDiffDetail.isEmpty {
                            firstDiffDetail = "HL y:\(y) x:\(x) enc=\(ev) dec=\(dv)"
                        }
                    }
                }
            }
            // LH
            for y in 0..<16 {
                for x in 0..<16 {
                    let ev = encView.base.advanced(by: (y + 16) * 32)[x]
                    let dv = decView.base.advanced(by: (y + 16) * 32)[x]
                    if ev != dv { diffLH += 1 }
                }
            }
            // HH
            for y in 0..<16 {
                for x in 0..<16 {
                    let ev = encView.base.advanced(by: (y + 16) * 32 + 16)[x]
                    let dv = decView.base.advanced(by: (y + 16) * 32 + 16)[x]
                    if ev != dv { diffHH += 1 }
                }
            }
                            
            print("=== Block[\(bi)] ===")
            print("  HL diff: \(diffHL), LH diff: \(diffLH), HH diff: \(diffHH)")
            if firstDiffDetail.isEmpty != true {
                print("  First diff: \(firstDiffDetail)")
            }
        }
        
        // 全ブロックの不一致カウント
        var failBlocks: [Int] = []
        for bi in 0..<blocks2.count {
            let encBlk = blocks2[bi]
            let decBlk = decBlocks[bi]
            var hasDiff = false
            let encView = encBlk
            let decView = decBlk
            for y in 0..<16 {
                for x in 0..<16 {
                    if encView.base.advanced(by: y * 32 + 16)[x] != decView.base.advanced(by: y * 32 + 16)[x] { hasDiff = true }
                    if encView.base.advanced(by: (y + 16) * 32)[x] != decView.base.advanced(by: (y + 16) * 32)[x] { hasDiff = true }
                    if encView.base.advanced(by: (y + 16) * 32 + 16)[x] != decView.base.advanced(by: (y + 16) * 32 + 16)[x] { hasDiff = true }
                }
            }
                            if hasDiff { failBlocks.append(bi) }
        }
        
        print("=== Total fail blocks: \(failBlocks) ===")
        
        XCTAssertEqual(failBlocks.count, 0, "Fail blocks: \(failBlocks)")
    }
}
