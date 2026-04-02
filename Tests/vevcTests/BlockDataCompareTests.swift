import XCTest
@testable import vevc

/// エンコーダとデコーダのブロック配列データを直接比較
final class BlockDataCompareTests: XCTestCase {
    
    /// encodePlaneLayer32 が返す blocks (HL/LH/HH) と decodePlaneSubbands32 が返す blocks を直接比較
    func testLayer32BlocksMatch() async throws {
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
        let qtC = QuantizationTable(baseStep: 6)
        
        // エンコーダ: Layer32 のバイト + blocks を取得
        var (_, encYBlocks, encCbBlocks, encCrBlocks) = try await preparePlaneLayer32(pd: pd, sads: nil, layer: 2, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        let layer2Bytes = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY, qtC: qtC, zeroThreshold: 3, yBlocks: &encYBlocks, cbBlocks: &encCbBlocks, crBlocks: &encCrBlocks, parentYBlocks: nil, parentCbBlocks: nil, parentCrBlocks: nil)

        
        // デコーダ: 同じバイトからblocks をデコード
        // decodePlaneSubbands32 は decodePlaneSubbands32(data:blockCount:) で直接呼べる
        // ただしLayer32ヘッダを解析してdataを取り出す必要がある
        var offset = 0
        let _ = Int(try readUInt16BEFromBytes(Array(layer2Bytes), offset: &offset)) // qtY step
        let _ = Int(try readUInt16BEFromBytes(Array(layer2Bytes), offset: &offset)) // qtC step
        
        let bufYLen = Int(try readUInt32BEFromBytes(Array(layer2Bytes), offset: &offset))
        let bufY = Array(layer2Bytes[offset..<(offset + bufYLen)])
        
        let rowCountY = (height + 32 - 1) / 32
        let colCountY = (width + 32 - 1) / 32
        let decYBlocks = try decodePlaneSubbands32(data: bufY, blockCount: rowCountY * colCountY, parentBlocks: nil)
        
        // encYBlocks vs decYBlocks のHL/LH/HHサブバンドを比較
        XCTAssertEqual(encYBlocks.count, decYBlocks.count, "blocks count mismatch: enc=\(encYBlocks.count) dec=\(decYBlocks.count)")
        
        var totalDiffHL = 0
        var totalDiffLH = 0
        var totalDiffHH = 0
        var maxDiffHL = 0
        var firstDiffBlockHL = -1
        
        for i in 0..<min(encYBlocks.count, decYBlocks.count) {
            let encBlk = encYBlocks[i]
            let decBlk = decYBlocks[i]
            
            encBlk.withView { encView in
                decBlk.withView { decView in
                    let half = 16
                    // HL comparison
                    for y in 0..<half {
                        let encPtr = encView.base.advanced(by: y * 32 + half)
                        let decPtr = decView.base.advanced(by: y * 32 + half)
                        for x in 0..<half {
                            let d = abs(Int(encPtr[x]) - Int(decPtr[x]))
                            if 0 < d {
                                totalDiffHL += 1
                                if maxDiffHL < d { maxDiffHL = d }
                                if firstDiffBlockHL < 0 { firstDiffBlockHL = i }
                            }
                        }
                    }
                    // LH comparison
                    for y in 0..<half {
                        let encPtr = encView.base.advanced(by: (y + half) * 32)
                        let decPtr = decView.base.advanced(by: (y + half) * 32)
                        for x in 0..<half {
                            if encPtr[x] != decPtr[x] { totalDiffLH += 1 }
                        }
                    }
                    // HH comparison
                    for y in 0..<half {
                        let encPtr = encView.base.advanced(by: (y + half) * 32 + half)
                        let decPtr = decView.base.advanced(by: (y + half) * 32 + half)
                        for x in 0..<half {
                            if encPtr[x] != decPtr[x] { totalDiffHH += 1 }
                        }
                    }
                }
            }
        }
        
        XCTAssertEqual(totalDiffHL, 0, "Layer32 Y HL不一致: \(totalDiffHL) pixels, maxDiff=\(maxDiffHL), firstBlock=\(firstDiffBlockHL)")
        XCTAssertEqual(totalDiffLH, 0, "Layer32 Y LH不一致: \(totalDiffLH) pixels")
        XCTAssertEqual(totalDiffHH, 0, "Layer32 Y HH不一致: \(totalDiffHH) pixels")
    }
}
