import XCTest
@testable import vevc

/// EntropyEncoder/Decoder のpairsを128x128 PD420データで直接比較
final class EntropyPairsCompareTests: XCTestCase {
    
    func testEntropyPairsRoundtrip_128x128() async throws {
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
        
        var (blocks, _) = await extractSingleTransformBlocks32(r: pd.rY, width: width, height: height)
        for i in blocks.indices {
            evaluateQuantizeLayer32(block: &blocks[i], qt: qtY)
        }
        
        let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
        
        // encodePlaneSubbands32 と同じフラグ判定を行い、encode16タスクを構築
        var bwFlags = BypassWriter()
        var tasks: [(Int, EncodeTask32)] = []
        
        for i in blocks.indices {
            let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
                return isEffectivelyZero32(data: ptr, threshold: safeThreshold)
            }
            if isZero {
                bwFlags.writeBit(true)
                blocks[i].withView { view in
                    let half = 32 / 2
                    let base = view.base
                    let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
                    let lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
                    let hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
                    hlView.clearAll()
                    lhView.clearAll()
                    hhView.clearAll()
                }
            } else {
                bwFlags.writeBit(false)
                
                let forceSplit = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
                    return shouldSplit32(data: ptr, skipLL: true)
                }
                if forceSplit {
                    bwFlags.writeBit(true)
                    bwFlags.writeBit(false); bwFlags.writeBit(false)
                    bwFlags.writeBit(false); bwFlags.writeBit(false)
                    bwFlags.writeBit(false); bwFlags.writeBit(false)
                    bwFlags.writeBit(false); bwFlags.writeBit(false)
                    tasks.append((i, .split8(true, true, true, true)))
                } else {
                    bwFlags.writeBit(false)
                    tasks.append((i, .encode16))
                }
            }
        }
        bwFlags.flush()
        
        // エンコーダ: EntropyEncoderを直接使って blockEncode16 を呼ぶ
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        
        for (i, task) in tasks {
            blocks[i].withView { view in
                let subs = getSubbands32(view: view)
                switch task {
                case .encode16:
                    blockEncode16(encoder: &encoder, block: subs.hl, parentBlock: nil)
                    blockEncode16(encoder: &encoder, block: subs.lh, parentBlock: nil)
                    blockEncode16(encoder: &encoder, block: subs.hh, parentBlock: nil)
                case .split8(let tl, let tr, let bl, let br):
                    if tl {
                        blockEncode8(encoder: &encoder, block: BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32), parentBlock: nil)
                        blockEncode8(encoder: &encoder, block: BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32), parentBlock: nil)
                        blockEncode8(encoder: &encoder, block: BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32), parentBlock: nil)
                    }
                    if tr {
                        blockEncode8(encoder: &encoder, block: BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32), parentBlock: nil)
                        blockEncode8(encoder: &encoder, block: BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32), parentBlock: nil)
                        blockEncode8(encoder: &encoder, block: BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32), parentBlock: nil)
                    }
                    if bl {
                        blockEncode8(encoder: &encoder, block: BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32), parentBlock: nil)
                        blockEncode8(encoder: &encoder, block: BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32), parentBlock: nil)
                        blockEncode8(encoder: &encoder, block: BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32), parentBlock: nil)
                    }
                    if br {
                        blockEncode8(encoder: &encoder, block: BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32), parentBlock: nil)
                        blockEncode8(encoder: &encoder, block: BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32), parentBlock: nil)
                        blockEncode8(encoder: &encoder, block: BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32), parentBlock: nil)
                    }
                }
            }
        }
        
        // エンコーダのpairとbypassを保存
        let encPairs = encoder.pairs
        encoder.flush()
        let encBypassBytes = encoder.bypassWriter.bytes
        let encCoeffCount = encoder.coeffCount
        
        // getData()でバイト列を取得
        let entropyData = encoder.getData()
        
        // デコーダでpairsを復元
        var decoder = try EntropyDecoder(data: entropyData)
        var decPairs: [(run: Int, val: Int16)] = []
        for i in 0..<encoder.pairs.count {
            let pair = decoder.readPair(contextIdx: encoder.pairs[i].contextIdx)
            decPairs.append(pair)
        }
        
        print("=== EntropyEncoder pairs count: \(encPairs.count) ===")
        print("=== EntropyDecoder pairs count: \(decPairs.count) ===")
        print("=== coeffCount: \(encCoeffCount) ===")
        print("=== encBypassBytes: \(encBypassBytes.count) ===")
        print("=== entropyData: \(entropyData.count) bytes ===")
        
        // pairs数の比較
        XCTAssertEqual(encPairs.count, decPairs.count, "pairs count mismatch: enc=\(encPairs.count) dec=\(decPairs.count)")
        
        // 各pairの比較
        var firstDiff = -1
        for i in 0..<min(encPairs.count, decPairs.count) {
            if encPairs[i].run != decPairs[i].run || encPairs[i].val != decPairs[i].val {
                if firstDiff < 0 {
                    firstDiff = i
                    print("First pair diff at [\(i)]: enc=(\(encPairs[i].run), \(encPairs[i].val)) dec=(\(decPairs[i].run), \(decPairs[i].val))")
                }
            }
        }
        
        XCTAssertEqual(firstDiff, -1, "Pairs differ starting at index \(firstDiff)")
        
        // bypassのデコード比較（blockDecode16内のdecodeBypass呼び出しを再現）
        var decoder2 = try EntropyDecoder(data: entropyData)
        var decBlocks = (0..<blocks.count).map { _ in Block2D(width: 32, height: 32) }
        
        for (i, task) in tasks {
            try decBlocks[i].withView { view in
                let subs = getSubbands32(view: view)
                switch task {
                case .encode16:
                    var hlView = BlockView(base: subs.hl.base, width: 16, height: 16, stride: 32)
                    try blockDecode16(decoder: &decoder2, block: &hlView, parentBlock: nil)
                    var lhView = BlockView(base: subs.lh.base, width: 16, height: 16, stride: 32)
                    try blockDecode16(decoder: &decoder2, block: &lhView, parentBlock: nil)
                    var hhView = BlockView(base: subs.hh.base, width: 16, height: 16, stride: 32)
                    try blockDecode16(decoder: &decoder2, block: &hhView, parentBlock: nil)
                case .split8(let tl, let tr, let bl, let br):
                    if tl {
                        var hl = BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32)
                        var lh = BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32)
                        var hh = BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32)
                        try blockDecode8(decoder: &decoder2, block: &hl, parentBlock: nil)
                        try blockDecode8(decoder: &decoder2, block: &lh, parentBlock: nil)
                        try blockDecode8(decoder: &decoder2, block: &hh, parentBlock: nil)
                    }
                    if tr {
                        var hl = BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                        var lh = BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                        var hh = BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                        try blockDecode8(decoder: &decoder2, block: &hl, parentBlock: nil)
                        try blockDecode8(decoder: &decoder2, block: &lh, parentBlock: nil)
                        try blockDecode8(decoder: &decoder2, block: &hh, parentBlock: nil)
                    }
                    if bl {
                        var hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                        var lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                        var hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                        try blockDecode8(decoder: &decoder2, block: &hl, parentBlock: nil)
                        try blockDecode8(decoder: &decoder2, block: &lh, parentBlock: nil)
                        try blockDecode8(decoder: &decoder2, block: &hh, parentBlock: nil)
                    }
                    if br {
                        var hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                        var lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                        var hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                        try blockDecode8(decoder: &decoder2, block: &hl, parentBlock: nil)
                        try blockDecode8(decoder: &decoder2, block: &lh, parentBlock: nil)
                        try blockDecode8(decoder: &decoder2, block: &hh, parentBlock: nil)
                    }
                }
            }
        }
        
        // HL/LH/HH比較
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
            XCTFail("Direct entropy roundtrip: totalDiff=\(totalDiff) firstDiffBlock=\(firstDiffBlock)")
        }
    }
}
