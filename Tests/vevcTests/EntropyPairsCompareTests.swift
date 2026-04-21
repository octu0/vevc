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
        let pool = BlockViewPool()
        
        let (blocks, _, rel) = await extractSingleTransformBlocks32(r: pd.rY, width: width, height: height, pool: pool, qt: qtY)
        defer { rel() }
        for i in blocks.indices {
            evaluateQuantizeLayer32(view: blocks[i], qt: qtY)
        }
        
        let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
        
        // encodePlaneSubbands32 と同じフラグ判定を行い、encode16タスクを構築
        var bwFlags = BypassWriter()
        var tasks: [(Int, EncodeTask32)] = []
        
        for i in blocks.indices {
            let isZero = isEffectivelyZero32(data: blocks[i].base, threshold: safeThreshold)
            if isZero {
                bwFlags.writeBit(true)
                let view = blocks[i]
                let half = 32 / 2
                let base = view.base
                let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 32)
                let lhView = BlockView(base: base.advanced(by: half * 32), width: half, height: half, stride: 32)
                let hhView = BlockView(base: base.advanced(by: half * 32 + half), width: half, height: half, stride: 32)
                hlView.clearAll()
                lhView.clearAll()
                hhView.clearAll()
                        } else {
                bwFlags.writeBit(false)
                
                let forceSplit = shouldSplit32WithoutLL(data: blocks[i].base)
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
            let view = blocks[i]
            let subs = getSubbands32(view: view)
            switch task {
            case .encode16:
                blockEncode16V(encoder: &encoder, block: subs.hl)
                blockEncode16H(encoder: &encoder, block: subs.lh)
                blockEncode16H(encoder: &encoder, block: subs.hh)
            case .split8(let tl, let tr, let bl, let br):
                if tl {
                    blockEncode8V(encoder: &encoder, block: BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32))
                    blockEncode8H(encoder: &encoder, block: BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32))
                    blockEncode8H(encoder: &encoder, block: BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32))
                }
                if tr {
                    blockEncode8V(encoder: &encoder, block: BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32))
                    blockEncode8H(encoder: &encoder, block: BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32))
                    blockEncode8H(encoder: &encoder, block: BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32))
                }
                if bl {
                    blockEncode8V(encoder: &encoder, block: BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32))
                    blockEncode8H(encoder: &encoder, block: BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32))
                    blockEncode8H(encoder: &encoder, block: BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32))
                }
                if br {
                    blockEncode8V(encoder: &encoder, block: BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32))
                    blockEncode8H(encoder: &encoder, block: BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32))
                    blockEncode8H(encoder: &encoder, block: BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32))
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
        try entropyData.withUnsafeBufferPointer { ptr in
            var decoder = try EntropyDecoder(base: ptr.baseAddress!, count: ptr.count)
            var decPairs: [(run: Int, val: Int16)] = []
            for i in 0..<encoder.pairs.count {
                let pair = decoder.readPair(isParentZero: encoder.pairs[i].isParentZero)
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
        }
        
        // bypassのデコード比較（blockDecode16内のdecodeBypass呼び出しを再現）
        let decBlocks = (0..<blocks.count).map { _ in BlockView.allocate(width: 32, height: 32) }
        try entropyData.withUnsafeBufferPointer { ptr in
            var decoder2 = try EntropyDecoder(base: ptr.baseAddress!, count: ptr.count)
            
            for (i, task) in tasks {
                let view = decBlocks[i]
                let subs = getSubbands32(view: view)
                switch task {
                case .encode16:
                    let hlView = BlockView(base: subs.hl.base, width: 16, height: 16, stride: 32)
                    try blockDecode16V(decoder: &decoder2, block: hlView)
                    let lhView = BlockView(base: subs.lh.base, width: 16, height: 16, stride: 32)
                    try blockDecode16H(decoder: &decoder2, block: lhView)
                    let hhView = BlockView(base: subs.hh.base, width: 16, height: 16, stride: 32)
                    try blockDecode16H(decoder: &decoder2, block: hhView)
                case .split8(let tl, let tr, let bl, let br):
                    if tl {
                        let hl = BlockView(base: subs.hl.base, width: 8, height: 8, stride: 32)
                        let lh = BlockView(base: subs.lh.base, width: 8, height: 8, stride: 32)
                        let hh = BlockView(base: subs.hh.base, width: 8, height: 8, stride: 32)
                        try blockDecode8V(decoder: &decoder2, block: hl)
                        try blockDecode8H(decoder: &decoder2, block: lh)
                        try blockDecode8H(decoder: &decoder2, block: hh)
                    }
                    if tr {
                        let hl = BlockView(base: subs.hl.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                        let lh = BlockView(base: subs.lh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                        let hh = BlockView(base: subs.hh.base.advanced(by: 8), width: 8, height: 8, stride: 32)
                        try blockDecode8V(decoder: &decoder2, block: hl)
                        try blockDecode8H(decoder: &decoder2, block: lh)
                        try blockDecode8H(decoder: &decoder2, block: hh)
                    }
                    if bl {
                        let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                        let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                        let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32), width: 8, height: 8, stride: 32)
                        try blockDecode8V(decoder: &decoder2, block: hl)
                        try blockDecode8H(decoder: &decoder2, block: lh)
                        try blockDecode8H(decoder: &decoder2, block: hh)
                    }
                    if br {
                        let hl = BlockView(base: subs.hl.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                        let lh = BlockView(base: subs.lh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                        let hh = BlockView(base: subs.hh.base.advanced(by: 8 * 32 + 8), width: 8, height: 8, stride: 32)
                        try blockDecode8V(decoder: &decoder2, block: hl)
                        try blockDecode8H(decoder: &decoder2, block: lh)
                        try blockDecode8H(decoder: &decoder2, block: hh)
                    }
                }
            }
        }
        
        // HL/LH/HH比較
        var totalDiff = 0
        var firstDiffBlock = -1
        for bi in 0..<blocks.count {
            let encBlk = blocks[bi]
            let decBlk = decBlocks[bi]
            let encView = encBlk
            let decView = decBlk
            for y in 0..<16 {
                for x in 0..<16 {
                    if encView.base.advanced(by: y * 32 + 16)[x] != decView.base.advanced(by: y * 32 + 16)[x] { totalDiff += 1; if firstDiffBlock < 0 { firstDiffBlock = bi } }
                    if encView.base.advanced(by: (y + 16) * 32)[x] != decView.base.advanced(by: (y + 16) * 32)[x] { totalDiff += 1 }
                    if encView.base.advanced(by: (y + 16) * 32 + 16)[x] != decView.base.advanced(by: (y + 16) * 32 + 16)[x] { totalDiff += 1 }
                }
            }
        }
        
        if 0 < totalDiff {
            XCTFail("Direct entropy roundtrip: totalDiff=\(totalDiff) firstDiffBlock=\(firstDiffBlock)")
        }
    }
}
