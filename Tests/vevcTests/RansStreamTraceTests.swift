import XCTest
@testable import vevc

/// 4-way rANS エンコーダ/デコーダの各laneのstream消費量を直接比較
final class RansStreamTraceTests: XCTestCase {
    
    func testStreamConsumptionPerLane() async throws {
        // 実DWTデータのpairsを生成
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
        for i in blocks.indices { evaluateQuantizeLayer32(block: &blocks[i], qt: qtY) }
        let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        for i in blocks.indices {
            let isZero = blocks[i].data.withUnsafeMutableBufferPointer { ptr in
                return isEffectivelyZero32(data: ptr, threshold: safeThreshold)
            }
            if isZero { continue }
            blocks[i].withView { view in
                let subs = getSubbands32(view: view)
                blockEncode16(encoder: &encoder, block: subs.hl, parentBlock: nil)
                blockEncode16(encoder: &encoder, block: subs.lh, parentBlock: nil)
                blockEncode16(encoder: &encoder, block: subs.hh, parentBlock: nil)
            }
        }
        
        let pairs = encoder.pairs
        let pairCount = pairs.count
        
        print("=== Total pairs: \(pairCount) ===")
        
        // --- エンコーダ側：各laneのstream.count変化を追跡 ---
        let chunkBase = pairCount / 4
        let chunkRemainder = pairCount % 4
        var chunkStarts = [Int](repeating: 0, count: 5)
        for i in 0..<4 {
            chunkStarts[i + 1] = chunkStarts[i] + chunkBase + (i < chunkRemainder ? 1 : 0)
        }
        
        // tokenization
        var chunkRunTokens = [[UInt8]](repeating: [], count: 4)
        var chunkValTokens = [[UInt8]](repeating: [], count: 4)
        var chunkBypassWriters = [BypassWriter](repeating: BypassWriter(), count: 4)
        var runTokenCounts = [Int](repeating: 0, count: 64)
        var valTokenCounts = [Int](repeating: 0, count: 64)
        
        for lane in 0..<4 {
            let start = chunkStarts[lane]
            let end = chunkStarts[lane + 1]
            for idx in start..<end {
                let pair = pairs[idx]
                let runResult = valueTokenizeUnsigned(pair.run)
                runTokenCounts[Int(runResult.token)] += 1
                chunkRunTokens[lane].append(runResult.token)
                chunkBypassWriters[lane].writeBits(runResult.bypassBits, count: runResult.bypassLen)
                let valResult = valueTokenize(pair.val)
                valTokenCounts[Int(valResult.token)] += 1
                chunkValTokens[lane].append(valResult.token)
                chunkBypassWriters[lane].writeBits(valResult.bypassBits, count: valResult.bypassLen)
            }
        }
        for lane in 0..<4 { chunkBypassWriters[lane].flush() }
        
        var runModel = rANSModel()
        runModel.normalize(sigCounts: [0, 1], tokenCounts: runTokenCounts)
        var valModel = rANSModel()
        valModel.normalize(sigCounts: [0, 1], tokenCounts: valTokenCounts)
        
        // エンコード with stream tracking
        var enc = Interleaved4rANSEncoder()
        var encStreamCountPerLane = [Int](repeating: 0, count: 4)
        
        for lane in stride(from: 3, through: 0, by: -1) {
            let streamCountBefore = enc.stream.count
            
            let runTokens = chunkRunTokens[lane]
            let valTokens = chunkValTokens[lane]
            let pairEnd = valTokens.count
            
            for i in stride(from: pairEnd - 1, through: 0, by: -1) {
                let vt = valTokens[i]
                enc.encodeSymbol(lane: lane, cumFreq: valModel.tokenCumFreqs[Int(vt)], freq: valModel.tokenFreqs[Int(vt)])
                let rt = runTokens[i]
                enc.encodeSymbol(lane: lane, cumFreq: runModel.tokenCumFreqs[Int(rt)], freq: runModel.tokenFreqs[Int(rt)])
            }
            
            encStreamCountPerLane[lane] = enc.stream.count - streamCountBefore
        }
        
        enc.flush()  // adds 8 words (4 lanes * 2 words)
        let bitstream = enc.getBitstream()
        
        print("=== Encoder stream per lane: \(encStreamCountPerLane) ===")
        print("=== Total stream words: \(enc.stream.count) (overflow: \(enc.stream.count - 8)) ===")
        print("=== Bitstream bytes: \(bitstream.count) ===")
        
        // --- デコーダ側：各laneのreadWord回数を追跡 ---
        // Interleaved4rANSDecoderを直接使わず、advanceSymbolの各ステップでoffset変化を追跡
        
        var decStates: (UInt32, UInt32, UInt32, UInt32) = (0, 0, 0, 0)
        var padded = bitstream
        padded.append(contentsOf: [0, 0, 0, 0, 0, 0, 0, 0])
        var offset = 0
        
        // state読み込み
        func readWord() -> UInt32 {
            if offset + 1 < padded.count {
                let b0 = UInt32(padded[offset])
                let b1 = UInt32(padded[offset + 1])
                offset += 2
                return (b0 << 8) | b1
            }
            return 0
        }
        
        func readState(_ off: Int) -> UInt32 {
            let b0 = UInt32(padded[off])
            let b1 = UInt32(padded[off + 1])
            let b2 = UInt32(padded[off + 2])
            let b3 = UInt32(padded[off + 3])
            return ((b0 << 8) | b1) << 16 | ((b2 << 8) | b3)
        }
        
        decStates.0 = readState(0)
        decStates.1 = readState(4)
        decStates.2 = readState(8)
        decStates.3 = readState(12)
        offset = 16
        
        print("=== Decoder initial states: \(decStates) ===")
        print("=== Encoder final states: \(enc.states) ===")
        XCTAssertEqual(decStates.0, enc.states.0, "state.0")
        XCTAssertEqual(decStates.1, enc.states.1, "state.1")
        XCTAssertEqual(decStates.2, enc.states.2, "state.2")
        XCTAssertEqual(decStates.3, enc.states.3, "state.3")
        
        var decReadWordsPerLane = [Int](repeating: 0, count: 4)
        let mask = RANS_SCALE - 1
        
        // 復元されたtokenFreqsとcumFreqsでモデル再構築
        let decRunModel = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: runModel.tokenFreqs)
        let decValModel = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: valModel.tokenFreqs)
        
        for lane in 0..<4 {
            let chunkSize = chunkStarts[lane + 1] - chunkStarts[lane]
            let offsetBefore = offset
            
            for _ in 0..<chunkSize {
                // decode run token
                let cfRun: UInt32
                switch lane {
                case 0: cfRun = decStates.0 & mask
                case 1: cfRun = decStates.1 & mask
                case 2: cfRun = decStates.2 & mask
                default: cfRun = decStates.3 & mask
                }
                let rtInfo = decRunModel.findToken(cf: cfRun)
                
                // advance run
                switch lane {
                case 0:
                    decStates.0 = rtInfo.freq * (decStates.0 >> UInt32(RANS_SCALE_BITS)) + (decStates.0 & mask) - rtInfo.cumFreq
                    while decStates.0 < RANS_L { decStates.0 = (decStates.0 << 16) | readWord(); decReadWordsPerLane[lane] += 1 }
                case 1:
                    decStates.1 = rtInfo.freq * (decStates.1 >> UInt32(RANS_SCALE_BITS)) + (decStates.1 & mask) - rtInfo.cumFreq
                    while decStates.1 < RANS_L { decStates.1 = (decStates.1 << 16) | readWord(); decReadWordsPerLane[lane] += 1 }
                case 2:
                    decStates.2 = rtInfo.freq * (decStates.2 >> UInt32(RANS_SCALE_BITS)) + (decStates.2 & mask) - rtInfo.cumFreq
                    while decStates.2 < RANS_L { decStates.2 = (decStates.2 << 16) | readWord(); decReadWordsPerLane[lane] += 1 }
                default:
                    decStates.3 = rtInfo.freq * (decStates.3 >> UInt32(RANS_SCALE_BITS)) + (decStates.3 & mask) - rtInfo.cumFreq
                    while decStates.3 < RANS_L { decStates.3 = (decStates.3 << 16) | readWord(); decReadWordsPerLane[lane] += 1 }
                }
                
                // decode val token
                let cfVal: UInt32
                switch lane {
                case 0: cfVal = decStates.0 & mask
                case 1: cfVal = decStates.1 & mask
                case 2: cfVal = decStates.2 & mask
                default: cfVal = decStates.3 & mask
                }
                let vtInfo = decValModel.findToken(cf: cfVal)
                
                // advance val
                switch lane {
                case 0:
                    decStates.0 = vtInfo.freq * (decStates.0 >> UInt32(RANS_SCALE_BITS)) + (decStates.0 & mask) - vtInfo.cumFreq
                    while decStates.0 < RANS_L { decStates.0 = (decStates.0 << 16) | readWord(); decReadWordsPerLane[lane] += 1 }
                case 1:
                    decStates.1 = vtInfo.freq * (decStates.1 >> UInt32(RANS_SCALE_BITS)) + (decStates.1 & mask) - vtInfo.cumFreq
                    while decStates.1 < RANS_L { decStates.1 = (decStates.1 << 16) | readWord(); decReadWordsPerLane[lane] += 1 }
                case 2:
                    decStates.2 = vtInfo.freq * (decStates.2 >> UInt32(RANS_SCALE_BITS)) + (decStates.2 & mask) - vtInfo.cumFreq
                    while decStates.2 < RANS_L { decStates.2 = (decStates.2 << 16) | readWord(); decReadWordsPerLane[lane] += 1 }
                default:
                    decStates.3 = vtInfo.freq * (decStates.3 >> UInt32(RANS_SCALE_BITS)) + (decStates.3 & mask) - vtInfo.cumFreq
                    while decStates.3 < RANS_L { decStates.3 = (decStates.3 << 16) | readWord(); decReadWordsPerLane[lane] += 1 }
                }
            }
            
            let offsetAfter = offset
            let bytesConsumed = offsetAfter - offsetBefore
            print("=== Lane[\(lane)]: readWords=\(decReadWordsPerLane[lane]) bytesConsumed=\(bytesConsumed) encStreamWords=\(encStreamCountPerLane[lane]) ===")
        }
        
        // 各laneのreadWords と encStreamCountを比較  
        for lane in 0..<4 {
            XCTAssertEqual(decReadWordsPerLane[lane], encStreamCountPerLane[lane], 
                          "Lane[\(lane)] stream mismatch: dec.readWords=\(decReadWordsPerLane[lane]) enc.streamWords=\(encStreamCountPerLane[lane])")
        }
    }
}
