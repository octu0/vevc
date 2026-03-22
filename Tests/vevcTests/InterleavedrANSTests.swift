import XCTest
@testable import vevc

final class InterleavedrANSTests: XCTestCase {
    
    struct EncodedData {
        let isSignificant: Bool
        let token: UInt8
        let bypassBits: UInt32
        let bypassLen: Int
    }
    
    func testInterleavedrANSEncodeDecodeAndPerformance() {
        var rng = SystemRandomNumberGenerator()
        
        let count = 8192 * 4 // 約32K係数
        var testData = [Int16]()
        testData.reserveCapacity(count)
        
        // ヘビーテール分布を模倣
        for _ in 0..<count {
            let p = Int.random(in: 0..<100, using: &rng)
            switch true {
            case p < 80:
                testData.append(0)
            case p < 98:
                testData.append(Int16.random(in: -3...3, using: &rng))
            default:
                testData.append(Int16.random(in: -255...255, using: &rng))
            }
        }
        
        var tokens = [EncodedData]()
        tokens.reserveCapacity(count)
        
        var sigCounts: [Int] = [0, 0]
        var tokenCounts: [Int] = Array(repeating: 0, count: 64)
        
        for val in testData {
            let isSig = val != 0
            if isSig {
                let t = ValueTokenizer.tokenize(val)
                tokens.append(EncodedData(isSignificant: true, token: t.token, bypassBits: t.bypassBits, bypassLen: t.bypassLen))
                sigCounts[1] += 1
                tokenCounts[Int(t.token)] += 1
            } else {
                tokens.append(EncodedData(isSignificant: false, token: 0, bypassBits: 0, bypassLen: 0))
                sigCounts[0] += 1
            }
        }
        
        var model = rANSModel()
        model.normalize(sigCounts: sigCounts, tokenCounts: tokenCounts)
        
        // ---- 4-way Encode ----
        var encoder = InterleavedrANSEncoder()
        var bypassWriters = [BypassWriter](repeating: BypassWriter(), count: 4)
        
        let chunkSize = count / 4
        
        // Backward rANS
        for lane in 0..<4 {
            let start = lane * chunkSize
            let end = start + chunkSize
            for i in stride(from: end - 1, through: start, by: -1) {
                let t = tokens[i]
                if t.isSignificant {
                    let freq = model.tokenFreqs[Int(t.token)]
                    let cumFreq = model.tokenCumFreqs[Int(t.token)]
                    encoder.encodeSymbol(lane: lane, cumFreq: cumFreq, freq: freq)
                    encoder.encodeSymbol(lane: lane, cumFreq: 0, freq: model.sigFreq)
                } else {
                    encoder.encodeSymbol(lane: lane, cumFreq: model.sigFreq, freq: RANS_SCALE - model.sigFreq)
                }
            }
        }
        encoder.flush()
        
        // Forward Bypass
        // Bypassデータは順方向に書き込んでおき、デコード後に順方向のまま取り出して、
        // 逆順で得られたrANSのデコード結果配列（restoredByLane）を反転させた後で結合する。
        for lane in 0..<4 {
            let start = lane * chunkSize
            let end = start + chunkSize
            for i in start..<end {
                let t = tokens[i]
                if t.isSignificant {
                    bypassWriters[lane].writeBits(t.bypassBits, count: t.bypassLen)
                }
            }
            bypassWriters[lane].flush()
        }
        
        let ransStream = encoder.getBitstream()
        let bypassStreams = bypassWriters.map { $0.bytes }
        
        // ---- 4-way Decode ----
        var decoder = InterleavedrANSDecoder(bitstream: ransStream)
        var bypassReaders = bypassStreams.map { BypassReader(data: $0) }
        
        let sigFreqVec = SIMD4<UInt32>(repeating: model.sigFreq)
        let invSigFreqVec = SIMD4<UInt32>(repeating: RANS_SCALE - model.sigFreq)
        let zeroVec = SIMD4<UInt32>(repeating: 0)
        
        var finalDecodingResult = [[(isSignificant: Bool, token: UInt8)]](repeating: [], count: 4)
        
        measure {
            decoder = InterleavedrANSDecoder(bitstream: ransStream)
            var rANSDecodedByLane = [[(isSignificant: Bool, token: UInt8)]](repeating: [], count: 4)
            for i in 0..<4 {
                rANSDecodedByLane[i].reserveCapacity(chunkSize)
            }
            
            // Decoding loop (LIFO Order)
            for _ in 0..<chunkSize {
                let cfs = decoder.getCumulativeFreqs()
                var isSigs = [Bool](repeating: false, count: 4)
                
                var sigAdvanceCumFreq = SIMD4<UInt32>(repeating: 0)
                var sigAdvanceFreq = SIMD4<UInt32>(repeating: 0)
                
                for lane in 0..<4 {
                    if cfs[lane] < model.sigFreq {
                        isSigs[lane] = true
                        sigAdvanceFreq[lane] = sigFreqVec[lane]
                        sigAdvanceCumFreq[lane] = zeroVec[lane]
                    } else {
                        isSigs[lane] = false
                        sigAdvanceFreq[lane] = invSigFreqVec[lane]
                        sigAdvanceCumFreq[lane] = sigFreqVec[lane]
                    }
                }
                
                decoder.advanceSymbols(cumFreqs: sigAdvanceCumFreq, freqs: sigAdvanceFreq)
                
                let cfTokens = decoder.getCumulativeFreqs()
                var tokenAdvanceCumFreq = SIMD4<UInt32>(repeating: 0)
                var tokenAdvanceFreq = SIMD4<UInt32>(repeating: 0)
                var readToken = [UInt8](repeating: 0, count: 4)
                
                var advanceMask = SIMD4<UInt32>(repeating: 0)
                for lane in 0..<4 {
                    if isSigs[lane] {
                        let tInfo = model.findToken(cf: cfTokens[lane])
                        readToken[lane] = tInfo.token
                        tokenAdvanceCumFreq[lane] = tInfo.cumFreq
                        tokenAdvanceFreq[lane] = tInfo.freq
                        advanceMask[lane] = 0xFFFFFFFF
                    }
                }
                
                decoder.advanceSymbols(cumFreqs: tokenAdvanceCumFreq, freqs: tokenAdvanceFreq, activeMask: advanceMask)
                
                // Read bypass ではなくまずは rANS 結果のみ保存
                for lane in 0..<4 {
                    rANSDecodedByLane[lane].append((isSignificant: isSigs[lane], token: readToken[lane]))
                }
            }
            finalDecodingResult = rANSDecodedByLane
        }
        
        // Assemble and read Bypass in Forward Order
        var restoredByLane = [[EncodedData]](repeating: [], count: 4)
        for i in 0..<4 {
            restoredByLane[i].reserveCapacity(chunkSize)
        }
        
        for lane in 0..<4 {
            // エンコードがBackwardなので、デコード結果(rANSDecodedByLane)はForwardで返ってくる
            let forwardLaneTokens = finalDecodingResult[lane]
            
            for t in forwardLaneTokens {
                if t.isSignificant {
                    let bypassLen = ValueTokenizer.bypassLength(for: t.token)
                    let bypassBits = bypassReaders[lane].readBits(count: bypassLen)
                    restoredByLane[lane].append(EncodedData(isSignificant: true, token: t.token, bypassBits: bypassBits, bypassLen: bypassLen))
                } else {
                    restoredByLane[lane].append(EncodedData(isSignificant: false, token: 0, bypassBits: 0, bypassLen: 0))
                }
            }
        }
        
        // Assemble and verify
        // tokens is sequential [0..<count]
        var allTokensMatches = true
        for lane in 0..<4 {
            let start = lane * chunkSize
            // restoredByLane[lane] は既に Forward 順序に直っているためそのまま比較
            for i in 0..<chunkSize {
                let orig = tokens[start + i]
                let rest = restoredByLane[lane][i]
                
                if orig.isSignificant != rest.isSignificant {
                    XCTFail("Mismatch at lane \(lane) offset \(i): Expected sig \(orig.isSignificant), got \(rest.isSignificant)")
                    allTokensMatches = false
                    break
                }
                if orig.isSignificant {
                    if orig.token != rest.token {
                        XCTFail("Mismatch at lane \(lane) offset \(i): Expected token \(orig.token), got \(rest.token)")
                        allTokensMatches = false
                        break
                    }
                    if orig.bypassBits != rest.bypassBits {
                        XCTFail("Mismatch at lane \(lane) offset \(i): Expected bypass \(orig.bypassBits), got \(rest.bypassBits)")
                        allTokensMatches = false
                        break
                    }
                }
            }
            if allTokensMatches != true {
                break
            }
        }
    }
}
