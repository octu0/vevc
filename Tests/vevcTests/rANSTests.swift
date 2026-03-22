import XCTest
@testable import vevc

final class rANSTests: XCTestCase {
    
    struct EncodedData {
        let isSignificant: Bool
        let token: UInt8
        let bypassBits: UInt32
        let bypassLen: Int
    }
    
    func testRANSEncodeDecodeAndCompare() {
        var rng = SystemRandomNumberGenerator()
        
        let count = 8192
        var testData = [Int16]()
        testData.reserveCapacity(count)
        
        for _ in 0..<count {
            let p = Int.random(in: 0..<100, using: &rng)
            switch true {
            case p < 80:
                // 80%: 0
                testData.append(0)
            case p < 98:
                // 18%: -3 to 3
                testData.append(Int16.random(in: -3...3, using: &rng))
            default:
                // 2%: 外れ値
                testData.append(Int16.random(in: -255...255, using: &rng))
            }
        }
        
        // Tokenizationと周波数のカウント
        var tokens = [EncodedData]()
        tokens.reserveCapacity(count)
        
        var sigCounts: [Int] = [0, 0] // [falseCount, trueCount]
        var tokenCounts: [Int] = Array(repeating: 0, count: 64)
        
        for val in testData {
            let isSig = val != 0
            if isSig {
                let t = valueTokenize(val)
                tokens.append(EncodedData(isSignificant: true, token: t.token, bypassBits: t.bypassBits, bypassLen: t.bypassLen))
                sigCounts[1] += 1
                tokenCounts[Int(t.token)] += 1
            } else {
                tokens.append(EncodedData(isSignificant: false, token: 0, bypassBits: 0, bypassLen: 0))
                sigCounts[0] += 1
            }
        }
        
        // rANSModelの構築
        var model = rANSModel()
        model.normalize(sigCounts: sigCounts, tokenCounts: tokenCounts)
        
        // -------- rANS Encode --------
        var encoder = rANSEncoder()
        var bypassWriter = BypassWriter()
        
        // Pass 1: Forward for Bypass
        for t in tokens {
            if t.isSignificant {
                bypassWriter.writeBits(t.bypassBits, count: t.bypassLen)
            }
        }
        bypassWriter.flush()
        
        // Pass 2: Backward for rANS
        for t in tokens.reversed() {
            if t.isSignificant {
                let tFreq = model.tokenFreqs[Int(t.token)]
                let tCumFreq = model.tokenCumFreqs[Int(t.token)]
                encoder.encodeSymbol(cumFreq: tCumFreq, freq: tFreq)
                encoder.encodeSymbol(cumFreq: 0, freq: model.sigFreq)
            } else {
                encoder.encodeSymbol(cumFreq: model.sigFreq, freq: RANS_SCALE - model.sigFreq)
            }
        }
        encoder.flush()
        
        let ransStream = encoder.getBitstream()
        let bypassStream = bypassWriter.bytes
        
        let totalRANSSize = ransStream.count + bypassStream.count
        print("[rANSTests] Compressed Size (rANS): \(totalRANSSize) bytes")
        
        // -------- rANS Decode --------
        var decoder = rANSDecoder(bitstream: ransStream)
        var bypassReader = BypassReader(data: bypassStream)
        
        var restoredTokens = [EncodedData]()
        restoredTokens.reserveCapacity(count)
        
        for _ in 0..<count {
            let cfSig = decoder.getCumulativeFreq()
            let isSig: Bool
            let freqSig: UInt32
            let cumFreqSig: UInt32
            
            if cfSig < model.sigFreq {
                // true
                isSig = true
                freqSig = model.sigFreq
                cumFreqSig = 0
            } else {
                // false
                isSig = false
                freqSig = RANS_SCALE - model.sigFreq
                cumFreqSig = model.sigFreq
            }
            decoder.advanceSymbol(cumFreq: cumFreqSig, freq: freqSig)
            
            if isSig {
                let cfToken = decoder.getCumulativeFreq()
                let tInfo = model.findToken(cf: cfToken)
                
                decoder.advanceSymbol(cumFreq: tInfo.cumFreq, freq: tInfo.freq)
                
                let bypassLen = valueBypassLength(for: tInfo.token)
                let bypassBits = bypassReader.readBits(count: bypassLen)
                
                restoredTokens.append(EncodedData(isSignificant: true, token: tInfo.token, bypassBits: bypassBits, bypassLen: bypassLen))
            } else {
                restoredTokens.append(EncodedData(isSignificant: false, token: 0, bypassBits: 0, bypassLen: 0))
            }
        }
        
        // データの検証
        XCTAssertEqual(restoredTokens.count, tokens.count)
        for i in 0..<count {
            let orig = tokens[i]
            let rest = restoredTokens[i]
            XCTAssertEqual(orig.isSignificant, rest.isSignificant, "Mismatch at \(i)")
            if orig.isSignificant {
                XCTAssertEqual(orig.token, rest.token, "Token mismatch at \(i)")
                XCTAssertEqual(orig.bypassBits, rest.bypassBits, "BypassBits mismatch at \(i)")
            }
        }
    }
}
