import XCTest
@testable import vevc

final class InterleavedrANSTests: XCTestCase {

    /// Interleaved4rANSEncoder/Decoder のラウンドトリップテスト
    /// エンコード/デコード順序:
    ///   エンコード: lane 3→0, 各lane内で逆順 (backward rANS)
    ///   デコード: lane 0→3, 各lane内で順に1シンボルずつ交互 (interleaved)
    func testInterleavedrANSEncodeDecodeAndPerformance() {
        var rng = SystemRandomNumberGenerator()

        let pairCount = 4096
        // run/val ペアを生成
        var runTokens = [UInt8](repeating: 0, count: pairCount)
        var valTokens = [UInt8](repeating: 0, count: pairCount)
        var runTokenCounts = [Int](repeating: 0, count: 64)
        var valTokenCounts = [Int](repeating: 0, count: 64)

        for i in 0..<pairCount {
            let p = Int.random(in: 0..<100, using: &rng)
            let val: Int16
            switch true {
            case p < 80:
                val = Int16.random(in: -3...3, using: &rng)
            default:
                val = Int16.random(in: -64...64, using: &rng)
            }
            let run = UInt32.random(in: 0...15, using: &rng)

            let rt = valueTokenizeUnsigned(run)
            let vt = valueTokenize(val)
            runTokens[i] = rt.token
            valTokens[i] = vt.token
            runTokenCounts[Int(rt.token)] += 1
            valTokenCounts[Int(vt.token)] += 1
        }

        var runModel = rANSModel()
        runModel.normalize(tokenCounts: runTokenCounts)
        var valModel = rANSModel()
        valModel.normalize(tokenCounts: valTokenCounts)

        // ---- エンコード ----
        // Interleaved4rANS: 4レーンに均等分割してエンコード
        let chunkBase = pairCount / 4
        let chunkRemainder = pairCount % 4
        var chunkStarts = [Int](repeating: 0, count: 5)
        for i in 0..<4 {
            chunkStarts[i + 1] = chunkStarts[i] + chunkBase + (i < chunkRemainder ? 1 : 0)
        }

        var enc = Interleaved4rANSEncoder()

        // Backward encode: lane 3→0, each lane backward
        for lane in stride(from: 3, through: 0, by: -1) {
            let start = chunkStarts[lane]
            let end = chunkStarts[lane + 1]
            for i in stride(from: end - 1, through: start, by: -1) {
                // val first (will be decoded second), then run (decoded first)
                let vt = valTokens[i]
                enc.encodeSymbol(lane: lane, cumFreq: valModel.tokenCumFreqs[Int(vt)], freq: valModel.tokenFreqs[Int(vt)])
                let rt = runTokens[i]
                enc.encodeSymbol(lane: lane, cumFreq: runModel.tokenCumFreqs[Int(rt)], freq: runModel.tokenFreqs[Int(rt)])
            }
        }
        enc.flush()
        let bitstream = enc.getBitstream()

        // ---- デコード ----
        bitstream.withUnsafeBufferPointer { buf in
            var dec = Interleaved4rANSDecoder(base: buf.baseAddress!, count: buf.count)

            for lane in 0..<4 {
                let start = chunkStarts[lane]
                let end = chunkStarts[lane + 1]
                for i in start..<end {
                    // Decode run token
                    let cfRun = dec.getCumulativeFreq(lane: lane)
                    let rtInfo = runModel.findToken(cf: cfRun)
                    dec.advanceSymbol(lane: lane, cumFreq: rtInfo.cumFreq, freq: rtInfo.freq)
                    XCTAssertEqual(runTokens[i], rtInfo.token, "Run token mismatch at pair \(i) lane \(lane)")

                    // Decode val token
                    let cfVal = dec.getCumulativeFreq(lane: lane)
                    let vtInfo = valModel.findToken(cf: cfVal)
                    dec.advanceSymbol(lane: lane, cumFreq: vtInfo.cumFreq, freq: vtInfo.freq)
                    XCTAssertEqual(valTokens[i], vtInfo.token, "Val token mismatch at pair \(i) lane \(lane)")
                }
            }
        }
    }
}
