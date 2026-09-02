import XCTest
@testable import vevc

/// Backward-adaptive (history) entropy tables: encoder and decoder maintain
/// identical decayed token counts and the encoder may signal flags bit 0x20
/// to reuse them instead of transmitting frequency tables.
final class BackwardAdaptiveTests: XCTestCase {

    /// A synthetic pair stream with a stable, skewed distribution across
    /// contexts 0..3 plus LSCP-style pairs in context 5.
    private func makePairs(seed: Int, count: Int) -> [(run: UInt32, val: Int16, ctx: UInt8)] {
        var pairs = [(run: UInt32, val: Int16, ctx: UInt8)]()
        pairs.reserveCapacity(count)
        var state = UInt64(seed) &* 6364136223846793005 &+ 1442695040888963407
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let r = Int((state >> 33) % 5)          // short zero-runs
            let v = Int16(1 + Int((state >> 40) % 6))  // small values
            let sign: Int16 = ((state >> 50) & 1) == 0 ? 1 : -1
            let ctx: UInt8 = i % 16 == 0 ? 5 : UInt8((state >> 20) % 4)
            let val: Int16 = ctx == 5 ? 0 : v * sign
            pairs.append((run: UInt32(r), val: val, ctx: ctx))
        }
        return pairs
    }

    private func encodeFrame(pairs: [(run: UInt32, val: Int16, ctx: UInt8)], history: EntropyHistoryState?) -> [UInt8] {
        var enc = EntropyEncoder()
        for p in pairs {
            enc.addPair(run: p.run, val: p.val, context: p.ctx)
        }
        enc.flush()
        return enc.getData(selectModel: unifiedSelectModel, history: history, updateHistory: true)
    }

    private func decodeFrame(data: [UInt8], pairs: [(run: UInt32, val: Int16, ctx: UInt8)], history: EntropyHistoryState?) throws -> [(run: Int, val: Int16)] {
        try data.withUnsafeBufferPointer { buf in
            var dec = try EntropyDecoder(base: buf.baseAddress!, count: buf.count, startOffset: 0, history: history, parentFreeStatics: false, updateHistory: true)
            var out = [(run: Int, val: Int16)]()
            out.reserveCapacity(pairs.count)
            for p in pairs {
                let (run, val) = dec.readPair(context: p.ctx)
                out.append((run, val))
            }
            dec.finalizeHistory()
            return out
        }
    }

    private func flagsByte(of data: [UInt8]) throws -> UInt8 {
        var offset = 0
        let bypassLen = try readVLQSizeFromBytes(data, offset: &offset)
        offset += bypassLen
        _ = try readVLQSizeFromBytes(data, offset: &offset)  // coeffCount
        return data[offset]
    }

    func testHistoryModeRoundtripAcrossFrames() throws {
        let encHistory = EntropyHistoryState()
        let decHistory = EntropyHistoryState()
        var sawHistoryFlag = false

        for frame in 0..<6 {
            let pairs = makePairs(seed: 1000 + frame, count: 400)
            let data = encodeFrame(pairs: pairs, history: encHistory)
            let flags = try flagsByte(of: data)
            if (flags & 0x20) != 0 { sawHistoryFlag = true }

            let decoded = try decodeFrame(data: data, pairs: pairs, history: decHistory)
            for (i, p) in pairs.enumerated() {
                XCTAssertEqual(decoded[i].run, Int(p.run), "frame \(frame) pair \(i) run")
                XCTAssertEqual(decoded[i].val, p.val, "frame \(frame) pair \(i) val")
            }
        }
        XCTAssertTrue(sawHistoryFlag, "stable distribution should trigger history mode within a few frames")
    }

    func testEncoderAndDecoderHistoriesStayIdentical() throws {
        let encHistory = EntropyHistoryState()
        let decHistory = EntropyHistoryState()

        for frame in 0..<4 {
            let pairs = makePairs(seed: 42 + frame, count: 300)
            let data = encodeFrame(pairs: pairs, history: encHistory)
            _ = try decodeFrame(data: data, pairs: pairs, history: decHistory)
        }
        XCTAssertEqual(encHistory.primed, decHistory.primed)
        for c in 0..<entropyContextCount {
            XCTAssertEqual(encHistory.runCounts[c], decHistory.runCounts[c], "run counts ctx \(c)")
            XCTAssertEqual(encHistory.valCounts[c], decHistory.valCounts[c], "val counts ctx \(c)")
        }
    }

    func testEncodeIsDeterministic() throws {
        func run() -> [[UInt8]] {
            let h = EntropyHistoryState()
            return (0..<4).map { frame in
                encodeFrame(pairs: makePairs(seed: 7 + frame, count: 350), history: h)
            }
        }
        let a = run()
        let b = run()
        XCTAssertEqual(a, b)
    }

    func testFirstFrameAndResetNeverUseHistory() throws {
        let h = EntropyHistoryState()
        let pairs = makePairs(seed: 5, count: 400)

        let first = encodeFrame(pairs: pairs, history: h)
        XCTAssertEqual(try flagsByte(of: first) & 0x20, 0, "unprimed history must not be used")

        _ = encodeFrame(pairs: pairs, history: h)
        h.reset()
        let afterReset = encodeFrame(pairs: pairs, history: h)
        XCTAssertEqual(try flagsByte(of: afterReset) & 0x20, 0, "reset history must not be used")
    }

    func testRawModeStreamsDoNotTouchHistory() throws {
        let h = EntropyHistoryState()
        // <= 32 pairs → raw-bypass mode: no history use, no history update
        let small = makePairs(seed: 9, count: 10)
        let data = encodeFrame(pairs: small, history: h)
        XCTAssertNotEqual(data.count, 0)
        XCTAssertFalse(h.primed, "raw-mode stream must not prime history")

        let decHistory = EntropyHistoryState()
        let decoded = try decodeFrame(data: data, pairs: small, history: decHistory)
        for (i, p) in small.enumerated() {
            XCTAssertEqual(decoded[i].run, Int(p.run))
            XCTAssertEqual(decoded[i].val, p.val)
        }
        XCTAssertFalse(decHistory.primed, "raw-mode stream must not prime decoder history")
    }

    func testHistoryModeShrinksStableStreams() throws {
        let h = EntropyHistoryState()
        var sizes = [Int]()
        for frame in 0..<6 {
            let pairs = makePairs(seed: 11 + frame, count: 500)
            sizes.append(encodeFrame(pairs: pairs, history: h).count)
        }
        // Once history kicks in, frames must not be larger than the first
        // (header-bearing) frame of the same distribution family.
        XCTAssertLessThan(sizes.last!, sizes.first!)
    }
}
