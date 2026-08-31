import XCTest
@testable import vevc

/// Round-trip tests for the #36 context-conditioned syntax coders. The
/// encoder and decoder share one adaptive-model object per GOP, so the tests
/// exercise multi-frame sequences with a shared state, not just single frames.
final class SyntaxContextCodecTests: XCTestCase {

    private func randomSkipMap(count: Int, seed: UInt64, interBias: Int) -> [BlockMode] {
        var s = seed | 1
        var out = [BlockMode]()
        out.reserveCapacity(count)
        for _ in 0..<count {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17
            let r = Int(s % 100)
            if r < interBias { out.append(.inter) }
            else if r < interBias + (100 - interBias) / 2 { out.append(.skip_prev) }
            else { out.append(.skip_ltr) }
        }
        return out
    }

    func testSkipMapContextRoundTripRandom() throws {
        let cols = 60
        let count = cols * 34
        for bias in [5, 50, 95] {
            let encState = SyntaxContextModels()
            let decState = SyntaxContextModels()
            // Three consecutive frames on one state: frame 0 has no temporal
            // context, frames 1 and 2 do.
            for f in 0..<3 {
                let map = randomSkipMap(count: count, seed: UInt64(1000 + f * 7 + bias), interBias: bias)
                let data = encodeSkipMapContext(map: map, cols: cols, state: encState)
                XCTAssertEqual(data.first, skipMapModeContext, "mode byte must tag the ctx form")
                let back = try decodeSkipMapContext(data: data, count: count, cols: cols, state: decState)
                XCTAssertEqual(back, map, "skipMap round-trip failed at bias=\(bias) frame=\(f)")
            }
        }
    }

    func testSkipMapContextUniformAndSingleSymbol() throws {
        let cols = 8
        let count = 64
        for mode in [BlockMode.inter, .skip_prev, .skip_ltr] {
            let encState = SyntaxContextModels()
            let decState = SyntaxContextModels()
            let map = [BlockMode](repeating: mode, count: count)
            let data = encodeSkipMapContext(map: map, cols: cols, state: encState)
            let back = try decodeSkipMapContext(data: data, count: count, cols: cols, state: decState)
            XCTAssertEqual(back, map)
        }
        // Single block.
        let encState = SyntaxContextModels()
        let decState = SyntaxContextModels()
        let one: [BlockMode] = [.skip_ltr]
        let d = encodeSkipMapContext(map: one, cols: 1, state: encState)
        XCTAssertEqual(try decodeSkipMapContext(data: d, count: 1, cols: 1, state: decState), one)
    }

    func testSkipMapContextRejectsLegacyTag() {
        let legacy: [UInt8] = [0, 1, 2, 3, 4, 5]
        let st = SyntaxContextModels()
        XCTAssertThrowsError(try decodeSkipMapContext(data: legacy, count: 4, cols: 2, state: st))
    }

    func testRefDirContextRoundTrip() throws {
        let cols = 60
        let count = cols * 34
        for onePct in [0, 1, 25, 100] {
            var s: UInt64 = UInt64(onePct) | 1
            let map = randomSkipMap(count: count, seed: 42, interBias: 40)
            var refDirs = [Bool](repeating: false, count: count)
            for i in 0..<count {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17
                refDirs[i] = Int(s % 100) < onePct
            }
            let data = encodeRefDirsContextProfile2(refDirs: refDirs, skipMap: map)
            let back = decodeRefDirsContextProfile2(buf: data, count: count, skipMap: map)
            // Only inter positions are coded; skip positions are expanded from
            // the mode (skip_ltr -> true, skip_prev -> false), matching the
            // legacy decoder's behaviour.
            for i in 0..<count {
                switch map[i] {
                case .inter:
                    XCTAssertEqual(back[i], refDirs[i], "refDir mismatch at \(i) onePct=\(onePct)")
                case .skip_ltr:
                    XCTAssertTrue(back[i], "skip_ltr must expand to true")
                case .skip_prev:
                    XCTAssertFalse(back[i], "skip_prev must expand to false")
                }
            }
        }
    }

    func testRefDirContextEmptyWhenNoInter() {
        let map = [BlockMode](repeating: .skip_prev, count: 32)
        let refDirs = [Bool](repeating: true, count: 32)
        let data = encodeRefDirsContextProfile2(refDirs: refDirs, skipMap: map)
        XCTAssertTrue(data.isEmpty, "no inter blocks means no refDir payload")
        let back = decodeRefDirsContextProfile2(buf: data, count: 32, skipMap: map)
        // all skip_prev, so every position expands to false
        XCTAssertEqual(back, [Bool](repeating: false, count: 32))
    }

    func testTreeMapContextRoundTrip() throws {
        let colsY = 60, rowsY = 34
        let colsC = 30, rowsC = 17
        let nY = colsY * rowsY
        let nC = colsC * rowsC
        for onePct in [10, 57, 85, 100] {
            var s: UInt64 = UInt64(onePct * 13) | 1
            func gen(_ n: Int) -> ([Bool], [Bool]) {
                var tz = [Bool](repeating: false, count: n)
                var sk = [Bool](repeating: false, count: n)
                for i in 0..<n {
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17
                    sk[i] = Int(s % 100) < 30
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17
                    tz[i] = Int(s % 100) < onePct
                }
                return (tz, sk)
            }
            let (tzY, skY) = gen(nY)
            let (tzCb, skC) = gen(nC)
            let (tzCr, _) = gen(nC)

            let data = encodeTreeMapContextProfile2(
                isTreezY: tzY, ySkip: skY, colsY: colsY,
                isTreezCb: tzCb, cbSkip: skC,
                isTreezCr: tzCr, crSkip: skC, colsC: colsC
            )
            let back = try decodeTreeMapContextProfile2(
                buf: data, ySkip: skY, colsY: colsY,
                cbSkip: skC, crSkip: skC, colsC: colsC
            )
            for i in 0..<nY where skY[i] != true {
                XCTAssertEqual(back.isTreezY[i], tzY[i], "Y treeMap mismatch at \(i) onePct=\(onePct)")
            }
            for i in 0..<nC where skC[i] != true {
                XCTAssertEqual(back.isTreezCb[i], tzCb[i], "Cb treeMap mismatch at \(i)")
                XCTAssertEqual(back.isTreezCr[i], tzCr[i], "Cr treeMap mismatch at \(i)")
            }
        }
    }

    /// The adaptive models must keep their frequencies summing to exactly
    /// rANSScale, which is what the rANS coder assumes.
    func testAdaptiveModelsPreserveTotal() {
        var b = AdaptiveBinModel(oneIn: 16)
        for i in 0..<10_000 {
            b.update(i % 7 == 0 ? 1 : 0)
            let iv0 = b.interval(0), iv1 = b.interval(1)
            XCTAssertEqual(iv0.freq + iv1.freq, rANSScale)
            XCTAssertGreaterThan(iv0.freq, 0)
            XCTAssertGreaterThan(iv1.freq, 0)
        }
        var t = AdaptiveTriModel()
        for i in 0..<10_000 {
            t.update(i % 5 == 0 ? 2 : (i % 3 == 0 ? 1 : 0))
            let s = t.freq(0) + t.freq(1) + t.freq(2)
            XCTAssertEqual(s, rANSScale)
            XCTAssertGreaterThan(t.freq(0), 0)
            XCTAssertGreaterThan(t.freq(1), 0)
            XCTAssertGreaterThan(t.freq(2), 0)
        }
    }

    /// Real-data-shaped sequence: long runs of skip_ltr with sparse inter
    /// islands, which is what the production skip decider actually emits.
    func testSkipMapContextRealisticRuns() throws {
        let cols = 60
        let rows = 34
        let count = cols * rows
        let encState = SyntaxContextModels()
        let decState = SyntaxContextModels()
        for f in 0..<5 {
            var map = [BlockMode](repeating: .skip_ltr, count: count)
            // a moving band of inter blocks
            let band = (f * 3) % rows
            for c in 0..<cols {
                map[band * cols + c] = .inter
                if band + 1 < rows { map[(band + 1) * cols + c] = .skip_prev }
            }
            let data = encodeSkipMapContext(map: map, cols: cols, state: encState)
            let back = try decodeSkipMapContext(data: data, count: count, cols: cols, state: decState)
            XCTAssertEqual(back, map, "realistic skipMap round-trip failed at frame \(f)")
        }
    }
}
