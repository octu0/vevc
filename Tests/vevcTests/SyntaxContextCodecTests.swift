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
        var m16 = AdaptiveModel16()
        for i in 0..<10_000 {
            m16.update(i % 16)
            var s: UInt32 = 0
            for sym in 0..<16 {
                let iv = m16.interval(sym)
                s += iv.freq
                XCTAssertGreaterThan(iv.freq, 0)
            }
            XCTAssertEqual(s, rANSScale)
        }
    }

    func testMVClassifyDeclassifyRoundTrip() {
        for v in -2048...2048 {
            let val = Int16(v)
            let (classIndex, sign, offset, bits) = mvClassify(val)
            let reconstructed = mvDeclassify(classIndex: classIndex, sign: sign, offset: offset)
            XCTAssertEqual(reconstructed, val, "MV classification mismatch for value \(val)")
            if 0 < bits {
                XCTAssertLessThan(offset, UInt32(1 << bits))
            }
        }
    }

    func testMVsContextRoundTripRandom() throws {
        let cols = 60
        let rows = 34
        let count = cols * rows
        var s: UInt64 = 42
        let encState = SyntaxContextModels()
        let decState = SyntaxContextModels()

        var prevMVs: MotionVectors? = nil
        for f in 0..<5 {
            var dxs = [Int16](repeating: 0, count: count)
            var dys = [Int16](repeating: 0, count: count)
            var skipMap = [BlockMode](repeating: .inter, count: count)
            for i in 0..<count {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17
                if (s % 10) < 3 {
                    skipMap[i] = .skip_ltr
                } else {
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17
                    let rx = Int16(Int(s % 200) - 100)
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17
                    let ry = Int16(Int(s % 200) - 100)
                    dxs[i] = rx
                    dys[i] = ry
                }
            }
            let mvs = MotionVectors(dx: dxs, dy: dys)
            let data = encodeMVsContextProfile2(mvs: mvs, skipMap: skipMap, cols: cols, prevMVs: prevMVs, state: encState, updateHistory: true)
            let back = try decodeMVsContextProfile2(data: data, count: count, skipMap: skipMap, cols: cols, prevMVs: prevMVs, state: decState, updateHistory: true)

            for i in 0..<count {
                if skipMap[i] == .inter {
                    XCTAssertEqual(back.dx[i], dxs[i], "dx mismatch at f=\(f) i=\(i)")
                    XCTAssertEqual(back.dy[i], dys[i], "dy mismatch at f=\(f) i=\(i)")
                } else {
                    XCTAssertEqual(back.dx[i], 0)
                    XCTAssertEqual(back.dy[i], 0)
                }
            }
            prevMVs = mvs
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
                if (band + 1) < rows { map[(band + 1) * cols + c] = .skip_prev }
            }
            let data = encodeSkipMapContext(map: map, cols: cols, state: encState)
            let back = try decodeSkipMapContext(data: data, count: count, cols: cols, state: decState)
            XCTAssertEqual(back, map, "realistic skipMap round-trip failed at frame \(f)")
        }
    }

    /// Random-access GOP decode equality: decoding from an interior GOP keyframe
    /// must produce bit-identical reconstructions compared to full-sequence decode.
    func testRandomAccessGOPDecodeMatchesFullDecode() async throws {
        let width = 64
        let height = 64
        let gop = 8
        let totalFrames = 20

        let encoder = VEVCEncoder(width: width, height: height, profile: 0x02)
        encoder.maxbitrate = 500
        encoder.framerate = 30
        encoder.zeroThreshold = 0
        encoder.keyint = gop
        encoder.iqFloor = 0
        encoder.sceneChangeThreshold = 100
        encoder.gop = gop
        var encodedChunks = [[UInt8]]()
        var s: UInt64 = 12345
        for _ in 0..<totalFrames {
            var img = YCbCrImage(width: width, height: height)
            for y in 0..<height {
                for x in 0..<width {
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17
                    img.yPlane[y * width + x] = UInt8(s % 256)
                }
            }
            for y in 0..<(height / 2) {
                for x in 0..<(width / 2) {
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17
                    img.cbPlane[y * (width / 2) + x] = UInt8(s % 256)
                    s ^= s << 13; s ^= s >> 7; s ^= s << 17
                    img.crPlane[y * (width / 2) + x] = UInt8(s % 256)
                }
            }
            let chunk = try await encoder.encode(image: img)
            encodedChunks.append(chunk)
        }

        // Full decode of all frames using Decoder (serial)
        let fullDecoder = VEVCDecoder(maxLayer: 2)
        fullDecoder.maxConcurrency = 1
        var fullRecons = [YCbCrImage]()
        for try await img in fullDecoder.decodeStream(stream: AsyncChunks(chunks: encodedChunks)) {
            fullRecons.append(img)
        }
        XCTAssertEqual(fullRecons.count, totalFrames)

        // Random-access decode starting from GOP 1 (frame index 8)
        let gop1Start = gop
        var headerOffset = 0
        _ = try VEVCFileHeader.deserialize(from: encodedChunks[0], offset: &headerOffset)
        let fileHeaderBytes = Array(encodedChunks[0][0..<headerOffset])

        var raChunks = [[UInt8]]()
        // Combine file header with GOP 1 keyframe chunk
        var gop1FirstChunk = fileHeaderBytes
        gop1FirstChunk.append(contentsOf: encodedChunks[gop1Start])
        raChunks.append(gop1FirstChunk)
        for f in (gop1Start + 1)..<totalFrames {
            raChunks.append(encodedChunks[f])
        }

        let raDecoder = VEVCDecoder(maxLayer: 2)
        raDecoder.maxConcurrency = 1
        var raRecons = [YCbCrImage]()
        for try await img in raDecoder.decodeStream(stream: AsyncChunks(chunks: raChunks)) {
            raRecons.append(img)
        }
        XCTAssertEqual(raRecons.count, totalFrames - gop1Start)

        for i in 0..<raRecons.count {
            let fullImg = fullRecons[gop1Start + i]
            let raImg = raRecons[i]
            XCTAssertEqual(fullImg.yPlane, raImg.yPlane, "Y plane mismatch at GOP frame \(i)")
            XCTAssertEqual(fullImg.cbPlane, raImg.cbPlane, "Cb plane mismatch at GOP frame \(i)")
            XCTAssertEqual(fullImg.crPlane, raImg.crPlane, "Cr plane mismatch at GOP frame \(i)")
        }
    }
}

private struct AsyncChunks: AsyncSequence, Sendable {
    typealias Element = [UInt8]
    let chunks: [[UInt8]]

    struct AsyncIterator: AsyncIteratorProtocol {
        var index = 0
        let chunks: [[UInt8]]
        mutating func next() async throws -> [UInt8]? {
            if chunks.count <= index { return nil }
            let item = chunks[index]
            index += 1
            return item
        }
    }

    func makeAsyncIterator() -> AsyncIterator {
        return AsyncIterator(index: 0, chunks: chunks)
    }
}
