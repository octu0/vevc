import XCTest
@testable import vevc

final class rANSContextTests: XCTestCase {
    func testRANSContextCodecRoundtrip() throws {
        let ws = rANSContextWorkspace()
        let blockCount = 100
        var blocks: [[Int16]] = []
        var rand = 1234567

        for _ in 0..<blockCount {
            var b = [Int16](repeating: 0, count: 16)
            for pos in 0..<16 {
                rand = (rand * 1103515245 + 12345) & 0x7FFFFFFF
                let r = (rand % 31) - 15
                b[pos] = Int16(r)
            }
            blocks.append(b)
        }

        // 1. Encode
        ws.resetPlaneEncoder()

        // Forward escapes
        for bIdx in 0..<blockCount {
            let b = blocks[bIdx]
            for pos in 4..<16 {
                let val = b[pos]
                if val < -64 {
                    ws.planeEncodeEscape(val: val)
                } else {
                    if 64 < val {
                        ws.planeEncodeEscape(val: val)
                    }
                }
            }
        }

        // Backward rANS symbols (workspace buffers and flat weights borrowed
        // once outside the block loop, mirroring EncodeTransform)
        let flat = ws.flatWeights
        withUnsafePointers(mut: &ws.feat, mut: &ws.hidden, mut: &ws.rawCum, mut: &ws.freqs, mut: &ws.cumFreqs, flat.w1All, flat.b1All, flat.w2All) { featP, hiddenP, rawP, freqP, cumP, w1P, b1P, w2P in
            var rBlock = blockCount - 1
            while 0 <= rBlock {
                let b = blocks[rBlock]
                var rPos = 15
                while 4 <= rPos {
                    let val = b[rPos]
                    let (mu, invScale) = b.withUnsafeBufferPointer { bufPtr in
                        ws.predict(
                            pos: rPos,
                            blockCoeffs: bufPtr.baseAddress!,
                            topCoeffs: nil,
                            leftCoeffs: nil,
                            tempCoeffs: nil,
                            isPFrame: false,
                            plane: 0,
                            qstep: 4096,
                            featP: featP,
                            hiddenP: hiddenP,
                            w1AllP: w1P,
                            b1AllP: b1P,
                            w2AllP: w2P
                        )
                    }
                    ws.buildCDF(muQ12: mu, invScaleQ12: invScale, rawP: rawP, freqP: freqP, cumP: cumP)
                    let sym: Int
                    if val < -64 {
                        sym = 129
                    } else {
                        if 64 < val {
                            sym = 129
                        } else {
                            sym = Int(val + 64)
                        }
                    }
                    let freq = freqP[sym]
                    let cumFreq = cumP[sym]
                    ws.planeEncodeSymbol(sym: sym, freq: freq, cumFreq: cumFreq)
                    rPos -= 1
                }
                rBlock -= 1
            }
        }

        let bytes = ws.finalizePlaneEncoder()
        XCTAssertGreaterThan(bytes.count, 8)

        // 2. Decode
        var decodedBlocks: [[Int16]] = []
        try bytes.withUnsafeBufferPointer { bufPtr in
            try ws.initPlaneDecoder(inputPtr: bufPtr.baseAddress!, totalBytes: bytes.count)
            
            withUnsafePointers(mut: &ws.feat, mut: &ws.hidden, mut: &ws.rawCum, mut: &ws.freqs, mut: &ws.cumFreqs, ws.flatWeights.w1All, ws.flatWeights.b1All, ws.flatWeights.w2All) { featP, hiddenP, rawP, freqP, cumP, w1P, b1P, w2P in
                for bIdx in 0..<blockCount {
                    var decB = [Int16](repeating: 0, count: 16)
                    for pos in 0..<4 {
                        decB[pos] = blocks[bIdx][pos]
                    }
                    for pos in 4..<16 {
                        let (mu, invScale) = decB.withUnsafeBufferPointer { bufPtr in
                            ws.predict(
                                pos: pos,
                                blockCoeffs: bufPtr.baseAddress!,
                                topCoeffs: nil,
                                leftCoeffs: nil,
                                tempCoeffs: nil,
                                isPFrame: false,
                                plane: 0,
                                qstep: 4096,
                                featP: featP,
                                hiddenP: hiddenP,
                                w1AllP: w1P,
                                b1AllP: b1P,
                                w2AllP: w2P
                            )
                        }
                        ws.buildCDF(muQ12: mu, invScaleQ12: invScale, rawP: rawP, freqP: freqP, cumP: cumP)
                        let sym = ws.planeDecodeSymbol(freqP: UnsafePointer(freqP), cumP: UnsafePointer(cumP))
                        if sym == 129 {
                            decB[pos] = ws.planeDecodeEscape()
                        } else {
                            decB[pos] = Int16(sym - 64)
                        }
                    }
                    decodedBlocks.append(decB)
                }
            }
        }

        // 3. Verify
        for bIdx in 0..<blockCount {
            for pos in 0..<16 {
                XCTAssertEqual(blocks[bIdx][pos], decodedBlocks[bIdx][pos], "Mismatch at block \(bIdx), pos \(pos)")
            }
        }
    }

    func testRANSContextCodecEscapeRoundtrip() throws {
        let ws = rANSContextWorkspace()
        let blockCount = 100
        var blocks: [[Int16]] = []
        var rand = 987654321

        for _ in 0..<blockCount {
            var b = [Int16](repeating: 0, count: 16)
            for pos in 0..<16 {
                rand = (rand * 1103515245 + 12345) & 0x7FFFFFFF
                // 範囲を -250 ... +250 に広げて escape 経路 (|coeff| > 64) を確実に通す
                let r = (rand % 501) - 250
                b[pos] = Int16(r)
            }
            blocks.append(b)
        }

        // 1. Encode
        ws.resetPlaneEncoder()

        // Forward escapes
        var escapeCount = 0
        for bIdx in 0..<blockCount {
            let b = blocks[bIdx]
            for pos in 4..<16 {
                let val = b[pos]
                if val < -64 {
                    ws.planeEncodeEscape(val: val)
                    escapeCount += 1
                } else {
                    if 64 < val {
                        ws.planeEncodeEscape(val: val)
                        escapeCount += 1
                    }
                }
            }
        }
        XCTAssertLessThan(0, escapeCount, "Escape test must encode at least one escape symbol")

        // Backward rANS symbols (workspace buffers and flat weights borrowed
        // once outside the block loop, mirroring EncodeTransform)
        let flat = ws.flatWeights
        withUnsafePointers(mut: &ws.feat, mut: &ws.hidden, mut: &ws.rawCum, mut: &ws.freqs, mut: &ws.cumFreqs, flat.w1All, flat.b1All, flat.w2All) { featP, hiddenP, rawP, freqP, cumP, w1P, b1P, w2P in
            var rBlock = blockCount - 1
            while 0 <= rBlock {
                let b = blocks[rBlock]
                var rPos = 15
                while 4 <= rPos {
                    let val = b[rPos]
                    let (mu, invScale) = b.withUnsafeBufferPointer { bufPtr in
                        ws.predict(
                            pos: rPos,
                            blockCoeffs: bufPtr.baseAddress!,
                            topCoeffs: nil,
                            leftCoeffs: nil,
                            tempCoeffs: nil,
                            isPFrame: false,
                            plane: 0,
                            qstep: 4096,
                            featP: featP,
                            hiddenP: hiddenP,
                            w1AllP: w1P,
                            b1AllP: b1P,
                            w2AllP: w2P
                        )
                    }
                    ws.buildCDF(muQ12: mu, invScaleQ12: invScale, rawP: rawP, freqP: freqP, cumP: cumP)
                    let sym: Int
                    if val < -64 {
                        sym = 129
                    } else {
                        if 64 < val {
                            sym = 129
                        } else {
                            sym = Int(val + 64)
                        }
                    }
                    let freq = freqP[sym]
                    let cumFreq = cumP[sym]
                    ws.planeEncodeSymbol(sym: sym, freq: freq, cumFreq: cumFreq)
                    rPos -= 1
                }
                rBlock -= 1
            }
        }

        let bytes = ws.finalizePlaneEncoder()
        XCTAssertLessThan(8, bytes.count)

        // 2. Decode
        var decodedBlocks: [[Int16]] = []
        try bytes.withUnsafeBufferPointer { bufPtr in
            try ws.initPlaneDecoder(inputPtr: bufPtr.baseAddress!, totalBytes: bytes.count)
            
            withUnsafePointers(mut: &ws.feat, mut: &ws.hidden, mut: &ws.rawCum, mut: &ws.freqs, mut: &ws.cumFreqs, ws.flatWeights.w1All, ws.flatWeights.b1All, ws.flatWeights.w2All) { featP, hiddenP, rawP, freqP, cumP, w1P, b1P, w2P in
                for bIdx in 0..<blockCount {
                    var decB = [Int16](repeating: 0, count: 16)
                    for pos in 0..<4 {
                        decB[pos] = blocks[bIdx][pos]
                    }
                    for pos in 4..<16 {
                        let (mu, invScale) = decB.withUnsafeBufferPointer { bufPtr in
                            ws.predict(
                                pos: pos,
                                blockCoeffs: bufPtr.baseAddress!,
                                topCoeffs: nil,
                                leftCoeffs: nil,
                                tempCoeffs: nil,
                                isPFrame: false,
                                plane: 0,
                                qstep: 4096,
                                featP: featP,
                                hiddenP: hiddenP,
                                w1AllP: w1P,
                                b1AllP: b1P,
                                w2AllP: w2P
                            )
                        }
                        ws.buildCDF(muQ12: mu, invScaleQ12: invScale, rawP: rawP, freqP: freqP, cumP: cumP)
                        let sym = ws.planeDecodeSymbol(freqP: UnsafePointer(freqP), cumP: UnsafePointer(cumP))
                        if sym == 129 {
                            decB[pos] = ws.planeDecodeEscape()
                        } else {
                            decB[pos] = Int16(sym - 64)
                        }
                    }
                    decodedBlocks.append(decB)
                }
            }
        }

        // 3. Verify
        for bIdx in 0..<blockCount {
            for pos in 0..<16 {
                XCTAssertEqual(blocks[bIdx][pos], decodedBlocks[bIdx][pos], "Mismatch at block \(bIdx), pos \(pos)")
            }
        }
    }
}
