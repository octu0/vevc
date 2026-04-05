import XCTest
@testable import vevc

/// ValueTokenizer のラウンドトリップテスト
final class ValueTokenizerTests: XCTestCase {
    
    func testSignedRoundtrip() {
        let testValues: [Int16] = [
            0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8,
            9, -9, 10, -10, 15, -15, 16, -16, 20, -20, 50, -50, 100, -100,
            127, -128, 255, -256, 500, -500, 1000, -1000,
            Int16.max, Int16.min, Int16.max - 1, Int16.min + 1
        ]
        
        for val in testValues {
            let result = valueTokenize(val)
            let bypassLen = valueBypassLength(for: result.token)
            XCTAssertEqual(bypassLen, result.bypassLen, "bypassLength mismatch for val=\(val)")
            let restored = valueDetokenize(token: result.token, bypassBits: result.bypassBits)
            XCTAssertEqual(val, restored, "Signed roundtrip failed for val=\(val)")
        }
    }
    
    func testUnsignedRoundtrip() {
        let testValues: [UInt32] = [0, 1, 2, 3, 4, 5, 10, 15, 16, 17, 20, 30, 50, 100, 200, 500, 1000]
        
        for val in testValues {
            let result = valueTokenizeUnsigned(val)
            let bypassLen = valueBypassLengthUnsigned(for: result.token)
            XCTAssertEqual(bypassLen, result.bypassLen, "bypassLength mismatch for val=\(val)")
            let restored = valueDetokenizeUnsigned(token: result.token, bypassBits: result.bypassBits)
            XCTAssertEqual(val, restored, "Unsigned roundtrip failed for val=\(val)")
        }
    }
    
    /// 実際のDWTデータでEntropyEncoder→EntropyDecoder pairs roundtrip
    func testRansModeWithActualBlockData() async throws {
        let pool = BlockViewPool()
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
        
        var (blocks, _) = await extractSingleTransformBlocks32(r: pd.rY, width: width, height: height, pool: pool)
        for i in blocks.indices {
            evaluateQuantizeLayer32(block: &blocks[i], qt: qtY)
        }
        
        let safeThreshold = max(0, 3 - (Int(qtY.step) / 2))
        
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        
        for i in blocks.indices {
            let isZero = isEffectivelyZero32(data: blocks[i].base, threshold: safeThreshold)
            if isZero { continue }
            
            let view = blocks[i]
            let subs = getSubbands32(view: view)
            blockEncode16(encoder: &encoder, block: subs.hl, parentBlock: nil)
            blockEncode16(encoder: &encoder, block: subs.lh, parentBlock: nil)
            blockEncode16(encoder: &encoder, block: subs.hh, parentBlock: nil)
                }
        
        print("=== Encoder: pairs=\(encoder.pairs.count) coeffCount=\(encoder.coeffCount) trailingZeros=\(encoder.trailingZeros) ===")
        
        let encPairs = encoder.pairs
        let data = encoder.getData()
        
        var decoder = try EntropyDecoder(data: data)
        var decPairs: [(run: Int, val: Int16)] = []
        for i in 0..<encPairs.count {
            let pair = decoder.readPair(isParentZero: encPairs[i].isParentZero)
            decPairs.append(pair)
        }
        
        print("=== Decoder: pairs=\(decPairs.count) ===")
        
        XCTAssertEqual(encPairs.count, decPairs.count, "pairs count")
        
        var firstDiff = -1
        var diffCount = 0
        for i in 0..<min(encPairs.count, decPairs.count) {
            if encPairs[i].run != decPairs[i].run || encPairs[i].val != decPairs[i].val {
                if firstDiff < 0 {
                    firstDiff = i
                    print("DIFF[\(i)]: enc=(\(encPairs[i].run), \(encPairs[i].val)) dec=(\(decPairs[i].run), \(decPairs[i].val))")
                }
                diffCount += 1
            }
        }
        
        XCTAssertEqual(diffCount, 0, "Pairs diff: \(diffCount) total, first at \(firstDiff)")
    }
}
