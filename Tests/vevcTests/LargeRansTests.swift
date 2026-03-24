import XCTest
@testable import vevc

/// 大量pairs (>32 → rANSモード) の直接テスト  
final class LargeRansTests: XCTestCase {
    
    func testRansRoundtrip_6500Pairs() throws {
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        var expected: [(run: UInt32, val: Int16)] = []
        
        for i in 0..<6500 {
            let run = UInt32(i % 7)
            // 実データに近い値: 小さいval多め、大きいval少なめ
            let raw = (i &* 2654435761) % 100
            let val: Int16
            if raw < 20 { val = 1 }
            else if raw < 35 { val = -1 }
            else if raw < 45 { val = 2 }
            else if raw < 55 { val = -2 }
            else if raw < 62 { val = 3 }
            else if raw < 69 { val = -3 }
            else if raw < 75 { val = 4 }
            else if raw < 80 { val = -4 }
            else if raw < 85 { val = 5 }
            else if raw < 89 { val = -5 }
            else if raw < 93 { val = Int16(raw - 85) }
            else { val = Int16(-(raw - 85)) }
            
            encoder.addPair(run: run, val: val, isParentZero: false)
            expected.append((run: run, val: val))
        }
        
        print("=== Encoder: pairs=\(encoder.pairs.count) coeffCount=\(encoder.coeffCount) trailingZeros=\(encoder.trailingZeros) ===")
        
        let data = encoder.getData()
        var decoder = try EntropyDecoder(data: data)
        var decPairs: [(run: Int, val: Int16)] = []
        for i in 0..<encoder.pairs.count {
            let pair = decoder.readPair(isParentZero: encoder.pairs[i].isParentZero)
            decPairs.append(pair)
        }
        
        print("=== Decoder: pairs=\(decPairs.count) ===")
        
        XCTAssertEqual(expected.count, decPairs.count, "pairs count")
        
        var firstDiff = -1
        var diffCount = 0
        for i in 0..<min(expected.count, decPairs.count) {
            if expected[i].run != decPairs[i].run || expected[i].val != decPairs[i].val {
                if firstDiff < 0 {
                    firstDiff = i
                    print("DIFF[\(i)]: enc=(\(expected[i].run), \(expected[i].val)) dec=(\(decPairs[i].run), \(decPairs[i].val))")
                }
                diffCount += 1
            }
        }
        
        XCTAssertEqual(diffCount, 0, "Pairs diff: \(diffCount) total, first at \(firstDiff)")
    }
    
    /// bypass + pairs 混合で大量データ
    func testRansWithBypass_6500() throws {
        var encoder = EntropyEncoder<DynamicEntropyModel>()
        
        // bypassビット（blockEncode16のhasNonZero + lscp相当）
        for block in 0..<48 {
            encoder.encodeBypass(binVal: 1)
            encodeExpGolomb(val: UInt32(block % 16), encoder: &encoder)
            encodeExpGolomb(val: UInt32(block % 16), encoder: &encoder)
        }
        
        // 大量pairs
        var expected: [(run: UInt32, val: Int16)] = []
        for i in 0..<6500 {
            let run = UInt32(i % 5)
            let val = Int16(clamping: ((i &* 7 + 3) % 30) - 15)
            if val == 0 { continue }
            encoder.addPair(run: run, val: val, isParentZero: false)
            expected.append((run: run, val: val))
        }
        
        let data = encoder.getData()
        var decoder = try EntropyDecoder(data: data)
        
        // bypass読み飛ばし
        for _ in 0..<48 {
            let _ = try decoder.decodeBypass()
            let _ = try decodeExpGolomb(decoder: &decoder)
            let _ = try decodeExpGolomb(decoder: &decoder)
        }
        
        // pairs比較
        var decPairs: [(run: Int, val: Int16)] = []
        for i in 0..<encoder.pairs.count {
            let pair = decoder.readPair(isParentZero: encoder.pairs[i].isParentZero)
            decPairs.append(pair)
        }
        XCTAssertEqual(expected.count, decPairs.count, "pairs count")
        
        var diffCount = 0
        var firstDiff = -1
        for i in 0..<min(expected.count, decPairs.count) {
            if expected[i].run != decPairs[i].run || expected[i].val != decPairs[i].val {
                if firstDiff < 0 {
                    firstDiff = i
                    print("DIFF[\(i)]: enc=(\(expected[i].run), \(expected[i].val)) dec=(\(decPairs[i].run), \(decPairs[i].val))")
                }
                diffCount += 1
            }
        }
        
        XCTAssertEqual(diffCount, 0, "Pairs diff: \(diffCount) total, first at \(firstDiff)")
    }
}
