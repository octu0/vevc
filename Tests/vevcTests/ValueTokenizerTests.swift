//
//  ValueTokenizerTests.swift
//  vevcTests
//

import XCTest
@testable import vevc

final class ValueTokenizerTests: XCTestCase {
    
    func testTokenizeAndDetokenize() {
        // Zero
        let z = ValueTokenizer.tokenize(0)
        XCTAssertFalse(z.isSignificant)
        XCTAssertEqual(ValueTokenizer.detokenize(isSignificant: z.isSignificant, sign: z.sign, token: z.token, bypassBits: z.bypassBits), 0)
        
        // 1 and -1 (Token 0)
        let pos1 = ValueTokenizer.tokenize(1)
        XCTAssertTrue(pos1.isSignificant)
        XCTAssertFalse(pos1.sign)
        XCTAssertEqual(pos1.token, 0)
        XCTAssertEqual(pos1.bypassBits, 0)
        XCTAssertEqual(ValueTokenizer.detokenize(isSignificant: pos1.isSignificant, sign: pos1.sign, token: pos1.token, bypassBits: pos1.bypassBits), 1)
        
        let neg1 = ValueTokenizer.tokenize(-1)
        XCTAssertTrue(neg1.isSignificant)
        XCTAssertTrue(neg1.sign)
        XCTAssertEqual(neg1.token, 0)
        XCTAssertEqual(neg1.bypassBits, 0)
        XCTAssertEqual(ValueTokenizer.detokenize(isSignificant: neg1.isSignificant, sign: neg1.sign, token: neg1.token, bypassBits: neg1.bypassBits), -1)
        
        // 2 and 3 (Token 1)
        for v: Int16 in 2...3 {
            let t = ValueTokenizer.tokenize(v)
            XCTAssertEqual(t.token, 1)
            XCTAssertEqual(ValueTokenizer.bypassLength(for: t.token), 1)
            XCTAssertEqual(ValueTokenizer.detokenize(isSignificant: t.isSignificant, sign: t.sign, token: t.token, bypassBits: t.bypassBits), v)
        }
        
        for v: Int16 in -3...(-2) {
            let t = ValueTokenizer.tokenize(v)
            XCTAssertEqual(t.token, 1)
            XCTAssertEqual(ValueTokenizer.detokenize(isSignificant: t.isSignificant, sign: t.sign, token: t.token, bypassBits: t.bypassBits), v)
        }
        
        // 4 ... 7 (Token 2)
        for v: Int16 in 4...7 {
            let t = ValueTokenizer.tokenize(v)
            XCTAssertEqual(t.token, 2)
            XCTAssertEqual(ValueTokenizer.bypassLength(for: t.token), 2)
            XCTAssertEqual(ValueTokenizer.detokenize(isSignificant: t.isSignificant, sign: t.sign, token: t.token, bypassBits: t.bypassBits), v)
        }
        
        // 8 ... 15 (Token 3)
        for v: Int16 in 8...15 {
            let t = ValueTokenizer.tokenize(v)
            XCTAssertEqual(t.token, 3)
            XCTAssertEqual(ValueTokenizer.bypassLength(for: t.token), 3)
            XCTAssertEqual(ValueTokenizer.detokenize(isSignificant: t.isSignificant, sign: t.sign, token: t.token, bypassBits: t.bypassBits), v)
        }
    }
    
    func testLosslessWithRandomOutliers() {
        // ランダムな外れ値を含む配列での可逆変換テスト (Lossless)
        var rng = SystemRandomNumberGenerator()
        
        for _ in 0..<10000 {
            // Int16の全範囲でのランダム生成
            let original = Int16.random(in: Int16.min...Int16.max, using: &rng)
            
            let t = ValueTokenizer.tokenize(original)
            let restored = ValueTokenizer.detokenize(isSignificant: t.isSignificant, sign: t.sign, token: t.token, bypassBits: t.bypassBits)
            
            XCTAssertEqual(original, restored, "Failed to losslessly transform value: \(original)")
        }
        
        // 境界値テスト
        let extremes: [Int16] = [Int16.min, Int16.min + 1, Int16.max - 1, Int16.max]
        for original in extremes {
            let t = ValueTokenizer.tokenize(original)
            let restored = ValueTokenizer.detokenize(isSignificant: t.isSignificant, sign: t.sign, token: t.token, bypassBits: t.bypassBits)
            XCTAssertEqual(original, restored, "Failed at extreme value: \(original)")
        }
    }
}
