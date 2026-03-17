import XCTest
@testable import vevc

final class ValueTokenizerTests: XCTestCase {
    
    func testTokenizeAndDetokenize() {
        // 1 and -1 (Token 0 and 1)
        let pos1 = ValueTokenizer.tokenize(1)
        XCTAssertEqual(pos1.token, 0)
        XCTAssertEqual(pos1.bypassBits, 0)
        XCTAssertEqual(ValueTokenizer.detokenize(token: pos1.token, bypassBits: pos1.bypassBits), 1)
        
        let neg1 = ValueTokenizer.tokenize(-1)
        XCTAssertEqual(neg1.token, 1)
        XCTAssertEqual(neg1.bypassBits, 0)
        XCTAssertEqual(ValueTokenizer.detokenize(token: neg1.token, bypassBits: neg1.bypassBits), -1)
        
        // 2 and -2 (Token 2 and 3)
        let pos2 = ValueTokenizer.tokenize(2)
        XCTAssertEqual(pos2.token, 2)
        XCTAssertEqual(ValueTokenizer.detokenize(token: pos2.token, bypassBits: pos2.bypassBits), 2)
        
        let neg2 = ValueTokenizer.tokenize(-2)
        XCTAssertEqual(neg2.token, 3)
        XCTAssertEqual(ValueTokenizer.detokenize(token: neg2.token, bypassBits: neg2.bypassBits), -2)
        
        // 8 and -8 (Token 14 and 15)
        let pos8 = ValueTokenizer.tokenize(8)
        XCTAssertEqual(pos8.token, 14)
        XCTAssertEqual(ValueTokenizer.detokenize(token: pos8.token, bypassBits: pos8.bypassBits), 8)
        
        let neg8 = ValueTokenizer.tokenize(-8)
        XCTAssertEqual(neg8.token, 15)
        XCTAssertEqual(ValueTokenizer.detokenize(token: neg8.token, bypassBits: neg8.bypassBits), -8)
        
        // 9 (Token 16 or 17...)
        let pos9 = ValueTokenizer.tokenize(9)
        XCTAssertEqual(ValueTokenizer.detokenize(token: pos9.token, bypassBits: pos9.bypassBits), 9)
        
        let neg9 = ValueTokenizer.tokenize(-9)
        XCTAssertEqual(ValueTokenizer.detokenize(token: neg9.token, bypassBits: neg9.bypassBits), -9)
    }
    
    func testLosslessWithRandomOutliers() {
        var rng = SystemRandomNumberGenerator()
        
        for _ in 0..<10000 {
            let original = Int16.random(in: Int16.min...Int16.max, using: &rng)
            
            let t = ValueTokenizer.tokenize(original)
            let restored = ValueTokenizer.detokenize(token: t.token, bypassBits: t.bypassBits)
            
            XCTAssertEqual(original, restored, "Failed to losslessly transform value: \(original)")
        }
        
        let extremes: [Int16] = [Int16.min, Int16.min + 1, Int16.max - 1, Int16.max]
        for original in extremes {
            let t = ValueTokenizer.tokenize(original)
            let restored = ValueTokenizer.detokenize(token: t.token, bypassBits: t.bypassBits)
            XCTAssertEqual(original, restored, "Failed at extreme value: \(original)")
        }
    }
}
