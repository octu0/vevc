import XCTest
@testable import vevc

final class AQMapTests: XCTestCase {

    func testEncodeDecodeAQMapUniform() {
        let blockCount = 100
        let uniformLevels: [UInt8] = [0, 1, 2, 3, 4]
        
        for level in uniformLevels {
            let levels = [UInt8](repeating: level, count: blockCount)
            let encoded = encodeAQMap(levels: levels)
            
            // Should be highly compressed (RLE or default level optimization)
            XCTAssertLessThan(encoded.count, blockCount)
            
            let decoded = decodeAQMap(data: encoded, blockCount: blockCount)
            XCTAssertEqual(decoded.count, blockCount)
            XCTAssertEqual(levels, decoded)
        }
    }
    
    func testEncodeDecodeAQMapMixed() {
        let blockCount = 64
        var levels = [UInt8](repeating: 2, count: blockCount)
        
        // Add some noise
        levels[10] = 0
        levels[11] = 1
        levels[30] = 3
        levels[31] = 4
        levels[50] = 1
        
        let encoded = encodeAQMap(levels: levels)
        let decoded = decodeAQMap(data: encoded, blockCount: blockCount)
        
        XCTAssertEqual(decoded.count, blockCount)
        XCTAssertEqual(levels, decoded)
    }
    
    func testEncodeDecodeAQMapEdgeCases() {
        // Zero blocks
        let encoded0 = encodeAQMap(levels: [])
        let decoded0 = decodeAQMap(data: encoded0, blockCount: 0)
        XCTAssertTrue(decoded0.isEmpty)
        
        // 1 block
        let encoded1 = encodeAQMap(levels: [3])
        let decoded1 = decodeAQMap(data: encoded1, blockCount: 1)
        XCTAssertEqual(decoded1, [3])
        
        // Non-standard block count (e.g. odd number)
        var levelsOdd = [UInt8](repeating: 2, count: 17)
        levelsOdd[3] = 4
        levelsOdd[16] = 1
        let encodedOdd = encodeAQMap(levels: levelsOdd)
        let decodedOdd = decodeAQMap(data: encodedOdd, blockCount: 17)
        XCTAssertEqual(decodedOdd, levelsOdd)
    }
    
    func testEncodeDecodeAQMapEmptyDataFallback() {
        let blockCount = 42
        // When decoding an empty bitstream, it should fallback to an array of 2s
        let decoded = decodeAQMap(data: [], blockCount: blockCount)
        let expected = [UInt8](repeating: 2, count: blockCount)
        XCTAssertEqual(decoded, expected)
    }
}
