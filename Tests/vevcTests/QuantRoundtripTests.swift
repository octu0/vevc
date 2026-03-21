import XCTest
@testable import vevc

/// quantizeSignedMapping → dequantizeSignedMapping のラウンドトリップ整合性テスト
/// 白ノイズの根本原因特定のための最小テスト
final class QuantRoundtripTests: XCTestCase {

    /// 8x8ブロックでSignedMappingの量子化→逆量子化をテスト
    /// step=1 なら量子化損失はゼロで、元の値が完全に復元されるべき
    func testSignedMappingRoundTrip8x8_Step1() {
        let size = 8
        // テストデータ: 正の値、負の値、ゼロを含む
        let testValues: [Int16] = [
            0, 1, -1, 2, -2, 10, -10, 127,
            -128, 50, -50, 3, -3, 7, -7, 100,
            -100, 0, 0, 0, 5, -5, 15, -15,
            30, -30, 60, -60, 1, -1, 0, 0,
            42, -42, 99, -99, 11, -11, 22, -22,
            33, -33, 44, -44, 55, -55, 66, -66,
            77, -77, 88, -88, 0, 0, 0, 0,
            1, 2, 3, 4, -4, -3, -2, -1
        ]
        XCTAssertEqual(testValues.count, size * size)
        
        let qt = QuantizationTable(baseStep: 1)
        
        var block = Block2D(width: size, height: size)
        // テスト値をコピー
        for i in 0..<(size * size) {
            block.data[i] = testValues[i]
        }
        
        // 量子化（SignedMapping = ジグザグエンコード付き）
        block.withView { view in
            quantizeLowSignedMapping(&view, qt: qt)
        }
        
        // ジグザグエンコード後の値を保存
        let quantized = block.data
        
        // 逆量子化（SignedMapping = ジグザグデコード付き）
        block.withView { view in
            dequantizeLowSignedMapping(&view, qt: qt)
        }
        
        // 元の値と比較
        for i in 0..<(size * size) {
            let original = testValues[i]
            let restored = block.data[i]
            let zigzag = quantized[i]
            XCTAssertEqual(original, restored, 
                "位置[\(i/size),\(i%size)]: 元=\(original), zigzag=\(zigzag), 復元=\(restored)")
        }
    }
    
    /// 16x16ブロックでSignedMappingのラウンドトリップ
    func testSignedMappingRoundTrip16x16_Step1() {
        let size = 16
        let qt = QuantizationTable(baseStep: 1)
        
        var block = Block2D(width: size, height: size)
        // 正と負の両方を含むテストパターン
        for y in 0..<size {
            for x in 0..<size {
                let v: Int16
                if (x + y) % 3 == 0 {
                    v = Int16(x * 8 - 64)  // -64..+56
                } else if (x + y) % 3 == 1 {
                    v = -Int16(y * 6 + 1)  // -1..-97
                } else {
                    v = Int16(x + y)       // 0..+30
                }
                block.data[y * size + x] = v
            }
        }
        let original = block.data
        
        block.withView { view in
            quantizeLowSignedMapping(&view, qt: qt)
        }
        block.withView { view in
            dequantizeLowSignedMapping(&view, qt: qt)
        }
        
        var mismatches: [(Int, Int16, Int16)] = []
        for i in 0..<(size * size) {
            if original[i] != block.data[i] {
                mismatches.append((i, original[i], block.data[i]))
            }
        }
        XCTAssertTrue(mismatches.isEmpty, 
            "16x16 step=1 ミスマッチ \(mismatches.count)個: " + 
            mismatches.prefix(10).map { "[\($0.0)]: \($0.1)→\($0.2)" }.joined(separator: ", "))
    }
    
    /// 32x32ブロックでSignedMappingのラウンドトリップ 
    func testSignedMappingRoundTrip32x32_Step1() {
        let size = 32
        let qt = QuantizationTable(baseStep: 1)
        
        var block = Block2D(width: size, height: size)
        for y in 0..<size {
            for x in 0..<size {
                let hash = (x &* 2654435761) ^ (y &* 2246822519)
                block.data[y * size + x] = Int16(clamping: (hash % 256) - 128)
            }
        }
        let original = block.data
        
        block.withView { view in
            quantizeLowSignedMapping(&view, qt: qt)
        }
        block.withView { view in
            dequantizeLowSignedMapping(&view, qt: qt)
        }
        
        var mismatches: [(Int, Int16, Int16)] = []
        for i in 0..<(size * size) {
            if original[i] != block.data[i] {
                mismatches.append((i, original[i], block.data[i]))
            }
        }
        XCTAssertTrue(mismatches.isEmpty, 
            "32x32 step=1 ミスマッチ \(mismatches.count)個: " + 
            mismatches.prefix(10).map { "[\($0.0)]: \($0.1)→\($0.2)" }.joined(separator: ", "))
    }
    
    /// step=2でのSignedMappingラウンドトリップ（量子化損失はあるが符号は保持されるべき）
    func testSignedMappingRoundTrip_Step2_SignPreserved() {
        let size = 8
        let qt = QuantizationTable(baseStep: 2) // qLow.step=2
        
        var block = Block2D(width: size, height: size)
        let testValues: [Int16] = [
            10, -10, 20, -20, 30, -30, 0, 0,
            5, -5, 15, -15, 25, -25, 35, -35,
            40, -40, 50, -50, 60, -60, 70, -70,
            80, -80, 90, -90, 100, -100, 110, -110,
            3, -3, 7, -7, 11, -11, 13, -13,
            17, -17, 19, -19, 23, -23, 29, -29,
            31, -31, 37, -37, 41, -41, 43, -43,
            47, -47, 53, -53, 59, -59, 61, -61
        ]
        for i in 0..<(size * size) {
            block.data[i] = testValues[i]
        }
        
        block.withView { view in
            quantizeLowSignedMapping(&view, qt: qt)
        }
        block.withView { view in
            dequantizeLowSignedMapping(&view, qt: qt)
        }
        
        var signMismatches: [(Int, Int16, Int16)] = []
        for i in 0..<(size * size) {
            let orig = testValues[i]
            let restored = block.data[i]
            // ゼロ以外の値は符号が保持されるべき
            if orig != 0 && restored != 0 {
                let origSign = orig > 0
                let restoredSign = restored > 0
                if origSign != restoredSign {
                    signMismatches.append((i, orig, restored))
                }
            }
        }
        XCTAssertTrue(signMismatches.isEmpty, 
            "step=2で符号反転: " + 
            signMismatches.prefix(10).map { "[\($0.0)]: \($0.1)→\($0.2)" }.joined(separator: ", "))
    }
    
    /// Mid/High量子化のSignedMappingラウンドトリップもテスト
    func testMidHighSignedMappingRoundTrip() {
        let size = 8
        let qt = QuantizationTable(baseStep: 1) // qMid.step=2, qHigh.step=4
        
        let testValues: [Int16] = [
            10, -10, 20, -20, 30, -30, 0, 0,
            5, -5, 15, -15, 25, -25, 35, -35,
            40, -40, 50, -50, 60, -60, 70, -70,
            80, -80, 90, -90, 100, -100, 110, -110,
            1, -1, 2, -2, 3, -3, 4, -4,
            5, -5, 6, -6, 7, -7, 8, -8,
            9, -9, 10, -10, 11, -11, 12, -12,
            13, -13, 14, -14, 15, -15, 16, -16
        ]
        
        // Mid test
        var blockMid = Block2D(width: size, height: size)
        for i in 0..<(size * size) { blockMid.data[i] = testValues[i] }
        blockMid.withView { view in quantizeMidSignedMapping(&view, qt: qt) }
        blockMid.withView { view in dequantizeMidSignedMapping(&view, qt: qt) }
        
        var midSignMismatches: [(Int, Int16, Int16)] = []
        for i in 0..<(size * size) {
            let orig = testValues[i]
            let restored = blockMid.data[i]
            if orig != 0 && restored != 0 {
                if (orig > 0) != (restored > 0) {
                    midSignMismatches.append((i, orig, restored))
                }
            }
        }
        XCTAssertTrue(midSignMismatches.isEmpty,
            "Mid SignedMapping 符号反転: " + 
            midSignMismatches.prefix(10).map { "[\($0.0)]: \($0.1)→\($0.2)" }.joined(separator: ", "))
        
        // High test
        var blockHigh = Block2D(width: size, height: size)
        for i in 0..<(size * size) { blockHigh.data[i] = testValues[i] }
        blockHigh.withView { view in quantizeHighSignedMapping(&view, qt: qt) }
        blockHigh.withView { view in dequantizeHighSignedMapping(&view, qt: qt) }
        
        var highSignMismatches: [(Int, Int16, Int16)] = []
        for i in 0..<(size * size) {
            let orig = testValues[i]
            let restored = blockHigh.data[i]
            if orig != 0 && restored != 0 {
                if (orig > 0) != (restored > 0) {
                    highSignMismatches.append((i, orig, restored))
                }
            }
        }
        XCTAssertTrue(highSignMismatches.isEmpty,
            "High SignedMapping 符号反転: " + 
            highSignMismatches.prefix(10).map { "[\($0.0)]: \($0.1)→\($0.2)" }.joined(separator: ", "))
    }
}
