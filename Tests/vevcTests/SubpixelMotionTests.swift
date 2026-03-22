import XCTest
@testable import vevc

final class SubpixelMotionTests: XCTestCase {
    
    // HEVC準拠の輝度(Luma) 8-tap フィルタ係数
    // 1/4: [-1,  4, -10, 58, 17, -5,  1,  0]
    // 2/4 (Half): [-1,  4, -11, 40, 40, -11,  4, -1]
    // 3/4: [ 0,  1,  -5, 17, 58, -10,  4, -1]
    
    func testInterpolateHalfPelHorizontal() {
        // [0, 0, 0, 100, 100, 0, 0, 0] のようなエッジで補間テスト
        let source: [Int16] = [10, 10, 10, 100, 100, 10, 10, 10]
        
        let filtered = source.withUnsafeBufferPointer { ptr -> Int16 in
            // Offset 3 (値100) と Offset 4 (値100) の間を補間する
            return subpixelInterpolateHalfX(ptr: ptr.baseAddress!, offset: 3)
        }
        
        // 期待値の計算:
        // (-1*10 + 4*10 - 11*10 + 40*100 + 40*100 - 11*10 + 4*10 - 1*10) / 64
        // = 7840
        // (7840 + 32) >> 6 = 123
        XCTAssertEqual(filtered, 123)
    }
    
    func testInterpolateQuarterPelHorizontal() {
        let source: [Int16] = [10, 10, 10, 100, 100, 10, 10, 10]
        
        let filtered = source.withUnsafeBufferPointer { ptr -> Int16 in
            // Offset 3 (値100) の1/4ピクセル右を補間する (1/4 pel)
            return subpixelInterpolateQuarterX(ptr: ptr.baseAddress!, offset: 3)
        }
        
        // 期待値の計算:
        // (-1*10 + 4*10 - 10*10 + 58*100 + 17*100 - 5*10 + 1*10 + 0*10) / 64
        // = (-10 + 40 - 100 + 5800 + 1700 - 50 + 10 + 0) / 64
        // = 7390 / 64 = 115
        XCTAssertEqual(filtered, 115)
    }
    
    func testInterpolateThreeQuarterPelHorizontal() {
        let source: [Int16] = [10, 10, 10, 100, 100, 10, 10, 10]
        
        let filtered = source.withUnsafeBufferPointer { ptr -> Int16 in
            // Offset 3 (値100) の3/4ピクセル右を補間する (3/4 pel)
            return subpixelInterpolateThreeQuarterX(ptr: ptr.baseAddress!, offset: 3)
        }
        
        // 期待値の計算:
        // (0*10 + 1*10 - 5*10 + 17*100 + 58*100 - 10*10 + 4*10 - 1*10) / 64
        // = (0 + 10 - 50 + 1700 + 5800 - 100 + 40 - 10) / 64
        // = 7390 / 64 = 115
        XCTAssertEqual(filtered, 115)
    }
    
    // 2Dブロック（縦横の Fractional 補間）の統合テスト
    func testInterpolateBlockFractional() {
        let w = 16
        let h = 16
        var source = [Int16](repeating: 0, count: w * h)
        // 中央に四角形を描画
        for y in 4..<12 {
            for x in 4..<12 {
                source[y * w + x] = 200
            }
        }
        
        var dest = [Int16](repeating: 0, count: 8 * 8)
        
        source.withUnsafeBufferPointer { srcPtr in
            dest.withUnsafeMutableBufferPointer { dstPtr in
                // (3.5, 3.5) の位置から 8x8 ブロックを切り出す (Half-X, Half-Y)
                subpixelInterpolateBlock(
                    src: srcPtr.baseAddress!, srcStride: w,
                    dst: dstPtr.baseAddress!, dstStride: 8,
                    width: 8, height: 8,
                    fracX: 2, fracY: 2, // 2/4 = Half-pel
                    startX: 3, startY: 3
                )
            }
        }
        
        // 元の四角形は x:4~11, y:4~11 だが、
        // (3.5, 3.5)から8x8切り出すと、補間された滑らかなエッジを持つブロックが得られるかを確認
        // トップ左隅の (x=0,y=0) つまりオリジナルの (3.5, 3.5) 近辺はゼロより少し高い値を持つはず
        XCTAssertGreaterThan(dest[0], 0)
        XCTAssertLessThan(dest[0], 100)
    }
}
