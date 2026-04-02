import XCTest
@testable import vevc

final class QuantBandingTests: XCTestCase {

    /// 緩やかなグラデーション（諧調）が、量子化によってどのように階段状（バンディング）になるかを検証する
    /// 量子化ステップが適切であれば、完全な階段状にはならず、ある程度の中間値が保たれるべき。
    func testChromaGradientBanding() {
        let size = 16
        // baseStepを大きめ（強い圧縮状態）に設定
        let baseStep = 32
        let qt = QuantizationTable(baseStep: baseStep, isChroma: true, layerIndex: 0)
        
        var block = Block2D(width: size, height: size)
        
        // テストデータ: -15から+15までの緩やかな変化（グラデーション）
        // x軸方向にのみ変化する
        for y in 0..<size {
            for x in 0..<size {
                let v = Int16(x - 8) // -8, -7, ..., 0, ..., 7
                block.data[y * size + x] = v
            }
        }
        
        let original = block.data
        
        // 量子化 -> 逆量子化
        block.withView { view in
            quantizeSIMD(&view, q: qt.qLow)
        }
        block.withView { view in
            dequantizeSIMD(&view, q: qt.qLow)
        }
        
        // 結果の解析
        var steps: Set<Int16> = []
        for x in 0..<size {
            steps.insert(block.data[x])
        }
        
        // バンディングの度合いを測る
        // -8 から +7 までの16種類の色が、いくつの色（ステップ）に丸められたか？
        // qt.qLow.step = max(1, 32 / 8) = 4 となると、 -8, -4, 0, 4 の4種類くらいに減ってしまう。
        // もしステップが改善されれば、より多くの諧調（ユニークな値）が残るべきである。
        
        print("--- Chroma QLow (\(qt.qLow.step)) Gradient ---")
        print("Original: \((0..<size).map{ original[$0] })")
        print("Quantized: \((0..<size).map{ block.data[$0] })")
        print("Unique Steps Count: \(steps.count)")
        
        // 本来16諧調あったものが、例えば4つ以下に潰れている場合はバンディングが激しいとする
        // 改善後は、より多くのステップが残る（またはステップ間の段差が小さくなる）ことを期待する
        XCTAssertGreaterThanOrEqual(steps.count, 8, "バンディングが強すぎます。元の16諧調の内、ユニークな値が \(steps.count) 個しか維持されていません。")
    }
}
