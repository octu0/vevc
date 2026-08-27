import Foundation

/// SNN 予測器定数定義
public enum SNNPredictorConstants {
    public static let vThresh: Int16 = 2048
    public static let leakShift: Int16 = 2
    public static let kDefault: Int = 6
    public static let epsilonDefault: Int16 = 1
}

/// 純整数 SIMD16 SNN 4x4 残差予測器エンジン (Q7/Q11 固定小数点, LIF 整数ダイナミクス)
public struct SNNPureIntegerPredictor: DPCMPredictorEngine, Sendable {
    public init() {}

    @inline(__always)
    public func predictBlock(
        transmittedK: UnsafePointer<Int16>,
        context: DPCMPredictContext,
        output: UnsafeMutablePointer<Int16>
    ) {
        let k = context.k

        // 1. 初期推定 (p1 MED 外挿) を出力バッファ上に展開
        let medEngine = DPCMPredictorMED()
        medEngine.predictBlock(transmittedK: transmittedK, context: context, output: output)

        // 2. 4 チャンネル特徴量ベクトルのスタック構築 (Q11: SIMD16<Int16> x 4)
        var ch0 = SIMD16<Int16>()
        var ch1 = SIMD16<Int16>()
        var ch2 = SIMD16<Int16>()
        var ch3 = SIMD16<Int16>()

        let top0 = context.topBoundary.0
        let top1 = context.topBoundary.1
        let top2 = context.topBoundary.2
        let top3 = context.topBoundary.3

        let left0 = context.leftBoundary.0
        let left1 = context.leftBoundary.1
        let left2 = context.leftBoundary.2
        let left3 = context.leftBoundary.3

        let mc = context.mcPred

        // 走査順に沿って特徴量を展開
        for i in 0..<16 {
            let pt = context.scanOrder.toXY(index: i)
            let x = pt.x
            let y = pt.y
            let rawIdx = y * 4 + x

            // Ch 0: 初期推定値
            ch0[i] = output[rawIdx]

            // Ch 1: 境界距離加重射影 (Int32 拡張によりオーバーフローを完全防止)
            var topVal: Int16 = 0
            switch x {
            case 0: topVal = top0
            case 1: topVal = top1
            case 2: topVal = top2
            default: topVal = top3
            }

            var leftVal: Int16 = 0
            switch y {
            case 0: leftVal = left0
            case 1: leftVal = left1
            case 2: leftVal = left2
            default: leftVal = left3
            }

            let distL = Int32(x + 1)
            let distT = Int32(y + 1)
            let sumVal = Int32(leftVal) * distT + Int32(topVal) * distL
            ch1[i] = Int16(sumVal / (distL + distT))

            // Ch 2: MC 予測値
            switch rawIdx {
            case 0: ch2[i] = mc.0
            case 1: ch2[i] = mc.1
            case 2: ch2[i] = mc.2
            case 3: ch2[i] = mc.3
            case 4: ch2[i] = mc.4
            case 5: ch2[i] = mc.5
            case 6: ch2[i] = mc.6
            case 7: ch2[i] = mc.7
            case 8: ch2[i] = mc.8
            case 9: ch2[i] = mc.9
            case 10: ch2[i] = mc.10
            case 11: ch2[i] = mc.11
            case 12: ch2[i] = mc.12
            case 13: ch2[i] = mc.13
            case 14: ch2[i] = mc.14
            default: ch2[i] = mc.15
            }

            // Ch 3: 位置および量子化ステップ埋め込み
            let posBias = Int16((y * 4 + x) * 64)
            ch3[i] = posBias &+ (context.qLow &* 8)
        }

        // 3. Layer 1: 純整数 LIF ダイナミクス (T=2, Q7/Q11)
        var i1 = SIMD16<Int16>()
        for i in 0..<16 {
            let diffMC = ch2[i] &- ch0[i]
            let diffBound = ch1[i] &- ch0[i]
            let curr = (diffMC >> 1) &+ (diffBound >> 2) &+ (ch3[i] >> 4)
            i1[i] = curr
        }

        let vTh = SNNPredictorConstants.vThresh

        // タイムステップ t = 0
        var u1_0 = i1
        let spk1_0 = vTh .<= u1_0
        u1_0 = u1_0.replacing(with: u1_0 &- vTh, where: spk1_0)

        // タイムステップ t = 1 (リーク減衰 beta = 0.75)
        var u1_leak1 = SIMD16<Int16>()
        for i in 0..<16 {
            u1_leak1[i] = u1_0[i] &- (u1_0[i] >> SNNPredictorConstants.leakShift)
        }
        var u1_1 = u1_leak1 &+ i1
        let spk1_1 = vTh .<= u1_1
        u1_1 = u1_1.replacing(with: u1_1 &- vTh, where: spk1_1)

        // 4. Layer 2 / Output: 補正値累積 (SIMD16 整数積和)
        var delta = SIMD16<Int16>()
        for i in 0..<16 {
            var corr: Int16 = 0
            if spk1_0[i] {
                corr = corr &+ 1
            }
            if spk1_1[i] {
                corr = corr &+ 1
            }
            let fine = (u1_1[i] &+ 512) >> 10
            delta[i] = corr &+ fine
        }

        // 5. 送信済み K 要素の誤差ゼロ強制一致 & テール補正適用
        for i in 0..<16 {
            let pt = context.scanOrder.toXY(index: i)
            let rawIdx = pt.y * 4 + pt.x
            if i < k {
                output[rawIdx] = transmittedK[i]
            } else {
                output[rawIdx] = output[rawIdx] &+ delta[i]
            }
        }
    }
}
