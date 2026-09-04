import Foundation

// MARK: - 数学ヘルパー関数

@inline(__always)
func softplus(_ x: Float) -> Float {
    if 20.0 <= x {
        return x
    }
    if x <= -20.0 {
        return exp(x)
    }
    return log1p(exp(x))
}

@inline(__always)
func softplusInverse(_ y: Float) -> Float {
    if 20.0 <= y {
        return y
    }
    if y <= 0.00001 {
        return -11.5
    }
    return log(expm1(y))
}

@inline(__always)
func sigmoidFloat(_ x: Float) -> Float {
    if 20.0 <= x {
        return 1.0
    }
    if x <= -20.0 {
        return 0.0
    }
    return 1.0 / (1.0 + exp(-x))
}

@inline(__always)
func geluFloat(_ x: Float) -> Float {
    let sqrt2OverPi: Float = 0.7978845608
    let u = sqrt2OverPi * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + tanh(u))
}

@inline(__always)
func dGeluFloat(_ x: Float) -> Float {
    let sqrt2OverPi: Float = 0.7978845608
    let u = sqrt2OverPi * (x + 0.044715 * x * x * x)
    let th = tanh(u)
    let du = sqrt2OverPi * (1.0 + 3.0 * 0.044715 * x * x)
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du
}

// MARK: - 決定論的擬似乱数生成器 (Xorshift128+)

public struct DeterministicRNG {
    private var s0: UInt64
    private var s1: UInt64

    public init(seed: UInt64) {
        var s = seed
        if s == 0 {
            s = 1
        }
        // SplitMix64 でシードを拡散
        var z = s &+ 0x9e3779b97f4a7c15
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        self.s0 = z ^ (z >> 31)

        z = z &+ 0x9e3779b97f4a7c15
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        self.s1 = z ^ (z >> 31)
    }

    public mutating func nextUInt64() -> UInt64 {
        var x = s0
        let y = s1
        s0 = y
        x ^= x << 23
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        return s1 &+ y
    }

    public mutating func nextFloat() -> Float {
        let v = nextUInt64() >> 40
        return Float(v) / Float(1 << 24)
    }
}

// MARK: - 単一 pos モデル (Float32 Adam 学習器)

public final class PositionModelTrainer {
    public let pos: Int
    public let inDim: Int

    // モデルパラメータ (Float32)
    public var w1: [Float] // [32 * inDim]
    public var b1: [Float] // [32]
    public var w2: [Float] // [32]
    public var b2: Float
    public var rawInvScale: Float

    // Adam モーメンタム
    private var m_w1: [Float]
    private var v_w1: [Float]
    private var m_b1: [Float]
    private var v_b1: [Float]
    private var m_w2: [Float]
    private var v_w2: [Float]
    private var m_b2: Float = 0.0
    private var v_b2: Float = 0.0
    private var m_rawInvScale: Float = 0.0
    private var v_rawInvScale: Float = 0.0
    private var stepCount: Int = 0

    private let maxWeightFloat: Float = 6300.0 / 4096.0

    public init(pos: Int, inDim: Int, initialWeights: RANSContextWeightsContainer? = nil) {
        self.pos = pos
        self.inDim = inDim

        let w1Count = 32 * inDim
        self.m_w1 = [Float](repeating: 0.0, count: w1Count)
        self.v_w1 = [Float](repeating: 0.0, count: w1Count)
        self.m_b1 = [Float](repeating: 0.0, count: 32)
        self.v_b1 = [Float](repeating: 0.0, count: 32)
        self.m_w2 = [Float](repeating: 0.0, count: 32)
        self.v_w2 = [Float](repeating: 0.0, count: 32)

        if let initW = initialWeights {
            var w1F = [Float](repeating: 0.0, count: w1Count)
            let srcW1 = initW.w1FlatQ[pos]
            for i in 0..<w1Count {
                w1F[i] = Float(srcW1[i]) / 4096.0
            }
            self.w1 = w1F

            var b1F = [Float](repeating: 0.0, count: 32)
            let srcB1 = initW.b1Q[pos]
            for h in 0..<32 {
                b1F[h] = Float(srcB1[h]) / 4096.0
            }
            self.b1 = b1F

            var w2F = [Float](repeating: 0.0, count: 32)
            let srcW2 = initW.w2Q[pos]
            for h in 0..<32 {
                w2F[h] = Float(srcW2[h]) / 4096.0
            }
            self.w2 = w2F

            self.b2 = Float(initW.b2Q[pos]) / 4096.0
            let invSF = Float(initW.invScalesQ[pos]) / 4096.0
            self.rawInvScale = softplusInverse(max(0.01, invSF))
        } else {
            self.w1 = [Float](repeating: 0.0, count: w1Count)
            self.b1 = [Float](repeating: 0.0, count: 32)
            self.w2 = [Float](repeating: 0.0, count: 32)
            self.b2 = 0.0
            self.rawInvScale = softplusInverse(1.2)
        }
    }

    /// Forward 計算: 入力特徴から (mu, invScale) を計算
    public func forward(feat: [Float]) -> (mu: Float, invScale: Float, z1: [Float], h: [Float]) {
        var z1 = [Float](repeating: 0.0, count: 32)
        var h = [Float](repeating: 0.0, count: 32)
        for i in 0..<32 {
            var sum = b1[i]
            let wOff = i * inDim
            for d in 0..<inDim {
                sum += w1[wOff + d] * feat[d]
            }
            z1[i] = sum
            h[i] = geluFloat(sum)
        }

        var mu = b2
        for i in 0..<32 {
            mu += w2[i] * h[i]
        }
        let invScale = softplus(rawInvScale)
        return (mu, invScale, z1, h)
    }

    /// ミニバッチ勾配計算および Adam パラメータ更新
    public func trainBatch(
        feats: [[Float]],
        vals: [Int16],
        learningRate: Float = 0.001
    ) -> Float {
        let batchSize = feats.count
        if batchSize <= 0 {
            return 0.0
        }

        var grad_w1 = [Float](repeating: 0.0, count: 32 * inDim)
        var grad_b1 = [Float](repeating: 0.0, count: 32)
        var grad_w2 = [Float](repeating: 0.0, count: 32)
        var grad_b2: Float = 0.0
        var grad_rawInvScale: Float = 0.0
        var totalLoss: Float = 0.0

        for b in 0..<batchSize {
            let feat = feats[b]
            let val = vals[b]
            let (mu, invScale, z1, h) = forward(feat: feat)

            let y = Float(val) + 0.5
            let z = invScale * (y - mu)
            let sigZ = sigmoidFloat(z)

            // NLL loss (nats)
            let loss = -log(max(0.00001, invScale)) + abs(z) + 2.0 * log1p(exp(-abs(z)))
            totalLoss += loss

            // 勾配計算
            let dL_dmu = invScale * (1.0 - 2.0 * sigZ)
            let dL_dinvScale = (z * (2.0 * sigZ - 1.0) - 1.0) / max(0.00001, invScale)
            let sigAlpha = sigmoidFloat(rawInvScale)
            let dL_drawInvScale = dL_dinvScale * sigAlpha

            grad_b2 += dL_dmu
            grad_rawInvScale += dL_drawInvScale

            var dL_dh = [Float](repeating: 0.0, count: 32)
            for i in 0..<32 {
                grad_w2[i] += dL_dmu * h[i]
                dL_dh[i] = dL_dmu * w2[i]
            }

            for i in 0..<32 {
                let dL_dz1 = dL_dh[i] * dGeluFloat(z1[i])
                grad_b1[i] += dL_dz1
                let wOff = i * inDim
                for d in 0..<inDim {
                    grad_w1[wOff + d] += dL_dz1 * feat[d]
                }
            }
        }

        let scale = 1.0 / Float(batchSize)
        stepCount += 1
        let beta1: Float = 0.9
        let beta2: Float = 0.999
        let eps: Float = 1e-8

        let bc1 = 1.0 - pow(beta1, Float(stepCount))
        let bc2 = 1.0 - pow(beta2, Float(stepCount))

        // b2 更新
        let gb2 = grad_b2 * scale
        m_b2 = beta1 * m_b2 + (1.0 - beta1) * gb2
        v_b2 = beta2 * v_b2 + (1.0 - beta2) * gb2 * gb2
        let m_b2_hat = m_b2 / bc1
        let v_b2_hat = v_b2 / bc2
        b2 -= learningRate * m_b2_hat / (sqrt(v_b2_hat) + eps)
        b2 = min(maxWeightFloat, max(-maxWeightFloat, b2))

        // rawInvScale 更新
        let gInv = grad_rawInvScale * scale
        m_rawInvScale = beta1 * m_rawInvScale + (1.0 - beta1) * gInv
        v_rawInvScale = beta2 * v_rawInvScale + (1.0 - beta2) * gInv * gInv
        let m_inv_hat = m_rawInvScale / bc1
        let v_inv_hat = v_rawInvScale / bc2
        rawInvScale -= learningRate * m_inv_hat / (sqrt(v_inv_hat) + eps)
        let curInv = softplus(rawInvScale)
        if maxWeightFloat < curInv {
            rawInvScale = softplusInverse(maxWeightFloat)
        }

        // w2 更新
        for i in 0..<32 {
            let gw2 = grad_w2[i] * scale
            m_w2[i] = beta1 * m_w2[i] + (1.0 - beta1) * gw2
            v_w2[i] = beta2 * v_w2[i] + (1.0 - beta2) * gw2 * gw2
            let mw_hat = m_w2[i] / bc1
            let vw_hat = v_w2[i] / bc2
            w2[i] -= learningRate * mw_hat / (sqrt(vw_hat) + eps)
            w2[i] = min(maxWeightFloat, max(-maxWeightFloat, w2[i]))
        }

        // b1 更新
        for i in 0..<32 {
            let gb1 = grad_b1[i] * scale
            m_b1[i] = beta1 * m_b1[i] + (1.0 - beta1) * gb1
            v_b1[i] = beta2 * v_b1[i] + (1.0 - beta2) * gb1 * gb1
            let mb_hat = m_b1[i] / bc1
            let vb_hat = v_b1[i] / bc2
            b1[i] -= learningRate * mb_hat / (sqrt(vb_hat) + eps)
            b1[i] = min(maxWeightFloat, max(-maxWeightFloat, b1[i]))
        }

        // w1 更新
        let w1Count = 32 * inDim
        for k in 0..<w1Count {
            let gw1 = grad_w1[k] * scale
            m_w1[k] = beta1 * m_w1[k] + (1.0 - beta1) * gw1
            v_w1[k] = beta2 * v_w1[k] + (1.0 - beta2) * gw1 * gw1
            let mw_hat = m_w1[k] / bc1
            let vw_hat = v_w1[k] / bc2
            w1[k] -= learningRate * mw_hat / (sqrt(vw_hat) + eps)
            w1[k] = min(maxWeightFloat, max(-maxWeightFloat, w1[k]))
        }

        return totalLoss * scale
    }

    /// Q12 固定小数点への量子化
    public func quantizeQ12() -> (w2: [Int32], b1: [Int32], w1: [Int32], b2: Int32, invScale: Int32) {
        var q_w2 = [Int32](repeating: 0, count: 32)
        for i in 0..<32 {
            let v = Int32(round(w2[i] * 4096.0))
            q_w2[i] = min(6300, max(-6300, v))
        }

        var q_b1 = [Int32](repeating: 0, count: 32)
        for i in 0..<32 {
            let v = Int32(round(b1[i] * 4096.0))
            q_b1[i] = min(6300, max(-6300, v))
        }

        let w1Count = 32 * inDim
        var q_w1 = [Int32](repeating: 0, count: w1Count)
        for k in 0..<w1Count {
            let v = Int32(round(w1[k] * 4096.0))
            q_w1[k] = min(6300, max(-6300, v))
        }

        let q_b2_val = Int32(round(b2 * 4096.0))
        let q_b2 = min(6300, max(-6300, q_b2_val))

        let invSF = softplus(rawInvScale)
        let q_inv_val = Int32(round(invSF * 4096.0))
        let q_inv = min(6300, max(1, q_inv_val))

        return (q_w2, q_b1, q_w1, q_b2, q_inv)
    }
}
