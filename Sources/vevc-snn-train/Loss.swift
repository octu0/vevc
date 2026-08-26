import Foundation

public struct LossResult: Sendable {
    public var totalLoss: Float
    public var charbLoss: Float
    public var ssimLoss: Float

    public init(totalLoss: Float, charbLoss: Float, ssimLoss: Float) {
        self.totalLoss = totalLoss
        self.charbLoss = charbLoss
        self.ssimLoss = ssimLoss
    }
}

public enum LossFunctions {
    public static let epsilonCharbonnier: Float = 1e-3
    public static let c1SSIM: Float = 0.0004
    public static let c2SSIM: Float = 0.0036

    /// Computes Charbonnier Loss and its analytical gradient.
    /// L_charb = (1 / N) * \sum \sqrt{(pred - target)^2 + eps^2}
    /// \partial L / \partial pred = (pred - target) / (N * \sqrt{(pred - target)^2 + eps^2})
    @inline(__always)
    public static func charbonnierLoss(
        pred: [Float],
        target: [Float],
        grad: inout [Float]
    ) -> Float {
        let count = pred.count
        let invN: Float = 1.0 / Float(max(1, count))
        let eps2 = epsilonCharbonnier * epsilonCharbonnier
        var totalLoss: Float = 0.0

        pred.withUnsafeBufferPointer { pPtr in
            target.withUnsafeBufferPointer { tPtr in
                grad.withUnsafeMutableBufferPointer { gPtr in
                    var idx = 0
                    while idx < count {
                        let diff = pPtr[idx] - tPtr[idx]
                        let sq = (diff * diff) + eps2
                        let sqrtVal = sqrtf(sq)
                        totalLoss += sqrtVal
                        gPtr[idx] = (diff / (sqrtVal * Float(max(1, count))))
                        idx &+= 1
                    }
                }
            }
        }

        return totalLoss * invN
    }

    /// Computes SSIM proxy loss and analytical gradient over 2D patches.
    /// L_ssim = 1 - mean(SSIM(pred, target))
    public static func ssimProxyLoss(
        pred: [Float],
        target: [Float],
        width: Int,
        height: Int,
        grad: inout [Float]
    ) -> Float {
        let count = width * height
        guard 3 <= width && 3 <= height else {
            return 0.0
        }

        var ssimSum: Float = 0.0
        let inv9: Float = 1.0 / 9.0

        var dSSIM_dP = [Float](repeating: 0.0, count: count)

        // Pass 1: compute 3x3 local statistics and per-pixel SSIM derivatives
        var y = 1
        while y < height - 1 {
            var x = 1
            while x < width - 1 {
                var sumP: Float = 0.0
                var sumT: Float = 0.0
                var sumPP: Float = 0.0
                var sumTT: Float = 0.0
                var sumPT: Float = 0.0

                var dy = -1
                while dy <= 1 {
                    var dx = -1
                    while dx <= 1 {
                        let pIdx = ((y + dy) * width) + (x + dx)
                        let pVal = pred[pIdx]
                        let tVal = target[pIdx]
                        sumP += pVal
                        sumT += tVal
                        sumPP += pVal * pVal
                        sumTT += tVal * tVal
                        sumPT += pVal * tVal
                        dx &+= 1
                    }
                    dy &+= 1
                }

                let muP = sumP * inv9
                let muT = sumT * inv9
                let sigmaP2 = max(0.0, (sumPP * inv9) - (muP * muP))
                let sigmaT2 = max(0.0, (sumTT * inv9) - (muT * muT))
                let sigmaPT = (sumPT * inv9) - (muP * muT)

                let A = (2.0 * muP * muT) + c1SSIM
                let B = (2.0 * sigmaPT) + c2SSIM
                let C = (muP * muP) + (muT * muT) + c1SSIM
                let D = sigmaP2 + sigmaT2 + c2SSIM

                let denom = C * D
                let ssimLocal = (A * B) / denom
                ssimSum += ssimLocal

                // Analytical derivatives w.r.t muP, sigmaP2, sigmaPT
                let dSSIM_dmuP = ((2.0 * muT * B) - (2.0 * muP * ssimLocal * D)) / denom
                let dSSIM_dsigmaP2 = -(ssimLocal / D)
                let dSSIM_dsigmaPT = (2.0 * A) / denom

                // Distribute gradient to 3x3 local neighborhood
                dy = -1
                while dy <= 1 {
                    var dx = -1
                    while dx <= 1 {
                        let pIdx = ((y + dy) * width) + (x + dx)
                        let pVal = pred[pIdx]
                        let tVal = target[pIdx]

                        // dSSIM / dP_k = inv9 * (dSSIM_dmuP + 2*P_k*dSSIM_dsigmaP2 + T_k*dSSIM_dsigmaPT)
                        let gradPixel = inv9 * (dSSIM_dmuP + (2.0 * pVal * dSSIM_dsigmaP2) + (tVal * dSSIM_dsigmaPT))
                        dSSIM_dP[pIdx] += gradPixel
                        dx &+= 1
                    }
                    dy &+= 1
                }

                x &+= 1
            }
            y &+= 1
        }

        // L_ssim = 1.0 - (ssimSum / validWindows)
        let validWindows = max(1, (width - 2) * (height - 2))
        let meanSSIM = ssimSum / Float(validWindows)
        let loss = 1.0 - meanSSIM

        // Grad = - (1 / validWindows) * dSSIM_dP
        let invW = 1.0 / Float(validWindows)
        var i = 0
        while i < count {
            grad[i] -= dSSIM_dP[i] * invW
            i &+= 1
        }

        return loss
    }

    /// Evaluates total loss combining Charbonnier Loss and SSIM Proxy Loss with analytical gradients.
    public static func computeTotalLoss(
        pred: [Float],
        target: [Float],
        width: Int,
        height: Int,
        ssimWeight: Float,
        grad: inout [Float]
    ) -> LossResult {
        let count = pred.count
        var charbGrad = [Float](repeating: 0.0, count: count)
        let charb = charbonnierLoss(pred: pred, target: target, grad: &charbGrad)

        var ssim: Float = 0.0
        var ssimGrad = [Float](repeating: 0.0, count: count)
        if 0.0 < ssimWeight {
            ssim = ssimProxyLoss(pred: pred, target: target, width: width, height: height, grad: &ssimGrad)
        }

        var idx = 0
        while idx < count {
            grad[idx] = charbGrad[idx] + (ssimWeight * ssimGrad[idx])
            idx &+= 1
        }

        let total = charb + (ssimWeight * ssim)
        return LossResult(totalLoss: total, charbLoss: charb, ssimLoss: ssim)
    }
}
