import Foundation

/// Leaky Integrate-and-Fire (LIF) neuron parameters and dynamics.
public struct LIFConfig: Sendable {
    public var beta: Float = 0.75
    public var threshold: Float = 1.0
    public var surrogateK: Float = 1.0

    public init(beta: Float = 0.75, threshold: Float = 1.0, surrogateK: Float = 1.0) {
        self.beta = beta
        self.threshold = threshold
        self.surrogateK = surrogateK
    }
}

/// Cached state of a LIF layer at one time-step for BPTT backward pass.
public struct LIFStepState {
    public var u: [Float]
    public var spike: [Float]
    public var uReset: [Float]

    public init(count: Int) {
        self.u = [Float](repeating: 0.0, count: count)
        self.spike = [Float](repeating: 0.0, count: count)
        self.uReset = [Float](repeating: 0.0, count: count)
    }
}

public enum LIFNeuron {
    /// Evaluates the Fast Sigmoid surrogate gradient: \sigma'(x) = 1 / (1 + k|x|)^2
    @inline(__always)
    public static func fastSigmoidDerivative(x: Float, k: Float) -> Float {
        let ax = abs(x)
        let denom = 1.0 + (k * ax)
        return 1.0 / (denom * denom)
    }

    /// Forward pass of LIF neuron layer for a single time step with Soft Reset (Reset-by-Subtraction).
    /// - Parameters:
    ///   - inputCurrent: Synaptic input current I[t]
    ///   - uResetPrev: Reset membrane potential from previous time step u_reset[t-1]
    ///   - config: LIF configuration (decay beta, threshold theta)
    ///   - state: Inout step state recording u, spike, and uReset for BPTT
    @inline(__always)
    public static func forwardStep(
        inputCurrent: [Float],
        uResetPrev: [Float]?,
        config: LIFConfig,
        state: inout LIFStepState
    ) {
        let count = inputCurrent.count
        let beta = config.beta
        let theta = config.threshold

        inputCurrent.withUnsafeBufferPointer { inPtr in
            state.u.withUnsafeMutableBufferPointer { uPtr in
                state.spike.withUnsafeMutableBufferPointer { spkPtr in
                    state.uReset.withUnsafeMutableBufferPointer { rstPtr in
                        var idx = 0
                        while idx < count {
                            var prevU: Float = 0.0
                            if let uResetPrev {
                                prevU = uResetPrev[idx]
                            }
                            let uVal = (beta * prevU) + inPtr[idx]
                            uPtr[idx] = uVal

                            var spikeVal: Float = 0.0
                            if theta <= uVal {
                                spikeVal = 1.0
                            }
                            spkPtr[idx] = spikeVal
                            rstPtr[idx] = uVal - (theta * spikeVal)

                            idx &+= 1
                        }
                    }
                }
            }
        }
    }

    /// Backward pass (BPTT) of LIF neuron layer for a single time step.
    /// - Parameters:
    ///   - gradSpike: Loss gradient w.r.t. spike \partial L / \partial s[t]
    ///   - gradResetNext: Loss gradient from next time step w.r.t. reset potential \partial L / \partial u_reset[t] (nil if t = T-1)
    ///   - stepState: Step state recorded during forward pass
    ///   - config: LIF configuration
    ///   - gradCurrent: Output loss gradient w.r.t. synaptic input \partial L / \partial I[t]
    ///   - gradResetPrev: Output loss gradient w.r.t. previous reset potential \partial L / \partial u_reset[t-1]
    @inline(__always)
    public static func backwardStep(
        gradSpike: [Float],
        gradResetNext: [Float]?,
        stepState: LIFStepState,
        config: LIFConfig,
        gradCurrent: inout [Float],
        gradResetPrev: inout [Float]
    ) {
        let count = gradSpike.count
        let beta = config.beta
        let theta = config.threshold
        let k = config.surrogateK

        gradSpike.withUnsafeBufferPointer { spkGradPtr in
            stepState.u.withUnsafeBufferPointer { uPtr in
                gradCurrent.withUnsafeMutableBufferPointer { curGradPtr in
                    gradResetPrev.withUnsafeMutableBufferPointer { prevGradPtr in
                        var idx = 0
                        while idx < count {
                            let uVal = uPtr[idx]
                            let x = uVal - theta
                            let sigmaPrime = fastSigmoidDerivative(x: x, k: k)
                            let dL_ds = spkGradPtr[idx]

                            var nextGrad: Float = 0.0
                            if let gradResetNext {
                                nextGrad = gradResetNext[idx]
                            }

                            // dL / du[t] = nextGrad + (dL_ds - theta * nextGrad) * sigmaPrime
                            let gradU = nextGrad + ((dL_ds - (theta * nextGrad)) * sigmaPrime)
                            curGradPtr[idx] = gradU
                            prevGradPtr[idx] = beta * gradU

                            idx &+= 1
                        }
                    }
                }
            }
        }
    }
}
