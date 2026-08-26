import Foundation

public struct SNNModelParams: Sendable {
    public var w1: [Float]    // Shape: [8, 4, 3, 3] = 288
    public var b1: [Float]    // Shape: [8] = 8
    public var w2: [Float]    // Shape: [8, 8, 1, 1] = 64
    public var b2: [Float]    // Shape: [8] = 8
    public var wOut: [Float]  // Shape: [1, 8, 1, 1] = 8
    public var bOut: [Float]  // Shape: [1] = 1

    public init(randomSeed: UInt64 = 42) {
        self.w1 = [Float](repeating: 0.0, count: 8 * 4 * 3 * 3)
        self.b1 = [Float](repeating: 0.0, count: 8)
        self.w2 = [Float](repeating: 0.0, count: 8 * 8)
        self.b2 = [Float](repeating: 0.0, count: 8)
        self.wOut = [Float](repeating: 0.0, count: 8)
        self.bOut = [Float](repeating: 0.0, count: 1)

        var rng = SplitMix64(seed: randomSeed)
        // He uniform initialization
        let limit1 = sqrtf(6.0 / Float(4 * 9))
        for i in 0..<w1.count {
            w1[i] = rng.nextFloat(in: -limit1...limit1)
        }
        let limit2 = sqrtf(6.0 / Float(8))
        for i in 0..<w2.count {
            w2[i] = rng.nextFloat(in: -limit2...limit2)
        }
        let limitOut = sqrtf(6.0 / Float(8))
        for i in 0..<wOut.count {
            wOut[i] = rng.nextFloat(in: -limitOut...limitOut)
        }
    }

    public func toParameterArray() -> [[Float]] {
        return [w1, b1, w2, b2, wOut, bOut]
    }

    public mutating func updateFromParameterArray(_ arr: [[Float]]) {
        guard arr.count == 6 else { return }
        self.w1 = arr[0]
        self.b1 = arr[1]
        self.w2 = arr[2]
        self.b2 = arr[3]
        self.wOut = arr[4]
        self.bOut = arr[5]
    }
}

public struct SNNGradients: Sendable {
    public var dw1: [Float]
    public var db1: [Float]
    public var dw2: [Float]
    public var db2: [Float]
    public var dwOut: [Float]
    public var dbOut: [Float]

    public init() {
        self.dw1 = [Float](repeating: 0.0, count: 8 * 4 * 3 * 3)
        self.db1 = [Float](repeating: 0.0, count: 8)
        self.dw2 = [Float](repeating: 0.0, count: 8 * 8)
        self.db2 = [Float](repeating: 0.0, count: 8)
        self.dwOut = [Float](repeating: 0.0, count: 8)
        self.dbOut = [Float](repeating: 0.0, count: 1)
    }

    public mutating func reset() {
        dw1.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0.0) }
        db1.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0.0) }
        dw2.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0.0) }
        db2.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0.0) }
        dwOut.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0.0) }
        dbOut.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0.0) }
    }

    public func toGradientArray() -> [[Float]] {
        return [dw1, db1, dw2, db2, dwOut, dbOut]
    }

    public mutating func updateFromGradientArray(_ arr: [[Float]]) {
        guard arr.count == 6 else { return }
        self.dw1 = arr[0]
        self.db1 = arr[1]
        self.dw2 = arr[2]
        self.db2 = arr[3]
        self.dwOut = arr[4]
        self.dbOut = arr[5]
    }
}

public struct ForwardCache {
    public var width: Int
    public var height: Int
    public var numSteps: Int
    public var inputFeatures: [Float] // [4 * H * W]
    public var l1States: [LIFStepState]
    public var l2States: [LIFStepState]
    public var l1Current: [[Float]]   // [T][8 * H * W]
    public var l2Current: [[Float]]   // [T][8 * H * W]

    public init(width: Int, height: Int, numSteps: Int = 2) {
        self.width = width
        self.height = height
        self.numSteps = numSteps
        let spatialCount = width * height
        self.inputFeatures = [Float](repeating: 0.0, count: 4 * spatialCount)
        self.l1States = (0..<numSteps).map { _ in LIFStepState(count: 8 * spatialCount) }
        self.l2States = (0..<numSteps).map { _ in LIFStepState(count: 8 * spatialCount) }
        self.l1Current = (0..<numSteps).map { _ in [Float](repeating: 0.0, count: 8 * spatialCount) }
        self.l2Current = (0..<numSteps).map { _ in [Float](repeating: 0.0, count: 8 * spatialCount) }
    }
}

public struct SplitMix64 {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }

    public mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
        let u = Float(next() >> 40) / Float(1 << 24)
        return range.lowerBound + (u * (range.upperBound - range.lowerBound))
    }
}

public final class SNNModel: @unchecked Sendable {
    public var params: SNNModelParams
    public var lifConfig1: LIFConfig
    public var lifConfig2: LIFConfig
    public var numSteps: Int

    public init(
        params: SNNModelParams = SNNModelParams(),
        lifConfig1: LIFConfig = LIFConfig(beta: 0.75, threshold: 1.0, surrogateK: 1.0),
        lifConfig2: LIFConfig = LIFConfig(beta: 0.75, threshold: 1.0, surrogateK: 1.0),
        numSteps: Int = 2
    ) {
        self.params = params
        self.lifConfig1 = lifConfig1
        self.lifConfig2 = lifConfig2
        self.numSteps = numSteps
    }

    // MARK: - Spatial Conv 3x3 (4ch -> 8ch)
    @inline(__always)
    public static func conv3x3Forward(
        input: [Float],
        weights: [Float],
        biases: [Float],
        width: Int,
        height: Int,
        output: inout [Float]
    ) {
        let spatialCount = width * height
        for outCh in 0..<8 {
            let bias = biases[outCh]
            let outOffset = outCh * spatialCount

            var y = 0
            while y < height {
                var x = 0
                while x < width {
                    var sum = bias
                    for inC in 0..<4 {
                        let inOffset = inC * spatialCount
                        let wOffset = (outCh * 4 + inC) * 9

                        var ky = -1
                        while ky <= 1 {
                            let iy = min(height - 1, max(0, y + ky))
                            var kx = -1
                            while kx <= 1 {
                                let ix = min(width - 1, max(0, x + kx))
                                let inVal = input[inOffset + (iy * width) + ix]
                                let wVal = weights[wOffset + ((ky + 1) * 3) + (kx + 1)]
                                sum += inVal * wVal
                                kx &+= 1
                            }
                            ky &+= 1
                        }
                    }
                    output[outOffset + (y * width) + x] = sum
                    x &+= 1
                }
                y &+= 1
            }
        }
    }

    // MARK: - Pointwise Conv 1x1 (8ch -> 8ch)
    @inline(__always)
    public static func conv1x1Forward8to8(
        input: [Float],
        weights: [Float],
        biases: [Float],
        width: Int,
        height: Int,
        output: inout [Float]
    ) {
        let spatialCount = width * height
        for outCh in 0..<8 {
            let bias = biases[outCh]
            let outOffset = outCh * spatialCount
            let wOffset = outCh * 8

            var idx = 0
            while idx < spatialCount {
                var sum = bias
                for inC in 0..<8 {
                    let inVal = input[(inC * spatialCount) + idx]
                    let wVal = weights[wOffset + inC]
                    sum += inVal * wVal
                }
                output[outOffset + idx] = sum
                idx &+= 1
            }
        }
    }

    // MARK: - Pointwise Conv 1x1 (8ch -> 1ch Output Accumulator)
    @inline(__always)
    public static func conv1x1Accumulate8to1(
        spikes2: [Float],
        weights: [Float],
        bias: Float,
        width: Int,
        height: Int,
        accumulator: inout [Float]
    ) {
        let spatialCount = width * height
        var idx = 0
        while idx < spatialCount {
            var sum = bias
            for inC in 0..<8 {
                let spk = spikes2[(inC * spatialCount) + idx]
                let w = weights[inC]
                sum += spk * w
            }
            accumulator[idx] += sum
            idx &+= 1
        }
    }

    // MARK: - Forward Pass
    public func forward(
        inputFeatures: [Float],
        width: Int,
        height: Int,
        cache: inout ForwardCache
    ) -> [Float] {
        let spatialCount = width * height
        cache.width = width
        cache.height = height
        cache.numSteps = numSteps
        cache.inputFeatures = inputFeatures

        var outputResidual = [Float](repeating: 0.0, count: spatialCount)

        // Time-step integration T=2
        for t in 0..<numSteps {
            // Layer 1: Conv 3x3 (4ch -> 8ch)
            Self.conv3x3Forward(
                input: inputFeatures,
                weights: params.w1,
                biases: params.b1,
                width: width,
                height: height,
                output: &cache.l1Current[t]
            )

            // Layer 1: LIF neuron forward
            var l1PrevReset: [Float]? = nil
            if 0 < t {
                l1PrevReset = cache.l1States[t - 1].uReset
            }
            LIFNeuron.forwardStep(
                inputCurrent: cache.l1Current[t],
                uResetPrev: l1PrevReset,
                config: lifConfig1,
                state: &cache.l1States[t]
            )

            // Layer 2: Pointwise Conv 1x1 (8ch -> 8ch)
            Self.conv1x1Forward8to8(
                input: cache.l1States[t].spike,
                weights: params.w2,
                biases: params.b2,
                width: width,
                height: height,
                output: &cache.l2Current[t]
            )

            // Layer 2: LIF neuron forward
            var l2PrevReset: [Float]? = nil
            if 0 < t {
                l2PrevReset = cache.l2States[t - 1].uReset
            }
            LIFNeuron.forwardStep(
                inputCurrent: cache.l2Current[t],
                uResetPrev: l2PrevReset,
                config: lifConfig2,
                state: &cache.l2States[t]
            )

            // Layer 3: Output Linear Accumulator (8ch -> 1ch)
            Self.conv1x1Accumulate8to1(
                spikes2: cache.l2States[t].spike,
                weights: params.wOut,
                bias: params.bOut[0],
                width: width,
                height: height,
                accumulator: &outputResidual
            )
        }

        return outputResidual
    }

    // MARK: - BPTT Backward Pass
    public func backward(
        gradOutput: [Float],
        cache: ForwardCache,
        grads: inout SNNGradients
    ) {
        let width = cache.width
        let height = cache.height
        let spatialCount = width * height
        let tSteps = cache.numSteps

        var gradL2ResetNext: [Float]? = nil
        var gradL1ResetNext: [Float]? = nil

        var dL_dI2 = [Float](repeating: 0.0, count: 8 * spatialCount)
        var dL_du2ResetPrev = [Float](repeating: 0.0, count: 8 * spatialCount)

        var dL_dI1 = [Float](repeating: 0.0, count: 8 * spatialCount)
        var dL_du1ResetPrev = [Float](repeating: 0.0, count: 8 * spatialCount)

        var dL_dS1 = [Float](repeating: 0.0, count: 8 * spatialCount)

        // Iterate backward through time steps T-1 down to 0
        var t = tSteps - 1
        while 0 <= t {
            let s2 = cache.l2States[t].spike
            let s1 = cache.l1States[t].spike

            // Output layer gradients: dL / dS2[t] = wOut * gradOutput
            var dL_dS2 = [Float](repeating: 0.0, count: 8 * spatialCount)
            for inC in 0..<8 {
                let wVal = params.wOut[inC]
                let inOffset = inC * spatialCount
                var idx = 0
                while idx < spatialCount {
                    let gOut = gradOutput[idx]
                    dL_dS2[inOffset + idx] = gOut * wVal
                    grads.dwOut[inC] += gOut * s2[inOffset + idx]
                    idx &+= 1
                }
            }

            // Output bias gradient
            var sumGOut: Float = 0.0
            for val in gradOutput {
                sumGOut += val
            }
            grads.dbOut[0] += sumGOut

            // Layer 2 LIF Backward
            LIFNeuron.backwardStep(
                gradSpike: dL_dS2,
                gradResetNext: gradL2ResetNext,
                stepState: cache.l2States[t],
                config: lifConfig2,
                gradCurrent: &dL_dI2,
                gradResetPrev: &dL_du2ResetPrev
            )
            gradL2ResetNext = dL_du2ResetPrev

            // Layer 2 Pointwise Conv Gradients (W2, b2) & Backprop to S1
            dL_dS1.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0.0) }
            for outCh in 0..<8 {
                let outOffset = outCh * spatialCount
                let wOffset = outCh * 8

                var bSum: Float = 0.0
                var idx = 0
                while idx < spatialCount {
                    let gI2 = dL_dI2[outOffset + idx]
                    bSum += gI2
                    for inC in 0..<8 {
                        let sVal = s1[(inC * spatialCount) + idx]
                        grads.dw2[wOffset + inC] += gI2 * sVal
                        dL_dS1[(inC * spatialCount) + idx] += gI2 * params.w2[wOffset + inC]
                    }
                    idx &+= 1
                }
                grads.db2[outCh] += bSum
            }

            // Layer 1 LIF Backward
            LIFNeuron.backwardStep(
                gradSpike: dL_dS1,
                gradResetNext: gradL1ResetNext,
                stepState: cache.l1States[t],
                config: lifConfig1,
                gradCurrent: &dL_dI1,
                gradResetPrev: &dL_du1ResetPrev
            )
            gradL1ResetNext = dL_du1ResetPrev

            // Layer 1 Spatial Conv 3x3 Gradients (W1, b1)
            for outCh in 0..<8 {
                let outOffset = outCh * spatialCount

                var bSum: Float = 0.0
                var y = 0
                while y < height {
                    var x = 0
                    while x < width {
                        let gI1 = dL_dI1[outOffset + (y * width) + x]
                        bSum += gI1

                        for inC in 0..<4 {
                            let inOffset = inC * spatialCount
                            let wOffset = (outCh * 4 + inC) * 9

                            var ky = -1
                            while ky <= 1 {
                                let iy = min(height - 1, max(0, y + ky))
                                var kx = -1
                                while kx <= 1 {
                                    let ix = min(width - 1, max(0, x + kx))
                                    let inVal = cache.inputFeatures[inOffset + (iy * width) + ix]
                                    let kIdx = ((ky + 1) * 3) + (kx + 1)
                                    grads.dw1[wOffset + kIdx] += gI1 * inVal
                                    kx &+= 1
                                }
                                ky &+= 1
                            }
                        }
                        x &+= 1
                    }
                    y &+= 1
                }
                grads.db1[outCh] += bSum
            }

            t -= 1
        }
    }
}
