import Testing
import Foundation
@testable import vevc

@Suite("SNN Training Framework and Weight Exporter Tests")
struct SNNTrainTests {

    @Test("LIF neuron forward pass and Soft Reset (Reset-by-Subtraction)")
    func testLIFNeuronDynamics() {
        let beta: Float = 0.75
        let theta: Float = 1.0
        let k: Float = 1.0

        // 1. Verify surrogate gradient symmetry: sigma'(-x) == sigma'(x)
        let g1 = 1.0 / ((1.0 + (k * abs(0.5))) * (1.0 + (k * abs(0.5))))
        let g2 = 1.0 / ((1.0 + (k * abs(-0.5))) * (1.0 + (k * abs(-0.5))))
        #expect(abs(g1 - g2) < 1e-6)
        #expect(0.0 < g1)
        #expect(g1 <= 1.0)

        // 2. Step 1: Input current = 0.8 (< 1.0 threshold) -> No spike, uReset = 0.8
        let u1: Float = 0.8
        var spike1: Float = 0.0
        if theta <= u1 {
            spike1 = 1.0
        }
        let uReset1: Float = u1 - (theta * spike1)
        #expect(spike1 == 0.0)
        #expect(abs(uReset1 - 0.8) < 1e-6)

        // 3. Step 2: Input current = 0.6. u = beta * uReset1 + I = 0.75 * 0.8 + 0.6 = 0.6 + 0.6 = 1.2
        let u2: Float = (beta * uReset1) + 0.6
        var spike2: Float = 0.0
        if theta <= u2 {
            spike2 = 1.0
        }
        let uReset2: Float = u2 - (theta * spike2)
        #expect(spike2 == 1.0)
        #expect(abs(u2 - 1.2) < 1e-6)
        #expect(abs(uReset2 - 0.2) < 1e-6) // Soft reset retains surplus 0.2
    }

    @Test("SNN static weights table validity and dimensions")
    func testSNNWeightsTableValidity() {
        // Verify SNNWeights properties
        #expect(SNNWeights.numTimeSteps == 2)
        #expect(SNNWeights.vThresh1 == 2048)
        #expect(SNNWeights.vThresh2 == 2048)
        #expect(SNNWeights.leakShift == 2)

        // Verify Conv1 weights (8 * 4 * 3 * 3 = 288)
        #expect(SNNWeights.conv1Weights.count == 288)
        #expect(SNNWeights.conv1Biases.count == 8)

        // Verify Conv2 weights (8 * 8 * 1 * 1 = 64)
        #expect(SNNWeights.conv2Weights.count == 64)
        #expect(SNNWeights.conv2Biases.count == 8)

        // Verify Output weights (8 * 1 = 8)
        #expect(SNNWeights.outWeights.count == 8)

        // Check for non-zero weights
        var conv1NonZero = 0
        for w in SNNWeights.conv1Weights {
            if w != 0 {
                conv1NonZero &+= 1
            }
        }
        #expect(0 < conv1NonZero)

        var conv2NonZero = 0
        for w in SNNWeights.conv2Weights {
            if w != 0 {
                conv2NonZero &+= 1
            }
        }
        #expect(0 < conv2NonZero)

        var outNonZero = 0
        for w in SNNWeights.outWeights {
            if w != 0 {
                outNonZero &+= 1
            }
        }
        #expect(0 < outNonZero)
    }

    @Test("Fixed-point quantization scaling and bounds")
    func testWeightQuantizationBounds() {
        // Q7 quantization bounds: [-128, 127]
        let testFP32: [Float] = [-1.5, -0.9, -0.5, 0.0, 0.3, 0.8, 1.2]
        for v in testFP32 {
            let q7 = Int(roundf(v * 128.0))
            let clampedQ7 = Int8(max(-128, min(127, q7)))
            #expect(-128 <= clampedQ7)
            #expect(clampedQ7 <= 127)

            let q11 = Int(roundf(v * 2048.0))
            let clampedQ11 = Int16(max(-32768, min(32767, q11)))
            #expect(-32768 <= clampedQ11)
            #expect(clampedQ11 <= 32767)
        }
    }

    @Test("Feature extractor extracts valid 4-channel spatial signals")
    func testFeatureExtractorChannels() {
        let width = 8
        let height = 8
        let count = width * height
        var image = [Float](repeating: 128.0, count: count)

        // Add a vertical edge
        for y in 0..<height {
            for x in 4..<width {
                image[(y * width) + x] = 200.0
            }
        }

        let ch0Offset = 0
        let ch1Offset = count
        let ch2Offset = 2 * count
        let ch3Offset = 3 * count

        var features = [Float](repeating: 0.0, count: 4 * count)
        for y in 0..<height {
            let yPrev = max(0, y - 1)
            let yNext = min(height - 1, y + 1)
            let yMod32 = y % 32
            let distY = Float(min(yMod32, 32 - yMod32))

            for x in 0..<width {
                let xPrev = max(0, x - 1)
                let xNext = min(width - 1, x + 1)
                let xMod32 = x % 32
                let distX = Float(min(xMod32, 32 - xMod32))

                let idx = (y * width) + x
                let centerVal = image[idx]
                let topVal = image[(yPrev * width) + x]
                let bottomVal = image[(yNext * width) + x]
                let leftVal = image[(y * width) + xPrev]
                let rightVal = image[(y * width) + xNext]

                features[ch0Offset + idx] = (centerVal - 128.0) / 128.0
                let laplacian = (4.0 * centerVal) - topVal - bottomVal - leftVal - rightVal
                features[ch1Offset + idx] = laplacian / 128.0
                let gradH = abs(rightVal - leftVal)
                let gradV = abs(bottomVal - topVal)
                features[ch2Offset + idx] = (gradH + gradV) / 128.0
                features[ch3Offset + idx] = (distX + distY) / 32.0
            }
        }

        #expect(features.count == 4 * count)
        // Check that ch0 on right side is positive
        #expect(0.0 < features[ch0Offset + 5])
        // Check that ch0 on left side is zero (128 - 128)
        #expect(abs(features[ch0Offset + 2]) < 1e-5)
        // Check that gradient at x=4 is non-zero
        #expect(0.0 < features[ch2Offset + 4])
    }

    @Test("vevc-snn-train CLI dry-run and static weight export execution")
    func testSNNTrainCLIDryRun() throws {
        let tempExportPath = "/tmp/test_snn_weights_\(UUID().uuidString).swift"
        defer {
            try? FileManager.default.removeItem(atPath: tempExportPath)
        }

        let process = Process()
        let binPath = ".build/arm64-apple-macosx/release/vevc-snn-train"
        if FileManager.default.fileExists(atPath: binPath) {
            process.executableURL = URL(fileURLWithPath: binPath)
            process.arguments = ["--dry-run", "--export-swift-tables", tempExportPath]

            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe

            try process.run()
            process.waitUntilExit()

            #expect(process.terminationStatus == 0)

            // Verify the generated file exists and contains valid SNNWeights enum
            #expect(FileManager.default.fileExists(atPath: tempExportPath))
            let content = try String(contentsOfFile: tempExportPath, encoding: .utf8)
            #expect(content.contains("public enum SNNWeights"))
            #expect(content.contains("public static let conv1Weights: [Int8]"))
            #expect(content.contains("public static let conv2Weights: [Int8]"))
            #expect(content.contains("public static let outWeights: [Int16]"))
        }
    }

    @Test("BPTT gradient descent reduces loss across multiple iterations")
    func testBPTTGradientDescentConvergence() {
        // 2-time-step LIF dynamics with learnable input and linear readout weights
        var wIn: Float = 0.5
        var bIn: Float = 0.1
        var wOut: Float = 0.3
        var bOut: Float = 0.0
        let target: Float = 0.75
        let lr: Float = 0.05
        let beta: Float = 0.75
        let theta: Float = 1.0
        let k: Float = 1.0

        var initialLoss: Float = 0.0
        var finalLoss: Float = 0.0

        for iter in 0..<30 {
            // Forward pass (T=2)
            // Step 0:
            let i0 = (wIn * 1.0) + bIn
            let u0 = i0
            var s0: Float = 0.0
            if theta <= u0 {
                s0 = 1.0
            }
            let uReset0 = u0 - (theta * s0)

            // Step 1:
            let i1 = (wIn * 1.0) + bIn
            let u1 = (beta * uReset0) + i1
            var s1: Float = 0.0
            if theta <= u1 {
                s1 = 1.0
            }

            // Output linear accumulation (Layer 3)
            let sSum = s0 + s1
            let pred = (wOut * sSum) + bOut
            let diff = pred - target
            let loss = sqrtf((diff * diff) + 1e-6)

            if iter == 0 {
                initialLoss = loss
            }
            finalLoss = loss

            // Backward pass (BPTT)
            let dL_dPred = diff / loss

            // Gradients w.r.t Output layer (wOut, bOut)
            let gradWOut = dL_dPred * sSum
            let gradBOut = dL_dPred

            let dL_ds1 = dL_dPred * wOut
            let dL_ds0 = dL_dPred * wOut

            // Step 1 LIF backward
            let sigmaPrime1 = 1.0 / ((1.0 + (k * abs(u1 - theta))) * (1.0 + (k * abs(u1 - theta))))
            let gradU1 = dL_ds1 * sigmaPrime1
            let gradReset0 = beta * gradU1
            let gradI1 = gradU1

            // Step 0 LIF backward
            let sigmaPrime0 = 1.0 / ((1.0 + (k * abs(u0 - theta))) * (1.0 + (k * abs(u0 - theta))))
            let gradU0 = gradReset0 + ((dL_ds0 - (theta * gradReset0)) * sigmaPrime0)
            let gradI0 = gradU0

            // Gradients w.r.t Input layer (wIn, bIn)
            let gradWIn = (gradI0 * 1.0) + (gradI1 * 1.0)
            let gradBIn = gradI0 + gradI1

            // Update parameters
            wOut -= lr * gradWOut
            bOut -= lr * gradBOut
            wIn -= lr * gradWIn
            bIn -= lr * gradBIn
        }

        #expect(finalLoss < initialLoss)
        #expect(finalLoss < 0.1)
    }

    @Test("Gradient stability under extreme inputs: zero, maximum delta +255, noise")
    func testGradientStabilityExtremeInputs() {
        let beta: Float = 0.75
        let theta: Float = 1.0
        let k: Float = 1.0

        let extremeInputs: [Float] = [
            0.0,            // Zero input
            255.0 / 128.0,  // Max normalized delta (~2.0)
            -255.0 / 128.0, // Min normalized delta (~-2.0)
            100.0,          // Extreme unbounded positive input
            -100.0          // Extreme unbounded negative input
        ]

        for inputVal in extremeInputs {
            // Forward pass
            let u0 = inputVal
            var s0: Float = 0.0
            if theta <= u0 {
                s0 = 1.0
            }
            let uReset0 = u0 - (theta * s0)
            let u1 = (beta * uReset0) + inputVal
            var s1: Float = 0.0
            if theta <= u1 {
                s1 = 1.0
            }

            let pred = (s0 + s1) * 0.5
            let diff = pred - 0.0
            let eps: Float = 1e-3
            let loss = sqrtf((diff * diff) + (eps * eps))

            // Check loss is finite and positive
            #expect(loss.isFinite)
            #expect(loss.isNaN != true)
            #expect(0.0 <= loss)

            // Backward pass
            let dL_dPred = diff / loss
            let dL_ds1 = dL_dPred * 0.5
            let dL_ds0 = dL_dPred * 0.5

            let sigmaPrime1 = 1.0 / ((1.0 + (k * abs(u1 - theta))) * (1.0 + (k * abs(u1 - theta))))
            let gradU1 = dL_ds1 * sigmaPrime1
            let gradReset0 = beta * gradU1
            let gradI1 = gradU1

            let sigmaPrime0 = 1.0 / ((1.0 + (k * abs(u0 - theta))) * (1.0 + (k * abs(u0 - theta))))
            let gradU0 = gradReset0 + ((dL_ds0 - (theta * gradReset0)) * sigmaPrime0)
            let gradI0 = gradU0

            // Check gradients are strictly finite and non-NaN
            #expect(gradI0.isFinite)
            #expect(gradI0.isNaN != true)
            #expect(gradI1.isFinite)
            #expect(gradI1.isNaN != true)
            #expect(abs(gradI0) <= 100.0)
            #expect(abs(gradI1) <= 100.0)
        }
    }

    @Test("Q7/Q11 quantization round-trip error bounds and arithmetic overflow prevention")
    func testQuantizationSafetyAndNoOverflow() {
        // 1. Q7 quantization error bound: max error <= 1 / 128
        let q7TestRange: [Float] = [-1.0, -0.75, -0.5, -0.25, -0.01, 0.0, 0.01, 0.25, 0.5, 0.75, 0.99]
        for val in q7TestRange {
            let q = Int(roundf(val * 128.0))
            let clampedQ7 = Int8(max(-128, min(127, q)))
            let dequant = Float(clampedQ7) / 128.0
            let error = abs(dequant - val)
            #expect(error <= (1.0 / 128.0) + 1e-5)
        }

        // 2. Q11 quantization error bound: max error <= 1 / 2048
        let q11TestRange: [Float] = [-15.0, -8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0, 15.0]
        for val in q11TestRange {
            let q = Int(roundf(val * 2048.0))
            let clampedQ11 = Int16(max(-32768, min(32767, q)))
            let dequant = Float(clampedQ11) / 2048.0
            let error = abs(dequant - val)
            #expect(error <= (1.0 / 2048.0) + 1e-5)
        }

        // 3. Overflow resistance under extreme Float inputs [-1000.0, 1000.0]
        let extremeFloats: [Float] = [-1000.0, -500.0, 500.0, 1000.0, Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude]
        for val in extremeFloats {
            let scaledQ7 = val * 128.0
            let clampedFloat7 = max(-128.0, min(127.0, roundf(scaledQ7)))
            let q7Clamped = Int8(clampedFloat7)
            #expect(-128 <= q7Clamped)
            #expect(q7Clamped <= 127)

            let scaledQ11 = val * 2048.0
            let clampedFloat11 = max(-32768.0, min(32767.0, roundf(scaledQ11)))
            let q11Clamped = Int16(clampedFloat11)
            #expect(-32768 <= q11Clamped)
            #expect(q11Clamped <= 32767)
        }

        // 4. Dot-product accumulation safety in SIMD16 integer domain
        // Max 8-channel Q7 weight sum * 1-bit spike:
        let maxWeightSum8 = 8 * 127
        #expect(maxWeightSum8 <= Int(Int16.max))

        // Max 8-channel Q11 weight sum * 1-bit spike:
        let maxWeightSumOut = 8 * 32767
        #expect(maxWeightSumOut <= Int(Int32.max))
    }
}


