import Foundation
import vevc

public struct TrainingSample: Sendable {
    public var inputFeatures: [Float] // [4 * H * W]
    public var targetResidual: [Float] // [H * W], normalized by 1/128.0
    public var width: Int
    public var height: Int

    public init(inputFeatures: [Float], targetResidual: [Float], width: Int, height: Int) {
        self.inputFeatures = inputFeatures
        self.targetResidual = targetResidual
        self.width = width
        self.height = height
    }
}

public enum FeatureExtractor {
    /// Extracts 4-channel spatial features from a single-plane float image in range [0, 255].
    public static func extract4ChFeatures(
        image: [Float],
        width: Int,
        height: Int
    ) -> [Float] {
        let spatialCount = width * height
        var features = [Float](repeating: 0.0, count: 4 * spatialCount)

        let ch0Offset = 0
        let ch1Offset = spatialCount
        let ch2Offset = 2 * spatialCount
        let ch3Offset = 3 * spatialCount

        var y = 0
        while y < height {
            let yPrev = max(0, y - 1)
            let yNext = min(height - 1, y + 1)
            let yMod32 = y % 32
            let distY = Float(min(yMod32, 32 - yMod32))

            var x = 0
            while x < width {
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

                // Ch 0: Centered pixel [-1.0, 1.0]
                features[ch0Offset + idx] = (centerVal - 128.0) / 128.0

                // Ch 1: 2nd-order Laplacian (ringing detector)
                let laplacian = (4.0 * centerVal) - topVal - bottomVal - leftVal - rightVal
                features[ch1Offset + idx] = laplacian / 128.0

                // Ch 2: Local gradient magnitude
                let gradH = abs(rightVal - leftVal)
                let gradV = abs(bottomVal - topVal)
                features[ch2Offset + idx] = (gradH + gradV) / 128.0

                // Ch 3: Block boundary distance indicator [0.0, 1.0]
                features[ch3Offset + idx] = (distX + distY) / 32.0

                x &+= 1
            }
            y &+= 1
        }

        return features
    }
}

public enum DatasetGenerator {
    @inline(__always)
    private static func liftRow16(_ row: inout [Int16], offset: Int) {
        var low = [Int16](repeating: 0, count: 8)
        var high = [Int16](repeating: 0, count: 8)
        for i in 0..<8 {
            low[i] = row[offset + (2 * i)]
            high[i] = row[offset + ((2 * i) + 1)]
        }

        // Predict
        for i in 0..<8 {
            var nextLow = low[7]
            if i + 1 < 8 {
                nextLow = low[i + 1]
            }
            var dither: Int16 = 0
            if i % 2 == 1 {
                dither = 1
            }
            high[i] &-= (low[i] &+ nextLow &+ dither) &>> 1
        }

        // Update
        for i in 0..<8 {
            var prevHigh = high[0]
            if 0 < i {
                prevHigh = high[i - 1]
            }
            var dither: Int16 = 1
            if i % 2 == 1 {
                dither = 2
            }
            low[i] &+= (prevHigh &+ high[i] &+ dither) &>> 2
        }

        for i in 0..<8 {
            row[offset + i] = low[i]
            row[offset + 8 + i] = high[i]
        }
    }

    @inline(__always)
    private static func inverseLiftRow16(_ row: inout [Int16], offset: Int) {
        var low = [Int16](repeating: 0, count: 8)
        var high = [Int16](repeating: 0, count: 8)
        for i in 0..<8 {
            low[i] = row[offset + i]
            high[i] = row[offset + 8 + i]
        }

        // Inverse update
        for i in 0..<8 {
            var prevHigh = high[0]
            if 0 < i {
                prevHigh = high[i - 1]
            }
            var dither: Int16 = 1
            if i % 2 == 1 {
                dither = 2
            }
            low[i] &-= (prevHigh &+ high[i] &+ dither) &>> 2
        }

        // Inverse predict
        for i in 0..<8 {
            var nextLow = low[7]
            if i + 1 < 8 {
                nextLow = low[i + 1]
            }
            var dither: Int16 = 0
            if i % 2 == 1 {
                dither = 1
            }
            high[i] &+= (low[i] &+ nextLow &+ dither) &>> 1
        }

        for i in 0..<8 {
            row[offset + (2 * i)] = low[i]
            row[offset + ((2 * i) + 1)] = high[i]
        }
    }

    /// Applies 2D LeGall 5/3 DWT, deadzone quantization with zero-threshold culling, and Inverse DWT to simulate VEVC degradation.
    public static func simulateDWTCompression(
        patch: [Float],
        width: Int,
        height: Int,
        qStep: Int = 128,
        zeroThreshold: Int = 3
    ) -> [Float] {
        let count = width * height
        var intBuffer = [Int16](repeating: 0, count: count)
        for i in 0..<count {
            let clamped = max(0.0, min(255.0, patch[i]))
            intBuffer[i] = Int16(clamped) - 128
        }

        // Apply 2D DWT per 16x16 block
        let blockSize = 16
        if 16 <= width && 16 <= height {
            var by = 0
            while by <= height - blockSize {
                var bx = 0
                while bx <= width - blockSize {
                    // Row DWT
                    for r in 0..<blockSize {
                        let offset = ((by + r) * width) + bx
                        liftRow16(&intBuffer, offset: offset)
                    }

                    // Deadzone quantization & zeroThreshold culling
                    var r = 0
                    while r < blockSize {
                        var c = 0
                        while c < blockSize {
                            let offset = ((by + r) * width) + (bx + c)
                            let coeff = Int(intBuffer[offset])
                            let absCoeff = abs(coeff)
                            if absCoeff <= zeroThreshold {
                                intBuffer[offset] = 0
                            } else {
                                var sign = 1
                                if coeff < 0 {
                                    sign = -1
                                }
                                let quantized = (absCoeff / qStep) * qStep
                                intBuffer[offset] = Int16(sign * quantized)
                            }
                            c &+= 1
                        }
                        r &+= 1
                    }

                    // Row IDWT
                    for r in 0..<blockSize {
                        let offset = ((by + r) * width) + bx
                        inverseLiftRow16(&intBuffer, offset: offset)
                    }

                    bx &+= blockSize
                }
                by &+= blockSize
            }
        }

        var degraded = [Float](repeating: 0.0, count: count)
        for i in 0..<count {
            let val = Float(intBuffer[i]) + 128.0
            degraded[i] = max(0.0, min(255.0, val))
        }

        return degraded
    }

    /// Generates synthetic training patches with diverse spatial frequencies, geometric edges, and ramps.
    public static func generateSyntheticDataset(
        numSamples: Int = 256,
        patchSize: Int = 32,
        seed: UInt64 = 12345
    ) -> [TrainingSample] {
        var rng = SplitMix64(seed: seed)
        var samples: [TrainingSample] = []
        samples.reserveCapacity(numSamples)

        let count = patchSize * patchSize

        for _ in 0..<numSamples {
            var gtPatch = [Float](repeating: 128.0, count: count)

            let patternType = Int(rng.next() % 5)
            switch patternType {
            case 0:
                // Step edge with arbitrary angle and contrast
                let angle = rng.nextFloat(in: 0.0...6.283)
                let cosA = cosf(angle)
                let sinA = sinf(angle)
                let offset = rng.nextFloat(in: -10.0...10.0)
                let val1 = rng.nextFloat(in: 30.0...100.0)
                let val2 = rng.nextFloat(in: 150.0...230.0)

                for y in 0..<patchSize {
                    for x in 0..<patchSize {
                        let fx = Float(x - patchSize / 2)
                        let fy = Float(y - patchSize / 2)
                        let proj = (fx * cosA) + (fy * sinA) + offset
                        var pixelVal = val1
                        if 0.0 <= proj {
                            pixelVal = val2
                        }
                        gtPatch[(y * patchSize) + x] = pixelVal
                    }
                }

            case 1:
                // High frequency sinusoidal grid / texture
                let freqX = rng.nextFloat(in: 0.2...0.8)
                let freqY = rng.nextFloat(in: 0.2...0.8)
                let base = rng.nextFloat(in: 100.0...150.0)
                let amp = rng.nextFloat(in: 20.0...60.0)

                for y in 0..<patchSize {
                    for x in 0..<patchSize {
                        let s = sinf(Float(x) * freqX) * cosf(Float(y) * freqY)
                        gtPatch[(y * patchSize) + x] = base + (amp * s)
                    }
                }

            case 2:
                // Thin bar / line stroke (HUD / text simulation)
                let lineX = Int(rng.next() % UInt64(patchSize - 6)) + 3
                let lineY = Int(rng.next() % UInt64(patchSize - 6)) + 3
                let bgVal = rng.nextFloat(in: 40.0...80.0)
                let strokeVal = rng.nextFloat(in: 200.0...245.0)

                for i in 0..<count { gtPatch[i] = bgVal }
                for y in 0..<patchSize {
                    gtPatch[(y * patchSize) + lineX] = strokeVal
                    gtPatch[(y * patchSize) + lineX + 1] = strokeVal
                }
                for x in 0..<patchSize {
                    gtPatch[(lineY * patchSize) + x] = strokeVal
                }

            case 3:
                // Smooth gradient ramp with high dynamic range
                let startVal = rng.nextFloat(in: 20.0...60.0)
                let endVal = rng.nextFloat(in: 190.0...240.0)
                for y in 0..<patchSize {
                    for x in 0..<patchSize {
                        let t = Float(x + y) / Float(2 * patchSize)
                        gtPatch[(y * patchSize) + x] = startVal + (t * (endVal - startVal))
                    }
                }

            default:
                // Random smooth noise blobs
                let base = rng.nextFloat(in: 80.0...160.0)
                for y in 0..<patchSize {
                    for x in 0..<patchSize {
                        let n = rng.nextFloat(in: -30.0...30.0)
                        gtPatch[(y * patchSize) + x] = max(0.0, min(255.0, base + n))
                    }
                }
            }

            // Simulate DWT LeGall 5/3 degradation
            let qStepChoice = [64, 128, 256, 384][Int(rng.next() % 4)]
            let degraded = simulateDWTCompression(
                patch: gtPatch,
                width: patchSize,
                height: patchSize,
                qStep: qStepChoice,
                zeroThreshold: 3
            )

            // Extract features and residual
            let features = FeatureExtractor.extract4ChFeatures(
                image: degraded,
                width: patchSize,
                height: patchSize
            )

            var targetRes = [Float](repeating: 0.0, count: count)
            for i in 0..<count {
                targetRes[i] = (gtPatch[i] - degraded[i]) / 128.0
            }

            samples.append(TrainingSample(
                inputFeatures: features,
                targetResidual: targetRes,
                width: patchSize,
                height: patchSize
            ))
        }

        return samples
    }

    /// Extracts training patches from an actual Y4M reader stream.
    public static func extractY4MPatches(
        y4mPath: String,
        patchSize: Int = 32,
        maxPatches: Int = 512
    ) throws -> [TrainingSample] {
        guard let handle = FileHandle(forReadingAtPath: y4mPath) else {
            return []
        }
        defer { try? handle.close() }

        let reader = try Y4MReader(fileHandle: handle)
        var samples: [TrainingSample] = []

        while let frame = try reader.readFrame(), samples.count < maxPatches {
            let width = frame.width
            let height = frame.height

            var yFloat = [Float](repeating: 0.0, count: width * height)
            frame.yPlane.withUnsafeBufferPointer { ptr in
                for i in 0..<(width * height) {
                    yFloat[i] = Float(ptr[i])
                }
            }

            var by = 0
            while by <= height - patchSize && samples.count < maxPatches {
                var bx = 0
                while bx <= width - patchSize && samples.count < maxPatches {
                    var gtPatch = [Float](repeating: 0.0, count: patchSize * patchSize)
                    for py in 0..<patchSize {
                        for px in 0..<patchSize {
                            gtPatch[(py * patchSize) + px] = yFloat[((by + py) * width) + (bx + px)]
                        }
                    }

                    let degraded = simulateDWTCompression(
                        patch: gtPatch,
                        width: patchSize,
                        height: patchSize,
                        qStep: 128,
                        zeroThreshold: 3
                    )

                    let features = FeatureExtractor.extract4ChFeatures(
                        image: degraded,
                        width: patchSize,
                        height: patchSize
                    )

                    var targetRes = [Float](repeating: 0.0, count: patchSize * patchSize)
                    for i in 0..<(patchSize * patchSize) {
                        targetRes[i] = (gtPatch[i] - degraded[i]) / 128.0
                    }

                    samples.append(TrainingSample(
                        inputFeatures: features,
                        targetResidual: targetRes,
                        width: patchSize,
                        height: patchSize
                    ))

                    bx &+= patchSize
                }
                by &+= patchSize
            }
        }

        return samples
    }
}
