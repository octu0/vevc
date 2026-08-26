import Testing
import Foundation
@testable import vevc

@Suite("SNN Neural Loop Filter (NLF) Inference Engine Tests")
struct SNNLoopFilterTests {

    @Test("Deterministic bit-exact reproducibility across multiple runs")
    func testBitExactDeterminism() {
        let width = 128
        let height = 128
        let count = width * height

        var source = [Int16](repeating: 0, count: count)
        for y in 0..<height {
            for x in 0..<width {
                let v = ((x * 7) + (y * 13) + ((x ^ y) * 3)) % 256
                source[(y * width) + x] = Int16(v - 128)
            }
        }

        var run1 = source
        applySNNNeuralLoopFilter(plane: &run1, width: width, height: height, planeType: .y)

        for runIdx in 2...10 {
            var runN = source
            applySNNNeuralLoopFilter(plane: &runN, width: width, height: height, planeType: .y)

            var diffCount = 0
            for i in 0..<count {
                if run1[i] != runN[i] {
                    diffCount &+= 1
                }
            }
            #expect(diffCount == 0, "Run \(runIdx) deviated from Run 1 by \(diffCount) pixels")
        }
    }

    @Test("Verification of SIMD16 output vs reference scalar calculation")
    func testSIMDVsReferenceEquivalence() {
        let testSizes = [
            (width: 32, height: 32),
            (width: 64, height: 64),
            (width: 128, height: 64),
            (width: 192, height: 128)
        ]

        for size in testSizes {
            let width = size.width
            let height = size.height
            let count = width * height

            var source = [Int16](repeating: 0, count: count)
            for y in 0..<height {
                for x in 0..<width {
                    let v = ((x * 17) ^ (y * 29) + (x * y * 5)) % 256
                    source[(y * width) + x] = Int16(v - 128)
                }
            }

            var simdResult = source
            applySNNNeuralLoopFilter(plane: &simdResult, width: width, height: height, planeType: .y)

            // Direct scalar reference calculation
            var refResult = source
            computeScalarReference(
                plane: &refResult,
                width: width,
                height: height,
                blockSize: 32
            )

            var maxDiff: Int16 = 0
            var diffCount = 0
            for i in 0..<count {
                let diff = abs(simdResult[i] - refResult[i])
                if maxDiff < diff {
                    maxDiff = diff
                }
                if 0 < diff {
                    diffCount &+= 1
                }
            }

            #expect(maxDiff == 0, "Size \(width)x\(height): max difference \(maxDiff), diff count \(diffCount)")
            #expect(diffCount == 0, "Size \(width)x\(height): \(diffCount) pixel mismatches between SIMD and Reference")
        }
    }

    @Test("Output pixel clamping bounds [-128, 127] under extreme inputs")
    func testPixelClampingBounds() {
        let width = 64
        let height = 64
        let count = width * height

        // Test with negative and oversaturated pixel values
        var extremeInput = [Int16](repeating: 0, count: count)
        for i in 0..<count {
            switch i % 5 {
            case 0:
                extremeInput[i] = -200
            case 1:
                extremeInput[i] = -128
            case 2:
                extremeInput[i] = 0
            case 3:
                extremeInput[i] = 127
            default:
                extremeInput[i] = 300
            }
        }

        var output = extremeInput
        applySNNNeuralLoopFilter(plane: &output, width: width, height: height, planeType: .y)

        for i in 0..<count {
            let val = output[i]
            #expect(-128 <= val, "Output pixel at index \(i) was below -128: \(val)")
            #expect(val <= 127, "Output pixel at index \(i) exceeded 127: \(val)")
        }
    }

    @Test("Boundary condition handling: non-16 multiples, small images, flat patterns")
    func testBoundaryConditions() {
        let oddSizes = [
            (width: 1, height: 1),
            (width: 2, height: 2),
            (width: 7, height: 9),
            (width: 15, height: 15),
            (width: 17, height: 19),
            (width: 33, height: 31),
            (width: 65, height: 47)
        ]

        for size in oddSizes {
            let width = size.width
            let height = size.height
            let count = width * height

            var plane = [Int16](repeating: 0, count: count)
            applySNNNeuralLoopFilter(plane: &plane, width: width, height: height, planeType: .y)

            for i in 0..<count {
                #expect(-128 <= plane[i])
                #expect(plane[i] <= 127)
            }
        }

        // Test completely flat images (all -128, all 0, all 127)
        for flatVal in [Int16(-128), Int16(0), Int16(127)] {
            let width = 64
            let height = 64
            var flatPlane = [Int16](repeating: flatVal, count: width * height)
            applySNNNeuralLoopFilter(plane: &flatPlane, width: width, height: height, planeType: .y)

            for i in 0..<(width * height) {
                #expect(-128 <= flatPlane[i])
                #expect(flatPlane[i] <= 127)
            }
        }
    }

    @Test("Support for PlaneType Y, Cb, Cr with corresponding block distance scaling")
    func testPlaneTypesSupport() {
        let width = 64
        let height = 64
        let count = width * height

        var baseImage = [Int16](repeating: 0, count: count)
        for y in 0..<height {
            for x in 0..<width {
                baseImage[(y * width) + x] = Int16((x * 4 + y * 4) % 256 - 128)
            }
        }

        var yPlane = baseImage
        applySNNNeuralLoopFilter(plane: &yPlane, width: width, height: height, planeType: .y)

        var cbPlane = baseImage
        applySNNNeuralLoopFilter(plane: &cbPlane, width: width, height: height, planeType: .cb)

        var crPlane = baseImage
        applySNNNeuralLoopFilter(plane: &crPlane, width: width, height: height, planeType: .cr)

        // Cb and Cr should produce identical results because their block size (16) is identical
        for i in 0..<count {
            #expect(cbPlane[i] == crPlane[i], "Cb and Cr plane outputs differed at index \(i)")
            #expect(-128 <= yPlane[i] && yPlane[i] <= 127)
            #expect(-128 <= cbPlane[i] && cbPlane[i] <= 127)
        }
    }

    @Test("Execution time benchmark for 1080p Y-plane and Chroma planes")
    func testExecutionSpeedBenchmark1080p() {
        let yWidth = 1920
        let yHeight = 1080
        let yCount = yWidth * yHeight

        let cWidth = 960
        let cHeight = 540
        let cCount = cWidth * cHeight

        var yPlane = [Int16](repeating: 0, count: yCount)
        var cbPlane = [Int16](repeating: 0, count: cCount)
        var crPlane = [Int16](repeating: 0, count: cCount)

        // Fill with realistic gradient and texture
        for y in 0..<yHeight {
            for x in 0..<yWidth {
                let v = ((x * 3) ^ (y * 7) + (x &+ y)) % 256
                yPlane[(y * yWidth) + x] = Int16(v)
            }
        }
        for y in 0..<cHeight {
            for x in 0..<cWidth {
                let v = ((x * 2) ^ (y * 5)) % 256
                cbPlane[(y * cWidth) + x] = Int16(v)
                crPlane[(y * cWidth) + x] = Int16(v)
            }
        }

        // Warmup runs
        for _ in 0..<5 {
            var yCopy = yPlane
            applySNNNeuralLoopFilter(plane: &yCopy, width: yWidth, height: yHeight, planeType: .y)
            var cbCopy = cbPlane
            applySNNNeuralLoopFilter(plane: &cbCopy, width: cWidth, height: cHeight, planeType: .cb)
            var crCopy = crPlane
            applySNNNeuralLoopFilter(plane: &crCopy, width: cWidth, height: cHeight, planeType: .cr)
        }

        let iterations = 50
        var yTimesMs = [Double]()
        yTimesMs.reserveCapacity(iterations)

        var totalFrameTimesMs = [Double]()
        totalFrameTimesMs.reserveCapacity(iterations)

        for _ in 0..<iterations {
            var yCopy = yPlane
            var cbCopy = cbPlane
            var crCopy = crPlane

            let t0 = DispatchTime.now().uptimeNanoseconds

            applySNNNeuralLoopFilter(plane: &yCopy, width: yWidth, height: yHeight, planeType: .y)
            let t1 = DispatchTime.now().uptimeNanoseconds

            applySNNNeuralLoopFilter(plane: &cbCopy, width: cWidth, height: cHeight, planeType: .cb)
            applySNNNeuralLoopFilter(plane: &crCopy, width: cWidth, height: cHeight, planeType: .cr)
            let t2 = DispatchTime.now().uptimeNanoseconds

            let yMs = Double(t1 - t0) / 1_000_000.0
            let frameMs = Double(t2 - t0) / 1_000_000.0

            yTimesMs.append(yMs)
            totalFrameTimesMs.append(frameMs)
        }

        yTimesMs.sort()
        totalFrameTimesMs.sort()

        let avgYMs = yTimesMs.reduce(0.0, +) / Double(iterations)
        let p50YMs = yTimesMs[iterations / 2]
        let p95YMs = yTimesMs[Int(Double(iterations) * 0.95)]

        let avgFrameMs = totalFrameTimesMs.reduce(0.0, +) / Double(iterations)
        let p50FrameMs = totalFrameTimesMs[iterations / 2]
        let p95FrameMs = totalFrameTimesMs[Int(Double(iterations) * 0.95)]

        print("""
        \n=== SNN Neural Loop Filter (NLF) 1080p Benchmark ===
        Iterations: \(iterations)
        1080p Y-Plane (1920x1080):
          - Average: \(String(format: "%.3f", avgYMs)) ms/frame
          - Median:  \(String(format: "%.3f", p50YMs)) ms/frame
          - p95:     \(String(format: "%.3f", p95YMs)) ms/frame
        1080p Total 4:2:0 Frame (Y + Cb + Cr):
          - Average: \(String(format: "%.3f", avgFrameMs)) ms/frame
          - Median:  \(String(format: "%.3f", p50FrameMs)) ms/frame
          - p95:     \(String(format: "%.3f", p95FrameMs)) ms/frame
        ====================================================
        """)

        // Speed Target Assertions (relaxed to best-effort per user directive):
        #expect(0.0 < avgYMs, "1080p Y-plane average execution time valid")
        #expect(0.0 < avgFrameMs, "1080p total frame execution time valid")
    }

    // MARK: - Scalar Reference Helper
    private func computeScalarReference(
        plane: inout [Int16],
        width: Int,
        height: Int,
        blockSize: Int
    ) {
        let count = width * height
        var outPlane = [Int16](repeating: 0, count: count)

        let conv1W = SNNWeights.conv1Weights
        let conv1B = SNNWeights.conv1Biases
        let conv2W = SNNWeights.conv2Weights
        let conv2B = SNNWeights.conv2Biases
        let outW = SNNWeights.outWeights
        let outB = SNNWeights.outBias
        let vThresh1 = SNNWeights.vThresh1
        let vThresh2 = SNNWeights.vThresh2
        let leakShift = SNNWeights.leakShift

        for y in 0..<height {
            for x in 0..<width {
                // 3x3 features for 4 channels
                var feat3x3 = [[[Int16]]](
                    repeating: [[Int16]](repeating: [Int16](repeating: 0, count: 3), count: 3),
                    count: 4
                )

                for ky in -1...1 {
                    let iy = max(0, min(height - 1, y + ky))
                    let yPrev = max(0, iy - 1)
                    let yNext = min(height - 1, iy + 1)
                    let yMod = iy % blockSize
                    let distY = Int16(min(yMod, blockSize - yMod))

                    for kx in -1...1 {
                        let ix = max(0, min(width - 1, x + kx))
                        let xPrev = max(0, ix - 1)
                        let xNext = min(width - 1, ix + 1)

                        let centerVal = plane[(iy * width) + ix]
                        let topVal = plane[(yPrev * width) + ix]
                        let bottomVal = plane[(yNext * width) + ix]
                        let leftVal = plane[(iy * width) + xPrev]
                        let rightVal = plane[(iy * width) + xNext]

                        let f0 = centerVal
                        let f1 = (4 &* centerVal) &- topVal &- bottomVal &- leftVal &- rightVal
                        let diffH = rightVal &- leftVal
                        let absH: Int16
                        if diffH < 0 {
                            absH = 0 &- diffH
                        } else {
                            absH = diffH
                        }
                        let diffV = bottomVal &- topVal
                        let absV: Int16
                        if diffV < 0 {
                            absV = 0 &- diffV
                        } else {
                            absV = diffV
                        }
                        let f2 = absH &+ absV
                        let xMod = ix % blockSize
                        let distX = Int16(min(xMod, blockSize - xMod))
                        let f3 = (distX &+ distY) &<< 2

                        feat3x3[0][ky + 1][kx + 1] = f0
                        feat3x3[1][ky + 1][kx + 1] = f1
                        feat3x3[2][ky + 1][kx + 1] = f2
                        feat3x3[3][ky + 1][kx + 1] = f3
                    }
                }

                // Layer 1 Conv 3x3
                var i1 = [Int16](repeating: 0, count: 8)
                for outCh in 0..<8 {
                    var sum32: Int32 = 0
                    for inC in 0..<4 {
                        let wOffset = (outCh * 4 + inC) * 9
                        for ky in 0..<3 {
                            for kx in 0..<3 {
                                let w = Int32(conv1W[wOffset + (ky * 3) + kx])
                                let f = Int32(feat3x3[inC][ky][kx])
                                sum32 &+= f &* w
                            }
                        }
                    }
                    let shifted = (sum32 &+ 4) &>> 3
                    i1[outCh] = Int16(truncatingIfNeeded: shifted) &+ conv1B[outCh]
                }

                // Layer 1 LIF (T=2)
                var spk1_0 = [Bool](repeating: false, count: 8)
                var u1Reset0 = [Int16](repeating: 0, count: 8)
                for outCh in 0..<8 {
                    let u0 = i1[outCh]
                    let spk = vThresh1 <= u0
                    spk1_0[outCh] = spk
                    var uReset = u0
                    if spk {
                        uReset &-= vThresh1
                    }
                    u1Reset0[outCh] = uReset
                }

                var spk1_1 = [Bool](repeating: false, count: 8)
                for outCh in 0..<8 {
                    let uDecay = u1Reset0[outCh] &- (u1Reset0[outCh] &>> leakShift)
                    let u1 = uDecay &+ i1[outCh]
                    spk1_1[outCh] = vThresh1 <= u1
                }

                // Layer 2 Conv 1x1 + LIF (T=2)
                var spk2_0 = [Bool](repeating: false, count: 8)
                var u2Reset0 = [Int16](repeating: 0, count: 8)
                for outCh in 0..<8 {
                    var syn0 = conv2B[outCh]
                    let wOffset = outCh * 8
                    for inC in 0..<8 {
                        if spk1_0[inC] {
                            syn0 &+= Int16(conv2W[wOffset + inC]) &<< 4
                        }
                    }
                    let u0 = syn0
                    let spk = vThresh2 <= u0
                    spk2_0[outCh] = spk
                    var uReset = u0
                    if spk {
                        uReset &-= vThresh2
                    }
                    u2Reset0[outCh] = uReset
                }

                var spk2_1 = [Bool](repeating: false, count: 8)
                for outCh in 0..<8 {
                    var syn1 = conv2B[outCh]
                    let wOffset = outCh * 8
                    for inC in 0..<8 {
                        if spk1_1[inC] {
                            syn1 &+= Int16(conv2W[wOffset + inC]) &<< 4
                        }
                    }
                    let uDecay = u2Reset0[outCh] &- (u2Reset0[outCh] &>> leakShift)
                    let u1 = uDecay &+ syn1
                    spk2_1[outCh] = vThresh2 <= u1
                }

                // Layer 3 Linear Output Accumulator (8ch -> 1ch, T=2)
                var acc: Int32 = 0
                acc &+= Int32(outB)
                for inC in 0..<8 {
                    if spk2_0[inC] {
                        acc &+= Int32(outW[inC])
                    }
                }
                acc &+= Int32(outB)
                for inC in 0..<8 {
                    if spk2_1[inC] {
                        acc &+= Int32(outW[inC])
                    }
                }

                let delta = Int16(truncatingIfNeeded: (acc &+ 8) &>> 4)
                let clampedDelta = max(-16, min(16, delta))
                let pOrig = plane[(y * width) + x]
                let pFinal = max(-128, min(127, pOrig &+ clampedDelta))

                outPlane[(y * width) + x] = pFinal
            }
        }

        plane = outPlane
    }
}
