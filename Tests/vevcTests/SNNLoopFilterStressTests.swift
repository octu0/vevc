import Testing
import Foundation
@testable import vevc

@Suite("SNN Neural Loop Filter (NLF) Empirical Stress & Adversarial Challenge Tests", .serialized)
struct SNNLoopFilterStressTests {

    // MARK: - 1. Non-Standard Image Dimensions & Boundary Edge Cases

    @Test("Guard against empty, zero, negative, and mismatched buffer dimensions")
    func testZeroAndInvalidDimensionGuards() {
        var emptyPlane = [Int16]()
        applySNNNeuralLoopFilter(plane: &emptyPlane, width: 0, height: 0, planeType: .y)
        #expect(emptyPlane.isEmpty)

        var plane1 = [Int16](repeating: 128, count: 100)
        applySNNNeuralLoopFilter(plane: &plane1, width: 0, height: 10, planeType: .y)
        #expect(plane1.count == 100)

        applySNNNeuralLoopFilter(plane: &plane1, width: 10, height: 0, planeType: .y)
        #expect(plane1.count == 100)

        applySNNNeuralLoopFilter(plane: &plane1, width: -10, height: 10, planeType: .y)
        #expect(plane1.count == 100)

        applySNNNeuralLoopFilter(plane: &plane1, width: 10, height: -10, planeType: .y)
        #expect(plane1.count == 100)

        // Mismatched buffer length (plane has 100 elements, but width*height is 64)
        applySNNNeuralLoopFilter(plane: &plane1, width: 8, height: 8, planeType: .y)
        #expect(plane1.count == 100)
    }

    @Test("Stress-test comprehensive non-standard and prime image dimensions")
    func testNonStandardDimensions() {
        let testDimensions: [(width: Int, height: Int)] = [
            (width: 1, height: 1),
            (width: 1, height: 2),
            (width: 2, height: 1),
            (width: 2, height: 2),
            (width: 3, height: 3),
            (width: 7, height: 9),
            (width: 9, height: 7),
            (width: 15, height: 1),
            (width: 1, height: 15),
            (width: 15, height: 15),
            (width: 16, height: 1),
            (width: 1, height: 16),
            (width: 16, height: 2),
            (width: 17, height: 1),
            (width: 17, height: 17),
            (width: 17, height: 19),
            (width: 19, height: 17),
            (width: 31, height: 33),
            (width: 33, height: 31),
            (width: 33, height: 65),
            (width: 47, height: 53),
            (width: 63, height: 63),
            (width: 64, height: 64),
            (width: 65, height: 47),
            (width: 127, height: 127),
            (width: 128, height: 129),
            (width: 255, height: 255),
            (width: 1920, height: 1),
            (width: 1, height: 1080),
            (width: 1920, height: 1080)
        ]

        for dim in testDimensions {
            let width = dim.width
            let height = dim.height
            let count = width * height

            var plane = [Int16](repeating: 0, count: count)
            for y in 0..<height {
                for x in 0..<width {
                    let v = ((x * 11) ^ (y * 23) + ((x + y) * 7)) % 256
                    plane[(y * width) + x] = Int16(v - 128)
                }
            }

            let originalPlane = plane
            applySNNNeuralLoopFilter(plane: &plane, width: width, height: height, planeType: .y)

            #expect(plane.count == count, "Size \(width)x\(height): output count altered")

            // Verify all pixels clamped to [-128, 127] and delta within [-16, 16]
            for i in 0..<count {
                let outVal = plane[i]
                #expect(-128 <= outVal, "Size \(width)x\(height): pixel at \(i) is below -128 (\(outVal))")
                #expect(outVal <= 127, "Size \(width)x\(height): pixel at \(i) exceeds 127 (\(outVal))")

                let delta = outVal - originalPlane[i]
                #expect(-16 <= delta, "Size \(width)x\(height): delta at \(i) is below -16 (\(delta))")
                #expect(delta <= 16, "Size \(width)x\(height): delta at \(i) exceeds 16 (\(delta))")
            }
        }
    }

    @Test("Exhaustive bit-exact equivalence between SIMD16 and Scalar oracle across odd dimensions")
    func testSIMDVsScalarBitExactnessOddDimensions() {
        let oddDimensions: [(width: Int, height: Int)] = [
            (width: 16, height: 16),
            (width: 17, height: 16),
            (width: 16, height: 17),
            (width: 17, height: 19),
            (width: 31, height: 33),
            (width: 33, height: 31),
            (width: 48, height: 49),
            (width: 65, height: 47),
            (width: 97, height: 83),
            (width: 129, height: 65)
        ]

        for dim in oddDimensions {
            let width = dim.width
            let height = dim.height
            let count = width * height

            var source = [Int16](repeating: 0, count: count)
            for y in 0..<height {
                for x in 0..<width {
                    let v = ((x * 37) ^ (y * 43) + (x * y)) % 256
                    source[(y * width) + x] = Int16(v)
                }
            }

            var simdResult = source
            applySNNNeuralLoopFilter(plane: &simdResult, width: width, height: height, planeType: .y)

            var scalarResult = source
            runScalarReferenceOracle(
                plane: &scalarResult,
                width: width,
                height: height,
                blockSize: 32
            )

            var mismatchCount = 0
            for i in 0..<count {
                if simdResult[i] != scalarResult[i] {
                    mismatchCount &+= 1
                }
            }

            #expect(
                mismatchCount == 0,
                "Dimension \(width)x\(height): \(mismatchCount) mismatches between SIMD16 and Scalar Reference"
            )
        }
    }

    // MARK: - 2. Extreme Pixel Inputs & Arithmetic Boundary / Overflow Immunity

    @Test("Extreme pixel inputs: Int16.min, Int16.max, negative values, saturated values")
    func testExtremePixelValuesAndClamping() {
        let width = 64
        let height = 64
        let count = width * height

        let extremeTestCases: [(name: String, generator: (Int, Int) -> Int16)] = [
            (name: "All Int16.min (-32768)", generator: { _, _ in Int16.min }),
            (name: "All Int16.max (+32767)", generator: { _, _ in Int16.max }),
            (name: "Extreme negative (-5000)", generator: { _, _ in -5000 }),
            (name: "Extreme positive (+5000)", generator: { _, _ in 5000 }),
            (name: "Alternating min/max checkerboard", generator: { x, y in
                let isEven = ((x + y) & 1) == 0
                if isEven {
                    return Int16.min
                } else {
                    return Int16.max
                }
            }),
            (name: "Step impulse at center", generator: { x, y in
                if x == 32 && y == 32 {
                    return Int16.max
                } else {
                    return 0
                }
            }),
            (name: "Negative impulse at center", generator: { x, y in
                if x == 32 && y == 32 {
                    return Int16.min
                } else {
                    return 127
                }
            }),
            (name: "High frequency noise [-1000, 1000]", generator: { x, y in
                let modVal = Int16(((x * 101) ^ (y * 307)) % 2001)
                return modVal - 1000
            })
        ]

        for tc in extremeTestCases {
            var plane = [Int16](repeating: 0, count: count)
            for y in 0..<height {
                for x in 0..<width {
                    plane[(y * width) + x] = tc.generator(x, y)
                }
            }

            // Apply filter - must execute without arithmetic trap/overflow crash
            applySNNNeuralLoopFilter(plane: &plane, width: width, height: height, planeType: .y)

            for i in 0..<count {
                let val = plane[i]
                #expect(
                    -128 <= val,
                    "[\(tc.name)] Output pixel at index \(i) was below -128: \(val)"
                )
                #expect(
                    val <= 127,
                    "[\(tc.name)] Output pixel at index \(i) exceeded 127: \(val)"
                )
            }
        }
    }

    @Test("Flat planes of various constants remain stable and within range")
    func testFlatPlanesStability() {
        let width = 64
        let height = 64
        let count = width * height

        let flatValues: [Int16] = [-1000, -128, -64, -1, 0, 1, 64, 127, 256, 1000]

        for flatVal in flatValues {
            var plane = [Int16](repeating: flatVal, count: count)
            applySNNNeuralLoopFilter(plane: &plane, width: width, height: height, planeType: .y)

            for i in 0..<count {
                let val = plane[i]
                #expect(
                    -128 <= val,
                    "FlatVal \(flatVal): Output pixel at \(i) was below -128: \(val)"
                )
                #expect(
                    val <= 127,
                    "FlatVal \(flatVal): Output pixel at \(i) exceeded 127: \(val)"
                )
            }
        }
    }

    // MARK: - 3. Plane Types (.y, .cb, .cr) & Block Size Invariants

    @Test("PlaneType block size invariants: .y is 32, .cb and .cr are 16")
    func testPlaneTypeBlockSizes() {
        #expect(SNNLoopFilter.getBlockSize(for: .y) == 32)
        #expect(SNNLoopFilter.getBlockSize(for: .cb) == 16)
        #expect(SNNLoopFilter.getBlockSize(for: .cr) == 16)
    }

    @Test("Chroma planes .cb and .cr produce identical bit-exact output on identical inputs")
    func testChromaSymmetryCbCr() {
        let testSizes = [
            (width: 16, height: 16),
            (width: 32, height: 32),
            (width: 64, height: 64),
            (width: 128, height: 64),
            (width: 960, height: 540)
        ]

        for size in testSizes {
            let width = size.width
            let height = size.height
            let count = width * height

            var inputPlane = [Int16](repeating: 0, count: count)
            for y in 0..<height {
                for x in 0..<width {
                    let v = ((x * 19) ^ (y * 31) + (x * 3)) % 256
                    inputPlane[(y * width) + x] = Int16(v)
                }
            }

            var cbResult = inputPlane
            applySNNNeuralLoopFilter(plane: &cbResult, width: width, height: height, planeType: .cb)

            var crResult = inputPlane
            applySNNNeuralLoopFilter(plane: &crResult, width: width, height: height, planeType: .cr)

            var diffCount = 0
            for i in 0..<count {
                if cbResult[i] != crResult[i] {
                    diffCount &+= 1
                }
            }

            #expect(
                diffCount == 0,
                "Size \(width)x\(height): Cb and Cr deviated by \(diffCount) pixels"
            )
        }
    }

    @Test("PlaneType .y vs .cb/.cr produce distinct outputs due to 32 vs 16 block boundary distance features")
    func testLumaVsChromaBlockDistanceDistinction() {
        let width = 64
        let height = 64
        let count = width * height

        var inputPlane = [Int16](repeating: 0, count: count)
        for y in 0..<height {
            for x in 0..<width {
                let v = ((x * 7) + (y * 11)) % 256
                inputPlane[(y * width) + x] = Int16(v)
            }
        }

        var yResult = inputPlane
        applySNNNeuralLoopFilter(plane: &yResult, width: width, height: height, planeType: .y)

        var cbResult = inputPlane
        applySNNNeuralLoopFilter(plane: &cbResult, width: width, height: height, planeType: .cb)

        var hasDifference = false
        for i in 0..<count {
            if yResult[i] != cbResult[i] {
                hasDifference = true
                break
            }
        }

        #expect(
            hasDifference,
            "Y (blockSize 32) and Cb (blockSize 16) should produce distinct filtering due to feature channel 3"
        )
    }

    // MARK: - 4. Multi-Threaded Determinism & Sequential Consistency Stress

    @Test("Sequential consistency and determinism across 50 iterations on 1080p frame")
    func testConsistencyDeterminism1080p() {
        let width = 1920
        let height = 1080
        let count = width * height

        var source = [Int16](repeating: 0, count: count)
        for y in 0..<height {
            for x in 0..<width {
                let v = ((x * 13) ^ (y * 29) + (x + y * 7)) % 256
                source[(y * width) + x] = Int16(v)
            }
        }

        var referenceRun = source
        applySNNNeuralLoopFilter(plane: &referenceRun, width: width, height: height, planeType: .y)

        // Run multiple times in sequence to detect any internal multi-threading race conditions or nondeterminism
        for runIndex in 1...20 {
            var candidateRun = source
            applySNNNeuralLoopFilter(plane: &candidateRun, width: width, height: height, planeType: .y)

            var mismatchCount = 0
            for i in 0..<count {
                if referenceRun[i] != candidateRun[i] {
                    mismatchCount &+= 1
                }
            }

            #expect(
                mismatchCount == 0,
                "Run \(runIndex): detected \(mismatchCount) non-deterministic pixel mismatches on 1080p frame"
            )
        }
    }

    // MARK: - Helper: Scalar Reference Oracle

    private func runScalarReferenceOracle(
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

    // MARK: - 5. Additional Pattern Stress Tests (Ramps, Noise, Checkerboard, Constants)

    @Test("SIMD16 vs Scalar equivalence on extreme constant values (0..255)")
    func testConstantValuesEquivalence() {
        let testSizes = [
            (w: 16, h: 16),
            (w: 32, h: 32),
            (w: 64, h: 64),
            (w: 128, h: 64),
            (w: 48, h: 32),
            (w: 33, h: 33),
            (w: 17, h: 17)
        ]

        let constantValues: [Int16] = [-128, -127, -64, 0, 1, 64, 126, 127]

        for size in testSizes {
            for val in constantValues {
                let width = size.w
                let height = size.h
                let count = width * height

                let src = [Int16](repeating: val, count: count)

                var simdOut = src
                applySNNNeuralLoopFilter(plane: &simdOut, width: width, height: height, planeType: .y)

                var scalarOut = src
                runScalarReferenceOracle(plane: &scalarOut, width: width, height: height, blockSize: 32)

                var diffCount = 0
                for i in 0..<count {
                    if simdOut[i] != scalarOut[i] {
                        diffCount &+= 1
                    }
                }
                #expect(diffCount == 0, "Constant value \(val) size \(width)x\(height): \(diffCount) pixel mismatches between SIMD and scalar")
            }
        }
    }

    @Test("SIMD16 vs Scalar equivalence on gradient ramps")
    func testGradientRampsEquivalence() {
        let testSizes = [
            (w: 32, h: 32),
            (w: 64, h: 64),
            (w: 128, h: 96),
            (w: 47, h: 31),
            (w: 192, h: 128)
        ]

        for size in testSizes {
            let width = size.w
            let height = size.h
            let count = width * height

            // 1. Horizontal ramp
            var hRamp = [Int16](repeating: 0, count: count)
            for y in 0..<height {
                for x in 0..<width {
                    hRamp[(y * width) + x] = Int16(((x * 255) / max(1, width - 1)) - 128)
                }
            }
            verifyRamp(src: hRamp, width: width, height: height, pattern: "Horizontal Ramp")

            // 2. Vertical ramp
            var vRamp = [Int16](repeating: 0, count: count)
            for y in 0..<height {
                for x in 0..<width {
                    vRamp[(y * width) + x] = Int16(((y * 255) / max(1, height - 1)) - 128)
                }
            }
            verifyRamp(src: vRamp, width: width, height: height, pattern: "Vertical Ramp")

            // 3. Diagonal ramp
            var dRamp = [Int16](repeating: 0, count: count)
            for y in 0..<height {
                for x in 0..<width {
                    dRamp[(y * width) + x] = Int16((((x + y) * 255) / max(1, width + height - 2)) - 128)
                }
            }
            verifyRamp(src: dRamp, width: width, height: height, pattern: "Diagonal Ramp")
        }
    }

    @Test("SIMD16 vs Scalar equivalence on high-frequency checkerboard & impulse spikes")
    func testHighFrequencyAndImpulsesEquivalence() {
        let testSizes = [
            (w: 17, h: 17),
            (w: 19, h: 23),
            (w: 32, h: 32),
            (w: 33, h: 31),
            (w: 64, h: 64),
            (w: 80, h: 48),
            (w: 128, h: 128)
        ]

        for size in testSizes {
            let width = size.w
            let height = size.h
            let count = width * height

            // 1. 1x1 Checkerboard
            var cb1x1 = [Int16](repeating: 0, count: count)
            for y in 0..<height {
                for x in 0..<width {
                    let isEven = ((x + y) % 2) == 0
                    if isEven {
                        cb1x1[(y * width) + x] = 127
                    } else {
                        cb1x1[(y * width) + x] = -128
                    }
                }
            }
            verifyRamp(src: cb1x1, width: width, height: height, pattern: "1x1 Checkerboard")

            // 2. 8x8 Checkerboard
            var cb8x8 = [Int16](repeating: 0, count: count)
            for y in 0..<height {
                for x in 0..<width {
                    let bx = (x / 8) % 2
                    let by = (y / 8) % 2
                    if bx == by {
                        cb8x8[(y * width) + x] = 72
                    } else {
                        cb8x8[(y * width) + x] = -78
                    }
                }
            }
            verifyRamp(src: cb8x8, width: width, height: height, pattern: "8x8 Checkerboard")

            // 3. Single impulse delta spikes
            var impulse = [Int16](repeating: 0, count: count)
            impulse[0] = 127
            impulse[width - 1] = -128
            impulse[(height - 1) * width] = 127
            impulse[count - 1] = -128
            if 16 < width && 16 < height {
                impulse[(15 * width) + 15] = 127
                impulse[(16 * width) + 16] = -128
            }
            if 32 < width && 32 < height {
                impulse[(31 * width) + 31] = 127
                impulse[(32 * width) + 32] = -128
            }
            verifyRamp(src: impulse, width: width, height: height, pattern: "Impulse Spikes")
        }
    }

    private func verifyRamp(src: [Int16], width: Int, height: Int, pattern: String) {
        let count = width * height
        var simdOut = src
        applySNNNeuralLoopFilter(plane: &simdOut, width: width, height: height, planeType: .y)

        var scalarOut = src
        runScalarReferenceOracle(plane: &scalarOut, width: width, height: height, blockSize: 32)

        var diffCount = 0
        var maxDiff: Int16 = 0
        for i in 0..<count {
            let diff = abs(simdOut[i] - scalarOut[i])
            if maxDiff < diff {
                maxDiff = diff
            }
            if 0 < diff {
                diffCount &+= 1
            }
        }

        #expect(diffCount == 0, "\(pattern) size \(width)x\(height): \(diffCount) pixel mismatches (maxDiff: \(maxDiff))")
    }
}

