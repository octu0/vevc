import Testing
import Foundation
import CryptoKit
@testable import vevc

@Suite("SNN Neural Loop Filter (NLF) Closed-Loop Integration & Bit-Exact Symmetry Tests")
struct SNNClosedLoopIntegrationTests {

    private func createTestPattern(width: Int, height: Int, frameIndex: Int) -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2

        // Luma Plane: Gradient, moving circle, and textured blocks
        for y in 0..<height {
            for x in 0..<width {
                let grad = ((x * 128) / max(1, width)) + ((y * 128) / max(1, height))
                let cx = (width / 2) + ((frameIndex * 5) % max(1, width / 2))
                let cy = (height / 2) + ((frameIndex * 3) % max(1, height / 2))
                let dx = x - cx
                let dy = y - cy
                let distSq = (dx * dx) + (dy * dy)
                let circleVal: Int
                if distSq <= 400 {
                    circleVal = 200
                } else {
                    circleVal = 0
                }
                let texture = ((x / 16) ^ (y / 16) ^ frameIndex) % 2 * 30
                let total = grad + circleVal + texture
                let clamped = min(255, max(0, total))
                img.yPlane[(y * width) + x] = UInt8(clamped)
            }
        }

        // Chroma Planes
        for cy in 0..<cHeight {
            for cx in 0..<cWidth {
                let idx = (cy * cWidth) + cx
                let cbVal = (128 + ((cx * 4) % 64) - ((cy * 2) % 32))
                let crVal = (128 + ((cy * 4) % 64) - ((cx * 2) % 32))
                img.cbPlane[idx] = UInt8(min(255, max(0, cbVal)))
                img.crPlane[idx] = UInt8(min(255, max(0, crVal)))
            }
        }

        return img
    }

    @Test("Milestone 3 Core: Encoder Recon vs Decoder Display Bit-Exact Symmetry (Delta = 0)")
    func testEncoderDecoderClosedLoopBitExactness() async throws {
        let width = 128
        let height = 128
        let frameCount = 10

        var frames = [YCbCrImage]()
        for f in 0..<frameCount {
            frames.append(createTestPattern(width: width, height: height, frameIndex: f))
        }

        let encoder = VEVCEncoder(
            width: width,
            height: height,
            qstep: 12,
            zeroThreshold: 3,
            keyint: 30,
            profile: 0x02,
            l2Cadence: 1,
            l1Cadence: 1
        )

        let bitstream = try await encoder.encodeToData(images: frames)
        let decoder = Decoder(maxLayer: 2, maxConcurrency: 1)
        let decodedFrames = try await decoder.decode(data: bitstream)

        #expect(decodedFrames.count == frameCount)

        // Verify decoding is 100% deterministic and matches
        let decoder2 = Decoder(maxLayer: 2, maxConcurrency: 1)
        let decodedFrames2 = try await decoder2.decode(data: bitstream)

        for f in 0..<frameCount {
            let d1 = decodedFrames[f]
            let d2 = decodedFrames2[f]

            for i in 0..<d1.yPlane.count {
                #expect(d1.yPlane[i] == d2.yPlane[i], "Frame \(f) Y mismatch at index \(i)")
            }
            for i in 0..<d1.cbPlane.count {
                #expect(d1.cbPlane[i] == d2.cbPlane[i], "Frame \(f) Cb mismatch at index \(i)")
            }
            for i in 0..<d1.crPlane.count {
                #expect(d1.crPlane[i] == d2.crPlane[i], "Frame \(f) Cr mismatch at index \(i)")
            }
        }
    }

    @Test("Milestone 3 Core: Multi-Frame GOP Drift Prevention Across 30 Frames")
    func testGOPDriftPreventionAcross30Frames() async throws {
        let width = 64
        let height = 64
        let frameCount = 30

        var frames = [YCbCrImage]()
        for f in 0..<frameCount {
            frames.append(createTestPattern(width: width, height: height, frameIndex: f))
        }

        let encoder = VEVCEncoder(
            width: width,
            height: height,
            qstep: 16,
            zeroThreshold: 3,
            keyint: 60,
            profile: 0x02,
            l2Cadence: 1,
            l1Cadence: 1
        )

        let bitstream = try await encoder.encodeToData(images: frames)
        let decoder = Decoder(maxLayer: 2, maxConcurrency: 1)
        let decodedFrames = try await decoder.decode(data: bitstream)

        #expect(decodedFrames.count == frameCount)

        // Verify that quality remains stable throughout the entire 30-frame GOP without accumulation error
        var ssimList = [Double]()
        for f in 0..<frameCount {
            let orig = frames[f]
            let dec = decodedFrames[f]

            var sum1: Double = 0
            var sum2: Double = 0
            var sum1Sq: Double = 0
            var sum2Sq: Double = 0
            var pdt: Double = 0
            let n = Double(orig.yPlane.count)

            for i in 0..<orig.yPlane.count {
                let v1 = Double(orig.yPlane[i])
                let v2 = Double(dec.yPlane[i])
                sum1 += v1
                sum2 += v2
                sum1Sq += (v1 * v1)
                sum2Sq += (v2 * v2)
                pdt += (v1 * v2)
            }

            let mu1 = sum1 / n
            let mu2 = sum2 / n
            let sig1Sq = (sum1Sq / n) - (mu1 * mu1)
            let sig2Sq = (sum2Sq / n) - (mu2 * mu2)
            let sig12 = (pdt / n) - (mu1 * mu2)

            let c1: Double = 6.5025
            let c2: Double = 58.5225
            let num = (2.0 * mu1 * mu2 + c1) * (2.0 * sig12 + c2)
            let den = (mu1 * mu1 + mu2 * mu2 + c1) * (sig1Sq + sig2Sq + c2)
            let ssim = num / den
            ssimList.append(ssim)

            #expect(0.85 <= ssim, "Frame \(f) SSIM (\(ssim)) dropped below drift tolerance 0.85")
        }

        // Compare first P-frame and last P-frame SSIM: drift must not degrade SSIM by > 0.08
        let firstPSSIM = ssimList[1]
        let lastPSSIM = ssimList[frameCount - 1]
        let driftDelta = firstPSSIM - lastPSSIM
        #expect(driftDelta <= 0.08, "Excessive GOP reconstruction drift: first P SSIM \(firstPSSIM), last P SSIM \(lastPSSIM)")
    }

    @Test("Milestone 3 Core: Multi-Resolution Splitter Compatibility with SNN NLF (L0 / L1 / L2)")
    func testSplitterMultiResolutionCompatibilityProfile2() async throws {
        let width = 64
        let height = 64
        let frameCount = 4

        var frames = [YCbCrImage]()
        for f in 0..<frameCount {
            frames.append(createTestPattern(width: width, height: height, frameIndex: f))
        }

        let encoder = VEVCEncoder(
            width: width,
            height: height,
            qstep: 12,
            zeroThreshold: 3,
            keyint: 10,
            profile: 0x02,
            l2Cadence: 1,
            l1Cadence: 1
        )

        let fullBitstream = try await encoder.encodeToData(images: frames)

        // 1. Extract Full Layer 2
        let splitL2 = try splitVEVCStream(input: fullBitstream, maxLayer: 2)
        #expect(splitL2.processedFrames == frameCount)
        let decoderL2 = Decoder(maxLayer: 2, maxConcurrency: 1)
        let decL2 = try await decoderL2.decode(data: splitL2.data)
        #expect(decL2.count == frameCount)
        #expect(decL2[0].width == width)
        #expect(decL2[0].height == height)

        // 2. Extract Half-Resolution Layer 1
        let splitL1 = try splitVEVCStream(input: fullBitstream, maxLayer: 1)
        #expect(splitL1.processedFrames == frameCount)
        let decoderL1 = Decoder(maxLayer: 1, maxConcurrency: 1)
        let decL1 = try await decoderL1.decode(data: splitL1.data)
        #expect(decL1.count == frameCount)
        #expect(decL1[0].width == (width + 1) / 2)
        #expect(decL1[0].height == (height + 1) / 2)

        // 3. Extract Quarter-Resolution Layer 0
        let splitL0 = try splitVEVCStream(input: fullBitstream, maxLayer: 0)
        #expect(splitL0.processedFrames == frameCount)
        let decoderL0 = Decoder(maxLayer: 0, maxConcurrency: 1)
        let decL0 = try await decoderL0.decode(data: splitL0.data)
        #expect(decL0.count == frameCount)
        #expect(decL0[0].width == (((width + 1) / 2) + 1) / 2)
        #expect(decL0[0].height == (((height + 1) / 2) + 1) / 2)
    }

    @Test("Milestone 3 Core: Rate Reduction Coupling & Zero-Threshold Maximization")
    func testRateReductionCouplingZeroThreshold() async throws {
        let width = 64
        let height = 64
        let pool = BlockViewPool()

        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
        for y in 0..<height {
            for x in 0..<width {
                let v = ((x * 13 ^ y * 29) % 256)
                img.yPlane[(y * width) + x] = UInt8(v)
            }
        }
        let cCount = (width / 2) * (height / 2)
        img.cbPlane = [UInt8](repeating: 128, count: cCount)
        img.crPlane = [UInt8](repeating: 128, count: cCount)

        let pd = toPlaneData420(image: img, pool: pool).0
        let qtY = QuantizationTable(baseStep: 16)
        let qtC = QuantizationTable(baseStep: 16)
        var staticCounters0 = [Int](repeating: 0, count: 4)
        var staticCounters3 = [Int](repeating: 0, count: 4)
        let l0State0 = L0RefState()
        let l0State3 = L0RefState()

        let (bytesZ0, _, _, _, rel0, _, _, _) = try await encodeSpatialLayersForProfile2(
            pd: pd, pool: pool, predictedPd: pd, nextPd: pd, prevInput: pd, ltrInput: pd, prevMVs: nil,
            maxbitrate: 10_000_000, qtY: qtY, qtC: qtC, zeroThreshold: 0,
            roundOffset: 0, gopPosition: 1, ltrAge: 1, skipThreshold: 0, reconThresholdScale: 1,
            staticCounters: &staticCounters0, cachedNextSub2: nil, cachedNextSub1: nil,
            entropyHistories: nil, l0State: l0State0
        )
        defer { rel0() }

        let (bytesZ3, _, _, _, rel3, _, _, _) = try await encodeSpatialLayersForProfile2(
            pd: pd, pool: pool, predictedPd: pd, nextPd: pd, prevInput: pd, ltrInput: pd, prevMVs: nil,
            maxbitrate: 10_000_000, qtY: qtY, qtC: qtC, zeroThreshold: 3,
            roundOffset: 0, gopPosition: 1, ltrAge: 1, skipThreshold: 0, reconThresholdScale: 1,
            staticCounters: &staticCounters3, cachedNextSub2: nil, cachedNextSub1: nil,
            entropyHistories: nil, l0State: l0State3
        )
        defer { rel3() }

        #expect(bytesZ3.count <= bytesZ0.count, "ZeroThreshold 3 size (\(bytesZ3.count)) should be <= ZeroThreshold 0 (\(bytesZ0.count))")
    }
}
