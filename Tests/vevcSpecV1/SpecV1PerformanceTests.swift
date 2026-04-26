import XCTest
@testable import vevc

final class SpecV1PerformanceTests: XCTestCase {
    
    let y4mPath = "Tests/vevcSpecV1/testdata/spec_1080p_60f.y4m"
    var allFrames: [YCbCrImage] = []
    
    override func setUp() async throws {
        if allFrames.isEmpty {
            let fileHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: y4mPath))
            let reader = try Y4MReader(fileHandle: fileHandle)
            var frames = [YCbCrImage]()
            while let frame = try reader.readFrame() {
                frames.append(frame)
            }
            allFrames = frames
        }
    }
    
    private func calcPlaneSSIM(p1: [UInt8], p2: [UInt8], w: Int, h: Int, stride1: Int, stride2: Int) -> Double {
        var ssimSum: Double = 0
        var blocks = 0
        let C1: Double = 6.5025
        let C2: Double = 58.5225

        p1.withUnsafeBufferPointer { ptr1 in
            p2.withUnsafeBufferPointer { ptr2 in
                guard let b1 = ptr1.baseAddress, let b2 = ptr2.baseAddress else { return }
                for y in stride(from: 0, to: h - 7, by: 8) {
                    for x in stride(from: 0, to: w - 7, by: 8) {
                        var sum1 = 0, sum2 = 0, sum1sq = 0, sum2sq = 0, sum12 = 0
                        for dy in 0..<8 {
                            let r1 = b1.advanced(by: (y + dy) * stride1 + x)
                            let r2 = b2.advanced(by: (y + dy) * stride2 + x)
                            for dx in 0..<8 {
                                let v1 = Int(r1[dx])
                                let v2 = Int(r2[dx])
                                sum1 += v1
                                sum2 += v2
                                sum1sq += v1 * v1
                                sum2sq += v2 * v2
                                sum12 += v1 * v2
                            }
                        }
                        let n = 64.0
                        let mu1 = Double(sum1) / n
                        let mu2 = Double(sum2) / n
                        let mu1sq = mu1 * mu1
                        let mu2sq = mu2 * mu2
                        let mu12 = mu1 * mu2
                        let sigma1sq = (Double(sum1sq) / n) - mu1sq
                        let sigma2sq = (Double(sum2sq) / n) - mu2sq
                        let sigma12 = (Double(sum12) / n) - mu12
                        let num = (2.0 * mu12 + C1) * (2.0 * sigma12 + C2)
                        let den = (mu1sq + mu2sq + C1) * (sigma1sq + sigma2sq + C2)
                        ssimSum += num / den
                        blocks += 1
                    }
                }
            }
        }
        return blocks == 0 ? 1.0 : ssimSum / Double(blocks)
    }

    private func calculateSSIMAll(img1: YCbCrImage, img2: YCbCrImage) -> Double {
        let w = min(img1.width, img2.width)
        let h = min(img1.height, img2.height)
        let ssimY = calcPlaneSSIM(p1: img1.yPlane, p2: img2.yPlane, w: w, h: h, stride1: img1.width, stride2: img2.width)
        let cw = min((img1.width + 1) / 2, (img2.width + 1) / 2)
        let ch = min((img1.height + 1) / 2, (img2.height + 1) / 2)
        let ssimU = calcPlaneSSIM(p1: img1.cbPlane, p2: img2.cbPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
        let ssimV = calcPlaneSSIM(p1: img1.crPlane, p2: img2.crPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
        return (4.0 * ssimY + ssimU + ssimV) / 6.0
    }
    
    func testPerformanceAndQualityTargets() async throws {
        let frames = allFrames
        guard !frames.isEmpty else { return }
        
        let width = frames[0].width
        let height = frames[0].height
        
        let encodeStart = Date()
        let encoder = VEVCEncoder(
            width: width,
            height: height,
            maxbitrate: 500 * 1000,
            zeroThreshold: 0, // strict quality
            keyint: 4,
            sceneChangeThreshold: 10
        )
        let bitstream = try await encoder.encodeToData(images: frames)
        let encodeElapsed = Date().timeIntervalSince(encodeStart)
        
        let decodeStart = Date()
        let decoder = Decoder(maxLayer: 2, maxConcurrency: 4)
        let decodedFrames = try await decoder.decode(data: [UInt8](bitstream))
        let decodeElapsed = Date().timeIntervalSince(decodeStart)
        
        let encodeTimePerFrame = (encodeElapsed * 1000) / Double(frames.count)
        let decodeTimePerFrame = (decodeElapsed * 1000) / Double(decodedFrames.count)
        let fileSizeKB = Double(bitstream.count) / 1024.0
        
        var totalSSIM: Double = 0
        for i in 0..<frames.count {
            let ssim = calculateSSIMAll(img1: frames[i], img2: decodedFrames[i])
            totalSSIM += ssim
        }
        let avgSSIM = totalSSIM / Double(frames.count)
        
        print(String(format: "E2E SpecV1 Results: Encode %.3fms/f, Decode %.3fms/f, Size %.2fKB, SSIM %.4f", encodeTimePerFrame, decodeTimePerFrame, fileSizeKB, avgSSIM))
        
        // Assertions (Can be adjusted based on first run)
        XCTAssertLessThanOrEqual(encodeTimePerFrame, 15.0, "Encode time should be <= 15.0ms/frame (Baseline for procedural data)")
        XCTAssertLessThanOrEqual(decodeTimePerFrame, 3.0, "Decode time should be <= 3.00ms/frame (Baseline for procedural data)")
        XCTAssertGreaterThanOrEqual(avgSSIM, 0.9044, "SSIM should be >= 0.9044")
        
        // Let's set a loose threshold for size initially. e.g. 5000KB (5MB) for these 60 frames.
        XCTAssertLessThanOrEqual(fileSizeKB, 2000, "File size should be <= 2000KB for the 60f procedural sequence")
    }
}
