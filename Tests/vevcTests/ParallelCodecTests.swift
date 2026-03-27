import XCTest
@testable import vevc

final class ParallelCodecTests: XCTestCase {
    
    // Helper to calculate PSNR
    private func calculatePSNR(original: [UInt8], decoded: [UInt8]) -> Double {
        let count = min(original.count, decoded.count)
        guard 0 < count else { return 0 }
        var mse: Double = 0
        for i in 0..<count {
            let diff = Double(Int(original[i]) - Int(decoded[i]))
            mse += diff * diff
        }
        mse /= Double(count)
        if mse < 0.0001 { return 100.0 }
        return 10.0 * log10(255.0 * 255.0 / mse)
    }

    // Generate a simple test sequence
    private func generateSequence(width: Int, height: Int, count: Int) -> [YCbCrImage] {
        var images: [YCbCrImage] = []
        for i in 0..<count {
            var img = YCbCrImage(width: width, height: height)
            let cWidth = (width + 1) / 2
            let cHeight = (height + 1) / 2
            for y in 0..<height {
                for x in 0..<width {
                    img.yPlane[y * width + x] = UInt8(clamping: (x + y + i * 5) % 256)
                }
            }
            for cy in 0..<cHeight {
                for cx in 0..<cWidth {
                    img.cbPlane[cy * cWidth + cx] = 128
                    img.crPlane[cy * cWidth + cx] = 128
                }
            }
            images.append(img)
        }
        return images
    }

    func testParallelEncodeDecodeRoundtrip() async throws {
        let width = 320
        let height = 240
        let frameCount = 10
        // Note: with temporal DWT (GOP=4), 10 frames → 2 VTGI chunks (4+4) + 2 individual I-frames = 4 chunks

        let images = generateSequence(width: width, height: height, count: frameCount)

        // Emulate streaming input frames
        let frameStream = AsyncStream<YCbCrImage> { continuation in
            for img in images {
                continuation.yield(img)
            }
            continuation.finish()
        }

        // 1. Parallel Encoding Test
        let encoder = VEVCEncoder(width: width, height: height, maxbitrate: 1000 * 1024, framerate: 30, zeroThreshold: 3, keyint: 60, sceneChangeThreshold: 8, maxConcurrency: 2)
        
        var chunks: [[UInt8]] = []
        let chunkStream = encoder.encode(stream: frameStream)
        for try await chunk in chunkStream {
            chunks.append(chunk)
        }
        // Temporal encoding: 10 frames → 1 VEVC Header + 2 VTGI (4+4=8 frames) + 2 individual (2 frames) = 5 chunks
        XCTAssertEqual(chunks.count, 5, "10 frames with temporal GOP=4 should produce 5 chunks (1 Header + 2 VTGI + 2 individual).")
        
        // Emulate streaming bitstream chunks
        let encodedStream = AsyncStream<[UInt8]> { continuation in
            for chunk in chunks {
                continuation.yield(chunk)
            }
            continuation.finish()
        }

        // 2. Parallel Decoding Test
        let decoder = Decoder(maxLayer: 2, maxConcurrency: 2, width: width, height: height)
        var decodedImages: [YCbCrImage] = []
        let imageStream = decoder.decode(stream: encodedStream)
        for try await img in imageStream {
            decodedImages.append(img)
        }
        
        XCTAssertEqual(decodedImages.count, frameCount, "Should output exactly one decoded image per frame.")

        // 3. Verify quality/correctness
        for i in 0..<frameCount {
            XCTAssertEqual(decodedImages[i].width, width)
            XCTAssertEqual(decodedImages[i].height, height)
            
            let psnr = calculatePSNR(original: images[i].yPlane, decoded: decodedImages[i].yPlane)
            // Expect reasonable PSNR even with multiple GOPs compressed in parallel
            XCTAssertGreaterThan(psnr, 25.0, "Frame \(i) PSNR (\(psnr)dB) is too low.")
        }
    }
}
