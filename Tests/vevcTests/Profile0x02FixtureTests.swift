import XCTest
import CryptoKit
@testable import vevc

final class Profile0x02FixtureTests: XCTestCase {
    func testProfile0x02MixedMotionDeterminismAndDecoding() async throws {
        let width = 128
        let height = 128
        let frameCount = 10
        
        var frames = [YCbCrImage]()
        for f in 0..<frameCount {
            var y = [UInt8](repeating: 128, count: width * height)
            var cb = [UInt8](repeating: 128, count: width * height / 4)
            var cr = [UInt8](repeating: 128, count: width * height / 4)
            
            for row in 0..<height {
                let isTopHalf = (row < height / 2)
                for col in 0..<width {
                    let idx = row * width + col
                    if isTopHalf {
                        // Top half: static pattern (checkerboard)
                        let val = UInt8(((row / 8) + (col / 8)) % 2 * 180 + 30)
                        y[idx] = val
                    } else {
                        // Bottom half: moving pattern (horizontal shift by frame index f)
                        let shiftedCol = (col + f * 4) % width
                        let val = UInt8(((row / 8) + (shiftedCol / 8)) % 2 * 200 + 20)
                        y[idx] = val
                    }
                }
            }
            
            for row in 0..<(height / 2) {
                for col in 0..<(width / 2) {
                    let idx = row * (width / 2) + col
                    cb[idx] = UInt8((128 + row * 2) % 256)
                    cr[idx] = UInt8((128 + col * 2) % 256)
                }
            }
            
            var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
            img.yPlane = y
            img.cbPlane = cb
            img.crPlane = cr
            frames.append(img)
        }
        
        // 1. Encoding Determinism Test (2 consecutive runs produce bit-exact same SHA-256)
        let encoder1 = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 30, profile: 0x02)
        let bitstream1 = try await encoder1.encodeToData(images: frames)
        let hash1 = SHA256.hash(data: Data(bitstream1)).compactMap { String(format: "%02x", $0) }.joined()
        
        let encoder2 = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 30, profile: 0x02)
        let bitstream2 = try await encoder2.encodeToData(images: frames)
        let hash2 = SHA256.hash(data: Data(bitstream2)).compactMap { String(format: "%02x", $0) }.joined()
        
        XCTAssertEqual(hash1, hash2, "Profile 0x02 encoding must be strictly deterministic across repeated runs.")
        
        // 2. Roundtrip Decoding Test
        let decodedFrames = try await Decoder().decode(data: bitstream1)
        
        XCTAssertEqual(decodedFrames.count, frameCount, "Decoded frame count must match input frame count.")
        
        // Verify PSNR quality on the decoded frames (minimum PSNR > 25.0 dB)
        for (fIdx, original) in frames.enumerated() {
            let decodedImg = decodedFrames[fIdx]
            XCTAssertEqual(decodedImg.width, original.width)
            XCTAssertEqual(decodedImg.height, original.height)
            
            var mseY: Double = 0
            for i in 0..<original.yPlane.count {
                let origVal = Double(original.yPlane[i])
                let decVal = Double(decodedImg.yPlane[i])
                let diffVal = origVal - decVal
                mseY += diffVal * diffVal
            }
            mseY /= Double(original.yPlane.count)
            let psnrY = 10.0 * log10((255.0 * 255.0) / max(0.0001, mseY))
            XCTAssertGreaterThan(psnrY, 25.0, "Frame \(fIdx) Y PSNR (\(psnrY) dB) must exceed 25.0 dB threshold.")
        }
    }
}
