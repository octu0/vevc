import XCTest
import CryptoKit
@testable import vevc

final class Profile0x02FixtureTests: XCTestCase {
    func testProfile0x02FixtureMatch() async throws {
        let width = 64
        let height = 64
        var y = [UInt8](repeating: 128, count: width * height)
        var cb = [UInt8](repeating: 128, count: width * height / 4)
        var cr = [UInt8](repeating: 128, count: width * height / 4)
        
        for i in 0..<height {
            for j in 0..<width {
                y[i * width + j] = UInt8((i / 8 * 8) % 256)
            }
        }
        
        var frames = [YCbCrImage]()
        for f in 0..<5 {
            var mY = y
            mY[0] = UInt8(((Int(y[0]) + f * 10) % 256))
            var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
            img.yPlane = mY
            img.cbPlane = cb
            img.crPlane = cr
            frames.append(img)
        }
        
        let encoder = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 30, profile: 0x02)
        let bitstream = try await encoder.encodeToData(images: frames)
        
        let hash = SHA256.hash(data: Data(bitstream))
        let hashString = hash.compactMap { String(format: "%02x", $0) }.joined()
        
        print("Generated Hash: \(hashString)")
        // Expected hash before optimization. Once generated, update this string.
        let expectedHash = "2ba7991b589f15f27a49d089ad889f1451bd21295be857707084374147b265a5"
        
        if expectedHash != "PLEASE_UPDATE_ME" {
            XCTAssertEqual(hashString, expectedHash, "Profile 0x02 bitstream output must be bit-exact matching the fixture.")
        } else {
            XCTFail("Run again and update PLEASE_UPDATE_ME to \(hashString)")
        }
    }
}
