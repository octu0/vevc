import XCTest
@testable import vevc

final class CompressionFeatureTests: XCTestCase {

    func testPMVRoundTrip() async throws {
        // Create 2 frames. Second frame shifted slightly to induce MVs.
        var img1 = YCbCrImage(width: 64, height: 64)
        for y in 0..<64 {
            for x in 0..<64 {
                img1.yPlane[y * 64 + x] = UInt8(x * 2 + y * 2)
            }
        }
        var img2 = YCbCrImage(width: 64, height: 64)
        for y in 0..<64 {
            for x in 0..<64 {
                // shifted x and y
                let srcY = max(0, min(63, y - 2))
                let srcX = max(0, min(63, x - 2))
                img2.yPlane[y * 64 + x] = img1.yPlane[srcY * 64 + srcX]
            }
        }

        let pmvEncoder = VEVCEncoder(width: 64, height: 64, maxbitrate: 1000 * 1024)
        let encoded = try await pmvEncoder.encodeToData(images: [img1, img2])
        XCTAssertFalse(encoded.isEmpty)

        let decoded = try await Decoder().decode(data: encoded)
        XCTAssertEqual(decoded.count, 2)

        // Assert PMV recovered correctly (frame 2 should decode without error and maintain features)
        XCTAssertEqual(decoded[1].width, 64)
        XCTAssertEqual(decoded[1].height, 64)
    }

    func testLSCPRoundTrip() async throws {
        // We will encode and decode a block with many trailing zeros to test LSCP logic
        var encoder = EntropyEncoder<DynamicEntropyModel>()

        let size = 8
        var blockData = [Int16](repeating: 0, count: size * size)
        blockData[0] = 5
        blockData[1] = -3
        blockData[8] = 2
        // Rest are 0

        var block = Block2D(width: size, height: size)
        var view = block.view
        for y in 0..<size {
            let ptr = view.rowPointer(y: y)
            for x in 0..<size {
                ptr[x] = blockData[y * size + x]
            }
        }
    
        view = block.view
        blockEncode(encoder: &encoder, block: view, size: size)
            encoder.flush()

        let encodedData = encoder.getData()
        var decoder = try EntropyDecoder(data: encodedData)

        var outBlock = Block2D(width: size, height: size)

        view = outBlock.view
        try blockDecode(decoder: &decoder, block: view, size: size)
    
        view = outBlock.view
        for y in 0..<size {
            let ptr = view.rowPointer(y: y)
            for x in 0..<size {
                XCTAssertEqual(ptr[x], blockData[y * size + x])
            }
        }
        }
}
