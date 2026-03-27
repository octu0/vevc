import XCTest
@testable import vevc

final class BufferStructureTests: XCTestCase {
    
    func testBufferStructureOneMode() async throws {
        // 1. Create a dummy test image
        let width = 128
        let height = 128
        var img = YCbCrImage(width: width, height: height)
        for i in 0..<img.yPlane.count { img.yPlane[i] = 128 }
        for i in 0..<img.cbPlane.count { img.cbPlane[i] = 128 }
        for i in 0..<img.crPlane.count { img.crPlane[i] = 128 }
        
        // 2. Encode using CoreEncoder ()
        let encoder = CoreEncoder(
            width: width,
            height: height,
            maxbitrate: 500_000,
            framerate: 30,
            zeroThreshold: 3,
            keyint: 60,
            sceneChangeThreshold: 32,
            
        )
        
        let chunk = try await encoder.encode(image: img)
        
        // 3. Verify structure of the encoded chunk
        var offset = 0
        let dataSize = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
        XCTAssertEqual(dataSize, chunk.count - 4, "DataSize should be total chunk size minus 4")
        
        let mode = chunk[offset]
        XCTAssertEqual(mode, 0x01, "Mode should be Direct (0x01)")
        offset += 1
        
        let gopSize = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
        XCTAssertEqual(gopSize, 1, "GOPSize should be 1 for Direct Mode")
        
        let nLow = Int(try readUInt16BEFromBytes(chunk, offset: &offset))
        XCTAssertEqual(nLow, 0, "nLow should be 0 for Direct Mode")
        
        let frameLength = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
        XCTAssertEqual(frameLength, chunk.count - 15, "FrameLength should match the payload size")
        
        // At this point, offset is 15. The payload bytes start here.
        // In encodePlaneBase32, the first two fields are qtY.step (2B) and qtC.step (2B)
        let qtYStep = Int(try readUInt16BEFromBytes(chunk, offset: &offset))
        let qtCStep = Int(try readUInt16BEFromBytes(chunk, offset: &offset))
        XCTAssertGreaterThan(qtYStep, 0, "qtY.step should be > 0")
        XCTAssertGreaterThan(qtCStep, 0, "qtC.step should be > 0")
        
        // The next 4 bytes should be bufY.count
        let bufYLen = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
        XCTAssertGreaterThan(bufYLen, 0, "bufYLen should be > 0")
        XCTAssertLessThanOrEqual(bufYLen, frameLength, "bufYLen cannot exceed FrameLength")
        
        // Check if the next offset actually matches bufY's content bounds
        offset += bufYLen
        
        let bufCbLen = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
        XCTAssertGreaterThan(bufCbLen, 0, "bufCbLen should be > 0")
        XCTAssertLessThanOrEqual(bufCbLen, frameLength, "bufCbLen cannot exceed FrameLength")
        
        offset += bufCbLen
        
        let bufCrLen = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
        XCTAssertGreaterThan(bufCrLen, 0, "bufCrLen should be > 0")
        XCTAssertLessThanOrEqual(bufCrLen, frameLength, "bufCrLen cannot exceed FrameLength")
        
        offset += bufCrLen
        
        XCTAssertEqual(offset, chunk.count, "Total consumed bytes should exactly match chunk size")
        
        // 4. Test Decoder
        let decoder = CoreDecoder(maxLayer: 0, width: width, height: height)
        let decodedImages = try await decoder.decodeGOP(chunk: chunk)
        XCTAssertEqual(decodedImages.count, 1, "Should decode exactly 1 image")
        XCTAssertEqual(decodedImages[0].width, width)
        XCTAssertEqual(decodedImages[0].height, height)
    }
    
    func testHighLevelAPIOneMode() async throws {
        let width = 128
        let height = 128
        var images: [YCbCrImage] = []
        for j in 0..<3 {
            var img = YCbCrImage(width: width, height: height)
            for i in 0..<img.yPlane.count { img.yPlane[i] = UInt8(j * 10) }
            for i in 0..<img.cbPlane.count { img.cbPlane[i] = 128 }
            for i in 0..<img.crPlane.count { img.crPlane[i] = 128 }
            images.append(img)
        }
        
        let encodedBytes = try await vevc.encode(images: images, maxbitrate: 500_000, framerate: 30)
        
        var offset = 0
        let byte0 = encodedBytes[offset]
        XCTAssertEqual(byte0, 0x56)
        offset += 4
        let metaSize = Int(try readUInt16BEFromBytes(encodedBytes, offset: &offset))
        offset += metaSize // skip meta
        
        let decoder = CoreDecoder(maxLayer: 0, width: width, height: height)
        var decodedCount = 0
        
        while offset < encodedBytes.count {
            let chunkStart = offset
            let gopDataSize = Int(try readUInt32BEFromBytes(encodedBytes, offset: &offset))
            let chunkEnd = offset + gopDataSize
            let chunk = Array(encodedBytes[chunkStart..<chunkEnd])
            
            do {
                let decoded = try await decoder.decodeGOP(chunk: chunk)
                decodedCount += decoded.count
            } catch {
                print("decodeGOP failed at offset \(chunkStart) with error: \(error)")
                XCTFail("decodeGOP threw an error")
                break
            }
            offset = chunkEnd
        }
        
        XCTAssertEqual(decodedCount, 3, "Should decode exactly 3 images")
    }
}
