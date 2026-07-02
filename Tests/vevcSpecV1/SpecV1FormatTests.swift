import XCTest
@testable import vevc

final class SpecV1FormatTests: XCTestCase {
    
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
            allFrames = Array(frames.prefix(4)) // only need a few frames to verify format
        }
    }
    
    func testDataFormatStructure() async throws {
        guard allFrames.isEmpty != true else { return }
        let width = allFrames[0].width
        let height = allFrames[0].height
        let encoder = VEVCEncoder(
            width: width,
            height: height,
            maxbitrate: 1000 * 1024,
            zeroThreshold: 3,
            keyint: 4,
            sceneChangeThreshold: 10
        )
        
        let bitstream = try await encoder.encodeToData(images: allFrames)
        let bytes = [UInt8](bitstream)
        
        // --- 1. Magic Header Verification ---
        XCTAssertGreaterThan(bytes.count, 4)
        XCTAssertEqual(bytes[0], 0x56) // 'V'
        XCTAssertEqual(bytes[1], 0x45) // 'E'
        XCTAssertEqual(bytes[2], 0x56) // 'V'
        XCTAssertEqual(bytes[3], 0x43) // 'C'
        
        var offset = 4
        
        // --- 2. Metadata Size ---
        XCTAssertGreaterThan(bytes.count, offset + 2)
        let metadataSize = Int(try readUInt16BEFromBytes(bytes, offset: &offset))
        let metaStart = offset
        
        // --- 3. Profile Version ---
        XCTAssertGreaterThan(bytes.count, offset + 1)
        let profileVersion = bytes[offset]
        offset += 1
        XCTAssertEqual(profileVersion, 0x01, "Profile version must be 0x01")
        
        // --- 4. Dimensions ---
        let w = Int(try readUInt16BEFromBytes(bytes, offset: &offset))
        let h = Int(try readUInt16BEFromBytes(bytes, offset: &offset))
        XCTAssertEqual(w, width)
        XCTAssertEqual(h, height)
        
        // --- 5. Color Gamut and FPS ---
        let gamut = bytes[offset]
        offset += 1
        XCTAssertEqual(gamut, 0x01, "Color gamut must be 0x01 (BT.709) by default")
        
        _ = Int(try readUInt16BEFromBytes(bytes, offset: &offset))
        let timescale = bytes[offset]
        offset += 1
        XCTAssertEqual(timescale, 0x00, "Timescale is typically 0x00 for fps representation")
        
        // rANS models: 4 tables, each 256 bytes = 1024 bytes (or dynamic/static flags)
        // Skip the rest of the metadata based on metadataSize
        offset = metaStart + metadataSize
        
        // --- 6. Frame Parsing ---
        // Expect a GOP header, then frames
        var frameCount = 0
        while offset < bytes.count {
            // Read status flag of frame packet
            let status = bytes[offset]
            offset += 1
            
            if status == 0x01 {
                // IsCopyFrame
                // No payload sizes, just the flag
                frameCount += 1
            } else {
                let frameTypeBits = status & 0x0F
                let hasRefDir = (status & 0x10) != 0
                XCTAssertTrue(frameTypeBits == 0x00 || frameTypeBits == 0x02, "Frame type must be pFrame or iFrame, got \(frameTypeBits)")
                
                // Normal or B-Frame: 5 x UInt32BE = 20 bytes
                let mvsSize = Int(try readUInt32BEFromBytes(bytes, offset: &offset))
                let refDirSize = Int(try readUInt32BEFromBytes(bytes, offset: &offset))
                let layer0Size = Int(try readUInt32BEFromBytes(bytes, offset: &offset))
                let layer1Size = Int(try readUInt32BEFromBytes(bytes, offset: &offset))
                let layer2Size = Int(try readUInt32BEFromBytes(bytes, offset: &offset))
                
                // Verify refDirSize matching hasRefDir logic
                let expectedRefDirBytes = if hasRefDir { (deriveMVCount(width: width, height: height) + 7) / 8 } else { 0 }
                XCTAssertEqual(refDirSize, expectedRefDirBytes, "refDirSize in header must match expectedRefDirBytes")
                
                let totalPayload = mvsSize + refDirSize + layer0Size + layer1Size + layer2Size
                XCTAssertLessThanOrEqual(offset + totalPayload, bytes.count, "Payload exceeds bitstream size")
                
                offset += totalPayload
                frameCount += 1
            }
        }
        
        XCTAssertEqual(frameCount, allFrames.count, "Bitstream must contain exactly the encoded number of frames")
    }
    
    func testFrameHeaderRefDirSizeValidation() {
        // Case 1: hasRefDir == true, but refDirSize == 0 (invalid)
        var badBytes1 = [UInt8]()
        let flag1: UInt8 = 0x10 | 0x00 // hasRefDir | pFrame
        badBytes1.append(flag1)
        appendUInt32BE(&badBytes1, 100) // mvsSize
        appendUInt32BE(&badBytes1, 0)   // refDirSize (0 is bad when hasRefDir is true)
        appendUInt32BE(&badBytes1, 100) // layer0Size
        appendUInt32BE(&badBytes1, 100) // layer1Size
        appendUInt32BE(&badBytes1, 100) // layer2Size
        
        var offset1 = 0
        XCTAssertThrowsError(try VEVCFrameHeader.deserialize(from: badBytes1, offset: &offset1)) { error in
            XCTAssertTrue(error is DecodeError)
            if let decodeError = error as? DecodeError {
                XCTAssertEqual(decodeError.description, "DecodeError.invalidHeader")
            }
        }
        
        // Case 2: hasRefDir == false, but refDirSize > 0 (invalid)
        var badBytes2 = [UInt8]()
        let flag2: UInt8 = 0x00 | 0x00 // hasRefDir=0 | pFrame
        badBytes2.append(flag2)
        appendUInt32BE(&badBytes2, 100) // mvsSize
        appendUInt32BE(&badBytes2, 50)  // refDirSize (50 is bad when hasRefDir is false)
        appendUInt32BE(&badBytes2, 100) // layer0Size
        appendUInt32BE(&badBytes2, 100) // layer1Size
        appendUInt32BE(&badBytes2, 100) // layer2Size
        
        var offset2 = 0
        XCTAssertThrowsError(try VEVCFrameHeader.deserialize(from: badBytes2, offset: &offset2)) { error in
            XCTAssertTrue(error is DecodeError)
            if let decodeError = error as? DecodeError {
                XCTAssertEqual(decodeError.description, "DecodeError.invalidHeader")
            }
        }
    }
}

// Helper to read Big Endian sizes
private func readUInt16BEFromBytes(_ bytes: [UInt8], offset: inout Int) throws -> UInt16 {
    guard offset + 2 <= bytes.count else { throw NSError(domain: "Test", code: 1, userInfo: nil) }
    let val = (UInt16(bytes[offset]) << 8) | UInt16(bytes[offset + 1])
    offset += 2
    return val
}

private func readUInt32BEFromBytes(_ bytes: [UInt8], offset: inout Int) throws -> UInt32 {
    guard offset + 4 <= bytes.count else { throw NSError(domain: "Test", code: 1, userInfo: nil) }
    let val = (UInt32(bytes[offset]) << 24) |
              (UInt32(bytes[offset + 1]) << 16) |
              (UInt32(bytes[offset + 2]) << 8) |
              UInt32(bytes[offset + 3])
    offset += 4
    return val
}
