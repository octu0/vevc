import Testing
import Foundation
@testable import vevc

struct SplitterTests {

    @Test
    func testSplitVEVCStreamInvalidMagic() {
        let input: [UInt8] = [0x00, 0x00, 0x00, 0x00, 0x00]
        let maxLayer: Int = 2

        #expect(throws: SplitterError.self) {
            try splitVEVCStream(input: input, maxLayer: maxLayer)
        }

        do {
            _ = try splitVEVCStream(input: input, maxLayer: maxLayer)
        } catch let error as SplitterError {
            if case .invalidMagic = error {
                // Success
            } else {
                 Issue.record("Expected .invalidMagic, but got \\(error)")
            }
        } catch {
            Issue.record("Expected SplitterError, but got \\(error)")
        }
    }

    @Test
    func testUnexpectedEOF() throws {
        // Construct a valid VEVC bitstream
        let metadataPayload: [UInt8] = [
            0x01,       // Profile (1B)
            0x00, 0x10, // Width (16) (2B)
            0x00, 0x10, // Height (16) (2B)
            0x01,       // ColorGamut (1B)
            0x00, 0x1E, // Framerate (30) (2B)
            0x00,       // Timescale (1B)
            0x00        // Table Flag (1B)
        ]

        var bitstream: [UInt8] = [0x56, 0x45, 0x56, 0x43] // Magic
        bitstream.append(contentsOf: [UInt8(metadataPayload.count >> 8), UInt8(metadataPayload.count & 0xFF)])
        bitstream.append(contentsOf: metadataPayload)

        let headerSizes: [UInt8] = [
            0x00, 0x00, 0x00, 0x01, // mvsSize = 1
            0x00, 0x00, 0x00, 0x00, // refDirSize = 0 (hasRefDir is false in flag 0x02)
            0x00, 0x00, 0x00, 0x01, // layer0Size = 1
            0x00, 0x00, 0x00, 0x01, // layer1Size = 1
            0x00, 0x00, 0x00, 0x01  // layer2Size = 1
        ]

        bitstream.append(0x02) // I-Frame flag (FrameType = 0x02, no ref dir)
        bitstream.append(contentsOf: headerSizes)

        bitstream.append(0xFF) // mvs payload
        bitstream.append(0xFE) // layer0 payload
        bitstream.append(0xFD) // layer1 payload
        bitstream.append(0xFC) // layer2 payload

        // Full bitstream works
        let result = try splitVEVCStream(input: bitstream, maxLayer: 2)
        #expect(result.processedFrames == 1)

        // Truncate at every possible length less than full size
        for i in 0..<bitstream.count {
            let truncated = Array(bitstream[0..<i])

            // If we truncate exactly at EOF boundary of a frame, it might be valid if no more frames are read.
            // But here we are within a frame, or before metadata, or in metadata.
            // However, after metadata, if we truncate exactly, the loop `readOffset < input.count`
            // will terminate. So `i = 16` (end of metadata) would be a valid empty stream.
            let isEndOfMetadata = (i == 4 + 2 + metadataPayload.count) // 16

            if isEndOfMetadata {
                let truncResult = try splitVEVCStream(input: truncated, maxLayer: 2)
                #expect(truncResult.processedFrames == 0)
                continue
            }

            #expect(throws: SplitterError.self) {
                try splitVEVCStream(input: truncated, maxLayer: 2)
            }

            do {
                _ = try splitVEVCStream(input: truncated, maxLayer: 2)
            } catch let error as SplitterError {
                if case .unexpectedEOF = error {
                    // Success
                } else if case .invalidMagic = error {
                     #expect(i < 4) // Only valid if magic was truncated or wrong
                } else {
                     Issue.record("Unexpected SplitterError: \(error) at length \(i)")
                }
            } catch {
                Issue.record("Unexpected error type: \(error) at length \(i)")
            }
        }
    }

    @Test
    func testInvalidMetadataSizeThrowsUnexpectedEOF() throws {
        // Test metadata size < 5
        let bitstream: [UInt8] = [
            0x56, 0x45, 0x56, 0x43, // Magic
            0x00, 0x04, // metadata size 4
            0x01, 0x00, 0x10, 0x00  // 4 bytes payload
        ]

        #expect(throws: SplitterError.self) {
            try splitVEVCStream(input: bitstream, maxLayer: 2)
        }

        do {
            _ = try splitVEVCStream(input: bitstream, maxLayer: 2)
        } catch let error as SplitterError {
            if case .unexpectedEOF = error {
                // Success
            } else {
                 Issue.record("Unexpected SplitterError: \(error)")
            }
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testSplitVEVCRoundtrip() async throws {
        let y4mPath = "Tests/vevcSpecV1/testdata/spec_1080p_60f.y4m"
        let fileHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: y4mPath))
        let reader = try Y4MReader(fileHandle: fileHandle)
        var frames = [YCbCrImage]()
        while let frame = try reader.readFrame() {
            frames.append(frame)
        }
        
        let encoder = VEVCEncoder(
            width: frames[0].width,
            height: frames[0].height,
            maxbitrate: 1000 * 1024,
            zeroThreshold: 3,
            keyint: 10
        )
        
        let bitstream = try await encoder.encodeToData(images: Array(frames.prefix(5)))
        let bytes = [UInt8](bitstream)
        
        // 1. Split to maxLayer = 1
        let l1Result = try splitVEVCStream(input: bytes, maxLayer: 1)
        #expect(l1Result.processedFrames == 5)
        #expect(l1Result.droppedLayer2Bytes > 0)
        #expect(l1Result.droppedLayer1Bytes == 0)
        
        // Decode l1
        let decoder1 = Decoder(maxLayer: 1)
        let decL1Frames = try await decoder1.decode(data: l1Result.data)
        #expect(decL1Frames.count == 5)
        
        // 2. Split to maxLayer = 0
        let l0Result = try splitVEVCStream(input: bytes, maxLayer: 0)
        #expect(l0Result.processedFrames == 5)
        #expect(l0Result.droppedLayer2Bytes > 0)
        #expect(l0Result.droppedLayer1Bytes > 0)
        
        // Decode l0
        let decoder0 = Decoder(maxLayer: 0)
        let decL0Frames = try await decoder0.decode(data: l0Result.data)
        #expect(decL0Frames.count == 5)
        
        // 3. Split to maxLayer = 2 (should be same as original)
        let l2Result = try splitVEVCStream(input: bytes, maxLayer: 2)
        #expect(l2Result.processedFrames == 5)
        #expect(l2Result.droppedLayer2Bytes == 0)
        #expect(l2Result.droppedLayer1Bytes == 0)
        
        // Decode l2
        let decoder2 = Decoder(maxLayer: 2)
        let decL2Frames = try await decoder2.decode(data: l2Result.data)
        #expect(decL2Frames.count == 5)
    }
}
