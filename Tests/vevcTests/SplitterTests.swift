import Testing
import Foundation
@testable import vevc

struct SplitterTests {

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
}
