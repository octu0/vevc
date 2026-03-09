import XCTest
@testable import vevc

final class BitReaderWriterTests: XCTestCase {
    
    func testWriteAndReadSingleBits() throws {
        var bw = BitWriter()
        
        // Write: 1011 0100
        bw.writeBit(1)
        bw.writeBit(0)
        bw.writeBit(1)
        bw.writeBit(1)
        
        bw.writeBit(0)
        bw.writeBit(1)
        bw.writeBit(0)
        bw.writeBit(0)
        
        bw.flush()
        
        XCTAssertEqual(bw.data.count, 1)
        XCTAssertEqual(bw.data[0 + 0], 0xB4) // 10110100 in binary is 0xB4 (180)
        
        var br = BitReader(data: bw.data)
        
        let b1 = try br.readBit()
        let b2 = try br.readBit()
        let b3 = try br.readBit()
        let b4 = try br.readBit()
        let b5 = try br.readBit()
        let b6 = try br.readBit()
        let b7 = try br.readBit()
        let b8 = try br.readBit()
        
        XCTAssertEqual(b1, 1)
        XCTAssertEqual(b2, 0)
        XCTAssertEqual(b3, 1)
        XCTAssertEqual(b4, 1)
        XCTAssertEqual(b5, 0)
        XCTAssertEqual(b6, 1)
        XCTAssertEqual(b7, 0)
        XCTAssertEqual(b8, 0)
        
        // Ensure reading past EOF throws
        do {
            _ = try br.readBit()
            XCTFail("Expected eof error")
        } catch DecodeError.eof {
            // Expected
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
    
    func testWriteAndReadMultipleBits() throws {
        var bw = BitWriter()
        
        // Write a 12-bit value: 0xABC (1010 1011 1100) -> 2748
        bw.writeBits(val: 0xABC, n: 12)
        // Write a 4-bit value: 0x5 (0101)
        bw.writeBits(val: 0x5, n: 4)
        
        bw.flush()
        
        // Total 16 bits = 2 bytes
        XCTAssertEqual(bw.data.count, 2)
        XCTAssertEqual(bw.data[0 + 0], 0xAB)
        XCTAssertEqual(bw.data[1 + 0], 0xC5)
        
        var br = BitReader(data: bw.data)
        
        let val1 = try br.readBits(n: 12)
        let val2 = try br.readBits(n: 4)
        
        XCTAssertEqual(val1, 0xABC)
        XCTAssertEqual(val2, 0x5)
        
        // Ensure EOF
        do {
            _ = try br.readBit()
            XCTFail("Expected eof error")
        } catch DecodeError.eof {
            // Expected
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
    
    func testBufferFlush() throws {
        var bw = BitWriter()
        
        // The BitWriter has a buffer of size 4096.
        // Let's write 4096 bytes (32768 bits) to fill it.
        for _ in 0..<4096 {
            bw.writeBits(val: 0xFF, n: 8)
        }
        
        // At exactly 4096 bytes, the buffer is flushed into `data`.
        XCTAssertEqual(bw.data.count, 4096)
        
        // Write one more bit to start a new byte
        bw.writeBit(1)
        
        // Still 4096 bytes in `data`
        XCTAssertEqual(bw.data.count, 4096)
        
        // Flush the remaining bits
        bw.flush()
        
        XCTAssertEqual(bw.data.count, 4097)
        
        var br = BitReader(data: bw.data)
        for _ in 0..<4096 {
            let val = try br.readBits(n: 8)
            XCTAssertEqual(val, 0xFF)
        }
        let lastBit = try br.readBit()
        XCTAssertEqual(lastBit, 1)
        
        // Ensure pad bits are 0 (the remaining 7 bits in the last byte)
        for _ in 0..<7 {
            let pad = try br.readBit()
            XCTAssertEqual(pad, 0)
        }
    }

    func testRoundTripRandomData() throws {
        var bw = BitWriter()
        
        let testValues: [(val: UInt16, n: UInt8)] = [
            (0, 1),
            (1, 1),
            (3, 2),
            (15, 4),
            (1023, 10),
            (4095, 12),
            (65535, 16),
            (42, 6),
            (128, 8),
            (0, 5)
        ]
        
        for item in testValues {
            bw.writeBits(val: item.val, n: item.n)
        }
        
        bw.flush()
        
        var br = BitReader(data: bw.data)
        
        for item in testValues {
            let val = try br.readBits(n: item.n)
            XCTAssertEqual(val, item.val)
        }
    }
}
