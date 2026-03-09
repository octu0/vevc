import XCTest
@testable import vevc

final class RiceTests: XCTestCase {
    
    func testBitWriterReader() throws {
        var bw = BitWriter()
        bw.writeBit(1)
        bw.writeBit(0)
        bw.writeBits(val: 5, n: 3) // 101
        bw.flush()
        
        let data = bw.data
        XCTAssertEqual(data.count, 1)
        // 1010 1000 = 168
        let firstByte = data.first
        XCTAssertEqual(firstByte, 168)
        
        var br = BitReader(data: data)
        let b1 = try br.readBit()
        let b2 = try br.readBit()
        let val = try br.readBits(n: 3)
        
        XCTAssertEqual(b1, 1)
        XCTAssertEqual(b2, 0)
        XCTAssertEqual(val, 5)
    }

    func testRiceWriterReaderRoundTrip() throws {
        var bw = BitWriter()
        
        // Write some values
        struct TestValue {
            let val: UInt16
            let k: UInt8
        }
        let values: [TestValue] = [
            TestValue(val: 5, k: 2),
            TestValue(val: 0, k: 2),
            TestValue(val: 0, k: 2),
            TestValue(val: 0, k: 2),
            TestValue(val: 10, k: 3),
            TestValue(val: 0, k: 3),
            TestValue(val: 100, k: 6)
        ]
        
        var rw = RiceWriter(bw: bw)
        for item in values {
            rw.write(val: item.val, k: item.k)
        }
        rw.flush()
        
        bw = rw.extractWriter()
        
        let data = bw.data
        
        let br = BitReader(data: data)
        var rr = RiceReader(br: br)
        
        for item in values {
            let val = try rr.read(k: item.k)
            XCTAssertEqual(val, item.val)
        }
    }

    func testRiceWriterZeroFlush() throws {
        var bw = BitWriter()
        var rw = RiceWriter(bw: bw)
        
        // maxVal in RiceWriter is 64. Let's write 65 zeros to trigger flushZeros
        for _ in 0..<65 {
            rw.write(val: 0, k: 2)
        }
        
        rw.flush()
        bw = rw.extractWriter()
        let data = bw.data
        
        let br = BitReader(data: data)
        var rr = RiceReader(br: br)
        
        for _ in 0..<65 {
            let val = try rr.read(k: 2)
            XCTAssertEqual(val, 0)
        }
    }
    
    func testRiceWriterWithWriter() throws {
        var bw = BitWriter()
        
        RiceWriter.withWriter(&bw) { rw in
            rw.write(val: 42, k: 4)
            rw.write(val: 0, k: 4)
        }
        
        let data = bw.data
        let br = BitReader(data: data)
        var rr = RiceReader(br: br)
        
        let v1 = try rr.read(k: 4)
        let v2 = try rr.read(k: 4)
        
        XCTAssertEqual(v1, 42)
        XCTAssertEqual(v2, 0)
    }

    func testBitReaderEOF() throws {
        let data: [UInt8] = []
        var br = BitReader(data: data)
        XCTAssertThrowsError(try br.readBit()) { error in
            guard let decodeError = error as? DecodeError else {
                XCTFail("Expected DecodeError")
                return
            }
            XCTAssertEqual(decodeError, DecodeError.eof)
        }
    }
}
