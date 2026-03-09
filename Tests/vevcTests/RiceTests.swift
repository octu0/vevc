import XCTest
@testable import vevc

final class RiceTests: XCTestCase {
    func testBitWriterReader() throws {
        var bw = BitWriter()
        bw.writeBit(1)
        bw.writeBit(0)
        bw.writeBit(1)
        bw.writeBit(1)
        bw.writeBits(val: 0b101, n: 3)
        bw.flush()

        var br = BitReader(data: bw.data)
        XCTAssertEqual(try br.readBit(), 1)
        XCTAssertEqual(try br.readBit(), 0)
        XCTAssertEqual(try br.readBit(), 1)
        XCTAssertEqual(try br.readBit(), 1)
        XCTAssertEqual(try br.readBits(n: 3), 0b101)
    }

    func testRiceCodingRoundTrip() throws {
        let values: [UInt16] = [0, 1, 2, 3, 4, 10, 100, 500, 0, 0, 0, 1]
        let ks: [UInt8] = [0, 1, 2, 4, 8]

        for k in ks {
            let bw = BitWriter()
            var rw = RiceWriter(bw: bw)
            for val in values {
                rw.write(val: val, k: k)
            }
            rw.flush()

            let data = rw.extractWriter().data
            let br = BitReader(data: data)
            var rr = RiceReader(br: br)

            for val in values {
                let decoded = try rr.read(k: k)
                XCTAssertEqual(decoded, val, "Failed for val \(val) with k \(k)")
            }
        }
    }

    func testRiceCodingZeroRuns() throws {
        let values: [UInt16] = Array(repeating: 0, count: 100) + [5] + Array(repeating: 0, count: 10)
        let k: UInt8 = 4

        let bw = BitWriter()
        var rw = RiceWriter(bw: bw)
        for val in values {
            rw.write(val: val, k: k)
        }
        rw.flush()

        let data = rw.extractWriter().data
        let br = BitReader(data: data)
        var rr = RiceReader(br: br)

        for (i, val) in values.enumerated() {
            let decoded = try rr.read(k: k)
            XCTAssertEqual(decoded, val, "Failed at index \(i)")
        }
    }

    func testRiceCodingMaxValZeroRun() throws {
        // RiceWriter has maxVal = 64 for zero runs
        let values: [UInt16] = Array(repeating: 0, count: 64) + [0] + [1]
        let k: UInt8 = 2

        let bw = BitWriter()
        var rw = RiceWriter(bw: bw)
        for val in values {
            rw.write(val: val, k: k)
        }
        rw.flush()

        let data = rw.extractWriter().data
        let br = BitReader(data: data)
        var rr = RiceReader(br: br)

        for (i, val) in values.enumerated() {
            let decoded = try rr.read(k: k)
            XCTAssertEqual(decoded, val, "Failed at index \(i)")
        }
    }

    func testRiceWithWriterHelper() {
        var bw = BitWriter()
        let values: [UInt16] = [1, 2, 3, 0, 0, 4]
        let k: UInt8 = 3

        RiceWriter.withWriter(&bw) { rw in
            for val in values {
                rw.write(val: val, k: k)
            }
        }

        let br = BitReader(data: bw.data)
        var rr = RiceReader(br: br)
        for val in values {
            do {
                let decoded = try rr.read(k: k)
                XCTAssertEqual(decoded, val)
            } catch {
                XCTFail("Failed to read: \(error)")
            }
        }
    }
}
