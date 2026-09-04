import Foundation

public struct VSDSubband {
    public let width: Int
    public let height: Int
    public let data: [Int16]

    public init(width: Int, height: Int, data: [Int16]) {
        self.width = width
        self.height = height
        self.data = data
    }
}

public struct VSDFrame {
    public let gop: Int
    public let width: Int
    public let height: Int
    public let qstep: Int32
    public let layer0Bytes: Int
    public let subbands: [VSDSubband]

    public init(gop: Int, width: Int, height: Int, qstep: Int32, layer0Bytes: Int, subbands: [VSDSubband]) {
        self.gop = gop
        self.width = width
        self.height = height
        self.qstep = qstep
        self.layer0Bytes = layer0Bytes
        self.subbands = subbands
    }
}

public final class VSDReader {
    private let data: Data
    private var offset: Int = 0

    public init(path: String) throws {
        self.data = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
        guard 4 <= data.count else {
            throw VSDError.invalidFormat("File too short")
        }
        let magic = String(decoding: data[0..<4], as: UTF8.self)
        guard magic == "VSD1" else {
            throw VSDError.invalidFormat("Invalid magic: \(magic)")
        }
        self.offset = 4
    }

    public func nextFrame() throws -> VSDFrame? {
        if data.count <= offset {
            return nil
        }
        guard offset + 4 <= data.count else {
            return nil
        }
        let tag = String(decoding: data[offset..<(offset + 4)], as: UTF8.self)
        guard tag == "FRAM" else {
            return nil
        }
        offset += 4

        let gop = readInt32()
        let width = readInt32()
        let height = readInt32()

        var qsteps: [[Int]] = []
        for _ in 0..<6 {
            let low = readInt32()
            let mid = readInt32()
            let high = readInt32()
            qsteps.append([low, mid, high])
        }

        let layer0Bytes = readInt32()
        _ = readInt32() // layer1Bytes
        _ = readInt32() // layer2Bytes

        let qstep = Int32(qsteps[4][0]) // qtY0.qLow.step

        // coded entries: 9 entries
        // [0: L2Y, 1: L2Cb, 2: L2Cr, 3: L1Y, 4: L1Cb, 5: L1Cr, 6: L0Y, 7: L0Cb, 8: L0Cr]
        var l0ySubbands: [VSDSubband]?
        for i in 0..<9 {
            if i == 6 {
                l0ySubbands = readSubbands()
            } else {
                skipSubbands()
            }
        }

        // parents entries: 6 entries
        for _ in 0..<6 {
            skipSubbands()
        }

        // pyr entries: 9 entries
        for _ in 0..<9 {
            skipSubbands()
        }

        guard let subbands = l0ySubbands, 4 <= subbands.count else {
            throw VSDError.invalidFormat("Failed to read L0Y subbands")
        }

        return VSDFrame(
            gop: gop,
            width: width,
            height: height,
            qstep: qstep,
            layer0Bytes: layer0Bytes,
            subbands: subbands
        )
    }

    private func readInt32() -> Int {
        guard offset + 4 <= data.count else { return 0 }
        let val = data.withUnsafeBytes { ptr -> Int32 in
            let base = ptr.baseAddress!.advanced(by: offset)
            return base.loadUnaligned(as: Int32.self)
        }
        offset += 4
        return Int(Int32(littleEndian: val))
    }

    private func readSubbands() -> [VSDSubband] {
        let nSub = readInt32()
        var subs: [VSDSubband] = []
        subs.reserveCapacity(nSub)
        for _ in 0..<nSub {
            let w = readInt32()
            let h = readInt32()
            let count = w * h
            let byteCount = count * 2
            guard offset + byteCount <= data.count else {
                offset = data.count
                break
            }
            var subData = [Int16](repeating: 0, count: count)
            subData.withUnsafeMutableBytes { dstPtr in
                data.withUnsafeBytes { srcPtr in
                    let src = srcPtr.baseAddress!.advanced(by: offset)
                    dstPtr.baseAddress!.copyMemory(from: src, byteCount: byteCount)
                }
            }
            #if _endian(big)
            for k in 0..<count {
                subData[k] = Int16(littleEndian: subData[k])
            }
            #endif
            offset += byteCount
            subs.append(VSDSubband(width: w, height: h, data: subData))
        }
        return subs
    }

    private func skipSubbands() {
        let nSub = readInt32()
        for _ in 0..<nSub {
            let w = readInt32()
            let h = readInt32()
            let byteCount = w * h * 2
            offset += byteCount
        }
    }
}

public enum VSDError: Error {
    case invalidFormat(String)
}
