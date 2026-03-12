// MARK: - CABAC (Context-Adaptive Binary Arithmetic Coding)

import Foundation

// MARK: - Probability State and LUT

let rangeLPS_table: [[UInt32]] = [
    (0...127).map { s in
        let p = pow(0.5, Double(s) / 10.0)
        return UInt32(max(1, min(255, Int(p * 256.0))))
    },
    (0...127).map { s in
        let p = pow(0.5, Double(s) / 10.0)
        return UInt32(max(1, min(255, Int(p * 256.0))))
    },
    (0...127).map { s in
        let p = pow(0.5, Double(s) / 10.0)
        return UInt32(max(1, min(255, Int(p * 256.0))))
    },
    (0...127).map { s in
        let p = pow(0.5, Double(s) / 10.0)
        return UInt32(max(1, min(255, Int(p * 256.0))))
    }
]

public struct ContextModel {
    public var pStateIdx: UInt8
    public var valMPS: UInt8

    public init() {
        self.pStateIdx = 0
        self.valMPS = 0
    }

    @inline(__always)
    public mutating func update(binVal: UInt8) {
        if binVal == valMPS {
            if pStateIdx < 127 {
                pStateIdx += 1
            }
        } else {
            if pStateIdx > 0 {
                pStateIdx -= 1
            } else {
                valMPS = 1 - valMPS
            }
        }
    }
}

// MARK: - BitWriter for CABAC

public struct CABACBitWriter {
    public var data: [UInt8]
    private var cache: UInt8
    private var bits: UInt8

    public init() {
        self.data = []
        self.cache = 0
        self.bits = 0
    }

    @inline(__always)
    public mutating func writeBit(_ bit: UInt8) {
        if 0 < bit {
            cache |= (1 << (7 - bits))
        }
        bits += 1
        if bits == 8 {
            data.append(cache)
            bits = 0
            cache = 0
        }
    }

    @inline(__always)
    public mutating func flush() {
        if 0 < bits {
            data.append(cache)
            bits = 0
            cache = 0
        }
    }
}

// MARK: - CABAC Encoder

public struct CABACEncoder {
    private var bw: CABACBitWriter
    private var low: UInt32
    private var range: UInt32
    private var bitsPending: Int

    public init() {
        self.bw = CABACBitWriter()
        self.low = 0
        self.range = 510
        self.bitsPending = 0
    }

    @inline(__always)
    private mutating func renormE() {
        while range < 256 {
            if low < 256 {
                putBit(0)
                putPendingBits(1)
            } else if low >= 512 {
                low -= 512
                putBit(1)
                putPendingBits(0)
            } else {
                low -= 256
                bitsPending += 1
            }
            low <<= 1
            range <<= 1
        }
    }

    @inline(__always)
    private mutating func putBit(_ bit: UInt8) {
        bw.writeBit(bit)
    }

    @inline(__always)
    private mutating func putPendingBits(_ bit: UInt8) {
        while bitsPending > 0 {
            bw.writeBit(bit)
            bitsPending -= 1
        }
    }

    @inline(__always)
    public mutating func encodeBin(binVal: UInt8, ctx: inout ContextModel) {
        let qIdx = (range >> 6) & 3
        let rLPS = rangeLPS_table[Int(qIdx)][Int(ctx.pStateIdx)]

        range -= rLPS

        if binVal != ctx.valMPS {
            low += range
            range = rLPS
        }

        ctx.update(binVal: binVal)
        renormE()
    }

    @inline(__always)
    public mutating func encodeBypass(binVal: UInt8) {
        low <<= 1
        if binVal != 0 {
            low += range
        }
        if low >= 1024 {
            low -= 1024
            putBit(1)
            putPendingBits(0)
        } else if low < 512 {
            putBit(0)
            putPendingBits(1)
        } else {
            low -= 512
            bitsPending += 1
        }
    }

    @inline(__always)
    public mutating func encodeTerminal(binVal: UInt8) {
        range -= 2
        if binVal != 0 {
            low += range
            range = 2
        }
        renormE()
    }

    @inline(__always)
    public mutating func flush() {
        bitsPending += 1
        let bit = UInt8((low >> 9) & 1)
        putBit(bit)
        putPendingBits(1 - bit)

        for _ in 0..<8 {
            let b = UInt8((low >> 8) & 1)
            bw.writeBit(b)
            low <<= 1
        }

        bw.flush()
    }

    @inline(__always)
    public func getData() -> [UInt8] {
        return bw.data
    }
}

// MARK: - CABAC Decoder

public struct CABACBitReader {
    private let data: [UInt8]
    private var offset: Int
    private var cache: UInt8
    private var bits: UInt8

    public init(data: [UInt8]) {
        self.data = data
        self.offset = 0
        self.cache = 0
        self.bits = 0
    }

    @inline(__always)
    public mutating func readBit() throws -> UInt8 {
        if bits == 0 {
            if offset >= data.count {
                return 0
            }
            cache = data[offset]
            offset += 1
            bits = 8
        }
        bits -= 1
        let bit = (cache >> bits) & 1
        return bit
    }
}

public struct CABACDecoder {
    private var br: CABACBitReader
    private var range: UInt32
    private var value: UInt32

    public init(data: [UInt8]) throws {
        self.br = CABACBitReader(data: data)
        self.range = 510
        self.value = 0

        for _ in 0..<10 {
            let b = try br.readBit()
            value = (value << 1) | UInt32(b)
        }
    }

    @inline(__always)
    private mutating func renormD() throws {
        while range < 256 {
            range <<= 1
            let b = try br.readBit()
            value = ((value << 1) | UInt32(b)) & 0x3FF
        }
    }

    @inline(__always)
    public mutating func decodeBin(ctx: inout ContextModel) throws -> UInt8 {
        let qIdx = (range >> 6) & 3
        let rLPS = rangeLPS_table[Int(qIdx)][Int(ctx.pStateIdx)]

        range -= rLPS

        let binVal: UInt8
        if value < range {
            binVal = ctx.valMPS
        } else {
            binVal = 1 - ctx.valMPS
            value -= range
            range = rLPS
        }

        ctx.update(binVal: binVal)
        try renormD()

        return binVal
    }

    @inline(__always)
    public mutating func decodeBypass() throws -> UInt8 {
        let b = try br.readBit()
        value = ((value << 1) | UInt32(b)) & 0x3FF

        let binVal: UInt8
        if value >= range {
            binVal = 1
            value -= range
        } else {
            binVal = 0
        }
        return binVal
    }
}
