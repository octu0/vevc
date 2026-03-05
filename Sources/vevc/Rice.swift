// MARK: - Rice Coding

// MARK: - BitWriter

public struct BitWriter {
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
    public mutating func writeBits(val: UInt16, n: UInt8) {
        for i in 0..<n {
            let bit = ((val >> (n - 1 - i)) & 1)
            writeBit(UInt8(bit))
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

// MARK: - RiceWriter

public struct RiceWriter {
    private var bw: BitWriter
    private let maxVal: UInt16
    private var zeroCount: UInt16
    private var lastK: UInt8
    
    public init(bw: BitWriter) {
        self.bw = bw
        self.maxVal = UInt16.max
        self.zeroCount = 0
        self.lastK = 0
    }
    
    @inline(__always)
    internal mutating func writePrimitive(val: UInt16, k: UInt8) {
        let m = (UInt16(1) << k)
        let q = (val / m)
        let r = (val % m)
        
        for _ in 0..<q {
            bw.writeBit(1)
        }
        bw.writeBit(0)
        
        bw.writeBits(val: r, n: k)
    }
    
    @inline(__always)
    public mutating func write(val: UInt16, k: UInt8) {
        if val == 0 {
            let currentK = (zeroCount == 0) ? k : lastK
            if zeroCount == maxVal {
                flushZeros(k: currentK)
                // flushZeros resets zeroCount to 0
            }
            zeroCount += 1
            lastK = currentK
            return
        }
        
        if 0 < zeroCount {
             flushZeros(k: lastK)
        }
        
        lastK = k
        writePrimitive(val: val, k: k)
    }
    
    @inline(__always)
    public mutating func flushZeros(k: UInt8) {
        if zeroCount == 0 {
            return
        }
        writePrimitive(val: 0, k: k)
        
        let escapeVal: UInt16 = 255
        if zeroCount < escapeVal {
            writePrimitive(val: zeroCount, k: k)
        } else {
            writePrimitive(val: escapeVal, k: k)
            bw.writeBits(val: zeroCount, n: 16)
        }
        
        zeroCount = 0
    }
    
    @inline(__always)
    public mutating func flushRice() {
        if 0 < zeroCount {
            flushZeros(k: lastK)
        }
    }
    
    @inline(__always)
    public mutating func flush() {
        flushRice()
        bw.flush()
    }
    
    @inline(__always)
    internal mutating func extractWriter() -> BitWriter {
        return bw
    }
    
    @inline(__always)
    public static func withWriter(_ bw: inout BitWriter, flushBits: Bool = true, body: (inout RiceWriter) throws -> Void) rethrows {
        var rw = RiceWriter(bw: bw)
        try body(&rw)
        if flushBits {
            rw.flush()
        } else {
            rw.flushRice()
        }
        bw = rw.extractWriter()
    }
}

// MARK: - BitReader

public struct BitReader {
    private let data: [UInt8]
    private var offset: Int
    private var cache: UInt8
    private var bits: UInt8
    private let dataCount: Int
    
    public init(data: [UInt8]) {
        self.data = data
        self.offset = 0
        self.cache = 0
        self.bits = 0
        self.dataCount = data.count
    }
    
    @inline(__always)
    public mutating func readBit() throws -> UInt8 {
        if bits == 0 {
            if dataCount <= offset {
                throw DecodeError.eof
            }
            cache = data[offset]
            offset += 1
            bits = 8
        }
        bits -= 1
        let bit = ((cache >> bits) & 1)
        return bit
    }
    
    @inline(__always)
    public mutating func readBits(n: UInt8) throws -> UInt16 {
        var val: UInt16 = 0
        for _ in 0..<n {
            let bit = try readBit()
            val = ((val << 1) | UInt16(bit))
        }
        return val
    }
}

// MARK: - RiceReader

public struct RiceReader {
    private var br: BitReader
    private var pendingZeros: Int
    
    public init(br: BitReader) {
        self.br = br
        self.pendingZeros = 0
    }
    
    @inline(__always)
    internal mutating func readPrimitive(k: UInt8) throws -> UInt16 {
        var q: UInt16 = 0
        while true {
            let bit = try br.readBit()
            if bit == 0 {
                break
            }
            q += 1
        }
        
        let rem64 = try br.readBits(n: k)
        let val = ((q << k) | rem64)
        return val
    }
    
    @inline(__always)
    public mutating func read(k: UInt8) throws -> UInt16 {
        if 0 < pendingZeros {
            pendingZeros -= 1
            return 0
        }
        
        let val = try readPrimitive(k: k)
        
        if val == 0 {
            let count = try readPrimitive(k: k)
            let escapeVal: UInt16 = 255
            if count == escapeVal {
                let actualCount = try br.readBits(n: 16)
                pendingZeros = (Int(actualCount) - 1)
            } else {
                pendingZeros = (Int(count) - 1)
            }
            return 0
        }
        
        return val
    }
    
    @inline(__always)
    internal mutating func extractReader() -> BitReader {
        return br
    }
    
    @inline(__always)
    public static func withReader(_ br: inout BitReader, body: (inout RiceReader) throws -> Void) rethrows {
        var rr = RiceReader(br: br)
        try body(&rr)
        br = rr.extractReader()
    }
}

