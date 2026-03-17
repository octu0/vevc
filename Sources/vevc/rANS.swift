import Foundation

let RANS_SCALE_BITS: UInt32 = 14
let RANS_SCALE: UInt32 = 1 << RANS_SCALE_BITS
let RANS_L: UInt32 = 1 << 15
let RANS_XMAX: UInt32 = ((RANS_L << 16) - 1)

// MARK: - rANS Encoder

struct rANSEncoder {
    public private(set) var state: UInt32
    public private(set) var stream: [UInt16]
    
    public init() {
        self.state = RANS_L
        self.stream = []
        self.stream.reserveCapacity(4096)
    }
    
    @inline(__always)
    public mutating func encodeSymbol(cumFreq: UInt32, freq: UInt32) {
        let xMax = RANS_XMAX / RANS_SCALE * freq
        while state > xMax {
            stream.append(UInt16(truncatingIfNeeded: state))
            state >>= 16
        }
        let q = state / freq
        state = (q << RANS_SCALE_BITS) + (state - q * freq) + cumFreq
    }
    
    public mutating func flush() {
        stream.append(UInt16(truncatingIfNeeded: state))
        stream.append(UInt16(truncatingIfNeeded: state >> 16))
    }
    
    public func getBitstream() -> [UInt8] {
        let count = stream.count
        var bytes = [UInt8](repeating: 0, count: count * 2)
        bytes.withUnsafeMutableBufferPointer { ptr in
            var idx = 0
            for i in stride(from: count - 1, through: 0, by: -1) {
                let word = stream[i]
                ptr[idx] = UInt8(truncatingIfNeeded: word >> 8)
                ptr[idx + 1] = UInt8(truncatingIfNeeded: word & 0xFF)
                idx += 2
            }
        }
        return bytes
    }
}

// MARK: - rANS Decoder

struct rANSDecoder {
    public private(set) var state: UInt32
    private let stream: [UInt8]
    private var offset: Int
    
    public init(bitstream: [UInt8]) {
        self.stream = bitstream
        self.offset = 0
        self.state = 0
        
        if bitstream.count >= 4 {
            let b0 = UInt32(bitstream[0])
            let b1 = UInt32(bitstream[1])
            let b2 = UInt32(bitstream[2])
            let b3 = UInt32(bitstream[3])
            let w1 = (b0 << 8) | b1
            let w0 = (b2 << 8) | b3
            self.state = (w1 << 16) | w0
            self.offset = 4
        }
    }
    
    @inline(__always)
    public func getCumulativeFreq() -> UInt32 {
        return state & (RANS_SCALE - 1)
    }
    
    @inline(__always)
    public mutating func advanceSymbol(cumFreq: UInt32, freq: UInt32) {
        let mask = RANS_SCALE - 1
        state = freq * (state >> RANS_SCALE_BITS) + (state & mask) - cumFreq
        
        while state < RANS_L {
            if offset + 1 < stream.count {
                let b0 = UInt32(stream[offset])
                let b1 = UInt32(stream[offset + 1])
                let word = (b0 << 8) | b1
                offset += 2
                state = (state << 16) | word
            } else {
                state = (state << 16)
            }
        }
    }
}

// MARK: - Interleaved 4-way rANS Encoder

struct Interleaved4rANSEncoder {
    public private(set) var states: (UInt32, UInt32, UInt32, UInt32)
    public private(set) var stream: [UInt16]
    
    public init() {
        self.states = (RANS_L, RANS_L, RANS_L, RANS_L)
        self.stream = []
        self.stream.reserveCapacity(4096)
    }
    
    @inline(__always)
    public mutating func encodeSymbol(lane: Int, cumFreq: UInt32, freq: UInt32) {
        let xMax = RANS_XMAX / RANS_SCALE * freq
        
        switch lane {
        case 0:
            while states.0 > xMax {
                stream.append(UInt16(truncatingIfNeeded: states.0))
                states.0 >>= 16
            }
            let q = states.0 / freq
            states.0 = (q << RANS_SCALE_BITS) + (states.0 - q * freq) + cumFreq
        case 1:
            while states.1 > xMax {
                stream.append(UInt16(truncatingIfNeeded: states.1))
                states.1 >>= 16
            }
            let q = states.1 / freq
            states.1 = (q << RANS_SCALE_BITS) + (states.1 - q * freq) + cumFreq
        case 2:
            while states.2 > xMax {
                stream.append(UInt16(truncatingIfNeeded: states.2))
                states.2 >>= 16
            }
            let q = states.2 / freq
            states.2 = (q << RANS_SCALE_BITS) + (states.2 - q * freq) + cumFreq
        case 3:
            while states.3 > xMax {
                stream.append(UInt16(truncatingIfNeeded: states.3))
                states.3 >>= 16
            }
            let q = states.3 / freq
            states.3 = (q << RANS_SCALE_BITS) + (states.3 - q * freq) + cumFreq
        default:
            break
        }
    }
    
    public mutating func flush() {
        stream.append(UInt16(truncatingIfNeeded: states.3))
        stream.append(UInt16(truncatingIfNeeded: states.3 >> 16))
        stream.append(UInt16(truncatingIfNeeded: states.2))
        stream.append(UInt16(truncatingIfNeeded: states.2 >> 16))
        stream.append(UInt16(truncatingIfNeeded: states.1))
        stream.append(UInt16(truncatingIfNeeded: states.1 >> 16))
        stream.append(UInt16(truncatingIfNeeded: states.0))
        stream.append(UInt16(truncatingIfNeeded: states.0 >> 16))
    }
    
    public func getBitstream() -> [UInt8] {
        let count = stream.count
        var bytes = [UInt8](repeating: 0, count: count * 2)
        bytes.withUnsafeMutableBufferPointer { ptr in
            var idx = 0
            for i in stride(from: count - 1, through: 0, by: -1) {
                let word = stream[i]
                ptr[idx] = UInt8(truncatingIfNeeded: word >> 8)
                ptr[idx + 1] = UInt8(truncatingIfNeeded: word & 0xFF)
                idx += 2
            }
        }
        return bytes
    }
}

// MARK: - Interleaved 4-way rANS Decoder

struct Interleaved4rANSDecoder {
    public private(set) var states: (UInt32, UInt32, UInt32, UInt32)
    private let stream: [UInt8]
    private var offset: Int
    
    public init(bitstream: [UInt8]) {
        // Add padding to eliminate bounds checks in readWord
        var padded = bitstream
        padded.append(contentsOf: [0, 0, 0, 0, 0, 0, 0, 0])
        self.stream = padded
        self.offset = 0
        self.states = (RANS_L, RANS_L, RANS_L, RANS_L)
        
        guard bitstream.count >= 16 else { return }
        
        @inline(__always)
        func readState(_ off: Int) -> UInt32 {
            let b0 = UInt32(padded[off])
            let b1 = UInt32(padded[off + 1])
            let b2 = UInt32(padded[off + 2])
            let b3 = UInt32(padded[off + 3])
            return ((b0 << 8) | b1) << 16 | ((b2 << 8) | b3)
        }
        
        self.states.0 = readState(0)
        self.states.1 = readState(4)
        self.states.2 = readState(8)
        self.states.3 = readState(12)
        self.offset = 16
    }
    
    @inline(__always)
    public func getCumulativeFreq(lane: Int) -> UInt32 {
        let mask = RANS_SCALE - 1
        switch lane {
        case 0: return states.0 & mask
        case 1: return states.1 & mask
        case 2: return states.2 & mask
        case 3: return states.3 & mask
        default: return 0
        }
    }
    
    @inline(__always)
    private mutating func readWord() -> UInt32 {
        // bounds check eliminated by padding in init
        let b0 = UInt32(stream[offset])
        let b1 = UInt32(stream[offset + 1])
        offset += 2
        return (b0 << 8) | b1
    }
    
    @inline(__always)
    public mutating func advanceSymbol(lane: Int, cumFreq: UInt32, freq: UInt32) {
        let mask = RANS_SCALE - 1
        
        switch lane {
        case 0:
            states.0 = freq * (states.0 >> RANS_SCALE_BITS) + (states.0 & mask) - cumFreq
            while states.0 < RANS_L { states.0 = (states.0 << 16) | readWord() }
        case 1:
            states.1 = freq * (states.1 >> RANS_SCALE_BITS) + (states.1 & mask) - cumFreq
            while states.1 < RANS_L { states.1 = (states.1 << 16) | readWord() }
        case 2:
            states.2 = freq * (states.2 >> RANS_SCALE_BITS) + (states.2 & mask) - cumFreq
            while states.2 < RANS_L { states.2 = (states.2 << 16) | readWord() }
        case 3:
            states.3 = freq * (states.3 >> RANS_SCALE_BITS) + (states.3 & mask) - cumFreq
            while states.3 < RANS_L { states.3 = (states.3 << 16) | readWord() }
        default:
            break
        }
    }
}

struct BypassWriter {
    public private(set) var bytes: [UInt8]
    private var buffer: UInt32
    private var bitsInBuffer: Int
    
    public init() {
        self.bytes = []
        self.bytes.reserveCapacity(256)
        self.buffer = 0
        self.bitsInBuffer = 0
    }
    
    @inline(__always)
    public mutating func writeBit(_ bit: Bool) {
        buffer = (buffer << 1) | (bit ? 1 : 0)
        bitsInBuffer += 1
        if bitsInBuffer == 32 {
            bytes.append(UInt8(truncatingIfNeeded: buffer >> 24))
            bytes.append(UInt8(truncatingIfNeeded: buffer >> 16))
            bytes.append(UInt8(truncatingIfNeeded: buffer >> 8))
            bytes.append(UInt8(truncatingIfNeeded: buffer))
            buffer = 0
            bitsInBuffer = 0
        }
    }
    
    @inline(__always)
    public mutating func writeBits(_ value: UInt16, count: Int) {
        guard 0 < count else { return }
        buffer = (buffer << count) | UInt32(value & ((1 << count) - 1))
        bitsInBuffer += count
        while bitsInBuffer >= 8 {
            bitsInBuffer -= 8
            bytes.append(UInt8(truncatingIfNeeded: buffer >> bitsInBuffer))
            if bitsInBuffer > 0 {
                buffer &= (1 << bitsInBuffer) - 1
            } else {
                buffer = 0
            }
        }
    }
    
    public mutating func flush() {
        guard 0 < bitsInBuffer else { return }
        while bitsInBuffer >= 8 {
            bitsInBuffer -= 8
            bytes.append(UInt8(truncatingIfNeeded: buffer >> bitsInBuffer))
            if bitsInBuffer > 0 {
                buffer &= (1 << bitsInBuffer) - 1
            } else {
                buffer = 0
            }
        }
        if 0 < bitsInBuffer {
            let shifted = buffer << (8 - bitsInBuffer)
            bytes.append(UInt8(truncatingIfNeeded: shifted))
            buffer = 0
            bitsInBuffer = 0
        }
    }
}

// MARK: - Bypass Reader

struct BypassReader {
    private let bytes: [UInt8]
    private var byteOffset: Int
    private var buffer: UInt32
    private var bitsInBuffer: Int
    
    public init(data: [UInt8]) {
        var padded = data
        padded.append(contentsOf: [0, 0, 0, 0, 0, 0, 0, 0])
        self.bytes = padded
        self.byteOffset = 0
        self.buffer = 0
        self.bitsInBuffer = 0
    }
    
    @inline(__always)
    private mutating func ensureBits(_ needed: Int) {
        while bitsInBuffer < needed {
            buffer = (buffer << 8) | UInt32(bytes[byteOffset])
            byteOffset += 1
            bitsInBuffer += 8
        }
    }
    
    @inline(__always)
    public mutating func readBit() -> Bool {
        ensureBits(1)
        bitsInBuffer -= 1
        let bit = (buffer >> bitsInBuffer) & 1
        buffer &= (1 << bitsInBuffer) &- 1
        return bit == 1
    }
    
    @inline(__always)
    public mutating func readBits(count: Int) -> UInt16 {
        guard 0 < count else { return 0 }
        ensureBits(count)
        bitsInBuffer -= count
        let value = (buffer >> bitsInBuffer) & ((1 << count) - 1)
        buffer &= (1 << bitsInBuffer) &- 1
        return UInt16(value)
    }
    
    public var consumedBytes: Int {
        let totalBitsRead = byteOffset * 8 - bitsInBuffer
        return (totalBitsRead + 7) / 8
    }
}

// MARK: - rANS Probability Model

struct rANSModel {
    public private(set) var sigFreq: UInt32
    public private(set) var tokenFreqs: [UInt32]
    public private(set) var tokenCumFreqs: [UInt32]
    public private(set) var tokenLUT: [UInt8]
    
    public init() {
        self.sigFreq = RANS_SCALE / 2
        self.tokenFreqs = Array(repeating: RANS_SCALE / 32, count: 32)
        self.tokenCumFreqs = (0..<32).map { UInt32($0) * (RANS_SCALE / 32) }
        self.tokenLUT = [UInt8](repeating: 0, count: Int(RANS_SCALE))
        buildLUT()
    }
    
    public init(sigFreq: UInt32, tokenFreqs: [UInt32]) {
        self.sigFreq = sigFreq
        self.tokenFreqs = tokenFreqs
        self.tokenCumFreqs = [UInt32](repeating: 0, count: 32)
        var sum: UInt32 = 0
        for i in 0..<32 {
            self.tokenCumFreqs[i] = sum
            sum += tokenFreqs[i]
        }
        self.tokenLUT = [UInt8](repeating: 0, count: Int(RANS_SCALE))
        buildLUT()
    }
    
    private mutating func buildLUT() {
        tokenLUT.withUnsafeMutableBufferPointer { ptr in
            for sym in 0..<32 {
                let start = Int(tokenCumFreqs[sym])
                let end = start + Int(tokenFreqs[sym])
                let s = UInt8(sym)
                for j in start..<min(end, Int(RANS_SCALE)) {
                    ptr[j] = s
                }
            }
        }
    }
    
    public mutating func normalize(sigCounts: [Int], tokenCounts: [Int]) {
        let totalSig = sigCounts[0] + sigCounts[1]
        if totalSig == 0 {
            self.sigFreq = RANS_SCALE / 2
        } else {
            let f = UInt32((Int(RANS_SCALE) * sigCounts[1]) / totalSig)
            self.sigFreq = max(1, min(RANS_SCALE - 1, f))
        }
        
        let totalTokens = tokenCounts.reduce(0, +)
        if totalTokens == 0 {
            self.tokenFreqs = Array(repeating: RANS_SCALE / 32, count: 32)
        } else {
            var sum: UInt32 = 0
            for i in 0..<32 {
                let count = tokenCounts[i]
                if count == 0 {
                    self.tokenFreqs[i] = 1
                } else {
                    let maxVal = RANS_SCALE - 32
                    self.tokenFreqs[i] = max(1, UInt32((Int(maxVal) * count) / totalTokens))
                }
                sum += self.tokenFreqs[i]
            }
            
            if sum != RANS_SCALE {
                var maxIdx = 0
                var maxVal = self.tokenFreqs[0]
                for i in 1..<32 {
                    if self.tokenFreqs[i] > maxVal {
                        maxVal = self.tokenFreqs[i]
                        maxIdx = i
                    }
                }
                if sum < RANS_SCALE {
                    self.tokenFreqs[maxIdx] += (RANS_SCALE - sum)
                } else if self.tokenFreqs[maxIdx] > (sum - RANS_SCALE) + 1 {
                    self.tokenFreqs[maxIdx] -= (sum - RANS_SCALE)
                }
            }
        }
        
        var cumSum: UInt32 = 0
        for i in 0..<32 {
            self.tokenCumFreqs[i] = cumSum
            cumSum += self.tokenFreqs[i]
        }
        
        buildLUT()
    }
    
    @inline(__always)
    public func findToken(cf: UInt32) -> (token: UInt8, freq: UInt32, cumFreq: UInt32) {
        let sym = Int(tokenLUT[Int(cf)])
        return (UInt8(sym), tokenFreqs[sym], tokenCumFreqs[sym])
    }
}
