// 14-bit scale balances precision vs compression efficiency
let rANSScaleBits: UInt32 = 14
let rANSScale: UInt32 = 1 << rANSScaleBits
let rANSL: UInt32 = 1 << 15
let rANSXMax: UInt32 = (rANSL >> rANSScaleBits) << 16

// MARK: - Static rANS Frequency Tables
// static tables eliminate per-stream frequency table headers (~120B),
// reducing compression overhead for small blocks
//
// Token mapping:
//   val tokens: 0..31 = values ±1..±16 (even=positive, odd=negative)
//   val tokens: 32..63 = values ±17+ (exp-golomb)
//   run tokens: 0..31 = run lengths 0..31
//   run tokens: 32..63 = run lengths 32+ (exp-golomb)

/// Build static rANS model from predetermined frequency data.
/// Normalizes the provided raw frequency array to sum exactly to rANSScale.
@inline(__always)
internal func buildStaticModel(rawFreqs: [UInt32]) -> rANSModel {
    var freqs = rawFreqs
    let sum: UInt32 = freqs.reduce(0, +)
    
    // rounding error is absorbed by the largest-frequency element
    // to minimize impact on the rest of the distribution
    if sum != rANSScale {
        var maxIdx = 0
        var maxVal: UInt32 = 0
        for i in 0..<64 {
            if maxVal < freqs[i] {
                maxVal = freqs[i]
                maxIdx = i
            }
        }
        if sum < rANSScale {
            freqs[maxIdx] += (rANSScale - sum)
        }
        if rANSScale < sum {
            var diff = sum - rANSScale
            while 0 < diff {
                var currentMaxIdx = 0
                var currentMaxVal = freqs[0]
                for i in 1..<64 {
                    if currentMaxVal < freqs[i] {
                        currentMaxVal = freqs[i]
                        currentMaxIdx = i
                    }
                }
                // freq <= 1 cannot be reduced further without causing division by zero during decode
                if currentMaxVal <= 1 { break }
                freqs[currentMaxIdx] -= 1
                diff -= 1
            }
        }
    }
    
    return rANSModel(tokenFreqs: freqs)
}

final class StaticRANSModels: @unchecked Sendable {
    static let shared = StaticRANSModels()

    var runModel0 = buildStaticModel(rawFreqs: [
        2695, 1335, 1025, 896, 982, 785, 633, 692, 721, 519, 441, 428, 712, 489, 420, 476,
        113, 79, 70, 64, 63, 59, 54, 105, 88, 65, 59, 57, 49, 46, 45, 86,
        82, 55, 46, 43, 43, 41, 37, 73, 67, 47, 41, 40, 39, 32, 34, 66,
        68, 47, 40, 36, 34, 33, 32, 66, 147, 89, 81, 73, 64, 58, 62, 288,
    ])

    var valModel0 = buildStaticModel(rawFreqs: [
        1, 7838, 1949, 6104, 120, 109, 48, 99, 25, 22, 15, 16, 8, 7, 4, 4,
        2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var runModel1 = buildStaticModel(rawFreqs: [
        6833, 1908, 1205, 1200, 564, 477, 497, 779, 271, 202, 169, 165, 142, 150, 164, 224,
        97, 69, 54, 48, 46, 45, 46, 61, 40, 35, 34, 33, 33, 34, 36, 43,
        30, 27, 23, 21, 19, 19, 18, 22, 17, 16, 16, 15, 15, 14, 14, 16,
        13, 12, 11, 11, 9, 8, 8, 7, 7, 8, 8, 8, 7, 7, 6, 214,
    ])

    var valModel1 = buildStaticModel(rawFreqs: [
        1, 7371, 1510, 6504, 228, 278, 76, 248, 33, 42, 15, 32, 8, 12, 5, 6,
        1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var runModel2 = buildStaticModel(rawFreqs: [
        1501, 970, 864, 907, 793, 675, 656, 810, 659, 534, 502, 556, 671, 549, 519, 666,
        130, 96, 88, 88, 81, 82, 81, 163, 109, 84, 77, 78, 68, 71, 69, 145,
        89, 66, 61, 62, 59, 60, 63, 128, 81, 62, 58, 58, 56, 56, 57, 131,
        85, 62, 60, 56, 55, 57, 60, 132, 189, 137, 130, 122, 119, 117, 121, 364,
    ])

    var valModel2 = buildStaticModel(rawFreqs: [
        1, 7494, 162, 8474, 76, 52, 33, 47, 14, 7, 6, 5, 2, 2, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var runModel3 = buildStaticModel(rawFreqs: [
        4629, 1566, 1210, 1523, 726, 615, 707, 1381, 398, 279, 233, 223, 188, 184, 198, 292,
        138, 99, 83, 77, 73, 71, 72, 104, 63, 53, 48, 45, 44, 45, 47, 66,
        40, 34, 29, 29, 28, 28, 28, 37, 24, 21, 21, 21, 19, 18, 19, 24,
        16, 16, 14, 14, 13, 12, 13, 14, 10, 9, 8, 9, 7, 6, 6, 279,
    ])

    var valModel3 = buildStaticModel(rawFreqs: [
        1, 7638, 33, 8352, 16, 134, 11, 150, 8, 10, 6, 11, 2, 1, 1, 1,
        2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var dpcmRunModel = buildStaticModel(rawFreqs: [
        6358, 2738, 1575, 1049,  757,  582,  490,  392,
         298,  254,  226,  200,  192,  199,  269,  209,
          45,   31,   25,   21,   18,   17,   19,   18,
          15,   13,   14,   14,   15,   19,   36,   35,
          10,    7,   11,   19,   52,   47,   34,   25,
          17,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,
    ])

    var dpcmValModel = buildStaticModel(rawFreqs: [
        7247, 7891,  495,  516,   90,   91,   19,   20,
           4,    4,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,
    ])

    var lscpRunModel = buildStaticModel(rawFreqs: [
        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
          1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
          1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
    ])
}

// MARK: - rANS Probability Model
// why: LUT reverse-lookup reduces symbol search from O(log n) binary search to O(1)

struct rANSModel {
    private(set) var tokenFreqs: [UInt32]
    private(set) var tokenCumFreqs: [UInt32]
    private(set) var tokenLUT: [UInt8]
    
    init(buildLUT: Bool = true) {
        self.tokenFreqs = Array(repeating: rANSScale / 64, count: 64)
        self.tokenCumFreqs = (0..<64).map { UInt32($0) * (rANSScale / 64) }
        self.tokenLUT = if buildLUT { [UInt8](repeating: 0, count: Int(rANSScale)) } else { [] }
        if buildLUT { self.buildLUT() }
    }
    
    init(tokenFreqs: [UInt32]) {
        self.tokenFreqs = tokenFreqs
        self.tokenCumFreqs = [UInt32](repeating: 0, count: 64)
        var sum: UInt32 = 0
        for i in 0..<64 {
            self.tokenCumFreqs[i] = sum
            sum += tokenFreqs[i]
        }
        self.tokenLUT = [UInt8](repeating: 0, count: Int(rANSScale))
        buildLUT()
    }
    
    @inline(__always)
    private mutating func buildLUT() {
        guard tokenLUT.isEmpty != true else { return }
        tokenLUT.withUnsafeMutableBufferPointer { ptr in
            for sym in 0..<64 {
                let start = Int(tokenCumFreqs[sym])
                let end = start + Int(tokenFreqs[sym])
                let s = UInt8(sym)
                for j in start..<min(end, Int(rANSScale)) {
                    ptr[j] = s
                }
            }
        }
    }
    
    @inline(__always)
    mutating func normalize(tokenCounts: [Int]) {
        
        let totalTokens = tokenCounts.reduce(0, +)
        if totalTokens == 0 {
            self.tokenFreqs = Array(repeating: rANSScale / 64, count: 64)
        } else {
            // Count unused tokens to maximize scale allocation for valid tokens
            var zeroCount: UInt32 = 0
            for i in 0..<64 {
                if tokenCounts[i] == 0 { zeroCount += 1 }
            }
            // Assign minimum freq=1 to unused tokens, allocating the rest to valid tokens
            let availableScale = rANSScale - zeroCount
            
            var sum: UInt32 = 0
            for i in 0..<64 {
                let count = tokenCounts[i]
                if count == 0 {
                    self.tokenFreqs[i] = 1
                } else {
                    self.tokenFreqs[i] = max(1, UInt32((Int(availableScale) * count) / totalTokens))
                }
                sum += self.tokenFreqs[i]
            }
            
            var maxIdx = 0
            var maxVal = self.tokenFreqs[0]
            for i in 1..<64 {
                if maxVal < self.tokenFreqs[i] {
                    maxVal = self.tokenFreqs[i]
                    maxIdx = i
                }
            }
            
            // why: absorb deficit into the largest frequency to preserve distribution shape
            if sum < rANSScale {
                self.tokenFreqs[maxIdx] += (rANSScale - sum)
            }
            if rANSScale < sum {
                var diff = sum - rANSScale
                while 0 < diff {
                    var currentMaxIdx = 0
                    var currentMaxVal = self.tokenFreqs[0]
                    for i in 1..<64 {
                        if currentMaxVal < self.tokenFreqs[i] {
                            currentMaxVal = self.tokenFreqs[i]
                            currentMaxIdx = i
                        }
                    }
                    // why: freq <= 1 cannot be reduced further without causing division by zero during decode
                    if currentMaxVal <= 1 { break }
                    self.tokenFreqs[currentMaxIdx] -= 1
                    diff -= 1
                }
            }
        }
        
        var cumSum: UInt32 = 0
        for i in 0..<64 {
            self.tokenCumFreqs[i] = cumSum
            cumSum += self.tokenFreqs[i]
        }
        
        buildLUT()
    }
    
    @inline(__always)
    func findToken(cf: UInt32) -> (token: UInt8, freq: UInt32, cumFreq: UInt32) {
        let sym = Int(tokenLUT[Int(cf)])
        return (UInt8(sym), tokenFreqs[sym], tokenCumFreqs[sym])
    }
}

// MARK: - rANS Encoder

struct rANSEncoder {
    private(set) var state: UInt32
    private(set) var stream: [UInt16]
    
    init() {
        self.state = rANSL
        self.stream = []
        self.stream.reserveCapacity(4096)
    }
    
    @inline(__always)
    mutating func encodeSymbol(cumFreq: UInt32, freq: UInt32) {
        let xMax = rANSXMax * freq
        while xMax <= state {
            stream.append(UInt16(truncatingIfNeeded: state))
            state >>= 16
        }
        let q = state / freq
        state = (q << rANSScaleBits) + (state - (q * freq)) + cumFreq
    }
    
    @inline(__always)
    mutating func flush() {
        stream.append(UInt16(truncatingIfNeeded: state))
        stream.append(UInt16(truncatingIfNeeded: state >> 16))
    }
    
    @inline(__always)
    func getBitstream() -> [UInt8] {
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
    private(set) var state: UInt32
    private let base: UnsafePointer<UInt8>
    private let count: Int
    private var offset: Int
    
    init(base: UnsafePointer<UInt8>, count: Int) {
        self.base = base
        self.count = count
        self.offset = 0
        self.state = 0
        
        if 4 <= count {
            let b0 = UInt32(base[0])
            let b1 = UInt32(base[1])
            let b2 = UInt32(base[2])
            let b3 = UInt32(base[3])
            let w1 = (b0 << 8) | b1
            let w0 = (b2 << 8) | b3
            self.state = (w1 << 16) | w0
            self.offset = 4
        }
    }
    
    @inline(__always)
    func getCumulativeFreq() -> UInt32 {
        return state & (rANSScale - 1)
    }
    
    @inline(__always)
    mutating func advanceSymbol(cumFreq: UInt32, freq: UInt32) {
        let mask = rANSScale - 1
        state = freq * (state >> rANSScaleBits) + (state & mask) - cumFreq
        
        while state < rANSL {
            if offset + 1 < count {
                let b0 = UInt32(base[offset])
                let b1 = UInt32(base[offset + 1])
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
    private(set) var states: (UInt32, UInt32, UInt32, UInt32)
    private(set) var stream: [UInt16]
    
    init() {
        self.states = (rANSL, rANSL, rANSL, rANSL)
        self.stream = []
        self.stream.reserveCapacity(4096)
    }
    
    @inline(__always)
    mutating func encodeSymbol(lane: Int, cumFreq: UInt32, freq: UInt32) {
        let xMax = rANSXMax * freq
        
        switch lane {
        case 0:
            while xMax <= states.0 {
                stream.append(UInt16(truncatingIfNeeded: states.0))
                states.0 >>= 16
            }
            let q = states.0 / freq
            states.0 = (q << rANSScaleBits) + (states.0 - (q * freq)) + cumFreq
        case 1:
            while xMax <= states.1 {
                stream.append(UInt16(truncatingIfNeeded: states.1))
                states.1 >>= 16
            }
            let q = states.1 / freq
            states.1 = (q << rANSScaleBits) + (states.1 - (q * freq)) + cumFreq
        case 2:
            while xMax <= states.2 {
                stream.append(UInt16(truncatingIfNeeded: states.2))
                states.2 >>= 16
            }
            let q = states.2 / freq
            states.2 = (q << rANSScaleBits) + (states.2 - (q * freq)) + cumFreq
        case 3:
            while xMax <= states.3 {
                stream.append(UInt16(truncatingIfNeeded: states.3))
                states.3 >>= 16
            }
            let q = states.3 / freq
            states.3 = (q << rANSScaleBits) + (states.3 - (q * freq)) + cumFreq
        default:
            break
        }
    }
    
    @inline(__always)
    mutating func flush() {
        stream.append(UInt16(truncatingIfNeeded: states.3))
        stream.append(UInt16(truncatingIfNeeded: states.3 >> 16))
        stream.append(UInt16(truncatingIfNeeded: states.2))
        stream.append(UInt16(truncatingIfNeeded: states.2 >> 16))
        stream.append(UInt16(truncatingIfNeeded: states.1))
        stream.append(UInt16(truncatingIfNeeded: states.1 >> 16))
        stream.append(UInt16(truncatingIfNeeded: states.0))
        stream.append(UInt16(truncatingIfNeeded: states.0 >> 16))
    }
    
    @inline(__always)
    func getBitstream() -> [UInt8] {
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
    private(set) var states: (UInt32, UInt32, UInt32, UInt32)
    private let base: UnsafePointer<UInt8>
    private let count: Int
    private var offset: Int
    
    init(base: UnsafePointer<UInt8>, count: Int) {
        self.base = base
        self.count = count
        self.offset = 0
        self.states = (rANSL, rANSL, rANSL, rANSL)
        
        guard 16 <= count else { return }
        
        @inline(__always)
        func readState(_ off: Int) -> UInt32 {
            let b0 = UInt32(base[off])
            let b1 = UInt32(base[off + 1])
            let b2 = UInt32(base[off + 2])
            let b3 = UInt32(base[off + 3])
            return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
        }
        
        self.states.0 = readState(0)
        self.states.1 = readState(4)
        self.states.2 = readState(8)
        self.states.3 = readState(12)
        self.offset = 16
    }
    
    @inline(__always)
    func getCumulativeFreq(lane: Int) -> UInt32 {
        let mask = rANSScale - 1
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
        if offset + 1 < count {
            let b0 = UInt32(base[offset])
            let b1 = UInt32(base[offset + 1])
            offset += 2
            return (b0 << 8) | b1
        }
        return 0
    }
    
    @inline(__always)
    mutating func advanceSymbol(lane: Int, cumFreq: UInt32, freq: UInt32) {
        let mask = rANSScale - 1
        
        switch lane {
        case 0:
            states.0 = freq * (states.0 >> rANSScaleBits) + (states.0 & mask) - cumFreq
            while states.0 < rANSL { states.0 = (states.0 << 16) | readWord() }
        case 1:
            states.1 = freq * (states.1 >> rANSScaleBits) + (states.1 & mask) - cumFreq
            while states.1 < rANSL { states.1 = (states.1 << 16) | readWord() }
        case 2:
            states.2 = freq * (states.2 >> rANSScaleBits) + (states.2 & mask) - cumFreq
            while states.2 < rANSL { states.2 = (states.2 << 16) | readWord() }
        case 3:
            states.3 = freq * (states.3 >> rANSScaleBits) + (states.3 & mask) - cumFreq
            while states.3 < rANSL { states.3 = (states.3 << 16) | readWord() }
        default:
            break
        }
    }
}


struct BypassWriter {
    private(set) var bytes: [UInt8]
    private var buffer: UInt64
    private var bitsInBuffer: Int
    
    init() {
        self.bytes = []
        self.bytes.reserveCapacity(1024)
        self.buffer = 0
        self.bitsInBuffer = 0
    }
    
    @inline(__always)
    mutating func writeBit(_ bit: Bool) {
        let b: UInt64 = if bit { 1 } else { 0 }
        buffer = (buffer << 1) | b
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
    mutating func writeBits(_ value: UInt32, count: Int) {
        guard 0 < count else { return }
        buffer = (buffer << count) | UInt64(value & ((1 << count) - 1))
        bitsInBuffer += count
        while 8 <= bitsInBuffer {
            bitsInBuffer -= 8
            bytes.append(UInt8(truncatingIfNeeded: buffer >> bitsInBuffer))
            if 0 < bitsInBuffer {
                buffer &= (1 << bitsInBuffer) - 1
            } else {
                buffer = 0
            }
        }
    }
    
    @inline(__always)
    mutating func flush() {
        guard 0 < bitsInBuffer else { return }
        while 8 <= bitsInBuffer {
            bitsInBuffer -= 8
            bytes.append(UInt8(truncatingIfNeeded: buffer >> bitsInBuffer))
            if 0 < bitsInBuffer {
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
    private let base: UnsafePointer<UInt8>
    private let count: Int
    private var byteOffset: Int
    private var buffer: UInt64
    private var bitsInBuffer: Int
    
    init(base: UnsafePointer<UInt8>, count: Int) {
        self.base = base
        self.count = count
        self.byteOffset = 0
        self.buffer = 0
        self.bitsInBuffer = 0
    }
    
    @inline(__always)
    private mutating func ensureBits(_ needed: Int) {
        while bitsInBuffer < needed {
            if byteOffset < count {
                buffer = (buffer << 8) | UInt64(base[byteOffset])
                byteOffset += 1
            } else {
                buffer = (buffer << 8) | 0
            }
            bitsInBuffer += 8
        }
    }
    
    
    @inline(__always)
    mutating func readBit() -> Bool {
        ensureBits(1)
        bitsInBuffer -= 1
        let bit = (buffer >> bitsInBuffer) & 1
        buffer &= (1 << bitsInBuffer) &- 1
        return bit == 1
    }
    
    @inline(__always)
    mutating func readBits(count: Int) -> UInt32 {
        guard 0 < count else { return 0 }
        let safeCount = min(count, 32)
        ensureBits(safeCount)
        bitsInBuffer -= safeCount
        let mask = (UInt64(1) << safeCount) - 1
        let value = (buffer >> bitsInBuffer) & mask
        buffer &= (1 << bitsInBuffer) &- 1
        return UInt32(truncatingIfNeeded: value)
    }
    
    var consumedBytes: Int {
        let totalBitsRead = byteOffset * 8 - bitsInBuffer
        return (totalBitsRead + 7) / 8
    }
}

// why: |val| <= 15 maps to token only (no bypass bits, covers most frequent range)
// |val| >= 16 splits into token + variable-length bypass bits
@inline(__always)
func valueTokenize(_ value: Int16) -> (token: UInt8, bypassBits: UInt32, bypassLen: Int) {
    if value == 0 { return (0, 0, 0) }
    let sign = value < 0
    let absValue = UInt16(value.magnitude)
    
    if absValue <= 15 {
        let offset: UInt16 = if sign { 1 } else { 0 }
        let token = UInt8(((absValue - 1) * 2) + 1 + offset)
        return (token, 0, 0)
    }
    
    let v = UInt32(absValue - 16)
    if v == 0 {
        if sign {
            return (32, 1, 1)
        }
        return (32, 0, 1)
    }
    
    let bits = UInt32.bitWidth - v.leadingZeroBitCount
    let subToken = UInt8(bits)
    let bypass = UInt32(v & ((1 << (bits - 1)) - 1))
    let bypassLen = bits - 1
    
    let token = 32 + subToken
    let signBit: UInt32 = if sign { 1 } else { 0 }
    let finalBypass = (bypass << 1) | signBit
    let finalBypassLen = bypassLen + 1
    
    return (token, finalBypass, finalBypassLen)
}

@inline(__always)
func valueDetokenize(token: UInt8, bypassBits: UInt32) -> Int16 {
    if token == 0 { return 0 }
    if token < 32 {
        let t = token - 1
        let absValue = (UInt16(t) / 2) + 1
        let isNegative = (t % 2) == 1
        if isNegative {
            return Int16(bitPattern: 0 &- absValue)
        }
        return Int16(bitPattern: absValue)
    }
    
    let subToken = token - 32
    let sign = (bypassBits & 1) == 1
    let bypass = bypassBits >> 1
    
    let v: UInt32
    if subToken == 0 {
        v = 0
    } else {
        let base: UInt32 = 1 << (UInt32(subToken) - 1)
        v = base | UInt32(bypass)
    }
    
    let absValue = v + 16
    if sign {
        let neg = 0 &- absValue
        return Int16(truncatingIfNeeded: neg)
    } else {
        return Int16(truncatingIfNeeded: absValue)
    }
}

@inline(__always)
func valueTokenizeUnsigned(_ value: UInt32) -> (token: UInt8, bypassBits: UInt32, bypassLen: Int) {
    if value < 32 {
        return (UInt8(value), 0, 0)
    }
    
    let v = value - 32
    if v == 0 {
        return (32, 0, 0)
    }
    
    let bits = UInt32.bitWidth - v.leadingZeroBitCount
    let subToken = UInt8(bits)
    let bypass = UInt32(v & ((1 << (bits - 1)) - 1))
    let bypassLen = bits - 1
    
    return (32 + subToken, bypass, bypassLen)
}

@inline(__always)
func valueDetokenizeUnsigned(token: UInt8, bypassBits: UInt32) -> UInt32 {
    if token < 32 {
        return UInt32(token)
    }
    
    let subToken = token - 32
    let v: UInt32
    if subToken == 0 {
        v = 0
    } else {
        let base: UInt32 = 1 << (UInt32(subToken) - 1)
        v = base | UInt32(bypassBits)
    }
    return v + 32
}

@inline(__always)
func valueBypassLength(for token: UInt8) -> Int {
    if token < 32 { return 0 }
    if token == 32 { return 1 }
    let t = min(token, 63)
    return Int(t - 32)
}

@inline(__always)
func valueBypassLengthUnsigned(for token: UInt8) -> Int {
    if token < 32 { return 0 }
    if token == 32 { return 0 }
    let t = min(token, 63)
    return Int(t - 33)
}