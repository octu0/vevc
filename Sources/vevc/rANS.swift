import Foundation

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
            let diff = sum - rANSScale
            if diff < freqs[maxIdx] {
                freqs[maxIdx] -= diff
            }
        }
    }
    
    return rANSModel(sigFreq: rANSScale / 2, tokenFreqs: freqs)
}

final class StaticRANSModels: @unchecked Sendable {
    static let shared = StaticRANSModels()

    var runModel0 = buildStaticModel(rawFreqs: [
        3170, 1572, 1281, 1030, 1140, 804, 719, 714, 656, 420, 383, 376, 637, 420, 381, 407,
        98, 57, 56, 54, 49, 44, 42, 69, 80, 48, 44, 43, 36, 33, 32, 62,
        61, 37, 35, 32, 32, 28, 31, 50, 49, 32, 29, 28, 27, 24, 24, 47,
        50, 32, 29, 26, 24, 23, 26, 45, 94, 58, 49, 44, 40, 38, 40, 212,
    ])

    var valModel0 = buildStaticModel(rawFreqs: [
        1, 7863, 1978, 6072, 138, 105, 42, 82, 20, 18, 11, 15, 7, 6, 5, 4,
        2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var runModel1 = buildStaticModel(rawFreqs: [
        6744, 2248, 1437, 1367, 632, 507, 503, 663, 264, 182, 150, 140, 122, 127, 140, 167,
        79, 52, 39, 33, 32, 31, 31, 38, 28, 24, 24, 21, 23, 22, 25, 29,
        21, 18, 16, 15, 13, 12, 12, 14, 11, 11, 12, 11, 10, 9, 10, 11,
        8, 9, 8, 7, 6, 5, 5, 6, 5, 5, 5, 5, 4, 4, 4, 137,
    ])

    var valModel1 = buildStaticModel(rawFreqs: [
        1, 7547, 2541, 5234, 248, 288, 75, 244, 34, 47, 20, 43, 13, 12, 6, 9,
        3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var runModel2 = buildStaticModel(rawFreqs: [
        1708, 962, 831, 895, 745, 613, 614, 778, 666, 509, 497, 577, 680, 519, 527, 679,
        147, 90, 92, 87, 87, 84, 82, 150, 118, 83, 78, 82, 77, 75, 76, 143,
        93, 69, 64, 66, 64, 62, 65, 122, 95, 63, 62, 58, 59, 57, 63, 124,
        99, 68, 59, 58, 59, 65, 68, 129, 194, 120, 113, 107, 109, 111, 116, 344,
    ])

    var valModel2 = buildStaticModel(rawFreqs: [
        1, 7558, 264, 8153, 120, 80, 55, 68, 24, 14, 12, 9, 6, 4, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var runModel3 = buildStaticModel(rawFreqs: [
        5099, 1531, 1160, 1547, 667, 577, 757, 1477, 402, 246, 199, 190, 159, 166, 208, 293,
        132, 91, 67, 61, 57, 55, 68, 97, 56, 43, 39, 38, 35, 38, 44, 61,
        35, 30, 27, 22, 21, 23, 24, 32, 19, 18, 17, 16, 14, 16, 16, 22,
        14, 14, 14, 12, 11, 9, 9, 13, 8, 6, 7, 7, 5, 5, 5, 198,
    ])

    var valModel3 = buildStaticModel(rawFreqs: [
        1, 7591, 92, 8090, 41, 201, 26, 216, 22, 27, 16, 23, 8, 5, 4, 5,
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
}

// MARK: - rANS Probability Model
// why: LUT reverse-lookup reduces symbol search from O(log n) binary search to O(1)

struct rANSModel {
    private(set) var sigFreq: UInt32
    private(set) var tokenFreqs: [UInt32]
    private(set) var tokenCumFreqs: [UInt32]
    private(set) var tokenLUT: [UInt8]
    
    init() {
        self.sigFreq = rANSScale / 2
        self.tokenFreqs = Array(repeating: rANSScale / 64, count: 64)
        self.tokenCumFreqs = (0..<64).map { UInt32($0) * (rANSScale / 64) }
        self.tokenLUT = [UInt8](repeating: 0, count: Int(rANSScale))
        buildLUT()
    }
    
    init(sigFreq: UInt32, tokenFreqs: [UInt32]) {
        self.sigFreq = sigFreq
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
    mutating func normalize(sigCounts: [Int], tokenCounts: [Int]) {
        let totalSig = sigCounts[0] + sigCounts[1]
        if totalSig == 0 {
            self.sigFreq = rANSScale / 2
        } else {
            let f = UInt32((Int(rANSScale) * sigCounts[1]) / totalSig)
            self.sigFreq = max(1, min(rANSScale - 1, f))
        }
        
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

// MARK: - rANS Model Serialization helpers

@inline(__always)
internal func serializeRANSModel(_ rANS: rANSModel) -> [UInt8] {
    var out: [UInt8] = []
    out.reserveCapacity(256)
    for freq in rANS.tokenFreqs {
        out.append(UInt8((freq >> 24) & 0xFF))
        out.append(UInt8((freq >> 16) & 0xFF))
        out.append(UInt8((freq >> 8) & 0xFF))
        out.append(UInt8(freq & 0xFF))
    }
    return out
}

@inline(__always)
internal func deserializeRANSModel(from chunk: [UInt8], offset: inout Int) -> rANSModel {
    var freqs = [UInt32](repeating: 0, count: 64)
    for i in 0..<64 {
        let b0 = UInt32(chunk[offset])
        let b1 = UInt32(chunk[offset+1])
        let b2 = UInt32(chunk[offset+2])
        let b3 = UInt32(chunk[offset+3])
        offset += 4
        freqs[i] = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
    }
    // rANSModel's sum may not be exactly rANSScale if we blindly use freqs without rebuilding
    // Actually, freqs *were* written off a valid rANSModel, but to ensure lookup table is built:
    return buildStaticModel(rawFreqs: freqs)
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
        
        if count >= 4 {
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
        
        guard count >= 16 else { return }
        
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

// MARK: - 4-way Interleaved rANS Encoder

struct InterleavedrANSEncoder {
    private(set) var states: [UInt32]
    private(set) var streams: [[UInt16]]
    
    init() {
        self.states = [rANSL, rANSL, rANSL, rANSL]
        self.streams = [
            [UInt16](), [UInt16](), [UInt16](), [UInt16]()
        ]
        for i in 0..<4 {
            self.streams[i].reserveCapacity(1024)
        }
    }
    
    @inline(__always)
    mutating func encodeSymbol(lane: Int, cumFreq: UInt32, freq: UInt32) {
        var state = states[lane]
        let xMax = rANSXMax * freq
        
        while xMax <= state {
            streams[lane].append(UInt16(truncatingIfNeeded: state))
            state >>= 16
        }
        
        state = ((state / freq) << rANSScaleBits) + (state % freq) + cumFreq
        states[lane] = state
    }
    
    @inline(__always)
    mutating func flush() {
        for lane in 0..<4 {
            streams[lane].append(UInt16(truncatingIfNeeded: states[lane]))
            streams[lane].append(UInt16(truncatingIfNeeded: states[lane] >> 16))
        }
    }
    
    @inline(__always)
    func getBitstream() -> [UInt8] {
        var bytes = [UInt8]()
        
        var lengths = [Int](repeating: 0, count: 4)
        for i in 0..<4 {
            lengths[i] = streams[i].count * 2
        }
        
        for len in lengths {
            let l = UInt32(len)
            bytes.append(UInt8(truncatingIfNeeded: l >> 24))
            bytes.append(UInt8(truncatingIfNeeded: (l >> 16) & 0xFF))
            bytes.append(UInt8(truncatingIfNeeded: (l >> 8) & 0xFF))
            bytes.append(UInt8(truncatingIfNeeded: l & 0xFF))
        }
        
        for lane in 0..<4 {
            for word in streams[lane].reversed() {
                bytes.append(UInt8(truncatingIfNeeded: word >> 8))
                bytes.append(UInt8(truncatingIfNeeded: word & 0xFF))
            }
        }
        
        return bytes
    }
}

// MARK: - 4-way Interleaved rANS SIMD Decoder


struct InterleavedrANSDecoder {
    private(set) var states: SIMD4<UInt32>
    
    private let base: UnsafePointer<UInt8>
    private let count: Int
    private var offsets: SIMD4<Int>
    private let limits: SIMD4<Int>
    
    init(base: UnsafePointer<UInt8>, count: Int) {
        self.base = base
        self.count = count
        
        guard count >= 16 else {
            self.states = SIMD4<UInt32>(repeating: 0)
            self.offsets = SIMD4<Int>(repeating: 0)
            self.limits = SIMD4<Int>(repeating: 0)
            return
        }
        
        var lens = SIMD4<Int>(repeating: 0)
        var offset = 0
        
        @inline(__always)
        func readUInt32() -> UInt32 {
            if offset + 3 < count {
                let b0 = UInt32(base[offset])
                let b1 = UInt32(base[offset+1])
                let b2 = UInt32(base[offset+2])
                let b3 = UInt32(base[offset+3])
                offset += 4
                return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
            }
            return 0
        }
        
        for i in 0..<4 {
            lens[i] = Int(readUInt32())
        }
        
        var currentOffset = 16
        var initOffsets = SIMD4<Int>(repeating: 0)
        var initLimits = SIMD4<Int>(repeating: 0)
        var initStates = SIMD4<UInt32>(repeating: 0)
        
        for i in 0..<4 {
            let limit = currentOffset + lens[i]
            if currentOffset + 4 <= limit && limit <= count {
                let w1 = (UInt32(base[currentOffset]) << 8) | UInt32(base[currentOffset + 1])
                let w0 = (UInt32(base[currentOffset + 2]) << 8) | UInt32(base[currentOffset + 3])
                initStates[i] = (w1 << 16) | w0
                
                initOffsets[i] = currentOffset + 4
            } else {
                initStates[i] = 0
                initOffsets[i] = limit
            }
            initLimits[i] = min(limit, count)
            currentOffset = limit
        }
        
        self.offsets = initOffsets
        self.limits = initLimits
        self.states = initStates
    }
    
    @inline(__always)
    func getCumulativeFreqs() -> SIMD4<UInt32> {
        let mask = SIMD4<UInt32>(repeating: rANSScale - 1)
        return states & mask
    }
    
    @inline(__always)
    mutating func advanceSymbols(cumFreqs: SIMD4<UInt32>, freqs: SIMD4<UInt32>, activeMask: SIMD4<UInt32> = SIMD4<UInt32>(repeating: 0xFFFFFFFF)) {
        let mask = SIMD4<UInt32>(repeating: rANSScale - 1)
        let nextStates = freqs &* (states &>> rANSScaleBits) &+ (states & mask) &- cumFreqs
        
        let boolMask = activeMask .== SIMD4<UInt32>(repeating: 0xFFFFFFFF)
        states.replace(with: nextStates, where: boolMask)
        
        // why: renormalize lanes where state fell below rANSL;
        // at most 2 passes needed to restore all lanes
        let th = SIMD4<UInt32>(repeating: rANSL)
        
        // first renormalization pass (sufficient for most cases)
        var renormMask = (states .< th) .& boolMask
        if any(renormMask) {
            for lane in 0..<4 {
                if renormMask[lane] {
                    let off = offsets[lane]
                    if off + 1 < limits[lane] {
                        let word = (UInt32(base[off]) << 8) | UInt32(base[off + 1])
                        offsets[lane] = off + 2
                        states[lane] = (states[lane] &<< 16) | word
                    } else {
                        states[lane] = states[lane] &<< 16
                    }
                }
            }
        }
        
        // second renormalization pass (rare case)
        renormMask = (states .< th) .& boolMask
        if any(renormMask) {
            for lane in 0..<4 {
                if renormMask[lane] {
                    let off = offsets[lane]
                    if off + 1 < limits[lane] {
                        let word = (UInt32(base[off]) << 8) | UInt32(base[off + 1])
                        offsets[lane] = off + 2
                        states[lane] = (states[lane] &<< 16) | word
                    } else {
                        states[lane] = states[lane] &<< 16
                    }
                }
            }
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
    
    @inline(__always)
    mutating func flush() {
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
    mutating func skipBit() {
        ensureBits(1)
        bitsInBuffer -= 1
        buffer &= (1 << bitsInBuffer) &- 1
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
        ensureBits(count)
        bitsInBuffer -= count
        let value = (buffer >> bitsInBuffer) & ((1 << count) - 1)
        buffer &= (1 << bitsInBuffer) &- 1
        return UInt32(value)
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