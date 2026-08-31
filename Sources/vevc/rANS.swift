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

    // Static fallback models: used until the backward-adaptive history is
    // primed. Trained offline from quantized-coefficient dumps
    // (vevc-training train-tables). dpcmRunModel/dpcmValModel are the
    // original hand-tuned tables: ctx4 only occurs in I-frames, which the
    // P-frame dumps do not cover.
    var runModel0 = buildStaticModel(rawFreqs: [
        4796, 1845, 1296, 979, 1002, 577, 527, 504, 714, 407, 356, 344, 561, 327, 296, 296,
        81, 45, 39, 41, 35, 34, 31, 48, 61, 36, 32, 31, 27, 27, 24, 39,
        38, 24, 42, 89, 173, 449, 34, 34, 20, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var valModel0 = buildStaticModel(rawFreqs: [
        1, 6642, 1781, 5762, 316, 523, 128, 511, 56, 137, 33, 139, 20, 55, 14, 56,
        9, 26, 7, 29, 5, 14, 3, 15, 3, 8, 2, 9, 2, 5, 2, 1,
        6, 5, 8, 10, 10, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var runModel1 = buildStaticModel(rawFreqs: [
        8761, 2545, 1361, 1039, 502, 370, 333, 350, 167, 111, 91, 84, 70, 69, 71, 75,
        42, 27, 20, 17, 15, 15, 15, 17, 13, 10, 9, 9, 9, 9, 10, 12,
        9, 6, 9, 16, 24, 23, 15, 9, 2, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var valModel1 = buildStaticModel(rawFreqs: [
        1, 5810, 2098, 4194, 440, 993, 184, 919, 82, 343, 49, 334, 30, 161, 21, 160,
        13, 87, 10, 86, 7, 49, 5, 49, 4, 31, 3, 31, 2, 20, 2, 1,
        22, 15, 25, 31, 28, 15, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var runModel2 = buildStaticModel(rawFreqs: [
        1688, 1153, 1026, 1057, 949, 711, 701, 822, 793, 594, 544, 647, 799, 588, 554, 688,
        111, 70, 69, 74, 63, 68, 68, 106, 81, 60, 57, 60, 54, 50, 53, 94,
        69, 43, 88, 211, 383, 1030, 28, 30, 27, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var valModel2 = buildStaticModel(rawFreqs: [
        1, 7497, 1, 7984, 1, 307, 1, 342, 1, 55, 1, 67, 1, 21, 1, 25,
        1, 8, 1, 10, 1, 4, 1, 4, 1, 3, 1, 3, 1, 1, 1, 1,
        2, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var runModel3 = buildStaticModel(rawFreqs: [
        6042, 1932, 1334, 1365, 748, 613, 653, 841, 358, 245, 205, 192, 172, 163, 187, 206,
        117, 80, 60, 53, 46, 43, 49, 59, 43, 31, 27, 26, 26, 30, 34, 38,
        28, 19, 28, 55, 78, 65, 39, 25, 6, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var valModel3 = buildStaticModel(rawFreqs: [
        1, 6201, 1, 6473, 1, 1065, 1, 1129, 1, 333, 1, 342, 1, 164, 1, 174,
        1, 85, 1, 90, 1, 49, 1, 51, 1, 28, 1, 32, 1, 19, 1, 1,
        19, 11, 21, 24, 21, 9, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
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
        2276, 2301, 2825, 6098, 328, 391, 455, 1363, 14, 18, 17, 24, 20, 31, 28, 147,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    // Parent-free AC tables (profile 0x02, EntropyCodec.swift): all AC
    // traffic lands in contexts 0-1, so these are trained on the merged
    // parent-free assignment. Contexts 2-3 reuse the shipped models above
    // but carry no data in profile 0x02.
    var pfRunModel0 = buildStaticModel(rawFreqs: [
        3804, 1660, 1260, 1042, 945, 619, 581, 610, 700, 452, 417, 423, 550, 365, 345, 400,
        98, 56, 50, 54, 47, 49, 48, 78, 76, 48, 44, 45, 40, 37, 39, 69,
        58, 34, 62, 142, 257, 618, 53, 57, 29, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var pfRunModel1 = buildStaticModel(rawFreqs: [
        8097, 2351, 1338, 1039, 562, 438, 418, 450, 216, 153, 129, 119, 110, 111, 121, 126,
        67, 42, 31, 27, 23, 22, 23, 26, 20, 16, 14, 14, 14, 16, 19, 21,
        15, 10, 15, 25, 39, 38, 27, 16, 3, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var pfValModel0 = buildStaticModel(rawFreqs: [
        1, 6757, 1270, 6249, 210, 504, 70, 526, 36, 150, 21, 162, 14, 68, 9, 70,
        6, 35, 4, 38, 4, 20, 3, 21, 2, 13, 2, 14, 1, 7, 2, 1,
        9, 6, 12, 17, 14, 9, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ])

    var pfValModel1 = buildStaticModel(rawFreqs: [
        1, 5556, 1324, 4681, 293, 1081, 112, 1090, 61, 410, 36, 414, 23, 218, 17, 220,
        11, 125, 9, 129, 6, 74, 5, 76, 4, 49, 3, 51, 3, 33, 2, 1,
        36, 23, 43, 52, 50, 28, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
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
        // memset per symbol run: this runs per frame for dynamic/history
        // models, so the fill must be vectorized, not a byte loop
        tokenLUT.withUnsafeMutableBufferPointer { ptr in
            let base = ptr.baseAddress!
            for sym in 0..<64 {
                let start = Int(tokenCumFreqs[sym])
                guard start < Int(rANSScale) else { continue }
                let len = min(Int(tokenFreqs[sym]), Int(rANSScale) - start)
                if 0 < len {
                    memset(base.advanced(by: start), Int32(sym), len)
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
            
            // absorb deficit into the largest frequency to preserve distribution shape
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
                    // freq <= 1 cannot be reduced further without causing division by zero during decode
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
    mutating func skipBit() {
        ensureBits(1)
        bitsInBuffer -= 1
        buffer &= (1 << bitsInBuffer) &- 1
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
    }
    return Int16(truncatingIfNeeded: absValue)
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