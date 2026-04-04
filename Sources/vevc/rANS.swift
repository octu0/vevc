import Foundation

let RANS_SCALE_BITS: UInt32 = 14
let RANS_SCALE: UInt32 = 1 << RANS_SCALE_BITS
let RANS_L: UInt32 = 1 << 15
let RANS_XMAX: UInt32 = (RANS_L >> RANS_SCALE_BITS) << 16

// MARK: - Static rANS Frequency Tables
// These tables replace per-stream frequency table headers (~120B savings per encoder).
// Each table contains raw frequency counts normalized to RANS_SCALE=16384.
//
// Token mapping:
//   val tokens: 0..31 = values ±1..±16 (even=positive, odd=negative)
//   val tokens: 32..63 = values ±17+ (exp-golomb)
//   run tokens: 0..31 = run lengths 0..31
//   run tokens: 32..63 = run lengths 32+ (exp-golomb)

/// Build static rANS model from predetermined frequency data.
/// Normalizes the provided raw frequency array to sum exactly to RANS_SCALE.
private func buildStaticModel(rawFreqs: [UInt32]) -> rANSModel {
    var freqs = rawFreqs
    let sum: UInt32 = freqs.reduce(0, +)
    
    // Adjust the largest element to make sum == RANS_SCALE
    if sum != RANS_SCALE {
        var maxIdx = 0
        var maxVal: UInt32 = 0
        for i in 0..<64 {
            if freqs[i] > maxVal {
                maxVal = freqs[i]
                maxIdx = i
            }
        }
        if sum < RANS_SCALE {
            freqs[maxIdx] += (RANS_SCALE - sum)
        } else {
            let diff = sum - RANS_SCALE
            if freqs[maxIdx] > diff {
                freqs[maxIdx] -= diff
            }
        }
    }
    
    return rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: freqs)
}

// Static tables derived from actual data:
// runNorm0: run tokens, isParentZero=false (34M pairs)
//   Dominated by token 0 (run=0, ~68%), exponential decay
let staticRunModel0 = buildStaticModel(rawFreqs: [
    11172, 2025, 1036, 793, 299, 185, 165, 181,
      104,   62,   52,  54,  57,  36,  32,  37,
        9,    7,    5,   4,   4,   3,   4,   4,
        3,    2,    2,   2,   2,   1,   1,   2,
        1,    1,    2,   4,   5,   6,   1,   1,
        1,    1,    1,   1,   1,   1,   1,   1,
        1,    1,    1,   1,   1,   1,   1,   1,
        1,    1,    1,   1,   1,   1,   1,   1,
])

// valNorm0: value tokens, isParentZero=false
//   Asymmetric: positive values 3-6x more frequent than negative
//   Token 0(+1)=3235, token 2(+2)=2731, token 6(+4)=1134 (sharp right-decay)
let staticValModel0 = buildStaticModel(rawFreqs: [
    3235,  964, 2731,  425, 1173,  289, 1134,  222,
     618,  179,  610,  151,  379,  129,  373,  111,
     254,   97,  251,   86,  181,   76,  180,   69,
     135,   63,  136,   56,  103,   52,  104,   47,
     124,  123,  201,  302,  363,  337,  221,   76,
       3,    1,    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
])

// runNorm1: run tokens, isParentZero=true
//   Less concentrated: token 3(run=3)=2459 is the mode (not token 0)
let staticRunModel1 = buildStaticModel(rawFreqs: [
    3339, 1389, 1644, 2459, 822, 668, 725, 1009,
     417,  306,  338,  429, 512, 332, 329,  435,
      82,   66,   61,   63,  48,  50,  46,   67,
      56,   37,   33,   36,  26,  26,  28,   42,
      23,   16,   32,   77, 103, 195,   1,    1,
       1,    1,    1,    1,   1,   1,   1,    1,
       1,    1,    1,    1,   1,   1,   1,    1,
       1,    1,    1,    1,   1,   1,   1,    1,
])

// valNorm1: value tokens, isParentZero=true
//   Almost exclusively positive values (odd tokens ≈ 0)
//   Token 2(+2) is the mode, not token 0(+1)
let staticValModel1 = buildStaticModel(rawFreqs: [
    5838,    1, 6989,    1, 1004,   1, 1582,   1,
     219,    1,  465,    1,   52,   1,  141,   1,
      13,    1,   44,    1,    5,   1,   17,   1,
       1,    1,    5,    1,    1,   1,    1,   1,
       1,    1,    1,    1,    1,   1,    1,   1,
       1,    1,    1,    1,    1,   1,    1,   1,
       1,    1,    1,    1,    1,   1,    1,   1,
       1,    1,    1,    1,    1,   1,    1,   1,
])

// MARK: - DPCM Static Tables
// Measured from real 1080p video (ToS-4k-1080.y4m, 1802 frames, 44139379 pairs).
// DPCM has no isParentZero concept, so only one run/val model pair is needed.
// Both isParentZero=false and isParentZero=true use the same tables.

// DPCM run tokens: run=0 dominant (~39%), smooth exponential decay
let staticDPCMRunModel = buildStaticModel(rawFreqs: [
    6358, 2738, 1575, 1049,  757,  582,  490,  392,
     298,  254,  226,  200,  192,  199,  269,  209,
      45,   31,   25,   21,   18,   17,   19,   18,
      15,   13,   14,   14,   15,   19,   36,   35,
      10,    7,   11,   19,   52,   47,   34,   25,
      17,    1,    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
])

// DPCM val tokens: token0(+1)=44.2%, token1(-1)=48.2% (ultra-concentrated)
// DPCM residuals are almost exclusively ±1
let staticDPCMValModel = buildStaticModel(rawFreqs: [
    7247, 7891,  495,  516,   90,   91,   19,   20,
       4,    4,    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,    1,    1,
])

// MARK: - rANS Probability Model

struct rANSModel {
    private(set) var sigFreq: UInt32
    private(set) var tokenFreqs: [UInt32]
    private(set) var tokenCumFreqs: [UInt32]
    private(set) var tokenLUT: [UInt8]
    
    init() {
        self.sigFreq = RANS_SCALE / 2
        self.tokenFreqs = Array(repeating: RANS_SCALE / 64, count: 64)
        self.tokenCumFreqs = (0..<64).map { UInt32($0) * (RANS_SCALE / 64) }
        self.tokenLUT = [UInt8](repeating: 0, count: Int(RANS_SCALE))
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
        self.tokenLUT = [UInt8](repeating: 0, count: Int(RANS_SCALE))
        buildLUT()
    }
    
    @inline(__always)
    private mutating func buildLUT() {
        tokenLUT.withUnsafeMutableBufferPointer { ptr in
            for sym in 0..<64 {
                let start = Int(tokenCumFreqs[sym])
                let end = start + Int(tokenFreqs[sym])
                let s = UInt8(sym)
                for j in start..<min(end, Int(RANS_SCALE)) {
                    ptr[j] = s
                }
            }
        }
    }
    
    @inline(__always)
    mutating func normalize(sigCounts: [Int], tokenCounts: [Int]) {
        let totalSig = sigCounts[0] + sigCounts[1]
        if totalSig == 0 {
            self.sigFreq = RANS_SCALE / 2
        } else {
            let f = UInt32((Int(RANS_SCALE) * sigCounts[1]) / totalSig)
            self.sigFreq = max(1, min(RANS_SCALE - 1, f))
        }
        
        let totalTokens = tokenCounts.reduce(0, +)
        if totalTokens == 0 {
            self.tokenFreqs = Array(repeating: RANS_SCALE / 64, count: 64)
        } else {
            // Count unused tokens to maximize scale allocation for valid tokens
            var zeroCount: UInt32 = 0
            for i in 0..<64 {
                if tokenCounts[i] == 0 { zeroCount += 1 }
            }
            // Assign minimum freq=1 to unused tokens, allocating the rest to valid tokens
            let availableScale = RANS_SCALE - zeroCount
            
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
                if self.tokenFreqs[i] > maxVal {
                    maxVal = self.tokenFreqs[i]
                    maxIdx = i
                }
            }
            
            if sum < RANS_SCALE {
                self.tokenFreqs[maxIdx] += (RANS_SCALE - sum)
            } else if sum > RANS_SCALE {
                var diff = sum - RANS_SCALE
                while diff > 0 {
                    // find the max frequency to subtract from
                    var currentMaxIdx = 0
                    var currentMaxVal = self.tokenFreqs[0]
                    for i in 1..<64 {
                        if self.tokenFreqs[i] > currentMaxVal {
                            currentMaxVal = self.tokenFreqs[i]
                            currentMaxIdx = i
                        }
                    }
                    if currentMaxVal <= 1 { break } // cannot reduce further
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
        self.state = RANS_L
        self.stream = []
        self.stream.reserveCapacity(4096)
    }
    
    @inline(__always)
    mutating func encodeSymbol(cumFreq: UInt32, freq: UInt32) {
        let xMax = RANS_XMAX * freq
        while state >= xMax {
            stream.append(UInt16(truncatingIfNeeded: state))
            state >>= 16
        }
        let q = state / freq
        state = (q << RANS_SCALE_BITS) + (state - q * freq) + cumFreq
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
    private let stream: [UInt8]
    private var offset: Int
    
    init(bitstream: [UInt8]) {
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
    func getCumulativeFreq() -> UInt32 {
        return state & (RANS_SCALE - 1)
    }
    
    @inline(__always)
    mutating func advanceSymbol(cumFreq: UInt32, freq: UInt32) {
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
    private(set) var states: (UInt32, UInt32, UInt32, UInt32)
    private(set) var stream: [UInt16]
    
    init() {
        self.states = (RANS_L, RANS_L, RANS_L, RANS_L)
        self.stream = []
        self.stream.reserveCapacity(4096)
    }
    
    @inline(__always)
    mutating func encodeSymbol(lane: Int, cumFreq: UInt32, freq: UInt32) {
        let xMax = RANS_XMAX * freq
        
        switch lane {
        case 0:
            while states.0 >= xMax {
                stream.append(UInt16(truncatingIfNeeded: states.0))
                states.0 >>= 16
            }
            let q = states.0 / freq
            states.0 = (q << RANS_SCALE_BITS) + (states.0 - q * freq) + cumFreq
        case 1:
            while states.1 >= xMax {
                stream.append(UInt16(truncatingIfNeeded: states.1))
                states.1 >>= 16
            }
            let q = states.1 / freq
            states.1 = (q << RANS_SCALE_BITS) + (states.1 - q * freq) + cumFreq
        case 2:
            while states.2 >= xMax {
                stream.append(UInt16(truncatingIfNeeded: states.2))
                states.2 >>= 16
            }
            let q = states.2 / freq
            states.2 = (q << RANS_SCALE_BITS) + (states.2 - q * freq) + cumFreq
        case 3:
            while states.3 >= xMax {
                stream.append(UInt16(truncatingIfNeeded: states.3))
                states.3 >>= 16
            }
            let q = states.3 / freq
            states.3 = (q << RANS_SCALE_BITS) + (states.3 - q * freq) + cumFreq
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
    private let stream: [UInt8]
    private var offset: Int
    
    init(bitstream: [UInt8]) {
        // Add padding to eliminate bounds checks in readWord
        var padded = bitstream
        padded.append(contentsOf: [0, 0, 0, 0, 0, 0, 0, 0])
        self.stream = padded
        self.offset = 0
        self.states = (RANS_L, RANS_L, RANS_L, RANS_L)
        
        guard bitstream.count >= 16 else { return }
        
        @inline(__always)
        func readState(_ off: Int) -> UInt32 {
            var localOffset = off
            return try! readUInt32BEFromBytes(padded, offset: &localOffset)
        }
        
        self.states.0 = readState(0)
        self.states.1 = readState(4)
        self.states.2 = readState(8)
        self.states.3 = readState(12)
        self.offset = 16
    }
    
    @inline(__always)
    func getCumulativeFreq(lane: Int) -> UInt32 {
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
        if offset + 1 < stream.count {
            let b0 = UInt32(stream[offset])
            let b1 = UInt32(stream[offset + 1])
            offset += 2
            return (b0 << 8) | b1
        }
        return 0
    }
    
    @inline(__always)
    mutating func advanceSymbol(lane: Int, cumFreq: UInt32, freq: UInt32) {
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

// MARK: - 4-way Interleaved rANS Encoder

struct InterleavedrANSEncoder {
    private(set) var states: [UInt32]
    private(set) var streams: [[UInt16]]
    
    init() {
        self.states = [RANS_L, RANS_L, RANS_L, RANS_L]
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
        let xMax = RANS_XMAX * freq
        
        while state >= xMax {
            streams[lane].append(UInt16(truncatingIfNeeded: state))
            state >>= 16
        }
        
        state = ((state / freq) << RANS_SCALE_BITS) + (state % freq) + cumFreq
        states[lane] = state
    }
    
    @inline(__always)
    mutating func flush() {
        for lane in 0..<4 {
            streams[lane].append(UInt16(truncatingIfNeeded: states[lane]))
            streams[lane].append(UInt16(truncatingIfNeeded: states[lane] >> 16))
        }
    }
    
    /// merge 4 streams into a single bitstream
    /// format: [len0(4bytes)][len1(4bytes)][len2(4bytes)][len3(4bytes)][stream0][stream1][stream2][stream3]
    @inline(__always)
    func getBitstream() -> [UInt8] {
        var bytes = [UInt8]()
        
        var lengths = [Int](repeating: 0, count: 4)
        for i in 0..<4 {
            lengths[i] = streams[i].count * 2
        }
        
        // header (length of each stream)
        for len in lengths {
            let l = UInt32(len)
            bytes.append(UInt8(truncatingIfNeeded: l >> 24))
            bytes.append(UInt8(truncatingIfNeeded: (l >> 16) & 0xFF))
            bytes.append(UInt8(truncatingIfNeeded: (l >> 8) & 0xFF))
            bytes.append(UInt8(truncatingIfNeeded: l & 0xFF))
        }
        
        // body (reverse order: LIFO)
        for lane in 0..<4 {
            for word in streams[lane].reversed() {
                bytes.append(UInt8(truncatingIfNeeded: word >> 8)) // BE
                bytes.append(UInt8(truncatingIfNeeded: word & 0xFF))
            }
        }
        
        return bytes
    }
}

// MARK: - 4-way Interleaved rANS SIMD Decoder

struct InterleavedrANSDecoder {
    private(set) var states: SIMD4<UInt32>
    
    private let stream: [UInt8]
    // 4 lanes independent offsets
    private var offsets: SIMD4<Int>
    private let limits: SIMD4<Int>
    
    init(bitstream: [UInt8]) {
        self.stream = bitstream
        
        // Header parse
        guard bitstream.count >= 16 else {
            self.states = SIMD4<UInt32>(repeating: 0)
            self.offsets = SIMD4<Int>(repeating: 0)
            self.limits = SIMD4<Int>(repeating: 0)
            return
        }
        
        var lens = SIMD4<Int>(repeating: 0)
        var offset = 0
        for i in 0..<4 {
            lens[i] = Int(try! readUInt32BEFromBytes(bitstream, offset: &offset))
        }
        
        var currentOffset = 16
        var initOffsets = SIMD4<Int>(repeating: 0)
        var initLimits = SIMD4<Int>(repeating: 0)
        var initStates = SIMD4<UInt32>(repeating: 0)
        
        for i in 0..<4 {
            let limit = currentOffset + lens[i]
            if currentOffset + 4 <= limit {
                // stream is written in reverse order (LIFO)
                // so the last flushed state (4 bytes) is at the beginning of each stream block
                let w1 = (UInt32(bitstream[currentOffset]) << 8) | UInt32(bitstream[currentOffset + 1])
                let w0 = (UInt32(bitstream[currentOffset + 2]) << 8) | UInt32(bitstream[currentOffset + 3])
                initStates[i] = (w1 << 16) | w0
                
                // next renorm reads from the 4 bytes after this
                initOffsets[i] = currentOffset + 4
            } else {
                initStates[i] = 0
                initOffsets[i] = limit
            }
            initLimits[i] = limit
            currentOffset = limit
        }
        
        self.offsets = initOffsets
        self.limits = initLimits
        self.states = initStates
    }
    
    @inline(__always)
    func getCumulativeFreqs() -> SIMD4<UInt32> {
        let mask = SIMD4<UInt32>(repeating: RANS_SCALE - 1)
        return states & mask
    }
    
    @inline(__always)
    mutating func advanceSymbols(cumFreqs: SIMD4<UInt32>, freqs: SIMD4<UInt32>, activeMask: SIMD4<UInt32> = SIMD4<UInt32>(repeating: 0xFFFFFFFF)) {
        // [SIMD] Data parallel non-dependent update
        let mask = SIMD4<UInt32>(repeating: RANS_SCALE - 1)
        let nextStates = freqs &* (states &>> RANS_SCALE_BITS) &+ (states & mask) &- cumFreqs
        
        let boolMask = activeMask .== SIMD4<UInt32>(repeating: 0xFFFFFFFF)
        states.replace(with: nextStates, where: boolMask)
        
        // SIMD Renormalization
        let th = SIMD4<UInt32>(repeating: RANS_L)
        
        // first renormalization
        var renormMask = (states .< th) .& boolMask
        if any(renormMask) {
            for lane in 0..<4 {
                if renormMask[lane] {
                    let off = offsets[lane]
                    if off + 1 < limits[lane] {
                        let word = (UInt32(stream[off]) << 8) | UInt32(stream[off + 1])
                        offsets[lane] = off + 2
                        states[lane] = (states[lane] &<< 16) | word
                    } else {
                        states[lane] = states[lane] &<< 16
                    }
                }
            }
        }
        
        // second renormalization
        renormMask = (states .< th) .& boolMask
        if any(renormMask) {
            for lane in 0..<4 {
                if renormMask[lane] {
                    let off = offsets[lane]
                    if off + 1 < limits[lane] {
                        let word = (UInt32(stream[off]) << 8) | UInt32(stream[off + 1])
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
    private let bytes: [UInt8]
    private var byteOffset: Int
    private var buffer: UInt64
    private var bitsInBuffer: Int
    
    init(data: [UInt8]) {
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
            if byteOffset < bytes.count {
                buffer = (buffer << 8) | UInt64(bytes[byteOffset])
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

@inline(__always)
func valueTokenize(_ value: Int16) -> (token: UInt8, bypassBits: UInt32, bypassLen: Int) {
    if value == 0 { return (0, 0, 0) }
    let sign = value < 0
    let absValue = UInt16(value.magnitude)
    
    if absValue <= 15 {
        let token = UInt8((absValue - 1) * 2 + 1 + (sign ? 1 : 0))
        return (token, 0, 0)
    }
    
    let v = UInt32(absValue - 16)
    if v == 0 {
        return (32, sign ? 1 : 0, 1)
    }
    
    let bits = UInt32.bitWidth - v.leadingZeroBitCount
    let subToken = UInt8(bits)
    let bypass = UInt32(v & ((1 << (bits - 1)) - 1))
    let bypassLen = bits - 1
    
    let token = 32 + subToken
    let finalBypass = (bypass << 1) | (sign ? 1 : 0)
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
        return isNegative ? Int16(bitPattern: 0 &- absValue) : Int16(bitPattern: absValue)
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