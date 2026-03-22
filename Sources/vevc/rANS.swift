import Foundation

let RANS_SCALE_BITS: UInt32 = 14
let RANS_SCALE: UInt32 = 1 << RANS_SCALE_BITS
let RANS_L: UInt32 = 1 << 15
let RANS_XMAX: UInt32 = (RANS_L >> RANS_SCALE_BITS) << 16

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
            // 未使用トークン数をカウントし、有効トークンへのスケール配分を最大化
            var zeroCount: UInt32 = 0
            for i in 0..<64 {
                if tokenCounts[i] == 0 { zeroCount += 1 }
            }
            // 未使用トークンは最小freq=1を割り当て、残りを有効トークンに配分
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
    
    mutating func flush() {
        stream.append(UInt16(truncatingIfNeeded: state))
        stream.append(UInt16(truncatingIfNeeded: state >> 16))
    }
    
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

enum rANSCompressorError: Error {
    case insufficientData
}

func rANSCompress(_ data: [Int16]) -> [UInt8] {
        if data.isEmpty {
            return []
        }
        var outData = [UInt8]()
        outData.append(UInt8((data.count >> 24) & 0xFF))
        outData.append(UInt8((data.count >> 16) & 0xFF))
        outData.append(UInt8((data.count >> 8) & 0xFF))
        outData.append(UInt8(data.count & 0xFF))
        
        var tokenInfos: [(isSignificant: Bool, token: UInt8, bypassBits: UInt32, bypassLen: Int)] = []
        tokenInfos.reserveCapacity(data.count)
        
        var sigCounts = [1, 1] // [falseCount, trueCount], initialize with 1 to avoid 0 frequency
        var tokenCounts = [Int](repeating: 1, count: 64)
        
        for v in data {
            let isSig = v != 0
            if isSig {
                let t = valueTokenize(v)
                tokenInfos.append((isSignificant: true, token: t.token, bypassBits: t.bypassBits, bypassLen: t.bypassLen))
                sigCounts[1] += 1
                tokenCounts[Int(t.token)] += 1
            } else {
                tokenInfos.append((isSignificant: false, token: 0, bypassBits: 0, bypassLen: 0))
                sigCounts[0] += 1
            }
        }
        
        var model = rANSModel()
        model.normalize(sigCounts: sigCounts, tokenCounts: tokenCounts)
        
        for c in sigCounts {
            outData.append(UInt8((c >> 24) & 0xFF))
            outData.append(UInt8((c >> 16) & 0xFF))
            outData.append(UInt8((c >> 8) & 0xFF))
            outData.append(UInt8(c & 0xFF))
        }
        for c in tokenCounts {
            outData.append(UInt8((c >> 24) & 0xFF))
            outData.append(UInt8((c >> 16) & 0xFF))
            outData.append(UInt8((c >> 8) & 0xFF))
            outData.append(UInt8(c & 0xFF))
        }
        
        var encoder = InterleavedrANSEncoder()
        var bypassWriters = [BypassWriter](repeating: BypassWriter(), count: 4)
        
        let baseChunkSize = data.count / 4
        let remainder = data.count % 4
        
        var chunkStarts = [Int](repeating: 0, count: 4)
        var chunkSizes = [Int](repeating: 0, count: 4)
        var currentStart = 0
        for i in 0..<4 {
            let size = baseChunkSize + (i < remainder ? 1 : 0)
            chunkStarts[i] = currentStart
            chunkSizes[i] = size
            currentStart += size
        }
        
        for lane in 0..<4 {
            let start = chunkStarts[lane]
            let size = chunkSizes[lane]
            let end = start + size
            
            // Backward pass for rANS
            if size > 0 {
                for i in stride(from: end - 1, through: start, by: -1) {
                    let t = tokenInfos[i]
                    if t.isSignificant {
                        encoder.encodeSymbol(lane: lane, cumFreq: model.tokenCumFreqs[Int(t.token)], freq: model.tokenFreqs[Int(t.token)])
                        encoder.encodeSymbol(lane: lane, cumFreq: 0, freq: model.sigFreq)
                    } else {
                        encoder.encodeSymbol(lane: lane, cumFreq: model.sigFreq, freq: RANS_SCALE - model.sigFreq)
                    }
                }
            }
            // Forward pass for Bypass
            for i in start..<end {
                let t = tokenInfos[i]
                if t.isSignificant {
                    bypassWriters[lane].writeBits(t.bypassBits, count: t.bypassLen)
                }
            }
            bypassWriters[lane].flush()
        }
        encoder.flush()
        
        let ransStream = encoder.getBitstream()
        
        outData.append(UInt8((ransStream.count >> 24) & 0xFF))
        outData.append(UInt8((ransStream.count >> 16) & 0xFF))
        outData.append(UInt8((ransStream.count >> 8) & 0xFF))
        outData.append(UInt8(ransStream.count & 0xFF))
        
        outData.append(contentsOf: ransStream)
        
        for lane in 0..<4 {
            let bytes = bypassWriters[lane].bytes
            outData.append(UInt8((bytes.count >> 24) & 0xFF))
            outData.append(UInt8((bytes.count >> 16) & 0xFF))
            outData.append(UInt8((bytes.count >> 8) & 0xFF))
            outData.append(UInt8(bytes.count & 0xFF))
            outData.append(contentsOf: bytes)
        }
        
        return outData
    }
    
func rANSDecompress(_ data: [UInt8]) throws -> [Int16] {
        if data.isEmpty {
            return []
        }
        var offset = 0
        
        func readUInt32() throws -> Int {
            guard offset + 4 <= data.count else { throw rANSCompressorError.insufficientData }
            let b0 = Int(data[offset])
            let b1 = Int(data[offset+1])
            let b2 = Int(data[offset+2])
            let b3 = Int(data[offset+3])
            offset += 4
            return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
        }
        
        let targetCount = try readUInt32()
        if targetCount == 0 { return [] }
        var sigCounts = [Int](repeating: 0, count: 2)
        sigCounts[0] = try readUInt32()
        sigCounts[1] = try readUInt32()
        
        var tokenCounts = [Int](repeating: 0, count: 64)
        for i in 0..<64 {
            tokenCounts[i] = try readUInt32()
        }
        
        var model = rANSModel()
        model.normalize(sigCounts: sigCounts, tokenCounts: tokenCounts)
        
        let ransStreamLen = try readUInt32()
        guard offset + ransStreamLen <= data.count else { throw rANSCompressorError.insufficientData }
        let ransStream = Array(data[offset..<offset+ransStreamLen])
        offset += ransStreamLen
        
        var bypassBytes = [[UInt8]](repeating: [], count: 4)
        for i in 0..<4 {
            let len = try readUInt32()
            guard offset + len <= data.count else { throw rANSCompressorError.insufficientData }
            bypassBytes[i] = Array(data[offset..<offset+len])
            offset += len
        }
        
        var decoder = InterleavedrANSDecoder(bitstream: ransStream)
        var bypassReaders = bypassBytes.map { BypassReader(data: $0) }
        
        var outCoeffs = [Int16](repeating: 0, count: targetCount)
        
        let baseChunkSize = targetCount / 4
        let remainder = targetCount % 4
        
        var chunkStarts = [Int](repeating: 0, count: 4)
        var chunkSizes = [Int](repeating: 0, count: 4)
        var currentStart = 0
        var maxChunkSize = 0
        for i in 0..<4 {
            let size = baseChunkSize + (i < remainder ? 1 : 0)
            chunkStarts[i] = currentStart
            chunkSizes[i] = size
            if size > maxChunkSize { maxChunkSize = size }
            currentStart += size
        }
        
        var rANSDecodedByLane = [[(isSignificant: Bool, token: UInt8)]](repeating: [], count: 4)
        for i in 0..<4 {
            rANSDecodedByLane[i].reserveCapacity(chunkSizes[i])
        }
        
        let sigFreqVec = SIMD4<UInt32>(repeating: model.sigFreq)
        let invSigFreqVec = SIMD4<UInt32>(repeating: RANS_SCALE - model.sigFreq)
        let zeroVec = SIMD4<UInt32>(repeating: 0)
        
        for idx in 0..<maxChunkSize {
            let cfs = decoder.getCumulativeFreqs()
            var isSigs = [Bool](repeating: false, count: 4)
            
            var sigAdvanceCumFreq = SIMD4<UInt32>(repeating: 0)
            var sigAdvanceFreq = SIMD4<UInt32>(repeating: 0)
            
            for lane in 0..<4 {
                if idx < chunkSizes[lane] {
                    if cfs[lane] < model.sigFreq {
                        isSigs[lane] = true
                        sigAdvanceFreq[lane] = sigFreqVec[lane]
                        sigAdvanceCumFreq[lane] = zeroVec[lane]
                    } else {
                        isSigs[lane] = false
                        sigAdvanceFreq[lane] = invSigFreqVec[lane]
                        sigAdvanceCumFreq[lane] = sigFreqVec[lane]
                    }
                } else {
                    sigAdvanceFreq[lane] = 1
                    sigAdvanceCumFreq[lane] = 0
                }
            }
            
            decoder.advanceSymbols(cumFreqs: sigAdvanceCumFreq, freqs: sigAdvanceFreq)
            
            let cfTokens = decoder.getCumulativeFreqs()
            var tokenAdvanceCumFreq = SIMD4<UInt32>(repeating: 0)
            var tokenAdvanceFreq = SIMD4<UInt32>(repeating: 0)
            var readToken = [UInt8](repeating: 0, count: 4)
            var advanceMask = SIMD4<UInt32>(repeating: 0)
            
            for lane in 0..<4 {
                if idx < chunkSizes[lane] && isSigs[lane] {
                    let tInfo = model.findToken(cf: cfTokens[lane])
                    readToken[lane] = tInfo.token
                    tokenAdvanceCumFreq[lane] = tInfo.cumFreq
                    tokenAdvanceFreq[lane] = tInfo.freq
                    advanceMask[lane] = 0xFFFFFFFF
                }
            }
            
            decoder.advanceSymbols(cumFreqs: tokenAdvanceCumFreq, freqs: tokenAdvanceFreq, activeMask: advanceMask)
            
            for lane in 0..<4 {
                if idx < chunkSizes[lane] {
                    rANSDecodedByLane[lane].append((isSignificant: isSigs[lane], token: readToken[lane]))
                }
            }
        }
        
        for lane in 0..<4 {
            let size = chunkSizes[lane]
            let start = chunkStarts[lane]
            
            // Forward pass reading
            for i in 0..<size {
                let t = rANSDecodedByLane[lane][i] // DO NOT EXTRACT IN REVERSE! LIFO pop output is already in original forward order.
                if t.isSignificant {
                    let bypassLen = valueBypassLength(for: t.token)
                    let bypassBits = bypassReaders[lane].readBits(count: bypassLen)
                    outCoeffs[start + i] = valueDetokenize(token: t.token, bypassBits: bypassBits)
                } else {
                    outCoeffs[start + i] = 0
                }
            }
        }
        
        return outCoeffs
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
        } else {
            return 0
        }
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
    
    mutating func flush() {
        for lane in 0..<4 {
            streams[lane].append(UInt16(truncatingIfNeeded: states[lane]))
            streams[lane].append(UInt16(truncatingIfNeeded: states[lane] >> 16))
        }
    }
    
    /// merge 4 streams into a single bitstream
    /// format: [len0(4bytes)][len1(4bytes)][len2(4bytes)][len3(4bytes)][stream0][stream1][stream2][stream3]
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
        for i in 0..<4 {
            let idx = i * 4
            let b0 = UInt32(bitstream[idx])
            let b1 = UInt32(bitstream[idx + 1])
            let b2 = UInt32(bitstream[idx + 2])
            let b3 = UInt32(bitstream[idx + 3])
            lens[i] = Int((b0 << 24) | (b1 << 16) | (b2 << 8) | b3)
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
        self.bytes.reserveCapacity(256)
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