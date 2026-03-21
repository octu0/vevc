import Foundation

// MARK: - ContextModel / Legacy CABAC Support (To be removed)

struct ContextModel {
    public var pStateIdx: UInt8 = 0
    public var valMPS: UInt8 = 0
    public init() {}
}

// MARK: - VevcEncoder

struct EntropyEncoder {
    var bypassWriter: BypassWriter
    public var pairs: [(run: UInt32, val: Int16)]
    public var trailingZeros: UInt32
    public private(set) var coeffCount: Int

    public init() {
        self.bypassWriter = BypassWriter()
        self.pairs = []
        self.trailingZeros = 0
        self.coeffCount = 0
    }

    @inline(__always)
    public mutating func encodeBypass(binVal: UInt8) {
        bypassWriter.writeBit(binVal != 0)
    }

    @inline(__always)
    public mutating func addPair(run: UInt32, val: Int16) {
        pairs.append((run: run, val: val))
        coeffCount += Int(run) + 1
    }
    
    @inline(__always)
    public mutating func addTrailingZeros(_ count: UInt32) {
        trailingZeros += count
        coeffCount += Int(count)
    }

    @inline(__always)
    public mutating func flush() {
        bypassWriter.flush()
    }

    @inline(__always)
    public mutating func getData() -> [UInt8] {
        var out = [UInt8]()
        out.reserveCapacity(pairs.count * 4 + 128)
        
        bypassWriter.flush()
        let metaBypassData = bypassWriter.bytes
        appendUInt32BE(&out, UInt32(metaBypassData.count))
        out.append(contentsOf: metaBypassData)
        
        appendUInt32BE(&out, UInt32(coeffCount))
        
        guard pairs.isEmpty != true || 0 < trailingZeros else { return out }
        
        let pairCount = pairs.count
        let hasTrailingZeros = trailingZeros > 0
        let nonZeroCount = pairCount
        
        if nonZeroCount <= 32 {
            out.append(0x80)
            var rawBypass = BypassWriter()
            for pair in pairs {
                // for zero-run
                for _ in 0..<pair.run {
                    rawBypass.writeBit(false)
                }
                rawBypass.writeBit(true)
                let valResult = ValueTokenizer.tokenize(pair.val)
                rawBypass.writeBits(UInt32(valResult.token), count: 5)
                rawBypass.writeBits(valResult.bypassBits, count: valResult.bypassLen)
            }
            // for trailing zeros
            for _ in 0..<trailingZeros {
                rawBypass.writeBit(false)
            }
            rawBypass.flush()
            let rawData = rawBypass.bytes
            appendUInt32BE(&out, UInt32(rawData.count))
            out.append(contentsOf: rawData)
            return out
        }
        
        // pairs: 4-way rANS mode
        let totalPairEntries = pairCount + (hasTrailingZeros ? 1 : 0)
        let chunkBase = pairCount / 4
        let chunkRemainder = pairCount % 4
        
        // chunk boundary: chunk[i] is responsible for pairs[chunkStarts[i]..<chunkStarts[i+1]]
        var chunkStarts = [Int](repeating: 0, count: 5)
        for i in 0..<4 {
            chunkStarts[i + 1] = chunkStarts[i] + chunkBase + (i < chunkRemainder ? 1 : 0)
        }
        
        // tokenization + bypass writing for each chunk
        var chunkRunTokens = [[UInt8]](repeating: [], count: 4)
        var chunkValTokens = [[UInt8]](repeating: [], count: 4)
        var chunkBypassWriters = [BypassWriter](repeating: BypassWriter(), count: 4)
        var runTokenCounts = Array(repeating: 0, count: 64)
        var valTokenCounts = Array(repeating: 0, count: 64)
        
        for lane in 0..<4 {
            let start = chunkStarts[lane]
            let end = chunkStarts[lane + 1]
            let chunkSize = end - start
            chunkRunTokens[lane].reserveCapacity(chunkSize + 1)
            chunkValTokens[lane].reserveCapacity(chunkSize)
            
            for idx in start..<end {
                let pair = pairs[idx]
                
                let runResult = ValueTokenizer.tokenizeUnsigned(pair.run)
                runTokenCounts[Int(runResult.token)] += 1
                chunkRunTokens[lane].append(runResult.token)
                chunkBypassWriters[lane].writeBits(runResult.bypassBits, count: runResult.bypassLen)
                
                let valResult = ValueTokenizer.tokenize(pair.val)
                valTokenCounts[Int(valResult.token)] += 1
                chunkValTokens[lane].append(valResult.token)
                chunkBypassWriters[lane].writeBits(valResult.bypassBits, count: valResult.bypassLen)
            }
        }
        
        // trailing zeros: add to lane3
        if hasTrailingZeros {
            let runResult = ValueTokenizer.tokenizeUnsigned(trailingZeros)
            runTokenCounts[Int(runResult.token)] += 1
            chunkRunTokens[3].append(runResult.token)
            chunkBypassWriters[3].writeBits(runResult.bypassBits, count: runResult.bypassLen)
        }
        
        for lane in 0..<4 {
            chunkBypassWriters[lane].flush()
        }
        // Cap frequencies to 16-bit to ensure bitstream serialization perfectly matches what the decoder reads
        for i in 0..<64 {
            if runTokenCounts[i] > 65535 { runTokenCounts[i] = 65535 }
            if valTokenCounts[i] > 65535 { valTokenCounts[i] = 65535 }
        }
        
        var runModel = rANSModel()
        var valModel = rANSModel()
        runModel.normalize(sigCounts: [0, 1], tokenCounts: runTokenCounts)
        valModel.normalize(sigCounts: [0, 1], tokenCounts: valTokenCounts)
        
        // header
        out.append(hasTrailingZeros ? 1 : 0)
        appendUInt32BE(&out, UInt32(totalPairEntries))
        
        // chunk size (4B × 4)
        for lane in 0..<4 {
            let chunkPairCount = chunkStarts[lane + 1] - chunkStarts[lane]
            let extraTrailing = (lane == 3 && hasTrailingZeros) ? 1 : 0
            appendUInt32BE(&out, UInt32(chunkPairCount + extraTrailing))
        }
        
        // compressed frequency table
        writeCompressedFreqTable(&out, freqs: runModel.tokenFreqs)
        writeCompressedFreqTable(&out, freqs: valModel.tokenFreqs)
        
        // 4-way bypass data
        for lane in 0..<4 {
            let bpData = chunkBypassWriters[lane].bytes
            appendUInt32BE(&out, UInt32(bpData.count))
            out.append(contentsOf: bpData)
        }
        
        // Interleaved 4-way rANS encode (reverse order)
        var enc = Interleaved4rANSEncoder()
        
        // encode each lane in reverse order
        // trailing zeros (lane 3)
        if hasTrailingZeros {
            let trailingRunToken = chunkRunTokens[3].last!
            enc.encodeSymbol(lane: 3, cumFreq: runModel.tokenCumFreqs[Int(trailingRunToken)], freq: runModel.tokenFreqs[Int(trailingRunToken)])
        }
        
        // encode each lane in reverse order
        for lane in stride(from: 3, through: 0, by: -1) {
            let runTokens = chunkRunTokens[lane]
            let valTokens = chunkValTokens[lane]
            let pairEnd = valTokens.count
            
            for i in stride(from: pairEnd - 1, through: 0, by: -1) {
                let vt = valTokens[i]
                enc.encodeSymbol(lane: lane, cumFreq: valModel.tokenCumFreqs[Int(vt)], freq: valModel.tokenFreqs[Int(vt)])
                
                let rt = runTokens[i]
                enc.encodeSymbol(lane: lane, cumFreq: runModel.tokenCumFreqs[Int(rt)], freq: runModel.tokenFreqs[Int(rt)])
            }
        }
        
        enc.flush()
        out.append(contentsOf: enc.getBitstream())
        
        return out
    }
    
    @inline(__always)
    private func writeCompressedFreqTable(_ out: inout [UInt8], freqs: [UInt32]) {
        var bitmap: UInt64 = 0
        for i in 0..<64 {
            if freqs[i] > 1 {
                bitmap |= UInt64(1) << i
            }
        }
        out.append(UInt8((bitmap >> 56) & 0xFF))
        out.append(UInt8((bitmap >> 48) & 0xFF))
        out.append(UInt8((bitmap >> 40) & 0xFF))
        out.append(UInt8((bitmap >> 32) & 0xFF))
        out.append(UInt8((bitmap >> 24) & 0xFF))
        out.append(UInt8((bitmap >> 16) & 0xFF))
        out.append(UInt8((bitmap >> 8) & 0xFF))
        out.append(UInt8(bitmap & 0xFF))
        
        for i in 0..<64 {
            if (bitmap & (UInt64(1) << i)) != 0 {
                out.append(UInt8(truncatingIfNeeded: freqs[i] >> 8))
                out.append(UInt8(truncatingIfNeeded: freqs[i] & 0xFF))
            }
        }
    }
    
    @inline(__always)
    private func appendUInt32BE(_ out: inout [UInt8], _ val: UInt32) {
        out.append(UInt8((val >> 24) & 0xFF))
        out.append(UInt8((val >> 16) & 0xFF))
        out.append(UInt8((val >> 8) & 0xFF))
        out.append(UInt8(val & 0xFF))
    }
}

// MARK: - VevcDecoder

struct EntropyDecoder {
    var bypassReader: BypassReader
    public var pairs: [(run: UInt32, val: Int16)]
    private var pairIndex: Int = 0

    public init(data: [UInt8]) throws {
        var offset = 0
        
        guard 4 <= data.count else { throw DecodeError.insufficientData }
        let bypassLen = vevc.readUInt32BE(data, at: &offset)
        
        guard offset + Int(bypassLen) <= data.count else { throw DecodeError.insufficientData }
        let bypassData = Array(data[offset..<(offset + Int(bypassLen))])
        self.bypassReader = BypassReader(data: bypassData)
        offset += Int(bypassLen)
        
        guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
        let coeffCount = Int(vevc.readUInt32BE(data, at: &offset))
        
        guard 0 < coeffCount else {
            self.pairs = []
            return
        }
        
        guard offset < data.count else { throw DecodeError.insufficientData }
        let flags = data[offset]
        offset += 1
        
        let isRawMode = (flags & 0x80) != 0
        
        if isRawMode {
            guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
            let rawDataLen = Int(vevc.readUInt32BE(data, at: &offset))
            guard offset + rawDataLen <= data.count else { throw DecodeError.insufficientData }
            let rawData = Array(data[offset..<(offset + rawDataLen)])
            var rawReader = BypassReader(data: rawData)
            
            var decodedPairs = [(run: UInt32, val: Int16)]()
            var zeroRun: UInt32 = 0
            for _ in 0..<coeffCount {
                let isNonZero = rawReader.readBit()
                if isNonZero {
                    let tokenBits = rawReader.readBits(count: 5)
                    let token = UInt8(tokenBits)
                    let bypassLen = ValueTokenizer.bypassLength(for: token)
                    let bypassBits = rawReader.readBits(count: bypassLen)
                    let val = ValueTokenizer.detokenize(token: token, bypassBits: bypassBits)
                    decodedPairs.append((run: zeroRun, val: val))
                    zeroRun = 0
                } else {
                    zeroRun += 1
                }
            }
            // trailing zeros are not included in pairs (implicitly managed by coeffCount)
            if 0 < zeroRun {
                decodedPairs.append((run: zeroRun, val: 0))
            }
            self.pairs = decodedPairs
            return
        }
        
        // rANS mode
        let hasTrailingZeros = (flags & 1) != 0
        
        guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
        let totalPairEntries = Int(vevc.readUInt32BE(data, at: &offset))
        
        // chunk size (4 lanes)
        guard offset + 16 <= data.count else { throw DecodeError.insufficientData }
        var chunkSizes = [Int](repeating: 0, count: 4)
        for lane in 0..<4 {
            chunkSizes[lane] = Int(vevc.readUInt32BE(data, at: &offset))
        }
        
        let runTokenFreqs = try EntropyDecoder.readCompressedFreqTable(data, at: &offset)
        let runModel = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: runTokenFreqs)
        
        let valTokenFreqs = try EntropyDecoder.readCompressedFreqTable(data, at: &offset)
        let valModel = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: valTokenFreqs)
        
        // 4-way bypass data
        var chunkBypassReaders = [BypassReader]()
        for _ in 0..<4 {
            guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
            let bpLen = Int(vevc.readUInt32BE(data, at: &offset))
            guard offset + bpLen <= data.count else { throw DecodeError.insufficientData }
            let bpData = Array(data[offset..<(offset + bpLen)])
            chunkBypassReaders.append(BypassReader(data: bpData))
            offset += bpLen
        }
        
        // rANS stream
        let ransData = Array(data[offset...])
        var ransDecoder = Interleaved4rANSDecoder(bitstream: ransData)
        
        // decode pairs from each lane
        var chunkPairs = [[(run: UInt32, val: Int16)]](repeating: [], count: 4)
        
        for lane in 0..<4 {
            let chunkSize = chunkSizes[lane]
            let hasTZ = (lane == 3 && hasTrailingZeros)
            let pairCountInChunk = hasTZ ? chunkSize - 1 : chunkSize
            chunkPairs[lane].reserveCapacity(chunkSize)
            
            for _ in 0..<pairCountInChunk {
                let cfRun = ransDecoder.getCumulativeFreq(lane: lane)
                let rtInfo = runModel.findToken(cf: cfRun)
                ransDecoder.advanceSymbol(lane: lane, cumFreq: rtInfo.cumFreq, freq: rtInfo.freq)
                
                let runBypassLen = ValueTokenizer.bypassLengthUnsigned(for: rtInfo.token)
                let runBypassBits = chunkBypassReaders[lane].readBits(count: runBypassLen)
                let zeroRun = UInt32(ValueTokenizer.detokenizeUnsigned(token: rtInfo.token, bypassBits: runBypassBits))
                
                let cfVal = ransDecoder.getCumulativeFreq(lane: lane)
                let vtInfo = valModel.findToken(cf: cfVal)
                ransDecoder.advanceSymbol(lane: lane, cumFreq: vtInfo.cumFreq, freq: vtInfo.freq)
                
                let valBypassLen = ValueTokenizer.bypassLength(for: vtInfo.token)
                let valBypassBits = chunkBypassReaders[lane].readBits(count: valBypassLen)
                let val = ValueTokenizer.detokenize(token: vtInfo.token, bypassBits: valBypassBits)
                
                chunkPairs[lane].append((run: zeroRun, val: val))
            }
            
            // trailing zeros (lane 3)
            if hasTZ {
                let cfRun = ransDecoder.getCumulativeFreq(lane: lane)
                let rtInfo = runModel.findToken(cf: cfRun)
                ransDecoder.advanceSymbol(lane: lane, cumFreq: rtInfo.cumFreq, freq: rtInfo.freq)
                
                let runBypassLen = ValueTokenizer.bypassLengthUnsigned(for: rtInfo.token)
                let runBypassBits = chunkBypassReaders[lane].readBits(count: runBypassLen)
                let zeroRun = UInt32(ValueTokenizer.detokenizeUnsigned(token: rtInfo.token, bypassBits: runBypassBits))
                chunkPairs[lane].append((run: zeroRun, val: 0))
            }
        }
        
        // combine 4 chunks to generate pairs array
        var allPairs = [(run: UInt32, val: Int16)]()
        allPairs.reserveCapacity(totalPairEntries)
        for lane in 0..<4 {
            allPairs.append(contentsOf: chunkPairs[lane])
        }
        
        self.pairs = allPairs
    }
    
    @inline(__always)
    private static func readCompressedFreqTable(_ data: [UInt8], at offset: inout Int) throws -> [UInt32] {
        guard offset + 8 <= data.count else { throw DecodeError.insufficientData }
        let b0 = UInt64(data[offset])
        let b1 = UInt64(data[offset+1])
        let b2 = UInt64(data[offset+2])
        let b3 = UInt64(data[offset+3])
        let b4 = UInt64(data[offset+4])
        let b5 = UInt64(data[offset+5])
        let b6 = UInt64(data[offset+6])
        let b7 = UInt64(data[offset+7])
        let bitmap = (b0 << 56) | (b1 << 48) | (b2 << 40) | (b3 << 32) | (b4 << 24) | (b5 << 16) | (b6 << 8) | b7
        offset += 8
        
        var freqs = [UInt32](repeating: 1, count: 64)
        for i in 0..<64 {
            if (bitmap & (UInt64(1) << i)) != 0 {
                guard offset + 2 <= data.count else { throw DecodeError.insufficientData }
                freqs[i] = (UInt32(data[offset]) << 8) | UInt32(data[offset + 1])
                offset += 2
            }
        }
        return freqs
    }

    @inline(__always)
    public mutating func decodeBypass() throws -> UInt8 {
        return bypassReader.readBit() ? 1 : 0
    }
    
    @inline(__always)
    public mutating func readPair() -> (run: Int, val: Int16) {
        guard pairIndex < pairs.count else { return (0, 0) }
        let pair = pairs[pairIndex]
        pairIndex += 1
        return (Int(pair.run), pair.val)
    }
}

@inline(__always)
private func readUInt32BE(_ data: [UInt8], at offset: inout Int) -> UInt32 {
    let val = (UInt32(data[offset]) << 24) | (UInt32(data[offset+1]) << 16) | (UInt32(data[offset+2]) << 8) | UInt32(data[offset+3])
    offset += 4
    return val
}
