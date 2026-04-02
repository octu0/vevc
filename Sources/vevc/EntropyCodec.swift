import Foundation

// MARK: - ContextModel / Legacy CABAC Support (To be removed)

struct ContextModel {
    var pStateIdx: UInt8 = 0
    var valMPS: UInt8 = 0
    init() {}
}

// MARK: - VevcEncoder

protocol EntropyModelProvider {
    static var isStaticMode: Bool { get }
    static var isDPCMMode: Bool { get }
    static func generateModels(
        runTokenCounts0: inout [Int], valTokenCounts0: inout [Int],
        runTokenCounts1: inout [Int], valTokenCounts1: inout [Int]
    ) -> (runModel0: rANSModel, valModel0: rANSModel, runModel1: rANSModel, valModel1: rANSModel)
    
    static func writeHeaders(
        into out: inout [UInt8],
        runModel0: rANSModel, valModel0: rANSModel,
        runModel1: rANSModel, valModel1: rANSModel
    )
}

struct StaticEntropyModel: EntropyModelProvider {
    static var isStaticMode: Bool { true }
    static var isDPCMMode: Bool { false }
    
    @inline(__always)
    static func generateModels(
        runTokenCounts0: inout [Int], valTokenCounts0: inout [Int],
        runTokenCounts1: inout [Int], valTokenCounts1: inout [Int]
    ) -> (runModel0: rANSModel, valModel0: rANSModel, runModel1: rANSModel, valModel1: rANSModel) {
        return (staticRunModel0, staticValModel0, staticRunModel1, staticValModel1)
    }
    
    @inline(__always)
    static func writeHeaders(
        into out: inout [UInt8],
        runModel0: rANSModel, valModel0: rANSModel,
        runModel1: rANSModel, valModel1: rANSModel
    ) {}
}

struct DynamicEntropyModel: EntropyModelProvider {
    // isStaticMode is dynamic: determined at encode time based on pair count.
    // When pair count < threshold, static tables are used (no header overhead).
    // The actual flag is written in getData() based on the generated models.
    static var isStaticMode: Bool { false }
    static var isDPCMMode: Bool { false }
    
    /// Minimum pair count for dynamic tables to be cost-effective.
    /// Below this threshold, the frequency table header overhead (~400B)
    /// exceeds the compression improvement from data-specific tables.
    /// Determined by breakeven analysis in StaticVsDynamicModelTests.
    private static let dynamicThreshold: Int = 500
    
    @inline(__always)
    static func generateModels(
        runTokenCounts0: inout [Int], valTokenCounts0: inout [Int],
        runTokenCounts1: inout [Int], valTokenCounts1: inout [Int]
    ) -> (runModel0: rANSModel, valModel0: rANSModel, runModel1: rANSModel, valModel1: rANSModel) {
        // Estimate pair count from run token counts (each pair produces one run token)
        let totalPairs: Int = runTokenCounts0.reduce(0, +) + runTokenCounts1.reduce(0, +)
        
        // Fallback to static tables when pair count is below threshold
        // to avoid frequency table header overhead exceeding compression benefit
        if totalPairs < dynamicThreshold {
            return (staticRunModel0, staticValModel0, staticRunModel1, staticValModel1)
        }
        
        var rm0 = rANSModel()
        var vm0 = rANSModel()
        var rm1 = rANSModel()
        var vm1 = rANSModel()
        rm0.normalize(sigCounts: [0, 1], tokenCounts: runTokenCounts0)
        vm0.normalize(sigCounts: [0, 1], tokenCounts: valTokenCounts0)
        rm1.normalize(sigCounts: [0, 1], tokenCounts: runTokenCounts1)
        vm1.normalize(sigCounts: [0, 1], tokenCounts: valTokenCounts1)
        return (rm0, vm0, rm1, vm1)
    }
    
    @inline(__always)
    static func writeHeaders(
        into out: inout [UInt8],
        runModel0: rANSModel, valModel0: rANSModel,
        runModel1: rANSModel, valModel1: rANSModel
    ) {
        // Check if we fell back to static tables by comparing tokenFreqs pointers
        // If the model's tokenFreqs match the static tables, skip writing headers
        if runModel0.tokenFreqs == staticRunModel0.tokenFreqs
            && valModel0.tokenFreqs == staticValModel0.tokenFreqs {
            return
        }
        writeCompressedFreqTable(&out, freqs: runModel0.tokenFreqs)
        writeCompressedFreqTable(&out, freqs: valModel0.tokenFreqs)
        writeCompressedFreqTable(&out, freqs: runModel1.tokenFreqs)
        writeCompressedFreqTable(&out, freqs: valModel1.tokenFreqs)
    }
}

/// Static entropy model specialized for DPCM mode.
/// DPCM does not distinguish isParentZero, so the same tables are used for both isParentZero=false and true.
struct StaticDPCMEntropyModel: EntropyModelProvider {
    static var isStaticMode: Bool { true }
    static var isDPCMMode: Bool { true }
    
    @inline(__always)
    static func generateModels(
        runTokenCounts0: inout [Int], valTokenCounts0: inout [Int],
        runTokenCounts1: inout [Int], valTokenCounts1: inout [Int]
    ) -> (runModel0: rANSModel, valModel0: rANSModel, runModel1: rANSModel, valModel1: rANSModel) {
        // DPCM uses the same model for both isParentZero=false and true
        return (staticDPCMRunModel, staticDPCMValModel, staticDPCMRunModel, staticDPCMValModel)
    }
    
    @inline(__always)
    static func writeHeaders(
        into out: inout [UInt8],
        runModel0: rANSModel, valModel0: rANSModel,
        runModel1: rANSModel, valModel1: rANSModel
    ) {}
}

struct EntropyEncoder<Model: EntropyModelProvider> {
    var bypassWriter: BypassWriter
    var pairs: [(run: UInt32, val: Int16, isParentZero: Bool)]
    var trailingZeros: UInt32
    private(set) var coeffCount: Int

    init() {
        self.bypassWriter = BypassWriter()
        self.pairs = []
        self.pairs.reserveCapacity(512)
        self.trailingZeros = 0
        self.coeffCount = 0
    }

    @inline(__always)
    mutating func encodeBypass(binVal: UInt8) {
        bypassWriter.writeBit(binVal != 0)
    }

    @inline(__always)
    mutating func addPair(run: UInt32, val: Int16, isParentZero: Bool = false) {
        pairs.append((run: run, val: val, isParentZero: isParentZero))
        coeffCount += Int(run) + 1
    }
    
    @inline(__always)
    mutating func addTrailingZeros(_ count: UInt32) {
        trailingZeros += count
        coeffCount += Int(count)
    }

    @inline(__always)
    mutating func flush() {
        bypassWriter.flush()
    }

    @inline(__always)
    mutating func getData() -> [UInt8] {
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
                let valResult = valueTokenize(pair.val)
                rawBypass.writeBits(UInt32(valResult.token), count: 6)
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
        var runTokenCounts0 = Array(repeating: 0, count: 64)
        var valTokenCounts0 = Array(repeating: 0, count: 64)
        var runTokenCounts1 = Array(repeating: 0, count: 64)
        var valTokenCounts1 = Array(repeating: 0, count: 64)
        
        for lane in 0..<4 {
            let start = chunkStarts[lane]
            let end = chunkStarts[lane + 1]
            let chunkSize = end - start
            chunkRunTokens[lane].reserveCapacity(chunkSize + 1)
            chunkValTokens[lane].reserveCapacity(chunkSize)
            
            for idx in start..<end {
                let pair = pairs[idx]
                
                let runResult = valueTokenizeUnsigned(pair.run)
                chunkRunTokens[lane].append(runResult.token)
                chunkBypassWriters[lane].writeBits(runResult.bypassBits, count: runResult.bypassLen)
                
                let valResult = valueTokenize(pair.val)
                chunkValTokens[lane].append(valResult.token)
                chunkBypassWriters[lane].writeBits(valResult.bypassBits, count: valResult.bypassLen)

                if pair.isParentZero {
                    runTokenCounts1[Int(runResult.token)] += 1
                    valTokenCounts1[Int(valResult.token)] += 1
                } else {
                    runTokenCounts0[Int(runResult.token)] += 1
                    valTokenCounts0[Int(valResult.token)] += 1
                }
            }
        }
        
        // trailing zeros: add to lane3
        if hasTrailingZeros {
            let runResult = valueTokenizeUnsigned(trailingZeros)
            runTokenCounts0[Int(runResult.token)] += 1
            chunkRunTokens[3].append(runResult.token)
            chunkBypassWriters[3].writeBits(runResult.bypassBits, count: runResult.bypassLen)
        }
        
        for lane in 0..<4 {
            chunkBypassWriters[lane].flush()
        }
        // Cap frequencies to 16-bit to ensure bitstream serialization perfectly matches what the decoder reads
        for i in 0..<64 {
            if runTokenCounts0[i] > 65535 { runTokenCounts0[i] = 65535 }
            if valTokenCounts0[i] > 65535 { valTokenCounts0[i] = 65535 }
            if runTokenCounts1[i] > 65535 { runTokenCounts1[i] = 65535 }
            if valTokenCounts1[i] > 65535 { valTokenCounts1[i] = 65535 }
        }
        
        let models = Model.generateModels(
            runTokenCounts0: &runTokenCounts0, valTokenCounts0: &valTokenCounts0,
            runTokenCounts1: &runTokenCounts1, valTokenCounts1: &valTokenCounts1
        )
        let runModel0 = models.runModel0
        let valModel0 = models.valModel0
        let runModel1 = models.runModel1
        let valModel1 = models.valModel1
        
        // Flags byte: bit6=isStatic, bit5=isDPCM, bit0=hasTrailingZeros
        // Determine isStatic dynamically: if writeHeaders writes nothing,
        // the model fell back to static tables (hybrid mode).
        let trailBit: UInt8  = hasTrailingZeros    ? 0x01 : 0
        let dpcmBit: UInt8   = Model.isDPCMMode   ? 0x20 : 0
        
        // Write a placeholder for flags byte, then headers, then fix the flag
        let flagsOffset = out.count
        out.append(0) // placeholder
        appendUInt32BE(&out, UInt32(totalPairEntries))
        
        // chunk size (4B × 4)
        for lane in 0..<4 {
            let chunkPairCount = chunkStarts[lane + 1] - chunkStarts[lane]
            let extraTrailing = (lane == 3 && hasTrailingZeros) ? 1 : 0
            appendUInt32BE(&out, UInt32(chunkPairCount + extraTrailing))
        }
        
        let preHeaderSize = out.count
        Model.writeHeaders(into: &out, runModel0: runModel0, valModel0: valModel0, runModel1: runModel1, valModel1: valModel1)
        let headerWasWritten = out.count > preHeaderSize
        
        // Set the isStatic flag based on whether headers were actually written
        let staticBit: UInt8 = (Model.isStaticMode || !headerWasWritten) ? 0x40 : 0
        out[flagsOffset] = staticBit | dpcmBit | trailBit
        
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
            enc.encodeSymbol(lane: 3, cumFreq: runModel0.tokenCumFreqs[Int(trailingRunToken)], freq: runModel0.tokenFreqs[Int(trailingRunToken)])
        }
        
        // encode each lane in reverse order
        for lane in stride(from: 3, through: 0, by: -1) {
            let runTokens = chunkRunTokens[lane]
            let valTokens = chunkValTokens[lane]
            let pairEnd = valTokens.count
            let chunkStartIdx = chunkStarts[lane]
            
            for i in stride(from: pairEnd - 1, through: 0, by: -1) {
                let pairIdx = chunkStartIdx + i
                let isParentZero = pairs[pairIdx].isParentZero
                let vt = valTokens[i]
                if isParentZero {
                    enc.encodeSymbol(lane: lane, cumFreq: valModel1.tokenCumFreqs[Int(vt)], freq: valModel1.tokenFreqs[Int(vt)])
                } else {
                    enc.encodeSymbol(lane: lane, cumFreq: valModel0.tokenCumFreqs[Int(vt)], freq: valModel0.tokenFreqs[Int(vt)])
                }
                
                let rt = runTokens[i]
                if isParentZero {
                    enc.encodeSymbol(lane: lane, cumFreq: runModel1.tokenCumFreqs[Int(rt)], freq: runModel1.tokenFreqs[Int(rt)])
                } else {
                    enc.encodeSymbol(lane: lane, cumFreq: runModel0.tokenCumFreqs[Int(rt)], freq: runModel0.tokenFreqs[Int(rt)])
                }
            }
        }
        
        enc.flush()
        out.append(contentsOf: enc.getBitstream())
        
        return out
    }
    
    @inline(__always)
    private func appendUInt32BE(_ out: inout [UInt8], _ val: UInt32) {
        out.append(UInt8((val >> 24) & 0xFF))
        out.append(UInt8((val >> 16) & 0xFF))
        out.append(UInt8((val >> 8) & 0xFF))
        out.append(UInt8(val & 0xFF))
    }
}

@inline(__always)
internal func writeCompressedFreqTable(_ out: inout [UInt8], freqs: [UInt32]) {
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

// MARK: - VevcDecoder

struct EntropyDecoder {
    var bypassReader: BypassReader
    var pairs: [(run: UInt32, val: Int16)] = []
    private var pairIndex: Int = 0
    
    private var isRawMode: Bool = false
    private var totalPairEntries: Int = 0
    private var chunkStarts: [Int] = []
    private var hasTrailingZeros: Bool = false
    private var runModel0: rANSModel!
    private var valModel0: rANSModel!
    private var runModel1: rANSModel!
    private var valModel1: rANSModel!
    private var chunkBypassReaders: [BypassReader] = []
    private var ransDecoder: Interleaved4rANSDecoder!

    init(data: [UInt8]) throws {
        var offset = 0
        
        guard 4 <= data.count else { throw DecodeError.insufficientData }
        let bypassLen = try readUInt32BEFromBytes(data, offset: &offset)
        
        guard offset + Int(bypassLen) <= data.count else { throw DecodeError.insufficientData }
        let bypassData = Array(data[offset..<(offset + Int(bypassLen))])
        self.bypassReader = BypassReader(data: bypassData)
        offset += Int(bypassLen)
        
        let coeffCount = Int(try readUInt32BEFromBytes(data, offset: &offset))
        
        guard 0 < coeffCount else {
            self.pairs = []
            return
        }
        
        guard offset < data.count else { throw DecodeError.insufficientData }
        let flags = data[offset]
        offset += 1
        
        let isRawMode = (flags & 0x80) != 0
        self.isRawMode = isRawMode
        
        if isRawMode {
            let rawDataLen = Int(try readUInt32BEFromBytes(data, offset: &offset))
            guard offset + rawDataLen <= data.count else { throw DecodeError.insufficientData }
            let rawData = Array(data[offset..<(offset + rawDataLen)])
            var rawReader = BypassReader(data: rawData)
            
            var decodedPairs = [(run: UInt32, val: Int16)]()
            var zeroRun: UInt32 = 0
            for _ in 0..<coeffCount {
                let isNonZero = rawReader.readBit()
                if isNonZero {
                    let tokenBits = rawReader.readBits(count: 6)
                    let token = UInt8(tokenBits)
                    let bypassLen = valueBypassLength(for: token)
                    let bypassBits = rawReader.readBits(count: bypassLen)
                    let val = valueDetokenize(token: token, bypassBits: bypassBits)
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
        let isStaticTable = (flags & 0x40) != 0
        let hasTrailingZeros = (flags & 1) != 0
        self.hasTrailingZeros = hasTrailingZeros
        
        let totalPairEntries = Int(try readUInt32BEFromBytes(data, offset: &offset))
        self.totalPairEntries = totalPairEntries
        
        // chunk size (4 lanes)
        var chunkSizes = [Int](repeating: 0, count: 4)
        for lane in 0..<4 {
            chunkSizes[lane] = Int(try readUInt32BEFromBytes(data, offset: &offset))
        }
        
        var starts = [Int](repeating: 0, count: 5)
        for i in 0..<4 {
            starts[i+1] = starts[i] + chunkSizes[i]
        }
        self.chunkStarts = starts
        
        let isDPCMTable = (flags & 0x20) != 0
        
        if isStaticTable, isDPCMTable {
            // Static DPCM mode: use DPCM-specific static tables
            self.runModel0 = staticDPCMRunModel
            self.valModel0 = staticDPCMValModel
            self.runModel1 = staticDPCMRunModel
            self.valModel1 = staticDPCMValModel
        } else if isStaticTable {
            // Static DWT mode: no frequency tables in bitstream
            self.runModel0 = staticRunModel0
            self.valModel0 = staticValModel0
            self.runModel1 = staticRunModel1
            self.valModel1 = staticValModel1
        } else {
            // Legacy dynamic table mode: read frequency tables from bitstream
            let runTokenFreqs0 = try EntropyDecoder.readCompressedFreqTable(data, at: &offset)
            self.runModel0 = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: runTokenFreqs0)
            
            let valTokenFreqs0 = try EntropyDecoder.readCompressedFreqTable(data, at: &offset)
            self.valModel0 = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: valTokenFreqs0)
            
            let runTokenFreqs1 = try EntropyDecoder.readCompressedFreqTable(data, at: &offset)
            self.runModel1 = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: runTokenFreqs1)
            
            let valTokenFreqs1 = try EntropyDecoder.readCompressedFreqTable(data, at: &offset)
            self.valModel1 = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: valTokenFreqs1)
        }
        
        // 4-way bypass data
        var chunkBypassReaders = [BypassReader]()
        for _ in 0..<4 {
            let bpLen = Int(try readUInt32BEFromBytes(data, offset: &offset))
            guard offset + bpLen <= data.count else { throw DecodeError.insufficientData }
            let bpData = Array(data[offset..<(offset + bpLen)])
            chunkBypassReaders.append(BypassReader(data: bpData))
            offset += bpLen
        }
        self.chunkBypassReaders = chunkBypassReaders
        
        // rANS stream
        let ransData = Array(data[offset...])
        self.ransDecoder = Interleaved4rANSDecoder(bitstream: ransData)
    }
    
    @inline(__always)
    internal static func readCompressedFreqTable(_ data: [UInt8], at offset: inout Int) throws -> [UInt32] {
        let bitmap = try readUInt64BEFromBytes(data, offset: &offset)
        
        var freqs = [UInt32](repeating: 1, count: 64)
        for i in 0..<64 {
            if (bitmap & (UInt64(1) << i)) != 0 {
                freqs[i] = UInt32(try readUInt16BEFromBytes(data, offset: &offset))
            }
        }
        return freqs
    }

    @inline(__always)
    mutating func decodeBypass() throws -> UInt8 {
        return bypassReader.readBit() ? 1 : 0
    }
    
    @inline(__always)
    mutating func readPair(isParentZero: Bool = false) -> (run: Int, val: Int16) {
        if isRawMode {
            guard pairIndex < pairs.count else { return (0, 0) }
            let pair = pairs[pairIndex]
            pairIndex += 1
            return (Int(pair.run), pair.val)
        }
        
        guard pairIndex < totalPairEntries else { return (0, 0) }
        
        var lane = 0
        if pairIndex >= chunkStarts[3] { lane = 3 }
        else if pairIndex >= chunkStarts[2] { lane = 2 }
        else if pairIndex >= chunkStarts[1] { lane = 1 }
        
        let isTZPair = (lane == 3 && hasTrailingZeros && pairIndex == chunkStarts[4] - 1)
        
        if isTZPair {
            let cfRun = ransDecoder.getCumulativeFreq(lane: lane)
            let rtInfo = runModel0.findToken(cf: cfRun)
            ransDecoder.advanceSymbol(lane: lane, cumFreq: rtInfo.cumFreq, freq: rtInfo.freq)
            
            let runBypassLen = valueBypassLengthUnsigned(for: rtInfo.token)
            let runBypassBits = chunkBypassReaders[lane].readBits(count: runBypassLen)
            let zeroRun = UInt32(valueDetokenizeUnsigned(token: rtInfo.token, bypassBits: runBypassBits))
            
            pairIndex += 1
            return (Int(zeroRun), 0)
        } else {
            let cfRun = ransDecoder.getCumulativeFreq(lane: lane)
            let rtInfo = isParentZero ? runModel1.findToken(cf: cfRun) : runModel0.findToken(cf: cfRun)
            ransDecoder.advanceSymbol(lane: lane, cumFreq: rtInfo.cumFreq, freq: rtInfo.freq)
            
            let runBypassLen = valueBypassLengthUnsigned(for: rtInfo.token)
            let runBypassBits = chunkBypassReaders[lane].readBits(count: runBypassLen)
            let zeroRun = UInt32(valueDetokenizeUnsigned(token: rtInfo.token, bypassBits: runBypassBits))
            
            let cfVal = ransDecoder.getCumulativeFreq(lane: lane)
            let vtInfo = isParentZero ? valModel1.findToken(cf: cfVal) : valModel0.findToken(cf: cfVal)
            ransDecoder.advanceSymbol(lane: lane, cumFreq: vtInfo.cumFreq, freq: vtInfo.freq)
            
            let valBypassLen = valueBypassLength(for: vtInfo.token)
            let valBypassBits = chunkBypassReaders[lane].readBits(count: valBypassLen)
            let val = valueDetokenize(token: vtInfo.token, bypassBits: valBypassBits)
            
            pairIndex += 1
            return (Int(zeroRun), val)
        }
    }
}

// MARK: - Motion Vector rANS Codec

@inline(__always)
func encodeMVs(mvs: [MotionVector]) -> [UInt8] {
    var tokensDx = [UInt8]()
    var tokensDy = [UInt8]()
    tokensDx.reserveCapacity(mvs.count)
    tokensDy.reserveCapacity(mvs.count)
    
    var bypass = BypassWriter()
    var freqsDx = [UInt32](repeating: 0, count: 64)
    var freqsDy = [UInt32](repeating: 0, count: 64)
    
    for mv in mvs {
        let tx = valueTokenize(mv.dx)
        tokensDx.append(tx.token)
        freqsDx[Int(tx.token)] += 1
        bypass.writeBits(tx.bypassBits, count: tx.bypassLen)

        let ty = valueTokenize(mv.dy)
        tokensDy.append(ty.token)
        freqsDy[Int(ty.token)] += 1
        bypass.writeBits(ty.bypassBits, count: ty.bypassLen)
    }
    bypass.flush()
    
    var modelDx = rANSModel()
    modelDx.normalize(sigCounts: [0, 0], tokenCounts: freqsDx.map(Int.init))
    
    var modelDy = rANSModel()
    modelDy.normalize(sigCounts: [0, 0], tokenCounts: freqsDy.map(Int.init))
    
    var enc = rANSEncoder()
    for i in stride(from: mvs.count - 1, through: 0, by: -1) {
        let ty = tokensDy[i]
        enc.encodeSymbol(cumFreq: modelDy.tokenCumFreqs[Int(ty)], freq: modelDy.tokenFreqs[Int(ty)])
        let tx = tokensDx[i]
        enc.encodeSymbol(cumFreq: modelDx.tokenCumFreqs[Int(tx)], freq: modelDx.tokenFreqs[Int(tx)])
    }
    enc.flush()
    
    var out = [UInt8]()
    writeCompressedFreqTable(&out, freqs: modelDx.tokenFreqs)
    writeCompressedFreqTable(&out, freqs: modelDy.tokenFreqs)
    
    let bpData = bypass.bytes
    out.append(UInt8((bpData.count >> 24) & 0xFF))
    out.append(UInt8((bpData.count >> 16) & 0xFF))
    out.append(UInt8((bpData.count >> 8) & 0xFF))
    out.append(UInt8(bpData.count & 0xFF))
    out.append(contentsOf: bpData)
    
    out.append(contentsOf: enc.getBitstream())
    return out
}

@inline(__always)
func decodeMVs(data: [UInt8], count: Int) throws -> [MotionVector] {
    var offset = 0
    let freqsDx = try EntropyDecoder.readCompressedFreqTable(data, at: &offset)
    let modelDx = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: freqsDx)
    
    let freqsDy = try EntropyDecoder.readCompressedFreqTable(data, at: &offset)
    let modelDy = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: freqsDy)
    
    let bpLen = Int(try readUInt32BEFromBytes(data, offset: &offset))
    guard offset + bpLen <= data.count else { throw DecodeError.insufficientData }
    var bypassReader = BypassReader(data: Array(data[offset..<(offset + bpLen)]))
    offset += bpLen
    
    guard offset < data.count else { throw DecodeError.insufficientData }
    var dec = rANSDecoder(bitstream: Array(data[offset...]))
    
    var mvs = [MotionVector]()
    mvs.reserveCapacity(count)
    
    for _ in 0..<count {
        let cfDx = dec.getCumulativeFreq()
        let txInfo = modelDx.findToken(cf: cfDx)
        dec.advanceSymbol(cumFreq: txInfo.cumFreq, freq: txInfo.freq)
        
        let cfDy = dec.getCumulativeFreq()
        let tyInfo = modelDy.findToken(cf: cfDy)
        dec.advanceSymbol(cumFreq: tyInfo.cumFreq, freq: tyInfo.freq)
        
        let dxBypassLen = valueBypassLength(for: txInfo.token)
        let dxBypassBits = bypassReader.readBits(count: dxBypassLen)
        let dx = valueDetokenize(token: txInfo.token, bypassBits: dxBypassBits)
        
        let dyBypassLen = valueBypassLength(for: tyInfo.token)
        let dyBypassBits = bypassReader.readBits(count: dyBypassLen)
        let dy = valueDetokenize(token: tyInfo.token, bypassBits: dyBypassBits)
        
        mvs.append(MotionVector(dx: dx, dy: dy))
    }
    return mvs
}

// MARK: - Intra Mode rANS Codec

@inline(__always)
func encodeModes(modes: [UInt8]) -> [UInt8] {
    var freqs = [UInt32](repeating: 0, count: 64)
    for m in modes {
        freqs[Int(m)] += 1
    }
    
    var model = rANSModel()
    model.normalize(sigCounts: [0, 0], tokenCounts: freqs.map(Int.init))
    
    var enc = rANSEncoder()
    for i in stride(from: modes.count - 1, through: 0, by: -1) {
        let m = Int(modes[i])
        enc.encodeSymbol(cumFreq: model.tokenCumFreqs[m], freq: model.tokenFreqs[m])
    }
    enc.flush()
    
    var out = [UInt8]()
    writeCompressedFreqTable(&out, freqs: model.tokenFreqs)
    out.append(contentsOf: enc.getBitstream())
    return out
}

@inline(__always)
func decodeModes(data: [UInt8], count: Int) throws -> ([UInt8], Int) {
    var offset = 0
    let freqs = try EntropyDecoder.readCompressedFreqTable(data, at: &offset)
    let model = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: freqs)
    
    var dec = rANSDecoder(bitstream: Array(data[offset...]))
    var modes = [UInt8]()
    modes.reserveCapacity(count)
    
    for _ in 0..<count {
        let cf = dec.getCumulativeFreq()
        let tInfo = model.findToken(cf: cf)
        dec.advanceSymbol(cumFreq: tInfo.cumFreq, freq: tInfo.freq)
        modes.append(tInfo.token)
    }
    
    // bitstream consumed bytes = offset + dec.consumedBytes
    return (modes, offset + dec.getConsumedBytes())
}
