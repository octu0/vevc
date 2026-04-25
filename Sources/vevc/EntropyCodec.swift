import Foundation

// MARK: - VevcEncoder

protocol EntropyModelProvider {
    static var isStaticMode: Bool { get }
    static var isDPCMMode: Bool { get }
    static func generateModels(
        runTokenCounts: inout [[Int]], valTokenCounts: inout [[Int]]
    ) -> (runModels: [rANSModel], valModels: [rANSModel])
    
    static func writeHeaders(
        into out: inout [UInt8],
        runModels: [rANSModel], valModels: [rANSModel]
    )
}

struct StaticEntropyModel: EntropyModelProvider {
    static var isStaticMode: Bool { true }
    static var isDPCMMode: Bool { false }
    
    @inline(__always)
    static func generateModels(
        runTokenCounts: inout [[Int]], valTokenCounts: inout [[Int]]
    ) -> (runModels: [rANSModel], valModels: [rANSModel]) {
        return (
            [StaticRANSModels.shared.runModel0, StaticRANSModels.shared.runModel1, StaticRANSModels.shared.runModel2, StaticRANSModels.shared.runModel3],
            [StaticRANSModels.shared.valModel0, StaticRANSModels.shared.valModel1, StaticRANSModels.shared.valModel2, StaticRANSModels.shared.valModel3]
        )
    }
    
    @inline(__always)
    static func writeHeaders(
        into out: inout [UInt8],
        runModels: [rANSModel], valModels: [rANSModel]
    ) {}
}

struct DynamicEntropyModel: EntropyModelProvider {
    static var isStaticMode: Bool { false }
    static var isDPCMMode: Bool { false }
    
    private static let dynamicThreshold: Int = 100
    
    @inline(__always)
    static func generateModels(
        runTokenCounts: inout [[Int]], valTokenCounts: inout [[Int]]
    ) -> (runModels: [rANSModel], valModels: [rANSModel]) {
        let totalPairs = runTokenCounts.reduce(0) { $0 + $1.reduce(0, +) }
        
        if totalPairs < dynamicThreshold {
            return (
                [StaticRANSModels.shared.runModel0, StaticRANSModels.shared.runModel1, StaticRANSModels.shared.runModel2, StaticRANSModels.shared.runModel3],
                [StaticRANSModels.shared.valModel0, StaticRANSModels.shared.valModel1, StaticRANSModels.shared.valModel2, StaticRANSModels.shared.valModel3]
            )
        }
        
        var runModels = [rANSModel]()
        var valModels = [rANSModel]()
        for i in 0..<4 {
            var rm = rANSModel()
            var vm = rANSModel()
            rm.normalize(sigCounts: [0, 1], tokenCounts: runTokenCounts[i])
            vm.normalize(sigCounts: [0, 1], tokenCounts: valTokenCounts[i])
            runModels.append(rm)
            valModels.append(vm)
        }
        return (runModels, valModels)
    }
    
    @inline(__always)
    static func writeHeaders(
        into out: inout [UInt8],
        runModels: [rANSModel], valModels: [rANSModel]
    ) {
        if runModels[0].tokenFreqs == StaticRANSModels.shared.runModel0.tokenFreqs && valModels[0].tokenFreqs == StaticRANSModels.shared.valModel0.tokenFreqs {
            return
        }
        for i in 0..<4 {
            writeCompressedFreqTable(&out, freqs: runModels[i].tokenFreqs)
            writeCompressedFreqTable(&out, freqs: valModels[i].tokenFreqs)
        }
    }
}

struct StaticDPCMEntropyModel: EntropyModelProvider {
    static var isStaticMode: Bool { true }
    static var isDPCMMode: Bool { true }
    
    @inline(__always)
    static func generateModels(
        runTokenCounts: inout [[Int]], valTokenCounts: inout [[Int]]
    ) -> (runModels: [rANSModel], valModels: [rANSModel]) {
        let dpcmRun = StaticRANSModels.shared.dpcmRunModel
        let dpcmVal = StaticRANSModels.shared.dpcmValModel
        return ([dpcmRun, dpcmRun, dpcmRun, dpcmRun], [dpcmVal, dpcmVal, dpcmVal, dpcmVal])
    }
    
    @inline(__always)
    static func writeHeaders(
        into out: inout [UInt8],
        runModels: [rANSModel], valModels: [rANSModel]
    ) {}
}

struct EntropyEncoder<Model: EntropyModelProvider> {
    var bypassWriter: BypassWriter
    /// SoA (Structure of Arrays): eliminates tuple-array padding for better cache efficiency
    var pairRuns: [UInt32]
    var pairVals: [Int16]
    var pairContexts: [UInt8]
    var trailingZeros: UInt32
    private(set) var coeffCount: Int

    init() {
        self.bypassWriter = BypassWriter()
        self.pairRuns = []
        self.pairVals = []
        self.pairContexts = []
        self.pairRuns.reserveCapacity(512)
        self.pairVals.reserveCapacity(512)
        self.pairContexts.reserveCapacity(512)
        self.trailingZeros = 0
        self.coeffCount = 0
    }
    
    /// Computed property for test compatibility (not used in production)
    var pairs: [(run: UInt32, val: Int16, context: UInt8)] {
        (0..<pairRuns.count).map { i in
            (run: pairRuns[i], val: pairVals[i], context: pairContexts[i])
        }
    }

    @inline(__always)
    mutating func addTrailingZeros(_ count: UInt32) {
        trailingZeros += count
        coeffCount += Int(count)
    }
    
    @inline(__always)
    mutating func addPair(run: UInt32, val: Int16, context: UInt8) {
        pairRuns.append(run)
        pairVals.append(val)
        pairContexts.append(context)
        coeffCount += Int(run) + 1
        
        if ModelTrainer.shared.isTrainingMode {
            ModelTrainer.shared.record(run: run, val: val, context: context)
        }
    }

    @inline(__always)
    mutating func encodeBypass(binVal: UInt8) {
        bypassWriter.writeBit(binVal != 0)
    }

    @inline(__always)
    mutating func flush() {
        bypassWriter.flush()
    }

    @inline(__always)
    mutating func getData() -> [UInt8] {
        var out = [UInt8]()
        let pairCount = pairRuns.count
        out.reserveCapacity(pairCount * 4 + 128)
        
        bypassWriter.flush()
        let metaBypassData = bypassWriter.bytes
        appendUInt32BE(&out, UInt32(metaBypassData.count))
        out.append(contentsOf: metaBypassData)
        
        appendUInt32BE(&out, UInt32(coeffCount))
        
        guard 0 < pairCount || 0 < trailingZeros else { return out }
        
        let hasTrailingZeros = 0 < trailingZeros
        let nonZeroCount = pairCount
        
        if nonZeroCount <= 32 {
            out.append(0x80)
            var rawBypass = BypassWriter()
            for i in 0..<pairCount {
                // for zero-run
                for _ in 0..<pairRuns[i] {
                    rawBypass.writeBit(false)
                }
                rawBypass.writeBit(true)
                let valResult = valueTokenize(pairVals[i])
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
        let totalPairEntries = if hasTrailingZeros { pairCount + 1 } else { pairCount }
        let chunkBase = pairCount / 4
        let chunkRemainder = pairCount % 4
        
        // chunk boundary: chunk[i] is responsible for pairs[chunkStarts[i]..<chunkStarts[i+1]]
        var chunkStarts = [Int](repeating: 0, count: 5)
        for i in 0..<4 {
            chunkStarts[i + 1] = if i < chunkRemainder { chunkStarts[i] + chunkBase + 1 } else { chunkStarts[i] + chunkBase }
        }
        
        // tokenization + bypass writing for each chunk
        var chunkRunTokens = [[UInt8]](repeating: [], count: 4)
        var chunkValTokens = [[UInt8]](repeating: [], count: 4)
        var chunkBypassWriters = [BypassWriter](repeating: BypassWriter(), count: 4)
        var runTokenCounts = [[Int]](repeating: Array(repeating: 0, count: 64), count: 4)
        var valTokenCounts = [[Int]](repeating: Array(repeating: 0, count: 64), count: 4)
        
        for lane in 0..<4 {
            let start = chunkStarts[lane]
            let end = chunkStarts[lane + 1]
            let chunkSize = end - start
            chunkRunTokens[lane].reserveCapacity(chunkSize + 1)
            chunkValTokens[lane].reserveCapacity(chunkSize)
            
            for idx in start..<end {
                let runResult = valueTokenizeUnsigned(pairRuns[idx])
                chunkRunTokens[lane].append(runResult.token)
                chunkBypassWriters[lane].writeBits(runResult.bypassBits, count: runResult.bypassLen)
                
                let valResult = valueTokenize(pairVals[idx])
                chunkValTokens[lane].append(valResult.token)
                chunkBypassWriters[lane].writeBits(valResult.bypassBits, count: valResult.bypassLen)

                let ctx = Int(pairContexts[idx])
                runTokenCounts[ctx][Int(runResult.token)] += 1
                valTokenCounts[ctx][Int(valResult.token)] += 1
            }
        }
        
        // trailing zeros: add to lane3
        if hasTrailingZeros {
            let runResult = valueTokenizeUnsigned(trailingZeros)
            runTokenCounts[0][Int(runResult.token)] += 1
            chunkRunTokens[3].append(runResult.token)
            chunkBypassWriters[3].writeBits(runResult.bypassBits, count: runResult.bypassLen)
        }
        
        for lane in 0..<4 {
            chunkBypassWriters[lane].flush()
        }
        // Cap frequencies to 16-bit to ensure bitstream serialization perfectly matches what the decoder reads
        for c in 0..<4 {
            for i in 0..<64 {
                if 65535 < runTokenCounts[c][i] { runTokenCounts[c][i] = 65535 }
                if 65535 < valTokenCounts[c][i] { valTokenCounts[c][i] = 65535 }
            }
        }
        
        let models = Model.generateModels(
            runTokenCounts: &runTokenCounts, valTokenCounts: &valTokenCounts
        )
        let runModels = models.runModels
        let valModels = models.valModels
        
        // Flags byte: bit6=isStatic, bit5=isDPCM, bit0=hasTrailingZeros
        // Determine isStatic dynamically: if writeHeaders writes nothing,
        // the model fell back to static tables (hybrid mode).
        let FLAG_TRAILING_ZEROS: UInt8 = 0x01
        let FLAG_DPCM: UInt8           = 0x20
        let FLAG_STATIC: UInt8         = 0x40
        
        let trailBit: UInt8  = if hasTrailingZeros { FLAG_TRAILING_ZEROS } else { 0 }
        let dpcmBit: UInt8   = if Model.isDPCMMode { FLAG_DPCM } else { 0 }
        
        // Write a placeholder for flags byte, then headers, then fix the flag
        let flagsOffset = out.count
        out.append(0) // placeholder
        appendUInt32BE(&out, UInt32(totalPairEntries))
        
        // chunk size (4B × 4)
        for lane in 0..<4 {
            let chunkPairCount = chunkStarts[lane + 1] - chunkStarts[lane]
            let extraTrailing = if lane == 3 && hasTrailingZeros { 1 } else { 0 }
            appendUInt32BE(&out, UInt32(chunkPairCount + extraTrailing))
        }
        
        let preHeaderSize = out.count
        Model.writeHeaders(into: &out, runModels: runModels, valModels: valModels)
        let headerWasWritten = preHeaderSize < out.count
        
        // why: when pair count is small, static tables produce better compression
        // than writing per-stream frequency headers
        let staticBit: UInt8 = if Model.isStaticMode || (headerWasWritten != true) { FLAG_STATIC } else { 0 }
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
            // Trailing zeros always use context 0
            enc.encodeSymbol(lane: 3, cumFreq: runModels[0].tokenCumFreqs[Int(trailingRunToken)], freq: runModels[0].tokenFreqs[Int(trailingRunToken)])
        }
        
        // encode each lane in reverse order
        for lane in stride(from: 3, through: 0, by: -1) {
            let runTokens = chunkRunTokens[lane]
            let valTokens = chunkValTokens[lane]
            let pairEnd = valTokens.count
            let chunkStartIdx = chunkStarts[lane]
            
            for i in stride(from: pairEnd - 1, through: 0, by: -1) {
                let pairIdx = chunkStartIdx + i
                let ctx = Int(pairContexts[pairIdx])
                
                let vt = valTokens[i]
                enc.encodeSymbol(lane: lane, cumFreq: valModels[ctx].tokenCumFreqs[Int(vt)], freq: valModels[ctx].tokenFreqs[Int(vt)])
                
                let rt = runTokens[i]
                enc.encodeSymbol(lane: lane, cumFreq: runModels[ctx].tokenCumFreqs[Int(rt)], freq: runModels[ctx].tokenFreqs[Int(rt)])
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
        if 1 < freqs[i] {
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

// MARK: - ModelTrainer

public final class ModelTrainer: @unchecked Sendable {
    public static let shared = ModelTrainer()
    
    public var isTrainingMode = false
    
    // 4 contexts, 64 tokens each
    public var runFreqs: [[UInt32]]
    public var valFreqs: [[UInt32]]
    
    private init() {
        self.runFreqs = Array(repeating: Array(repeating: 0, count: 64), count: 4)
        self.valFreqs = Array(repeating: Array(repeating: 0, count: 64), count: 4)
    }
    
    public func reset() {
        self.runFreqs = Array(repeating: Array(repeating: 0, count: 64), count: 4)
        self.valFreqs = Array(repeating: Array(repeating: 0, count: 64), count: 4)
    }
    
    @inline(__always)
    public func record(run: UInt32, val: Int16, context: UInt8) {
        let ctx = Int(context)
        guard ctx >= 0 && ctx < 4 else { return }
        
        let runToken = min(run, 63)
        let valToken = valueTokenize(val).token
        
        runFreqs[ctx][Int(runToken)] &+= 1
        
        if valToken < 64 {
            valFreqs[ctx][Int(valToken)] &+= 1
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
    private var runModels: [rANSModel] = []
    private var valModels: [rANSModel] = []
    private var chunkBypassReaders: [BypassReader] = []
    private var ransDecoder: Interleaved4rANSDecoder!
    private var currentLane: Int = 0

    init(base: UnsafePointer<UInt8>, count: Int, startOffset: Int = 0) throws {
        var offset = startOffset
        
        let bypassLen = try readUInt32BEFromPtr(base, offset: &offset, count: count)
        guard offset + Int(bypassLen) <= count else { throw DecodeError.insufficientData }
        
        self.bypassReader = BypassReader(base: base.advanced(by: offset), count: Int(bypassLen))
        offset += Int(bypassLen)
        
        let coeffCount = Int(try readUInt32BEFromPtr(base, offset: &offset, count: count))
        
        guard 0 < coeffCount else {
            self.pairs = []
            return
        }
        
        guard offset < count else { throw DecodeError.insufficientData }
        let flags = base[offset]
        offset += 1
        
        let isRawMode = (flags & 0x80) != 0
        self.isRawMode = isRawMode
        
        if isRawMode {
            let rawDataLen = Int(try readUInt32BEFromPtr(base, offset: &offset, count: count))
            guard offset + rawDataLen <= count else { throw DecodeError.insufficientData }
            var rawReader = BypassReader(base: base.advanced(by: offset), count: rawDataLen)
            
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
        
        let totalPairEntries = Int(try readUInt32BEFromPtr(base, offset: &offset, count: count))
        self.totalPairEntries = totalPairEntries
        
        // chunk size (4 lanes)
        var chunkSizes = [Int](repeating: 0, count: 4)
        for lane in 0..<4 {
            chunkSizes[lane] = Int(try readUInt32BEFromPtr(base, offset: &offset, count: count))
        }
        
        var starts = [Int](repeating: 0, count: 5)
        for i in 0..<4 {
            starts[i+1] = starts[i] + chunkSizes[i]
        }
        self.chunkStarts = starts
        
        let isDPCMTable = (flags & 0x20) != 0
        
        switch (isStaticTable, isDPCMTable) {
        case (true, true):
            let runM = StaticRANSModels.shared.dpcmRunModel
            let valM = StaticRANSModels.shared.dpcmValModel
            self.runModels = [runM, runM, runM, runM]
            self.valModels = [valM, valM, valM, valM]
        case (true, false):
            self.runModels = [StaticRANSModels.shared.runModel0, StaticRANSModels.shared.runModel1, StaticRANSModels.shared.runModel2, StaticRANSModels.shared.runModel3]
            self.valModels = [StaticRANSModels.shared.valModel0, StaticRANSModels.shared.valModel1, StaticRANSModels.shared.valModel2, StaticRANSModels.shared.valModel3]
        case (false, _):
            var rModels = [rANSModel]()
            var vModels = [rANSModel]()
            for _ in 0..<4 {
                let runFreqs = try EntropyDecoder.readCompressedFreqTable(base, at: &offset, count: count)
                rModels.append(rANSModel(sigFreq: rANSScale / 2, tokenFreqs: runFreqs))
                
                let valFreqs = try EntropyDecoder.readCompressedFreqTable(base, at: &offset, count: count)
                vModels.append(rANSModel(sigFreq: rANSScale / 2, tokenFreqs: valFreqs))
            }
            self.runModels = rModels
            self.valModels = vModels
        }
        
        // 4-way bypass data
        var chunkBypassReaders = [BypassReader]()
        for _ in 0..<4 {
            let bpLen = Int(try readUInt32BEFromPtr(base, offset: &offset, count: count))
            guard offset + bpLen <= count else { throw DecodeError.insufficientData }
            chunkBypassReaders.append(BypassReader(base: base.advanced(by: offset), count: bpLen))
            offset += bpLen
        }
        self.chunkBypassReaders = chunkBypassReaders
        
        // rANS stream
        self.ransDecoder = Interleaved4rANSDecoder(base: base.advanced(by: offset), count: count - offset)
    }
    
    @inline(__always)
    internal static func readCompressedFreqTable(_ base: UnsafePointer<UInt8>, at offset: inout Int, count: Int) throws -> [UInt32] {
        let bitmap = try readUInt64BEFromPtr(base, offset: &offset, count: count)
        
        var freqs = [UInt32](repeating: 1, count: 64)
        for i in 0..<64 {
            if (bitmap & (UInt64(1) << i)) != 0 {
                freqs[i] = UInt32(try readUInt16BEFromPtr(base, offset: &offset, count: count))
            }
        }
        return freqs
    }

    @inline(__always)
    mutating func decodeBypass() throws -> UInt8 {
        if bypassReader.readBit() {
            return 1
        }
        return 0
    }
    
    @inline(__always)
    mutating func readPair(context: UInt8) -> (run: Int, val: Int16) {
        if isRawMode {
            guard pairIndex < pairs.count else { return (0, 0) }
            let pair = pairs[pairIndex]
            pairIndex += 1
            return (Int(pair.run), pair.val)
        }
        
        guard pairIndex < totalPairEntries else { return (0, 0) }
        
        // pairIndex increases monotonically; increment chunk index only at lane boundaries
        while currentLane < 3 && pairIndex >= chunkStarts[currentLane + 1] {
            currentLane += 1
        }
        let lane = currentLane
        
        let isTZPair = (lane == 3 && hasTrailingZeros && pairIndex == chunkStarts[4] - 1)
        
        if isTZPair {
            let cfRun = ransDecoder.getCumulativeFreq(lane: lane)
            let rtInfo = runModels[0].findToken(cf: cfRun)
            ransDecoder.advanceSymbol(lane: lane, cumFreq: rtInfo.cumFreq, freq: rtInfo.freq)
            
            let runBypassLen = valueBypassLengthUnsigned(for: rtInfo.token)
            let runBypassBits = chunkBypassReaders[lane].readBits(count: runBypassLen)
            let zeroRun = UInt32(valueDetokenizeUnsigned(token: rtInfo.token, bypassBits: runBypassBits))
            
            pairIndex += 1
            return (Int(zeroRun), 0)
        } else {
            let cfRun = ransDecoder.getCumulativeFreq(lane: lane)
            let ctx = Int(context)
            let rtInfo = runModels[ctx].findToken(cf: cfRun)
            ransDecoder.advanceSymbol(lane: lane, cumFreq: rtInfo.cumFreq, freq: rtInfo.freq)
            
            let runBypassLen = valueBypassLengthUnsigned(for: rtInfo.token)
            let runBypassBits = chunkBypassReaders[lane].readBits(count: runBypassLen)
            let zeroRun = UInt32(valueDetokenizeUnsigned(token: rtInfo.token, bypassBits: runBypassBits))
            
            let cfVal = ransDecoder.getCumulativeFreq(lane: lane)
            let vtInfo = valModels[ctx].findToken(cf: cfVal)
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
    return try data.withUnsafeBufferPointer { buf -> [MotionVector] in
        guard let base = buf.baseAddress else { return [] }
        var offset = 0
        let bufCount = buf.count
        
        let freqsDx = try EntropyDecoder.readCompressedFreqTable(base, at: &offset, count: bufCount)
        let modelDx = rANSModel(sigFreq: rANSScale / 2, tokenFreqs: freqsDx)
        
        let freqsDy = try EntropyDecoder.readCompressedFreqTable(base, at: &offset, count: bufCount)
        let modelDy = rANSModel(sigFreq: rANSScale / 2, tokenFreqs: freqsDy)
        
        let bpLen = Int(try readUInt32BEFromPtr(base, offset: &offset, count: bufCount))
        guard offset + bpLen <= bufCount else { throw DecodeError.insufficientData }
        var bypassReader = BypassReader(base: base.advanced(by: offset), count: bpLen)
        offset += bpLen
        
        guard offset < bufCount else { throw DecodeError.insufficientData }
        var dec = rANSDecoder(base: base.advanced(by: offset), count: bufCount - offset)
        
        var mvs = [MotionVector]()
        mvs.reserveCapacity(count)
        
        for _ in 0..<count {
            let txCf = dec.getCumulativeFreq()
            let tx = modelDx.findToken(cf: txCf)
            dec.advanceSymbol(cumFreq: tx.cumFreq, freq: tx.freq)
            
            let tyCf = dec.getCumulativeFreq()
            let ty = modelDy.findToken(cf: tyCf)
            dec.advanceSymbol(cumFreq: ty.cumFreq, freq: ty.freq)
            
            let dxBp = valueBypassLength(for: tx.token)
            let dxBv = bypassReader.readBits(count: dxBp)
            let dx = valueDetokenize(token: tx.token, bypassBits: dxBv)
            
            let dyBp = valueBypassLength(for: ty.token)
            let dyBv = bypassReader.readBits(count: dyBp)
            let dy = valueDetokenize(token: ty.token, bypassBits: dyBv)
            
            mvs.append(MotionVector(dx: dx, dy: dy))
        }
        
        return mvs
    }
}

// MARK: - Subband Entropy Codecs

struct SubbandEncoders<MLL: EntropyModelProvider, M: EntropyModelProvider> {
    var ll: EntropyEncoder<MLL>
    var hl: EntropyEncoder<M>
    var lh: EntropyEncoder<M>
    var hh: EntropyEncoder<M>
    
    init() {
        self.ll = EntropyEncoder<MLL>()
        self.hl = EntropyEncoder<M>()
        self.lh = EntropyEncoder<M>()
        self.hh = EntropyEncoder<M>()
    }
    
    @inline(__always)
    mutating func flush() {
        ll.flush()
        hl.flush()
        lh.flush()
        hh.flush()
    }
    
    @inline(__always)
    mutating func getData() -> [UInt8] {
        var out = [UInt8]()
        
        let dLL = ll.getData()
        let dHL = hl.getData()
        let dLH = lh.getData()
        let dHH = hh.getData()
        
        out.append(UInt8((dLL.count >> 24) & 0xFF))
        out.append(UInt8((dLL.count >> 16) & 0xFF))
        out.append(UInt8((dLL.count >> 8) & 0xFF))
        out.append(UInt8(dLL.count & 0xFF))
        out.append(contentsOf: dLL)
        
        out.append(UInt8((dHL.count >> 24) & 0xFF))
        out.append(UInt8((dHL.count >> 16) & 0xFF))
        out.append(UInt8((dHL.count >> 8) & 0xFF))
        out.append(UInt8(dHL.count & 0xFF))
        out.append(contentsOf: dHL)
        
        out.append(UInt8((dLH.count >> 24) & 0xFF))
        out.append(UInt8((dLH.count >> 16) & 0xFF))
        out.append(UInt8((dLH.count >> 8) & 0xFF))
        out.append(UInt8(dLH.count & 0xFF))
        out.append(contentsOf: dLH)
        
        out.append(UInt8((dHH.count >> 24) & 0xFF))
        out.append(UInt8((dHH.count >> 16) & 0xFF))
        out.append(UInt8((dHH.count >> 8) & 0xFF))
        out.append(UInt8(dHH.count & 0xFF))
        out.append(contentsOf: dHH)
        
        return out
    }
}

struct SubbandDecoders {
    var ll: EntropyDecoder
    var hl: EntropyDecoder
    var lh: EntropyDecoder
    var hh: EntropyDecoder
    private(set) var consumedBytes: Int
    
    init(base: UnsafePointer<UInt8>, count: Int, startOffset: Int) throws {
        var offset = startOffset
        
        let llSize = Int(try readUInt32BEFromPtr(base, offset: &offset, count: count))
        guard offset + llSize <= count else { throw DecodeError.insufficientData }
        self.ll = try EntropyDecoder(base: base, count: count, startOffset: offset)
        offset += llSize
        
        let hlSize = Int(try readUInt32BEFromPtr(base, offset: &offset, count: count))
        guard offset + hlSize <= count else { throw DecodeError.insufficientData }
        self.hl = try EntropyDecoder(base: base, count: count, startOffset: offset)
        offset += hlSize
        
        let lhSize = Int(try readUInt32BEFromPtr(base, offset: &offset, count: count))
        guard offset + lhSize <= count else { throw DecodeError.insufficientData }
        self.lh = try EntropyDecoder(base: base, count: count, startOffset: offset)
        offset += lhSize
        
        let hhSize = Int(try readUInt32BEFromPtr(base, offset: &offset, count: count))
        guard offset + hhSize <= count else { throw DecodeError.insufficientData }
        self.hh = try EntropyDecoder(base: base, count: count, startOffset: offset)
        offset += hhSize
        
        self.consumedBytes = offset
    }
}

