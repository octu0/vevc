// MARK: - Context Constants

/// Number of rANS contexts in the unified entropy stream.
/// Contexts 0-3: AC coefficients (HL/LH/HH subbands, context selected by getContext())
/// Context 4:    DPCM coefficients (LL subband)
let kEntropyContextCount = 6
let kDPCMContext: UInt8 = 4

// MARK: - EntropyModelSelection

/// Result of model selection: contains the chosen models and flags for the encoder/decoder.
struct EntropyModelSelection {
    let runModels: [rANSModel]  // always kEntropyContextCount elements (may be identical for merged)
    let valModels: [rANSModel]  // always kEntropyContextCount elements
    let isStatic: Bool
    let isMerged: Bool
}

// MARK: - EntropyModelProvider

protocol EntropyModelProvider {
    static func selectModel(
        runTokenCounts: inout [[Int]], valTokenCounts: inout [[Int]]
    ) -> EntropyModelSelection
}

// MARK: - Bit Cost Estimation

/// Fast integer log2 approximation in Q8 fixed-point.
/// Returns log2(x) * 256 for x >= 1, 0 for x <= 0.
@inline(__always)
func log2Q8(_ x: Int) -> Int {
    guard 1 <= x else { return 0 }
    let bits = Int.bitWidth - 1 - x.leadingZeroBitCount  // floor(log2(x))
    let base = 1 << bits
    // Linear interpolation for fractional part: frac = (x - 2^bits) / 2^bits
    let fracQ8 = ((x - base) << 8) / base
    return (bits << 8) + fracQ8
}

/// Estimate the rANS bit cost for encoding tokenCounts with the given model.
/// Returns cost in Q8 fixed-point (bits * 256).
/// cost = Σ count(token_i) * log2(rANSScale / freq(token_i))
///      = Σ count(token_i) * (log2(rANSScale) - log2(freq(token_i)))
@inline(__always)
func estimateBitCostQ8(tokenCounts: [Int], model: rANSModel) -> Int {
    let scaleLog2Q8 = log2Q8(Int(rANSScale))
    var totalCostQ8: Int = 0
    for i in 0..<64 {
        let count = tokenCounts[i]
        if count == 0 { continue }
        let freq = Int(model.tokenFreqs[i])
        let bitsPerSymbolQ8 = scaleLog2Q8 - log2Q8(freq)
        totalCostQ8 += count * bitsPerSymbolQ8
    }
    return totalCostQ8
}

/// Estimate the byte cost of writing a compressed frequency table header.
/// Format: bitmap(8B) + VLQ (1B or 2B) per non-trivial frequency.
@inline(__always)
func headerCostBits(model: rANSModel) -> Int {
    var sizeBits = 64
    for i in 0..<64 {
        let freq = model.tokenFreqs[i]
        if 1 < freq {
            if freq < 128 {
                sizeBits += 8
            } else {
                sizeBits += 16
            }
        }
    }
    return sizeBits
}

// MARK: - Model Providers

struct StaticEntropyModel: EntropyModelProvider {
    
    @inline(__always)
    static func selectModel(
        runTokenCounts: inout [[Int]], valTokenCounts: inout [[Int]]
    ) -> EntropyModelSelection {
        return EntropyModelSelection(
            runModels: [StaticRANSModels.shared.runModel0, StaticRANSModels.shared.runModel1, StaticRANSModels.shared.runModel2, StaticRANSModels.shared.runModel3, StaticRANSModels.shared.dpcmRunModel, StaticRANSModels.shared.lscpRunModel],
            valModels: [StaticRANSModels.shared.valModel0, StaticRANSModels.shared.valModel1, StaticRANSModels.shared.valModel2, StaticRANSModels.shared.valModel3, StaticRANSModels.shared.dpcmValModel, StaticRANSModels.shared.dpcmValModel],
            isStatic: true,
            isMerged: false
        )
    }
}

struct AdaptiveEntropyModel: EntropyModelProvider {
    
    @inline(__always)
    static func selectModel(
        runTokenCounts: inout [[Int]], valTokenCounts: inout [[Int]]
    ) -> EntropyModelSelection {
        let totalPairs = runTokenCounts.reduce(0) { $0 + $1.reduce(0, +) }
        
        let staticRunModels = [
            StaticRANSModels.shared.runModel0, StaticRANSModels.shared.runModel1,
            StaticRANSModels.shared.runModel2, StaticRANSModels.shared.runModel3,
            StaticRANSModels.shared.dpcmRunModel, StaticRANSModels.shared.lscpRunModel,
        ]
        let staticValModels = [
            StaticRANSModels.shared.valModel0, StaticRANSModels.shared.valModel1,
            StaticRANSModels.shared.valModel2, StaticRANSModels.shared.valModel3,
            StaticRANSModels.shared.dpcmValModel, StaticRANSModels.shared.dpcmValModel,
        ]
        
        // Too few pairs: static tables are always the best choice (no header overhead)
        if totalPairs == 0 {
            return EntropyModelSelection(
                runModels: staticRunModels, valModels: staticValModels,
                isStatic: true, isMerged: false
            )
        }
        
        // --- Option 1: Static 4-context (no header cost) ---
        var staticCostQ8: Int = 0
        for c in 0..<4 {
            staticCostQ8 += estimateBitCostQ8(tokenCounts: runTokenCounts[c], model: staticRunModels[c])
            staticCostQ8 += estimateBitCostQ8(tokenCounts: valTokenCounts[c], model: staticValModels[c])
        }
        
        // --- Option 2: Dynamic 4-context (8 tables header cost) ---
        var dynRunModels = [rANSModel]()
        var dynValModels = [rANSModel]()
        var dynamic4CostQ8: Int = 0
        var dynamic4HeaderBits: Int = 0
        dynRunModels.reserveCapacity(kEntropyContextCount)
        dynValModels.reserveCapacity(kEntropyContextCount)
        for c in 0..<kEntropyContextCount {
            var rm = rANSModel()
            var vm = rANSModel()
            rm.normalize(tokenCounts: runTokenCounts[c])
            vm.normalize(tokenCounts: valTokenCounts[c])
            dynRunModels.append(rm)
            dynValModels.append(vm)
            dynamic4CostQ8 += estimateBitCostQ8(tokenCounts: runTokenCounts[c], model: rm)
            dynamic4CostQ8 += estimateBitCostQ8(tokenCounts: valTokenCounts[c], model: vm)
            dynamic4HeaderBits += headerCostBits(model: rm)
            dynamic4HeaderBits += headerCostBits(model: vm)
        }
        // Add header overhead in Q8
        dynamic4CostQ8 += dynamic4HeaderBits << 8
        
        // --- Option 3: Dynamic merged (2 tables header cost) ---
        var mergedRunCounts = [Int](repeating: 0, count: 64)
        var mergedValCounts = [Int](repeating: 0, count: 64)
        for c in 0..<4 {
            for t in 0..<64 {
                mergedRunCounts[t] += runTokenCounts[c][t]
                mergedValCounts[t] += valTokenCounts[c][t]
            }
        }
        var mergedRunModel = rANSModel()
        var mergedValModel = rANSModel()
        mergedRunModel.normalize(tokenCounts: mergedRunCounts)
        mergedValModel.normalize(tokenCounts: mergedValCounts)
        
        var mergedCostQ8: Int = 0
        // Merged model: all contexts use the same model
        for c in 0..<4 {
            mergedCostQ8 += estimateBitCostQ8(tokenCounts: runTokenCounts[c], model: mergedRunModel)
            mergedCostQ8 += estimateBitCostQ8(tokenCounts: valTokenCounts[c], model: mergedValModel)
        }
        let mergedHeaderBits = headerCostBits(model: mergedRunModel) + headerCostBits(model: mergedValModel)
        mergedCostQ8 += mergedHeaderBits << 8
        
        // --- Choose minimum cost ---
        let minCost = min(staticCostQ8, min(dynamic4CostQ8, mergedCostQ8))
        
        if minCost == staticCostQ8 {
            return EntropyModelSelection(
                runModels: staticRunModels, valModels: staticValModels,
                isStatic: true, isMerged: false
            )
        }
        if minCost == mergedCostQ8 {
            let mergedRun6 = [rANSModel](repeating: mergedRunModel, count: kEntropyContextCount)
            let mergedVal6 = [rANSModel](repeating: mergedValModel, count: kEntropyContextCount)
            return EntropyModelSelection(
                runModels: mergedRun6, valModels: mergedVal6,
                isStatic: false, isMerged: true
            )
        }
        // dynamic 4-context fallback padded to 6
        return EntropyModelSelection(
            runModels: dynRunModels, valModels: dynValModels,
            isStatic: false, isMerged: false
        )
    }
}

struct StaticDPCMEntropyModel: EntropyModelProvider {
    
    @inline(__always)
    static func selectModel(
        runTokenCounts: inout [[Int]], valTokenCounts: inout [[Int]]
    ) -> EntropyModelSelection {
        let dpcmRun = StaticRANSModels.shared.dpcmRunModel
        let dpcmVal = StaticRANSModels.shared.dpcmValModel
        return EntropyModelSelection(
            runModels: [dpcmRun, dpcmRun, dpcmRun, dpcmRun, dpcmRun, dpcmRun],
            valModels: [dpcmVal, dpcmVal, dpcmVal, dpcmVal, dpcmVal, dpcmVal],
            isStatic: true,
            isMerged: false
        )
    }
}



// MARK: - Unified Model (5-context: AC 0-3 + DPCM 4)

/// Unified model selection for the single-stream encoder.
/// Contexts 0-3 use AC static/dynamic tables, context 4 uses DPCM static/dynamic tables.
/// This replaces separate SubbandEncoders with per-model selection.
@inline(__always)
func unifiedSelectModel(
    runTokenCounts: inout [[Int]], valTokenCounts: inout [[Int]]
) -> EntropyModelSelection {
    // --- AC contexts (0-3) ---
    let staticACRunModels = [
        StaticRANSModels.shared.runModel0, StaticRANSModels.shared.runModel1,
        StaticRANSModels.shared.runModel2, StaticRANSModels.shared.runModel3,
    ]
    let staticACValModels = [
        StaticRANSModels.shared.valModel0, StaticRANSModels.shared.valModel1,
        StaticRANSModels.shared.valModel2, StaticRANSModels.shared.valModel3,
    ]
    let staticDPCMRun = StaticRANSModels.shared.dpcmRunModel
    let staticDPCMVal = StaticRANSModels.shared.dpcmValModel
    let staticLSCPRun = StaticRANSModels.shared.lscpRunModel

    let totalPairs = runTokenCounts.reduce(0) { $0 + $1.reduce(0, +) }
    if totalPairs == 0 {
        return EntropyModelSelection(
            runModels: staticACRunModels + [staticDPCMRun, staticLSCPRun],
            valModels: staticACValModels + [staticDPCMVal, staticDPCMVal],
            isStatic: true, isMerged: false
        )
    }

    // --- Option 1: Static 6-context (no header cost) ---
    var staticCostQ8: Int = 0
    for c in 0..<4 {
        staticCostQ8 += estimateBitCostQ8(tokenCounts: runTokenCounts[c], model: staticACRunModels[c])
        staticCostQ8 += estimateBitCostQ8(tokenCounts: valTokenCounts[c], model: staticACValModels[c])
    }
    staticCostQ8 += estimateBitCostQ8(tokenCounts: runTokenCounts[4], model: staticDPCMRun)
    staticCostQ8 += estimateBitCostQ8(tokenCounts: valTokenCounts[4], model: staticDPCMVal)
    staticCostQ8 += estimateBitCostQ8(tokenCounts: runTokenCounts[5], model: staticLSCPRun)
    staticCostQ8 += estimateBitCostQ8(tokenCounts: valTokenCounts[5], model: staticDPCMVal)

    // --- Option 2: Dynamic 5-context (10 tables header cost) ---
    var dynRunModels = [rANSModel]()
    var dynValModels = [rANSModel]()
    var dynamic5CostQ8: Int = 0
    var dynamic5HeaderBits: Int = 0
    dynRunModels.reserveCapacity(kEntropyContextCount)
    dynValModels.reserveCapacity(kEntropyContextCount)
    for c in 0..<kEntropyContextCount {
        var rm = rANSModel()
        var vm = rANSModel()
        rm.normalize(tokenCounts: runTokenCounts[c])
        vm.normalize(tokenCounts: valTokenCounts[c])
        dynRunModels.append(rm)
        dynValModels.append(vm)
        dynamic5CostQ8 += estimateBitCostQ8(tokenCounts: runTokenCounts[c], model: rm)
        dynamic5CostQ8 += estimateBitCostQ8(tokenCounts: valTokenCounts[c], model: vm)
        dynamic5HeaderBits += headerCostBits(model: rm)
        dynamic5HeaderBits += headerCostBits(model: vm)
    }
    dynamic5CostQ8 += dynamic5HeaderBits << 8

    // --- Option 3: Dynamic merged (2 tables header cost) ---
    var mergedRunCounts = [Int](repeating: 0, count: 64)
    var mergedValCounts = [Int](repeating: 0, count: 64)
    for c in 0..<kEntropyContextCount {
        for t in 0..<64 {
            mergedRunCounts[t] += runTokenCounts[c][t]
            mergedValCounts[t] += valTokenCounts[c][t]
        }
    }
    var mergedRunModel = rANSModel()
    var mergedValModel = rANSModel()
    mergedRunModel.normalize(tokenCounts: mergedRunCounts)
    mergedValModel.normalize(tokenCounts: mergedValCounts)

    var mergedCostQ8: Int = 0
    for c in 0..<kEntropyContextCount {
        mergedCostQ8 += estimateBitCostQ8(tokenCounts: runTokenCounts[c], model: mergedRunModel)
        mergedCostQ8 += estimateBitCostQ8(tokenCounts: valTokenCounts[c], model: mergedValModel)
    }
    let mergedHeaderBits = headerCostBits(model: mergedRunModel) + headerCostBits(model: mergedValModel)
    mergedCostQ8 += mergedHeaderBits << 8

    // --- Choose minimum cost ---
    let minCost = min(staticCostQ8, min(dynamic5CostQ8, mergedCostQ8))

    if minCost == staticCostQ8 {
        return EntropyModelSelection(
            runModels: staticACRunModels + [staticDPCMRun, staticLSCPRun],
            valModels: staticACValModels + [staticDPCMVal, staticDPCMVal],
            isStatic: true, isMerged: false
        )
    }
    if minCost == mergedCostQ8 {
        let merged5Run = [rANSModel](repeating: mergedRunModel, count: kEntropyContextCount)
        let merged5Val = [rANSModel](repeating: mergedValModel, count: kEntropyContextCount)
        return EntropyModelSelection(
            runModels: merged5Run, valModels: merged5Val,
            isStatic: false, isMerged: true
        )
    }
    // dynamic 5-context
    return EntropyModelSelection(
        runModels: dynRunModels, valModels: dynValModels,
        isStatic: false, isMerged: false
    )
}

// MARK: - EntropyEncoder

/// Model selection function type: given mutable token count arrays, returns the best model selection.
typealias ModelSelectorFn = (inout [[Int]], inout [[Int]]) -> EntropyModelSelection

struct EntropyEncoder {
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
    mutating func getData(selectModel: ModelSelectorFn) -> [UInt8] {
        var out = [UInt8]()
        let pairCount = pairRuns.count
        out.reserveCapacity(pairCount * 4 + 128)
        
        bypassWriter.flush()
        let metaBypassData = bypassWriter.bytes
        writeVLQSize(&out, metaBypassData.count)
        out.append(contentsOf: metaBypassData)
        
        writeVLQSize(&out, coeffCount)
        
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
            writeVLQSize(&out, rawData.count)
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
        var runTokenCounts = [[Int]](repeating: Array(repeating: 0, count: 64), count: kEntropyContextCount)
        var valTokenCounts = [[Int]](repeating: Array(repeating: 0, count: 64), count: kEntropyContextCount)
        
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
        for c in 0..<kEntropyContextCount {
            for i in 0..<64 {
                if 65535 < runTokenCounts[c][i] { runTokenCounts[c][i] = 65535 }
                if 65535 < valTokenCounts[c][i] { valTokenCounts[c][i] = 65535 }
            }
        }
        
        let selection = selectModel(
            &runTokenCounts, &valTokenCounts
        )
        let runModels = selection.runModels
        let valModels = selection.valModels
        
        // Flags byte: bit6=isStatic, bit4=isMerged, bit0=hasTrailingZeros
        let flagTrailingZeros: UInt8 = 0x01
        let flagMerged: UInt8        = 0x10
        let flagStatic: UInt8        = 0x40
        
        let trailBit: UInt8  = if hasTrailingZeros { flagTrailingZeros } else { 0 }
        let staticBit: UInt8 = if selection.isStatic { flagStatic } else { 0 }
        let mergedBit: UInt8 = if selection.isMerged { flagMerged } else { 0 }
        
        out.append(staticBit | mergedBit | trailBit)
        writeVLQSize(&out, totalPairEntries)
        
        // Write dynamic frequency table headers when not static
        if selection.isStatic != true {
            if selection.isMerged {
                // Merged: 2 tables (run + val for single merged context)
                writeCompressedFreqTable(&out, freqs: runModels[0].tokenFreqs)
                writeCompressedFreqTable(&out, freqs: valModels[0].tokenFreqs)
            } else {
                // N-context: 2*N tables (run + val for each context)
                for i in 0..<kEntropyContextCount {
                    writeCompressedFreqTable(&out, freqs: runModels[i].tokenFreqs)
                    writeCompressedFreqTable(&out, freqs: valModels[i].tokenFreqs)
                }
            }
        }
        
        // 4-way bypass data
        for lane in 0..<4 {
            let bpData = chunkBypassWriters[lane].bytes
            writeVLQSize(&out, bpData.count)
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
    

}

@inline(__always)
internal func writeVLQSize(_ out: inout [UInt8], _ val: Int) {
    var v = val
    if v == 0 {
        out.append(0)
        return
    }
    var temp = [UInt8]()
    temp.reserveCapacity(10)
    while v != 0 {
        temp.append(UInt8(v & 0x7F))
        v >>= 7
    }
    var i = temp.count - 1
    while i != 0 {
        out.append(temp[i] | 0x80)
        i -= 1
    }
    out.append(temp[0])
}

@inline(__always)
internal func readVLQSize(_ base: UnsafePointer<UInt8>, at offset: inout Int, count: Int) throws -> Int {
    var val = 0
    var bytesRead = 0
    while true {
        guard offset < count else { throw DecodeError.insufficientData }
        let b = base[offset]
        offset += 1
        
        val = (val << 7) | Int(b & 0x7F)
        bytesRead += 1
        if (b & 0x80) == 0 {
            break
        }
        if 5 < bytesRead {
            throw DecodeError.invalidBlockData
        }
    }
    return val
}

@inline(__always)
internal func readVLQSizeFromBytes(_ r: [UInt8], offset: inout Int) throws -> Int {
    var val = 0
    var bytesRead = 0
    while true {
        guard offset < r.count else { throw DecodeError.insufficientData }
        let b = r[offset]
        offset += 1
        
        val = (val << 7) | Int(b & 0x7F)
        bytesRead += 1
        if (b & 0x80) == 0 {
            break
        }
        if 5 < bytesRead {
            throw DecodeError.invalidBlockData
        }
    }
    return val
}

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
            let val = freqs[i]
            if val < 128 {
                out.append(UInt8(val))
            } else {
                out.append(UInt8((val >> 8) | 0x80))
                out.append(UInt8(val & 0xFF))
            }
        }
    }
}

// MARK: - ModelTrainer

public final class ModelTrainer: @unchecked Sendable {
    public static let shared = ModelTrainer()
    
    public var isTrainingMode = false
    
    // 5 contexts (0-3: AC, 4: DPCM), 64 tokens each
    public var runFreqs: [[UInt32]]
    public var valFreqs: [[UInt32]]
    
    private init() {
        self.runFreqs = Array(repeating: Array(repeating: 0, count: 64), count: kEntropyContextCount)
        self.valFreqs = Array(repeating: Array(repeating: 0, count: 64), count: kEntropyContextCount)
    }
    
    @inline(__always)
    public func reset() {
        self.runFreqs = Array(repeating: Array(repeating: 0, count: 64), count: kEntropyContextCount)
        self.valFreqs = Array(repeating: Array(repeating: 0, count: 64), count: kEntropyContextCount)
    }
    
    @inline(__always)
    public func record(run: UInt32, val: Int16, context: UInt8) {
        let ctx = Int(context)
        guard 0 <= ctx && ctx < kEntropyContextCount else { return }
        
        let runToken = min(run, 63)
        let valToken = valueTokenize(val).token
        
        runFreqs[ctx][Int(runToken)] &+= 1
        
        if valToken < 64 {
            valFreqs[ctx][Int(valToken)] &+= 1
        }
    }
}

// MARK: - EntropyDecoder

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
        
        let bypassLen = try readVLQSize(base, at: &offset, count: count)
        guard offset + bypassLen <= count else { throw DecodeError.insufficientData }
        
        self.bypassReader = BypassReader(base: base.advanced(by: offset), count: bypassLen)
        offset += bypassLen
        
        let coeffCount = try readVLQSize(base, at: &offset, count: count)
        
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
            let rawDataLen = try readVLQSize(base, at: &offset, count: count)
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
        
        let totalPairEntries = try readVLQSize(base, at: &offset, count: count)
        self.totalPairEntries = totalPairEntries
        
        // chunk size (4 lanes) - dynamically reconstructed from totalPairEntries
        let totalPairs = hasTrailingZeros ? (totalPairEntries - 1) : totalPairEntries
        let chunkBase = totalPairs / 4
        let chunkRemainder = totalPairs % 4
        var starts = [Int](repeating: 0, count: 5)
        for i in 0..<4 {
            let size = (i < chunkRemainder) ? (chunkBase + 1) : chunkBase
            starts[i+1] = starts[i] + size
        }
        if hasTrailingZeros {
            starts[4] += 1
        }
        self.chunkStarts = starts
        
        let isMergedContext = (flags & 0x10) != 0
        
        if isStaticTable {
            // Static 6-context: AC models for ctx 0-3, DPCM model for ctx 4, LSCP for ctx 5
            self.runModels = [
                StaticRANSModels.shared.runModel0, StaticRANSModels.shared.runModel1,
                StaticRANSModels.shared.runModel2, StaticRANSModels.shared.runModel3,
                StaticRANSModels.shared.dpcmRunModel, StaticRANSModels.shared.lscpRunModel,
            ]
            self.valModels = [
                StaticRANSModels.shared.valModel0, StaticRANSModels.shared.valModel1,
                StaticRANSModels.shared.valModel2, StaticRANSModels.shared.valModel3,
                StaticRANSModels.shared.dpcmValModel, StaticRANSModels.shared.dpcmValModel,
            ]
        } else if isMergedContext {
            // Dynamic merged: read 2 tables (run + val), replicate to all 5 contexts
            let runFreqs = try EntropyDecoder.readCompressedFreqTable(base, at: &offset, count: count)
            let runM = rANSModel(tokenFreqs: runFreqs)
            let valFreqs = try EntropyDecoder.readCompressedFreqTable(base, at: &offset, count: count)
            let valM = rANSModel(tokenFreqs: valFreqs)
            self.runModels = [rANSModel](repeating: runM, count: kEntropyContextCount)
            self.valModels = [rANSModel](repeating: valM, count: kEntropyContextCount)
        } else {
            // Dynamic N-context: read 2*N tables
            var rModels = [rANSModel]()
            var vModels = [rANSModel]()
            for _ in 0..<kEntropyContextCount {
                let runFreqs = try EntropyDecoder.readCompressedFreqTable(base, at: &offset, count: count)
                rModels.append(rANSModel(tokenFreqs: runFreqs))
                
                let valFreqs = try EntropyDecoder.readCompressedFreqTable(base, at: &offset, count: count)
                vModels.append(rANSModel(tokenFreqs: valFreqs))
            }
            self.runModels = rModels
            self.valModels = vModels
        }
        
        // 4-way bypass data
        var chunkBypassReaders = [BypassReader]()
        for _ in 0..<4 {
            let bpLen = try readVLQSize(base, at: &offset, count: count)
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
                guard offset < count else { throw DecodeError.insufficientData }
                let b0 = base[offset]
                offset += 1
                if (b0 & 0x80) == 0 {
                    freqs[i] = UInt32(b0)
                } else {
                    guard offset < count else { throw DecodeError.insufficientData }
                    let b1 = base[offset]
                    offset += 1
                    freqs[i] = (UInt32(b0 & 0x7F) << 8) | UInt32(b1)
                }
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
        while currentLane < 3 && chunkStarts[currentLane + 1] <= pairIndex {
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
func encodeMVs(mvs: MotionVectors) -> [UInt8] {
    var tokensDx = [UInt8]()
    var tokensDy = [UInt8]()
    tokensDx.reserveCapacity(mvs.count)
    tokensDy.reserveCapacity(mvs.count)
    
    var bypass = BypassWriter()
    var freqsDx = [UInt32](repeating: 0, count: 64)
    var freqsDy = [UInt32](repeating: 0, count: 64)
    
    for i in 0..<mvs.count {
        let tx = valueTokenize(mvs.dx[i])
        tokensDx.append(tx.token)
        freqsDx[Int(tx.token)] += 1
        bypass.writeBits(tx.bypassBits, count: tx.bypassLen)

        let ty = valueTokenize(mvs.dy[i])
        tokensDy.append(ty.token)
        freqsDy[Int(ty.token)] += 1
        bypass.writeBits(ty.bypassBits, count: ty.bypassLen)
    }
    bypass.flush()
    
    var modelDx = rANSModel()
    modelDx.normalize(tokenCounts: freqsDx.map(Int.init))
    
    var modelDy = rANSModel()
    modelDy.normalize(tokenCounts: freqsDy.map(Int.init))
    
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
    writeVLQSize(&out, bpData.count)
    out.append(contentsOf: bpData)
    
    out.append(contentsOf: enc.getBitstream())
    return out
}

@inline(__always)
func decodeMVs(data: [UInt8], count: Int) throws -> MotionVectors {
    return try data.withUnsafeBufferPointer { buf -> MotionVectors in
        guard let base = buf.baseAddress else { return MotionVectors.empty }
        var offset = 0
        let bufCount = buf.count
        
        let freqsDx = try EntropyDecoder.readCompressedFreqTable(base, at: &offset, count: bufCount)
        let modelDx = rANSModel(tokenFreqs: freqsDx)
        
        let freqsDy = try EntropyDecoder.readCompressedFreqTable(base, at: &offset, count: bufCount)
        let modelDy = rANSModel(tokenFreqs: freqsDy)
        
        let bpLen = try readVLQSize(base, at: &offset, count: bufCount)
        guard offset + bpLen <= bufCount else { throw DecodeError.insufficientData }
        var bypassReader = BypassReader(base: base.advanced(by: offset), count: bpLen)
        offset += bpLen
        
        guard offset < bufCount else { throw DecodeError.insufficientData }
        var dec = rANSDecoder(base: base.advanced(by: offset), count: bufCount - offset)
        
        var mvsDx = [Int16]()
        var mvsDy = [Int16]()
        mvsDx.reserveCapacity(count)
        mvsDy.reserveCapacity(count)
        
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
            
            mvsDx.append(dx)
            mvsDy.append(dy)
        }
        
        return MotionVectors(dx: mvsDx, dy: mvsDy)
    }
}
