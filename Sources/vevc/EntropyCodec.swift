import Foundation

// MARK: - ContextModel / Legacy CABAC Support (To be removed)

public struct ContextModel {
    public var pStateIdx: UInt8 = 0
    public var valMPS: UInt8 = 0
    public init() {}
}

// MARK: - VevcEncoder

public struct VevcEncoder {
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

    public mutating func flush() {
        bypassWriter.flush()
    }

    public mutating func getData() -> [UInt8] {
        var out = [UInt8]()
        out.reserveCapacity(pairs.count * 4 + 128)
        
        bypassWriter.flush()
        let metaBypassData = bypassWriter.bytes
        appendUInt32BE(&out, UInt32(metaBypassData.count))
        out.append(contentsOf: metaBypassData)
        
        appendUInt32BE(&out, UInt32(coeffCount))
        
        guard !pairs.isEmpty || trailingZeros > 0 else { return out }
        
        let pairCount = pairs.count
        let hasTrailingZeros = trailingZeros > 0
        let nonZeroCount = pairCount
        
        // ---- Raw フォールバック (非ゼロ ≤ 32) ----
        if nonZeroCount <= 32 {
            out.append(0x80)
            var rawBypass = BypassWriter()
            for pair in pairs {
                // ゼロラン分
                for _ in 0..<pair.run {
                    rawBypass.writeBit(false)
                }
                // 非ゼロ値
                rawBypass.writeBit(true)
                rawBypass.writeBit(pair.val < 0)
                let absVal = UInt32(abs(Int(pair.val))) - 1
                let result = ValueTokenizer.tokenizeUnsigned(absVal)
                rawBypass.writeBits(UInt16(result.token), count: 4)
                rawBypass.writeBits(result.bypassBits, count: result.bypassLen)
            }
            // 末尾ゼロ
            for _ in 0..<trailingZeros {
                rawBypass.writeBit(false)
            }
            rawBypass.flush()
            let rawData = rawBypass.bytes
            appendUInt32BE(&out, UInt32(rawData.count))
            out.append(contentsOf: rawData)
            return out
        }
        
        // ---- rANS mode: 4チャンク分割 ----
        
        // pairs を4チャンクに分割
        let totalPairEntries = pairCount + (hasTrailingZeros ? 1 : 0)
        let chunkBase = pairCount / 4
        let chunkRemainder = pairCount % 4
        
        // チャンク境界: chunk[i] は pairs[chunkStarts[i]..<chunkStarts[i+1]] を担当
        var chunkStarts = [Int](repeating: 0, count: 5)
        for i in 0..<4 {
            chunkStarts[i + 1] = chunkStarts[i] + chunkBase + (i < chunkRemainder ? 1 : 0)
        }
        
        // 各チャンクのトークン化 + Bypass書き出し
        var chunkRunTokens = [[UInt8]](repeating: [], count: 4)
        var chunkValTokens = [[UInt8]](repeating: [], count: 4)
        var chunkBypassWriters = [BypassWriter](repeating: BypassWriter(), count: 4)
        var runTokenCounts = Array(repeating: 0, count: 16)
        var valTokenCounts = Array(repeating: 0, count: 16)
        
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
                chunkBypassWriters[lane].writeBit(valResult.sign)
                chunkBypassWriters[lane].writeBits(valResult.bypassBits, count: ValueTokenizer.bypassLength(for: valResult.token))
            }
        }
        
        // 末尾ゼロラン: lane3 に追加
        if hasTrailingZeros {
            let runResult = ValueTokenizer.tokenizeUnsigned(trailingZeros)
            runTokenCounts[Int(runResult.token)] += 1
            chunkRunTokens[3].append(runResult.token)
            chunkBypassWriters[3].writeBits(runResult.bypassBits, count: runResult.bypassLen)
        }
        
        for lane in 0..<4 {
            chunkBypassWriters[lane].flush()
        }
        
        // rANS モデル構築
        var runModel = rANSModel()
        runModel.normalize(sigCounts: [0, 1], tokenCounts: runTokenCounts)
        var valModel = rANSModel()
        valModel.normalize(sigCounts: [0, 1], tokenCounts: valTokenCounts)
        
        // ---- ヘッダ書き込み ----
        // flags: bit7=0 (rANS), bit0=hasTrailingZeros
        out.append(hasTrailingZeros ? 1 : 0)
        appendUInt32BE(&out, UInt32(totalPairEntries))
        
        // チャンクサイズ (4B × 4)
        for lane in 0..<4 {
            let chunkPairCount = chunkStarts[lane + 1] - chunkStarts[lane]
            let extraTrailing = (lane == 3 && hasTrailingZeros) ? 1 : 0
            appendUInt32BE(&out, UInt32(chunkPairCount + extraTrailing))
        }
        
        // 周波数テーブル (圧縮版)
        writeCompressedFreqTable(&out, freqs: runModel.tokenFreqs)
        writeCompressedFreqTable(&out, freqs: valModel.tokenFreqs)
        
        // 4系統 Bypass データ
        for lane in 0..<4 {
            let bpData = chunkBypassWriters[lane].bytes
            appendUInt32BE(&out, UInt32(bpData.count))
            out.append(contentsOf: bpData)
        }
        
        // ---- Interleaved 4-way rANS エンコード（逆順） ----
        var enc = Interleaved4rANSEncoder()
        
        // 各レーンを逆順でエンコード
        // 末尾ゼロラン (lane 3)
        if hasTrailingZeros {
            let trailingRunToken = chunkRunTokens[3].last!
            enc.encodeSymbol(lane: 3, cumFreq: runModel.tokenCumFreqs[Int(trailingRunToken)], freq: runModel.tokenFreqs[Int(trailingRunToken)])
        }
        
        // 各レーンのペアを逆順エンコード
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
        var bitmap: UInt16 = 0
        for i in 0..<16 {
            if freqs[i] > 1 {
                bitmap |= UInt16(1 << i)
            }
        }
        out.append(UInt8(bitmap >> 8))
        out.append(UInt8(bitmap & 0xFF))
        for i in 0..<16 {
            if (bitmap & UInt16(1 << i)) != 0 {
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

public struct VevcDecoder {
    var bypassReader: BypassReader
    public var pairs: [(run: UInt32, val: Int16)]
    private var pairIndex: Int = 0

    public init(data: [UInt8]) throws {
        var offset = 0
        
        guard data.count >= 4 else { throw DecodeError.insufficientData }
        let bypassLen = vevc.readUInt32BE(data, at: &offset)
        
        guard offset + Int(bypassLen) <= data.count else { throw DecodeError.insufficientData }
        let bypassData = Array(data[offset..<(offset + Int(bypassLen))])
        self.bypassReader = BypassReader(data: bypassData)
        offset += Int(bypassLen)
        
        guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
        let coeffCount = Int(vevc.readUInt32BE(data, at: &offset))
        
        guard coeffCount > 0 else {
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
                    let isNeg = rawReader.readBit()
                    let tokenBits = rawReader.readBits(count: 4)
                    let token = UInt8(tokenBits)
                    let bypassBits = rawReader.readBits(count: max(0, Int(token) - 1))
                    let absVal = Int16(ValueTokenizer.detokenizeUnsigned(token: token, bypassBits: bypassBits)) + 1
                    let val = isNeg ? -absVal : absVal
                    decodedPairs.append((run: zeroRun, val: val))
                    zeroRun = 0
                } else {
                    zeroRun += 1
                }
            }
            // 末尾ゼロはペアに含めない（coeffCountで暗黙管理）
            if zeroRun > 0 {
                decodedPairs.append((run: zeroRun, val: 0))
            }
            self.pairs = decodedPairs
            return
        }
        
        // ---- rANS mode デコード ----
        let hasTrailingZeros = (flags & 1) != 0
        
        guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
        let totalPairEntries = Int(vevc.readUInt32BE(data, at: &offset))
        
        // チャンクサイズ (4レーン分)
        guard offset + 16 <= data.count else { throw DecodeError.insufficientData }
        var chunkSizes = [Int](repeating: 0, count: 4)
        for lane in 0..<4 {
            chunkSizes[lane] = Int(vevc.readUInt32BE(data, at: &offset))
        }
        
        let runTokenFreqs = try VevcDecoder.readCompressedFreqTable(data, at: &offset)
        let runModel = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: runTokenFreqs)
        
        let valTokenFreqs = try VevcDecoder.readCompressedFreqTable(data, at: &offset)
        let valModel = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: valTokenFreqs)
        
        // 4系統 Bypass データ
        var chunkBypassReaders = [BypassReader]()
        for _ in 0..<4 {
            guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
            let bpLen = Int(vevc.readUInt32BE(data, at: &offset))
            guard offset + bpLen <= data.count else { throw DecodeError.insufficientData }
            let bpData = Array(data[offset..<(offset + bpLen)])
            chunkBypassReaders.append(BypassReader(data: bpData))
            offset += bpLen
        }
        
        // rANS ストリーム
        let ransData = Array(data[offset...])
        var ransDecoder = Interleaved4rANSDecoder(bitstream: ransData)
        
        // 各レーンからペアをデコード
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
                
                let runBypassBits = chunkBypassReaders[lane].readBits(count: max(0, Int(rtInfo.token) - 1))
                let zeroRun = UInt32(ValueTokenizer.detokenizeUnsigned(token: rtInfo.token, bypassBits: runBypassBits))
                
                let cfVal = ransDecoder.getCumulativeFreq(lane: lane)
                let vtInfo = valModel.findToken(cf: cfVal)
                ransDecoder.advanceSymbol(lane: lane, cumFreq: vtInfo.cumFreq, freq: vtInfo.freq)
                
                let sign = chunkBypassReaders[lane].readBit()
                let valBypassBits = chunkBypassReaders[lane].readBits(count: ValueTokenizer.bypassLength(for: vtInfo.token))
                let val = ValueTokenizer.detokenize(isSignificant: true, sign: sign, token: vtInfo.token, bypassBits: valBypassBits)
                
                chunkPairs[lane].append((run: zeroRun, val: val))
            }
            
            // 末尾ゼロラン (lane 3)
            if hasTZ {
                let cfRun = ransDecoder.getCumulativeFreq(lane: lane)
                let rtInfo = runModel.findToken(cf: cfRun)
                ransDecoder.advanceSymbol(lane: lane, cumFreq: rtInfo.cumFreq, freq: rtInfo.freq)
                
                let runBypassBits = chunkBypassReaders[lane].readBits(count: max(0, Int(rtInfo.token) - 1))
                let zeroRun = UInt32(ValueTokenizer.detokenizeUnsigned(token: rtInfo.token, bypassBits: runBypassBits))
                chunkPairs[lane].append((run: zeroRun, val: 0))
            }
        }
        
        // 4チャンクを結合してペア配列を生成
        var allPairs = [(run: UInt32, val: Int16)]()
        allPairs.reserveCapacity(totalPairEntries)
        for lane in 0..<4 {
            allPairs.append(contentsOf: chunkPairs[lane])
        }
        
        self.pairs = allPairs
    }
    
    private static func readCompressedFreqTable(_ data: [UInt8], at offset: inout Int) throws -> [UInt32] {
        guard offset + 2 <= data.count else { throw DecodeError.insufficientData }
        let bitmap = (UInt16(data[offset]) << 8) | UInt16(data[offset + 1])
        offset += 2
        
        var freqs = [UInt32](repeating: 1, count: 16)
        for i in 0..<16 {
            if (bitmap & UInt16(1 << i)) != 0 {
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

    // 後方互換: readCoeff / tryReadCoeff は廃止予定だが
    // blockDecode 内の (run, val) = decodeCoeffRun() パターンが
    // readPair() に直接マッピングされるため不要
}

@inline(__always)
private func readUInt32BE(_ data: [UInt8], at offset: inout Int) -> UInt32 {
    let val = (UInt32(data[offset]) << 24) | (UInt32(data[offset+1]) << 16) | (UInt32(data[offset+2]) << 8) | UInt32(data[offset+3])
    offset += 4
    return val
}
