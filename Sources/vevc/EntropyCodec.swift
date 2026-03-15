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
    public var coeffs: [Int16]
    private var rANSCompressor: rANSEncoder

    public init() {
        self.bypassWriter = BypassWriter()
        self.coeffs = []
        self.rANSCompressor = rANSEncoder()
    }

    @inline(__always)
    public mutating func encodeBypass(binVal: UInt8) {
        bypassWriter.writeBit(binVal != 0)
    }

    @inline(__always)
    public mutating func addCoeff(_ val: Int16) {
        coeffs.append(val)
    }

    public mutating func flush() {
        bypassWriter.flush()
    }

    public mutating func getData() -> [UInt8] {
        var out = [UInt8]()
        out.reserveCapacity(coeffs.count + 128)
        
        // ---- Phase 1: メタデータ Bypass の書き出し ----
        bypassWriter.flush() // 冪等: 呼び出し元で既にflush済みでも安全
        let metaBypassData = bypassWriter.bytes
        appendUInt32BE(&out, UInt32(metaBypassData.count))
        out.append(contentsOf: metaBypassData)
        
        // ---- Phase 2: 係数の Zero-Run + Value ペア列への変換 ----
        let coeffCount = coeffs.count
        appendUInt32BE(&out, UInt32(coeffCount))
        
        guard coeffCount > 0 else { return out }
        
        // (zeroRun, value) ペアを構築
        // 同時に頻度カウントと bypass ビットの書き込みを行う
        var coeffBypassWriter = BypassWriter()
        
        var runTokenCounts = Array(repeating: 0, count: 16)
        var valTokenCounts = Array(repeating: 0, count: 16)
        
        // ペアリスト用の事前確保配列
        // 最悪ケース: 全て非ゼロ → coeffCount 個のペア
        var pairRunTokens = [UInt8]()
        pairRunTokens.reserveCapacity(coeffCount)
        var pairValTokens = [UInt8]()
        pairValTokens.reserveCapacity(coeffCount)
        
        var zeroRun: UInt32 = 0
        for i in 0..<coeffCount {
            let c = coeffs[i]
            if c == 0 {
                zeroRun += 1
            } else {
                // Zero-Run Token
                let runResult = ValueTokenizer.tokenizeUnsigned(zeroRun)
                runTokenCounts[Int(runResult.token)] += 1
                pairRunTokens.append(runResult.token)
                coeffBypassWriter.writeBits(runResult.bypassBits, count: runResult.bypassLen)
                
                // Value Token
                let valResult = ValueTokenizer.tokenize(c)
                valTokenCounts[Int(valResult.token)] += 1
                pairValTokens.append(valResult.token)
                coeffBypassWriter.writeBit(valResult.sign)
                coeffBypassWriter.writeBits(valResult.bypassBits, count: ValueTokenizer.bypassLength(for: valResult.token))
                
                zeroRun = 0
            }
        }
        // 末尾のゼロランが残る場合:  終端マーカーとして特殊ペアを追加
        // → 末尾ゼロは LSCP によりブロックエンコード側で切り捨て済みのため、
        //   通常は zeroRun == 0 だが、安全のためセンチネルを追加
        let hasTrailingZeros = zeroRun > 0
        if hasTrailingZeros {
            // 末尾ゼロランとして記録。valToken=0としてsignificant値=1のダミーを配置
            // → デコード時にtrailingZerosフラグで判断
            let runResult = ValueTokenizer.tokenizeUnsigned(zeroRun)
            runTokenCounts[Int(runResult.token)] += 1
            pairRunTokens.append(runResult.token)
            coeffBypassWriter.writeBits(runResult.bypassBits, count: runResult.bypassLen)
        }
        
        coeffBypassWriter.flush()
        
        let pairCount = pairValTokens.count
        let totalPairCount = hasTrailingZeros ? pairCount + 1 : pairCount
        
        // ---- Phase 3: rANS モデル構築 ----
        var runModel = rANSModel()
        runModel.normalize(sigCounts: [0, 1], tokenCounts: runTokenCounts)
        
        var valModel = rANSModel()
        valModel.normalize(sigCounts: [0, 1], tokenCounts: valTokenCounts)
        
        // ---- Phase 4: ヘッダ書き込み ----
        // フラグ: bit0 = hasTrailingZeros
        out.append(hasTrailingZeros ? 1 : 0)
        
        appendUInt32BE(&out, UInt32(totalPairCount))
        
        // runModel の周波数テーブル (32 bytes)
        for f in runModel.tokenFreqs {
            out.append(UInt8(truncatingIfNeeded: f >> 8))
            out.append(UInt8(truncatingIfNeeded: f & 0xFF))
        }
        
        // valModel の周波数テーブル (32 bytes)
        for f in valModel.tokenFreqs {
            out.append(UInt8(truncatingIfNeeded: f >> 8))
            out.append(UInt8(truncatingIfNeeded: f & 0xFF))
        }
        
        // 係数 Bypass データ
        let coeffBypassData = coeffBypassWriter.bytes
        appendUInt32BE(&out, UInt32(coeffBypassData.count))
        out.append(contentsOf: coeffBypassData)
        
        // ---- Phase 5: Interleaved 4-way rANS エンコード（逆順） ----
        var enc = Interleaved4rANSEncoder()
        
        // シンボル総数: 通常ペアは runToken + valToken = 2シンボル/ペア
        // 末尾ゼロランがある場合は +1 シンボル
        let totalSymbols = pairCount * 2 + (hasTrailingZeros ? 1 : 0)
        
        // 末尾ゼロラン（ペアの最後のrunTokenのみ、対応するvalTokenなし）
        if hasTrailingZeros {
            let symIdx = totalSymbols - 1
            let lane = symIdx & 3
            let runToken = pairRunTokens[pairCount]
            let rtFreq = runModel.tokenFreqs[Int(runToken)]
            let rtCumFreq = runModel.tokenCumFreqs[Int(runToken)]
            enc.encodeSymbol(lane: lane, cumFreq: rtCumFreq, freq: rtFreq)
        }
        
        // 通常ペア: (runToken, valToken) を逆順でエンコード
        for i in stride(from: pairCount - 1, through: 0, by: -1) {
            // valToken のシンボルインデックス: i*2 + 1
            let valSymIdx = i * 2 + 1
            let valLane = valSymIdx & 3
            let valToken = pairValTokens[i]
            let vtFreq = valModel.tokenFreqs[Int(valToken)]
            let vtCumFreq = valModel.tokenCumFreqs[Int(valToken)]
            enc.encodeSymbol(lane: valLane, cumFreq: vtCumFreq, freq: vtFreq)
            
            // runToken のシンボルインデックス: i*2
            let runSymIdx = i * 2
            let runLane = runSymIdx & 3
            let runToken = pairRunTokens[i]
            let rtFreq = runModel.tokenFreqs[Int(runToken)]
            let rtCumFreq = runModel.tokenCumFreqs[Int(runToken)]
            enc.encodeSymbol(lane: runLane, cumFreq: rtCumFreq, freq: rtFreq)
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

// MARK: - VevcDecoder

public struct VevcDecoder {
    var bypassReader: BypassReader
    public var coeffs: [Int16]
    private var index: Int = 0

    public init(data: [UInt8]) throws {
        var offset = 0
        
        // 1. メタデータ (Bypass) の読み込み
        guard data.count >= 4 else { throw DecodeError.insufficientData }
        let bypassLen = vevc.readUInt32BE(data, at: &offset)
        
        guard offset + Int(bypassLen) <= data.count else { throw DecodeError.insufficientData }
        let bypassData = Array(data[offset..<(offset + Int(bypassLen))])
        self.bypassReader = BypassReader(data: bypassData)
        offset += Int(bypassLen)
        
        // 2. 係数数
        guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
        let coeffCount = Int(vevc.readUInt32BE(data, at: &offset))
        
        guard coeffCount > 0 else {
            self.coeffs = []
            return
        }
        
        // 3. フラグ
        guard offset < data.count else { throw DecodeError.insufficientData }
        let flags = data[offset]
        offset += 1
        let hasTrailingZeros = (flags & 1) != 0
        
        // 4. ペア数
        guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
        let totalPairCount = Int(vevc.readUInt32BE(data, at: &offset))
        let pairCount = hasTrailingZeros ? totalPairCount - 1 : totalPairCount
        
        // 5. runModel の周波数テーブル
        guard offset + 32 <= data.count else { throw DecodeError.insufficientData }
        var runTokenFreqs = [UInt32]()
        runTokenFreqs.reserveCapacity(16)
        for _ in 0..<16 {
            let f = (UInt32(data[offset]) << 8) | UInt32(data[offset+1])
            offset += 2
            runTokenFreqs.append(f)
        }
        let runModel = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: runTokenFreqs)
        
        // 6. valModel の周波数テーブル
        guard offset + 32 <= data.count else { throw DecodeError.insufficientData }
        var valTokenFreqs = [UInt32]()
        valTokenFreqs.reserveCapacity(16)
        for _ in 0..<16 {
            let f = (UInt32(data[offset]) << 8) | UInt32(data[offset+1])
            offset += 2
            valTokenFreqs.append(f)
        }
        let valModel = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: valTokenFreqs)
        
        // 7. 係数 Bypass データ
        guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
        let coeffBypassLen = Int(vevc.readUInt32BE(data, at: &offset))
        guard offset + coeffBypassLen <= data.count else { throw DecodeError.insufficientData }
        let coeffBypassData = Array(data[offset..<(offset + coeffBypassLen)])
        var coeffBypassReader = BypassReader(data: coeffBypassData)
        offset += coeffBypassLen
        
        // 8. Interleaved 4-way rANS ストリーム
        let ransData = Array(data[offset...])
        var ransDecoder = Interleaved4rANSDecoder(bitstream: ransData)
        
        // 9. デコード: 事前確保した配列に直接書き込み (修正3: append排除)
        var outCoeffs = [Int16](repeating: 0, count: coeffCount)
        var writeIdx = 0
        
        for pairIdx in 0..<pairCount {
            // runToken のシンボルインデックス: pairIdx*2
            let runLane = (pairIdx * 2) & 3
            let cfRun = ransDecoder.getCumulativeFreq(lane: runLane)
            let rtInfo = runModel.findToken(cf: cfRun)
            ransDecoder.advanceSymbol(lane: runLane, cumFreq: rtInfo.cumFreq, freq: rtInfo.freq)
            
            let runBypassBits = coeffBypassReader.readBits(count: max(0, Int(rtInfo.token) - 1))
            let zeroRun = Int(ValueTokenizer.detokenizeUnsigned(token: rtInfo.token, bypassBits: runBypassBits))
            
            // ゼロ分インデックスを進める (既に0で初期化済み)
            writeIdx += zeroRun
            
            // valToken のシンボルインデックス: pairIdx*2 + 1
            let valLane = (pairIdx * 2 + 1) & 3
            let cfVal = ransDecoder.getCumulativeFreq(lane: valLane)
            let vtInfo = valModel.findToken(cf: cfVal)
            ransDecoder.advanceSymbol(lane: valLane, cumFreq: vtInfo.cumFreq, freq: vtInfo.freq)
            
            let sign = coeffBypassReader.readBit()
            let valBypassBits = coeffBypassReader.readBits(count: ValueTokenizer.bypassLength(for: vtInfo.token))
            let val = ValueTokenizer.detokenize(isSignificant: true, sign: sign, token: vtInfo.token, bypassBits: valBypassBits)
            
            if writeIdx < coeffCount {
                outCoeffs[writeIdx] = val
            }
            writeIdx += 1
        }
        
        // 末尾ゼロラン
        if hasTrailingZeros {
            let totalSymbols = pairCount * 2 + 1
            let symIdx = totalSymbols - 1
            let lane = symIdx & 3
            let cfRun = ransDecoder.getCumulativeFreq(lane: lane)
            let rtInfo = runModel.findToken(cf: cfRun)
            ransDecoder.advanceSymbol(lane: lane, cumFreq: rtInfo.cumFreq, freq: rtInfo.freq)
            
            let _ = coeffBypassReader.readBits(count: max(0, Int(rtInfo.token) - 1))
            // 末尾ゼロは outCoeffs が既に 0 初期化済みのため進めるだけ
        }
        
        self.coeffs = outCoeffs
    }
    


    @inline(__always)
    public mutating func decodeBypass() throws -> UInt8 {
        return bypassReader.readBit() ? 1 : 0
    }

    @inline(__always)
    public mutating func readCoeff() throws -> Int16 {
        guard index < coeffs.count else { throw DecodeError.insufficientData }
        let val = coeffs[index]
        index += 1
        return val
    }
    
    /// EOF安全なreadCoeff: 係数が尽きた場合はnilを返す
    @inline(__always)
    public mutating func tryReadCoeff() -> Int16? {
        guard index < coeffs.count else { return nil }
        let val = coeffs[index]
        index += 1
        return val
    }
}

@inline(__always)
private func readUInt32BE(_ data: [UInt8], at offset: inout Int) -> UInt32 {
    let val = (UInt32(data[offset]) << 24) | (UInt32(data[offset+1]) << 16) |
              (UInt32(data[offset+2]) << 8) | UInt32(data[offset+3])
    offset += 4
    return val
}
