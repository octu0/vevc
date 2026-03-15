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
        
        bypassWriter.flush()
        let metaBypassData = bypassWriter.bytes
        appendUInt32BE(&out, UInt32(metaBypassData.count))
        out.append(contentsOf: metaBypassData)
        
        let coeffCount = coeffs.count
        appendUInt32BE(&out, UInt32(coeffCount))
        
        guard coeffCount > 0 else { return out }
        
        var coeffBypassWriter = BypassWriter()
        var runTokenCounts = Array(repeating: 0, count: 16)
        var valTokenCounts = Array(repeating: 0, count: 16)
        
        var pairRunTokens = [UInt8]()
        pairRunTokens.reserveCapacity(coeffCount)
        var pairValTokens = [UInt8]()
        pairValTokens.reserveCapacity(coeffCount)
        
        var zeroRun: UInt32 = 0
        var nonZeroCount = 0
        for i in 0..<coeffCount {
            let c = coeffs[i]
            if c == 0 {
                zeroRun += 1
            } else {
                nonZeroCount += 1
                let runResult = ValueTokenizer.tokenizeUnsigned(zeroRun)
                runTokenCounts[Int(runResult.token)] += 1
                pairRunTokens.append(runResult.token)
                coeffBypassWriter.writeBits(runResult.bypassBits, count: runResult.bypassLen)
                
                let valResult = ValueTokenizer.tokenize(c)
                valTokenCounts[Int(valResult.token)] += 1
                pairValTokens.append(valResult.token)
                coeffBypassWriter.writeBit(valResult.sign)
                coeffBypassWriter.writeBits(valResult.bypassBits, count: ValueTokenizer.bypassLength(for: valResult.token))
                
                zeroRun = 0
            }
        }
        let hasTrailingZeros = zeroRun > 0
        if hasTrailingZeros {
            let runResult = ValueTokenizer.tokenizeUnsigned(zeroRun)
            runTokenCounts[Int(runResult.token)] += 1
            pairRunTokens.append(runResult.token)
            coeffBypassWriter.writeBits(runResult.bypassBits, count: runResult.bypassLen)
        }
        coeffBypassWriter.flush()
        
        let pairCount = pairValTokens.count
        
        if nonZeroCount <= 32 {
            // flags bit7 = 1 → Raw mode
            out.append(0x80)
            var rawBypass = BypassWriter()
            for i in 0..<coeffCount {
                let c = coeffs[i]
                if c == 0 {
                    rawBypass.writeBit(false)
                } else {
                    rawBypass.writeBit(true)
                    rawBypass.writeBit(c < 0)
                    let absVal = UInt32(abs(Int(c))) - 1
                    let result = ValueTokenizer.tokenizeUnsigned(absVal)
                    rawBypass.writeBits(UInt16(result.token), count: 4)
                    rawBypass.writeBits(result.bypassBits, count: result.bypassLen)
                }
            }
            rawBypass.flush()
            let rawData = rawBypass.bytes
            appendUInt32BE(&out, UInt32(rawData.count))
            out.append(contentsOf: rawData)
            return out
        }
        
        let totalPairCount = hasTrailingZeros ? pairCount + 1 : pairCount
        
        var runModel = rANSModel()
        runModel.normalize(sigCounts: [0, 1], tokenCounts: runTokenCounts)
        var valModel = rANSModel()
        valModel.normalize(sigCounts: [0, 1], tokenCounts: valTokenCounts)
        
        // flags: bit7=0 (rANS), bit0=hasTrailingZeros
        out.append(hasTrailingZeros ? 1 : 0)
        appendUInt32BE(&out, UInt32(totalPairCount))
        
        writeCompressedFreqTable(&out, freqs: runModel.tokenFreqs)
        writeCompressedFreqTable(&out, freqs: valModel.tokenFreqs)
        
        let coeffBypassData = coeffBypassWriter.bytes
        appendUInt32BE(&out, UInt32(coeffBypassData.count))
        out.append(contentsOf: coeffBypassData)
        
        var enc = Interleaved4rANSEncoder()
        let totalSymbols = pairCount * 2 + (hasTrailingZeros ? 1 : 0)
        
        if hasTrailingZeros {
            let symIdx = totalSymbols - 1
            let lane = symIdx & 3
            let runToken = pairRunTokens[pairCount]
            enc.encodeSymbol(lane: lane, cumFreq: runModel.tokenCumFreqs[Int(runToken)], freq: runModel.tokenFreqs[Int(runToken)])
        }
        
        for i in stride(from: pairCount - 1, through: 0, by: -1) {
            let valLane = (i * 2 + 1) & 3
            let valToken = pairValTokens[i]
            enc.encodeSymbol(lane: valLane, cumFreq: valModel.tokenCumFreqs[Int(valToken)], freq: valModel.tokenFreqs[Int(valToken)])
            
            let runLane = (i * 2) & 3
            let runToken = pairRunTokens[i]
            enc.encodeSymbol(lane: runLane, cumFreq: runModel.tokenCumFreqs[Int(runToken)], freq: runModel.tokenFreqs[Int(runToken)])
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
    public var coeffs: [Int16]
    private var index: Int = 0

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
            self.coeffs = []
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
            
            var outCoeffs = [Int16](repeating: 0, count: coeffCount)
            for i in 0..<coeffCount {
                let isNonZero = rawReader.readBit()
                if isNonZero {
                    let isNeg = rawReader.readBit()
                    let tokenBits = rawReader.readBits(count: 4)
                    let token = UInt8(tokenBits)
                    let bypassBits = rawReader.readBits(count: max(0, Int(token) - 1))
                    let absVal = Int16(ValueTokenizer.detokenizeUnsigned(token: token, bypassBits: bypassBits)) + 1
                    outCoeffs[i] = isNeg ? -absVal : absVal
                }
            }
            self.coeffs = outCoeffs
            return
        }
        
        let hasTrailingZeros = (flags & 1) != 0
        
        guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
        let totalPairCount = Int(vevc.readUInt32BE(data, at: &offset))
        let pairCount = hasTrailingZeros ? totalPairCount - 1 : totalPairCount
        
        let runTokenFreqs = try VevcDecoder.readCompressedFreqTable(data, at: &offset)
        let runModel = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: runTokenFreqs)
        
        let valTokenFreqs = try VevcDecoder.readCompressedFreqTable(data, at: &offset)
        let valModel = rANSModel(sigFreq: RANS_SCALE / 2, tokenFreqs: valTokenFreqs)
        
        guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
        let coeffBypassLen = Int(vevc.readUInt32BE(data, at: &offset))
        guard offset + coeffBypassLen <= data.count else { throw DecodeError.insufficientData }
        let coeffBypassData = Array(data[offset..<(offset + coeffBypassLen)])
        var coeffBypassReader = BypassReader(data: coeffBypassData)
        offset += coeffBypassLen
        
        let ransData = Array(data[offset...])
        var ransDecoder = Interleaved4rANSDecoder(bitstream: ransData)
        
        var outCoeffs = [Int16](repeating: 0, count: coeffCount)
        var writeIdx = 0
        
        for pairIdx in 0..<pairCount {
            let runLane = (pairIdx * 2) & 3
            let cfRun = ransDecoder.getCumulativeFreq(lane: runLane)
            let rtInfo = runModel.findToken(cf: cfRun)
            ransDecoder.advanceSymbol(lane: runLane, cumFreq: rtInfo.cumFreq, freq: rtInfo.freq)
            
            let runBypassBits = coeffBypassReader.readBits(count: max(0, Int(rtInfo.token) - 1))
            let zeroRun = Int(ValueTokenizer.detokenizeUnsigned(token: rtInfo.token, bypassBits: runBypassBits))
            
            writeIdx += zeroRun
            
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
        
        if hasTrailingZeros {
            let totalSymbols = pairCount * 2 + 1
            let symIdx = totalSymbols - 1
            let lane = symIdx & 3
            let cfRun = ransDecoder.getCumulativeFreq(lane: lane)
            let rtInfo = runModel.findToken(cf: cfRun)
            ransDecoder.advanceSymbol(lane: lane, cumFreq: rtInfo.cumFreq, freq: rtInfo.freq)
            
            let _ = coeffBypassReader.readBits(count: max(0, Int(rtInfo.token) - 1))
        }
        
        self.coeffs = outCoeffs
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
    public mutating func readCoeff() throws -> Int16 {
        guard index < coeffs.count else { throw DecodeError.insufficientData }
        let val = coeffs[index]
        index += 1
        return val
    }
    
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
    let val = (UInt32(data[offset]) << 24) | (UInt32(data[offset+1]) << 16) | (UInt32(data[offset+2]) << 8) | UInt32(data[offset+3])
    offset += 4
    return val
}
