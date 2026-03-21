import Foundation

struct rANSCompressor {
    public static func compress(_ data: [Int16]) -> [UInt8] {
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
                let t = ValueTokenizer.tokenize(v)
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
    
    enum DecodeError: Error {
        case insufficientData
    }

    public static func decompress(_ data: [UInt8]) throws -> [Int16] {
        if data.isEmpty {
            return []
        }
        var offset = 0
        
        func readUInt32() throws -> Int {
            guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
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
        guard offset + ransStreamLen <= data.count else { throw DecodeError.insufficientData }
        let ransStream = Array(data[offset..<offset+ransStreamLen])
        offset += ransStreamLen
        
        var bypassBytes = [[UInt8]](repeating: [], count: 4)
        for i in 0..<4 {
            let len = try readUInt32()
            guard offset + len <= data.count else { throw DecodeError.insufficientData }
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
                    let bypassLen = ValueTokenizer.bypassLength(for: t.token)
                    let bypassBits = bypassReaders[lane].readBits(count: bypassLen)
                    outCoeffs[start + i] = ValueTokenizer.detokenize(token: t.token, bypassBits: bypassBits)
                } else {
                    outCoeffs[start + i] = 0
                }
            }
        }
        
        return outCoeffs
    }
}
