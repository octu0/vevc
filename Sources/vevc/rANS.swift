//
//  rANS.swift
//  vevc
//

import Foundation

/// rANS Parameters
public let RANS_SCALE_BITS: UInt32 = 14
public let RANS_SCALE: UInt32 = 1 << RANS_SCALE_BITS
public let RANS_L: UInt32 = 1 << 15
public let RANS_XMAX: UInt32 = ((RANS_L << 16) - 1)

// MARK: - rANS Encoder

public struct rANSEncoder {
    public private(set) var state: UInt32
    public private(set) var stream: [UInt16]
    
    public init() {
        self.state = RANS_L
        self.stream = []
        self.stream.reserveCapacity(4096)
    }
    
    /// シンボルをエンコードして状態を更新する (LIFOなので逆から処理する)
    @inline(__always)
    public mutating func encodeSymbol(cumFreq: UInt32, freq: UInt32) {
        // Renormalization
        let xMax = RANS_XMAX / RANS_SCALE * freq
        while state > xMax {
            stream.append(UInt16(truncatingIfNeeded: state))
            state >>= 16
        }
        // State update
        state = ((state / freq) << RANS_SCALE_BITS) + (state % freq) + cumFreq
    }
    
    /// ストリームをフラッシュし、最終状態を書き込む
    public mutating func flush() {
        stream.append(UInt16(truncatingIfNeeded: state))
        stream.append(UInt16(truncatingIfNeeded: state >> 16))
    }
    
    /// 生成されたバイトストリーム（逆順）を反転させて正しいByte配列を返す
    public func getBitstream() -> [UInt8] {
        var bytes = [UInt8]()
        bytes.reserveCapacity(stream.count * 2)
        for word in stream.reversed() {
            bytes.append(UInt8(truncatingIfNeeded: word >> 8))
            bytes.append(UInt8(truncatingIfNeeded: word & 0xFF))
        }
        return bytes
    }
}

// MARK: - rANS Decoder

public struct rANSDecoder {
    public private(set) var state: UInt32
    private let stream: [UInt8]
    private var offset: Int
    
    public init(bitstream: [UInt8]) {
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
    public func getCumulativeFreq() -> UInt32 {
        return state & (RANS_SCALE - 1)
    }
    
    @inline(__always)
    public mutating func advanceSymbol(cumFreq: UInt32, freq: UInt32) {
        let mask = RANS_SCALE - 1
        state = freq * (state >> RANS_SCALE_BITS) + (state & mask) - cumFreq
        
        // Renormalization
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

public struct BypassWriter {
    public private(set) var bytes: [UInt8]
    private var buffer: UInt32
    private var bitsInBuffer: Int
    
    public init() {
        self.bytes = []
        self.bytes.reserveCapacity(256)
        self.buffer = 0
        self.bitsInBuffer = 0
    }
    
    @inline(__always)
    public mutating func writeBit(_ bit: Bool) {
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
    
    /// count ビットを一括で書き込む (最大16ビット)
    @inline(__always)
    public mutating func writeBits(_ value: UInt16, count: Int) {
        guard count > 0 else { return }
        // バッファに加える
        buffer = (buffer << count) | UInt32(value & ((1 << count) - 1))
        bitsInBuffer += count
        // 32bit以上溜まったらフラッシュ
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
    
    public mutating func flush() {
        guard bitsInBuffer > 0 else { return }
        // 完全なバイト分を先に書き出す
        while bitsInBuffer >= 8 {
            bitsInBuffer -= 8
            bytes.append(UInt8(truncatingIfNeeded: buffer >> bitsInBuffer))
            if bitsInBuffer > 0 {
                buffer &= (1 << bitsInBuffer) - 1
            } else {
                buffer = 0
            }
        }
        // 残りの端数ビットを左詰めパディングして書き出す
        if bitsInBuffer > 0 {
            let shifted = buffer << (8 - bitsInBuffer)
            bytes.append(UInt8(truncatingIfNeeded: shifted))
            buffer = 0
            bitsInBuffer = 0
        }
    }
}

// MARK: - Bypass Reader (UInt32 ビットバッファ)

public struct BypassReader {
    private let bytes: [UInt8]
    private var byteOffset: Int
    private var buffer: UInt32
    private var bitsInBuffer: Int
    
    public init(data: [UInt8]) {
        self.bytes = data
        self.byteOffset = 0
        self.buffer = 0
        self.bitsInBuffer = 0
    }
    
    @inline(__always)
    private mutating func ensureBits(_ needed: Int) {
        while bitsInBuffer < needed && byteOffset < bytes.count {
            buffer = (buffer << 8) | UInt32(bytes[byteOffset])
            byteOffset += 1
            bitsInBuffer += 8
        }
    }
    
    @inline(__always)
    public mutating func readBit() -> Bool {
        ensureBits(1)
        bitsInBuffer -= 1
        let bit = (buffer >> bitsInBuffer) & 1
        if bitsInBuffer > 0 {
            buffer &= (1 << bitsInBuffer) - 1
        } else {
            buffer = 0
        }
        return bit == 1
    }
    
    /// count ビットを一括で読み込む (最大16ビット)
    @inline(__always)
    public mutating func readBits(count: Int) -> UInt16 {
        guard count > 0 else { return 0 }
        ensureBits(count)
        bitsInBuffer -= count
        let value = (buffer >> bitsInBuffer) & ((1 << count) - 1)
        if bitsInBuffer > 0 {
            buffer &= (1 << bitsInBuffer) - 1
        } else {
            buffer = 0
        }
        return UInt16(value)
    }
    
    public var consumedBytes: Int {
        // 読み込んだビット数 = (先読みしたバイト数 × 8) - バッファに残っているビット数
        // Writer の flush() でパディングされた端数バイトも消費済みとして扱うために切り上げ
        let totalBitsRead = byteOffset * 8 - bitsInBuffer
        return (totalBitsRead + 7) / 8
    }
}

// MARK: - rANS Probability Model (O(1) LUT付き)

public struct rANSModel {
    public private(set) var sigFreq: UInt32
    public private(set) var tokenFreqs: [UInt32]
    public private(set) var tokenCumFreqs: [UInt32]
    public private(set) var tokenLUT: [UInt8]  // O(1) 逆引きテーブル
    
    public init() {
        self.sigFreq = RANS_SCALE / 2
        self.tokenFreqs = Array(repeating: RANS_SCALE / 16, count: 16)
        self.tokenCumFreqs = (0..<16).map { UInt32($0) * (RANS_SCALE / 16) }
        self.tokenLUT = [UInt8](repeating: 0, count: Int(RANS_SCALE))
        buildLUT()
    }
    
    public init(sigFreq: UInt32, tokenFreqs: [UInt32]) {
        self.sigFreq = sigFreq
        self.tokenFreqs = tokenFreqs
        self.tokenCumFreqs = [UInt32](repeating: 0, count: 16)
        var sum: UInt32 = 0
        for i in 0..<16 {
            self.tokenCumFreqs[i] = sum
            sum += tokenFreqs[i]
        }
        self.tokenLUT = [UInt8](repeating: 0, count: Int(RANS_SCALE))
        buildLUT()
    }
    
    private mutating func buildLUT() {
        tokenLUT.withUnsafeMutableBufferPointer { ptr in
            for sym in 0..<16 {
                let start = Int(tokenCumFreqs[sym])
                let end = start + Int(tokenFreqs[sym])
                let s = UInt8(sym)
                for j in start..<min(end, Int(RANS_SCALE)) {
                    ptr[j] = s
                }
            }
        }
    }
    
    /// 生の出現回数からRANS_SCALEに合わせて正規化する
    public mutating func normalize(sigCounts: [Int], tokenCounts: [Int]) {
        // Significance
        let totalSig = sigCounts[0] + sigCounts[1]
        if totalSig == 0 {
            self.sigFreq = RANS_SCALE / 2
        } else {
            let f = UInt32((Int(RANS_SCALE) * sigCounts[1]) / totalSig)
            self.sigFreq = max(1, min(RANS_SCALE - 1, f))
        }
        
        // Tokens
        let totalTokens = tokenCounts.reduce(0, +)
        if totalTokens == 0 {
            self.tokenFreqs = Array(repeating: RANS_SCALE / 16, count: 16)
        } else {
            var sum: UInt32 = 0
            for i in 0..<16 {
                let count = tokenCounts[i]
                if count == 0 {
                    self.tokenFreqs[i] = 1
                } else {
                    let maxVal = RANS_SCALE - 16
                    self.tokenFreqs[i] = max(1, UInt32((Int(maxVal) * count) / totalTokens))
                }
                sum += self.tokenFreqs[i]
            }
            
            while sum < RANS_SCALE {
                let maxIdx = self.tokenFreqs.firstIndex(of: self.tokenFreqs.max()!)!
                self.tokenFreqs[maxIdx] += 1
                sum += 1
            }
            while sum > RANS_SCALE {
                let maxIdx = self.tokenFreqs.firstIndex(of: self.tokenFreqs.max()!)!
                if self.tokenFreqs[maxIdx] > 1 {
                    self.tokenFreqs[maxIdx] -= 1
                    sum -= 1
                } else {
                    break
                }
            }
        }
        
        // 累積頻度の更新
        var cumSum: UInt32 = 0
        for i in 0..<16 {
            self.tokenCumFreqs[i] = cumSum
            cumSum += self.tokenFreqs[i]
        }
        
        // O(1) LUT の構築
        buildLUT()
    }
    
    /// O(1) デコード: cfからトークンをLUTで取得
    @inline(__always)
    public func findToken(cf: UInt32) -> (token: UInt8, freq: UInt32, cumFreq: UInt32) {
        let sym = Int(tokenLUT[Int(cf)])
        return (UInt8(sym), tokenFreqs[sym], tokenCumFreqs[sym])
    }
}
