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
        // State update: 除算1回のみ (q*freq を引いて剰余を計算)
        let q = state / freq
        state = (q << RANS_SCALE_BITS) + (state - q * freq) + cumFreq
    }
    
    /// ストリームをフラッシュし、最終状態を書き込む
    public mutating func flush() {
        stream.append(UInt16(truncatingIfNeeded: state))
        stream.append(UInt16(truncatingIfNeeded: state >> 16))
    }
    
    /// 生成されたバイトストリーム（逆順）を反転させてByte配列を返す
    public func getBitstream() -> [UInt8] {
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

// MARK: - Interleaved 4-way rANS Encoder

/// 4つの独立した rANS state をインターリーブしてエンコードする。
/// シンボル i は state[i % 4] に割り当てられる。
/// エンコードは逆順で行われ、renormalization words は共有ストリームに書き込まれる。
public struct Interleaved4rANSEncoder {
    public private(set) var states: (UInt32, UInt32, UInt32, UInt32)
    public private(set) var stream: [UInt16]
    
    public init() {
        self.states = (RANS_L, RANS_L, RANS_L, RANS_L)
        self.stream = []
        self.stream.reserveCapacity(4096)
    }
    
    /// 指定された lane (0-3) のシンボルをエンコード
    @inline(__always)
    public mutating func encodeSymbol(lane: Int, cumFreq: UInt32, freq: UInt32) {
        let xMax = RANS_XMAX / RANS_SCALE * freq
        
        switch lane {
        case 0:
            while states.0 > xMax {
                stream.append(UInt16(truncatingIfNeeded: states.0))
                states.0 >>= 16
            }
            let q = states.0 / freq
            states.0 = (q << RANS_SCALE_BITS) + (states.0 - q * freq) + cumFreq
        case 1:
            while states.1 > xMax {
                stream.append(UInt16(truncatingIfNeeded: states.1))
                states.1 >>= 16
            }
            let q = states.1 / freq
            states.1 = (q << RANS_SCALE_BITS) + (states.1 - q * freq) + cumFreq
        case 2:
            while states.2 > xMax {
                stream.append(UInt16(truncatingIfNeeded: states.2))
                states.2 >>= 16
            }
            let q = states.2 / freq
            states.2 = (q << RANS_SCALE_BITS) + (states.2 - q * freq) + cumFreq
        case 3:
            while states.3 > xMax {
                stream.append(UInt16(truncatingIfNeeded: states.3))
                states.3 >>= 16
            }
            let q = states.3 / freq
            states.3 = (q << RANS_SCALE_BITS) + (states.3 - q * freq) + cumFreq
        default:
            break
        }
    }
    
    /// 4つの state をフラッシュ (state3, state2, state1, state0 の順で書き込み)
    public mutating func flush() {
        // デコード側は state0, state1, state2, state3 の順で読むため逆順書き込み
        stream.append(UInt16(truncatingIfNeeded: states.3))
        stream.append(UInt16(truncatingIfNeeded: states.3 >> 16))
        stream.append(UInt16(truncatingIfNeeded: states.2))
        stream.append(UInt16(truncatingIfNeeded: states.2 >> 16))
        stream.append(UInt16(truncatingIfNeeded: states.1))
        stream.append(UInt16(truncatingIfNeeded: states.1 >> 16))
        stream.append(UInt16(truncatingIfNeeded: states.0))
        stream.append(UInt16(truncatingIfNeeded: states.0 >> 16))
    }
    
    /// 共有ストリームを逆順にしてバイト配列を返す
    public func getBitstream() -> [UInt8] {
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

/// 4つの独立した rANS state をインターリーブしてデコードする。
/// 共有ストリームから renormalization words を読み込み、各 state で交互にデコードする。
public struct Interleaved4rANSDecoder {
    public private(set) var states: (UInt32, UInt32, UInt32, UInt32)
    private let stream: [UInt8]
    private var offset: Int
    
    public init(bitstream: [UInt8]) {
        self.stream = bitstream
        self.offset = 0
        self.states = (RANS_L, RANS_L, RANS_L, RANS_L)
        
        // 4つの state を読み込み (state0, state1, state2, state3 の順)
        guard bitstream.count >= 16 else { return }
        
        @inline(__always)
        func readState(_ off: Int) -> UInt32 {
            let b0 = UInt32(bitstream[off])
            let b1 = UInt32(bitstream[off + 1])
            let b2 = UInt32(bitstream[off + 2])
            let b3 = UInt32(bitstream[off + 3])
            return ((b0 << 8) | b1) << 16 | ((b2 << 8) | b3)
        }
        
        self.states.0 = readState(0)
        self.states.1 = readState(4)
        self.states.2 = readState(8)
        self.states.3 = readState(12)
        self.offset = 16
    }
    
    /// 指定 lane の累積頻度を取得
    @inline(__always)
    public func getCumulativeFreq(lane: Int) -> UInt32 {
        let mask = RANS_SCALE - 1
        switch lane {
        case 0: return states.0 & mask
        case 1: return states.1 & mask
        case 2: return states.2 & mask
        case 3: return states.3 & mask
        default: return 0
        }
    }
    
    /// renormalization のための16ビットワード読み込み
    @inline(__always)
    private mutating func readWord() -> UInt32 {
        if offset + 1 < stream.count {
            let b0 = UInt32(stream[offset])
            let b1 = UInt32(stream[offset + 1])
            offset += 2
            return (b0 << 8) | b1
        }
        return 0
    }
    
    /// 指定 lane のシンボルをアドバンス
    @inline(__always)
    public mutating func advanceSymbol(lane: Int, cumFreq: UInt32, freq: UInt32) {
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
            
            // 合計をRANS_SCALEに調整: 最大頻度のインデックスを直接追跡
            if sum != RANS_SCALE {
                var maxIdx = 0
                var maxVal = self.tokenFreqs[0]
                for i in 1..<16 {
                    if self.tokenFreqs[i] > maxVal {
                        maxVal = self.tokenFreqs[i]
                        maxIdx = i
                    }
                }
                if sum < RANS_SCALE {
                    self.tokenFreqs[maxIdx] += (RANS_SCALE - sum)
                } else if self.tokenFreqs[maxIdx] > (sum - RANS_SCALE) + 1 {
                    self.tokenFreqs[maxIdx] -= (sum - RANS_SCALE)
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
