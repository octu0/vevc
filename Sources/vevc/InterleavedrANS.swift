//
//  InterleavedrANS.swift
//  vevc
//

import Foundation

// MARK: - 4-way Interleaved rANS Encoder

struct InterleavedrANSEncoder {
    public private(set) var states: [UInt32]
    public private(set) var streams: [[UInt16]]
    
    public init() {
        self.states = [RANS_L, RANS_L, RANS_L, RANS_L]
        self.streams = [
            [UInt16](), [UInt16](), [UInt16](), [UInt16]()
        ]
        for i in 0..<4 {
            self.streams[i].reserveCapacity(1024)
        }
    }
    
    @inline(__always)
    public mutating func encodeSymbol(lane: Int, cumFreq: UInt32, freq: UInt32) {
        var state = states[lane]
        let xMax = RANS_XMAX / RANS_SCALE * freq
        
        while state > xMax {
            streams[lane].append(UInt16(truncatingIfNeeded: state))
            state >>= 16
        }
        
        state = ((state / freq) << RANS_SCALE_BITS) + (state % freq) + cumFreq
        states[lane] = state
    }
    
    public mutating func flush() {
        for lane in 0..<4 {
            streams[lane].append(UInt16(truncatingIfNeeded: states[lane]))
            streams[lane].append(UInt16(truncatingIfNeeded: states[lane] >> 16))
        }
    }
    
    /// merge 4 streams into a single bitstream
    /// format: [len0(4bytes)][len1(4bytes)][len2(4bytes)][len3(4bytes)][stream0][stream1][stream2][stream3]
    public func getBitstream() -> [UInt8] {
        var bytes = [UInt8]()
        
        var lengths = [Int](repeating: 0, count: 4)
        for i in 0..<4 {
            lengths[i] = streams[i].count * 2
        }
        
        // ヘッダ (各ストリームのバイト長)
        for len in lengths {
            let l = UInt32(len)
            bytes.append(UInt8(truncatingIfNeeded: l >> 24))
            bytes.append(UInt8(truncatingIfNeeded: (l >> 16) & 0xFF))
            bytes.append(UInt8(truncatingIfNeeded: (l >> 8) & 0xFF))
            bytes.append(UInt8(truncatingIfNeeded: l & 0xFF))
        }
        
        // ボディ (逆順で書き出す: LIFO)
        for lane in 0..<4 {
            for word in streams[lane].reversed() {
                bytes.append(UInt8(truncatingIfNeeded: word >> 8)) // BE
                bytes.append(UInt8(truncatingIfNeeded: word & 0xFF))
            }
        }
        
        return bytes
    }
}

// MARK: - 4-way Interleaved rANS SIMD Decoder

public struct InterleavedrANSDecoder {
    public private(set) var states: SIMD4<UInt32>
    
    private let stream: [UInt8]
    // 4 lanes independent offsets
    private var offsets: SIMD4<Int>
    private let limits: SIMD4<Int>
    
    public init(bitstream: [UInt8]) {
        self.stream = bitstream
        
        // Header parse
        guard bitstream.count >= 16 else {
            self.states = SIMD4<UInt32>(repeating: 0)
            self.offsets = SIMD4<Int>(repeating: 0)
            self.limits = SIMD4<Int>(repeating: 0)
            return
        }
        
        var lens = SIMD4<Int>(repeating: 0)
        for i in 0..<4 {
            let idx = i * 4
            let b0 = UInt32(bitstream[idx])
            let b1 = UInt32(bitstream[idx + 1])
            let b2 = UInt32(bitstream[idx + 2])
            let b3 = UInt32(bitstream[idx + 3])
            lens[i] = Int((b0 << 24) | (b1 << 16) | (b2 << 8) | b3)
        }
        
        var currentOffset = 16
        var initOffsets = SIMD4<Int>(repeating: 0)
        var initLimits = SIMD4<Int>(repeating: 0)
        var initStates = SIMD4<UInt32>(repeating: 0)
        
        for i in 0..<4 {
            let limit = currentOffset + lens[i]
            if currentOffset + 4 <= limit {
                // stream is written in reverse order (LIFO)
                // so the last flushed state (4 bytes) is at the beginning of each stream block
                let w1 = (UInt32(bitstream[currentOffset]) << 8) | UInt32(bitstream[currentOffset + 1])
                let w0 = (UInt32(bitstream[currentOffset + 2]) << 8) | UInt32(bitstream[currentOffset + 3])
                initStates[i] = (w1 << 16) | w0
                
                // next renorm reads from the 4 bytes after this
                initOffsets[i] = currentOffset + 4
            } else {
                initStates[i] = 0
                initOffsets[i] = limit
            }
            initLimits[i] = limit
            currentOffset = limit
        }
        
        self.offsets = initOffsets
        self.limits = initLimits
        self.states = initStates
    }
    
    @inline(__always)
    public func getCumulativeFreqs() -> SIMD4<UInt32> {
        let mask = SIMD4<UInt32>(repeating: RANS_SCALE - 1)
        return states & mask
    }
    
    @inline(__always)
    public mutating func advanceSymbols(cumFreqs: SIMD4<UInt32>, freqs: SIMD4<UInt32>, activeMask: SIMD4<UInt32> = SIMD4<UInt32>(repeating: 0xFFFFFFFF)) {
        // [SIMD] Data parallel non-dependent update
        let mask = SIMD4<UInt32>(repeating: RANS_SCALE - 1)
        let nextStates = freqs &* (states &>> RANS_SCALE_BITS) &+ (states & mask) &- cumFreqs
        
        let boolMask = activeMask .== SIMD4<UInt32>(repeating: 0xFFFFFFFF)
        states.replace(with: nextStates, where: boolMask)
        
        // SIMD Renormalization
        let th = SIMD4<UInt32>(repeating: RANS_L)
        
        // first renormalization
        var renormMask = (states .< th) .& boolMask
        if any(renormMask) {
            for lane in 0..<4 {
                if renormMask[lane] {
                    let off = offsets[lane]
                    if off + 1 < limits[lane] {
                        let word = (UInt32(stream[off]) << 8) | UInt32(stream[off + 1])
                        offsets[lane] = off + 2
                        states[lane] = (states[lane] &<< 16) | word
                    } else {
                        states[lane] = states[lane] &<< 16
                    }
                }
            }
        }
        
        // second renormalization
        renormMask = (states .< th) .& boolMask
        if any(renormMask) {
            for lane in 0..<4 {
                if renormMask[lane] {
                    let off = offsets[lane]
                    if off + 1 < limits[lane] {
                        let word = (UInt32(stream[off]) << 8) | UInt32(stream[off + 1])
                        offsets[lane] = off + 2
                        states[lane] = (states[lane] &<< 16) | word
                    } else {
                        states[lane] = states[lane] &<< 16
                    }
                }
            }
        }
    }
}
