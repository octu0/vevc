//
//  ValueTokenizer.swift
//  vevc
//

import Foundation

/// DWT係数などの値をTokenとBypass Bitsに分離・復元する構造体
public enum ValueTokenizer {
    
    /// 係数値を各要素に分解する
    /// - Parameter value: 入力値 (Int16)
    /// - Returns:
    ///   - isSignificant: 値が0でないかどうか
    ///   - sign: 符号(true なら負、バイパス出力する対象)
    ///   - token: 絶対値の対数カテゴリ (確率モデルを通す)
    ///   - bypassBits: Token長で表現される残りの生ビット (バイパス出力する対象)
    @inline(__always)
    public static func tokenize(_ value: Int16) -> (isSignificant: Bool, sign: Bool, token: UInt8, bypassBits: UInt16) {
        if value == 0 {
            return (false, false, 0, 0)
        }
        
        let sign = value < 0
        let absValue = UInt16(value.magnitude)
        
        if absValue == 1 {
            return (true, sign, 0, 0)
        }
        
        let leadingZeros = absValue.leadingZeroBitCount
        let token = UInt8(UInt16.bitWidth - leadingZeros - 1)
        
        let bypassBits = absValue &- (1 &<< token)
        return (true, sign, token, bypassBits)
    }
    
    /// 分解された要素から元の係数値を復元する
    @inline(__always)
    public static func detokenize(isSignificant: Bool, sign: Bool, token: UInt8, bypassBits: UInt16) -> Int16 {
        if !isSignificant {
            return 0
        }
        
        let absValue: UInt16
        if token == 0 {
            absValue = 1
        } else {
            absValue = (1 &<< token) &+ bypassBits
        }
        
        let intAbs = Int(absValue)
        return sign ? Int16(-intAbs) : Int16(intAbs)
    }
    
    /// 特定のTokenにおけるBypassビットの長さを取得する（Tokenの値に等しい）
    @inline(__always)
    public static func bypassLength(for token: UInt8) -> Int {
        return Int(token)
    }
    
    // MARK: - 符号なし値のToken化（Zero-Run長用）
    
    /// 符号なし整数をToken化する（Zero-Run長用）
    /// value=0 → token=0, bypass=0 (runLength=0 は 0 を意味する)
    /// value=1 → token=1, bypass=0
    /// value=2..3 → token=2, bypass=0..1
    /// value=4..7 → token=3, bypass=0..3
    @inline(__always)
    public static func tokenizeUnsigned(_ value: UInt32) -> (token: UInt8, bypassBits: UInt16, bypassLen: Int) {
        if value == 0 {
            return (0, 0, 0)
        }
        let bits = UInt32.bitWidth - value.leadingZeroBitCount
        let token = UInt8(bits)
        let bypass = UInt16(value & ((1 << (bits - 1)) - 1))
        let bypassLen = bits - 1
        return (token, bypass, bypassLen)
    }
    
    /// 符号なしTokenからZero-Run長を復元する
    @inline(__always)
    public static func detokenizeUnsigned(token: UInt8, bypassBits: UInt16) -> UInt32 {
        if token == 0 {
            return 0
        }
        let base: UInt32 = 1 << (UInt32(token) - 1)
        return base | UInt32(bypassBits)
    }
}
