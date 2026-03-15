import Foundation

public enum ValueTokenizer {
    
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
    
    @inline(__always)
    public static func bypassLength(for token: UInt8) -> Int {
        return Int(token)
    }
    
    /// value=0 → token=0, bypass=0
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
    
    @inline(__always)
    public static func detokenizeUnsigned(token: UInt8, bypassBits: UInt16) -> UInt32 {
        if token == 0 {
            return 0
        }
        let base: UInt32 = 1 << (UInt32(token) - 1)
        return base | UInt32(bypassBits)
    }
}
