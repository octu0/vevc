import Foundation

enum ValueTokenizer {
    
    @inline(__always)
    public static func tokenize(_ value: Int16) -> (token: UInt8, bypassBits: UInt32, bypassLen: Int) {
        if value == 0 { return (0, 0, 0) }
        let sign = value < 0
        let absValue = UInt16(value.magnitude)
        
        if absValue <= 8 {
            let token = UInt8((absValue - 1) * 2 + (sign ? 1 : 0))
            return (token, 0, 0)
        }
        
        let v = UInt32(absValue - 9)
        if v == 0 {
            return (16, sign ? 1 : 0, 1)
        }
        
        let bits = UInt32.bitWidth - v.leadingZeroBitCount
        let subToken = UInt8(bits)
        let bypass = UInt32(v & ((1 << (bits - 1)) - 1))
        let bypassLen = bits - 1
        
        let token = 16 + subToken
        let finalBypass = (bypass << 1) | (sign ? 1 : 0)
        let finalBypassLen = bypassLen + 1
        
        return (token, finalBypass, finalBypassLen)
    }
    
    @inline(__always)
    public static func detokenize(token: UInt8, bypassBits: UInt32) -> Int16 {
        if token < 16 {
            let absValue = (UInt16(token) / 2) + 1
            let isNegative = (token % 2) == 1
            return isNegative ? Int16(bitPattern: 0 &- absValue) : Int16(bitPattern: absValue)
        }
        
        let subToken = token - 16
        let sign = (bypassBits & 1) == 1
        let bypass = bypassBits >> 1
        
        let v: UInt32
        if subToken == 0 {
            v = 0
        } else {
            let base: UInt32 = 1 << (UInt32(subToken) - 1)
            v = base | UInt32(bypass)
        }
        
        let absValue = v + 9
        if sign {
            let neg = 0 &- absValue
            return Int16(truncatingIfNeeded: neg)
        } else {
            return Int16(truncatingIfNeeded: absValue)
        }
    }
    
    @inline(__always)
    public static func tokenizeUnsigned(_ value: UInt32) -> (token: UInt8, bypassBits: UInt32, bypassLen: Int) {
        if value < 16 {
            return (UInt8(value), 0, 0)
        }
        
        let v = value - 16
        if v == 0 {
            return (16, 0, 0)
        }
        
        let bits = UInt32.bitWidth - v.leadingZeroBitCount
        let subToken = UInt8(bits)
        let bypass = UInt32(v & ((1 << (bits - 1)) - 1))
        let bypassLen = bits - 1
        
        return (16 + subToken, bypass, bypassLen)
    }
    
    @inline(__always)
    public static func detokenizeUnsigned(token: UInt8, bypassBits: UInt32) -> UInt32 {
        if token < 16 {
            return UInt32(token)
        }
        
        let subToken = token - 16
        let v: UInt32
        if subToken == 0 {
            v = 0
        } else {
            let base: UInt32 = 1 << (UInt32(subToken) - 1)
            v = base | UInt32(bypassBits)
        }
        return v + 16
    }
    
    @inline(__always)
    public static func bypassLength(for token: UInt8) -> Int {
        if token < 16 { return 0 }
        if token == 16 { return 1 }
        let t = min(token, 31)
        return Int(t - 16)
    }
    
    @inline(__always)
    public static func bypassLengthUnsigned(for token: UInt8) -> Int {
        if token < 16 { return 0 }
        if token == 16 { return 0 }
        let t = min(token, 31)
        return Int(t - 17)
    }
}
