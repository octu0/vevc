import Foundation

@inline(__always)
func valueTokenize(_ value: Int16) -> (token: UInt8, bypassBits: UInt32, bypassLen: Int) {
    if value == 0 { return (0, 0, 0) }
    let sign = value < 0
    let absValue = UInt16(value.magnitude)
    
    if absValue <= 16 {
        let token = UInt8((absValue - 1) * 2 + (sign ? 1 : 0))
        return (token, 0, 0)
    }
    
    let v = UInt32(absValue - 17)
    if v == 0 {
        return (32, sign ? 1 : 0, 1)
    }
    
    let bits = UInt32.bitWidth - v.leadingZeroBitCount
    let subToken = UInt8(bits)
    let bypass = UInt32(v & ((1 << (bits - 1)) - 1))
    let bypassLen = bits - 1
    
    let token = 32 + subToken
    let finalBypass = (bypass << 1) | (sign ? 1 : 0)
    let finalBypassLen = bypassLen + 1
    
    return (token, finalBypass, finalBypassLen)
}

@inline(__always)
func valueDetokenize(token: UInt8, bypassBits: UInt32) -> Int16 {
    if token < 32 {
        let absValue = (UInt16(token) / 2) + 1
        let isNegative = (token % 2) == 1
        return isNegative ? Int16(bitPattern: 0 &- absValue) : Int16(bitPattern: absValue)
    }
    
    let subToken = token - 32
    let sign = (bypassBits & 1) == 1
    let bypass = bypassBits >> 1
    
    let v: UInt32
    if subToken == 0 {
        v = 0
    } else {
        let base: UInt32 = 1 << (UInt32(subToken) - 1)
        v = base | UInt32(bypass)
    }
    
    let absValue = v + 17
    if sign {
        let neg = 0 &- absValue
        return Int16(truncatingIfNeeded: neg)
    } else {
        return Int16(truncatingIfNeeded: absValue)
    }
}

@inline(__always)
func valueTokenizeUnsigned(_ value: UInt32) -> (token: UInt8, bypassBits: UInt32, bypassLen: Int) {
    if value < 32 {
        return (UInt8(value), 0, 0)
    }
    
    let v = value - 32
    if v == 0 {
        return (32, 0, 0)
    }
    
    let bits = UInt32.bitWidth - v.leadingZeroBitCount
    let subToken = UInt8(bits)
    let bypass = UInt32(v & ((1 << (bits - 1)) - 1))
    let bypassLen = bits - 1
    
    return (32 + subToken, bypass, bypassLen)
}

@inline(__always)
func valueDetokenizeUnsigned(token: UInt8, bypassBits: UInt32) -> UInt32 {
    if token < 32 {
        return UInt32(token)
    }
    
    let subToken = token - 32
    let v: UInt32
    if subToken == 0 {
        v = 0
    } else {
        let base: UInt32 = 1 << (UInt32(subToken) - 1)
        v = base | UInt32(bypassBits)
    }
    return v + 32
}

@inline(__always)
func valueBypassLength(for token: UInt8) -> Int {
    if token < 32 { return 0 }
    if token == 32 { return 1 }
    let t = min(token, 63)
    return Int(t - 32)
}

@inline(__always)
func valueBypassLengthUnsigned(for token: UInt8) -> Int {
    if token < 32 { return 0 }
    if token == 32 { return 0 }
    let t = min(token, 63)
    return Int(t - 33)
}
