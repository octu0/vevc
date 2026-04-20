
enum BinaryError: Error, CustomStringConvertible {
    case eof
    case insufficientData(message: String)
    
    var description: String {
        switch self {
        case .eof: return "EOF"
        case .insufficientData(let msg): return "insufficientData: \(msg)"
        }
    }
}

// MARK: - Byte Serialization Helpers

@inline(__always)
func appendUInt16BE(_ out: inout [UInt8], _ val: UInt16) {
    out.append(UInt8(val >> 8))
    out.append(UInt8(val & 0xFF))
}

@inline(__always)
func appendUInt32BE(_ out: inout [UInt8], _ val: UInt32) {
    out.append(UInt8((val >> 24) & 0xFF))
    out.append(UInt8((val >> 16) & 0xFF))
    out.append(UInt8((val >> 8) & 0xFF))
    out.append(UInt8(val & 0xFF))
}


@inline(__always)
public func readUInt16BEFromBytes(_ r: [UInt8], offset: inout Int) throws -> UInt16 {
    guard (offset + 2) <= r.count else {
        throw BinaryError.insufficientData(message: "readUInt16BEFromBytes offset(\(offset)) + 2 > r.count(\(r.count))")
    }
    let b0 = UInt16(r[offset])
    let b1 = UInt16(r[offset + 1])
    offset += 2
    return (b0 << 8) | b1
}

@inline(__always)
public func readUInt32BEFromBytes(_ r: [UInt8], offset: inout Int) throws -> UInt32 {
    guard (offset + 4) <= r.count else {
        throw BinaryError.insufficientData(message: "readUInt32BEFromBytes offset(\(offset)) + 4 > r.count(\(r.count))")
    }
    let val = (UInt32(r[offset]) << 24) | (UInt32(r[offset + 1]) << 16) | (UInt32(r[offset + 2]) << 8) | UInt32(r[offset + 3])
    offset += 4
    return val
}

@inline(__always)
func readUInt64BEFromBytes(_ r: [UInt8], offset: inout Int) throws -> UInt64 {
    let high = try readUInt32BEFromBytes(r, offset: &offset)
    let low = try readUInt32BEFromBytes(r, offset: &offset)
    return (UInt64(high) << 32) | UInt64(low)
}

@inline(__always)
internal func readUInt16BEFromPtr(_ base: UnsafePointer<UInt8>, offset: inout Int, count: Int) throws -> UInt16 {
    guard offset + 2 <= count else { throw BinaryError.insufficientData(message: "readUInt16BEFromPtr") }
    let b0 = UInt16(base[offset + 0])
    let b1 = UInt16(base[offset + 1])
    offset += 2
    return (b0 << 8) | b1
}

@inline(__always)
internal func readUInt32BEFromPtr(_ base: UnsafePointer<UInt8>, offset: inout Int, count: Int) throws -> UInt32 {
    guard offset + 4 <= count else { throw BinaryError.insufficientData(message: "readUInt32BEFromPtr") }
    let b0 = UInt32(base[offset + 0])
    let b1 = UInt32(base[offset + 1])
    let b2 = UInt32(base[offset + 2])
    let b3 = UInt32(base[offset + 3])
    offset += 4
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
}

@inline(__always)
internal func readUInt64BEFromPtr(_ base: UnsafePointer<UInt8>, offset: inout Int, count: Int) throws -> UInt64 {
    guard offset + 8 <= count else { throw BinaryError.insufficientData(message: "readUInt64BEFromPtr") }
    let b0 = UInt64(base[offset + 0])
    let b1 = UInt64(base[offset + 1])
    let b2 = UInt64(base[offset + 2])
    let b3 = UInt64(base[offset + 3])
    let b4 = UInt64(base[offset + 4])
    let b5 = UInt64(base[offset + 5])
    let b6 = UInt64(base[offset + 6])
    let b7 = UInt64(base[offset + 7])
    offset += 8
    return (b0 << 56) | (b1 << 48) | (b2 << 40) | (b3 << 32) | (b4 << 24) | (b5 << 16) | (b6 << 8) | b7
}
