
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
