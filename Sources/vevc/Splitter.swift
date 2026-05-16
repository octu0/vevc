import Foundation

/// Result of a VEVC stream split operation.
public struct SplitterResult: Sendable {
    public let data: [UInt8]
    public let processedFrames: Int
    public let droppedLayer1Bytes: Int
    public let droppedLayer2Bytes: Int
    
    public var totalDroppedBytes: Int {
        droppedLayer1Bytes + droppedLayer2Bytes
    }
}

public enum SplitterError: Error {
    case invalidMagic
    case invalidFrameType(UInt8)
    case unexpectedEOF
    case invalidMaxLayer(Int)
}

/// Splits a VEVC bitstream in-memory, dropping layers above `maxLayer`.
///
/// - Parameters:
///   - input: Full VEVC encoded data (all 3 layers).
///   - maxLayer: Maximum layer to retain (0 = layer0 only, 1 = layer0+1, 2 = all layers).
/// - Returns: A `SplitterResult` containing the stripped bitstream and statistics.
@inline(__always)
public func splitVEVCStream(input: [UInt8], maxLayer: Int) throws -> SplitterResult {
    guard 0 <= maxLayer, maxLayer <= 2 else {
        throw SplitterError.invalidMaxLayer(maxLayer)
    }
    
    var readOffset = 0
    
    @inline(__always)
    func readFully(count: Int) throws -> ArraySlice<UInt8> {
        let end = readOffset + count
        guard end <= input.count else {
            throw SplitterError.unexpectedEOF
        }
        let slice = input[readOffset..<end]
        readOffset = end
        return slice
    }
    
    // 1. Read and copy FileHeader magic
    let magicSlice = try readFully(count: 4)
    guard magicSlice.elementsEqual([0x56, 0x45, 0x56, 0x43]) else {
        throw SplitterError.invalidMagic
    }
    
    // 2. Read metadataSize
    let metadataSizeSlice = try readFully(count: 2)
    let msOffset = metadataSizeSlice.startIndex
    let metadataSize = Int(UInt16(metadataSizeSlice[msOffset]) << 8 | UInt16(metadataSizeSlice[msOffset + 1]))
    
    // 3. Read metadata payload
    let metadataSlice = try readFully(count: metadataSize)
    
    // Pre-allocate output buffer
    var output = [UInt8]()
    output.reserveCapacity(input.count)
    
    // Write FileHeader to output
    output.append(contentsOf: magicSlice)
    output.append(contentsOf: metadataSizeSlice)
    output.append(contentsOf: metadataSlice)
    
    var processedFrames = 0
    var droppedLayer1Bytes = 0
    var droppedLayer2Bytes = 0
    
    while readOffset < input.count {
        let flagSlice = try readFully(count: 1)
        let flagByte = flagSlice[flagSlice.startIndex]
        
        guard let fType = VEVCFrameHeader.FrameType(rawValue: flagByte) else {
            throw SplitterError.invalidFrameType(flagByte)
        }
        
        // CopyFrame: write flag byte only
        if fType == .copyFrame {
            output.append(flagByte)
            processedFrames += 1
            continue
        }
        
        // Read 6 x UInt32BE = 24 bytes of frame header sizes
        let sizesSlice = try readFully(count: 24)
        var sizeBase = sizesSlice.startIndex
        
        @inline(__always)
        func readU32() -> Int {
            let v = Int(UInt32(input[sizeBase]) << 24 | UInt32(input[sizeBase+1]) << 16
                        | UInt32(input[sizeBase+2]) << 8 | UInt32(input[sizeBase+3]))
            sizeBase += 4
            return v
        }
        
        let mvsCount  = readU32()
        let mvsSize   = readU32()
        let refDirSize = readU32()
        let layer0Size = readU32()
        let layer1Size = readU32()
        let layer2Size = readU32()
        
        // Rebuild header with trimmed layer sizes
        let newLayer1Size = if 1 <= maxLayer { layer1Size } else { 0 }
        let newLayer2Size = if 2 <= maxLayer { layer2Size } else { 0 }
        
        let newHeader = VEVCFrameHeader(
            frameType: fType,
            mvsCount: mvsCount,
            mvsSize: mvsSize,
            refDirSize: refDirSize,
            layer0Size: layer0Size,
            layer1Size: newLayer1Size,
            layer2Size: newLayer2Size
        )
        output.append(contentsOf: newHeader.serialize())
        
        // MVs payload
        if 0 < mvsSize {
            let payload = try readFully(count: mvsSize)
            output.append(contentsOf: payload)
        }
        // RefDir payload
        if 0 < refDirSize {
            let payload = try readFully(count: refDirSize)
            output.append(contentsOf: payload)
        }
        // Layer 0 payload (always retained)
        if 0 < layer0Size {
            let payload = try readFully(count: layer0Size)
            output.append(contentsOf: payload)
        }
        
        // Layer 1 payload
        if 0 < layer1Size {
            let payload = try readFully(count: layer1Size)
            if 1 <= maxLayer {
                output.append(contentsOf: payload)
            } else {
                droppedLayer1Bytes += layer1Size
            }
        }
        // Layer 2 payload
        if 0 < layer2Size {
            let payload = try readFully(count: layer2Size)
            if 2 <= maxLayer {
                output.append(contentsOf: payload)
            } else {
                droppedLayer2Bytes += layer2Size
            }
        }
        processedFrames += 1
    }
    
    return SplitterResult(
        data: output,
        processedFrames: processedFrames,
        droppedLayer1Bytes: droppedLayer1Bytes,
        droppedLayer2Bytes: droppedLayer2Bytes
    )
}
