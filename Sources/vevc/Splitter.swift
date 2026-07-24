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
    let fileHeader = try VEVCFileHeader.deserialize(from: input, offset: &readOffset)
    let profile = fileHeader.profile
    
    // Copy the whole FileHeader to output
    let headerSlice = input[0..<readOffset]
    
    // Pre-allocate output buffer
    var output = [UInt8]()
    output.reserveCapacity(input.count)
    
    // Write FileHeader to output
    output.append(contentsOf: headerSlice)
    
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
    
    var processedFrames = 0
    var droppedLayer1Bytes = 0
    var droppedLayer2Bytes = 0
    
    while readOffset < input.count {
        var frameOffset = readOffset
        let frameHeader = try VEVCFrameHeader.deserialize(from: input, offset: &frameOffset, profile: profile)
        readOffset = frameOffset
        
        // CopyFrame
        if frameHeader.isCopyFrame {
            output.append(contentsOf: frameHeader.serialize(profile: profile))
            processedFrames += 1
            continue
        }
        
        // Rebuild header with trimmed layer sizes
        let newLayer1Size = if 1 <= maxLayer { frameHeader.layer1Size } else { 0 }
        let newLayer2Size = if 2 <= maxLayer { frameHeader.layer2Size } else { 0 }
        
        let newHeader = VEVCFrameHeader(
            frameType: frameHeader.frameType,
            hasRefDir: frameHeader.hasRefDir,
            skipMapSize: frameHeader.skipMapSize,
            mvsSize: frameHeader.mvsSize,
            refDirSize: frameHeader.refDirSize,
            layer0Size: frameHeader.layer0Size,
            layer1Size: newLayer1Size,
            layer2Size: newLayer2Size
        )
        output.append(contentsOf: newHeader.serialize(profile: profile))
        
        // SkipMap payload
        if 0 < frameHeader.skipMapSize {
            let payload = try readFully(count: frameHeader.skipMapSize)
            output.append(contentsOf: payload)
        }
        // MVs payload
        if 0 < frameHeader.mvsSize {
            let payload = try readFully(count: frameHeader.mvsSize)
            output.append(contentsOf: payload)
        }
        // RefDir payload
        if 0 < frameHeader.refDirSize {
            let payload = try readFully(count: frameHeader.refDirSize)
            output.append(contentsOf: payload)
        }
        // Layer 0 payload (always retained)
        if 0 < frameHeader.layer0Size {
            let payload = try readFully(count: frameHeader.layer0Size)
            output.append(contentsOf: payload)
        }
        
        // Layer 1 payload
        if 0 < frameHeader.layer1Size {
            let payload = try readFully(count: frameHeader.layer1Size)
            if 1 <= maxLayer {
                output.append(contentsOf: payload)
            } else {
                droppedLayer1Bytes += frameHeader.layer1Size
            }
        }
        // Layer 2 payload
        if 0 < frameHeader.layer2Size {
            let payload = try readFully(count: frameHeader.layer2Size)
            if 2 <= maxLayer {
                output.append(contentsOf: payload)
            } else {
                droppedLayer2Bytes += frameHeader.layer2Size
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
