public struct VEVCFileHeader {
    public static let magic: [UInt8] = [0x56, 0x45, 0x56, 0x43]
    public let profile: UInt8
    public let width: Int
    public let height: Int
    public let colorGamut: UInt8 = 0x01 // BT.709
    public let framerate: Int
    public let timescale: UInt8 = 0x00 // 1000ms
    public let gop: Int
    public let temporalLayers: Int
    
    public init(width: Int, height: Int, framerate: Int, profile: UInt8 = 0x01, gop: Int = 12, temporalLayers: Int = 1) {
        self.width = width
        self.height = height
        self.framerate = framerate
        self.profile = profile
        self.gop = gop
        self.temporalLayers = temporalLayers
    }
    
    @inline(__always)
    public func serialize() -> [UInt8] {
        var out = VEVCFileHeader.magic
        var payload = [UInt8]()
        payload.append(profile)
        appendUInt16BE(&payload, UInt16(width))
        appendUInt16BE(&payload, UInt16(height))
        payload.append(colorGamut)
        appendUInt16BE(&payload, UInt16(framerate))
        payload.append(timescale)
        
        // Table Flag: 0x00 = use built-in static tables (no table data follows)
        //             0x01 = custom tables follow in compressed format (reserved for future)
        payload.append(0x00)
        
        if profile == 0x02 {
            payload.append(UInt8(clamping: gop))
            if 1 < temporalLayers {
                payload.append(UInt8(clamping: temporalLayers))
            }
        }
        
        appendUInt16BE(&out, UInt16(payload.count))
        out.append(contentsOf: payload)
        return out
    }
    
    @inline(__always)
    public static func deserialize(from chunk: [UInt8], offset: inout Int) throws -> VEVCFileHeader {
        guard offset + 4 <= chunk.count, chunk[offset] == 0x56, chunk[offset + 1] == 0x45, chunk[offset + 2] == 0x56, chunk[offset + 3] == 0x43 else {
            throw DecodeError.insufficientDataContext("VEVC Magic NotFound")
        }
        offset += 4
        
        let metadataSize = Int(try readUInt16BEFromBytes(chunk, offset: &offset))
        let payloadEnd = offset + metadataSize
        guard payloadEnd <= chunk.count else {
            throw DecodeError.insufficientDataContext("VEVC FileHeader length overflow")
        }
        
        let readProfile = chunk[offset]
        guard readProfile == 0x01 || readProfile == 0x02 else {
            throw DecodeError.insufficientDataContext("VEVC Profile MUST be 0x01 or 0x02, reading: \(readProfile)")
        }
        offset += 1
        
        let w = Int(try readUInt16BEFromBytes(chunk, offset: &offset))
        let h = Int(try readUInt16BEFromBytes(chunk, offset: &offset))
        
        _ = chunk[offset] // ColorGamut
        offset += 1
        
        let fps = Int(try readUInt16BEFromBytes(chunk, offset: &offset))
        _ = chunk[offset] // Timescale
        offset += 1
        
        guard offset < payloadEnd else {
            throw DecodeError.insufficientDataContext("VEVC FileHeader Table Flag missing")
        }
        
        let tableFlag = chunk[offset]
        offset += 1
        
        if tableFlag == 0x00 {
            // Built-in static tables: no table data to read, StaticRANSModels keeps defaults
        } else {
            throw DecodeError.insufficientDataContext("VEVC FileHeader unsupported Table Flag: \(tableFlag)")
        }
        
        let gop: Int
        let temporalLayers: Int
        if readProfile == 0x02 {
            guard offset < payloadEnd else {
                throw DecodeError.insufficientDataContext("VEVC FileHeader GOP missing for Profile 0x02")
            }
            gop = Int(chunk[offset])
            offset += 1
            if offset < payloadEnd {
                temporalLayers = Int(chunk[offset])
                offset += 1
            } else {
                temporalLayers = 1
            }
        } else {
            gop = 0
            temporalLayers = 1
        }
        
        offset = payloadEnd
        return VEVCFileHeader(width: w, height: h, framerate: fps, profile: readProfile, gop: gop, temporalLayers: temporalLayers)
    }
}

public struct VEVCFrameHeader {
    public enum FrameType: UInt8 {
        case pFrame = 0x00
        case copyFrame = 0x01
        case iFrame = 0x02
    }
    
    // Declaration order mirrors the serialized layout.
    public let frameType: FrameType
    public let hasRefDir: Bool
    public let skipMapSize: Int
    public let mvsSize: Int
    public let refDirSize: Int
    public let treeMapSize: Int
    /// Weighted-prediction offsets (profile 0x02 P-frames, mandatory
    /// fields): the decoder forms P′ = P + offset on inter blocks per plane.
    /// Serialized as two Int8 bytes placed after refDirSize and BEFORE the
    /// layer sizes, so layer-dropping readers never have to reach past them.
    /// A stream without the bytes fails to parse (profile 0x02 is in
    /// development — no backward compatibility). chromaOffset is signaled
    /// for symmetry; the current encoder always sends 0.
    public let lumaOffset: Int
    public let chromaOffset: Int
    public let hasCtxRans: Bool
    public let layer0Size: Int
    public let layer1Size: Int
    public let layer2Size: Int

    public init(frameType: FrameType, hasRefDir: Bool, hasCtxRans: Bool = false, skipMapSize: Int, mvsSize: Int, refDirSize: Int, treeMapSize: Int = 0, lumaOffset: Int = 0, chromaOffset: Int = 0, layer0Size: Int, layer1Size: Int, layer2Size: Int) {
        self.frameType = frameType
        self.hasRefDir = hasRefDir
        self.hasCtxRans = hasCtxRans
        self.skipMapSize = skipMapSize
        self.mvsSize = mvsSize
        self.refDirSize = refDirSize
        self.treeMapSize = treeMapSize
        self.lumaOffset = lumaOffset
        self.chromaOffset = chromaOffset
        self.layer0Size = layer0Size
        self.layer1Size = layer1Size
        self.layer2Size = layer2Size
    }
    
    @inline(__always)
    public var isCopyFrame: Bool {
        return frameType == .copyFrame
    }
    
    @inline(__always)
    public var isIFrame: Bool {
        return frameType == .iFrame
    }
    
    /// Compute payload size including derived refDirSize and treeMapSize.
    @inline(__always)
    public var payloadSize: Int {
        if frameType == .copyFrame { return 0 }
        return skipMapSize + mvsSize + refDirSize + treeMapSize + layer0Size + layer1Size + layer2Size
    }
    
    @inline(__always)
    public func serialize(profile: UInt8 = 0x01) -> [UInt8] {
        var out = [UInt8]()
        let refDirFlag: UInt8
        switch hasRefDir {
        case true:
            refDirFlag = 0x10
        case false:
            refDirFlag = 0x00
        }
        let ctxRansFlag: UInt8
        switch profile == 0x02 && hasCtxRans {
        case true:
            ctxRansFlag = 0x20
        case false:
            ctxRansFlag = 0x00
        }
        let flag = frameType.rawValue | refDirFlag | ctxRansFlag
        out.append(flag)
        if frameType != .copyFrame {
            if profile == 0x02 && frameType == .pFrame {
                appendUInt32BE(&out, UInt32(skipMapSize))
            }
            appendUInt32BE(&out, UInt32(mvsSize))
            appendUInt32BE(&out, UInt32(refDirSize))
            if profile == 0x02 && frameType == .pFrame {
                appendUInt32BE(&out, UInt32(treeMapSize))
                out.append(UInt8(bitPattern: Int8(clamping: lumaOffset)))
                out.append(UInt8(bitPattern: Int8(clamping: chromaOffset)))
            }
            appendUInt32BE(&out, UInt32(layer0Size))
            appendUInt32BE(&out, UInt32(layer1Size))
            appendUInt32BE(&out, UInt32(layer2Size))
        }
        return out
    }
    
    @inline(__always)
    public static func deserialize(from r: [UInt8], offset: inout Int, profile: UInt8 = 0x01) throws -> VEVCFrameHeader {
        guard offset < r.count else { throw BinaryError.insufficientData(message: "VEVCFrameHeader flag") }
        let flag = r[offset]
        offset += 1
        
        let frameTypeBits = flag & 0x0F
        let hasRefDir = (flag & 0x10) != 0
        let hasCtxRans: Bool
        switch profile == 0x02 {
        case true:
            hasCtxRans = (flag & 0x20) != 0
        case false:
            hasCtxRans = false
        }
        
        guard let fType = FrameType(rawValue: frameTypeBits) else {
            throw BinaryError.insufficientData(message: "VEVCFrameHeader invalid frameType \(flag)")
        }
        
        if fType == .copyFrame {
            return VEVCFrameHeader(frameType: .copyFrame, hasRefDir: false, hasCtxRans: false, skipMapSize: 0, mvsSize: 0, refDirSize: 0, treeMapSize: 0, lumaOffset: 0, chromaOffset: 0, layer0Size: 0, layer1Size: 0, layer2Size: 0)
        }
        
        let skipMapSize: Int
        switch profile == 0x02 && fType == .pFrame {
        case true:
            skipMapSize = Int(try readUInt32BEFromBytes(r, offset: &offset))
        case false:
            skipMapSize = 0
        }
        let mvsSize = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let refDirSize = Int(try readUInt32BEFromBytes(r, offset: &offset))
        var treeMapSize = 0
        var lumaOffset = 0
        var chromaOffset = 0
        if profile == 0x02 && fType == .pFrame {
            treeMapSize = Int(try readUInt32BEFromBytes(r, offset: &offset))
            guard offset + 1 < r.count else { throw BinaryError.insufficientData(message: "VEVCFrameHeader prediction offsets") }
            lumaOffset = Int(Int8(bitPattern: r[offset]))
            chromaOffset = Int(Int8(bitPattern: r[offset + 1]))
            offset += 2
        }
        let layer0Size = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let layer1Size = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let layer2Size = Int(try readUInt32BEFromBytes(r, offset: &offset))
        
        if profile == 0x02 && fType == .pFrame {
            if hasRefDir != true && refDirSize != 0 {
                throw DecodeError.invalidHeader
            }
        } else {
            if (hasRefDir && refDirSize == 0) || (hasRefDir != true && refDirSize != 0) {
                throw DecodeError.invalidHeader
            }
        }
        
        return VEVCFrameHeader(frameType: fType, hasRefDir: hasRefDir, hasCtxRans: hasCtxRans, skipMapSize: skipMapSize, mvsSize: mvsSize, refDirSize: refDirSize, treeMapSize: treeMapSize, lumaOffset: lumaOffset, chromaOffset: chromaOffset, layer0Size: layer0Size, layer1Size: layer1Size, layer2Size: layer2Size)
    }
}

/// Derive MV block column count from frame width.
@inline(__always)
public func deriveMVColumns(width: Int) -> Int {
    let l1dx = (width + 1) / 2
    let l0dx = (l1dx + 1) / 2
    return (l0dx + 7) / 8
}

/// Derive MV block count from frame dimensions.
/// MV grid is 8x8 blocks at the **Base8 (L0) resolution**, which is the LL subband after 2 DWT stages.
/// L0 dimensions: l0dx = ((dx+1)/2+1)/2, l0dy = ((dy+1)/2+1)/2
@inline(__always)
public func deriveMVCount(width: Int, height: Int) -> Int {
    let cols = deriveMVColumns(width: width)
    let l1dy = (height + 1) / 2
    let l0dy = (l1dy + 1) / 2
    let rows = (l0dy + 7) / 8
    return cols * rows
}

/// Layer Data structure (Section 4 of DataLayout.md).
/// Encapsulates the serialization format for each spatial layer (Layer0/Layer1/Layer2):
///   [Quantization Step Y (2B UInt16BE)]
///   [Quantization Step CbCr (2B UInt16BE)]
///   [Y Payload Size (VLQ)] [Y Payload Data]
///   [Cb Payload Size (VLQ)] [Cb Payload Data]
///   [Cr Payload Size (VLQ)] [Cr Payload Data]
public struct VEVCLayerData {
    
    /// Serialize layer data into the bitstream format.
    /// Writes quantization steps followed by VLQ-prefixed Y/Cb/Cr plane payloads.
    @inline(__always)
    static func serialize(
        qtYStep: UInt16,
        qtCStep: UInt16,
        bufY: [UInt8],
        bufCb: [UInt8],
        bufCr: [UInt8]
    ) -> [UInt8] {
        var out: [UInt8] = []
        appendUInt16BE(&out, qtYStep)
        appendUInt16BE(&out, qtCStep)
        
        writeVLQSize(&out, bufY.count)
        out.append(contentsOf: bufY)
        
        writeVLQSize(&out, bufCb.count)
        out.append(contentsOf: bufCb)
        
        writeVLQSize(&out, bufCr.count)
        out.append(contentsOf: bufCr)
        
        return out
    }
    
    /// Deserialize layer data from a byte array.
    /// Returns quantization tables and the raw byte slices for Y/Cb/Cr plane payloads.
    @inline(__always)
    static func deserialize(
        from r: [UInt8],
        layer: UInt8,
        layerLabel: String
    ) throws -> (qtY: QuantizationTable, qtC: QuantizationTable, bufY: ArraySlice<UInt8>, bufCb: ArraySlice<UInt8>, bufCr: ArraySlice<UInt8>) {
        var offset = 0
        let qtY = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: false, layerIndex: Int(layer))
        let qtC = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: true, layerIndex: Int(layer))
        
        let bufYLen = try readVLQSizeFromBytes(r, offset: &offset)
        guard (offset + bufYLen) <= r.count else {
            throw DecodeError.invalidBlockDataContext("\(layerLabel) Y overflow: offset=\(offset) len=\(bufYLen) total=\(r.count)")
        }
        let bufY = r[offset..<(offset + bufYLen)]
        offset += bufYLen
        
        let bufCbLen = try readVLQSizeFromBytes(r, offset: &offset)
        guard (offset + bufCbLen) <= r.count else {
            throw DecodeError.invalidBlockDataContext("\(layerLabel) Cb overflow: offset=\(offset) len=\(bufCbLen) total=\(r.count)")
        }
        let bufCb = r[offset..<(offset + bufCbLen)]
        offset += bufCbLen
        
        let bufCrLen = try readVLQSizeFromBytes(r, offset: &offset)
        guard (offset + bufCrLen) <= r.count else {
            throw DecodeError.invalidBlockDataContext("\(layerLabel) Cr overflow: offset=\(offset) len=\(bufCrLen) total=\(r.count)")
        }
        let bufCr = r[offset..<(offset + bufCrLen)]
        offset += bufCrLen
        
        return (qtY, qtC, bufY, bufCb, bufCr)
    }

}

public enum BlockMode: UInt8, Sendable {
    case inter = 0
    case skip_prev = 1
    case skip_ltr = 2
}

// MARK: - Stream splitting
//
// The splitter rewrites the exact structures defined above (file header,
// frame header, payload order), so it lives next to them: any layout change
// fails to compile here instead of silently truncating rebuilt headers.

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

/// Splits a VEVC bitstream in-memory, dropping layers above `maxLayer` and optionally dropping temporal layers above `maxTemporalLayer`.
///
/// - Parameters:
///   - input: Full VEVC encoded data (all 3 layers).
///   - maxLayer: Maximum layer to retain (0 = layer0 only, 1 = layer0+1, 2 = all layers).
///   - maxTemporalLayer: Maximum temporal layer to retain (0 = T0 only / 30fps, 1 = T0+T1 / all frames).
/// - Returns: A `SplitterResult` containing the stripped bitstream and statistics.
@inline(__always)
public func splitVEVCStream(input: [UInt8], maxLayer: Int, maxTemporalLayer: Int = 1) throws -> SplitterResult {
    guard 0 <= maxLayer, maxLayer <= 2 else {
        throw SplitterError.invalidMaxLayer(maxLayer)
    }
    guard 0 <= maxTemporalLayer, maxTemporalLayer <= 1 else {
        throw SplitterError.invalidMaxLayer(maxTemporalLayer)
    }
    let dropT1 = (maxTemporalLayer == 0)

    guard 4 <= input.count else {
        throw SplitterError.unexpectedEOF
    }
    guard input[0] == 0x56, input[1] == 0x45, input[2] == 0x56, input[3] == 0x43 else {
        throw SplitterError.invalidMagic
    }

    var readOffset = 0
    let fileHeader = try VEVCFileHeader.deserialize(from: input, offset: &readOffset)
    let profile = fileHeader.profile

    // Pre-allocate output buffer
    var output = [UInt8]()
    output.reserveCapacity(input.count)

    // Write FileHeader to output
    if dropT1 {
        let newGop = max(1, fileHeader.gop / 2)
        let newFps = max(1, fileHeader.framerate / 2)
        let newFileHeader = VEVCFileHeader(
            width: fileHeader.width,
            height: fileHeader.height,
            framerate: newFps,
            profile: fileHeader.profile,
            gop: newGop,
            temporalLayers: 1
        )
        output.append(contentsOf: newFileHeader.serialize())
    } else {
        let headerSlice = input[0..<readOffset]
        output.append(contentsOf: headerSlice)
    }

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
    var temporalFrameIndex = 0

    while readOffset < input.count {
        var frameOffset = readOffset
        let frameHeader = try VEVCFrameHeader.deserialize(from: input, offset: &frameOffset, profile: profile)
        readOffset = frameOffset

        let isIFrame = frameHeader.isIFrame
        if isIFrame {
            temporalFrameIndex = 0
        }
        let isT0 = (fileHeader.temporalLayers <= 1) || (temporalFrameIndex % 2 == 0)
        temporalFrameIndex += 1

        let shouldDropFrame = dropT1 && (isT0 != true)

        // CopyFrame
        if frameHeader.isCopyFrame {
            if shouldDropFrame {
                // Drop this T1 frame entirely
                continue
            }
            output.append(contentsOf: frameHeader.serialize(profile: profile))
            processedFrames += 1
            continue
        }

        // Rebuild header with trimmed layer sizes
        let newLayer1Size: Int
        switch 1 <= maxLayer {
        case true:
            newLayer1Size = frameHeader.layer1Size
        case false:
            newLayer1Size = 0
        }
        let newLayer2Size: Int
        switch 2 <= maxLayer {
        case true:
            newLayer2Size = frameHeader.layer2Size
        case false:
            newLayer2Size = 0
        }

        let newHeader = VEVCFrameHeader(
            frameType: frameHeader.frameType,
            hasRefDir: frameHeader.hasRefDir,
            skipMapSize: frameHeader.skipMapSize,
            mvsSize: frameHeader.mvsSize,
            refDirSize: frameHeader.refDirSize,
            treeMapSize: frameHeader.treeMapSize,
            lumaOffset: frameHeader.lumaOffset,
            chromaOffset: frameHeader.chromaOffset,
            layer0Size: frameHeader.layer0Size,
            layer1Size: newLayer1Size,
            layer2Size: newLayer2Size
        )

        if shouldDropFrame != true {
            output.append(contentsOf: newHeader.serialize(profile: profile))
        }

        // SkipMap payload
        if 0 < frameHeader.skipMapSize {
            let payload = try readFully(count: frameHeader.skipMapSize)
            if shouldDropFrame != true {
                output.append(contentsOf: payload)
            }
        }
        // MVs payload
        if 0 < frameHeader.mvsSize {
            let payload = try readFully(count: frameHeader.mvsSize)
            if shouldDropFrame != true {
                output.append(contentsOf: payload)
            }
        }
        // RefDir payload
        if 0 < frameHeader.refDirSize {
            let payload = try readFully(count: frameHeader.refDirSize)
            if shouldDropFrame != true {
                output.append(contentsOf: payload)
            }
        }
        // TreeMap payload
        if 0 < frameHeader.treeMapSize {
            let payload = try readFully(count: frameHeader.treeMapSize)
            if shouldDropFrame != true {
                output.append(contentsOf: payload)
            }
        }
        // Layer 0 payload (always retained)
        if 0 < frameHeader.layer0Size {
            let payload = try readFully(count: frameHeader.layer0Size)
            if shouldDropFrame != true {
                output.append(contentsOf: payload)
            }
        }

        // Layer 1 payload
        if 0 < frameHeader.layer1Size {
            let payload = try readFully(count: frameHeader.layer1Size)
            if shouldDropFrame != true {
                if 1 <= maxLayer {
                    output.append(contentsOf: payload)
                } else {
                    droppedLayer1Bytes += frameHeader.layer1Size
                }
            }
        }
        // Layer 2 payload
        if 0 < frameHeader.layer2Size {
            let payload = try readFully(count: frameHeader.layer2Size)
            if shouldDropFrame != true {
                if 2 <= maxLayer {
                    output.append(contentsOf: payload)
                } else {
                    droppedLayer2Bytes += frameHeader.layer2Size
                }
            }
        }

        if shouldDropFrame != true {
            processedFrames += 1
        }
    }

    return SplitterResult(
        data: output,
        processedFrames: processedFrames,
        droppedLayer1Bytes: droppedLayer1Bytes,
        droppedLayer2Bytes: droppedLayer2Bytes
    )
}
