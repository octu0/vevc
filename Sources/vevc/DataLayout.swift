public struct VEVCFileHeader {
    public static let magic: [UInt8] = [0x56, 0x45, 0x56, 0x43]
    public let profile: UInt8 = 0x01
    public let width: Int
    public let height: Int
    public let colorGamut: UInt8 = 0x01 // BT.709
    public let framerate: Int
    public let timescale: UInt8 = 0x00 // 1000ms
    
    public init(width: Int, height: Int, framerate: Int) {
        self.width = width
        self.height = height
        self.framerate = framerate
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
        guard readProfile == 0x01 else {
            throw DecodeError.insufficientDataContext("VEVC Profile MUST be 0x01, reading: \(readProfile)")
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
        
        offset = payloadEnd
        return VEVCFileHeader(width: w, height: h, framerate: fps)
    }
}

public struct VEVCFrameHeader {
    public enum FrameType: UInt8 {
        case pFrame = 0x00
        case copyFrame = 0x01
        case iFrame = 0x02
    }
    
    public let frameType: FrameType
    public let hasRefDir: Bool
    public let mvsSize: Int
    public let refDirSize: Int
    public let layer0Size: Int
    public let layer1Size: Int
    public let layer2Size: Int
    
    public init(frameType: FrameType, hasRefDir: Bool = false, mvsSize: Int = 0, refDirSize: Int = 0, layer0Size: Int = 0, layer1Size: Int = 0, layer2Size: Int = 0) {
        self.frameType = frameType
        self.hasRefDir = hasRefDir
        self.mvsSize = mvsSize
        self.refDirSize = refDirSize
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
    
    /// Compute payload size including derived refDirSize.
    @inline(__always)
    public var payloadSize: Int {
        if frameType == .copyFrame { return 0 }
        return mvsSize + refDirSize + layer0Size + layer1Size + layer2Size
    }
    
    @inline(__always)
    public func serialize() -> [UInt8] {
        var out = [UInt8]()
        let refDirFlag: UInt8 = if hasRefDir { 0x10 } else { 0x00 }
        let flag = frameType.rawValue | refDirFlag
        out.append(flag)
        if frameType != .copyFrame {
            appendUInt32BE(&out, UInt32(mvsSize))
            appendUInt32BE(&out, UInt32(refDirSize))
            appendUInt32BE(&out, UInt32(layer0Size))
            appendUInt32BE(&out, UInt32(layer1Size))
            appendUInt32BE(&out, UInt32(layer2Size))
        }
        return out
    }
    
    @inline(__always)
    public static func deserialize(from r: [UInt8], offset: inout Int) throws -> VEVCFrameHeader {
        guard offset < r.count else { throw BinaryError.insufficientData(message: "VEVCFrameHeader flag") }
        let flag = r[offset]
        offset += 1
        
        let frameTypeBits = flag & 0x0F
        let hasRefDir = (flag & 0x10) != 0
        
        guard let fType = FrameType(rawValue: frameTypeBits) else {
            throw BinaryError.insufficientData(message: "VEVCFrameHeader invalid frameType \(flag)")
        }
        
        if fType == .copyFrame {
            return VEVCFrameHeader(frameType: .copyFrame)
        }
        
        let mvsSize = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let refDirSize = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let layer0Size = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let layer1Size = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let layer2Size = Int(try readUInt32BEFromBytes(r, offset: &offset))
        
        if (hasRefDir && refDirSize == 0) || (!hasRefDir && refDirSize != 0) {
            throw DecodeError.invalidHeader
        }
        
        return VEVCFrameHeader(frameType: fType, hasRefDir: hasRefDir, mvsSize: mvsSize, refDirSize: refDirSize, layer0Size: layer0Size, layer1Size: layer1Size, layer2Size: layer2Size)
    }
}

/// Derive MV block count from frame dimensions.
/// MV grid is 8x8 blocks at the **Base8 (L0) resolution**, which is the LL subband after 2 DWT stages.
/// L0 dimensions: l0dx = ((dx+1)/2+1)/2, l0dy = ((dy+1)/2+1)/2
@inline(__always)
public func deriveMVCount(width: Int, height: Int) -> Int {
    let l1dx = (width + 1) / 2
    let l1dy = (height + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    let cols = (l0dx + 7) / 8
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
        aqMap: [UInt8]? = nil,
        bufY: [UInt8],
        bufCb: [UInt8],
        bufCr: [UInt8]
    ) -> [UInt8] {
        var out: [UInt8] = []
        appendUInt16BE(&out, qtYStep)
        appendUInt16BE(&out, qtCStep)
        
        if let aqMap = aqMap {
            writeVLQSize(&out, aqMap.count)
            out.append(contentsOf: aqMap)
        }
        
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
    ) throws -> (qtY: QuantizationTable, qtC: QuantizationTable, aqMapData: [UInt8]?, bufY: [UInt8], bufCb: [UInt8], bufCr: [UInt8]) {
        var offset = 0
        let qtY = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: false, layerIndex: Int(layer))
        let qtC = QuantizationTable(baseStep: Int(try readUInt16BEFromBytes(r, offset: &offset)), isChroma: true, layerIndex: Int(layer))
        
        var aqMapData: [UInt8]? = nil
        if layer == 2 {
            let aqLen = try readVLQSizeFromBytes(r, offset: &offset)
            guard (offset + aqLen) <= r.count else {
                throw DecodeError.invalidBlockDataContext("\(layerLabel) AQMap overflow: offset=\(offset) len=\(aqLen) total=\(r.count)")
            }
            if 0 < aqLen {
                aqMapData = Array(r[offset..<(offset + aqLen)])
            } else {
                aqMapData = []
            }
            offset += aqLen
        }
        
        let bufYLen = try readVLQSizeFromBytes(r, offset: &offset)
        guard (offset + bufYLen) <= r.count else {
            throw DecodeError.invalidBlockDataContext("\(layerLabel) Y overflow: offset=\(offset) len=\(bufYLen) total=\(r.count)")
        }
        let bufY = Array(r[offset..<(offset + bufYLen)])
        offset += bufYLen
        
        let bufCbLen = try readVLQSizeFromBytes(r, offset: &offset)
        guard (offset + bufCbLen) <= r.count else {
            throw DecodeError.invalidBlockDataContext("\(layerLabel) Cb overflow: offset=\(offset) len=\(bufCbLen) total=\(r.count)")
        }
        let bufCb = Array(r[offset..<(offset + bufCbLen)])
        offset += bufCbLen
        
        let bufCrLen = try readVLQSizeFromBytes(r, offset: &offset)
        guard (offset + bufCrLen) <= r.count else {
            throw DecodeError.invalidBlockDataContext("\(layerLabel) Cr overflow: offset=\(offset) len=\(bufCrLen) total=\(r.count)")
        }
        let bufCr = Array(r[offset..<(offset + bufCrLen)])
        offset += bufCrLen
        
        return (qtY, qtC, aqMapData, bufY, bufCb, bufCr)
    }
}

// MARK: - AQ Map Data Layout

@inline(__always)
public func encodeAQMap(levels: [UInt8]) -> [UInt8] {
    if levels.allSatisfy({ $0 == 2 }) {
        return []
    }
    
    var writer = BypassWriter()
    let count = levels.count
    var i = 0
    while i < count {
        let level = levels[i]
        var runLength = 1
        while i + runLength < count && levels[i + runLength] == level {
            runLength += 1
        }
        
        writer.writeBits(UInt32(level), count: 3)
        var val = UInt32(runLength - 1)
        while 128 <= val {
            writer.writeBits((val & 0x7F) | 0x80, count: 8)
            val >>= 7
        }
        writer.writeBits(val & 0x7F, count: 8)
        
        i += runLength
    }
    writer.flush()
    return writer.bytes
}

@inline(__always)
public func decodeAQMap(data: [UInt8], blockCount: Int) -> [UInt8] {
    if data.isEmpty {
        return [UInt8](repeating: 2, count: blockCount)
    }
    
    return data.withUnsafeBufferPointer { ptr -> [UInt8] in
        var reader = BypassReader(base: ptr.baseAddress!, count: ptr.count)
        var levels = [UInt8]()
        levels.reserveCapacity(blockCount)
        
        while levels.count < blockCount {
            let level = UInt8(reader.readBits(count: 3))
            
            var run: UInt32 = 0
            var shift = 0
            while true {
                let b = reader.readBits(count: 8)
                run |= (b & 0x7F) << shift
                if (b & 0x80) == 0 { break }
                shift += 7
            }
            
            let runLength = Int(run) + 1
            for _ in 0..<runLength {
                levels.append(level)
            }
        }
        return levels
    }
}
