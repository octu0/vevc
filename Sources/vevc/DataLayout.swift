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
    public let layer0Size: Int
    public let layer1Size: Int
    public let layer2Size: Int
    
    public init(frameType: FrameType, hasRefDir: Bool = false, mvsSize: Int = 0, layer0Size: Int = 0, layer1Size: Int = 0, layer2Size: Int = 0) {
        self.frameType = frameType
        self.hasRefDir = hasRefDir
        self.mvsSize = mvsSize
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
    /// width/height are needed to derive mvsCount → refDirSize.
    @inline(__always)
    public func payloadSize(width: Int, height: Int) -> Int {
        if frameType == .copyFrame { return 0 }
        let refDirBytes = if hasRefDir { (deriveMVCount(width: width, height: height) + 7) / 8 } else { 0 }
        return mvsSize + refDirBytes + layer0Size + layer1Size + layer2Size
    }
    
    @inline(__always)
    public func serialize() -> [UInt8] {
        var out = [UInt8]()
        let flag = frameType.rawValue | (hasRefDir ? 0x10 : 0x00)
        out.append(flag)
        if frameType != .copyFrame {
            appendUInt32BE(&out, UInt32(mvsSize))
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
        let layer0Size = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let layer1Size = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let layer2Size = Int(try readUInt32BEFromBytes(r, offset: &offset))
        
        return VEVCFrameHeader(frameType: fType, hasRefDir: hasRefDir, mvsSize: mvsSize, layer0Size: layer0Size, layer1Size: layer1Size, layer2Size: layer2Size)
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
