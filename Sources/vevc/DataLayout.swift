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
        
        payload.append(contentsOf: serializeRANSModel(StaticRANSModels.shared.runModel0))
        payload.append(contentsOf: serializeRANSModel(StaticRANSModels.shared.valModel0))
        payload.append(contentsOf: serializeRANSModel(StaticRANSModels.shared.runModel1))
        payload.append(contentsOf: serializeRANSModel(StaticRANSModels.shared.valModel1))
        payload.append(contentsOf: serializeRANSModel(StaticRANSModels.shared.dpcmRunModel))
        payload.append(contentsOf: serializeRANSModel(StaticRANSModels.shared.dpcmValModel))
        
        appendUInt16BE(&out, UInt16(payload.count))
        out.append(contentsOf: payload)
        return out
    }
    
    @inline(__always)
    public static func deserialize(from chunk: [UInt8], offset: inout Int) throws -> VEVCFileHeader {
        guard chunk.count >= offset + 4, chunk[offset] == 0x56, chunk[offset + 1] == 0x45, chunk[offset + 2] == 0x56, chunk[offset + 3] == 0x43 else {
            throw DecodeError.insufficientDataContext("VEVC Magic NotFound")
        }
        offset += 4
        
        let metadataSize = Int(try readUInt16BEFromBytes(chunk, offset: &offset))
        let payloadEnd = offset + metadataSize
        guard chunk.count >= payloadEnd else {
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
        
        if offset + 1536 <= payloadEnd {
            StaticRANSModels.shared.runModel0 = deserializeRANSModel(from: chunk, offset: &offset)
            StaticRANSModels.shared.valModel0 = deserializeRANSModel(from: chunk, offset: &offset)
            StaticRANSModels.shared.runModel1 = deserializeRANSModel(from: chunk, offset: &offset)
            StaticRANSModels.shared.valModel1 = deserializeRANSModel(from: chunk, offset: &offset)
            StaticRANSModels.shared.dpcmRunModel = deserializeRANSModel(from: chunk, offset: &offset)
            StaticRANSModels.shared.dpcmValModel = deserializeRANSModel(from: chunk, offset: &offset)
        }
        
        offset = payloadEnd
        return VEVCFileHeader(width: w, height: h, framerate: fps)
    }
}

public struct VEVCGOPHeader {
    public let frameCount: Int
    
    public init(frameCount: Int) {
        self.frameCount = frameCount
    }
    
    @inline(__always)
    public func serialize() -> [UInt8] {
        var out = [UInt8]()
        appendUInt32BE(&out, UInt32(frameCount))
        return out
    }
    
    @inline(__always)
    public static func deserialize(from r: [UInt8], offset: inout Int) throws -> VEVCGOPHeader {
        let count = Int(try readUInt32BEFromBytes(r, offset: &offset))
        return VEVCGOPHeader(frameCount: count)
    }
}

public struct VEVCFrameHeader {
    public let isCopyFrame: Bool
    public let mvsCount: Int
    public let mvsSize: Int
    public let refDirSize: Int
    public let layer0Size: Int
    public let layer1Size: Int
    public let layer2Size: Int
    
    public init(isCopyFrame: Bool, mvsCount: Int = 0, mvsSize: Int = 0, refDirSize: Int = 0, layer0Size: Int = 0, layer1Size: Int = 0, layer2Size: Int = 0) {
        self.isCopyFrame = isCopyFrame
        self.mvsCount = mvsCount
        self.mvsSize = mvsSize
        self.refDirSize = refDirSize
        self.layer0Size = layer0Size
        self.layer1Size = layer1Size
        self.layer2Size = layer2Size
    }
    
    @inline(__always)
    public var payloadSize: Int {
        if isCopyFrame { return 0 }
        return mvsSize + refDirSize + layer0Size + layer1Size + layer2Size
    }
    
    @inline(__always)
    public func serialize() -> [UInt8] {
        var out = [UInt8]()
        if isCopyFrame {
            out.append(0x01)
        } else {
            out.append(0x00)
            appendUInt32BE(&out, UInt32(mvsCount))
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
        
        if flag == 0x01 {
            return VEVCFrameHeader(isCopyFrame: true)
        }
        
        let mvsCount = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let mvsSize = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let refDirSize = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let layer0Size = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let layer1Size = Int(try readUInt32BEFromBytes(r, offset: &offset))
        let layer2Size = Int(try readUInt32BEFromBytes(r, offset: &offset))
        
        return VEVCFrameHeader(isCopyFrame: false, mvsCount: mvsCount, mvsSize: mvsSize, refDirSize: refDirSize, layer0Size: layer0Size, layer1Size: layer1Size, layer2Size: layer2Size)
    }
}
