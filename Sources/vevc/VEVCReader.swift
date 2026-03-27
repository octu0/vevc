import Foundation

// MARK: - BlockView

struct BlockView {
    var base: UnsafeMutablePointer<Int16>
    let width: Int
    let height: Int
    let stride: Int

    init(base: UnsafeMutablePointer<Int16>, width: Int, height: Int, stride: Int) {
        self.base = base
        self.width = width
        self.height = height
        self.stride = stride
    }

    @inline(__always)
    subscript(y: Int, x: Int) -> Int16 {
        get { base[(y * stride) + x] }
        set { base[(y * stride) + x] = newValue }
    }

    @inline(__always)
    func rowPointer(y: Int) -> UnsafeMutablePointer<Int16> {
        return base.advanced(by: y * stride)
    }

    @inline(__always)
    func setRow(offsetY: Int, row: [Int16]) {
        let ptr = rowPointer(y: offsetY)
        row.withUnsafeBufferPointer { src in
            guard let srcBase = src.baseAddress else { return }
            ptr.update(from: srcBase, count: width)
        }
    }

    @inline(__always)
    func clearAll() {
        if width == stride {
            base.initialize(repeating: 0, count: width * height)
        } else {
            for y in 0..<height {
                rowPointer(y: y).initialize(repeating: 0, count: width)
            }
        }
    }
}

// MARK: - Block2D

struct Block2D: Sendable {
    var data: [Int16]
    let width: Int
    let height: Int

    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.data = [Int16](repeating: 0, count: (width * height))
    }
    
    @inline(__always)
    mutating func withView<R>(_ body: (inout BlockView) throws -> R) rethrows -> R {
        return try data.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else { 
                fatalError("Block2D buffer is empty")
            }
            var view = BlockView(base: base, width: width, height: height, stride: width)
            return try body(&view)
        }
    }
}

public enum VEVCReaderError: Error {
    case invalidHeader
    case unexpectedEOF
    case unknownMagic(String)
}

public class VEVCReader {
    public enum ColorGamut: UInt8 {
        case bt709 = 1
        case bt2020 = 2
        case unspecified = 0
    }
    
    public enum Timescale: UInt8 {
        case ms1000 = 0
        case hz90000 = 1
    }

    private let fileHandle: FileHandle
    public var width: Int = 0
    public var height: Int = 0
    public var colorGamut: ColorGamut = .bt709
    public var fps: UInt16 = 30
    public var timescale: Timescale = .ms1000
    
    public init(fileHandle: FileHandle) {
        self.fileHandle = fileHandle
    }
    
    public func readFrameChunk() throws -> [UInt8]? {
        // Read 4 bytes to determine if it's 'VEVC' magic or a GOP DataSize prefix
        let first4Bytes = readFully(count: 4)
        if first4Bytes.isEmpty { return nil } // Normal EOF
        guard first4Bytes.count == 4 else { throw VEVCReaderError.unexpectedEOF }
        
        let magicVEVC = Data([0x56, 0x45, 0x56, 0x43]) // 'VEVC'
        
        if first4Bytes == magicVEVC {
            // Read metadata: metadataSize(2B) first
            let metaSizeData = readFully(count: 2)
            guard metaSizeData.count == 2 else { throw VEVCReaderError.unexpectedEOF }
            var msOffset = 0
            let metadataSize = Int(try readUInt16BEFromBytes([UInt8](metaSizeData), offset: &msOffset))
            
            // Read metadata payload
            let metaData = readFully(count: metadataSize)
            guard metaData.count == metadataSize else { throw VEVCReaderError.unexpectedEOF }
            let metaBytes = [UInt8](metaData)
            
            // Parse Profile 1: Width(2B) + Height(2B) + ColorGamut(1B) + FPS(2B) + Timescale(1B)
            let profile = metaBytes[0]
            if profile == 0x01, metadataSize >= 9 {
                var mOffset = 1
                self.width = Int(try readUInt16BEFromBytes(metaBytes, offset: &mOffset))
                self.height = Int(try readUInt16BEFromBytes(metaBytes, offset: &mOffset))
                if let cg = ColorGamut(rawValue: metaBytes[mOffset]) {
                    self.colorGamut = cg
                }
                mOffset += 1
                self.fps = try readUInt16BEFromBytes(metaBytes, offset: &mOffset)
                if let ts = Timescale(rawValue: metaBytes[mOffset]) {
                    self.timescale = ts
                }
            }
            
            // Recursively read the next chunk (first GOP)
            return try readFrameChunk()
        }
        
        // GOP chunk: first4Bytes is actually DataSize(4B)
        // DataSize is the total size of the gopBody without the 4B prefix itself.
        var dsOffset = 0
        let dataSizeBytes = [UInt8](first4Bytes)
        let gopDataSize = Int(try readUInt32BEFromBytes(dataSizeBytes, offset: &dsOffset))
        
        // Read the entire GOP body at once using DataSize
        let gopBody = readFully(count: gopDataSize)
        guard gopBody.count == gopDataSize else { throw VEVCReaderError.unexpectedEOF }
        
        // Build complete chunk: DataSize(4B) + gopBody
        var chunk = [UInt8]()
        chunk.reserveCapacity(4 + gopDataSize)
        chunk.append(contentsOf: dataSizeBytes)
        chunk.append(contentsOf: gopBody)
        
        return chunk
    }
    
    private func readFully(count: Int) -> Data {
        var result = Data()
        var remaining = count
        while remaining > 0 {
            let data = fileHandle.readData(ofLength: remaining)
            if data.isEmpty { break }
            result.append(data)
            remaining -= data.count
        }
        return result
    }
}
