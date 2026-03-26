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
    private let fileHandle: FileHandle
    public var dataSize: UInt32 = 0
    public var width: Int = 0
    public var height: Int = 0
    public var colorGamut: UInt8 = 1  // 1=BT.709, 2=BT.2020
    public var fps: UInt16 = 30
    public var timescale: UInt8 = 0   // 0=1000ms, 1=90000hz
    
    public init(fileHandle: FileHandle) {
        self.fileHandle = fileHandle
    }
    
    public func readFrameChunk() throws -> [UInt8]? {
        // Peek first byte to determine chunk type
        let firstByte = readFully(count: 1)
        if firstByte.isEmpty { return nil } // Normal EOF
        guard firstByte.count == 1 else { throw VEVCReaderError.unexpectedEOF }
        
        let byte0 = firstByte[firstByte.startIndex]
        
        // VEVC file header starts with 'V' (0x56)
        if byte0 == 0x56 {
            // Read remaining 3 bytes of magic + 4 bytes dataSize = 7 bytes
            let restHeader = readFully(count: 7)
            guard restHeader.count == 7 else { throw VEVCReaderError.unexpectedEOF }
            
            let magic = Data([byte0]) + restHeader.prefix(3)
            guard magic == Data([0x56, 0x45, 0x56, 0x43]) else {
                let magicString = magic.map { String(format: "%02x", $0) }.joined()
                throw VEVCReaderError.unknownMagic(magicString)
            }
            
            // DataSize (4B) at restHeader[3..<7]
            var dsOffset = 3
            self.dataSize = try readUInt32BEFromBytes([UInt8](restHeader), offset: &dsOffset)
            
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
                self.colorGamut = metaBytes[mOffset]
                mOffset += 1
                self.fps = try readUInt16BEFromBytes(metaBytes, offset: &mOffset)
                self.timescale = metaBytes[mOffset]
            }
            
            // Recursively read the next chunk (first GOP)
            return try readFrameChunk()
        }
        
        // GOP chunk: Mode(1B) + GOPSize(4B) + nLow(2B) + frames
        guard byte0 == 0x00 || byte0 == 0x01 else {
            throw VEVCReaderError.unknownMagic(String(format: "%02x", byte0))
        }
        
        var chunk = [UInt8]()
        chunk.reserveCapacity(1024 * 16)
        chunk.append(byte0) // Mode
        
        // Read GOPSize(4B) + nLow(2B) = 6 bytes
        let headerData = readFully(count: 6)
        guard headerData.count == 6 else { throw VEVCReaderError.unexpectedEOF }
        chunk.append(contentsOf: headerData)
        
        var headerOffset = 0
        let gopSize = Int(try readUInt32BEFromBytes([UInt8](headerData), offset: &headerOffset))
        // nLow is read but not needed for framing
        
        // Read all frame payloads
        for _ in 0..<gopSize {
            let frameLenData = readFully(count: 4)
            guard frameLenData.count == 4 else { throw VEVCReaderError.unexpectedEOF }
            chunk.append(contentsOf: frameLenData)
            
            var frameLenOffset = 0
            let frameLen = Int(try readUInt32BEFromBytes([UInt8](frameLenData), offset: &frameLenOffset))
            guard frameLen >= 0 else { throw VEVCReaderError.invalidHeader }
            
            let framePayload = readFully(count: frameLen)
            guard framePayload.count == frameLen else { throw VEVCReaderError.unexpectedEOF }
            chunk.append(contentsOf: framePayload)
        }
        
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
