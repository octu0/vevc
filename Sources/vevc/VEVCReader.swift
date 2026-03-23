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
    public var fpsHeader: String? = nil
    
    public init(fileHandle: FileHandle) {
        self.fileHandle = fileHandle
    }
    
    public func readFrameChunk() throws -> [UInt8]? {
        let magicData = readFully(count: 4)
        if magicData.isEmpty { return nil } // Normal EOF
        guard magicData.count == 4 else { throw VEVCReaderError.unexpectedEOF }
        
        var chunk = [UInt8]()
        chunk.reserveCapacity(1024 * 16)
        chunk.append(contentsOf: magicData)
        
        switch magicData {
        case Data([0x56, 0x45, 0x56, 0x49]), Data([0x56, 0x45, 0x4F, 0x49]): // VEVI, VEOI
            let lenData = readFully(count: 4)
            guard lenData.count == 4 else { throw VEVCReaderError.unexpectedEOF }
            chunk.append(contentsOf: lenData)
            
            let len = Int((UInt32(lenData[0]) << 24) | (UInt32(lenData[1]) << 16) | (UInt32(lenData[2]) << 8) | UInt32(lenData[3]))
            
            guard len >= 0 else { throw VEVCReaderError.invalidHeader }
            let payload = readFully(count: len)
            guard payload.count == len else { throw VEVCReaderError.unexpectedEOF }
            chunk.append(contentsOf: payload)
            
            return chunk
            
        case Data([0x56, 0x45, 0x56, 0x50]), Data([0x56, 0x45, 0x4F, 0x50]): // VEVP, VEOP
            let headerData = readFully(count: 8) // mvsCount(4) + mvDataLen(4)
            guard headerData.count == 8 else { throw VEVCReaderError.unexpectedEOF }
            chunk.append(contentsOf: headerData)
            
            let mvDataLen = Int((UInt32(headerData[4]) << 24) | (UInt32(headerData[5]) << 16) | (UInt32(headerData[6]) << 8) | UInt32(headerData[7]))
            
            if mvDataLen > 0 {
                let mvData = readFully(count: mvDataLen)
                guard mvData.count == mvDataLen else { throw VEVCReaderError.unexpectedEOF }
                chunk.append(contentsOf: mvData)
            }
            
            let lenData = readFully(count: 4)
            guard lenData.count == 4 else { throw VEVCReaderError.unexpectedEOF }
            chunk.append(contentsOf: lenData)
            
            let len = Int((UInt32(lenData[0]) << 24) | (UInt32(lenData[1]) << 16) | (UInt32(lenData[2]) << 8) | UInt32(lenData[3]))
            
            guard len >= 0 else { throw VEVCReaderError.invalidHeader }
            let payload = readFully(count: len)
            guard payload.count == len else { throw VEVCReaderError.unexpectedEOF }
            chunk.append(contentsOf: payload)
            
            return chunk
            
        case Data([0x56, 0x45, 0x56, 0x48]): // VEVH
            let lenData = readFully(count: 4)
            guard lenData.count == 4 else { throw VEVCReaderError.unexpectedEOF }
            
            let len = Int((UInt32(lenData[0]) << 24) | (UInt32(lenData[1]) << 16) | (UInt32(lenData[2]) << 8) | UInt32(lenData[3]))
            guard len >= 0 else { throw VEVCReaderError.invalidHeader }
            
            if len > 0 {
                let payload = readFully(count: len)
                guard payload.count == len else { throw VEVCReaderError.unexpectedEOF }
                self.fpsHeader = String(data: payload, encoding: .utf8)
            }
            
            // recursively read the next chunk (which should be the first frame)
            return try readFrameChunk()
            
        default:
            let magicString = magicData.map { String(format: "%02x", $0) }.joined()
            throw VEVCReaderError.unknownMagic(magicString)
        }
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
