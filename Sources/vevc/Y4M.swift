import Foundation

public enum Y4MError: Error {
    case invalidHeader
    case unexpectedEOF
    case invalidFormat
}

public class Y4MReader {
    private let fileHandle: FileHandle
    public let width: Int
    public let height: Int
    public let fpsHeader: String
    
    private let frameHeader = "FRAME\n".data(using: .ascii)!

    public init(fileHandle: FileHandle) throws {
        self.fileHandle = fileHandle
        
        var headerData = Data()
        var byte: Data
        while true {
            byte = fileHandle.readData(ofLength: 1)
            guard !byte.isEmpty else { throw Y4MError.unexpectedEOF }
            headerData.append(byte)
            if byte[0] == 0x0A { break } // \n
        }
        
        guard let headerStr = String(data: headerData, encoding: .ascii),
              headerStr.hasPrefix("YUV4MPEG2 ") else {
            throw Y4MError.invalidHeader
        }
        
        let parts = headerStr.trimmingCharacters(in: .whitespacesAndNewlines).components(separatedBy: " ")
        var w = 0
        var h = 0
        var fps = "F30:1"
        for p in parts {
            if p.hasPrefix("W") { w = Int(p.dropFirst()) ?? 0 }
            else if p.hasPrefix("H") { h = Int(p.dropFirst()) ?? 0 }
            else if p.hasPrefix("F") { fps = p }
        }
        
        guard w > 0, h > 0 else { throw Y4MError.invalidFormat }
        self.width = w
        self.height = h
        self.fpsHeader = fps
    }
    
    @inline(__always)
    public func readFrame() throws -> YCbCrImage? {
        let header = fileHandle.readData(ofLength: 6)
        if header.isEmpty { return nil } // EOF
        guard header == frameHeader else { throw Y4MError.invalidFormat }
        
        let ySize = width * height
        let cSize = ((width + 1) / 2) * ((height + 1) / 2)
        let totalSize = ySize + cSize * 2
        
        let frameData = fileHandle.readData(ofLength: totalSize)
        guard frameData.count == totalSize else { throw Y4MError.unexpectedEOF }
        
        var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
        
        frameData.withUnsafeBytes { ptr in
            let base = ptr.bindMemory(to: UInt8.self).baseAddress!
            withUnsafePointers(mut: &img.yPlane, mut: &img.cbPlane, mut: &img.crPlane) { yBase, cbBase, crBase in
                yBase.update(from: base, count: ySize)
                cbBase.update(from: base.advanced(by: ySize), count: cSize)
                crBase.update(from: base.advanced(by: ySize + cSize), count: cSize)
            }
        }
        
        return img
    }
}

public class Y4MWriter {
    private let fileHandle: FileHandle
    private let frameHeader = "FRAME\n".data(using: .ascii)!

    public init(fileHandle: FileHandle, width: Int, height: Int, fpsHeader: String = "F30:1") throws {
        self.fileHandle = fileHandle
        let headerStr = "YUV4MPEG2 W\(width) H\(height) \(fpsHeader) Ip A0:0 C420mpeg2 XYSCSS=420JPEG\n"
        if let data = headerStr.data(using: .ascii) {
            fileHandle.write(data)
        }
    }
    
    @inline(__always)
    public func writeFrame(_ img: YCbCrImage) throws {
        fileHandle.write(frameHeader)
        
        guard img.ratio == .ratio420 else { throw Y4MError.invalidFormat }
        
        img.yPlane.withUnsafeBufferPointer { ptr in
            fileHandle.write(Data(buffer: ptr))
        }
        img.cbPlane.withUnsafeBufferPointer { ptr in
            fileHandle.write(Data(buffer: ptr))
        }
        img.crPlane.withUnsafeBufferPointer { ptr in
            fileHandle.write(Data(buffer: ptr))
        }
    }
}
