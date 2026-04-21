import Foundation
import vevc

func printUsage() {
    print("usage: vevc-splitter -i <input.vevc> -o <output.vevc> [-maxLayer 0-2]")
}

var inputPath: String? = nil
var outputPath: String? = nil
var maxLayer: Int = 1

var i = 1
while i < CommandLine.arguments.count {
    let arg = CommandLine.arguments[i]
    if arg == "-i", i + 1 < CommandLine.arguments.count {
        inputPath = CommandLine.arguments[i + 1]
        i += 2
    } else if arg == "-o", i + 1 < CommandLine.arguments.count {
        outputPath = CommandLine.arguments[i + 1]
        i += 2
    } else if arg == "-maxLayer", i + 1 < CommandLine.arguments.count {
        if let val = Int(CommandLine.arguments[i + 1]) {
            maxLayer = val
        }
        i += 2
    } else {
        i += 1
    }
}

guard let input = inputPath, let output = outputPath else {
    printUsage()
    exit(1)
}

func runSplitter(input: String, output: String, maxLayer: Int) throws {
    let fileManager = FileManager.default
    var inputHandle: FileHandle
    if input == "-" {
        inputHandle = FileHandle.standardInput
    } else {
        inputHandle = try FileHandle(forReadingFrom: URL(fileURLWithPath: input))
    }
    
    if output != "-" {
        if fileManager.fileExists(atPath: output) {
            try fileManager.removeItem(atPath: output)
        }
        fileManager.createFile(atPath: output, contents: nil, attributes: nil)
    }
    var outputHandle: FileHandle
    if output == "-" {
        outputHandle = FileHandle.standardOutput
    } else {
        outputHandle = try FileHandle(forWritingTo: URL(fileURLWithPath: output))
    }
    
    defer {
        try? inputHandle.close()
        try? outputHandle.close()
    }
    
    @inline(__always)
    func readFully(count: Int) throws -> [UInt8] {
        var buf: [UInt8] = []
        buf.reserveCapacity(count)
        while buf.count < count {
            let data = try inputHandle.read(upToCount: count - buf.count)
            guard let d = data, d.count > 0 else {
                throw NSError(domain: "vevc-splitter", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unexpected EOF"])
            }
            buf.append(contentsOf: d)
        }
        return buf
    }
    
    // 1. Read FileHeader magic
    let magicData = try readFully(count: 4)
    guard magicData == [0x56, 0x45, 0x56, 0x43] else {
        print("Invalid magic")
        exit(1)
    }
    
    // 2. Read metadataSize
    let metadataSizeData = try readFully(count: 2)
    var msOffset = 0
    let metadataSize = Int(try readUInt16BEFromBytes(metadataSizeData, offset: &msOffset))
    
    // 3. Read metadata payload
    let metadataPayload = try readFully(count: metadataSize)
    
    // Write out FileHeader identical to source
    try outputHandle.write(contentsOf: magicData)
    try outputHandle.write(contentsOf: metadataSizeData)
    try outputHandle.write(contentsOf: metadataPayload)
    
    var processedFrames = 0
    var droppedLayer1Bytes = 0
    var droppedLayer2Bytes = 0
    
    while true {
        let gopHeaderData: [UInt8]
        do {
            gopHeaderData = try readFully(count: 4) // GOPHeader is frameCount(UInt32)
        } catch {
            break // EOF expected here
        }
        
        var goOffset = 0
        let gopHeader = try VEVCGOPHeader.deserialize(from: gopHeaderData, offset: &goOffset)
        try outputHandle.write(contentsOf: gopHeaderData)
        
        for _ in 0..<gopHeader.frameCount {
            let flagData = try readFully(count: 1)
            let isCopyFrame = (flagData[0] == 0x01)
            
            if isCopyFrame {
                try outputHandle.write(contentsOf: flagData)
                processedFrames += 1
                continue
            }
            
            let sizesData = try readFully(count: 24)
            var offset = 0
            let mvsCount = Int(try readUInt32BEFromBytes(sizesData, offset: &offset))
            let mvsSize = Int(try readUInt32BEFromBytes(sizesData, offset: &offset))
            let refDirSize = Int(try readUInt32BEFromBytes(sizesData, offset: &offset))
            let layer0Size = Int(try readUInt32BEFromBytes(sizesData, offset: &offset))
            let layer1Size = Int(try readUInt32BEFromBytes(sizesData, offset: &offset))
            let layer2Size = Int(try readUInt32BEFromBytes(sizesData, offset: &offset))
            
            // Reconstruct sizes based on maxLayer
            let newLayer1Size = (maxLayer >= 1) ? layer1Size : 0
            let newLayer2Size = (maxLayer >= 2) ? layer2Size : 0
            
            let newFrameHeader = VEVCFrameHeader(
                isCopyFrame: false,
                mvsCount: mvsCount,
                mvsSize: mvsSize,
                refDirSize: refDirSize,
                layer0Size: layer0Size,
                layer1Size: newLayer1Size,
                layer2Size: newLayer2Size
            )
            
            try outputHandle.write(contentsOf: newFrameHeader.serialize())
            
            // Read all payloads & slice
            if mvsSize > 0 {
                let mvsPayload = try readFully(count: mvsSize)
                try outputHandle.write(contentsOf: mvsPayload)
            }
            if refDirSize > 0 {
                let refDirPayload = try readFully(count: refDirSize)
                try outputHandle.write(contentsOf: refDirPayload)
            }
            if layer0Size > 0 {
                let layer0Payload = try readFully(count: layer0Size)
                try outputHandle.write(contentsOf: layer0Payload)
            }
            
            if layer1Size > 0 {
                let bytes = try readFully(count: layer1Size)
                if maxLayer >= 1 {
                    try outputHandle.write(contentsOf: bytes)
                } else {
                    droppedLayer1Bytes += bytes.count
                }
            }
            if layer2Size > 0 {
                let bytes = try readFully(count: layer2Size)
                if maxLayer >= 2 {
                    try outputHandle.write(contentsOf: bytes)
                } else {
                    droppedLayer2Bytes += bytes.count
                }
            }
            processedFrames += 1
        }
    }
    
    let totalDropped = droppedLayer1Bytes + droppedLayer2Bytes
    print("------------------------------------------")
    print(" Splitter Status                          ")
    print("------------------------------------------")
    print("  Processed        : \(processedFrames) frames")
    print("  Dropped Layer 1  : \(droppedLayer1Bytes) bytes")
    print("  Dropped Layer 2  : \(droppedLayer2Bytes) bytes")
    print("  Total Dropped    : \(totalDropped) bytes")
    print("------------------------------------------")
}

do {
    try runSplitter(input: input, output: output, maxLayer: maxLayer)
} catch {
    print("Error: \(error)")
    exit(1)
}
