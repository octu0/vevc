import Foundation
import vevc

/// CLI wrapper that performs VEVC stream splitting via file I/O.
/// Core logic is provided by `splitVEVCStream` from the vevc library.
func runSplitter(input: String, output: String, maxLayer: Int) throws {
    // Read input data
    let inputData: [UInt8]
    if input == "-" {
        var buf = [UInt8]()
        while let chunk = try FileHandle.standardInput.read(upToCount: 65536) {
            if chunk.isEmpty { break }
            buf.append(contentsOf: chunk)
        }
        inputData = buf
    } else {
        let data = try Data(contentsOf: URL(fileURLWithPath: input))
        inputData = [UInt8](data)
    }

    // Split using the library API
    let result = try splitVEVCStream(input: inputData, maxLayer: maxLayer)

    // Write output data
    if output == "-" {
        FileHandle.standardOutput.write(Data(result.data))
    } else {
        try Data(result.data).write(to: URL(fileURLWithPath: output))
    }

    print("------------------------------------------")
    print(" Splitter Status                          ")
    print("------------------------------------------")
    print("  Processed        : \(result.processedFrames) frames")
    print("  Dropped Layer 1  : \(result.droppedLayer1Bytes) bytes")
    print("  Dropped Layer 2  : \(result.droppedLayer2Bytes) bytes")
    print("  Total Dropped    : \(result.totalDroppedBytes) bytes")
    print("------------------------------------------")
}
