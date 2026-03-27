import Foundation
import PNG
import vevc

let args = CommandLine.arguments
var inputPath = ""
var outDir = ".out/"
var maxLayer = 2
var maxFrames = 4

var i = 1
while i < args.count {
    let arg = args[i]
    switch arg {
    case "-i":
        if (i + 1) < args.count {
            inputPath = args[i + 1]
            i += 1
        }
    case "-o":
        if (i + 1) < args.count {
            outDir = args[i + 1]
            i += 1
        }
    case "-maxLayer":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { maxLayer = v }
            i += 1
        }
    case "-maxFrames":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { maxFrames = v }
            i += 1
        }
    default:
        ()
    }
    i += 1
}

if inputPath.isEmpty {
    print("Usage: vevc-dec -i <input.vevc> [-o output_dir] [-maxLayer 0-2] [-maxFrames 1|2|4] [-one]")
    exit(1)
}

guard let inputData = try? Data(contentsOf: URL(fileURLWithPath: inputPath)) else {
    print("Failed to read \(inputPath)")
    exit(1)
}

do {
    let startTime = Date()
    let images: [YCbCrImage]
    let opts = vevc.DecodeOptions(maxLayer: maxLayer, maxFrames: maxFrames)
    images = try await vevc.decode(data: Array(inputData), opts: opts)
    let elapsed = Date().timeIntervalSince(startTime)
    print(String(
        format: "Decoded %d frames in %.4fms (%.4fms/frame)",
        images.count,
        elapsed * 1000,
        elapsed * 1000 / Double(images.count),
    ))
    
    let outputURL: URL = URL(fileURLWithPath: outDir).standardized

    try? FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

    for (idx, img) in images.enumerated() {
        let rgba: [UInt8] = vevc.ycbcrToRGBA(img: img)
        let imgSize: (x: Int, y: Int) = (x: img.width, y: img.height)
        let layout: PNG.Layout = .init(format: .rgba8(palette: [], fill: nil))

        var packed: [PNG.RGBA<UInt8>] = []
        packed.reserveCapacity(img.width * img.height)
        let total: Int = img.width * img.height
        for j: Int in 0..<total {
            let offset: Int = j * 4
            packed.append(PNG.RGBA<UInt8>(rgba[offset], rgba[offset + 1], rgba[offset + 2], rgba[offset + 3]))
        }

        let ppng: PNG.Image = PNG.Image(packing: packed, size: imgSize, layout: layout)
        let fileName: String = "frame_\(String(format: "%04d", idx)).png"
        let fileURL: URL = outputURL.appendingPathComponent(fileName)
        try ppng.compress(path: fileURL.path)
        print("Saved \(fileURL.path)")
    }
} catch {
    print("Failed to decode: \(error)")
    exit(1)
}