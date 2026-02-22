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
    print("Usage: vevc-dec -i <input.vevc> [-o output_dir] [-maxLayer 0-2] [-maxFrames 1|2|4]")
    exit(1)
}

guard let inputData = try? Data(contentsOf: URL(fileURLWithPath: inputPath)) else {
    print("Failed to read \(inputPath)")
    exit(1)
}

do {
    let startTime = Date()
    let opts = vevc.DecodeOptions(maxLayer: maxLayer, maxFrames: maxFrames)
    let images = try await vevc.decode(data: Array(inputData), opts: opts)
    let elapsed = Date().timeIntervalSince(startTime)
    print(String(
        format: "Decoded %d frames in %.4fms (%.4fms/frame)",
        images.count,
        elapsed * 1000,
        elapsed * 1000 / Double(images.count),
    ))
    
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    
    for (idx, img) in images.enumerated() {
        let rgba = vevc.ycbcrToRGBA(img: img)
        let imgSize: (x:Int, y:Int) = (x:img.width, y:img.height)
        let layout: PNG.Layout = .init(format: .rgba8(palette: [], fill: nil))
        
        var packed: [PNG.RGBA<UInt8>] = []
        packed.reserveCapacity(img.width * img.height)
        let total = img.width * img.height
        for j in 0..<total {
            let offset = j * 4
            packed.append(PNG.RGBA<UInt8>(rgba[offset], rgba[offset+1], rgba[offset+2], rgba[offset+3]))
        }
        
        let ppng = PNG.Image(packing: packed, size: imgSize, layout: layout)
        let outPath = "\(outDir)/frame_\(String(format: "%04d", idx)).png"
        try ppng.compress(path: outPath)
        print("Saved \(outPath)")
    }
} catch {
    print("Failed to decode: \(error)")
    exit(1)
}