import Foundation
import PNG
import vevc

let args = CommandLine.arguments
var bitrate = 500
var positionalArgs: [String] = []
var outPath = "a.vevc"
var zeroThreshold = 0
var gopSize = 15
var sceneThreshold = 8
var isOne = false

var i = 1
while i < args.count {
    let arg = args[i]
    switch arg {
    case "-bitrate":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { bitrate = v }
            i += 1
        }
    case "-o":
        if (i + 1) < args.count {
            outPath = args[i + 1]
            i += 1
        }
    case "-zeroThreshold":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { zeroThreshold = v }
            i += 1
        }
    case "-gopSize":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { gopSize = v }
            i += 1
        }
    case "-sceneThreshold":
        if (i + 1) < args.count {
            if let v = Int(args[i + 1]) { sceneThreshold = v }
            i += 1
        }
    case "-one":
        isOne = true
    default:
        positionalArgs.append(arg)
    }
    i += 1
}

if positionalArgs.isEmpty {
    print("Usage: vevc-enc -o <output.vevc> [-bitrate <kbits>] [-zeroThreshold <threshold>] [-gopSize <frames>] [-sceneThreshold <sad>] [-one] <input1.png> [input2.png ...]")
    exit(1)
}

func readPNG(path: String) -> YCbCrImage? {
    guard let image: PNG.Image = try? .decompress(path: path) else { return nil }
    let rgba: [PNG.RGBA<UInt8>] = image.unpack(as: PNG.RGBA<UInt8>.self)
    var data = [UInt8](repeating: 0, count: rgba.count * 4)
    for j in 0..<rgba.count {
        let offset = j * 4
        data[offset + 0] = rgba[j].r
        data[offset + 1] = rgba[j].g
        data[offset + 2] = rgba[j].b
        data[offset + 3] = rgba[j].a
    }
    return vevc.rgbaToYCbCr(data: data, width: image.size.x, height: image.size.y)
}

var images: [YCbCrImage] = []
for p in positionalArgs {
    if let img = readPNG(path: p) {
        images.append(img)
    } else {
        print("Failed to read \(p)")
    }
}

do {
    let startTime = Date()
    let out: [UInt8]
    if isOne {
        out = try await vevc.encodeOne(images: images, maxbitrate: bitrate * 1000, zeroThreshold: zeroThreshold, gopSize: gopSize, sceneChangeThreshold: sceneThreshold)
    } else {
        out = try await vevc.encode(images: images, maxbitrate: bitrate * 1000, zeroThreshold: zeroThreshold, gopSize: gopSize, sceneChangeThreshold: sceneThreshold)
    }
    let elapsed = Date().timeIntervalSince(startTime)
    
    let dataSize: Int
    if images.isEmpty != true {
        let first = images[0]
        dataSize = images.count * first.width * first.height * 3
    } else {
        dataSize = 0
    }
    
    print(String(
        format:"elapse= %.4fms (%.4fms/frame) %3.2fKB -> %3.2fKB compressed %3.2f%%",
        elapsed * 1000,
        elapsed * 1000 / Double(images.count),
        Double(dataSize) / 1024.0,
        Double(out.count) / 1024.0,
        Double(out.count) / Double(dataSize) * 100.0
    ))
    
    if FileManager.default.createFile(atPath: outPath, contents: Data(out)) {
        print("Successfully encoded \(images.count) frames to \(outPath)")
    } else {
        print("Failed to write \(outPath)")
        exit(1)
    }
} catch {
    print("Failed to encode: \(error)")
    exit(1)
}