import Foundation
import CryptoKit
import vevc

func parseVEVCLayerSizes(bitstream: [UInt8], profile: UInt8, width: Int, height: Int) -> (l0: Int, l1: Int, l2: Int, skips: (prev: Int, ltr: Int, inter: Int, copy: Int, total: Int)) {
    var offset = 0
    var headerAndBase = 0
    var l0 = 0
    var l1 = 0
    var l2 = 0
    
    var skipPrev = 0
    var skipLtr = 0
    var inter = 0
    var copyPrev = 0
    var totalBlocks = 0
    
    let bw = (width + 31) / 32
    let bh = (height + 31) / 32
    let blockCount = bw * bh
    let syntaxModels = SyntaxContextModels()
    
    while offset < bitstream.count {
        if offset + 4 <= bitstream.count && bitstream[offset] == 0x56 && bitstream[offset+1] == 0x45 && bitstream[offset+2] == 0x56 && bitstream[offset+3] == 0x43 {
            let start = offset
            if let _ = try? VEVCFileHeader.deserialize(from: bitstream, offset: &offset) {
                headerAndBase += (offset - start)
            } else {
                break
            }
        } else {
            let start = offset
            if let fh = try? VEVCFrameHeader.deserialize(from: bitstream, offset: &offset, profile: profile) {
                let headerSize = offset - start
                headerAndBase += headerSize
                if fh.isCopyFrame != true {
                    if fh.isIFrame {
                        syntaxModels.reset()
                        inter += blockCount
                        totalBlocks += blockCount
                    } else if profile == 0x02 {
                        if 0 < fh.skipMapSize {
                            let skipMapData = Array(bitstream[offset..<(offset + fh.skipMapSize)])
                            if let map = try? decodeSkipMapContext(data: skipMapData, count: blockCount, cols: bw, state: syntaxModels) {
                                for m in map {
                                    switch m {
                                    case .skip_prev:
                                        skipPrev += 1
                                    case .skip_ltr:
                                        skipLtr += 1
                                    case .inter:
                                        inter += 1
                                    }
                                    totalBlocks += 1
                                }
                            }
                        }
                    } else if profile == 0x01 {
                        inter += blockCount
                        totalBlocks += blockCount
                    }
                    
                    headerAndBase += fh.skipMapSize + fh.mvsSize + fh.refDirSize
                    l0 += fh.layer0Size
                    l1 += fh.layer1Size
                    l2 += fh.layer2Size
                    offset += fh.payloadSize
                } else {
                    copyPrev += blockCount
                    totalBlocks += blockCount
                }
            } else {
                break
            }
        }
    }
    
    let layer0Total = headerAndBase + l0
    let layer1Total = layer0Total + l1
    let layer2Total = layer1Total + l2
    
    return (layer0Total, layer1Total, layer2Total, (skipPrev, skipLtr, inter, copyPrev, totalBlocks))
}

func splitIntoChunks(data: [UInt8], profile: UInt8) -> [[UInt8]] {
    if data.isEmpty { return [] }
    var offset = 0
    var chunks: [[UInt8]] = []
    var currentProfile = profile
    
    while offset < data.count {
        if offset + 4 <= data.count && data[offset] == 0x56 && data[offset+1] == 0x45 && data[offset+2] == 0x56 && data[offset+3] == 0x43 {
            let headerStart = offset
            if let header = try? VEVCFileHeader.deserialize(from: data, offset: &offset) {
                currentProfile = header.profile
                chunks.append(Array(data[headerStart..<offset]))
            } else { break }
        } else {
            let chunkStart = offset
            if let frameHeader = try? VEVCFrameHeader.deserialize(from: data, offset: &offset, profile: currentProfile) {
                offset += frameHeader.payloadSize
                chunks.append(Array(data[chunkStart..<offset]))
            } else { break }
        }
    }
    return chunks
}

func runVEVC(y4mPath: String, config: Config) async throws -> (
    encTime: Double,
    bitstream: [UInt8],
    sizes: (l0: Int, l1: Int, l2: Int),
    skips: (prev: Int, ltr: Int, inter: Int, copy: Int, total: Int),
    l0Dec: (time: Double, metrics: [QualityMetrics]?),
    l1Dec: (time: Double, metrics: [QualityMetrics]?),
    l2Dec: (time: Double, metrics: [QualityMetrics]?),
    width: Int,
    height: Int
) {
    print("  -> runVEVC Encoding...")
    var encTime: Double = 0
    var outBytes: [UInt8] = []
    
    let encY4M = try Y4MIterator(path: y4mPath, config: config)
    guard let firstFrame = try encY4M.next() else {
        return (0, [], (0,0,0), (0,0,0,0,0), (0,nil), (0,nil), (0,nil), 0, 0)
    }
    
    // Profile 0x02 defaults: keyint 120 as an upper bound with the quality
    // floor at alpha 2.5 placing the I frames. Resolved here rather than in the
    // argument parser because -output-graph clones the config and flips
    // `profile` afterwards, and the defaults have to follow the final profile.
    // Profile 0x01 keeps keyint 30 and no floor.
    let effKeyint = (config.profile == 0x02 && config.keyintExplicit != true) ? 120 : config.keyint
    let effIqFloor = (config.profile == 0x02 && config.iqFloorExplicit != true) ? 250 : config.iqFloor

    let vevcEncoder: VEVCEncoder
    if let qstep = config.qstep {
        vevcEncoder = VEVCEncoder(width: firstFrame.width, height: firstFrame.height, qstep: qstep, framerate: config.framerate, zeroThreshold: config.zeroThreshold, keyint: effKeyint, sceneChangeThreshold: config.sceneThreshold, profile: config.profile, skipThreshold: config.skipThreshold, gop: config.gop, l2Cadence: config.l2Cadence, l1Cadence: config.l1Cadence, l0Cadence: config.l0Cadence, motionMaskingPx: config.motionMaskingPx, smooth: config.smooth, temporalLayers: config.temporalLayers, skipModel: config.skipModel, ransContext: config.ransContext, iqFloor: effIqFloor)
    } else {
        vevcEncoder = VEVCEncoder(width: firstFrame.width, height: firstFrame.height, maxbitrate: config.bitrate * 1000, framerate: config.framerate, zeroThreshold: config.zeroThreshold, keyint: effKeyint, sceneChangeThreshold: config.sceneThreshold, profile: config.profile, skipThreshold: config.skipThreshold, gop: config.gop, l2Cadence: config.l2Cadence, l1Cadence: config.l1Cadence, l0Cadence: config.l0Cadence, motionMaskingPx: config.motionMaskingPx, smooth: config.smooth, temporalLayers: config.temporalLayers, skipModel: config.skipModel, ransContext: config.ransContext, iqFloor: effIqFloor)
    }
    
    let encStart1 = Date()
    let chunk1 = try await vevcEncoder.encode(image: firstFrame.vevcImage)
    encTime += Date().timeIntervalSince(encStart1)
    outBytes.append(contentsOf: chunk1)
    
    while let imgInput = try encY4M.next() {
        let encStart = Date()
        let chunk = try await vevcEncoder.encode(image: imgInput.vevcImage)
        encTime += Date().timeIntervalSince(encStart)
        outBytes.append(contentsOf: chunk)
    }
    print("  -> runVEVC Encoded \(outBytes.count) bytes")
    
    let res = parseVEVCLayerSizes(bitstream: outBytes, profile: config.profile, width: firstFrame.width, height: firstFrame.height)
    let sizes = (l0: res.l0, l1: res.l1, l2: res.l2)
    let skips = res.skips
    
    let chunks = splitIntoChunks(data: outBytes, profile: config.profile)
    
    func decodeSpeed(maxLayer: Int) async throws -> Double {
        print("  -> runVEVC Decoding Layer \(maxLayer)...")
        let decoder = Decoder(maxLayer: maxLayer)
        let stream = AsyncStream<[UInt8]> { continuation in
            for c in chunks { continuation.yield(c) }
            continuation.finish()
        }
        let decStart = Date()
        var count = 0
        for try await _ in decoder.decodeStream(stream: stream) {
            count += 1
        }
        let decTime = Date().timeIntervalSince(decStart)
        print("  -> runVEVC Decoded Layer \(maxLayer) \(count) frames")
        return decTime
    }
    
    let l2DecTime = try await decodeSpeed(maxLayer: 2)
    let l1DecTime = try await decodeSpeed(maxLayer: 1)
    let l0DecTime = try await decodeSpeed(maxLayer: 0)
    
    var metrics: [QualityMetrics]? = nil
    if config.quality {
        print("  -> runVEVC Quality Pass (Layer 2)...")
        let qY4M = try Y4MIterator(path: y4mPath, config: config)
        let decoder = Decoder(maxLayer: 2)
        let stream = AsyncStream<[UInt8]> { continuation in
            for c in chunks { continuation.yield(c) }
            continuation.finish()
        }
        
        var mets = [QualityMetrics]()
        var perFrameCSV = "frame,psnr,ssim,ssimY\n"
        var frameIdx = 0
        for try await decodedImg in decoder.decodeStream(stream: stream) {
            guard let orig = try qY4M.next() else { break }
            let psnr = calculatePSNR(img1: orig.vevcImage, img2: decodedImg)
            let ssim = calculateSSIM(img1: orig.vevcImage, img2: decodedImg)
            mets.append(QualityMetrics(psnr: psnr, ssim: ssim))
            if config.dumpFrameMetrics != nil {
                let ssimY = calculateSSIMLuma(img1: orig.vevcImage, img2: decodedImg)
                perFrameCSV += "\(frameIdx),\(psnr),\(ssim),\(ssimY)\n"
            }
            frameIdx += 1
        }
        metrics = mets
        if let dumpPath = config.dumpFrameMetrics {
            try? perFrameCSV.write(toFile: dumpPath, atomically: true, encoding: .utf8)
            print("  -> per-frame metrics written to \(dumpPath)")
        }
    }
    
    if config.dumpHash {
        let encHash = SHA256.hash(data: outBytes).map { String(format: "%02x", $0) }.joined()
        var hasher = SHA256()
        let decoder = Decoder(maxLayer: 2)
        let stream = AsyncStream<[UInt8]> { continuation in
            for c in chunks { continuation.yield(c) }
            continuation.finish()
        }
        for try await frame in decoder.decodeStream(stream: stream) {
            frame.yPlane.withUnsafeBufferPointer { hasher.update(bufferPointer: UnsafeRawBufferPointer($0)) }
            frame.cbPlane.withUnsafeBufferPointer { hasher.update(bufferPointer: UnsafeRawBufferPointer($0)) }
            frame.crPlane.withUnsafeBufferPointer { hasher.update(bufferPointer: UnsafeRawBufferPointer($0)) }
        }
        let decHash = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        print("[FIXTURE_HASH] Encoded SHA-256: \(encHash)")
        print("[FIXTURE_HASH] Decoded SHA-256: \(decHash)")
    }
    
    return (encTime, outBytes, sizes, skips, (l0DecTime, nil), (l1DecTime, nil), (l2DecTime, metrics), firstFrame.width, firstFrame.height)
}

func extractVEVCFrames(bitstream: [UInt8], config: Config, indices: Set<Int>) async throws -> [Int: YCbCrImage] {
    let vevcDecoder = Decoder(maxLayer: config.maxLayer)
    let chunks = splitIntoChunks(data: bitstream, profile: config.profile)
    let stream = AsyncStream<[UInt8]> { continuation in
        for c in chunks { continuation.yield(c) }
        continuation.finish()
    }
    
    var extracted: [Int: YCbCrImage] = [:]
    var i = 0
    for try await frame in vevcDecoder.decodeStream(stream: stream) {
        if indices.contains(i) {
            extracted[i] = frame
        }
        i += 1
    }
    return extracted
}
