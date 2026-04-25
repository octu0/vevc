import XCTest
@testable import vevc

/// Encoder と Decoder の再構築結果がピクセル単位で一致するかを検証するテスト
final class ReconstructionDriftTests: XCTestCase {
    
    private func stats(_ arr: [Int16]) -> (min: Int16, max: Int16, mean: Double) {
        guard !arr.isEmpty else { return (0, 0, 0) }
        var mn: Int16 = .max
        var mx: Int16 = .min
        var sum: Int64 = 0
        for v in arr {
            if v < mn { mn = v }
            if mx < v { mx = v }
            sum += Int64(v)
        }
        return (mn, mx, Double(sum) / Double(arr.count))
    }
    
    func testIFrameReconstructionMatch_64x64() async throws {
        try await verifyIFrameReconstruction(width: 64, height: 64)
    }
    
    func testIFrameReconstructionMatch_640x480() async throws {
        try await verifyIFrameReconstruction(width: 640, height: 480)
    }
    
    private func verifyIFrameReconstruction(width: Int, height: Int) async throws {
        let pool = BlockViewPool()
        var img = YCbCrImage(width: width, height: height)
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        for y in 0..<height {
            for x in 0..<width {
                let base = (x + y * 2) % 256
                let noise = (x &* 2654435761 ^ y &* 2246822519) % 20
                img.yPlane[y * width + x] = UInt8(clamping: base + noise - 10)
            }
        }
        for cy in 0..<cHeight {
            for cx in 0..<cWidth {
                let idx = cy * cWidth + cx
                img.cbPlane[idx] = UInt8(clamping: 128 + (cx + cy) % 20 - 10)
                img.crPlane[idx] = UInt8(clamping: 128 + (cx - cy + 256) % 20 - 10)
            }
        }
        
        let pd = toPlaneData420(image: img, pool: BlockViewPool()).0
        let qtY = QuantizationTable(baseStep: 2)
        let qtC = QuantizationTable(baseStep: 6)
        
        let (bytes, encRecon, releaseFn) = try await encodeSpatialLayers(pd: pd, pool: pool, maxbitrate: 500 * 1024, qtY: qtY, qtC: qtC, zeroThreshold: 3, roundOffset: 0)
        defer { releaseFn() }
        
        let decImg16 = try await decodeSpatialLayers(r: bytes, pool: pool, maxLayer: 2, dx: width, dy: height, roundOffset: 0)
        let decRecon = PlaneData420(img16: decImg16)
        
        let encStats = stats(encRecon.y)
        let decStats = stats(decRecon.y)
        
        // 最初の不一致ピクセルを10個出力
        var firstDiffs: [(Int, Int16, Int16)] = []
        let count = min(encRecon.y.count, decRecon.y.count)
        for i in 0..<count {
            if encRecon.y[i] != decRecon.y[i] {
                firstDiffs.append((i, encRecon.y[i], decRecon.y[i]))
                if 10 <= firstDiffs.count { break }
            }
        }
        
        var diffCount = 0
        var maxD = 0
        var maxDIdx = -1
        for i in 0..<count {
            let d = abs(Int(encRecon.y[i]) - Int(decRecon.y[i]))
            if 0 < d { diffCount += 1 }
            if maxD < d { maxD = d; maxDIdx = i }
        }
        
        // 差が大きい上位10ピクセルを見つける
        var topDiffs: [(Int, Int16, Int16, Int)] = [] // (idx, enc, dec, diff)
        for i in 0..<count {
            let d = abs(Int(encRecon.y[i]) - Int(decRecon.y[i]))
            if 100 < d {
                topDiffs.append((i, encRecon.y[i], decRecon.y[i], d))
            }
        }
        topDiffs.sort { $1.3 < $0.3 }
        let showTop = topDiffs.prefix(10)
        
        if 0 < diffCount {
            let diffsStr = firstDiffs.map { "[\($0.0) y:\($0.0/width) x:\($0.0%width)]: enc=\($0.1) dec=\($0.2)" }.joined(separator: "\n  ")
            let topStr = showTop.map { "[\($0.0) y:\($0.0/width) x:\($0.0%width)]: enc=\($0.1) dec=\($0.2) diff=\($0.3)" }.joined(separator: "\n  ")
            XCTFail("""
                \(width)x\(height) I-Frame Y面不一致:
                Enc Y stats: min=\(encStats.min) max=\(encStats.max) mean=\(String(format: "%.1f", encStats.mean))
                Dec Y stats: min=\(decStats.min) max=\(decStats.max) mean=\(String(format: "%.1f", decStats.mean))
                diffPixels=\(diffCount)/\(count) maxDiff=\(maxD) at idx=\(maxDIdx) y:\(maxDIdx/width) x:\(maxDIdx%width)
                maxDiff pixel: enc=\(encRecon.y[maxDIdx]) dec=\(decRecon.y[maxDIdx])
                大差(>100)のピクセル数: \(topDiffs.count)
                最初の不一致:
                  \(diffsStr)
                差が大きい上位:
                  \(topStr)
            """)
        }
    }
    
    func testPFrameReconstructionMatch() async throws {
        let width = 640
        let height = 480
        
        let encoder = LayersEncodeActor(width: width, height: height, maxbitrate: 500 * 1024, framerate: 30, zeroThreshold: 3, keyint: 15, sceneChangeThreshold: 32, pool: BlockViewPool())
        let decoder = StreamingDecoderActor(width: width, height: height)
        
        var img0 = YCbCrImage(width: width, height: height)
        for y in 0..<height {
            for x in 0..<width {
                img0.yPlane[y * width + x] = UInt8(clamping: (x + y * 2) % 256)
            }
        }
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        for cy in 0..<cHeight {
            for cx in 0..<cWidth {
                img0.cbPlane[cy * cWidth + cx] = 128
                img0.crPlane[cy * cWidth + cx] = 128
            }
        }
        
        let chunk0 = try await encoder.encodeNextFrame(image: img0, isSceneChange: false)
        let dec0 = try await decoder.decodeNextFrame(chunk: chunk0)!
        
        var img1 = img0
        for y in 0..<height {
            for x in 0..<width {
                let v = Int(img0.yPlane[y * width + x]) + 3
                img1.yPlane[y * width + x] = UInt8(clamping: v)
            }
        }
        
        let chunk1 = try await encoder.encodeNextFrame(image: img1, isSceneChange: false)
        let dec1 = try await decoder.decodeNextFrame(chunk: chunk1)!
        
        let enc0psnr = calculatePSNR(original: img0, decoded: dec0)
        let enc1psnr = calculatePSNR(original: img1, decoded: dec1)
        
        XCTAssertGreaterThan(enc0psnr, 20.0, "I-Frame PSNR=\(String(format: "%.1f", enc0psnr))dB")
        XCTAssertGreaterThan(enc1psnr, 20.0, "P-Frame PSNR=\(String(format: "%.1f", enc1psnr))dB")
    }
    
    private func calculatePSNR(original: YCbCrImage, decoded: YCbCrImage) -> Double {
        let count = min(original.yPlane.count, decoded.yPlane.count)
        guard 0 < count else { return 0 }
        var mse: Double = 0
        for i in 0..<count {
            let diff = Double(Int(original.yPlane[i]) - Int(decoded.yPlane[i]))
            mse += diff * diff
        }
        mse /= Double(count)
        if mse < 0.0001 { return 100.0 }
        return 10.0 * log10(255.0 * 255.0 / mse)
    }
}
