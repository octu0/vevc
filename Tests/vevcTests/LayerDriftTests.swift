import XCTest
@testable import vevc

/// レイヤ別にEncoder/Decoder再構築の差異をテスト
final class LayerDriftTests: XCTestCase {
    
    private func stats(_ arr: [Int16]) -> String {
        guard !arr.isEmpty else { return "empty" }
        var mn: Int16 = .max
        var mx: Int16 = .min
        var sum: Int64 = 0
        for v in arr {
            if v < mn { mn = v }
            if mx < v { mx = v }
            sum += Int64(v)
        }
        return "min=\(mn) max=\(mx) mean=\(String(format: "%.1f", Double(sum)/Double(arr.count)))"
    }
    
    private func diffStats(_ a: [Int16], _ b: [Int16]) -> (maxDiff: Int, diffCount: Int, count: Int) {
        let count = min(a.count, b.count)
        var maxD = 0
        var diffCount = 0
        for i in 0..<count {
            let d = abs(Int(a[i]) - Int(b[i]))
            if 0 < d { diffCount += 1 }
            if maxD < d { maxD = d }
        }
        return (maxD, diffCount, count)
    }
    
    /// Base8のみ（エンコーダ vs デコーダ）
    func testBase8Only() async throws {
        let width = 640
        let height = 480
        
        var img = YCbCrImage(width: width, height: height)
        for y in 0..<height {
            for x in 0..<width {
                img.yPlane[y * width + x] = UInt8(clamping: (x + y * 2) % 256)
            }
        }
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        for cy in 0..<cHeight {
            for cx in 0..<cWidth {
                img.cbPlane[cy * cWidth + cx] = UInt8(clamping: 128 + (cx + cy) % 20 - 10)
                img.crPlane[cy * cWidth + cx] = UInt8(clamping: 128 + (cx - cy + 256) % 20 - 10)
            }
        }
        
        let pd = toPlaneData420(images: [img])[0]
        let qtY = QuantizationTable(baseStep: 2)
        let qtC = QuantizationTable(baseStep: 6)
        
        // エンコーダ: Base8のみ
        let (bytes, encRecon) = try await encodePlaneBase8(pd: pd, predictedPd: nil, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        
        // デコーダ: Base8のみ
        let decImg = try await decodeBase8(r: bytes, layer: 0)
        
        let d = diffStats(encRecon.y, decImg.y)
        XCTAssertEqual(d.maxDiff, 0, "Base8 Y不一致: maxDiff=\(d.maxDiff) diffPixels=\(d.diffCount)/\(d.count) enc=[\(stats(encRecon.y))] dec=[\(stats(decImg.y))]")
    }
    
    /// Base8 + Layer16 再構築（Y面のみ比較）
    func testBase8PlusLayer16() async throws {
        let width = 640
        let height = 480
        
        var img = YCbCrImage(width: width, height: height)
        for y in 0..<height {
            for x in 0..<width {
                img.yPlane[y * width + x] = UInt8(clamping: (x + y * 2) % 256)
            }
        }
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        for cy in 0..<cHeight {
            for cx in 0..<cWidth {
                img.cbPlane[cy * cWidth + cx] = UInt8(clamping: 128 + (cx + cy) % 20 - 10)
                img.crPlane[cy * cWidth + cx] = UInt8(clamping: 128 + (cx - cy + 256) % 20 - 10)
            }
        }
        
        let pd = toPlaneData420(images: [img])[0]
        let qtY = QuantizationTable(baseStep: 2)
        let qtC = QuantizationTable(baseStep: 6)
        
        // encodeSpatialLayersのLayer2→Layer1→Base8 チェーンの部分実行と同等
        let (_, sub2, subPred2, _, _, _) = try await encodePlaneLayer32(pd: pd, predictedPd: nil, layer: 2, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        let (layer1, sub1, subPred1, l1yBlocks, _, _) = try await encodePlaneLayer16(pd: sub2, predictedPd: subPred2, layer: 1, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        let (layer0, baseRecon) = try await encodePlaneBase8(pd: sub1, predictedPd: subPred1, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        
        // エンコーダ再構築: Base8 → Layer16
        let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)
        let l1dx = sub2.width
        let l1dy = sub2.height
        let reconL1Y = reconstructPlaneLayer(blocks: l1yBlocks, prevImg: baseImg, planeType: 0, width: l1dx, height: l1dy, blockSize: 16, qt: qtY)
        
        // デコーダ: Base8 → Layer16
        let decBase = try await decodeBase8(r: layer0, layer: 0)
        let decL1 = try await decodeLayer16(r: layer1, layer: 1, prev: decBase)
        
        let d = diffStats(reconL1Y, decL1.y)
        XCTAssertEqual(d.maxDiff, 0, "Base8+Layer16 Y不一致: maxDiff=\(d.maxDiff) diffPixels=\(d.diffCount)/\(d.count) enc=[\(stats(reconL1Y))] dec=[\(stats(decL1.y))]")
    }
    
    /// フルチェーン: Base8 + Layer16 + Layer32
    func testFullChain() async throws {
        let width = 640
        let height = 480
        
        var img = YCbCrImage(width: width, height: height)
        for y in 0..<height {
            for x in 0..<width {
                img.yPlane[y * width + x] = UInt8(clamping: (x + y * 2) % 256)
            }
        }
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        for cy in 0..<cHeight {
            for cx in 0..<cWidth {
                img.cbPlane[cy * cWidth + cx] = UInt8(clamping: 128 + (cx + cy) % 20 - 10)
                img.crPlane[cy * cWidth + cx] = UInt8(clamping: 128 + (cx - cy + 256) % 20 - 10)
            }
        }
        
        let pd = toPlaneData420(images: [img])[0]
        let qtY = QuantizationTable(baseStep: 2)
        let qtC = QuantizationTable(baseStep: 6)
        
        let (bytes, encRecon) = try await encodeSpatialLayers(pd: pd, predictedPd: nil, maxbitrate: 500 * 1024, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        
        let decImg16 = try await decodeSpatialLayers(r: bytes, maxLayer: 2)
        
        let dY = diffStats(encRecon.y, decImg16.y)
        let dCb = diffStats(encRecon.cb, decImg16.cb)
        let dCr = diffStats(encRecon.cr, decImg16.cr)
        
        XCTAssertEqual(dY.maxDiff, 0, "Full Y不一致: maxDiff=\(dY.maxDiff) diffPixels=\(dY.diffCount)/\(dY.count)")
        XCTAssertEqual(dCb.maxDiff, 0, "Full Cb不一致: maxDiff=\(dCb.maxDiff)")
        XCTAssertEqual(dCr.maxDiff, 0, "Full Cr不一致: maxDiff=\(dCr.maxDiff)")
    }
    
    /// ノイズパターンでフルチェーン
    func testFullChainNoisePattern() async throws {
        let width = 640
        let height = 480
        
        var img = YCbCrImage(width: width, height: height)
        for y in 0..<height {
            for x in 0..<width {
                let base = (x + y * 2) % 256
                let noise = (x &* 2654435761 ^ y &* 2246822519) % 20
                img.yPlane[y * width + x] = UInt8(clamping: base + noise - 10)
            }
        }
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        for cy in 0..<cHeight {
            for cx in 0..<cWidth {
                img.cbPlane[cy * cWidth + cx] = UInt8(clamping: 128 + (cx + cy) % 20 - 10)
                img.crPlane[cy * cWidth + cx] = UInt8(clamping: 128 + (cx - cy + 256) % 20 - 10)
            }
        }
        
        let pd = toPlaneData420(images: [img])[0]
        let qtY = QuantizationTable(baseStep: 2)
        let qtC = QuantizationTable(baseStep: 6)
        
        let (bytes, encRecon) = try await encodeSpatialLayers(pd: pd, predictedPd: nil, maxbitrate: 500 * 1024, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        
        let decImg16 = try await decodeSpatialLayers(r: bytes, maxLayer: 2)
        
        let dY = diffStats(encRecon.y, decImg16.y)
        XCTAssertEqual(dY.maxDiff, 0, "NoisePattern Full Y不一致: maxDiff=\(dY.maxDiff) diffPixels=\(dY.diffCount)/\(dY.count) enc=[\(stats(encRecon.y))] dec=[\(stats(decImg16.y))]")
    }
}
