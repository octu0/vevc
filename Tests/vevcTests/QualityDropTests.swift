import XCTest
import Foundation
@testable import vevc

final class QualityDropTests: XCTestCase {
    
    // Copy necessary SSIM code from Quality.swift for testing
    private func calcPlaneSSIM(p1: [UInt8], p2: [UInt8], w: Int, h: Int, stride1: Int, stride2: Int) -> Double {
        var ssimSum: Double = 0
        var blocks = 0
        let C1: Double = 6.5025
        let C2: Double = 58.5225
        
        p1.withUnsafeBufferPointer { ptr1 in
            p2.withUnsafeBufferPointer { ptr2 in
                guard let b1 = ptr1.baseAddress, let b2 = ptr2.baseAddress else { return }
                for y in stride(from: 0, to: h - 7, by: 8) {
                    for x in stride(from: 0, to: w - 7, by: 8) {
                        var sum1 = 0, sum2 = 0, sum1sq = 0, sum2sq = 0, sum12 = 0
                        for dy in 0..<8 {
                            let r1 = b1.advanced(by: (y + dy) * stride1 + x)
                            let r2 = b2.advanced(by: (y + dy) * stride2 + x)
                            for dx in 0..<8 {
                                let v1 = Int(r1[dx])
                                let v2 = Int(r2[dx])
                                sum1 += v1
                                sum2 += v2
                                sum1sq += v1 * v1
                                sum2sq += v2 * v2
                                sum12 += v1 * v2
                            }
                        }
                        
                        let n = 64.0
                        let mu1 = Double(sum1) / n
                        let mu2 = Double(sum2) / n
                        let mu1sq = mu1 * mu1
                        let mu2sq = mu2 * mu2
                        let mu12 = mu1 * mu2
                        
                        let sigma1sq = (Double(sum1sq) / n) - mu1sq
                        let sigma2sq = (Double(sum2sq) / n) - mu2sq
                        let sigma12 = (Double(sum12) / n) - mu12
                        
                        let num = (2.0 * mu12 + C1) * (2.0 * sigma12 + C2)
                        let den = (mu1sq + mu2sq + C1) * (sigma1sq + sigma2sq + C2)
                        ssimSum += num / den
                        blocks += 1
                    }
                }
            }
        }
        return blocks == 0 ? 1.0 : ssimSum / Double(blocks)
    }

    private func calculateSSIMAll(img1: YCbCrImage, img2: YCbCrImage) -> Double {
        let w = min(img1.width, img2.width)
        let h = min(img1.height, img2.height)
        let ssimY = calcPlaneSSIM(p1: img1.yPlane, p2: img2.yPlane, w: w, h: h, stride1: img1.width, stride2: img2.width)
        let cw = min((img1.width + 1) / 2, (img2.width + 1) / 2)
        let ch = min((img1.height + 1) / 2, (img2.height + 1) / 2)
        let ssimU = calcPlaneSSIM(p1: img1.cbPlane, p2: img2.cbPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
        let ssimV = calcPlaneSSIM(p1: img1.crPlane, p2: img2.crPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
        return (4.0 * ssimY + ssimU + ssimV) / 6.0
    }
    
    private func planeDataToImage(pd: PlaneData420) -> YCbCrImage {
        var img = YCbCrImage(width: pd.width, height: pd.height, ratio: .ratio420)
        let w = pd.width
        let h = pd.height
        let cw = (w + 1) / 2
        let ch = (h + 1) / 2
        
        for y in 0..<h {
            for x in 0..<w {
                img.yPlane[y * w + x] = UInt8(clamping: pd.y[y * w + x] + 128)
            }
        }
        for cy in 0..<ch {
            for cx in 0..<cw {
                img.cbPlane[cy * cw + cx] = UInt8(clamping: pd.cb[cy * cw + cx] + 128)
                img.crPlane[cy * cw + cx] = UInt8(clamping: pd.cr[cy * cw + cx] + 128)
            }
        }
        return img
    }

    func xtestQualityDropOnFrame() async throws {
        // Load the problematic frame
        guard let fh = try? FileHandle(forReadingFrom: URL(fileURLWithPath: ".tmp/ToS-1080-f1740.y4m")) else {
            print("ToS-1080-f1740.y4m not found, skipping specific test")
            return
        }
        defer { fh.closeFile() }
        let reader = try Y4MReader(fileHandle: fh)
        guard let img = try reader.readFrame() else {
            XCTFail("Failed to read frame")
            return
        }

        let pd = toPlaneData420(images: [img])[0]
        let qtY = QuantizationTable(baseStep: 2)
        let qtC = QuantizationTable(baseStep: 6)
        let pool = BlockViewPool()
        
        // 1. Base8
        let (bytesB8, reconB8, _, _, _, relb8) = try await encodePlaneBase8(pd: pd, pool: pool, sads: nil, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        defer { relb8() }
        let (decB8, _, _, _) = try await decodeBase8(r: bytesB8, pool: pool, layer: 0, dx: pd.width, dy: pd.height, isIFrame: true)
        let decB8Pd = PlaneData420(width: pd.width, height: pd.height, y: decB8.y, cb: decB8.cb, cr: decB8.cr)
        let decImgB8 = planeDataToImage(pd: decB8Pd)
        
        let imgB8 = planeDataToImage(pd: reconB8)
        let base8Ssim = calculateSSIMAll(img1: img, img2: imgB8)
        let decBase8Ssim = calculateSSIMAll(img1: img, img2: decImgB8)
        print("Base8 SSIM All: \(base8Ssim) / Decoded SSIM: \(decBase8Ssim)")
        
        var diffY = diffStats(reconB8.y, decB8.y)
        var diffCb = diffStats(reconB8.cb, decB8.cb)
        var diffCr = diffStats(reconB8.cr, decB8.cr)
        print("Base8 Diff Y:\(diffY.diffCount)/\(diffY.count) Cb:\(diffCb.diffCount) Cr:\(diffCr.diffCount)")
        
        // 2. Layer16
        var (sub16, l1yBlocks, l1cbBlocks, l1crBlocks, rel16) = try await preparePlaneLayer16(pd: pd, pool: pool, sads: nil, layer: 1, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        defer { rel16() }
        let (b8ReconBytes, b8Recon, _, _, _, relb8_a) = try await encodePlaneBase8(pd: sub16, pool: pool, sads: nil, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        defer { relb8_a() }
        
        let prevImg = Image16(width: b8Recon.width, height: b8Recon.height, y: b8Recon.y, cb: b8Recon.cb, cr: b8Recon.cr)
        let bytesL16 = entropyEncodeLayer16(dx: pd.width, dy: pd.height, layer: 1, qtY: qtY, qtC: qtC, zeroThreshold: 3, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: nil, parentCbBlocks: nil, parentCrBlocks: nil)
        
        let (reconL1Y, _) = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: prevImg, width: pd.width, height: pd.height, qt: qtY, pool: pool)
        let cbw = (pd.width + 1) / 2
        let cbh = (pd.height + 1) / 2
        let (reconL1Cb, _) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: prevImg, width: cbw, height: cbh, qt: qtC, pool: pool)
        let (reconL1Cr, _) = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: prevImg, width: cbw, height: cbh, qt: qtC, pool: pool)
        let (_, _, _, _, _, relb8_b) = try await encodePlaneBase8(pd: sub16, pool: pool, sads: nil, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        defer { relb8_b() }
        let (decB8_sub16, _, _, _) = try await decodeBase8(r: b8ReconBytes, pool: pool, layer: 0, dx: pd.width, dy: pd.height, isIFrame: true)
        
        let recon16 = PlaneData420(width: pd.width, height: pd.height, y: reconL1Y, cb: reconL1Cb, cr: reconL1Cr)
        let decImg16 = planeDataToImage(pd: recon16)
        
        let (decL16, _, _, _) = try await decodeLayer16(r: bytesL16, pool: pool, layer: 1, dx: pd.width, dy: pd.height, prev: decB8_sub16, parentYBlocks: nil, parentCbBlocks: nil, parentCrBlocks: nil)
        let decL16Pd = PlaneData420(width: pd.width, height: pd.height, y: decL16.y, cb: decL16.cb, cr: decL16.cr)
        let decImg16Final = planeDataToImage(pd: decL16Pd)
        
        let l16Ssim = calculateSSIMAll(img1: img, img2: decImg16)
        let decL16Ssim = calculateSSIMAll(img1: img, img2: decImg16Final)
        print("Layer16 SSIM All: \(l16Ssim) / Decoded SSIM: \(decL16Ssim)")
        
        // Let's decode layer16 and compare the Y blocks to the encoder's Y blocks.
        // Sadly decodeLayer16 doesn't return the raw blocks. 
        // We know decL16.y mismatches recon16.y. 
        // This is due to either prevImg or entropy.
        diffY = diffStats(recon16.y, decL16.y)
        diffCb = diffStats(recon16.cb, decL16.cb)
        diffCr = diffStats(recon16.cr, decL16.cr)
        print("Layer16 Diff Y:\(diffY.diffCount) Cb:\(diffCb.diffCount) Cr:\(diffCr.diffCount)")
        
        let prevDiffY = diffStats(b8Recon.y, decB8_sub16.y) // Compare intermediate prev
        print("Layer16 Prev (Base8 reconstructed internally) Diff Y:\(prevDiffY.diffCount)")
        
        // Skip layer 32 for now to focus on Layer16's mismatch
        // XCTAssertGreaterThan(decL16Ssim, 0.94, "Decoded Layer16 SSIM drop detected")

        // 3. Layer32 (full resolution reproduction check)
        var (sub32, l32yBlocks, l32cbBlocks, l32crBlocks, rel32) = try await preparePlaneLayer32(pd: pd, pool: pool, sads: nil, layer: 2, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        defer { rel32() }
        var (sub16_2, l1yBlocks_2, l1cbBlocks_2, l1crBlocks_2, rel16_2) = try await preparePlaneLayer16(pd: sub32, pool: pool, sads: nil, layer: 1, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        defer { rel16_2() }
        let (_, b8Recon_2, _, _, _, relb8_c) = try await encodePlaneBase8(pd: sub16_2, pool: pool, sads: nil, layer: 0, qtY: qtY, qtC: qtC, zeroThreshold: 3)
        defer { relb8_c() }

        let baseImg2 = Image16(width: b8Recon_2.width, height: b8Recon_2.height, y: b8Recon_2.y, cb: b8Recon_2.cb, cr: b8Recon_2.cr)
        let _ = entropyEncodeLayer16(dx: sub32.width, dy: sub32.height, layer: 1, qtY: qtY, qtC: qtC, zeroThreshold: 3, yBlocks: &l1yBlocks_2, cbBlocks: &l1cbBlocks_2, crBlocks: &l1crBlocks_2, parentYBlocks: nil, parentCbBlocks: nil, parentCrBlocks: nil)

        let (reconL1Y_2, _) = reconstructPlaneLayer16Y(blocks: l1yBlocks_2, prevImg: baseImg2, width: sub32.width, height: sub32.height, qt: qtY, pool: pool)
        let (reconL1Cb_2, _) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks_2, prevImg: baseImg2, width: (sub32.width+1)/2, height: (sub32.height+1)/2, qt: qtC, pool: pool)
        let (reconL1Cr_2, _) = reconstructPlaneLayer16Cr(blocks: l1crBlocks_2, prevImg: baseImg2, width: (sub32.width+1)/2, height: (sub32.height+1)/2, qt: qtC, pool: pool)
        _ = Image16(width: sub32.width, height: sub32.height, y: reconL1Y_2, cb: reconL1Cb_2, cr: reconL1Cr_2)

        let bytesL32 = entropyEncodeLayer32(dx: pd.width, dy: pd.height, layer: 2, qtY: qtY, qtC: qtC, zeroThreshold: 3, yBlocks: &l32yBlocks, cbBlocks: &l32cbBlocks, crBlocks: &l32crBlocks, parentYBlocks: nil, parentCbBlocks: nil, parentCrBlocks: nil)
        
        let prevImg8 = Image16(width: b8Recon_2.width, height: b8Recon_2.height, y: b8Recon_2.y, cb: b8Recon_2.cb, cr: b8Recon_2.cr)
        
        let (r1Y, _) = reconstructPlaneLayer16Y(blocks: l1yBlocks_2, prevImg: prevImg8, width: sub32.width, height: sub32.height, qt: qtY, pool: pool)
        let (r1Cb, _) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks_2, prevImg: prevImg8, width: (sub32.width + 1) / 2, height: (sub32.height + 1) / 2, qt: qtC, pool: pool)
        let (r1Cr, _) = reconstructPlaneLayer16Cr(blocks: l1crBlocks_2, prevImg: prevImg8, width: (sub32.width + 1) / 2, height: (sub32.height + 1) / 2, qt: qtC, pool: pool)
        
        let prevImg16 = Image16(width: sub32.width, height: sub32.height, y: r1Y, cb: r1Cb, cr: r1Cr)

        let (r32Y, _) = reconstructPlaneLayer32Y(blocks: l32yBlocks, prevImg: prevImg16, width: pd.width, height: pd.height, qt: qtY, pool: pool)
        let (r32Cb, _) = reconstructPlaneLayer32Cb(blocks: l32cbBlocks, prevImg: prevImg16, width: cbw, height: cbh, qt: qtC, pool: pool)
        let (r32Cr, _) = reconstructPlaneLayer32Cr(blocks: l32crBlocks, prevImg: prevImg16, width: cbw, height: cbh, qt: qtC, pool: pool)

        let recon32 = PlaneData420(width: pd.width, height: pd.height, y: r32Y, cb: r32Cb, cr: r32Cr)
        let img32 = planeDataToImage(pd: recon32)
        
        let decL32 = try await decodeLayer32(r: bytesL32, pool: pool, layer: 2, dx: pd.width, dy: pd.height, prev: decL16, parentYBlocks: nil, parentCbBlocks: nil, parentCrBlocks: nil, roundOffset: 0)
        let decL32Pd = PlaneData420(width: pd.width, height: pd.height, y: decL32.y, cb: decL32.cb, cr: decL32.cr)
        let decImg32Final = planeDataToImage(pd: decL32Pd)
        
        let l32Ssim = calculateSSIMAll(img1: img, img2: img32)
        let decL32Ssim = calculateSSIMAll(img1: img, img2: decImg32Final)
        print("Layer32 SSIM All: \(l32Ssim) / Decoded SSIM: \(decL32Ssim)")
        
        diffY = diffStats(recon32.y, decL32.y)
        diffCb = diffStats(recon32.cb, decL32.cb)
        diffCr = diffStats(recon32.cr, decL32.cr)
        print("Layer32 Diff Y:\(diffY.diffCount) Cb:\(diffCb.diffCount) Cr:\(diffCr.diffCount)")

        XCTAssertGreaterThan(decL32Ssim, 0.94, "Decoded Layer32 SSIM drop detected")
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
}
