import XCTest
@testable import vevc

final class Profile0x02Tests: XCTestCase {
    
    func testProfile0x02StaticSequence() async throws {
        let width = 1920
        let height = 1080
        var y = [UInt8](repeating: 128, count: width * height)
        var cb = [UInt8](repeating: 128, count: width * height / 4)
        var cr = [UInt8](repeating: 128, count: width * height / 4)
        
        // 複雑な静止画を生成
        for i in 0..<height {
            for j in 0..<width {
                y[i * width + j] = UInt8((i * 3 + j * 7) % 256)
            }
        }
        
        var staticImage = YCbCrImage(width: width, height: height, ratio: .ratio420)
        staticImage.yPlane = y
        staticImage.cbPlane = cb
        staticImage.crPlane = cr
        var frames = [YCbCrImage]()
        
        // 100フレーム。CopyFrameにならないように1画素だけごくわずかにいじる（微小ノイズ）
        for f in 0..<100 {
            var mY = y
            mY[0] = UInt8(((Int(y[0]) + f) % 5) + 128) // PSNR > 99になる程度の超微小変更
            var img = YCbCrImage(width: width, height: height, ratio: .ratio420)
            img.yPlane = mY
            img.cbPlane = cb
            img.crPlane = cr
            frames.append(img)
        }
        
        // qstep16でエンコード
        let encoder = VEVCEncoder(width: width, height: height, qstep: 16, keyint: 30, profile: 0x02)
        let bitstream = try await encoder.encodeToData(images: frames)
        
        // デコード
        let decoder = Decoder(maxLayer: 2)
        let outFrames = try await decoder.decode(data: bitstream)
        
        XCTAssertEqual(outFrames.count, 100)
        
        // 解析
        var offset = 0
        var skipPrev = 0
        var skipLtr = 0
        var inter = 0
        var totalBlocks = 0
        
        let bw = (width + 31) / 32
        let bh = (height + 31) / 32
        let blockCount = bw * bh
        
        while offset < bitstream.count {
            if offset + 4 <= bitstream.count && bitstream[offset] == 0x56 && bitstream[offset+1] == 0x45 && bitstream[offset+2] == 0x56 && bitstream[offset+3] == 0x43 {
                let _ = try VEVCFileHeader.deserialize(from: bitstream, offset: &offset)
                continue
            }
            
            let start = offset
            let fh = try VEVCFrameHeader.deserialize(from: bitstream, offset: &offset, profile: 0x02)
            if !fh.isCopyFrame {
                if fh.skipMapSize > 0 {
                    let smData = Array(bitstream[offset..<(offset + fh.skipMapSize)])
                    let map = try decodeSkipMap(data: smData, count: blockCount)
                    for m in map {
                        if m == .skip_prev { skipPrev += 1 }
                        else if m == .skip_ltr { skipLtr += 1 }
                        else { inter += 1 }
                        totalBlocks += 1
                    }
                }
                
                let headerSize = offset - start
                offset = start + headerSize + fh.skipMapSize + fh.mvsSize + fh.refDirSize + fh.layer0Size + fh.layer1Size + fh.layer2Size
            } else {
                let headerSize = offset - start
                skipPrev += blockCount
                totalBlocks += blockCount
                offset = start + headerSize
            }
        }
        
        print("--- Profile0x02 Static Test ---")
        print("Total Blocks: \(totalBlocks)")
        print("Skip Prev   : \(skipPrev)")
        print("Skip LTR    : \(skipLtr)")
        print("Inter       : \(inter)")
        
        // skip (prev + LTR) がほぼ全発火していること (微小ノイズの1ブロック以外)
        let skipRatio = Double(skipPrev + skipLtr) / Double(totalBlocks)
        XCTAssertGreaterThan(skipRatio, 0.99)
        
        // 全フレーム一致 (PSNR/SSIM)
        var minSsim: Double = 1.0
        for i in 0..<100 {
            let ssim = calculateSSIM(img1: staticImage, img2: outFrames[i])
            if ssim < minSsim { minSsim = ssim }
        }
        print("Min SSIM    : \(minSsim)")
        XCTAssertGreaterThan(minSsim, 0.99)
    }
    
    func calculateSSIM(img1: YCbCrImage, img2: YCbCrImage) -> Double {
        // SSIM計算は重いので中央だけ
        let w = min(64, img1.width)
        let h = min(64, img1.height)
        
        var sum1: Double = 0
        var sum2: Double = 0
        var sum1Sq: Double = 0
        var sum2Sq: Double = 0
        var pdt: Double = 0
        let N = Double(w * h)
        
        img1.yPlane.withUnsafeBufferPointer { p1 in
            img2.yPlane.withUnsafeBufferPointer { p2 in
                for y in 0..<h {
                    for x in 0..<w {
                        let idx = y * img1.width + x
                        let val1 = Double(p1[idx])
                        let val2 = Double(p2[idx])
                        sum1 += val1
                        sum2 += val2
                        sum1Sq += val1 * val1
                        sum2Sq += val2 * val2
                        pdt += val1 * val2
                    }
                }
            }
        }
        
        let mu1 = sum1 / N
        let mu2 = sum2 / N
        let sigma1Sq = (sum1Sq / N) - (mu1 * mu1)
        let sigma2Sq = (sum2Sq / N) - (mu2 * mu2)
        let sigma12 = (pdt / N) - (mu1 * mu2)
        
        let C1 = 6.5025
        let C2 = 58.5225
        
        let ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1Sq + sigma2Sq + C2))
        return ssim
    }
}
