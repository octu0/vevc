import Foundation
import CoreMedia
import VideoToolbox
import vevc

public struct QualityMetrics {
    public let psnr: Double
    public let ssim: Double
}

public struct QualityStats {
    public let avgPSNR: Double
    public let minPSNR: Double
    public let maxPSNR: Double
    public let p50PSNR: Double
    public let p90PSNR: Double
    public let stddevPSNR: Double
    
    public let avgSSIM: Double
    public let minSSIM: Double
    public let maxSSIM: Double
    public let p50SSIM: Double
    public let p90SSIM: Double
    public let stddevSSIM: Double
}

public func calculateQualityStats(metrics: [QualityMetrics]) -> QualityStats? {
    guard !metrics.isEmpty else { return nil }
    
    let count = Double(metrics.count)
    var sumPsnr = 0.0
    var sumSsim = 0.0
    var minPsnr = Double.greatestFiniteMagnitude
    var maxPsnr = -Double.greatestFiniteMagnitude
    var minSsim = Double.greatestFiniteMagnitude
    var maxSsim = -Double.greatestFiniteMagnitude
    
    var psnrs = [Double]()
    var ssims = [Double]()
    psnrs.reserveCapacity(metrics.count)
    ssims.reserveCapacity(metrics.count)
    
    for m in metrics {
        sumPsnr += m.psnr
        sumSsim += m.ssim
        if m.psnr < minPsnr { minPsnr = m.psnr }
        if m.psnr > maxPsnr { maxPsnr = m.psnr }
        if m.ssim < minSsim { minSsim = m.ssim }
        if m.ssim > maxSsim { maxSsim = m.ssim }
        psnrs.append(m.psnr)
        ssims.append(m.ssim)
    }
    
    let avgPsnr = sumPsnr / count
    let avgSsim = sumSsim / count
    
    var sqSumPsnr = 0.0
    var sqSumSsim = 0.0
    for m in metrics {
        sqSumPsnr += (m.psnr - avgPsnr) * (m.psnr - avgPsnr)
        sqSumSsim += (m.ssim - avgSsim) * (m.ssim - avgSsim)
    }
    let stddevPsnr = sqrt(sqSumPsnr / count)
    let stddevSsim = sqrt(sqSumSsim / count)
    
    psnrs.sort()
    ssims.sort()
    
    let p50Index = Int(count * 0.50)
    let p90Index = Int(count * 0.90)
    
    let p50Psnr = psnrs[min(p50Index, metrics.count - 1)]
    let p90Psnr = psnrs[min(p90Index, metrics.count - 1)]
    
    let p50Ssim = ssims[min(p50Index, metrics.count - 1)]
    let p90Ssim = ssims[min(p90Index, metrics.count - 1)]
    
    return QualityStats(
        avgPSNR: avgPsnr, minPSNR: minPsnr, maxPSNR: maxPsnr, p50PSNR: p50Psnr, p90PSNR: p90Psnr, stddevPSNR: stddevPsnr,
        avgSSIM: avgSsim, minSSIM: minSsim, maxSSIM: maxSsim, p50SSIM: p50Ssim, p90SSIM: p90Ssim, stddevSSIM: stddevSsim
    )
}

@inline(__always)
public func calculatePSNR(img1: YCbCrImage, img2: YCbCrImage) -> Double {
    let w = min(img1.width, img2.width)
    let h = min(img1.height, img2.height)
    var ssd = 0
    let count = w * h
    
    img1.yPlane.withUnsafeBufferPointer { p1 in
        img2.yPlane.withUnsafeBufferPointer { p2 in
            guard let b1 = p1.baseAddress, let b2 = p2.baseAddress else { return }
            for y in 0..<h {
                let r1 = b1.advanced(by: y * img1.width)
                let r2 = b2.advanced(by: y * img2.width)
                for x in 0..<w {
                    let diff = Int(r1[x]) - Int(r2[x])
                    ssd += diff * diff
                }
            }
        }
    }
    
    if ssd == 0 { return 100.0 }
    let mse = Double(ssd) / Double(count)
    return 10.0 * log10((255.0 * 255.0) / mse)
}

@inline(__always)
public func calculatePSNR(img1: YCbCrImage, bgraBuffer buffer: CVPixelBuffer) -> Double {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    
    let w = min(img1.width, CVPixelBufferGetWidth(buffer))
    let h = min(img1.height, CVPixelBufferGetHeight(buffer))
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let baseAddr = CVPixelBufferGetBaseAddress(buffer)!
    
    var ssd = 0
    let count = w * h
    let bias = 1 << 15
    
    img1.yPlane.withUnsafeBufferPointer { p1 in
        guard let b1 = p1.baseAddress else { return }
        for y in 0..<h {
            let bgraRow = baseAddr.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            let yRow = b1.advanced(by: y * img1.width)
            for x in 0..<w {
                let off = x * 4
                let b = Int(bgraRow[off + 0])
                let g = Int(bgraRow[off + 1])
                let r = Int(bgraRow[off + 2])
                let y2 = (19595 * r + 38470 * g + 7471 * b + bias) >> 16
                let y2Clamped = y2 < 0 ? 0 : (y2 > 255 ? 255 : y2)
                
                let diff = Int(yRow[x]) - y2Clamped
                ssd += diff * diff
            }
        }
    }
    
    if ssd == 0 { return 100.0 }
    let mse = Double(ssd) / Double(count)
    return 10.0 * log10((255.0 * 255.0) / mse)
}

@inline(__always)
public func calculateSSIM(img1: YCbCrImage, img2: YCbCrImage) -> Double {
    let w = min(img1.width, img2.width)
    let h = min(img1.height, img2.height)
    
    var ssimSum: Double = 0
    var blocks = 0
    let C1: Double = 6.5025
    let C2: Double = 58.5225
    
    img1.yPlane.withUnsafeBufferPointer { p1 in
        img2.yPlane.withUnsafeBufferPointer { p2 in
            guard let b1 = p1.baseAddress, let b2 = p2.baseAddress else { return }
            for y in stride(from: 0, to: h - 7, by: 8) {
                for x in stride(from: 0, to: w - 7, by: 8) {
                    var sum1 = 0, sum2 = 0, sum1sq = 0, sum2sq = 0, sum12 = 0
                    for dy in 0..<8 {
                        let r1 = b1.advanced(by: (y + dy) * img1.width + x)
                        let r2 = b2.advanced(by: (y + dy) * img2.width + x)
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

@inline(__always)
public func calculateSSIM(img1: YCbCrImage, bgraBuffer buffer: CVPixelBuffer) -> Double {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    
    let w = min(img1.width, CVPixelBufferGetWidth(buffer))
    let h = min(img1.height, CVPixelBufferGetHeight(buffer))
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let baseAddr = CVPixelBufferGetBaseAddress(buffer)!
    
    var ssimSum: Double = 0
    var blocks = 0
    let C1: Double = 6.5025
    let C2: Double = 58.5225
    let bias = 1 << 15
    
    img1.yPlane.withUnsafeBufferPointer { p1 in
        guard let b1 = p1.baseAddress else { return }
        for y in stride(from: 0, to: h - 7, by: 8) {
            for x in stride(from: 0, to: w - 7, by: 8) {
                var sum1 = 0, sum2 = 0, sum1sq = 0, sum2sq = 0, sum12 = 0
                for dy in 0..<8 {
                    let yRow = b1.advanced(by: (y + dy) * img1.width + x)
                    let bgraRow = baseAddr.advanced(by: (y + dy) * bytesPerRow).assumingMemoryBound(to: UInt8.self).advanced(by: x * 4)
                    for dx in 0..<8 {
                        let off = dx * 4
                        let cb = Int(bgraRow[off + 0])
                        let cg = Int(bgraRow[off + 1])
                        let cr = Int(bgraRow[off + 2])
                        let y2 = (19595 * cr + 38470 * cg + 7471 * cb + bias) >> 16
                        let v2 = y2 < 0 ? 0 : (y2 > 255 ? 255 : y2)
                        
                        let v1 = Int(yRow[dx])
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
    return blocks == 0 ? 1.0 : ssimSum / Double(blocks)
}
