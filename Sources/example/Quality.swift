import Foundation
import CoreMedia
import VideoToolbox
import vevc

public struct QualityMetrics: Sendable {
    public let psnr: Double
    public let ssim: Double
}

public struct QualityStats: Sendable {
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
    
    let psnrY = calcPlanePSNR(p1: img1.yPlane, p2: img2.yPlane, w: w, h: h, stride1: img1.width, stride2: img2.width)
    
    let cw = min((img1.width + 1) / 2, (img2.width + 1) / 2)
    let ch = min((img1.height + 1) / 2, (img2.height + 1) / 2)
    let psnrU = calcPlanePSNR(p1: img1.cbPlane, p2: img2.cbPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
    let psnrV = calcPlanePSNR(p1: img1.crPlane, p2: img2.crPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
    
    return (4.0 * psnrY + psnrU + psnrV) / 6.0
}

@inline(__always)
private func calcPlanePSNR(p1: [UInt8], p2: [UInt8], w: Int, h: Int, stride1: Int, stride2: Int) -> Double {
    var ssd = 0
    let count = w * h
    if count == 0 { return 100.0 }
    
    p1.withUnsafeBufferPointer { ptr1 in
        p2.withUnsafeBufferPointer { ptr2 in
            guard let b1 = ptr1.baseAddress, let b2 = ptr2.baseAddress else { return }
            for y in 0..<h {
                let r1 = b1.advanced(by: y * stride1)
                let r2 = b2.advanced(by: y * stride2)
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
    let w = min(img1.width, CVPixelBufferGetWidth(buffer))
    let h = min(img1.height, CVPixelBufferGetHeight(buffer))
    let img2 = createYCbCrImage(from: buffer, width: w, height: h)
    return calculatePSNR(img1: img1, img2: img2)
}

@inline(__always)
public func calculateSSIM(img1: YCbCrImage, img2: YCbCrImage) -> Double {
    let w = min(img1.width, img2.width)
    let h = min(img1.height, img2.height)
    
    let ssimY = calcPlaneSSIM(p1: img1.yPlane, p2: img2.yPlane, w: w, h: h, stride1: img1.width, stride2: img2.width)
    
    let cw = min((img1.width + 1) / 2, (img2.width + 1) / 2)
    let ch = min((img1.height + 1) / 2, (img2.height + 1) / 2)
    let ssimU = calcPlaneSSIM(p1: img1.cbPlane, p2: img2.cbPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
    let ssimV = calcPlaneSSIM(p1: img1.crPlane, p2: img2.crPlane, w: cw, h: ch, stride1: (img1.width + 1) / 2, stride2: (img2.width + 1) / 2)
    
    return (4.0 * ssimY + ssimU + ssimV) / 6.0
}

@inline(__always)
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

@inline(__always)
public func calculateSSIM(img1: YCbCrImage, bgraBuffer buffer: CVPixelBuffer) -> Double {
    let w = min(img1.width, CVPixelBufferGetWidth(buffer))
    let h = min(img1.height, CVPixelBufferGetHeight(buffer))
    let img2 = createYCbCrImage(from: buffer, width: w, height: h)
    return calculateSSIM(img1: img1, img2: img2)
}

@inline(__always)
public func createYCbCrImage(from buffer: CVPixelBuffer, width: Int, height: Int) -> YCbCrImage {
    var ycbcr = YCbCrImage(width: width, height: height, ratio: .ratio420)
    
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    
    let format = CVPixelBufferGetPixelFormatType(buffer)
    let isBiPlanar = CVPixelBufferIsPlanar(buffer)
    
    if isBiPlanar {
        // Y Plane
        if let ySrc = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) {
            let srcStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
            ycbcr.yPlane.withUnsafeMutableBufferPointer { yDest in
                guard let destBase = yDest.baseAddress else { return }
                for y in 0..<height {
                    memcpy(destBase.advanced(by: y * width), ySrc.advanced(by: y * srcStride), width)
                }
            }
        }
        
        // UV Plane
        if let uvSrc = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) {
            let srcStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
            let cWidth = (width + 1) / 2
            let cHeight = (height + 1) / 2
            
            ycbcr.cbPlane.withUnsafeMutableBufferPointer { cbDest in
                ycbcr.crPlane.withUnsafeMutableBufferPointer { crDest in
                    guard let cbBase = cbDest.baseAddress, let crBase = crDest.baseAddress else { return }
                    
                    for y in 0..<cHeight {
                        let srcRow = uvSrc.advanced(by: y * srcStride).assumingMemoryBound(to: UInt8.self)
                        let cbRow = cbBase.advanced(by: y * cWidth)
                        let crRow = crBase.advanced(by: y * cWidth)
                        
                        for x in 0..<cWidth {
                            cbRow[x] = srcRow[x * 2 + 0]
                            crRow[x] = srcRow[x * 2 + 1]
                        }
                    }
                }
            }
        }
    } else {
        guard format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32RGBA || format == kCVPixelFormatType_32ARGB || format == kCVPixelFormatType_32ABGR else {
            let p1 = String(UnicodeScalar((format >> 24) & 255) ?? "?")
            let p2 = String(UnicodeScalar((format >> 16) & 255) ?? "?")
            let p3 = String(UnicodeScalar((format >> 8) & 255) ?? "?")
            let p4 = String(UnicodeScalar(format & 255) ?? "?")
            fatalError("Unsupported format for createYCbCrImage: \(p1)\(p2)\(p3)\(p4) (\(format))")
        }
        
        // Fallback for BGRA
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let baseAddr = CVPixelBufferGetBaseAddress(buffer)!
        let biasY = 1 << 15
        let biasC = 1 << 15
        
        let actualWidth = min(width, CVPixelBufferGetWidth(buffer))
        let actualHeight = min(height, CVPixelBufferGetHeight(buffer))
        
        ycbcr.yPlane.withUnsafeMutableBufferPointer { yPtr in
            ycbcr.cbPlane.withUnsafeMutableBufferPointer { cbPtr in
                ycbcr.crPlane.withUnsafeMutableBufferPointer { crPtr in
                    guard let yBase = yPtr.baseAddress, let cbBase = cbPtr.baseAddress, let crBase = crPtr.baseAddress else { return }
                    
                    let strideY = width
                    let strideC = (width + 1) / 2
                    
                    for y in 0..<actualHeight {
                        let bgraRow = baseAddr.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
                        let yRow = yBase.advanced(by: y * strideY)
                        
                        for x in 0..<actualWidth {
                            let off = x * 4
                            let b = Int(bgraRow[off + 0])
                            let g = Int(bgraRow[off + 1])
                            let r = Int(bgraRow[off + 2])
                            
                            let y2 = (19595 * r + 38470 * g + 7471 * b + biasY) >> 16
                            yRow[x] = UInt8(clamping: y2)
                            
                            if x % 2 == 0 && y % 2 == 0 {
                                let cb2 = ((-11059 * r - 21709 * g + 32768 * b + biasC) >> 16) + 128
                                let cr2 = ((32768 * r - 27439 * g - 5329 * b + biasC) >> 16) + 128
                                
                                cbBase[(y / 2) * strideC + (x / 2)] = UInt8(clamping: cb2)
                                crBase[(y / 2) * strideC + (x / 2)] = UInt8(clamping: cr2)
                            }
                        }
                    }
                }
            }
        }
    }
    
    return ycbcr
}
