import Foundation
import Accelerate
import CoreVideo
import CoreMedia
import vevc

/// Resize YCbCrImage (4:2:0) to 720p (1280x720).
public func resizeYCbCrImage720p(image: YCbCrImage) -> YCbCrImage {
    let targetWidth = 1280
    let targetHeight = 720
    
    if image.width == targetWidth && image.height == targetHeight && image.ratio == .ratio420 {
        return image
    }
    
    var resized = YCbCrImage(width: targetWidth, height: targetHeight, ratio: .ratio420, fps: image.fps)
    
    // Y Plane: (image.width x image.height) -> (1280 x 720)
    image.yPlane.withUnsafeBufferPointer { srcPtr in
        guard let srcBase = srcPtr.baseAddress else { return }
        var srcYBuffer = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: srcBase),
            height: vImagePixelCount(image.height),
            width: vImagePixelCount(image.width),
            rowBytes: image.yStride
        )
        resized.yPlane.withUnsafeMutableBufferPointer { dstPtr in
            guard let dstBase = dstPtr.baseAddress else { return }
            var dstYBuffer = vImage_Buffer(
                data: dstBase,
                height: vImagePixelCount(targetHeight),
                width: vImagePixelCount(targetWidth),
                rowBytes: targetWidth
            )
            _ = vImageScale_Planar8(&srcYBuffer, &dstYBuffer, nil, vImage_Flags(kvImageHighQualityResampling))
        }
    }
    
    // Cb & Cr Planes: 4:2:0 chroma
    let srcCWidth = (image.width + 1) / 2
    let srcCHeight = (image.height + 1) / 2
    let targetCWidth = (targetWidth + 1) / 2 // 640
    let targetCHeight = (targetHeight + 1) / 2 // 360
    
    image.cbPlane.withUnsafeBufferPointer { srcPtr in
        guard let srcBase = srcPtr.baseAddress else { return }
        var srcCbBuffer = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: srcBase),
            height: vImagePixelCount(srcCHeight),
            width: vImagePixelCount(srcCWidth),
            rowBytes: image.cStride
        )
        resized.cbPlane.withUnsafeMutableBufferPointer { dstPtr in
            guard let dstBase = dstPtr.baseAddress else { return }
            var dstCbBuffer = vImage_Buffer(
                data: dstBase,
                height: vImagePixelCount(targetCHeight),
                width: vImagePixelCount(targetCWidth),
                rowBytes: targetCWidth
            )
            _ = vImageScale_Planar8(&srcCbBuffer, &dstCbBuffer, nil, vImage_Flags(kvImageHighQualityResampling))
        }
    }
    
    image.crPlane.withUnsafeBufferPointer { srcPtr in
        guard let srcBase = srcPtr.baseAddress else { return }
        var srcCrBuffer = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: srcBase),
            height: vImagePixelCount(srcCHeight),
            width: vImagePixelCount(srcCWidth),
            rowBytes: image.cStride
        )
        resized.crPlane.withUnsafeMutableBufferPointer { dstPtr in
            guard let dstBase = dstPtr.baseAddress else { return }
            var dstCrBuffer = vImage_Buffer(
                data: dstBase,
                height: vImagePixelCount(targetCHeight),
                width: vImagePixelCount(targetCWidth),
                rowBytes: targetCWidth
            )
            _ = vImageScale_Planar8(&srcCrBuffer, &dstCrBuffer, nil, vImage_Flags(kvImageHighQualityResampling))
        }
    }
    
    return resized
}

/// Create CVPixelBuffer (420YpCbCr8BiPlanarFullRange) from YCbCrImage
public func createPixelBuffer(from img: YCbCrImage) -> CVPixelBuffer? {
    let width = img.width
    let height = img.height
    
    let attrs = [
        kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
        kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue,
        kCVPixelBufferMetalCompatibilityKey: kCFBooleanTrue,
    ] as CFDictionary
    
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, attrs, &pixelBuffer)
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
    
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    
    if let yDest = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) {
        let destStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        img.yPlane.withUnsafeBufferPointer { ySrc in
            guard let srcBase = ySrc.baseAddress else { return }
            for y in 0..<height {
                let destRow = yDest.advanced(by: y * destStride)
                let srcRow = srcBase.advanced(by: y * width)
                memcpy(destRow, srcRow, width)
            }
        }
    }
    
    if let uvDest = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) {
        let destStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        
        img.cbPlane.withUnsafeBufferPointer { cbSrc in
            img.crPlane.withUnsafeBufferPointer { crSrc in
                guard let cbBase = cbSrc.baseAddress, let crBase = crSrc.baseAddress else { return }
                for y in 0..<cHeight {
                    let destRow = uvDest.advanced(by: y * destStride).assumingMemoryBound(to: UInt8.self)
                    let cbRow = cbBase.advanced(by: y * cWidth)
                    let crRow = crBase.advanced(by: y * cWidth)
                    for x in 0..<cWidth {
                        destRow[x * 2 + 0] = cbRow[x]
                        destRow[x * 2 + 1] = crRow[x]
                    }
                }
            }
        }
    }
    return buffer
}

/// Create YCbCrImage from CVPixelBuffer (BiPlanar or BGRA)
public func createYCbCrImage(from buffer: CVPixelBuffer, width: Int, height: Int) -> YCbCrImage {
    var ycbcr = YCbCrImage(width: width, height: height, ratio: .ratio420)
    
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    
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
        // Fallback for BGRA
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        if let baseAddr = CVPixelBufferGetBaseAddress(buffer) {
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
    }
    
    return ycbcr
}
