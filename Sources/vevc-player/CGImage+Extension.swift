import Foundation
import CoreGraphics
import vevc

func createCGImage(from img: YCbCrImage) throws -> CGImage {
    let width = img.width
    let height = img.height
    let bytesPerPixel = 4
    let bytesPerRow = (bytesPerPixel * width)
    var rawData = [UInt8](repeating: 0, count: (height * bytesPerRow))
    
    for y in 0..<height {
        for x in 0..<width {
            let yScaled = Int(img.yPlane[img.yOffset(x, y)]) << 10
            
            var cPx = x
            var cPy = y
            if img.ratio == .ratio420 {
                cPx = (x / 2)
                cPy = (y / 2)
            }
            
            let cOff = img.cOffset(cPx, cPy)
            let cbDiff = Int(img.cbPlane[cOff]) - 128
            let crDiff = Int(img.crPlane[cOff]) - 128
            
            let r = (yScaled + (1436 * crDiff)) >> 10
            let g = (yScaled - (352 * cbDiff) - (731 * crDiff)) >> 10
            let b = (yScaled + (1815 * cbDiff)) >> 10
            
            let offset = ((y * bytesPerRow) + (x * bytesPerPixel))
            rawData[offset + 0] = UInt8(clamping: r)
            rawData[offset + 1] = UInt8(clamping: g)
            rawData[offset + 2] = UInt8(clamping: b)
            rawData[offset + 3] = 255
        }
    }
    
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
        throw NSError(domain: "ImageError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create color space"])
    }
    guard let context = CGContext(
        data: &rawData,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    ) else {
        throw NSError(domain: "ImageError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create context for output"])
    }
    
    guard let cgImage = context.makeImage() else {
        throw NSError(domain: "ImageError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGImage"])
    }
    
    return cgImage
}
