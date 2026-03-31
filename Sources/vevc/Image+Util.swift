@inline(__always)
public func rgbaToYCbCr(data: [UInt8], width: Int, height: Int) -> YCbCrImage {
    var ycbcr = YCbCrImage(width: width, height: height, ratio: .ratio444)
    
    withUnsafePointers(data, mut: &ycbcr.yPlane, mut: &ycbcr.cbPlane, mut: &ycbcr.crPlane) { dataBase, yBase, cbBase, crBase in
        let totalPixels = width * height
        let yBias = 1 << 15
        
        var i = 0
        while i <= totalPixels - 4 {
            let offset = i * 4
            let r0 = Int(dataBase[offset + 0])
            let g0 = Int(dataBase[offset + 1])
            let b0 = Int(dataBase[offset + 2])
            
            let r1 = Int(dataBase[offset + 4])
            let g1 = Int(dataBase[offset + 5])
            let b1 = Int(dataBase[offset + 6])
            
            let r2 = Int(dataBase[offset + 8])
            let g2 = Int(dataBase[offset + 9])
            let b2 = Int(dataBase[offset + 10])
            
            let r3 = Int(dataBase[offset + 12])
            let g3 = Int(dataBase[offset + 13])
            let b3 = Int(dataBase[offset + 14])
            
            let yVal0 = (19595 * r0 + 38470 * g0 + 7471 * b0 + yBias) >> 16
            let cbVal0 = ((-11059 * r0 - 21709 * g0 + 32768 * b0 + yBias) >> 16) + 128
            let crVal0 = ((32768 * r0 - 27439 * g0 - 5329 * b0 + yBias) >> 16) + 128
            
            let yVal1 = (19595 * r1 + 38470 * g1 + 7471 * b1 + yBias) >> 16
            let cbVal1 = ((-11059 * r1 - 21709 * g1 + 32768 * b1 + yBias) >> 16) + 128
            let crVal1 = ((32768 * r1 - 27439 * g1 - 5329 * b1 + yBias) >> 16) + 128
            
            let yVal2 = (19595 * r2 + 38470 * g2 + 7471 * b2 + yBias) >> 16
            let cbVal2 = ((-11059 * r2 - 21709 * g2 + 32768 * b2 + yBias) >> 16) + 128
            let crVal2 = ((32768 * r2 - 27439 * g2 - 5329 * b2 + yBias) >> 16) + 128
            
            let yVal3 = (19595 * r3 + 38470 * g3 + 7471 * b3 + yBias) >> 16
            let cbVal3 = ((-11059 * r3 - 21709 * g3 + 32768 * b3 + yBias) >> 16) + 128
            let crVal3 = ((32768 * r3 - 27439 * g3 - 5329 * b3 + yBias) >> 16) + 128
            
            yBase[i] = UInt8(clamping: yVal0)
            cbBase[i] = UInt8(clamping: cbVal0)
            crBase[i] = UInt8(clamping: crVal0)
            
            yBase[i+1] = UInt8(clamping: yVal1)
            cbBase[i+1] = UInt8(clamping: cbVal1)
            crBase[i+1] = UInt8(clamping: crVal1)
            
            yBase[i+2] = UInt8(clamping: yVal2)
            cbBase[i+2] = UInt8(clamping: cbVal2)
            crBase[i+2] = UInt8(clamping: crVal2)
            
            yBase[i+3] = UInt8(clamping: yVal3)
            cbBase[i+3] = UInt8(clamping: cbVal3)
            crBase[i+3] = UInt8(clamping: crVal3)
            
            i += 4
        }
        
        while i < totalPixels {
            let offset = i * 4
            let r1 = Int(dataBase[offset + 0])
            let g1 = Int(dataBase[offset + 1])
            let b1 = Int(dataBase[offset + 2])
            
            let yVal = (19595 * r1 + 38470 * g1 + 7471 * b1 + yBias) >> 16
            let cbVal = ((-11059 * r1 - 21709 * g1 + 32768 * b1 + yBias) >> 16) + 128
            let crVal = ((32768 * r1 - 27439 * g1 - 5329 * b1 + yBias) >> 16) + 128
            
            yBase[i] = UInt8(clamping: yVal)
            cbBase[i] = UInt8(clamping: cbVal)
            crBase[i] = UInt8(clamping: crVal)
            i += 1
        }
    }
    return ycbcr
}

@inline(__always)
public func ycbcrToRGBA(img: YCbCrImage) -> [UInt8] {
    let width = img.width
    let height = img.height
    let totalPixels = width * height
    var rawData = [UInt8](repeating: 0, count: totalPixels * 4)
    
    withUnsafePointers(img.yPlane, img.cbPlane, img.crPlane, mut: &rawData) { yBase, cbBase, crBase, outBase in
        if img.ratio == .ratio444 {
            var i = 0
            while i <= totalPixels - 4 {
                let yVal0 = Int(yBase[i]) << 10
                let cbDiff0 = Int(cbBase[i]) - 128
                let crDiff0 = Int(crBase[i]) - 128
                let r0 = (yVal0 + (1436 * crDiff0)) >> 10
                let g0 = (yVal0 - (352 * cbDiff0) - (731 * crDiff0)) >> 10
                let b0 = (yVal0 + (1815 * cbDiff0)) >> 10
                
                let yVal1 = Int(yBase[i+1]) << 10
                let cbDiff1 = Int(cbBase[i+1]) - 128
                let crDiff1 = Int(crBase[i+1]) - 128
                let r1 = (yVal1 + (1436 * crDiff1)) >> 10
                let g1 = (yVal1 - (352 * cbDiff1) - (731 * crDiff1)) >> 10
                let b1 = (yVal1 + (1815 * cbDiff1)) >> 10
                
                let yVal2 = Int(yBase[i+2]) << 10
                let cbDiff2 = Int(cbBase[i+2]) - 128
                let crDiff2 = Int(crBase[i+2]) - 128
                let r2 = (yVal2 + (1436 * crDiff2)) >> 10
                let g2 = (yVal2 - (352 * cbDiff2) - (731 * crDiff2)) >> 10
                let b2 = (yVal2 + (1815 * cbDiff2)) >> 10
                
                let yVal3 = Int(yBase[i+3]) << 10
                let cbDiff3 = Int(cbBase[i+3]) - 128
                let crDiff3 = Int(crBase[i+3]) - 128
                let r3 = (yVal3 + (1436 * crDiff3)) >> 10
                let g3 = (yVal3 - (352 * cbDiff3) - (731 * crDiff3)) >> 10
                let b3 = (yVal3 + (1815 * cbDiff3)) >> 10
                
                let offset = i * 4
                outBase[offset + 0] = UInt8(clamping: r0)
                outBase[offset + 1] = UInt8(clamping: g0)
                outBase[offset + 2] = UInt8(clamping: b0)
                outBase[offset + 3] = 255
                
                outBase[offset + 4] = UInt8(clamping: r1)
                outBase[offset + 5] = UInt8(clamping: g1)
                outBase[offset + 6] = UInt8(clamping: b1)
                outBase[offset + 7] = 255
                
                outBase[offset + 8] = UInt8(clamping: r2)
                outBase[offset + 9] = UInt8(clamping: g2)
                outBase[offset + 10] = UInt8(clamping: b2)
                outBase[offset + 11] = 255
                
                outBase[offset + 12] = UInt8(clamping: r3)
                outBase[offset + 13] = UInt8(clamping: g3)
                outBase[offset + 14] = UInt8(clamping: b3)
                outBase[offset + 15] = 255
                
                i += 4
            }
            while i < totalPixels {
                let yVal = Int(yBase[i]) << 10
                let cbDiff = Int(cbBase[i]) - 128
                let crDiff = Int(crBase[i]) - 128
                
                let r = (yVal + (1436 * crDiff)) >> 10
                let g = (yVal - (352 * cbDiff) - (731 * crDiff)) >> 10
                let b = (yVal + (1815 * cbDiff)) >> 10
                
                let offset = i * 4
                outBase[offset + 0] = UInt8(clamping: r)
                outBase[offset + 1] = UInt8(clamping: g)
                outBase[offset + 2] = UInt8(clamping: b)
                outBase[offset + 3] = 255
                i += 1
            }
        } else {
            let cWidth = width / 2
            for y in 0..<height {
                let cRowOffset = (y / 2) * cWidth
                let yRowOffset = y * width
                let outRowOffset = yRowOffset * 4
                
                var x = 0
                while x <= width - 4 {
                    let cOff0 = cRowOffset + (x / 2)
                    let cOff2 = cRowOffset + ((x + 2) / 2)
                    
                    let yVal0 = Int(yBase[yRowOffset + x]) << 10
                    let cbDiff0 = Int(cbBase[cOff0]) - 128
                    let crDiff0 = Int(crBase[cOff0]) - 128
                    let r0 = (yVal0 + (1436 * crDiff0)) >> 10
                    let g0 = (yVal0 - (352 * cbDiff0) - (731 * crDiff0)) >> 10
                    let b0 = (yVal0 + (1815 * cbDiff0)) >> 10
                    
                    let yVal1 = Int(yBase[yRowOffset + x + 1]) << 10
                    let r1 = (yVal1 + (1436 * crDiff0)) >> 10
                    let g1 = (yVal1 - (352 * cbDiff0) - (731 * crDiff0)) >> 10
                    let b1 = (yVal1 + (1815 * cbDiff0)) >> 10
                    
                    let yVal2 = Int(yBase[yRowOffset + x + 2]) << 10
                    let cbDiff2 = Int(cbBase[cOff2]) - 128
                    let crDiff2 = Int(crBase[cOff2]) - 128
                    let r2 = (yVal2 + (1436 * crDiff2)) >> 10
                    let g2 = (yVal2 - (352 * cbDiff2) - (731 * crDiff2)) >> 10
                    let b2 = (yVal2 + (1815 * cbDiff2)) >> 10
                    
                    let yVal3 = Int(yBase[yRowOffset + x + 3]) << 10
                    let r3 = (yVal3 + (1436 * crDiff2)) >> 10
                    let g3 = (yVal3 - (352 * cbDiff2) - (731 * crDiff2)) >> 10
                    let b3 = (yVal3 + (1815 * cbDiff2)) >> 10
                    
                    let offset = outRowOffset + (x * 4)
                    outBase[offset + 0]  = UInt8(clamping: r0); outBase[offset + 1]  = UInt8(clamping: g0); outBase[offset + 2]  = UInt8(clamping: b0); outBase[offset + 3]  = 255
                    outBase[offset + 4]  = UInt8(clamping: r1); outBase[offset + 5]  = UInt8(clamping: g1); outBase[offset + 6]  = UInt8(clamping: b1); outBase[offset + 7]  = 255
                    outBase[offset + 8]  = UInt8(clamping: r2); outBase[offset + 9]  = UInt8(clamping: g2); outBase[offset + 10] = UInt8(clamping: b2); outBase[offset + 11] = 255
                    outBase[offset + 12] = UInt8(clamping: r3); outBase[offset + 13] = UInt8(clamping: g3); outBase[offset + 14] = UInt8(clamping: b3); outBase[offset + 15] = 255
                    
                    x += 4
                }
                while x < width {
                    let cOff = cRowOffset + (x / 2)
                    let yVal = Int(yBase[yRowOffset + x]) << 10
                    let cbDiff = Int(cbBase[cOff]) - 128
                    let crDiff = Int(crBase[cOff]) - 128
                    
                    let r = (yVal + (1436 * crDiff)) >> 10
                    let g = (yVal - (352 * cbDiff) - (731 * crDiff)) >> 10
                    let b = (yVal + (1815 * cbDiff)) >> 10
                    
                    let offset = outRowOffset + (x * 4)
                    outBase[offset + 0] = UInt8(clamping: r)
                    outBase[offset + 1] = UInt8(clamping: g)
                    outBase[offset + 2] = UInt8(clamping: b)
                    outBase[offset + 3] = 255
                    x += 1
                }
            }
        }
    }
    return rawData
}