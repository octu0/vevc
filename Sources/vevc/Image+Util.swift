@inline(__always)
public func rgbaToYCbCr(data: [UInt8], width: Int, height: Int) -> YCbCrImage {
    var ycbcr = YCbCrImage(width: width, height: height, ratio: .ratio444)
    
    data.withUnsafeBufferPointer { dataPtr in
        ycbcr.yPlane.withUnsafeMutableBufferPointer { yPtr in
            ycbcr.cbPlane.withUnsafeMutableBufferPointer { cbPtr in
                ycbcr.crPlane.withUnsafeMutableBufferPointer { crPtr in
                    let dataBase = dataPtr.baseAddress!
                    let yBase = yPtr.baseAddress!
                    let cbBase = cbPtr.baseAddress!
                    let crBase = crPtr.baseAddress!
                    
                    let totalPixels = width * height
                    for i in 0..<totalPixels {
                        let offset = i * 4
                        let r1 = Int32(dataBase[offset + 0])
                        let g1 = Int32(dataBase[offset + 1])
                        let b1 = Int32(dataBase[offset + 2])
                        
                        let yVal = (19595 * r1 + 38470 * g1 + 7471 * b1 + (1 << 15)) >> 16
                        let cbVal = ((-11059 * r1 - 21709 * g1 + 32768 * b1 + (1 << 15)) >> 16) + 128
                        let crVal = ((32768 * r1 - 27439 * g1 - 5329 * b1 + (1 << 15)) >> 16) + 128
                        
                        yBase[i] = UInt8(clamping: yVal)
                        cbBase[i] = UInt8(clamping: cbVal)
                        crBase[i] = UInt8(clamping: crVal)
                    }
                }
            }
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
    
    rawData.withUnsafeMutableBufferPointer { outPtr in
        img.yPlane.withUnsafeBufferPointer { yPtr in
            img.cbPlane.withUnsafeBufferPointer { cbPtr in
                img.crPlane.withUnsafeBufferPointer { crPtr in
                    let outBase = outPtr.baseAddress!
                    let yBase = yPtr.baseAddress!
                    let cbBase = cbPtr.baseAddress!
                    let crBase = crPtr.baseAddress!
                    
                    if img.ratio == .ratio444 {
                        for i in 0..<totalPixels {
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
                        }
                    } else {
                        for y in 0..<height {
                            let cPy = y / 2
                            let cRowOffset = cPy * (width / 2)
                            let yRowOffset = y * width
                            let outRowOffset = yRowOffset * 4
                            
                            for x in 0..<width {
                                let cPx = x / 2
                                let cOff = cRowOffset + cPx
                                
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
                            }
                        }
                    }
                }
            }
        }
    }
    return rawData
}