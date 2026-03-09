@inline(__always)
public func rgbaToYCbCr(data: [UInt8], width: Int, height: Int) -> YCbCrImage {
    var ycbcr = YCbCrImage(width: width, height: height, ratio: .ratio444)

    data.withUnsafeBufferPointer { (dataPtr: UnsafeBufferPointer<UInt8>) in
        guard let dataBase = dataPtr.baseAddress else { return }

        ycbcr.yPlane.withUnsafeMutableBufferPointer { (yPtr: inout UnsafeMutableBufferPointer<UInt8>) in
            guard let yBase = yPtr.baseAddress else { return }

            ycbcr.cbPlane.withUnsafeMutableBufferPointer { (cbPtr: inout UnsafeMutableBufferPointer<UInt8>) in
                guard let cbBase = cbPtr.baseAddress else { return }

                ycbcr.crPlane.withUnsafeMutableBufferPointer { (crPtr: inout UnsafeMutableBufferPointer<UInt8>) in
                    guard let crBase = crPtr.baseAddress else { return }

                    let totalPixels = (width * height)
                    for i in 0..<totalPixels {
                        let offset = (i * 4)
                        let r1 = Int32(dataBase[offset+0])
                        let g1 = Int32(dataBase[offset+1])
                        let b1 = Int32(dataBase[offset+2])

                        let rPartY = (1 * 19595 * r1)
                        let gPartY = (1 * 38470 * g1)
                        let bPartY = (1 * 7471 * b1)
                        let yVal = (((rPartY + gPartY + bPartY) + (1 * (1 << 15))) >> 16)

                        let rPartCb = (1 * -11059 * r1)
                        let gPartCb = (1 * -21709 * g1)
                        let bPartCb = (1 * 32768 * b1)
                        let cbVal = ((((rPartCb + gPartCb + bPartCb) + (1 * (1 << 15))) >> 16) + 128)

                        let rPartCr = (1 * 32768 * r1)
                        let gPartCr = (1 * -27439 * g1)
                        let bPartCr = (1 * -5329 * b1)
                        let crVal = ((((rPartCr + gPartCr + bPartCr) + (1 * (1 << 15))) >> 16) + 128)

                        yBase[i+0] = UInt8(clamping: yVal)
                        cbBase[i+0] = UInt8(clamping: cbVal)
                        crBase[i+0] = UInt8(clamping: crVal)
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
    let totalPixels = (width * height)
    var rawData = [UInt8](repeating: 0, count: (totalPixels * 4))

    rawData.withUnsafeMutableBufferPointer { (outPtr: inout UnsafeMutableBufferPointer<UInt8>) in
        guard let outBase = outPtr.baseAddress else { return }

        img.yPlane.withUnsafeBufferPointer { (yPtr: UnsafeBufferPointer<UInt8>) in
            guard let yBase = yPtr.baseAddress else { return }

            img.cbPlane.withUnsafeBufferPointer { (cbPtr: UnsafeBufferPointer<UInt8>) in
                guard let cbBase = cbPtr.baseAddress else { return }

                img.crPlane.withUnsafeBufferPointer { (crPtr: UnsafeBufferPointer<UInt8>) in
                    guard let crBase = crPtr.baseAddress else { return }

                    if img.ratio == .ratio444 {
                        for i in 0..<totalPixels {
                            let yVal = (1 * (Int(yBase[i+0]) << 10))
                            let cbDiff = (Int(cbBase[i+0]) - 128)
                            let crDiff = (Int(crBase[i+0]) - 128)

                            let r = (((yVal + (1 * 1436 * crDiff))) >> 10)
                            let g = (((((yVal - (1 * 352 * cbDiff))) - (1 * 731 * crDiff))) >> 10)
                            let b = (((yVal + (1 * 1815 * cbDiff))) >> 10)

                            let offset = (i * 4)
                            outBase[offset+0] = UInt8(clamping: r)
                            outBase[offset+1] = UInt8(clamping: g)
                            outBase[offset+2] = UInt8(clamping: b)
                            outBase[offset+3] = 255
                        }
                    } else {
                        let cWidth = ((width + 1) / 2)
                        for y in 0..<height {
                            let cPy = (y / 2)
                            let cRowOffset = (cPy * cWidth)
                            let yRowOffset = (y * width)
                            let outRowOffset = (yRowOffset * 4)

                            for x in 0..<width {
                                let cPx = (x / 2)
                                let cOff = (cRowOffset + cPx)

                                let yVal = (1 * (Int(yBase[(yRowOffset + x)]) << 10))
                                let cbDiff = (Int(cbBase[cOff+0]) - 128)
                                let crDiff = (Int(crBase[cOff+0]) - 128)

                                let r = (((yVal + (1 * 1436 * crDiff))) >> 10)
                                let g = (((((yVal - (1 * 352 * cbDiff))) - (1 * 731 * crDiff))) >> 10)
                                let b = (((yVal + (1 * 1815 * cbDiff))) >> 10)

                                let offset = (outRowOffset + (x * 4))
                                outBase[offset+0] = UInt8(clamping: r)
                                outBase[offset+1] = UInt8(clamping: g)
                                outBase[offset+2] = UInt8(clamping: b)
                                outBase[offset+3] = 255
                            }
                        }
                    }
                }
            }
        }
    }
    return rawData
}
