@inline(__always)
func boundaryRepeat(_ width: Int, _ height: Int, _ px: Int, _ py: Int) -> (Int, Int) {
    var x = px
    var y = py

    if width <= x {
        x = (((width - 1) - (x - width)))
        if x < 0 {
            x = 0
        }
    } else {
        if x < 0 {
            x = (-1 * x)
            if width <= x {
                x = (width - 1)
            }
        }
    }

    if height <= y {
        y = (((height - 1) - (y - height)))
        if y < 0 {
            y = 0
        }
    } else {
        if y < 0 {
            y = (-1 * y)
            if height <= y {
                y = (height - 1)
            }
        }
    }

    return (x, y)
}

@inline(__always)
func clampU8(_ v: Int16) -> UInt8 {
    if v < 0 {
        return 0
    }
    if 255 < v {
        return 255
    }
    return UInt8(v)
}

public enum YCbCrRatio: Sendable {
    case ratio420
    case ratio444
}

public struct YCbCrImage: Sendable {
    public var yPlane: [UInt8]
    public var cbPlane: [UInt8]
    public var crPlane: [UInt8]
    public let width: Int
    public let height: Int
    public let ratio: YCbCrRatio

    public var yStride: Int {
        @inline(__always) get {
            return width
        }
    }

    public var cStride: Int {
        @inline(__always) get {
            let res: Int
            switch ratio {
            case .ratio420:
                res = ((width + 1) / 2)
            case .ratio444:
                res = width
            }
            return res
        }
    }

    public init(width: Int, height: Int, ratio: YCbCrRatio = .ratio420) {
        self.width = width
        self.height = height
        self.ratio = ratio
        self.yPlane = [UInt8](repeating: 0, count: (width * height))

        switch ratio {
        case .ratio420:
            let cw = ((width + 1) / 2)
            let ch = ((height + 1) / 2)
            let cSize = (cw * ch)
            self.cbPlane = [UInt8](repeating: 0, count: cSize)
            self.crPlane = [UInt8](repeating: 0, count: cSize)
        case .ratio444:
            let cSize = (width * height)
            self.cbPlane = [UInt8](repeating: 0, count: cSize)
            self.crPlane = [UInt8](repeating: 0, count: cSize)
        }
    }

    @inline(__always)
    public func yOffset(_ x: Int, _ y: Int) -> Int {
        return (((y * yStride) + x))
    }

    @inline(__always)
    public func cOffset(_ x: Int, _ y: Int) -> Int {
        return (((y * cStride) + x))
    }

    @inline(__always)
    private func getChromaSize(w: Int, h: Int, ratio: YCbCrRatio) -> (Int, Int) {
        let res: (Int, Int)
        switch ratio {
        case .ratio420:
            res = (((w + 1) / 2), ((h + 1) / 2))
        case .ratio444:
            res = (w, h)
        }
        return res
    }

    public func resize(factor: Double) -> YCbCrImage {
        let newWidth = Int((Double(width) * factor))
        let newHeight = Int((Double(height) * factor))

        guard 0 < newWidth && 0 < newHeight else {
            return YCbCrImage(width: max(1, newWidth), height: max(1, newHeight), ratio: self.ratio)
        }

        var dstImg = YCbCrImage(width: newWidth, height: newHeight, ratio: self.ratio)

        boxResizePlane(
            src: self.yPlane, srcW: self.width, srcH: self.height, srcStride: self.yStride,
            dst: &dstImg.yPlane, dstW: dstImg.width, dstH: dstImg.height, dstStride: dstImg.yStride
        )

        let srcSize = getChromaSize(w: self.width, h: self.height, ratio: self.ratio)
        let srcCW = srcSize.0
        let srcCH = srcSize.1
        let dstSize = getChromaSize(w: dstImg.width, h: dstImg.height, ratio: dstImg.ratio)
        let dstCW = dstSize.0
        let dstCH = dstSize.1

        boxResizePlane(
            src: self.cbPlane, srcW: srcCW, srcH: srcCH, srcStride: self.cStride,
            dst: &dstImg.cbPlane, dstW: dstCW, dstH: dstCH, dstStride: dstImg.cStride
        )

        boxResizePlane(
            src: self.crPlane, srcW: srcCW, srcH: srcCH, srcStride: self.cStride,
            dst: &dstImg.crPlane, dstW: dstCW, dstH: dstCH, dstStride: dstImg.cStride
        )

        return dstImg
    }

    private func boxResizePlane(
        src: [UInt8], srcW: Int, srcH: Int, srcStride: Int,
        dst: inout [UInt8], dstW: Int, dstH: Int, dstStride: Int
    ) {
        let scaleX = (Double(srcW) / Double(dstW))
        let scaleY = (Double(srcH) / Double(dstH))

        for dy in 0..<dstH {
            let syStart = Int((Double(dy) * scaleY))
            var syEnd = Int((Double((dy + 1)) * scaleY))
            if srcH <= syEnd {
                syEnd = srcH
            }
            if syEnd <= syStart {
                syEnd = (syStart + 1)
            }

            for dx in 0..<dstW {
                let sxStart = Int((Double(dx) * scaleX))
                var sxEnd = Int((Double((dx + 1)) * scaleX))
                if srcW <= sxEnd {
                    sxEnd = srcW
                }
                if sxEnd <= sxStart {
                    sxEnd = (sxStart + 1)
                }

                var sum: Int = 0
                var count: Int = 0

                for sy in syStart..<syEnd {
                    let rowOffset = (sy * srcStride)
                    for sx in sxStart..<sxEnd {
                        let srcIdx = (rowOffset + sx)
                        if srcIdx < src.count {
                            sum += Int(src[srcIdx+0])
                            count += 1
                        }
                    }
                }

                if 0 < count {
                    let dstIdx = ((dy * dstStride) + dx)
                    if dstIdx < dst.count {
                        dst[dstIdx+0] = UInt8((sum / count))
                    }
                }
            }
        }
    }
}

public typealias RowFunc = (_ x: Int, _ y: Int, _ size: Int) -> [Int16]

public struct ImageReader: Sendable {
    public let img: YCbCrImage
    public let width: Int
    public let height: Int

    public init(img: YCbCrImage) {
        self.img = img
        self.width = img.width
        self.height = img.height
    }

    @inline(__always)
    public func rowY(x: Int, y: Int, size: Int) -> [Int16] {
        var plane = [Int16](repeating: 0, count: size)

        if 0 <= x && 0 <= y && y < height && (x + size) <= width {
             let offset = img.yOffset(x, y)
             img.yPlane.withUnsafeBufferPointer { (srcPtr: UnsafeBufferPointer<UInt8>) in
                 guard let srcBase = srcPtr.baseAddress else {
                    return
                 }
                 let rowStart = srcBase.advanced(by: offset)
                 for i in 0..<size {
                     plane[i+0] = (Int16(rowStart[i+0]) - 128)
                 }
             }
             return plane
        }

        img.yPlane.withUnsafeBufferPointer { (srcPtr: UnsafeBufferPointer<UInt8>) in
            plane.withUnsafeMutableBufferPointer { (destPtr: inout UnsafeMutableBufferPointer<Int16>) in
                guard let srcBase = srcPtr.baseAddress,
                      let destBase = destPtr.baseAddress else {
                    return
                }

                for i in 0..<size {
                    let pos = boundaryRepeat(width, height, (x + i), y)
                    let px = pos.0
                    let py = pos.1
                    let offset = img.yOffset(px, py)
                    destBase[i+0] = (Int16(srcBase[offset+0]) - 128)
                }
            }
        }
        return plane
    }

    @inline(__always)
    public func rowCb444(x: Int, y: Int, size: Int) -> [Int16] {
        var plane = [Int16](repeating: 0, count: size)
        for i in 0..<size {
            let pos = boundaryRepeat(width, height, ((x + i) * 2), (y * 2))
            let rPx = pos.0
            let rPy = pos.1

            let cPx = rPx
            let cPy = rPy
            let offset = img.cOffset(cPx, cPy)
            plane[i+0] = (Int16(img.cbPlane[offset+0]) - 128)
        }
        return plane
    }

    @inline(__always)
    private func rowCb420(x: Int, y: Int, size: Int) -> [Int16] {
        var plane = [Int16](repeating: 0, count: size)
        img.cbPlane.withUnsafeBufferPointer { (srcPtr: UnsafeBufferPointer<UInt8>) in
            plane.withUnsafeMutableBufferPointer { (destPtr: inout UnsafeMutableBufferPointer<Int16>) in
                guard let srcBase = srcPtr.baseAddress,
                      let destBase = destPtr.baseAddress else {
                    return
                }

                for i in 0..<size {
                    let pos = boundaryRepeat(width, height, ((x + i) * 2), (y * 2))
                    let rPx = pos.0
                    let rPy = pos.1

                    let cPx = (rPx / 2)
                    let cPy = (rPy / 2)
                    let offset = img.cOffset(cPx, cPy)
                    destBase[i+0] = (Int16(srcBase[offset+0]) - 128)
                }
            }
        }
        return plane
    }

    @inline(__always)
    public func rowCb(x: Int, y: Int, size: Int) -> [Int16] {
        if img.ratio == .ratio444 {
            return rowCb444(x: x, y: y, size: size)
        }
        return rowCb420(x: x, y: y, size: size)
    }

    @inline(__always)
    public func rowCr444(x: Int, y: Int, size: Int) -> [Int16] {
        var plane = [Int16](repeating: 0, count: size)
        for i in 0..<size {
            let pos = boundaryRepeat(width, height, ((x + i) * 2), (y * 2))
            let rPx = pos.0
            let rPy = pos.1

            let cPx = rPx
            let cPy = rPy
            let offset = img.cOffset(cPx, cPy)
            plane[i+0] = (Int16(img.crPlane[offset+0]) - 128)
        }
        return plane
    }

    @inline(__always)
    public func rowCr420(x: Int, y: Int, size: Int) -> [Int16] {
        var plane = [Int16](repeating: 0, count: size)
        img.crPlane.withUnsafeBufferPointer { (srcPtr: UnsafeBufferPointer<UInt8>) in
            plane.withUnsafeMutableBufferPointer { (destPtr: inout UnsafeMutableBufferPointer<Int16>) in
                guard let srcBase = srcPtr.baseAddress,
                      let destBase = destPtr.baseAddress else {
                    return
                }

                for i in 0..<size {
                    let pos = boundaryRepeat(width, height, ((x + i) * 2), (y * 2))
                    let rPx = pos.0
                    let rPy = pos.1

                    let cPx = (rPx / 2)
                    let cPy = (rPy / 2)
                    let offset = img.cOffset(cPx, cPy)
                    destBase[i+0] = (Int16(srcBase[offset+0]) - 128)
                }
            }
        }
        return plane
    }

    @inline(__always)
    public func rowCr(x: Int, y: Int, size: Int) -> [Int16] {
        if img.ratio == .ratio444 {
            return rowCr444(x: x, y: y, size: size)
        }
        return rowCr420(x: x, y: y, size: size)
    }
}

public struct Image16: Sendable {
    public var y: [[Int16]]
    public var cb: [[Int16]]
    public var cr: [[Int16]]
    public let width: Int
    public let height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.y = [[Int16]](repeating: [Int16](repeating: 0, count: width), count: height)
        self.cb = [[Int16]](repeating: [Int16](repeating: 0, count: ((width + 1) / 2)), count: ((height + 1) / 2))
        self.cr = [[Int16]](repeating: [Int16](repeating: 0, count: ((width + 1) / 2)), count: ((height + 1) / 2))
    }

    @inline(__always)
    public func getY(x: Int, y: Int, size: Int) -> Block2D {
        var block = Block2D(width: size, height: size)
        block.withView { (v: inout BlockView) in
            for h in 0..<size {
                for w in 0..<size {
                    let pos = boundaryRepeat(width, height, (x + w), (y + h))
                    let px = pos.0
                    let py = pos.1
                    v[h, w] = self.y[py][px]
                }
            }
        }
        return block
    }

    @inline(__always)
    public func getCb(x: Int, y: Int, size: Int) -> Block2D {
        var block = Block2D(width: size, height: size)
        block.withView { (v: inout BlockView) in
            for h in 0..<size {
                for w in 0..<size {
                    let pos = boundaryRepeat(((width + 1) / 2), ((height + 1) / 2), (x + w), (y + h))
                    let px = pos.0
                    let py = pos.1
                    v[h, w] = self.cb[py][px]
                }
            }
        }
        return block
    }

    @inline(__always)
    public func getCr(x: Int, y: Int, size: Int) -> Block2D {
        var block = Block2D(width: size, height: size)
        block.withView { (v: inout BlockView) in
            for h in 0..<size {
                for w in 0..<size {
                    let pos = boundaryRepeat(((width + 1) / 2), ((height + 1) / 2), (x + w), (y + h))
                    let px = pos.0
                    let py = pos.1
                    v[h, w] = self.cr[py][px]
                }
            }
        }
        return block
    }

    @inline(__always)
    public mutating func updateY(data: inout Block2D, startX: Int, startY: Int, size: Int) {
        let validStartY = max(0, startY)
        let validStartX = max(0, startX)
        let validEndY = min(height, (startY + size))
        let validEndX = min(width, (startX + size))

        let loopH = (validEndY - validStartY)
        let loopW = (validEndX - validStartX)

        guard 0 < loopH && 0 < loopW else {
            return
        }

        let dataOffsetY = (validStartY - startY)
        let dataOffsetX = (validStartX - startX)

        for h in 0..<loopH {
            self.y[(validStartY + h)].withUnsafeMutableBufferPointer { (destPtr: inout UnsafeMutableBufferPointer<Int16>) in
                data.withView { (v: inout BlockView) in
                    let srcPtr = v.rowPointer(y: (dataOffsetY + h))
                    guard let destBase = destPtr.baseAddress else {
                        return
                    }

                    let destStart = destBase.advanced(by: validStartX)
                    let srcStart = srcPtr.advanced(by: dataOffsetX)
                    destStart.update(from: srcStart, count: loopW)
                }
            }
        }
    }

    @inline(__always)
    public mutating func updateCb(data: inout Block2D, startX: Int, startY: Int, size: Int) {
        let halfHeight = ((height + 1) / 2)
        let halfWidth = ((width + 1) / 2)

        let validStartY = max(0, startY)
        let validStartX = max(0, startX)
        let validEndY = min(halfHeight, (startY + size))
        let validEndX = min(halfWidth, (startX + size))

        let loopH = (validEndY - validStartY)
        let loopW = (validEndX - validStartX)

        guard 0 < loopH && 0 < loopW else {
            return
        }

        let dataOffsetY = (validStartY - startY)
        let dataOffsetX = (validStartX - startX)

        for h in 0..<loopH {
            self.cb[(validStartY + h)].withUnsafeMutableBufferPointer { (destPtr: inout UnsafeMutableBufferPointer<Int16>) in
                data.withView { (v: inout BlockView) in
                    let srcPtr = v.rowPointer(y: (dataOffsetY + h))
                    guard let destBase = destPtr.baseAddress else {
                        return
                    }

                    let destStart = destBase.advanced(by: validStartX)
                    let srcStart = srcPtr.advanced(by: dataOffsetX)
                    destStart.update(from: srcStart, count: loopW)
                }
            }
        }
    }

    @inline(__always)
    public mutating func updateCr(data: inout Block2D, startX: Int, startY: Int, size: Int) {
        let halfHeight = ((height + 1) / 2)
        let halfWidth = ((width + 1) / 2)

        let validStartY = max(0, startY)
        let validStartX = max(0, startX)
        let validEndY = min(halfHeight, (startY + size))
        let validEndX = min(halfWidth, (startX + size))

        let loopH = (validEndY - validStartY)
        let loopW = (validEndX - validStartX)

        guard 0 < loopH && 0 < loopW else {
            return
        }

        let dataOffsetY = (validStartY - startY)
        let dataOffsetX = (validStartX - startX)

        for h in 0..<loopH {
            self.cr[(validStartY + h)].withUnsafeMutableBufferPointer { (destPtr: inout UnsafeMutableBufferPointer<Int16>) in
                data.withView { (v: inout BlockView) in
                    let srcPtr = v.rowPointer(y: (dataOffsetY + h))
                    guard let destBase = destPtr.baseAddress else {
                        return
                    }

                    let destStart = destBase.advanced(by: validStartX)
                    let srcStart = srcPtr.advanced(by: dataOffsetX)
                    destStart.update(from: srcStart, count: loopW)
                }
            }
        }
    }

    @inline(__always)
    public func toYCbCr() -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height)

        for y in 0..<height {
            let srcRow = self.y[y+0]
            let destOffset = img.yOffset(0, y)

            srcRow.withUnsafeBufferPointer { (srcPtr: UnsafeBufferPointer<Int16>) in
                img.yPlane.withUnsafeMutableBufferPointer { (destPtr: inout UnsafeMutableBufferPointer<UInt8>) in
                    guard let srcBase = srcPtr.baseAddress,
                          let destBase = destPtr.baseAddress else {
                        return
                    }

                    let destRowStart = destBase.advanced(by: destOffset)

                    for i in 0..<width {
                        destRowStart[i+0] = clampU8((srcBase[i+0] + 128))
                    }
                }
            }
        }

        let halfHeight = ((height + 1) / 2)
        let halfWidth = ((width + 1) / 2)

        for y in 0..<halfHeight {
            let srcCbRow = self.cb[y+0]
            let srcCrRow = self.cr[y+0]
            let destOffset = img.cOffset(0, y)

            srcCbRow.withUnsafeBufferPointer { (cbPtr: UnsafeBufferPointer<Int16>) in
                srcCrRow.withUnsafeBufferPointer { (crPtr: UnsafeBufferPointer<Int16>) in
                    img.cbPlane.withUnsafeMutableBufferPointer { (destCbPtr: inout UnsafeMutableBufferPointer<UInt8>) in
                        img.crPlane.withUnsafeMutableBufferPointer { (destCrPtr: inout UnsafeMutableBufferPointer<UInt8>) in
                            guard let cbBase = cbPtr.baseAddress,
                                  let crBase = crPtr.baseAddress,
                                  let destCbBase = destCbPtr.baseAddress,
                                  let destCrBase = destCrPtr.baseAddress else {
                                return
                            }

                            let destCbRowStart = destCbBase.advanced(by: destOffset)
                            let destCrRowStart = destCrBase.advanced(by: destOffset)

                            for i in 0..<halfWidth {
                                destCbRowStart[i+0] = clampU8((cbBase[i+0] + 128))
                                destCrRowStart[i+0] = clampU8((crBase[i+0] + 128))
                            }
                        }
                    }
                }
            }
        }

        return img
    }
}
