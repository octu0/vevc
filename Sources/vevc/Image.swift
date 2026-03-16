// MARK: - Image

struct Int16Reader {
    let data: [Int16]
    let width: Int
    let height: Int
    
    @inline(__always)
    func row(x: Int, y: Int, size: Int) -> [Int16] {
        var r = [Int16](repeating: 0, count: size)
        let safeY = min(y, height - 1)
        
        let limit = min(size, width - x)
        if 0 < limit {
            data.withUnsafeBufferPointer { ptr in
                guard let basePtr = ptr.baseAddress else { return }
                let base = basePtr.advanced(by: safeY * width + x)
                r.withUnsafeMutableBufferPointer { dst in
                    guard let dstBase = dst.baseAddress else { return }
                    dstBase.update(from: base, count: limit)
                    
                    if limit < size {
                        let lastVal = dst[limit - 1]
                        for i in limit..<size {
                            dst[i] = lastVal
                        }
                    }
                }
            }
        } else {
            let lastVal = data[safeY * width + (width - 1)]
            for i in 0..<size {
                r[i] = lastVal
            }
        }
        
        return r
    }
    
    @inline(__always)
    func readBlock(x: Int, y: Int, width blockWidth: Int, height blockHeight: Int, into view: inout BlockView) {
        data.withUnsafeBufferPointer { srcBuf in
            guard let srcBase = srcBuf.baseAddress else { return }
            
            for line in 0..<blockHeight {
                let currentY = y + line
                let safeY = min(currentY, self.height - 1)
                
                let dstPtr = view.rowPointer(y: line)
                let limit = min(blockWidth, self.width - x)
                
                if 0 < limit {
                    let srcPtr = srcBase.advanced(by: safeY * self.width + x)
                    dstPtr.update(from: srcPtr, count: limit)
                    
                    if limit < blockWidth {
                        let lastVal = dstPtr[limit - 1]
                        for i in limit..<blockWidth {
                            dstPtr[i] = lastVal
                        }
                    }
                } else {
                    let lastVal = srcBuf[safeY * self.width + (self.width - 1)]
                    for i in 0..<blockWidth {
                        dstPtr[i] = lastVal
                    }
                }
            }
        }
    }
}

struct PlaneData420 {
    let width: Int
    let height: Int
    var y: [Int16]
    var cb: [Int16]
    var cr: [Int16]
    
    var rY: Int16Reader {
        Int16Reader(data: y, width: width, height: height)
    }
    var rCb: Int16Reader {
        Int16Reader(data: cb, width: (width + 1) / 2, height: (height + 1) / 2)
    }
    var rCr: Int16Reader {
        Int16Reader(data: cr, width: (width + 1) / 2, height: (height + 1) / 2)
    }
}

extension PlaneData420 {
    init(img16: Image16) {
        self.width = img16.width
        self.height = img16.height
        self.y = img16.y
        self.cb = img16.cb
        self.cr = img16.cr
    }
    
    func toYCbCr() -> YCbCrImage {
        var img = YCbCrImage(width: width, height: height)
        if width < 1 || height < 1 { return img }

        // optimize SIMD Int16 -> UInt8 (+128 clamp)
        @inline(__always)
        func convertPlane(src: [Int16], dst: inout [UInt8]) {
            let count = min(src.count, dst.count)
            src.withUnsafeBufferPointer { srcBuf in
                dst.withUnsafeMutableBufferPointer { dstBuf in
                    guard let srcPtr = srcBuf.baseAddress, let dstPtr = dstBuf.baseAddress else { return }
                    var i = 0
                    #if arch(arm64) || arch(x86_64) || arch(wasm32)
                    let offset128 = SIMD8<Int16>(repeating: 128)
                    let zero8 = SIMD8<Int16>.zero
                    let max255 = SIMD8<Int16>(repeating: 255)
                    while i + 8 <= count {
                        let vals = UnsafeRawPointer(srcPtr.advanced(by: i)).load(as: SIMD8<Int16>.self)
                        let clamped = (vals &+ offset128).clamped(lowerBound: zero8, upperBound: max255)
                        let narrowed = SIMD8<UInt8>(truncatingIfNeeded: clamped)
                        UnsafeMutableRawPointer(dstPtr.advanced(by: i)).storeBytes(of: narrowed, as: SIMD8<UInt8>.self)
                        i += 8
                    }
                    #endif
                    while i < count {
                        let v = srcPtr[i]
                        switch v {
                        case ..<(-128): dstPtr[i] = 0
                        case 128...: dstPtr[i] = 255
                        default: dstPtr[i] = UInt8(v + 128)
                        }
                        i += 1
                    }
                }
            }
        }

        convertPlane(src: self.y, dst: &img.yPlane)
        convertPlane(src: self.cb, dst: &img.cbPlane)
        convertPlane(src: self.cr, dst: &img.crPlane)

        return img
    }
}

@inline(__always)
func toPlaneData420(images: [YCbCrImage]) -> [PlaneData420] {
    return images.map { (img: YCbCrImage) in
        let y = [Int16](unsafeUninitializedCapacity: img.yPlane.count) { (buffer: inout UnsafeMutableBufferPointer<Int16>, initializedCount: inout Int) in
            img.yPlane.withUnsafeBufferPointer { (src: UnsafeBufferPointer<UInt8>) in
                for i in 0..<src.count {
                    buffer[i] = (Int16(src[i]) - 128)
                }
            }
            initializedCount = img.yPlane.count
        }

        let cWidth = ((img.width + 1) / 2)
        let cHeight = ((img.height + 1) / 2)
        let cCount = (cWidth * cHeight)

        if img.ratio == .ratio444 {
            let cb = [Int16](unsafeUninitializedCapacity: cCount) { (buffer: inout UnsafeMutableBufferPointer<Int16>, initializedCount: inout Int) in
                img.cbPlane.withUnsafeBufferPointer { (src: UnsafeBufferPointer<UInt8>) in
                    for cy in 0..<cHeight {
                        let py = (cy * 2)
                        let srcRowOffset = (py * img.width)
                        let dstRowOffset = (cy * cWidth)
                        for cx in 0..<cWidth {
                            let px = (cx * 2)
                            let srcOffset = (srcRowOffset + px)
                            let dstOffset = (dstRowOffset + cx)
                            if srcOffset < src.count {
                                buffer[dstOffset] = (Int16(src[srcOffset]) - 128)
                            } else {
                                buffer[dstOffset] = 0
                            }
                        }
                    }
                }
                initializedCount = cCount
            }
            let cr = [Int16](unsafeUninitializedCapacity: cCount) { (buffer: inout UnsafeMutableBufferPointer<Int16>, initializedCount: inout Int) in
                img.crPlane.withUnsafeBufferPointer { (src: UnsafeBufferPointer<UInt8>) in
                    for cy in 0..<cHeight {
                        let py = (cy * 2)
                        let srcRowOffset = (py * img.width)
                        let dstRowOffset = (cy * cWidth)
                        for cx in 0..<cWidth {
                            let px = (cx * 2)
                            let srcOffset = (srcRowOffset + px)
                            let dstOffset = (dstRowOffset + cx)
                            if srcOffset < src.count {
                                buffer[dstOffset] = (Int16(src[srcOffset]) - 128)
                            } else {
                                buffer[dstOffset] = 0
                            }
                        }
                    }
                }
                initializedCount = cCount
            }
            return PlaneData420(width: img.width, height: img.height, y: y, cb: cb, cr: cr)
        }

        let cb = [Int16](unsafeUninitializedCapacity: img.cbPlane.count) { (buffer: inout UnsafeMutableBufferPointer<Int16>, initializedCount: inout Int) in
            img.cbPlane.withUnsafeBufferPointer { (src: UnsafeBufferPointer<UInt8>) in
                for i in 0..<src.count {
                    buffer[i] = (Int16(src[i]) - 128)
                }
            }
            initializedCount = img.cbPlane.count
        }
        let cr = [Int16](unsafeUninitializedCapacity: img.crPlane.count) { (buffer: inout UnsafeMutableBufferPointer<Int16>, initializedCount: inout Int) in
            img.crPlane.withUnsafeBufferPointer { (src: UnsafeBufferPointer<UInt8>) in
                for i in 0..<src.count {
                    buffer[i] = (Int16(src[i]) - 128)
                }
            }
            initializedCount = img.crPlane.count
        }
        return PlaneData420(width: img.width, height: img.height, y: y, cb: cb, cr: cr)
    }
}

@inline(__always)
func subPlanes(curr: PlaneData420, predicted: PlaneData420) async -> PlaneData420 {
    @Sendable @inline(__always)
    func sub(c: [Int16], p: [Int16]) -> [Int16] {
        let count = c.count
        if count < 1 { return [] }

        var res = [Int16](repeating: 0, count: count)
        c.withUnsafeBufferPointer { cBuf in
            p.withUnsafeBufferPointer { pBuf in
                res.withUnsafeMutableBufferPointer { resBuf in
                    guard let cPtr = cBuf.baseAddress,
                          let pPtr = pBuf.baseAddress,
                          let resPtr = resBuf.baseAddress else { return }
                    
                    var i = 0
                    #if arch(arm64) || arch(x86_64) || arch(wasm32)
                    let chunk = 16
                    while i + chunk <= count {
                        let cSimd = UnsafeRawPointer(cPtr.advanced(by: i)).load(as: SIMD16<Int16>.self)
                        let pSimd = UnsafeRawPointer(pPtr.advanced(by: i)).load(as: SIMD16<Int16>.self)
                        let diffSimd = cSimd &- pSimd
                        UnsafeMutableRawPointer(resPtr.advanced(by: i)).storeBytes(of: diffSimd, as: SIMD16<Int16>.self)
                        i += chunk
                    }
                    #endif
                    
                    while i < count {
                        resPtr[i] = cPtr[i] &- pPtr[i]
                        i += 1
                    }
                }
            }
        }
        return res
    }
    
    async let y = sub(c: curr.y, p: predicted.y)
    async let cb = sub(c: curr.cb, p: predicted.cb)
    async let cr = sub(c: curr.cr, p: predicted.cr)
    
    return PlaneData420(width: curr.width, height: curr.height, y: await y, cb: await cb, cr: await cr)
}

@inline(__always)
func addPlanes(residual: PlaneData420, predicted: PlaneData420) async -> PlaneData420 {
    @Sendable @inline(__always)
    func add(r: [Int16], p: [Int16]) -> [Int16] {
        let count = r.count
        if count < 1 { return [] }

        var curr = [Int16](repeating: 0, count: count)
        r.withUnsafeBufferPointer { rBuf in
            p.withUnsafeBufferPointer { pBuf in
                curr.withUnsafeMutableBufferPointer { cBuf in
                    guard let rPtr = rBuf.baseAddress,
                          let pPtr = pBuf.baseAddress,
                          let cPtr = cBuf.baseAddress else { return }
                    
                    var i = 0
                    #if arch(arm64) || arch(x86_64) || arch(wasm32)
                    let chunk = 16
                    while i + chunk <= count {
                        let rSimd = UnsafeRawPointer(rPtr.advanced(by: i)).load(as: SIMD16<Int16>.self)
                        let pSimd = UnsafeRawPointer(pPtr.advanced(by: i)).load(as: SIMD16<Int16>.self)
                        let sumSimd = rSimd &+ pSimd
                        UnsafeMutableRawPointer(cPtr.advanced(by: i)).storeBytes(of: sumSimd, as: SIMD16<Int16>.self)
                        i += chunk
                    }
                    #endif
                    
                    while i < count {
                        cPtr[i] = rPtr[i] &+ pPtr[i]
                        i += 1
                    }
                }
            }
        }
        return curr
    }
    
    async let y = add(r: residual.y, p: predicted.y)
    async let cb = add(r: residual.cb, p: predicted.cb)
    async let cr = add(r: residual.cr, p: predicted.cr)
    
    return PlaneData420(width: residual.width, height: residual.height, y: await y, cb: await cb, cr: await cr)
}

// MARK: - Utilities

@inline(__always)
func boundaryRepeat(_ width: Int, _ height: Int, _ px: Int, _ py: Int) -> (Int, Int) {
    var x = px
    var y = py
    
    if width <= x {
        x = (width - 1 - (x - width))
        if x <= -1 {
            x = 0
        }
    } else {
        if x <= -1 {
            x = (-1 * x)
            if width <= x {
                x = (width - 1)
            }
        }
    }
    
    if height <= y {
        y = (height - 1 - (y - height))
        if y <= -1 {
            y = 0
        }
    } else {
        if y <= -1 {
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
    if v <= -1 {
        return 0
    }
    if 255 < v {
        return 255
    }
    return UInt8(v)
}

// MARK: - Image Structures

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
        @inline(__always) get { width }
    }

    public var cStride: Int {
        @inline(__always) get {
            switch ratio {
            case .ratio420: return (width + 1) / 2
            case .ratio444: return width
            }
        }
    }
    
    public init(width: Int, height: Int, ratio: YCbCrRatio = .ratio420) {
        self.width = width
        self.height = height
        self.ratio = ratio
        self.yPlane = [UInt8](repeating: 0, count: (width * height))
        
        switch ratio {
        case .ratio420:
            let cw = (width + 1) / 2
            let ch = (height + 1) / 2
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
        return ((y * yStride) + x)
    }
    
    @inline(__always)
    public func cOffset(_ x: Int, _ y: Int) -> Int {
        return ((y * cStride) + x)
    }

    @inline(__always)
    private func getChromaSize(w: Int, h: Int, ratio: YCbCrRatio) -> (Int, Int) {
        switch ratio {
        case .ratio420:
            return ((w + 1) / 2, (h + 1) / 2)
        case .ratio444:
            return (w, h)
        }
    }

    public func resize(factor: Double) -> YCbCrImage {
        let newWidth = Int(Double(width) * factor)
        let newHeight = Int(Double(height) * factor)

        guard 0 < newWidth && 0 < newHeight else {
            return YCbCrImage(width: max(1, newWidth), height: max(1, newHeight), ratio: self.ratio)
        }

        var dstImg = YCbCrImage(width: newWidth, height: newHeight, ratio: self.ratio)

        boxResizePlane(
            src: self.yPlane, srcW: self.width, srcH: self.height, srcStride: self.yStride,
            dst: &dstImg.yPlane, dstW: dstImg.width, dstH: dstImg.height, dstStride: dstImg.yStride
        )

        let (srcCW, srcCH) = getChromaSize(w: self.width, h: self.height, ratio: self.ratio)
        let (dstCW, dstCH) = getChromaSize(w: dstImg.width, h: dstImg.height, ratio: dstImg.ratio)

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
        let scaleX = Double(srcW) / Double(dstW)
        let scaleY = Double(srcH) / Double(dstH)

        for dy in 0..<dstH {
            let syStart = Int(Double(dy) * scaleY)
            var syEnd = Int(Double(dy + 1) * scaleY)
            if srcH <= syEnd { syEnd = srcH }
            if syEnd <= syStart { syEnd = syStart + 1 }

            for dx in 0..<dstW {
                let sxStart = Int(Double(dx) * scaleX)
                var sxEnd = Int(Double(dx + 1) * scaleX)
                if srcW <= sxEnd { sxEnd = srcW }
                if sxEnd <= sxStart { sxEnd = sxStart + 1 }

                var sum: Int = 0
                var count: Int = 0

                for sy in syStart..<syEnd {
                    let rowOffset = sy * srcStride
                    for sx in sxStart..<sxEnd {
                        let srcIdx = rowOffset + sx
                        if srcIdx < src.count {
                            sum += Int(src[srcIdx])
                            count += 1
                        }
                    }
                }

                if 0 < count {
                    let dstIdx = (dy * dstStride) + dx
                    if dstIdx < dst.count {
                        dst[dstIdx] = UInt8(sum / count)
                    }
                }
            }
        }
    }
}

typealias RowFunc = (_ x: Int, _ y: Int, _ size: Int) -> [Int16]

struct ImageReader: Sendable {
    let img: YCbCrImage
    let width: Int
    let height: Int
    
    init(img: YCbCrImage) {
        self.img = img
        self.width = img.width
        self.height = img.height
    }
    
    @inline(__always)
    func rowY(x: Int, y: Int, size: Int) -> [Int16] {
        var plane = [Int16](repeating: 0, count: size)
        
        if 0 <= x && 0 <= y && y < height && (x + size) <= width {
             let offset = img.yOffset(x, y)
             img.yPlane.withUnsafeBufferPointer { srcPtr in
                 guard let srcBase = srcPtr.baseAddress else { return }
                 let rowStart = srcBase.advanced(by: offset)
                 for i in 0..<size {
                     plane[i] = Int16(rowStart[i]) - 128
                 }
             }
             return plane
        }
        
        img.yPlane.withUnsafeBufferPointer { srcPtr in
            plane.withUnsafeMutableBufferPointer { destPtr in
                guard let srcBase = srcPtr.baseAddress,
                      let destBase = destPtr.baseAddress else { return }
                
                for i in 0..<size {
                    let (px, py) = boundaryRepeat(width, height, (x + i), y)
                    let offset = img.yOffset(px, py)
                    destBase[i] = Int16(srcBase[offset]) - 128
                }
            }
        }
        return plane
    }

    @inline(__always)
    func rowCb444(x: Int, y: Int, size: Int) -> [Int16] {
        var plane = [Int16](repeating: 0, count: size)
        for i in 0..<size {
            let (rPx, rPy) = boundaryRepeat(width, height, ((x + i) * 2), (y * 2))
            
            let cPx = rPx
            let cPy = rPy
            let offset = img.cOffset(cPx, cPy)
            plane[i] = Int16(img.cbPlane[offset]) - 128
        }
        return plane
    }

    @inline(__always)
    private func rowCb420(x: Int, y: Int, size: Int) -> [Int16] {
        var plane = [Int16](repeating: 0, count: size)
        img.cbPlane.withUnsafeBufferPointer { srcPtr in
            plane.withUnsafeMutableBufferPointer { destPtr in
                guard let srcBase = srcPtr.baseAddress,
                      let destBase = destPtr.baseAddress else { return }
                
                for i in 0..<size {
                    let (rPx, rPy) = boundaryRepeat(width, height, ((x + i) * 2), (y * 2))
                    
                    let cPx = (rPx / 2)
                    let cPy = (rPy / 2)
                    let offset = img.cOffset(cPx, cPy)
                    destBase[i] = Int16(srcBase[offset]) - 128
                }
            }
        }
        return plane
    }

    @inline(__always)
    func rowCb(x: Int, y: Int, size: Int) -> [Int16] {
        if img.ratio == .ratio444 {
            return rowCb444(x: x, y: y, size: size)
        }
        return rowCb420(x: x, y: y, size: size)
    }

    @inline(__always)
    func rowCr444(x: Int, y: Int, size: Int) -> [Int16] {
        var plane = [Int16](repeating: 0, count: size)
        for i in 0..<size {
            let (rPx, rPy) = boundaryRepeat(width, height, ((x + i) * 2), (y * 2))
            
            let cPx = rPx
            let cPy = rPy
            let offset = img.cOffset(cPx, cPy)
            plane[i] = Int16(img.crPlane[offset]) - 128
        }
        return plane
    }

    @inline(__always)
    func rowCr420(x: Int, y: Int, size: Int) -> [Int16] {
        var plane = [Int16](repeating: 0, count: size)
        img.crPlane.withUnsafeBufferPointer { srcPtr in
            plane.withUnsafeMutableBufferPointer { destPtr in
                guard let srcBase = srcPtr.baseAddress,
                      let destBase = destPtr.baseAddress else { return }
                
                for i in 0..<size {
                    let (rPx, rPy) = boundaryRepeat(width, height, ((x + i) * 2), (y * 2))
                    
                    let cPx = (rPx / 2)
                    let cPy = (rPy / 2)
                    let offset = img.cOffset(cPx, cPy)
                    destBase[i] = Int16(srcBase[offset]) - 128
                }
            }
        }
        return plane
    }

    @inline(__always)
    func rowCr(x: Int, y: Int, size: Int) -> [Int16] {
        if img.ratio == .ratio444 {
            return rowCr444(x: x, y: y, size: size)
        }
        return rowCr420(x: x, y: y, size: size)
    }
}

struct Image16: Sendable {
    var y: [Int16]      // フラット配列: height * width
    var cb: [Int16]     // フラット配列: cHeight * cWidth
    var cr: [Int16]     // フラット配列: cHeight * cWidth
    let width: Int
    let height: Int
    
    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.y = [Int16](repeating: 0, count: width * height)
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        self.cb = [Int16](repeating: 0, count: cWidth * cHeight)
        self.cr = [Int16](repeating: 0, count: cWidth * cHeight)
    }
    
    @inline(__always)
    func getY(x: Int, y yPos: Int, size: Int) -> Block2D {
        var block = Block2D(width: size, height: size)
        block.withView { v in
            self.y.withUnsafeBufferPointer { srcBuf in
                guard let srcBase = srcBuf.baseAddress else { return }
                for h in 0..<size {
                    let dstPtr = v.rowPointer(y: h)
                    for w in 0..<size {
                        let (px, py) = boundaryRepeat(width, height, (x + w), (yPos + h))
                        dstPtr[w] = srcBase[py * width + px]
                    }
                }
            }
        }
        return block
    }
    
    @inline(__always)
    func getCb(x: Int, y yPos: Int, size: Int) -> Block2D {
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        var block = Block2D(width: size, height: size)
        block.withView { v in
            self.cb.withUnsafeBufferPointer { srcBuf in
                guard let srcBase = srcBuf.baseAddress else { return }
                for h in 0..<size {
                    let dstPtr = v.rowPointer(y: h)
                    for w in 0..<size {
                        let (px, py) = boundaryRepeat(cWidth, cHeight, (x + w), (yPos + h))
                        dstPtr[w] = srcBase[py * cWidth + px]
                    }
                }
            }
        }
        return block
    }
    
    @inline(__always)
    func getCr(x: Int, y yPos: Int, size: Int) -> Block2D {
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        var block = Block2D(width: size, height: size)
        block.withView { v in
            self.cr.withUnsafeBufferPointer { srcBuf in
                guard let srcBase = srcBuf.baseAddress else { return }
                for h in 0..<size {
                    let dstPtr = v.rowPointer(y: h)
                    for w in 0..<size {
                        let (px, py) = boundaryRepeat(cWidth, cHeight, (x + w), (yPos + h))
                        dstPtr[w] = srcBase[py * cWidth + px]
                    }
                }
            }
        }
        return block
    }
    
    @inline(__always)
    mutating func updateY(data: inout Block2D, startX: Int, startY: Int, size: Int) {
        let validStartY = max(0, startY)
        let validStartX = max(0, startX)
        let validEndY = min(height, startY + size)
        let validEndX = min(width, startX + size)
        
        let loopH = validEndY - validStartY
        let loopW = validEndX - validStartX
        
        if loopH <= 0 || loopW <= 0 { return }
        
        let dataOffsetY = validStartY - startY
        let dataOffsetX = validStartX - startX
        
        self.y.withUnsafeMutableBufferPointer { destBuf in
            guard let destBase = destBuf.baseAddress else { return }
            data.withView { v in
                for h in 0..<loopH {
                    let srcPtr = v.rowPointer(y: dataOffsetY + h)
                    let destPtr = destBase.advanced(by: (validStartY + h) * width + validStartX)
                    destPtr.update(from: srcPtr.advanced(by: dataOffsetX), count: loopW)
                }
            }
        }
    }
    
    @inline(__always)
    mutating func updateCb(data: inout Block2D, startX: Int, startY: Int, size: Int) {
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        
        let validStartY = max(0, startY)
        let validStartX = max(0, startX)
        let validEndY = min(cHeight, startY + size)
        let validEndX = min(cWidth, startX + size)
        
        let loopH = validEndY - validStartY
        let loopW = validEndX - validStartX
        
        if loopH <= 0 || loopW <= 0 { return }
        
        let dataOffsetY = validStartY - startY
        let dataOffsetX = validStartX - startX
        
        self.cb.withUnsafeMutableBufferPointer { destBuf in
            guard let destBase = destBuf.baseAddress else { return }
            data.withView { v in
                for h in 0..<loopH {
                    let srcPtr = v.rowPointer(y: dataOffsetY + h)
                    let destPtr = destBase.advanced(by: (validStartY + h) * cWidth + validStartX)
                    destPtr.update(from: srcPtr.advanced(by: dataOffsetX), count: loopW)
                }
            }
        }
    }
    
    @inline(__always)
    mutating func updateCr(data: inout Block2D, startX: Int, startY: Int, size: Int) {
        let cWidth = (width + 1) / 2
        let cHeight = (height + 1) / 2
        
        let validStartY = max(0, startY)
        let validStartX = max(0, startX)
        let validEndY = min(cHeight, startY + size)
        let validEndX = min(cWidth, startX + size)
        
        let loopH = validEndY - validStartY
        let loopW = validEndX - validStartX
        
        if loopH <= 0 || loopW <= 0 { return }
        
        let dataOffsetY = validStartY - startY
        let dataOffsetX = validStartX - startX
        
        self.cr.withUnsafeMutableBufferPointer { destBuf in
            guard let destBase = destBuf.baseAddress else { return }
            data.withView { v in
                for h in 0..<loopH {
                    let srcPtr = v.rowPointer(y: dataOffsetY + h)
                    let destPtr = destBase.advanced(by: (validStartY + h) * cWidth + validStartX)
                    destPtr.update(from: srcPtr.advanced(by: dataOffsetX), count: loopW)
                }
            }
        }
    }
}
