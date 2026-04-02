import Foundation

@inlinable @inline(__always)
func debugLog(_ message: @autoclosure () -> String) {
#if DEBUG
    if ProcessInfo.processInfo.environment["VEVC_DEBUG"] != nil {
        fputs(message() + "\n", stderr)
    }
#endif
}

public enum ColorGamut: UInt8 {
    case bt709 = 1
    case bt2020 = 2
    case unspecified = 0
}

public enum Timescale: UInt8 {
    case ms1000 = 0
    case hz90000 = 1
}

// MARK: - BlockView

struct BlockView {
    var base: UnsafeMutablePointer<Int16>
    let width: Int
    let height: Int
    let stride: Int

    init(base: UnsafeMutablePointer<Int16>, width: Int, height: Int, stride: Int) {
        self.base = base
        self.width = width
        self.height = height
        self.stride = stride
    }

    @inline(__always)
    subscript(y: Int, x: Int) -> Int16 {
        get { base[(y * stride) + x] }
        set { base[(y * stride) + x] = newValue }
    }

    @inline(__always)
    func rowPointer(y: Int) -> UnsafeMutablePointer<Int16> {
        return base.advanced(by: y * stride)
    }

    @inline(__always)
    func setRow(offsetY: Int, row: [Int16]) {
        let ptr = rowPointer(y: offsetY)
        row.withUnsafeBufferPointer { src in
            guard let srcBase = src.baseAddress else { return }
            ptr.update(from: srcBase, count: width)
        }
    }

    @inline(__always)
    func clearAll() {
        var i = 0
        let zero16 = SIMD16<Int16>.zero
        for y in 0..<height {
            let ptr = rowPointer(y: y)
            i = 0
            while i + 16 <= width {
                UnsafeMutableRawPointer(ptr + i).storeBytes(of: zero16, as: SIMD16<Int16>.self)
                i += 16
            }
            while i < width {
                ptr[i] = 0
                i += 1
            }
        }
    }
}

// MARK: - Block2D

struct Block2D: Sendable {
    var data: [Int16]
    let width: Int
    let height: Int

    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.data = [Int16](repeating: 0, count: (width * height))
    }
    
    @inline(__always)
    mutating func withView<R>(_ body: (inout BlockView) throws -> R) rethrows -> R {
        return try data.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else { 
                fatalError("Block2D buffer is empty")
            }
            var view = BlockView(base: base, width: width, height: height, stride: width)
            return try body(&view)
        }
    }
    
    @inline(__always)
    mutating func clearAll() {
        data.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            let count = buf.count
            var i = 0
            let zero16 = SIMD16<Int16>.zero
            while i + 16 <= count {
                UnsafeMutableRawPointer(base + i).storeBytes(of: zero16, as: SIMD16<Int16>.self)
                i += 16
            }
            while i < count {
                base[i] = 0
                i += 1
            }
        }
    }
}

enum IntraMode: UInt8 {
    case dc = 0
    case horizontal = 1
    case vertical = 2
    case planar = 3
}

struct IntraContext {
    let blockSize: Int
    let colCount: Int
    
    var topBuffer: [Int16]
    var leftBuffer: [Int16]
    
    init(blockSize: Int, colCount: Int) {
        self.blockSize = blockSize
        self.colCount = colCount
        self.topBuffer = [Int16](repeating: 0, count: blockSize * colCount)
        self.leftBuffer = [Int16](repeating: 0, count: blockSize)
    }
    
    @inline(__always)
    mutating func resetLeft() {
        for i in 0..<blockSize {
            leftBuffer[i] = 0
        }
    }
    
    @inline(__always)
    mutating func update(col: Int, block: BlockView) {
        let topBase = col * blockSize
        for y in 0..<blockSize {
            leftBuffer[y] = block.rowPointer(y: y)[blockSize - 1]
        }
        let bottomRowPtr = block.rowPointer(y: blockSize - 1)
        for x in 0..<blockSize {
            topBuffer[topBase + x] = bottomRowPtr[x]
        }
    }
}

@inline(__always)
func predictIntra4(mode: IntraMode, col: Int, row: Int, ctx: IntraContext, ptrOut: UnsafeMutablePointer<Int16>) {
    let topBase = col * 4
    let hasTop = (row > 0)
    let hasLeft = (col > 0)
    
    switch mode {
    case .dc:
        var sum: Int32 = 0
        var count: Int32 = 0
        if hasTop {
            sum += Int32(ctx.topBuffer[topBase]) + Int32(ctx.topBuffer[topBase + 1]) + Int32(ctx.topBuffer[topBase + 2]) + Int32(ctx.topBuffer[topBase + 3])
            count += 4
        }
        if hasLeft {
            sum += Int32(ctx.leftBuffer[0]) + Int32(ctx.leftBuffer[1]) + Int32(ctx.leftBuffer[2]) + Int32(ctx.leftBuffer[3])
            count += 4
        }
        let dcVal = count > 0 ? Int16(sum / count) : 0
        for i in 0..<16 { ptrOut[i] = dcVal }
        
    case .horizontal:
        for y in 0..<4 {
            let leftVal = hasLeft ? ctx.leftBuffer[y] : 0
            for x in 0..<4 {
                ptrOut[y * 4 + x] = leftVal
            }
        }
        
    case .vertical:
        for y in 0..<4 {
            for x in 0..<4 {
                ptrOut[y * 4 + x] = hasTop ? ctx.topBuffer[topBase + x] : 0
            }
        }
        
    case .planar:
        if hasTop && hasLeft {
            for y in 0..<4 {
                for x in 0..<4 {
                    let wH = 4 - x
                    let wV = 4 - y
                    let val = (Int32(ctx.leftBuffer[y]) * Int32(wH) + Int32(ctx.topBuffer[topBase + x]) * Int32(wV)) / Int32(wH + wV)
                    ptrOut[y * 4 + x] = Int16(val)
                }
            }
        } else if hasTop {
            for y in 0..<4 {
                for x in 0..<4 {
                    ptrOut[y * 4 + x] = ctx.topBuffer[topBase + x]
                }
            }
        } else if hasLeft {
            for y in 0..<4 {
                let leftVal = ctx.leftBuffer[y]
                for x in 0..<4 {
                    ptrOut[y * 4 + x] = leftVal
                }
            }
        } else {
            for i in 0..<16 { ptrOut[i] = 0 }
        }
    }
}

