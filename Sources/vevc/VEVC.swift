import Foundation

@inlinable
@inline(__always)
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
        for y in 0..<height {
            let ptr = rowPointer(y: y)
            for x in 0..<width {
                ptr[x] = 0
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
    
    mutating func clearAll() {
        self.data = [Int16](repeating: 0, count: self.data.count)
    }
}