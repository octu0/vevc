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

final class Block2D: @unchecked Sendable {
    let base: UnsafeMutablePointer<Int16>
    let width: Int
    let height: Int
    let stride: Int
    private let count: Int

    init(width: Int, height: Int, stride: Int? = nil) {
        self.width = width
        self.height = height
        self.stride = stride ?? width
        self.count = self.stride * height
        self.base = .allocate(capacity: self.count)
        self.base.initialize(repeating: 0, count: self.count)
    }

    deinit {
        base.deinitialize(count: count)
        base.deallocate()
    }

    /// BlockView互換の直接subscriptアクセス
    @inline(__always)
    subscript(y: Int, x: Int) -> Int16 {
        get { base[(y * stride) + x] }
        set { base[(y * stride) + x] = newValue }
    }

    /// BlockView互換の行ポインタ取得
    @inline(__always)
    func rowPointer(y: Int) -> UnsafeMutablePointer<Int16> {
        return base.advanced(by: y * stride)
    }

    /// 後方互換: BlockView経由のアクセス（段階的に廃止予定）
    @inline(__always)
    func withView<R>(_ body: (inout BlockView) throws -> R) rethrows -> R {
        var view = BlockView(base: base, width: width, height: height, stride: stride)
        return try body(&view)
    }

    /// BlockViewを動的に生成
    @inline(__always)
    var view: BlockView {
        return BlockView(base: base, width: width, height: height, stride: stride)
    }


    @inline(__always)
    func clearAll() {
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