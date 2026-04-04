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

struct BlockView: @unchecked Sendable {
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

    /// メモリ確保してBlockViewを生成する。使用後は必ずdeallocate()を呼ぶこと。
    /// Block2Dの代替として使用する。
    @inline(__always)
    static func allocate(width: Int, height: Int, stride strideVal: Int? = nil) -> BlockView {
        let s = strideVal ?? width
        let count = s * height
        let ptr = UnsafeMutablePointer<Int16>.allocate(capacity: count)
        ptr.initialize(repeating: 0, count: count)
        return BlockView(base: ptr, width: width, height: height, stride: s)
    }

    /// allocateで確保したメモリを解放する。
    @inline(__always)
    func deallocate() {
        base.deinitialize(count: stride * height)
        base.deallocate()
    }

    /// Block2D互換: .viewアクセスで自身を返す（既存コードとの互換性維持）
    @inline(__always)
    var view: BlockView {
        return self
    }
}

