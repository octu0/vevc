import Foundation
#if canImport(os)
import os
#endif

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
        if stride == width {
            let total = width * height
            UnsafeMutableRawPointer(base).initializeMemory(as: UInt8.self, repeating: 0, count: total * 2)
            return
        }
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

// MARK: - BlockViewPool

final class BlockViewPool: @unchecked Sendable {
    /// サイズ別のプール。キーは width * height（8x8=64, 16x16=256, 32x32=1024）
    private var pools: [Int: [BlockView]] = [:]
    
    /// サイズ別の保持上限（超過分は即時 deallocate）
    private let maxPerSize: Int
    
    #if arch(wasm32)
    // Wasm は単一スレッド環境のためロック不要
    #else
    private let lock = UnsafeMutablePointer<os_unfair_lock_s>.allocate(capacity: 1)
    #endif
    
    init(maxPerSize: Int = 256) {
        self.maxPerSize = maxPerSize
        #if !arch(wasm32)
        lock.initialize(to: os_unfair_lock())
        #endif
    }
    
    deinit {
        #if !arch(wasm32)
        lock.deinitialize(count: 1)
        lock.deallocate()
        #endif
        // 全プール内のブロックを解放
        for (_, blocks) in pools {
            for block in blocks {
                block.deallocate()
            }
        }
    }
    
    /// プールからBlockViewを取得する。プールに在庫があれば再利用し、なければ新規確保する。
    /// 返却されたブロックはゼロクリア済みであることを保証する。
    @inline(__always)
    func get(width: Int, height: Int) -> BlockView {
        let key = width * height
        
        #if arch(wasm32)
        if var bucket = pools[key], !bucket.isEmpty {
            let block = bucket.removeLast()
            pools[key] = bucket
            block.clearAll() // プール再利用時にゼロ保証
            return block
        }
        #else
        os_unfair_lock_lock(lock)
        if var bucket = pools[key], !bucket.isEmpty {
            let block = bucket.removeLast()
            pools[key] = bucket
            os_unfair_lock_unlock(lock)
            block.clearAll() // ロック解除後にクリア（ロック保持時間を短縮）
            return block
        }
        os_unfair_lock_unlock(lock)
        #endif
        
        // プールに在庫がないため新規確保（initialize(repeating:0) でゼロ保証済み）
        return BlockView.allocate(width: width, height: height)
    }
    
    /// BlockViewをプールに返却する。クリアは get() 時に実行するため、ここではクリアしない。
    /// これにより並列実行時の put() のロック保持時間を最小化する。
    @inline(__always)
    func put(_ block: BlockView) {
        let key = block.width * block.height
        
        #if arch(wasm32)
        var bucket = pools[key] ?? []
        if bucket.count < maxPerSize {
            bucket.append(block)
            pools[key] = bucket
        } else {
            block.deallocate()
        }
        #else
        os_unfair_lock_lock(lock)
        var bucket = pools[key] ?? []
        if bucket.count < maxPerSize {
            bucket.append(block)
            pools[key] = bucket
            os_unfair_lock_unlock(lock)
        } else {
            os_unfair_lock_unlock(lock)
            block.deallocate()
        }
        #endif
    }
    
    /// BlockView配列を一括でプールに返却する。長寿命ブロック（配列で受け渡されるもの）用。
    @inline(__always)
    func putAll(_ blocks: [BlockView]) {
        for block in blocks {
            put(block)
        }
    }
}
