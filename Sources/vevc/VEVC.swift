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

    @inline(__always)
    static func allocate(width: Int, height: Int, stride strideVal: Int? = nil) -> BlockView {
        let s = strideVal ?? width
        let count = s * height
        let ptr = UnsafeMutablePointer<Int16>.allocate(capacity: count)
        ptr.initialize(repeating: 0, count: count)
        return BlockView(base: ptr, width: width, height: height, stride: s)
    }

    @inline(__always)
    func deallocate() {
        base.deinitialize(count: stride * height)
        base.deallocate()
    }
}

@inline(__always)
func clearBlockRegion(base: UnsafeMutablePointer<Int16>, width: Int, height: Int, stride: Int) {
    if stride == width {
        let total = width * height
        UnsafeMutableRawPointer(base).initializeMemory(as: UInt8.self, repeating: 0, count: total * 2)
        return
    }
    var i = 0
    let zero16 = SIMD16<Int16>.zero
    for y in 0..<height {
        let ptr = base.advanced(by: y * stride)
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

// MARK: - BlockViewPool


final class BaseBlockViewPool: @unchecked Sendable {
    private var pools: [Int: [BlockView]] = [:]
    private var int16Pools: [Int: [[Int16]]] = [:]
    
    private let maxPerSize: Int
    
    // no lock Wasm is single thread
    #if !arch(wasm32)
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

        for (_, blocks) in pools {
            for block in blocks {
                block.deallocate()
            }
        }
    }

    @inline(__always)
    func get(width: Int, height: Int) -> BlockView {
        let key = width * height
        
        #if arch(wasm32)
        if var bucket = pools[key], bucket.isEmpty != true {
            let block = bucket.removeLast()
            pools[key] = bucket
            block.clearAll()
            return block
        }
        #else
        os_unfair_lock_lock(lock)
        if var bucket = pools[key], bucket.isEmpty != true {
            let block = bucket.removeLast()
            pools[key] = bucket
            os_unfair_lock_unlock(lock)
            block.clearAll()
            return block
        }
        os_unfair_lock_unlock(lock)
        #endif
        
        return BlockView.allocate(width: width, height: height)
    }
    
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
        } else {
            block.deallocate()
        }
        os_unfair_lock_unlock(lock)
        #endif
    }
    
    @inline(__always)
    func putAll(_ blocks: [BlockView]) {
        for block in blocks {
            put(block)
        }
    }

    @inline(__always)
    func getInt16(count: Int) -> [Int16] {
        #if arch(wasm32)
        if var bucket = int16Pools[count], bucket.isEmpty != true {
            let arr = bucket.removeLast()
            int16Pools[count] = bucket
            return arr
        }
        #else
        os_unfair_lock_lock(lock)
        if var bucket = int16Pools[count], bucket.isEmpty != true {
            let arr = bucket.removeLast()
            int16Pools[count] = bucket
            os_unfair_lock_unlock(lock)
            return arr
        }
        os_unfair_lock_unlock(lock)
        #endif
        return [Int16](unsafeUninitializedCapacity: count) { _, c in c = count }
    }
    
    @inline(__always)
    func putInt16(_ array: [Int16]) {
        let count = array.count
        #if arch(wasm32)
        var bucket = int16Pools[count] ?? []
        if bucket.count < maxPerSize {
            bucket.append(array)
            int16Pools[count] = bucket
        }
        #else
        os_unfair_lock_lock(lock)
        var bucket = int16Pools[count] ?? []
        if bucket.count < maxPerSize {
            bucket.append(array)
            int16Pools[count] = bucket
        }
        os_unfair_lock_unlock(lock)
        #endif
    }
}

#if !arch(wasm32)
@inline(__always)
private func currentThreadShardIndex(shardCount: Int) -> Int {
    var tid: UInt64 = 0
    pthread_threadid_np(nil, &tid)
    tid = (tid ^ (tid >> 30)) &* 0xbf58476d1ce4e5b9
    tid = (tid ^ (tid >> 27)) &* 0x94d049bb133111eb
    tid = tid ^ (tid >> 31)
    return Int(tid % UInt64(shardCount))
}
#endif

// Sharded pattern proxy
final class BlockViewPool: @unchecked Sendable {
    #if arch(wasm32)
    private let pool: BaseBlockViewPool
    #else
    private let shardCount: Int
    private let shards: [BaseBlockViewPool]
    #endif

    init(shardCount: Int = 32, maxPerSize: Int = 256) {
        #if arch(wasm32)
        self.pool = BaseBlockViewPool(maxPerSize: maxPerSize)
        #else
        self.shardCount = shardCount
        self.shards = (0..<shardCount).map { _ in BaseBlockViewPool(maxPerSize: maxPerSize) }
        #endif
    }

    @inline(__always)
    func get(width: Int, height: Int) -> BlockView {
        #if arch(wasm32)
        return pool.get(width: width, height: height)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        return shards[idx].get(width: width, height: height)
        #endif
    }

    @inline(__always)
    func put(_ block: BlockView) {
        #if arch(wasm32)
        pool.put(block)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        shards[idx].put(block)
        #endif
    }
    
    @inline(__always)
    func putAll(_ blocks: [BlockView]) {
        for block in blocks {
            put(block)
        }
    }

    @inline(__always)
    func getInt16(count: Int) -> [Int16] {
        #if arch(wasm32)
        return pool.getInt16(count: count)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        return shards[idx].getInt16(count: count)
        #endif
    }
    
    @inline(__always)
    func putInt16(_ array: [Int16]) {
        #if arch(wasm32)
        pool.putInt16(array)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        shards[idx].putInt16(array)
        #endif
    }
}
