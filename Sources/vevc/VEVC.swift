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

@inlinable @inline(__always)
func statsLog(_ action: @autoclosure () -> Void) {
#if DEBUG
    if ProcessInfo.processInfo.environment["VEVC_STATS"] != nil {
        action()
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
    if width == stride {
        memset(base, 0, width * height * 2)
    } else {
        for y in 0..<height {
            memset(base.advanced(by: y * stride), 0, width * 2)
        }
    }
}

// MARK: - PlatformLock

struct PlatformLock {
    #if canImport(os)
    private let _lock: UnsafeMutablePointer<os_unfair_lock_s>
    #else
    private let _lock: NSLock
    #endif

    init() {
        #if canImport(os)
        _lock = .allocate(capacity: 1)
        _lock.initialize(to: os_unfair_lock())
        #else
        _lock = NSLock()
        #endif
    }

    func deallocate() {
        #if canImport(os)
        _lock.deinitialize(count: 1)
        _lock.deallocate()
        #endif
    }

    @inline(__always)
    func lock() {
        #if canImport(os)
        os_unfair_lock_lock(_lock)
        #else
        _lock.lock()
        #endif
    }

    @inline(__always)
    func unlock() {
        #if canImport(os)
        os_unfair_lock_unlock(_lock)
        #else
        _lock.unlock()
        #endif
    }
}

// MARK: - BlockViewPool

final class BaseBlockViewPool: @unchecked Sendable {
    private var pools: [Int: [BlockView]] = [:]
    private var int16Pools: [Int: [[Int16]]] = [:]
    private var arrayPools: [Int: [[BlockView]]] = [:]
    private var pools1024: [BlockView] = []
    private var pools256: [BlockView] = []
    private var pools64: [BlockView] = []
    
    private let maxPerSize: Int
    
    // no lock Wasm is single thread
    #if !arch(wasm32)
    private let _lock = PlatformLock()
    #endif
    
    init(maxPerSize: Int = 256) {
        self.maxPerSize = maxPerSize
    }
    
    deinit {
        #if !arch(wasm32)
        _lock.deallocate()
        #endif

        for (_, blocks) in pools {
            for block in blocks {
                block.deallocate()
            }
        }
    }
    
    @inline(__always)
    func get1024() -> BlockView {
        #if arch(wasm32)
        if pools1024.isEmpty != true {
            let block = pools1024.removeLast()
            clearBlockRegion(base: block.base, width: block.width, height: block.height, stride: block.stride)
            return block
        }
        #else
        _lock.lock()
        if pools1024.isEmpty != true {
            let block = pools1024.removeLast()
            _lock.unlock()
            clearBlockRegion(base: block.base, width: block.width, height: block.height, stride: block.stride)
            return block
        }
        _lock.unlock()
        #endif
        let fresh = BlockView.allocate(width: 32, height: 32)
        clearBlockRegion(base: fresh.base, width: fresh.width, height: fresh.height, stride: fresh.stride)
        return fresh
    }

    @inline(__always)
    func get256() -> BlockView {
        #if arch(wasm32)
        if pools256.isEmpty != true {
            let block = pools256.removeLast()
            clearBlockRegion(base: block.base, width: block.width, height: block.height, stride: block.stride)
            return block
        }
        #else
        _lock.lock()
        if pools256.isEmpty != true {
            let block = pools256.removeLast()
            _lock.unlock()
            clearBlockRegion(base: block.base, width: block.width, height: block.height, stride: block.stride)
            return block
        }
        _lock.unlock()
        #endif
        let fresh = BlockView.allocate(width: 16, height: 16)
        clearBlockRegion(base: fresh.base, width: fresh.width, height: fresh.height, stride: fresh.stride)
        return fresh
    }

    @inline(__always)
    func get64() -> BlockView {
        #if arch(wasm32)
        if pools64.isEmpty != true {
            let block = pools64.removeLast()
            clearBlockRegion(base: block.base, width: block.width, height: block.height, stride: block.stride)
            return block
        }
        #else
        _lock.lock()
        if pools64.isEmpty != true {
            let block = pools64.removeLast()
            _lock.unlock()
            clearBlockRegion(base: block.base, width: block.width, height: block.height, stride: block.stride)
            return block
        }
        _lock.unlock()
        #endif
        let fresh = BlockView.allocate(width: 8, height: 8)
        clearBlockRegion(base: fresh.base, width: fresh.width, height: fresh.height, stride: fresh.stride)
        return fresh
    }

    @inline(__always)
    func put1024(_ block: BlockView) {
        #if arch(wasm32)
        if pools1024.count < 8000 {
            pools1024.append(block)
        } else {
            block.deallocate()
        }
        #else
        _lock.lock()
        if pools1024.count < 4096 {
            pools1024.append(block)
        } else {
            block.deallocate()
        }
        _lock.unlock()
        #endif
    }

    @inline(__always)
    func put256(_ block: BlockView) {
        #if arch(wasm32)
        if pools256.count < 30000 {
            pools256.append(block)
        } else {
            block.deallocate()
        }
        #else
        _lock.lock()
        if pools256.count < 16384 {
            pools256.append(block)
        } else {
            block.deallocate()
        }
        _lock.unlock()
        #endif
    }

    @inline(__always)
    func put64(_ block: BlockView) {
        #if arch(wasm32)
        if pools64.count < 100000 {
            pools64.append(block)
        } else {
            block.deallocate()
        }
        #else
        _lock.lock()
        if pools64.count < 65536 {
            pools64.append(block)
        } else {
            block.deallocate()
        }
        _lock.unlock()
        #endif
    }

    @inline(__always)
    func putBlockViewArray1024(_ array: [BlockView]) {
        for block in array { self.put1024(block) }
        let capacity = array.capacity
        var arr = array
        arr.removeAll(keepingCapacity: true)
        #if arch(wasm32)
        var bucket = arrayPools[capacity] ?? []
        if bucket.count < 16 {
            bucket.append(arr)
            arrayPools[capacity] = bucket
        }
        #else
        _lock.lock()
        if arrayPools[capacity] == nil { arrayPools[capacity] = [] }
        if arrayPools[capacity]!.count < 4096 { arrayPools[capacity]!.append(arr) }
        _lock.unlock()
        #endif
    }

    @inline(__always)
    func putBlockViewArray256(_ array: [BlockView]) {
        for block in array { self.put256(block) }
        let capacity = array.capacity
        var arr = array
        arr.removeAll(keepingCapacity: true)
        #if arch(wasm32)
        var bucket = arrayPools[capacity] ?? []
        if bucket.count < 16 {
            bucket.append(arr)
            arrayPools[capacity] = bucket
        }
        #else
        _lock.lock()
        if arrayPools[capacity] == nil { arrayPools[capacity] = [] }
        if arrayPools[capacity]!.count < 4096 { arrayPools[capacity]!.append(arr) }
        _lock.unlock()
        #endif
    }

    @inline(__always)
    func putBlockViewArray64(_ array: [BlockView]) {
        for block in array { self.put64(block) }
        let capacity = array.capacity
        var arr = array
        arr.removeAll(keepingCapacity: true)
        #if arch(wasm32)
        var bucket = arrayPools[capacity] ?? []
        if bucket.count < 16 {
            bucket.append(arr)
            arrayPools[capacity] = bucket
        }
        #else
        _lock.lock()
        if arrayPools[capacity] == nil { arrayPools[capacity] = [] }
        if arrayPools[capacity]!.count < 4096 { arrayPools[capacity]!.append(arr) }
        _lock.unlock()
        #endif
    }



    @inline(__always)
    func get(width: Int, height: Int) -> BlockView {
        let key = width * height
        
        #if arch(wasm32)
        if var bucket = pools[key], bucket.isEmpty != true {
            let block = bucket.removeLast()
            pools[key] = bucket
            clearBlockRegion(base: block.base, width: block.width, height: block.height, stride: block.stride)
            return block
        }
        #else
        _lock.lock()
        if let pool = pools[key], pool.isEmpty != true {
            let block = pools[key]!.removeLast()
            _lock.unlock()
            clearBlockRegion(base: block.base, width: block.width, height: block.height, stride: block.stride)
            return block
        }
        _lock.unlock()
        #endif
        
        let fresh = BlockView.allocate(width: width, height: height)
        clearBlockRegion(base: fresh.base, width: fresh.width, height: fresh.height, stride: fresh.stride)
        return fresh
    }
    
    @inline(__always)
    func put(_ block: BlockView) {
        let key = block.width * block.height
        
        #if arch(wasm32)
        var bucket = pools[key] ?? []
        let limit: Int
        switch key {
        case 1024: limit = 8000
        case 256: limit = 30000
        case 64: limit = 100000
        default: limit = maxPerSize
        }
        if bucket.count < limit {
            bucket.append(block)
            pools[key] = bucket
        } else {
            block.deallocate()
        }
        #else
        _lock.lock()
        if pools[key] == nil {
            pools[key] = []
        }
        let limit: Int
        switch key {
        case 1024: limit = 4096
        case 256: limit = 16384
        case 64: limit = 65536
        default: limit = 4096
        }
        if pools[key]!.count < limit {
            pools[key]!.append(block)
        } else {
            block.deallocate()
        }
        _lock.unlock()
        #endif
    }
    
    @inline(__always)
    func putAll(_ blocks: [BlockView]) {
        for block in blocks {
            put(block)
        }
    }

    @inline(__always)
    func getInt16(count: Int, zeroed: Bool = true) -> [Int16] {
        #if arch(wasm32)
        if var bucket = int16Pools[count], bucket.isEmpty != true {
            var arr = bucket.removeLast()
            int16Pools[count] = bucket
            if zeroed {
                arr.withUnsafeMutableBufferPointer { $0.baseAddress!.update(repeating: 0, count: count) }
            }
            return arr
        }
        #else
        _lock.lock()
        if let pool = int16Pools[count], pool.isEmpty != true {
            var arr = int16Pools[count]!.removeLast()
            _lock.unlock()
            if zeroed {
                arr.withUnsafeMutableBufferPointer { $0.baseAddress!.update(repeating: 0, count: count) }
            }
            return arr
        }
        _lock.unlock()
        #endif
        if zeroed {
            return [Int16](repeating: 0, count: count)
        } else {
            return [Int16](unsafeUninitializedCapacity: count) { _, c in c = count }
        }
    }
    
    @inline(__always)
    func putInt16(_ array: [Int16]) {
        let count = array.count
        #if arch(wasm32)
        var bucket = int16Pools[count] ?? []
        if bucket.count < 16 {
            bucket.append(array)
            int16Pools[count] = bucket
        }
        #else
        _lock.lock()
        if int16Pools[count] == nil {
            int16Pools[count] = []
        }
        if int16Pools[count]!.count < 4096 {
            int16Pools[count]!.append(array)
        }
        _lock.unlock()
        #endif
    }
    
    @inline(__always)
    func getBlockViewArray(capacity: Int) -> [BlockView] {
        #if arch(wasm32)
        if var bucket = arrayPools[capacity], bucket.isEmpty != true {
            let arr = bucket.removeLast()
            arrayPools[capacity] = bucket
            return arr
        }
        #else
        _lock.lock()
        if let pool = arrayPools[capacity], pool.isEmpty != true {
            let arr = arrayPools[capacity]!.removeLast()
            _lock.unlock()
            return arr
        }
        _lock.unlock()
        #endif
        var arr = [BlockView]()
        arr.reserveCapacity(capacity)
        return arr
    }

    @inline(__always)
    func putBlockViewArray(_ array: [BlockView]) {
        for block in array {
            self.put(block)
        }
        let capacity = array.capacity
        var arr = array
        arr.removeAll(keepingCapacity: true)
        
        #if arch(wasm32)
        var bucket = arrayPools[capacity] ?? []
        if bucket.count < 16 {
            bucket.append(arr)
            arrayPools[capacity] = bucket
        }
        #else
        _lock.lock()
        if arrayPools[capacity] == nil {
            arrayPools[capacity] = []
        }
        if arrayPools[capacity]!.count < 4096 {
            arrayPools[capacity]!.append(arr)
        }
        _lock.unlock()
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

    init(shardCount: Int = 4, maxPerSize: Int = 256) {
        #if arch(wasm32)
        self.pool = BaseBlockViewPool(maxPerSize: maxPerSize)
        #else
        self.shardCount = shardCount
        self.shards = (0..<shardCount).map { _ in BaseBlockViewPool(maxPerSize: maxPerSize) }
        #endif
    }

    



    @inline(__always)
    func get1024() -> BlockView {
        #if arch(wasm32)
        return pool.get1024()
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        return shards[idx].get1024()
        #endif
    }

    @inline(__always)
    func get256() -> BlockView {
        #if arch(wasm32)
        return pool.get256()
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        return shards[idx].get256()
        #endif
    }

    @inline(__always)
    func get64() -> BlockView {
        #if arch(wasm32)
        return pool.get64()
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        return shards[idx].get64()
        #endif
    }

    @inline(__always)
    func put1024(_ block: BlockView) {
        #if arch(wasm32)
        pool.put1024(block)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        shards[idx].put1024(block)
        #endif
    }

    @inline(__always)
    func put256(_ block: BlockView) {
        #if arch(wasm32)
        pool.put256(block)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        shards[idx].put256(block)
        #endif
    }

    @inline(__always)
    func put64(_ block: BlockView) {
        #if arch(wasm32)
        pool.put64(block)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        shards[idx].put64(block)
        #endif
    }

    @inline(__always)
    func putBlockViewArray1024(_ array: [BlockView]) {
        #if arch(wasm32)
        pool.putBlockViewArray1024(array)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        shards[idx].putBlockViewArray1024(array)
        #endif
    }

    @inline(__always)
    func putBlockViewArray256(_ array: [BlockView]) {
        #if arch(wasm32)
        pool.putBlockViewArray256(array)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        shards[idx].putBlockViewArray256(array)
        #endif
    }

    @inline(__always)
    func putBlockViewArray64(_ array: [BlockView]) {
        #if arch(wasm32)
        pool.putBlockViewArray64(array)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        shards[idx].putBlockViewArray64(array)
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
    func getInt16(count: Int, zeroed: Bool = true) -> [Int16] {
        #if arch(wasm32)
        return pool.getInt16(count: count, zeroed: zeroed)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        return shards[idx].getInt16(count: count, zeroed: zeroed)
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
    
    @inline(__always)
    func getBlockViewArray(capacity: Int) -> [BlockView] {
        #if arch(wasm32)
        return pool.getBlockViewArray(capacity: capacity)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        return shards[idx].getBlockViewArray(capacity: capacity)
        #endif
    }

    @inline(__always)
    func putBlockViewArray(_ array: [BlockView]) {
        #if arch(wasm32)
        pool.putBlockViewArray(array)
        #else
        let idx = currentThreadShardIndex(shardCount: shardCount)
        shards[idx].putBlockViewArray(array)
        #endif
    }
}

// MARK: - Bitstream inspection

/// Per-frame bitstream statistics as CSV: frame index, frame type (I/P/C),
/// luma quantizer steps per layer, per-layer payload sizes and skip-map
/// composition. This is the measurement view used by the quality/rate
/// tuning workflow (vevc-inspect CLI); it reads only container headers and
/// the skip map, never the coefficient payloads.
@inline(__always)
public func inspectBitstreamCSV(data: [UInt8]) throws -> String {
    var offset = 0
    var width = 0
    var height = 0
    var profile: UInt8 = 0x01
    var frameIdx = 0
    var csv = "frame,type,qy0,qy1,qy2,l0,l1,l2,total,skipPrev,skipLtr,inter,wpLuma,wpChroma,skipMapSize,mvsSize,refDirSize,treeMapSize\n"
    while offset < data.count {
        if offset + 4 <= data.count && Array(data[offset..<(offset + 4)]) == VEVCFileHeader.magic {
            let fh = try VEVCFileHeader.deserialize(from: data, offset: &offset)
            width = fh.width
            height = fh.height
            profile = fh.profile
            continue
        }
        let start = offset
        let fh = try VEVCFrameHeader.deserialize(from: data, offset: &offset, profile: profile)
        let headerSize = offset - start
        if fh.isCopyFrame {
            csv += "\(frameIdx),C,0,0,0,0,0,0,\(headerSize),0,0,0,0,0,0,0,0,0\n"
            offset = start + headerSize
            frameIdx += 1
            continue
        }
        var skipPrev = 0
        var skipLtr = 0
        var inter = 0
        if 0 < fh.skipMapSize {
            let bw = (width + 31) / 32
            let bh = (height + 31) / 32
            let smData = Array(data[offset..<(offset + fh.skipMapSize)])
            let map = try decodeSkipMap(data: smData, count: bw * bh)
            for m in map {
                switch m {
                case .skip_prev: skipPrev += 1
                case .skip_ltr: skipLtr += 1
                default: inter += 1
                }
            }
        }
        let l0Offset = offset + fh.skipMapSize + fh.mvsSize + fh.refDirSize + fh.treeMapSize
        let (qtY0, _, _, _, _) = try VEVCLayerData.deserialize(from: Array(data[l0Offset..<(l0Offset + fh.layer0Size)]), layer: 0, layerLabel: "Base8")
        let l1Offset = l0Offset + fh.layer0Size
        var qy1Step = 0
        if 0 < fh.layer1Size {
            let (qtY1, _, _, _, _) = try VEVCLayerData.deserialize(from: Array(data[l1Offset..<(l1Offset + fh.layer1Size)]), layer: 1, layerLabel: "Layer16")
            qy1Step = Int(qtY1.step)
        }
        let l2Offset = l1Offset + fh.layer1Size
        var qy2Step = 0
        if 0 < fh.layer2Size {
            let (qtY2, _, _, _, _) = try VEVCLayerData.deserialize(from: Array(data[l2Offset..<(l2Offset + fh.layer2Size)]), layer: 2, layerLabel: "Layer32")
            qy2Step = Int(qtY2.step)
        }
        let total = headerSize + fh.skipMapSize + fh.mvsSize + fh.refDirSize + fh.treeMapSize + fh.layer0Size + fh.layer1Size + fh.layer2Size
        let frameType = fh.isIFrame ? "I" : "P"
        csv += "\(frameIdx),\(frameType),\(qtY0.step),\(qy1Step),\(qy2Step),\(fh.layer0Size),\(fh.layer1Size),\(fh.layer2Size),\(total),\(skipPrev),\(skipLtr),\(inter),\(fh.lumaOffset),\(fh.chromaOffset),\(fh.skipMapSize),\(fh.mvsSize),\(fh.refDirSize),\(fh.treeMapSize)\n"
        offset = start + total
        frameIdx += 1
    }
    return csv
}

// MARK: - FrameRateConverter

/// Frame-rate conversion by frame repetition/dropping: feed input frames in
/// order and emit each one repeatCount() times.
public struct FrameRateConverter {
    public let inFps: Int
    public let outFps: Int
    private var acc: Int = 0

    public init(inFps: Int, outFps: Int) {
        precondition(inFps > 0 && outFps > 0, "inFps and outFps must be positive integers")
        self.inFps = inFps
        self.outFps = outFps
    }

    /// Output count for the next input frame (call order = input frame order).
    public mutating func repeatCount() -> Int {
        acc += outFps
        let count = acc / inFps
        acc %= inFps
        return count
    }
}




