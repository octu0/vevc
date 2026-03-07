import Foundation

struct Int16Reader {
    let data: [Int16]
    let width: Int
    let height: Int

    @inline(__always)
    func row(x: Int, y: Int, size: Int) -> [Int16] {
        var r = [Int16](repeating: 0, count: size)
        let safeY = min(y, height - 1)

        let limit = min(size, width - x)
        if limit > 0 {
            data.withUnsafeBufferPointer { ptr in
                let base = ptr.baseAddress!.advanced(by: safeY * width + x)
                r.withUnsafeMutableBufferPointer { dst in
                    dst.baseAddress!.update(from: base, count: limit)
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
    func fillRow(x: Int, y: Int, size: Int, dest: UnsafeMutablePointer<Int16>) {
        let safeY = min(y, height - 1)
        let limit = min(size, width - x)
        if limit > 0 {
            data.withUnsafeBufferPointer { ptr in
                let base = ptr.baseAddress!.advanced(by: safeY * width + x)
                dest.update(from: base, count: limit)
                if limit < size {
                    let lastVal = dest[limit - 1]
                    for i in limit..<size {
                        dest[i] = lastVal
                    }
                }
            }
        } else {
            let lastVal = data[safeY * width + (width - 1)]
            for i in 0..<size {
                dest[i] = lastVal
            }
        }
    }
}

public struct BlockView {
    public var base: UnsafeMutablePointer<Int16>
    public let width: Int
    public let height: Int
    public let stride: Int

    @inline(__always)
    public func rowPointer(y: Int) -> UnsafeMutablePointer<Int16> {
        return base.advanced(by: y * stride)
    }

    @inline(__always)
    public func setRow(offsetY: Int, row: [Int16]) {
        let ptr = rowPointer(y: offsetY)
        row.withUnsafeBufferPointer { src in
            ptr.update(from: src.baseAddress!, count: width)
        }
    }

    @inline(__always)
    func fillRow(offsetY: Int, from reader: Int16Reader, x: Int, y: Int) {
        let ptr = rowPointer(y: offsetY)
        reader.fillRow(x: x, y: y, size: width, dest: ptr)
    }
}

let width = 1920
let height = 1080
let size = 16
let iterations = 100

let data = [Int16](repeating: 1, count: width * height)
let reader = Int16Reader(data: data, width: width, height: height)

let blockDataSize = size * size
var blockDataOld = [Int16](repeating: 0, count: blockDataSize)
var blockDataNew = [Int16](repeating: 0, count: blockDataSize)

let startOld = CFAbsoluteTimeGetCurrent()
for _ in 0..<iterations {
    for h in stride(from: 0, to: height, by: size) {
        for w in stride(from: 0, to: width, by: size) {
            blockDataOld.withUnsafeMutableBufferPointer { ptr in
                var view = BlockView(base: ptr.baseAddress!, width: size, height: size, stride: size)
                for line in 0..<size {
                    let row = reader.row(x: w, y: (h + line), size: size)
                    view.setRow(offsetY: line, row: row)
                }
            }
        }
    }
}
let timeOld = CFAbsoluteTimeGetCurrent() - startOld

let startNew = CFAbsoluteTimeGetCurrent()
for _ in 0..<iterations {
    for h in stride(from: 0, to: height, by: size) {
        for w in stride(from: 0, to: width, by: size) {
            blockDataNew.withUnsafeMutableBufferPointer { ptr in
                var view = BlockView(base: ptr.baseAddress!, width: size, height: size, stride: size)
                for line in 0..<size {
                    view.fillRow(offsetY: line, from: reader, x: w, y: (h + line))
                }
            }
        }
    }
}
let timeNew = CFAbsoluteTimeGetCurrent() - startNew

print("--- Performance Benchmark ---")
print("Image Size: \(width)x\(height)")
print("Block Size: \(size)x\(size)")
print("Iterations: \(iterations)")
print("Old Method (Array Allocation): \(String(format: "%.4f", timeOld)) seconds")
print("New Method (Direct Pointer):   \(String(format: "%.4f", timeNew)) seconds")
print("Speedup: \(String(format: "%.2fx", timeOld / timeNew))")
