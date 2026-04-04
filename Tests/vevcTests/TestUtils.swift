import Foundation
@testable import vevc

extension BlockView {
    var data: [Int16] {
        Array(UnsafeBufferPointer(start: base, count: stride * height))
    }

    func setData(_ values: [Int16]) {
        let copyCount = min(values.count, stride * height)
        values.withUnsafeBufferPointer { src in
            guard let srcBase = src.baseAddress else { return }
            base.update(from: srcBase, count: copyCount)
        }
    }
}
