// MARK: - CABAC (Context-adaptive binary arithmetic coding)

import Foundation

// MARK: - ContextModel

public struct ContextModel {
    public var state: Int = 0 // 0 to 63
    public var mps: Int = 0   // 0 or 1

    public init() {
        self.state = 0
        self.mps = 0
    }

    private static let transIndexMPS: [Int] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 63,
    ]

    private static let transIndexLPS: [Int] = [
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
        8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15,
        16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23,
        24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31,
    ]

    public mutating func update(bit: Int) {
        if bit == mps {
            state = ContextModel.transIndexMPS[state]
        } else {
            if state == 0 {
                mps = (1 - mps)
            }
            state = ContextModel.transIndexLPS[state]
        }
    }

    public func getRLPS(range: Int) -> Int {
        let q = ((range >> 6) & 3)
        let rLPS = ContextModel.rangeLPS_table[state][q]
        return rLPS
    }

    private static let rangeLPS_table: [[Int]] = [
        [144, 176, 208, 240], [135, 165, 195, 226], [127, 155, 184, 212], [120, 146, 173, 199],
        [112, 137, 162, 187], [106, 129, 152, 176], [99, 121, 143, 165], [93, 114, 135, 155],
        [88, 107, 127, 146], [82, 101, 119, 137], [77, 95, 112, 129], [73, 89, 105, 121],
        [68, 84, 99, 114], [64, 79, 93, 107], [60, 74, 87, 101], [57, 69, 82, 95],
        [53, 65, 77, 89], [50, 61, 72, 84], [47, 58, 68, 78], [44, 54, 64, 74],
        [42, 51, 60, 69], [39, 48, 56, 65], [37, 45, 53, 61], [35, 42, 50, 58],
        [32, 40, 47, 54], [30, 37, 44, 51], [29, 35, 41, 48], [27, 33, 39, 45],
        [25, 31, 37, 42], [24, 29, 34, 40], [22, 27, 32, 37], [21, 26, 30, 35],
        [20, 24, 29, 33], [19, 23, 27, 31], [17, 21, 25, 29], [16, 20, 24, 27],
        [15, 19, 22, 26], [14, 18, 21, 24], [14, 17, 20, 23], [13, 16, 18, 21],
        [12, 15, 17, 20], [11, 14, 16, 19], [11, 13, 15, 18], [10, 12, 14, 17],
        [9, 11, 14, 16], [9, 11, 13, 15], [8, 10, 12, 14], [8, 10, 11, 13],
        [7, 9, 11, 12], [7, 8, 10, 11], [6, 8, 9, 11], [6, 7, 9, 10],
        [6, 7, 8, 10], [5, 7, 8, 9], [5, 6, 7, 8], [5, 6, 7, 8],
        [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 5, 6],
        [3, 4, 5, 6], [3, 4, 5, 5], [3, 4, 4, 5], [3, 4, 4, 5]
    ]
}

// MARK: - CABACEncoder

public struct CABACEncoder {
    private var range: Int = 510
    private var low: Int = 0
    private var bitsToFollow: Int = 0
    public var data: [UInt8] = []
    private var bitBuffer: Int = 0
    private var bitCount: Int = 0

    public init() {}

    public mutating func encode(bit: Int, context: inout ContextModel) {
        let rLPS = context.getRLPS(range: range)
        range = (range - rLPS)

        if (bit != context.mps) {
            low = (low + range)
            range = rLPS
        }
        context.update(bit: bit)

        renormalize()
    }

    public mutating func encodeBypass(bit: Int) {
        low = (low << 1)
        if (bit != 0) {
            low = (low + range)
        }

        if ((low & 1024) != 0) {
            writeBit(1)
            while (0 < bitsToFollow) {
                writeBit(0)
                bitsToFollow = (bitsToFollow - 1)
            }
            low = (low & 1023)
        } else if ((low & 512) != 0) {
            bitsToFollow = (bitsToFollow + 1)
            low = (low & 511)
        } else {
            writeBit(0)
            while (0 < bitsToFollow) {
                writeBit(1)
                bitsToFollow = (bitsToFollow - 1)
            }
        }
    }

    private mutating func renormalize() {
        while (range < 256) {
            if (low < 256) {
                writeBit(0)
                while (0 < bitsToFollow) {
                    writeBit(1)
                    bitsToFollow = (bitsToFollow - 1)
                }
            } else if (512 <= low) {
                writeBit(1)
                while (0 < bitsToFollow) {
                    writeBit(0)
                    bitsToFollow = (bitsToFollow - 1)
                }
                low = (low - 512)
            } else {
                bitsToFollow = (bitsToFollow + 1)
                low = (low - 256)
            }

            range = (range << 1)
            low = (low << 1)
        }
    }

    private mutating func writeBit(_ bit: Int) {
        bitBuffer = ((bitBuffer << 1) | (bit & 1))
        bitCount = (bitCount + 1)
        if (bitCount == 8) {
            data.append(UInt8(bitBuffer))
            bitCount = 0
            bitBuffer = 0
        }
    }

    public mutating func flush() {
        if (low < 256) {
            writeBit(0)
            writeBit(1)
            while (0 < bitsToFollow) {
                writeBit(1)
                bitsToFollow = (bitsToFollow - 1)
            }
        } else {
            writeBit(1)
            writeBit(0)
            while (0 < bitsToFollow) {
                writeBit(0)
                bitsToFollow = (bitsToFollow - 1)
            }
        }

        if (0 < bitCount) {
            bitBuffer = (bitBuffer << (8 - bitCount))
            data.append(UInt8(bitBuffer))
            bitCount = 0
            bitBuffer = 0
        }
    }
}

// MARK: - CABACDecoder

public struct CABACDecoder {
    private var range: Int = 510
    private var value: Int = 0
    private let data: [UInt8]
    private var offset: Int = 0
    private var bitBuffer: Int = 0
    private var bitCount: Int = 0

    public init(data: [UInt8]) {
        self.data = data
        for _ in 0..<10 {
            value = ((value << 1) | readBit())
        }
    }

    public mutating func decode(context: inout ContextModel) -> Int {
        let rLPS = context.getRLPS(range: range)
        range = (range - rLPS)

        let bit: Int
        if (value < range) {
            bit = context.mps
        } else {
            value = (value - range)
            bit = (1 - context.mps)
            range = rLPS
        }
        context.update(bit: bit)

        renormalize()
        return bit
    }

    public mutating func decodeBypass() -> Int {
        value = (value << 1)
        value = (value | readBit())
        var bit: Int = 0
        if (range <= value) {
            value = (value - range)
            bit = 1
        }
        return bit
    }

    private mutating func renormalize() {
        while (range < 256) {
            range = (range << 1)
            value = ((value << 1) | readBit())
        }
    }

    private mutating func readBit() -> Int {
        if (bitCount == 0) {
            if (offset < data.count) {
                bitBuffer = Int(data[offset])
                offset = (offset + 1)
                bitCount = 8
            } else {
                return 0
            }
        }
        bitCount = (bitCount - 1)
        return ((bitBuffer >> bitCount) & 1)
    }
}

// MARK: - Symbol Encoding/Decoding

extension CABACEncoder {
    public mutating func encodeVal(_ val: Int, ctxG1: inout [ContextModel]) {
        var v = val
        let limit = ctxG1.count
        for i in 0..<limit {
            if (v == i) {
                encode(bit: 1, context: &ctxG1[i])
                return
            }
            encode(bit: 0, context: &ctxG1[i])
        }
        v = (v - limit)

        var k: Int = 0
        while ((1 << k) <= v) {
            encodeBypass(bit: 1)
            v = (v - (1 << k))
            k = (k + 1)
        }
        encodeBypass(bit: 0)

        if (0 < k) {
            for i in (0..<k).reversed() {
                encodeBypass(bit: ((v >> i) & 1))
            }
        }
    }
}

extension CABACDecoder {
    public mutating func decodeVal(ctxG1: inout [ContextModel]) -> Int {
        let limit = ctxG1.count
        for i in 0..<limit {
            if (decode(context: &ctxG1[i]) == 1) {
                return i
            }
        }

        var k: Int = 0
        while (decodeBypass() == 1) {
            k = (k + 1)
        }

        var v: Int = 0
        if (0 < k) {
            for _ in 0..<k {
                v = ((v << 1) | decodeBypass())
            }
        }
        v = (v + ((1 << k) - 1))

        return (v + limit)
    }
}
