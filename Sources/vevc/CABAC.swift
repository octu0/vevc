// MARK: - CABAC (Context-adaptive binary arithmetic coding)

import Foundation

// MARK: - ContextModel

public struct ContextModel {
    public var prob: Int = 16384 // 15-bit probability of being 1. Initial 0.5.

    public init() {
        self.prob = 16384
    }

    public var mps: Int {
        @inline(__always) get {
            return prob > 16384 ? 1 : 0
        }
    }

    public mutating func update(bit: Int) {
        let shift = 5 // adaptation speed
        if bit == 1 {
            prob += (32768 - prob) >> shift
        } else {
            prob -= prob >> shift
        }
    }

    public func getRLPS(range: Int) -> Int {
        let pLPS = prob > 16384 ? (32768 - prob) : prob
        // qP: 0 to 127. pLPS is [0, 16384]. 16384 >> 7 = 128.
        let qP = min(127, pLPS >> 7)
        let qR = (range >> 6) & 3
        return ContextModel.rangeLPS_table[qP][qR]
    }

    private static let rangeLPS_table: [[Int]] = [
        [2, 2, 2, 2], [2, 2, 2, 3], [3, 3, 4, 5], [4, 5, 6, 7],
        [5, 6, 7, 8], [6, 8, 9, 10], [7, 9, 11, 12], [8, 10, 12, 14],
        [10, 12, 14, 16], [11, 13, 15, 18], [12, 14, 17, 20], [13, 16, 19, 22],
        [14, 17, 20, 23], [15, 19, 22, 25], [16, 20, 24, 27], [17, 21, 25, 29],
        [19, 23, 27, 31], [20, 24, 28, 33], [21, 25, 30, 35], [22, 27, 32, 37],
        [23, 28, 33, 38], [24, 30, 35, 40], [25, 31, 37, 42], [26, 32, 38, 44],
        [28, 34, 40, 46], [29, 35, 41, 48], [30, 36, 43, 50], [31, 38, 45, 52],
        [32, 39, 46, 53], [33, 41, 48, 55], [34, 42, 50, 57], [35, 43, 51, 59],
        [37, 45, 53, 61], [38, 46, 54, 63], [39, 47, 56, 65], [40, 49, 58, 67],
        [41, 50, 59, 68], [42, 52, 61, 70], [43, 53, 63, 72], [44, 54, 64, 74],
        [46, 56, 66, 76], [47, 57, 67, 78], [48, 58, 69, 80], [49, 60, 71, 82],
        [50, 61, 72, 83], [51, 63, 74, 85], [52, 64, 76, 87], [53, 65, 77, 89],
        [55, 67, 79, 91], [56, 68, 80, 93], [57, 69, 82, 95], [58, 71, 84, 97],
        [59, 72, 85, 98], [60, 74, 87, 100], [61, 75, 89, 102], [62, 76, 90, 104],
        [64, 78, 92, 106], [65, 79, 93, 108], [66, 80, 95, 110], [67, 82, 97, 112],
        [68, 83, 98, 113], [69, 85, 100, 115], [70, 86, 102, 117], [71, 87, 103, 119],
        [73, 89, 105, 121], [74, 90, 106, 123], [75, 91, 108, 125], [76, 93, 110, 127],
        [77, 94, 111, 128], [78, 96, 113, 130], [79, 97, 115, 132], [80, 98, 116, 134],
        [82, 100, 118, 136], [83, 101, 119, 138], [84, 102, 121, 140], [85, 104, 123, 142],
        [86, 105, 124, 143], [87, 107, 126, 145], [88, 108, 128, 147], [89, 109, 129, 149],
        [91, 111, 131, 151], [92, 112, 132, 153], [93, 113, 134, 155], [94, 115, 136, 157],
        [95, 116, 137, 158], [96, 118, 139, 160], [97, 119, 141, 162], [98, 120, 142, 164],
        [100, 122, 144, 166], [101, 123, 145, 168], [102, 124, 147, 170], [103, 126, 149, 172],
        [104, 127, 150, 173], [105, 129, 152, 175], [106, 130, 154, 177], [107, 131, 155, 179],
        [109, 133, 157, 181], [110, 134, 158, 183], [111, 135, 160, 185], [112, 137, 162, 187],
        [113, 138, 163, 188], [114, 140, 165, 190], [115, 141, 167, 192], [116, 142, 168, 194],
        [118, 144, 170, 196], [119, 145, 171, 198], [120, 146, 173, 200], [121, 148, 175, 202],
        [122, 149, 176, 203], [123, 151, 178, 205], [124, 152, 180, 207], [125, 153, 181, 209],
        [127, 155, 183, 211], [128, 156, 184, 213], [129, 157, 186, 215], [130, 159, 188, 217],
        [131, 160, 189, 218], [132, 162, 191, 220], [133, 163, 193, 222], [134, 164, 194, 224],
        [136, 166, 196, 226], [137, 167, 197, 228], [138, 168, 199, 230], [139, 170, 201, 232],
        [140, 171, 202, 233], [141, 173, 204, 235], [142, 174, 206, 237], [143, 175, 207, 239]
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
