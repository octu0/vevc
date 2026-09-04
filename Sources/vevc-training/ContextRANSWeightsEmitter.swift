import Foundation

// MARK: - rANSContextWeights 生成・検証 Emitter

public struct QuantizedContextModels {
    public let invScalesQ: [Int32] // [16]
    public let b2Q: [Int32]        // [16]
    public let dims: [Int32]       // [16]
    public let w2Q: [[Int32]]      // [16][32]
    public let b1Q: [[Int32]]      // [16][32]
    public let w1FlatQ: [[Int32]]  // [16][32 * inDim]

    public init(
        invScalesQ: [Int32],
        b2Q: [Int32],
        dims: [Int32],
        w2Q: [[Int32]],
        b1Q: [[Int32]],
        w1FlatQ: [[Int32]]
    ) {
        self.invScalesQ = invScalesQ
        self.b2Q = b2Q
        self.dims = dims
        self.w2Q = w2Q
        self.b1Q = b1Q
        self.w1FlatQ = w1FlatQ
    }

    /// フラットな [Int32] blob 配列 (26,352 要素) に変換する
    public func toBlob() -> [Int32] {
        var blob: [Int32] = []
        blob.reserveCapacity(26352)

        for i in 0..<16 {
            blob.append(invScalesQ[i])
        }
        for i in 0..<16 {
            blob.append(b2Q[i])
        }
        for i in 0..<16 {
            blob.append(dims[i])
        }
        for pos in 4..<16 {
            let w2 = w2Q[pos]
            for h in 0..<32 {
                blob.append(w2[h])
            }
            let b1 = b1Q[pos]
            for h in 0..<32 {
                blob.append(b1[h])
            }
            let inDim = Int(dims[pos])
            let w1 = w1FlatQ[pos]
            for k in 0..<(32 * inDim) {
                blob.append(w1[k])
            }
        }
        return blob
    }
}

public final class ContextRANSWeightsEmitter {
    public init() {}

    /// 完全な Swift ソースコード文字列を生成する
    public func emitSwiftSource(models: QuantizedContextModels) -> String {
        let blob = models.toBlob()

        var s = ""
        s += "// Generated rANSContextWeights.swift - Q12 Fixed-Point MLP AR Weights\n"
        s += "// Target: Profile 2 L0 LL tail coefficients (pos 4..15)\n"
        s += "// Scale: Q12 (1.0 = 4096)\n\n"
        s += "final class rANSContextWeights: @unchecked Sendable {\n"
        s += "    static let shared = rANSContextWeights()\n\n"
        s += "    let invScalesQ: [Int32]\n"
        s += "    let b2Q: [Int32]\n"
        s += "    let inDims: [Int]\n"
        s += "    let w2Q: [[Int32]]\n"
        s += "    let b1Q: [[Int32]]\n"
        s += "    let w1FlatQ: [[Int32]]\n\n"
        s += "    private init() {\n"
        s += "        let a = rANSContextWeightsData.blob\n"
        s += "        var invS = [Int32](repeating: 0, count: 16)\n"
        s += "        var b2 = [Int32](repeating: 0, count: 16)\n"
        s += "        var dims = [Int](repeating: 0, count: 16)\n"
        s += "        var w2 = [[Int32]](repeating: [], count: 16)\n"
        s += "        var b1 = [[Int32]](repeating: [], count: 16)\n"
        s += "        var w1 = [[Int32]](repeating: [], count: 16)\n\n"
        s += "        var offset = 0\n"
        s += "        for i in 0..<16 {\n"
        s += "            invS[i] = a[offset]\n"
        s += "            offset += 1\n"
        s += "        }\n"
        s += "        for i in 0..<16 {\n"
        s += "            b2[i] = a[offset]\n"
        s += "            offset += 1\n"
        s += "        }\n"
        s += "        for i in 0..<16 {\n"
        s += "            dims[i] = Int(a[offset])\n"
        s += "            offset += 1\n"
        s += "        }\n"
        s += "        for pos in 4..<16 {\n"
        s += "            var w2Row = [Int32](repeating: 0, count: 32)\n"
        s += "            for h in 0..<32 {\n"
        s += "                w2Row[h] = a[offset]\n"
        s += "                offset += 1\n"
        s += "            }\n"
        s += "            w2[pos] = w2Row\n\n"
        s += "            var b1Row = [Int32](repeating: 0, count: 32)\n"
        s += "            for h in 0..<32 {\n"
        s += "                b1Row[h] = a[offset]\n"
        s += "                offset += 1\n"
        s += "            }\n"
        s += "            b1[pos] = b1Row\n\n"
        s += "            let inDim = dims[pos]\n"
        s += "            var w1Row = [Int32](repeating: 0, count: 32 * inDim)\n"
        s += "            for i in 0..<(32 * inDim) {\n"
        s += "                w1Row[i] = a[offset]\n"
        s += "                offset += 1\n"
        s += "            }\n"
        s += "            w1[pos] = w1Row\n"
        s += "        }\n\n"
        s += "        self.invScalesQ = invS\n"
        s += "        self.b2Q = b2\n"
        s += "        self.inDims = dims\n"
        s += "        self.w2Q = w2\n"
        s += "        self.b1Q = b1\n"
        s += "        self.w1FlatQ = w1\n"
        s += "    }\n"
        s += "}\n\n"
        s += "// Generated table: flat little-endian Int32 serialization of the trained\n"
        s += "// weights, in the exact layout `rANSContextWeights.init` reads. Do not\n"
        s += "// hand-edit.\n"
        s += "enum rANSContextWeightsData {\n"
        s += "    static let blob: [Int32] = [\n"

        var i = 0
        let total = blob.count
        while i < total {
            s += "    "
            let end = min(i + 12, total)
            var k = i
            while k < end {
                s += "\(blob[k]),"
                k += 1
            }
            s += "\n"
            i += 12
        }

        s += "    ]\n"
        s += "}\n"
        return s
    }

    /// パース round-trip 検証: blob が rANSContextWeights.init の規則で完全に復元できるか検証する
    public func verifyRoundTrip(models: QuantizedContextModels) throws {
        let blob = models.toBlob()
        if blob.count != 26352 {
            throw EmitterError.invalidBlobSize("Expected 26352 elements, got \(blob.count)")
        }

        let parsed = RANSContextWeightsContainer(blob: blob)

        // 1. invScalesQ 検証
        for pos in 0..<16 {
            if parsed.invScalesQ[pos] != models.invScalesQ[pos] {
                throw EmitterError.mismatch("invScalesQ[\(pos)] mismatch: parsed \(parsed.invScalesQ[pos]) != expected \(models.invScalesQ[pos])")
            }
        }

        // 2. b2Q 検証
        for pos in 0..<16 {
            if parsed.b2Q[pos] != models.b2Q[pos] {
                throw EmitterError.mismatch("b2Q[\(pos)] mismatch: parsed \(parsed.b2Q[pos]) != expected \(models.b2Q[pos])")
            }
        }

        // 3. inDims 検証
        for pos in 0..<16 {
            if parsed.inDims[pos] != Int(models.dims[pos]) {
                throw EmitterError.mismatch("dims[\(pos)] mismatch: parsed \(parsed.inDims[pos]) != expected \(models.dims[pos])")
            }
        }

        // 4. pos 4..15 の w2Q, b1Q, w1FlatQ 検証
        for pos in 4..<16 {
            let expW2 = models.w2Q[pos]
            let actW2 = parsed.w2Q[pos]
            for h in 0..<32 {
                if actW2[h] != expW2[h] {
                    throw EmitterError.mismatch("w2Q[\(pos)][\(h)] mismatch: \(actW2[h]) != \(expW2[h])")
                }
            }

            let expB1 = models.b1Q[pos]
            let actB1 = parsed.b1Q[pos]
            for h in 0..<32 {
                if actB1[h] != expB1[h] {
                    throw EmitterError.mismatch("b1Q[\(pos)][\(h)] mismatch: \(actB1[h]) != \(expB1[h])")
                }
            }

            let inDim = Int(models.dims[pos])
            let expW1 = models.w1FlatQ[pos]
            let actW1 = parsed.w1FlatQ[pos]
            let w1Count = 32 * inDim
            for k in 0..<w1Count {
                if actW1[k] != expW1[k] {
                    throw EmitterError.mismatch("w1FlatQ[\(pos)][\(k)] mismatch: \(actW1[k]) != \(expW1[k])")
                }
            }
        }
    }
}

public enum EmitterError: Error {
    case invalidBlobSize(String)
    case mismatch(String)
}
