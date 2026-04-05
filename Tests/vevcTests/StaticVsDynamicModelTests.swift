import Testing
@testable import vevc

// MARK: - Static vs Dynamic Entropy Model Tests
//
// Intent: The static rANS tables were trained on aggregate data (34M pairs from all subbands).
// Different subbands (HL: horizontal detail, LH: vertical detail, HH: diagonal detail)
// have different coefficient distributions. Dynamic models create per-stream frequency
// tables that match the actual data distribution, at the cost of frequency table headers
// (~120-200B per stream).
//
// Hypothesis: For streams with sufficient data (>200 pairs), Dynamic Model should produce
// smaller output than Static Model because the frequency tables can adapt to the specific
// subband's distribution.

struct StaticVsDynamicModelTests {

    // MARK: - Helpers

    /// Generate HL-like pairs: moderate runs, moderate values
    /// HL captures horizontal edges → asymmetric, positive-biased values
    private func generateHLPairs(count: Int) -> [(run: UInt32, val: Int16)] {
        var pairs: [(run: UInt32, val: Int16)] = []
        pairs.reserveCapacity(count)
        for i in 0..<count {
            let run: UInt32 = UInt32(i % 5) // runs 0-4
            let val: Int16 = Int16(1 + (i % 6)) // values 1-6
            pairs.append((run: run, val: (i % 3 == 0) ? -val : val))
        }
        return pairs
    }

    /// Generate HH-like pairs: long runs, very small values
    /// HH captures diagonal detail → very sparse, mostly ±1
    private func generateHHPairs(count: Int) -> [(run: UInt32, val: Int16)] {
        var pairs: [(run: UInt32, val: Int16)] = []
        pairs.reserveCapacity(count)
        for i in 0..<count {
            let run: UInt32 = UInt32(5 + (i % 12)) // runs 5-16 (very sparse)
            let val: Int16 = (i % 2 == 0) ? 1 : -1 // almost exclusively ±1
            pairs.append((run: run, val: val))
        }
        return pairs
    }

    /// Generate DPCM-like pairs: run=0 dominant, values concentrated at ±1
    private func generateDPCMPairs(count: Int) -> [(run: UInt32, val: Int16)] {
        var pairs: [(run: UInt32, val: Int16)] = []
        pairs.reserveCapacity(count)
        for i in 0..<count {
            let run: UInt32 = (i % 8 == 0) ? 1 : 0 // mostly run=0
            let val: Int16 = (i % 2 == 0) ? 1 : -1 // ±1
            pairs.append((run: run, val: val))
        }
        return pairs
    }

    /// Encode with a specific model and return the byte count
    private func encodeWithModel<M: EntropyModelProvider>(
        _ modelType: M.Type,
        pairs: [(run: UInt32, val: Int16)],
        trailingZeros: UInt32 = 0
    ) -> Int {
        var encoder = EntropyEncoder<M>()
        encoder.encodeBypass(binVal: 1) // hasNonZero
        encoder.encodeBypass(binVal: 1) // lscpX
        encoder.encodeBypass(binVal: 1) // lscpY
        for pair in pairs {
            encoder.addPair(run: pair.run, val: pair.val, isParentZero: false)
        }
        if trailingZeros > 0 {
            encoder.addTrailingZeros(trailingZeros)
        }
        encoder.flush()
        return encoder.getData().count
    }

    // MARK: - Test: Static vs Dynamic at various data sizes

    /// Intent: Compare Static and Dynamic model output sizes for different pair counts.
    /// Dynamic model has ~120-200B header overhead, so it should only win when streams
    /// are large enough for the rANS compression improvement to exceed the header cost.
    @Test func staticVsDynamicSizeComparison() {
        let testCounts = [50, 100, 200, 500, 1000, 2000]

        print("  === HL-like data (moderate runs, moderate values) ===")
        for count in testCounts {
            let pairs = generateHLPairs(count: count)
            let staticSize = encodeWithModel(StaticEntropyModel.self, pairs: pairs)
            let dynamicSize = encodeWithModel(DynamicEntropyModel.self, pairs: pairs)
            let diff = staticSize - dynamicSize
            let pct = Double(diff) / Double(staticSize) * 100
            print("  pairs=\(count) static=\(staticSize)B dynamic=\(dynamicSize)B diff=\(diff)B (\(String(format: "%+.1f", pct))%)")
        }

        print("  === HH-like data (long runs, ±1 values) ===")
        for count in testCounts {
            let pairs = generateHHPairs(count: count)
            let staticSize = encodeWithModel(StaticEntropyModel.self, pairs: pairs)
            let dynamicSize = encodeWithModel(DynamicEntropyModel.self, pairs: pairs)
            let diff = staticSize - dynamicSize
            let pct = Double(diff) / Double(staticSize) * 100
            print("  pairs=\(count) static=\(staticSize)B dynamic=\(dynamicSize)B diff=\(diff)B (\(String(format: "%+.1f", pct))%)")
        }

        print("  === DPCM-like data (run=0 dominant, ±1 values) ===")
        for count in testCounts {
            let pairs = generateDPCMPairs(count: count)
            let staticSize = encodeWithModel(StaticEntropyModel.self, pairs: pairs)
            let dynamicSize = encodeWithModel(DynamicEntropyModel.self, pairs: pairs)
            let diff = staticSize - dynamicSize
            let pct = Double(diff) / Double(staticSize) * 100
            print("  pairs=\(count) static=\(staticSize)B dynamic=\(dynamicSize)B diff=\(diff)B (\(String(format: "%+.1f", pct))%)")
        }
    }

    // MARK: - Test: Dynamic model roundtrip correctness

    /// Intent: Ensure Dynamic model encode/decode roundtrip produces identical data.
    @Test func dynamicModelRoundtrip() throws {
        let testData: [[(run: UInt32, val: Int16)]] = [
            generateHLPairs(count: 100),
            generateHHPairs(count: 100),
            generateDPCMPairs(count: 100),
        ]

        for (idx, pairs) in testData.enumerated() {
            var encoder = EntropyEncoder<DynamicEntropyModel>()
            encoder.encodeBypass(binVal: 1) // hasNonZero
            encoder.encodeBypass(binVal: 1) // lscpX
            encoder.encodeBypass(binVal: 1) // lscpY
            for pair in pairs {
                encoder.addPair(run: pair.run, val: pair.val, isParentZero: false)
            }
            encoder.flush()
            let data = encoder.getData()

            try data.withUnsafeBufferPointer { ptr in
                var decoder = try EntropyDecoder(base: ptr.baseAddress!, count: ptr.count)
                let hasNonZero = try decoder.decodeBypass()
                #expect(hasNonZero == 1)
                let _ = try decoder.decodeBypass() // lscpX
                let _ = try decoder.decodeBypass() // lscpY
                
                for (pIdx, original) in pairs.enumerated() {
                    let decoded = decoder.readPair(isParentZero: false)
                    #expect(decoded.run == Int(original.run),
                           "Dataset \(idx) pair \(pIdx): run mismatch expected=\(String(original.run)) got=\(String(decoded.run))")
                    #expect(decoded.val == original.val,
                           "Dataset \(idx) pair \(pIdx): val mismatch expected=\(String(original.val)) got=\(String(decoded.val))")
                }
            }
        }
    }

    // MARK: - Test: Overhead breakeven point

    /// Intent: Find the minimum pair count where Dynamic Model output size <= Static Model.
    /// Below this point, the frequency table header cost exceeds the compression benefit.
    @Test func dynamicModelBreakevenPoint() {
        print("  === Breakeven analysis (HL data) ===")
        var found = false
        for count in stride(from: 10, through: 500, by: 10) {
            let pairs = generateHLPairs(count: count)
            let staticSize = encodeWithModel(StaticEntropyModel.self, pairs: pairs)
            let dynamicSize = encodeWithModel(DynamicEntropyModel.self, pairs: pairs)
            if dynamicSize <= staticSize && !found {
                print("  ★ Breakeven at pairs=\(count): static=\(staticSize)B dynamic=\(dynamicSize)B")
                found = true
            }
        }
        if !found {
            print("  Dynamic model never beats static in tested range (10-500)")
        }

        found = false
        print("  === Breakeven analysis (HH data) ===")
        for count in stride(from: 10, through: 500, by: 10) {
            let pairs = generateHHPairs(count: count)
            let staticSize = encodeWithModel(StaticEntropyModel.self, pairs: pairs)
            let dynamicSize = encodeWithModel(DynamicEntropyModel.self, pairs: pairs)
            if dynamicSize <= staticSize && !found {
                print("  ★ Breakeven at pairs=\(count): static=\(staticSize)B dynamic=\(dynamicSize)B")
                found = true
            }
        }
        if !found {
            print("  Dynamic model never beats static in tested range (10-500)")
        }
    }
}
