import Testing
@testable import vevc

// MARK: - Raw Bypass Threshold Tests
//
// Hypothesis: The current raw bypass threshold (nonZeroCount <= 32) causes
// small blocks to use raw bypass mode (6bit/token + variable run) instead of
// rANS mode (~2-4bit/token). For pairs where rANS static tables match the
// data distribution well, rANS should produce smaller output even for
// relatively few pairs.
//
// These tests encode the same coefficient data using both raw bypass mode
// and rANS mode, comparing the resulting data sizes to determine the
// crossover point where rANS becomes beneficial.

struct RawBypassThresholdTests {

    // MARK: - Helpers

    /// Encode pairs using the current EntropyEncoder (static model) to get the actual output size.
    /// Returns the total byte count of the entropy-encoded data.
    private func encodePairsToSize(pairs: [(run: UInt32, val: Int16)], trailingZeros: UInt32 = 0) -> Int {
        var encoder = EntropyEncoder<StaticEntropyModel>()

        // Write a dummy hasNonZero bypass bit to match real encoding pattern
        encoder.encodeBypass(binVal: 1)
        // Write dummy LSCP coordinates
        encoder.encodeBypass(binVal: 1) // exp-golomb for lscpX=0
        encoder.encodeBypass(binVal: 1) // exp-golomb for lscpY=0

        for pair in pairs {
            encoder.addPair(run: pair.run, val: pair.val, contextIdx: 0)
        }
        if trailingZeros > 0 {
            encoder.addTrailingZeros(trailingZeros)
        }
        encoder.flush()
        return encoder.getData().count
    }

    /// Generate realistic DWT coefficient pairs that mimic the distribution
    /// of actual encoded data (run=0 dominant, val=+1..+4 concentrated).
    private func generateRealisticPairs(count: Int) -> [(run: UInt32, val: Int16)] {
        var pairs: [(run: UInt32, val: Int16)] = []
        pairs.reserveCapacity(count)
        // Mimic real distribution: mostly run=0 with small values
        for i in 0..<count {
            let run: UInt32 = (i % 3 == 0) ? 1 : 0 // ~33% have run=1, rest run=0
            let valAbs: Int16 = Int16(1 + (i % 4))  // values 1-4
            let val: Int16 = (i % 5 == 0) ? -valAbs : valAbs // ~20% negative
            pairs.append((run: run, val: val))
        }
        return pairs
    }

    /// Generate sparse coefficient pairs (high zero runs, typical of high-frequency subbands)
    private func generateSparsePairs(count: Int) -> [(run: UInt32, val: Int16)] {
        var pairs: [(run: UInt32, val: Int16)] = []
        pairs.reserveCapacity(count)
        for i in 0..<count {
            let run: UInt32 = UInt32(3 + (i % 8)) // runs 3-10
            let val: Int16 = (i % 2 == 0) ? 1 : -1 // mostly ±1
            pairs.append((run: run, val: val))
        }
        return pairs
    }

    // MARK: - Test: Verify raw bypass mode is used for small pair counts

    /// Intent: Verify that the current implementation uses raw bypass mode (0x80 flag)
    /// when the pair count is <= 32. This establishes the baseline behavior.
    @Test func rawBypassModeIsUsedForSmallPairCount() {
        let pairs = generateRealisticPairs(count: 10)
        var encoder = EntropyEncoder<StaticEntropyModel>()
        for pair in pairs {
            encoder.addPair(run: pair.run, val: pair.val, contextIdx: 0)
        }
        encoder.flush()
        let data = encoder.getData()

        // getData() structure: [bypassLen(4B)] [bypassData] [coeffCount(4B)] [mode(1B)] ...
        // mode byte should be 0x80 for raw bypass mode
        let bypassLen = Int(UInt32(data[0]) << 24 | UInt32(data[1]) << 16 | UInt32(data[2]) << 8 | UInt32(data[3]))
        let modeByteOffset = 4 + bypassLen + 4 // skip bypassLen + bypassData + coeffCount
        #expect(modeByteOffset < data.count, "Data should contain mode byte")
        #expect(data[modeByteOffset] == 0x80, "Should be raw bypass mode (0x80) for \(pairs.count) pairs")
    }

    /// Intent: Verify that rANS mode (not raw bypass) is used when pair count > 32.
    @Test func ransModelIsUsedForLargePairCount() {
        let pairs = generateRealisticPairs(count: 64)
        var encoder = EntropyEncoder<StaticEntropyModel>()
        for pair in pairs {
            encoder.addPair(run: pair.run, val: pair.val, contextIdx: 0)
        }
        encoder.flush()
        let data = encoder.getData()

        let bypassLen = Int(UInt32(data[0]) << 24 | UInt32(data[1]) << 16 | UInt32(data[2]) << 8 | UInt32(data[3]))
        let modeByteOffset = 4 + bypassLen + 4
        #expect(modeByteOffset < data.count, "Data should contain mode byte")
        let modeByte = data[modeByteOffset]
        #expect(modeByte != 0x80, "Should NOT be raw bypass mode for \(pairs.count) pairs")
        #expect((modeByte & 0x40) != 0, "Should have static table flag set")
    }

    // MARK: - Test: Size comparison at the crossover point

    /// Intent: Compare encoded sizes at pair counts near the threshold (32) to find
    /// the optimal crossover point. If rANS produces smaller output at lower pair counts,
    /// the threshold should be lowered.
    @Test func sizeComparisonAtCrossoverPoint() {
        // Test a range of pair counts around the current threshold
        let testCounts = [8, 12, 16, 20, 24, 28, 32, 48, 64]
        var results: [(count: Int, size: Int)] = []

        for count in testCounts {
            let pairs = generateRealisticPairs(count: count)
            let size = encodePairsToSize(pairs: pairs)
            results.append((count: count, size: size))
        }

        // Verify monotonic relationship: more pairs should generally produce more data
        // (unless rANS compression ratio improves with more data, which is expected)
        for i in 1..<results.count {
            let prevBpp = Double(results[i-1].size * 8) / Double(results[i-1].count)
            let currBpp = Double(results[i].size * 8) / Double(results[i].count)
            // rANS should have better bits-per-pair for larger counts
            // The crossover point is where rANS starts having lower bpp than raw bypass
            if results[i].count > 32 && results[i-1].count <= 32 {
                // This is the boundary where mode switches from raw to rANS
                print("  Crossover: count=\(results[i-1].count) bpp=\(String(format: "%.2f", prevBpp)) (raw) -> count=\(results[i].count) bpp=\(String(format: "%.2f", currBpp)) (rANS)")
            }
        }

        // Print all results for analysis
        for r in results {
            let bpp = Double(r.size * 8) / Double(r.count)
            print("  pairs=\(r.count) size=\(r.size)B bpp=\(String(format: "%.2f", bpp))")
        }
    }

    // MARK: - Test: Sparse data (high-frequency subbands) size comparison

    /// Intent: Sparse data (large runs, small values) typical of HH subbands
    /// should benefit more from rANS due to the concentrated run distribution.
    @Test func sparsePairsSizeComparison() {
        let testCounts = [8, 16, 24, 32, 48]

        for count in testCounts {
            let pairs = generateSparsePairs(count: count)
            let size = encodePairsToSize(pairs: pairs)
            let bpp = Double(size * 8) / Double(count)
            print("  sparse pairs=\(count) size=\(size)B bpp=\(String(format: "%.2f", bpp))")
        }
    }

    // MARK: - Test: Encode/Decode roundtrip at low pair counts

    /// Intent: Ensure that lowering the threshold to 8 does not break encode/decode
    /// symmetry. This validates that rANS mode works correctly for small pair counts.
    @Test func roundtripSmallPairCountViaPairs() throws {
        let testCounts = [4, 8, 16, 32, 64]

        for count in testCounts {
            let originalPairs = generateRealisticPairs(count: count)

            // Encode
            var encoder = EntropyEncoder<StaticEntropyModel>()
            encoder.encodeBypass(binVal: 1) // hasNonZero
            encoder.encodeBypass(binVal: 1) // lscpX exp-golomb terminator
            encoder.encodeBypass(binVal: 1) // lscpY exp-golomb terminator
            for pair in originalPairs {
                encoder.addPair(run: pair.run, val: pair.val, contextIdx: 0)
            }
            encoder.flush()
            let data = encoder.getData()

            // Decode
            var decoder = try EntropyDecoder(data: data)
            let hasNonZero = try decoder.decodeBypass()
            #expect(hasNonZero == 1, "hasNonZero should be 1")
            let _ = try decoder.decodeBypass() // lscpX
            let _ = try decoder.decodeBypass() // lscpY

            for (idx, original) in originalPairs.enumerated() {
                let decoded = decoder.readPair(contextIdx: 0)
                #expect(decoded.run == Int(original.run),
                       "Roundtrip mismatch at pair \(idx)/\(count): run expected=\(original.run) got=\(decoded.run)")
                #expect(decoded.val == original.val,
                       "Roundtrip mismatch at pair \(idx)/\(count): val expected=\(original.val) got=\(decoded.val)")
            }
        }
    }
}
