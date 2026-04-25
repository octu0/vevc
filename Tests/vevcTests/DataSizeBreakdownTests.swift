import Testing
@testable import vevc

// MARK: - Data Size Breakdown Tests
//
// Intent: Analyze the data size breakdown of the encoding pipeline to identify
// the true bottleneck. Previous tests showed that:
// - Improvement A (raw bypass threshold): REJECTED - rANS has higher bpp than raw bypass for small pair counts
// - Improvement C (zero block threshold): REJECTED - only 10KB reduction even at threshold=8
//
// This test suite measures:
// 1. The per-block overhead structure (hasNonZero + LSCP + run/val + rANS)
// 2. The ratio of bypass metadata vs rANS data in encoded output
// 3. Where the bytes actually go in a realistic encoding scenario

struct DataSizeBreakdownTests {

    // MARK: - Test: Per-block encoding overhead for 4x4 blocks

    /// Intent: Measure the fixed overhead per 4x4 block in the current encoding scheme.
    /// A 4x4 block has 16 coefficients. For each non-zero block, we pay:
    /// - 1 bit: hasNonZero flag
    /// - Exp-Golomb: lscpX coordinate (1+ bits)
    /// - Exp-Golomb: lscpY coordinate (1+ bits)
    /// - run/val pairs: variable
    /// This test measures the minimum overhead per block and the typical overhead.
    @Test func perBlockOverhead4x4() {
        // Scenario 1: Single non-zero coefficient at (0,0) = minimum data
        do {
            var encoder = EntropyEncoder<StaticEntropyModel>()
            encoder.encodeBypass(binVal: 1) // hasNonZero = true
            encoder.encodeBypass(binVal: 1) // lscpX = 0 (exp-golomb: 1-bit terminator)
            encoder.encodeBypass(binVal: 1) // lscpY = 0
            encoder.addPair(run: 0, val: 1, context: 0)
            encoder.addTrailingZeros(15)
            encoder.flush()
            let data = encoder.getData()
            print("  4x4 single-coeff-at-0,0: \(data.count)B")
        }

        // Scenario 2: Single non-zero coefficient at (3,3) = maximum LSCP cost
        do {
            var encoder = EntropyEncoder<StaticEntropyModel>()
            encoder.encodeBypass(binVal: 1) // hasNonZero
            // lscpX = 3: exp-golomb(3) = 0b0_1_00 = 4 bits (0, 1-bit, then 2 data bits)
            // But in our encoding, we use encodeExpGolomb differently
            // Let's just measure the full block
            encoder.addPair(run: 15, val: 1, context: 0) // run=15 to reach (3,3) in 4x4
            encoder.flush()
            let data = encoder.getData()
            print("  4x4 single-coeff-at-3,3: \(data.count)B")
        }

        // Scenario 3: All 16 coefficients non-zero (dense block)
        do {
            var encoder = EntropyEncoder<StaticEntropyModel>()
            encoder.encodeBypass(binVal: 1) // hasNonZero
            encoder.encodeBypass(binVal: 1) // lscpX = 3
            encoder.encodeBypass(binVal: 1) // lscpY = 3
            for _ in 0..<16 {
                encoder.addPair(run: 0, val: 1, context: 0)
            }
            encoder.flush()
            let data = encoder.getData()
            print("  4x4 all-nonzero (val=1): \(data.count)B")
        }

        // Scenario 4: All 16 coefficients non-zero with realistic values
        do {
            var encoder = EntropyEncoder<StaticEntropyModel>()
            encoder.encodeBypass(binVal: 1) // hasNonZero
            encoder.encodeBypass(binVal: 1) // lscpX
            encoder.encodeBypass(binVal: 1) // lscpY
            let vals: [Int16] = [8, 4, 2, 1, 3, 2, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0]
            var run: UInt32 = 0
            for v in vals {
                if v == 0 {
                    run += 1
                } else {
                    encoder.addPair(run: run, val: v, context: 0)
                    run = 0
                }
            }
            if 0 < run {
                encoder.addTrailingZeros(run)
            }
            encoder.flush()
            let data = encoder.getData()
            print("  4x4 realistic-dwt-coeffs: \(data.count)B (non-zero pairs: \(vals.filter { $0 != 0 }.count))")
        }

        // Scenario 5: Empty block (all zero)
        do {
            var encoder = EntropyEncoder<StaticEntropyModel>()
            encoder.encodeBypass(binVal: 0) // hasNonZero = false
            encoder.flush()
            let data = encoder.getData()
            print("  4x4 all-zero: \(data.count)B")
        }
    }

    // MARK: - Test: Encoding multiple blocks shows per-block overhead accumulation

    /// Intent: Measure how encoding N blocks (as would happen in a plane) affects
    /// the total size. This reveals the per-block entropy coding overhead.
    @Test func multipleBlocksOverhead() {
        let blockCounts = [10, 50, 100, 200, 500]

        for count in blockCounts {
            var encoder = EntropyEncoder<StaticEntropyModel>()
            // Simulate encoding N 4x4 blocks with typical DWT coefficients
            for _ in 0..<count {
                encoder.encodeBypass(binVal: 1) // hasNonZero
                encoder.encodeBypass(binVal: 1) // lscpX = 0
                encoder.encodeBypass(binVal: 1) // lscpY = 0
                // Typical: 3-5 non-zero coefficients per 4x4 block
                encoder.addPair(run: 0, val: 2, context: 0) // val=+2
                encoder.addPair(run: 1, val: 1, context: 0) // skip 1, val=+1
                encoder.addPair(run: 0, val: -1, context: 0) // val=-1
                encoder.addTrailingZeros(12)     // rest are zero
            }
            encoder.flush()
            let data = encoder.getData()
            let bytesPerBlock = Double(data.count) / Double(count)
            print("  blocks=\(count) total=\(data.count)B bytes/block=\(String(format: "%.1f", bytesPerBlock))")
        }
    }

    // MARK: - Test: Bypass metadata vs rANS data ratio

    /// Intent: Understand what fraction of the encoded data is bypass metadata
    /// (hasNonZero, LSCP, bypass bits) vs rANS-encoded run/val tokens.
    /// This tells us where optimization efforts should focus.
    @Test func bypassVsRansDataRatio() {
        // Encode 200 blocks with typical data
        var encoder = EntropyEncoder<StaticEntropyModel>()
        for _ in 0..<200 {
            encoder.encodeBypass(binVal: 1) // hasNonZero
            encoder.encodeBypass(binVal: 1) // lscpX
            encoder.encodeBypass(binVal: 1) // lscpY
            encoder.addPair(run: 0, val: 3, context: 0)
            encoder.addPair(run: 1, val: 1, context: 0)
            encoder.addPair(run: 0, val: -2, context: 0)
            encoder.addTrailingZeros(12)
        }
        encoder.flush()
        let data = encoder.getData()

        // Parse the structure to find the bypass/rANS split
        let bypassLen = Int(UInt32(data[0]) << 24 | UInt32(data[1]) << 16 | UInt32(data[2]) << 8 | UInt32(data[3]))

        let metaBypassSize = 4 + bypassLen // bypassLen field + bypass data
        let coeffCountSize = 4
        let ransDataSize = data.count - metaBypassSize - coeffCountSize

        print("  Total: \(data.count)B")
        print("  Bypass metadata (hasNonZero + LSCP): \(metaBypassSize)B (\(String(format: "%.1f", Double(metaBypassSize) / Double(data.count) * 100))%)")
        print("  CoeffCount header: \(coeffCountSize)B")
        print("  rANS data (run/val tokens + headers): \(ransDataSize)B (\(String(format: "%.1f", Double(ransDataSize) / Double(data.count) * 100))%)")
    }

    // MARK: - Test: Estimate actual frame encoding data distribution

    /// Intent: Using a realistic block layout (1080x450 @ 8x8 blocks = 135x56+1 blocks),
    /// estimate how much data each component consumes.
    @Test func estimateFrameDataDistribution() {
        // 1080x450 / 8 = 135x56 = 7,560 blocks for Y plane base layer
        // Each 8x8 block has LL(4x4) + HL(4x4) + LH(4x4) + HH(4x4)
        // For I-frames, LL uses DPCM, HL/LH/HH use standard encoding

        // Simulate: 50% of blocks are zero, 50% have data
        // Of non-zero blocks: average 4 non-zero coefficients per 4x4 subband

        let totalBlocks = 510 // Y blocks from debug log (270x113 / 8 rounded)
        let zeroRate = 0.10 // 10% zero for I-frame base (from debug log: 0%)
        let nonZeroBlocks = Int(Double(totalBlocks) * (1.0 - zeroRate))

        var encoder = EntropyEncoder<StaticDPCMEntropyModel>()

        // Encode LL subbands (DPCM)
        for _ in 0..<nonZeroBlocks {
            encoder.encodeBypass(binVal: 1) // hasNonZero
            encoder.encodeBypass(binVal: 1) // lscpX
            encoder.encodeBypass(binVal: 1) // lscpY
            // DPCM residuals are typically ±1
            for _ in 0..<16 {
                encoder.addPair(run: 0, val: 1, context: 0)
            }
        }
        encoder.flush()
        let llData = encoder.getData()

        var encoder2 = EntropyEncoder<StaticEntropyModel>()

        // Encode HL+LH+HH subbands (3 subbands × nonZeroBlocks blocks)
        for _ in 0..<(nonZeroBlocks * 3) {
            encoder2.encodeBypass(binVal: 1) // hasNonZero
            encoder2.encodeBypass(binVal: 1) // lscpX
            encoder2.encodeBypass(binVal: 1) // lscpY
            encoder2.addPair(run: 0, val: 2, context: 0)
            encoder2.addPair(run: 2, val: 1, context: 0)
            encoder2.addPair(run: 0, val: -1, context: 0)
            encoder2.addTrailingZeros(12)
        }
        encoder2.flush()
        let subbandData = encoder2.getData()

        print("  Estimated LL (DPCM): \(llData.count)B for \(nonZeroBlocks) blocks")
        print("  Estimated HL+LH+HH: \(subbandData.count)B for \(nonZeroBlocks * 3) blocks")
        print("  Total estimate: \(llData.count + subbandData.count)B")
        print("  Actual Layer0 Y from debug: ~18,762B")
    }
}
