import Foundation

// MARK: - Quantization

struct Quantizer: Sendable {
    let step: Int16
    let mul: Int32
    let bias: Int32
    let shift: Int16 = 16
    /// Whether AC dequantization applies the centroid offset. Disabled in the
    /// saturated regime (baseStep > 2048): at those step sizes the ±3Δ/16
    /// boost on every surviving coefficient reads as visible mottling on
    /// flat regions, outweighing the MSE gain.
    let centroidOffset: Bool

    /// - Parameters:
    ///   - step: Quantization step size.
    ///   - roundToNearest: If true, sets bias to 1<<15 for round-to-nearest behavior.
    ///   - deadZoneBias: Pre-computed dead zone bias in Q16 fixed-point.
    ///     Positive values narrow the dead zone (towards round-to-nearest).
    ///     Negative values widen the dead zone (more values become 0).
    ///     Common values: -6554 (-0.10), -3277 (-0.05), 0 (none).
    init(step: Int, roundToNearest: Bool = false, deadZoneBias: Int32 = 0, centroidOffset: Bool = true) {
        self.step = Int16(step)
        // reciprocal in Q16 fixed-point converts division to multiply+shift
        // Optimize by approximating division: val / step ≈ (val * mul) >> 16
        self.mul = Int32((1 << 20) / step)
        var b: Int32 = 0
        if roundToNearest {
            b = Int32(1 << 15)
        }
        if roundToNearest != true && deadZoneBias != 0 {
            b = deadZoneBias
        }
        self.bias = b
        self.centroidOffset = centroidOffset
    }
}

struct QuantizationTable: Sendable {
    let step: Int16
    let isChroma: Bool
    public let qLow: Quantizer
    public let qMid: Quantizer
    public let qHigh: Quantizer

    init(baseStep: Int, isChroma: Bool = false, layerIndex: Int = 0) {
        let s = max(16, min(baseStep, 4096))
        self.step = Int16(s)
        self.isChroma = isChroma

        // CSF-based perceptual quantization
        // DWT subbands map to spatial frequency bands:
        //   Layer 0 (Base8)  = lowest freq  → high CSF sensitivity → fine quantization
        //   Layer 1 (L16)    = mid freq      → moderate sensitivity
        //   Layer 2 (L32)    = highest freq  → low sensitivity → coarse quantization
        //   HH (diagonal) = √2× higher freq than HL/LH → even less perceptible
        //
        // Scale factors are expressed as integer ratios (numerator, denominator)
        // to avoid floating-point computation entirely.
        var qMidNum = 4  // HL/LH scale numerator   (4/4 = 1.0)
        var qMidDen = 4  // HL/LH scale denominator
        var qHighNum = 6  // HH scale numerator      (6/4 = 1.5)
        var qHighDen = 4  // HH scale denominator
        var qLowDivisor = 6

        switch layerIndex {
        case 2:
            qLowDivisor = 1
            qMidNum = 1
            qMidDen = 1  // 1.0
            qHighNum = 5
            qHighDen = 4  // 1.25
        case 1:
            qMidNum = 1
            qMidDen = 2  // 0.50
            qHighNum = 1
            qHighDen = 1  // 1.00
        default:  // layerIndex == 0
            qMidNum = 1
            qMidDen = 2  // 0.5
            qHighNum = 3
            qHighDen = 4  // 0.75
            qLowDivisor = 8
        }

        var dzMidY: Int32 = -4000
        var dzHighY: Int32 = -8000
        
        if layerIndex == 2 {
            dzMidY = 0
            dzHighY = -8000
        } else if layerIndex == 1 {
            dzMidY = 8192
            dzHighY = 0
        } else if layerIndex == 0 {
            dzMidY = 16384  // +0.25 positive bias to boost Luma SSIM
            dzHighY = 8192
        }
        
        let dzMidC: Int32
        let dzHighC: Int32
        
        if layerIndex == 0 {
            dzMidC = -8000
            dzHighC = -16000
        } else if layerIndex == 1 {
            dzMidC = -16000
            dzHighC = -32000
        } else {
            dzMidC = -32000
            dzHighC = -64000
        }


        // Saturation-gated extended range: below the knee (baseStep 2048) the
        // caps match the original tuning exactly, so normal-rate quality is
        // untouched. When the rate controller pushes baseStep past the knee
        // (i.e. the original caps have long been saturated and rate can no
        // longer drop), the effective cap ramps linearly up to 2x at
        // baseStep 4096, lowering the achievable rate floor instead of the
        // step silently clamping. Derived deterministically from the signaled
        // baseStep, so the decoder reproduces it without extra signaling.
        let ext = max(0, Int(s) - 2048)

        let offsetOn = Int(s) <= 2048

        if isChroma {
            // qLow is the DC component: NEVER scale it to avoid destroying base color/brightness!
            let cLow = min(qLowCapQ4, max(16, baseStep / 8))
            let cMid = min(384, max(16, (baseStep * qMidNum) / qMidDen))
            // No saturation extension for chroma: at extended steps (real 96)
            // surviving chroma HH coefficients of motion residuals appear as
            // red/green block splats on fast pans (measured on the Cr plane).
            let cHigh = min(768, max(16, (baseStep * qHighNum) / qHighDen))

            self.qLow = Quantizer(step: Int(cLow), roundToNearest: true)
            self.qMid = Quantizer(step: Int(cMid), roundToNearest: false, deadZoneBias: dzMidC, centroidOffset: offsetOn)
            self.qHigh = Quantizer(step: Int(cHigh), roundToNearest: false, deadZoneBias: dzHighC, centroidOffset: offsetOn)
        } else {
            // qLow is the DC component: NEVER scale it!
            let lLow = min(qLowCapQ4, max(16, baseStep / qLowDivisor))
            self.qLow = Quantizer(step: Int(lLow), roundToNearest: true)

            // Luma stepMult is 1: Never scale Luma steps because they ruin SSIM.
            let lMid = min(768, max(16, (baseStep * qMidNum) / qMidDen))
            self.qMid = Quantizer(step: Int(lMid), roundToNearest: false, deadZoneBias: dzMidY, centroidOffset: offsetOn)

            let lHigh = min(1024, max(16, (baseStep * qHighNum) / qHighDen)) + (ext * 1024) / 2048
            self.qHigh = Quantizer(step: Int(lHigh), roundToNearest: false, deadZoneBias: dzHighY, centroidOffset: offsetOn)
        }
    }
}

// MARK: - Adaptive Quantization Table

/// Block-level adaptive quantization.
/// Pre-generates a discrete set of QuantizationTables with scaled qMid/qHigh
/// to redistribute bits from flat blocks to edge/texture blocks.
///
/// Selection logic:
///   - Measure each block's AC energy (sum of |HL| + |LH| + |HH| coefficients)
///   - Compare against the frame-wide average AC energy
///   - High-energy blocks (edges/textures) → higher qStep (coarser quantization, noise is masked)
///   - Low-energy blocks (flat regions, faces) → lower qStep (finer quantization, prevents visible blockiness)
///
/// qLow is NOT scaled — base frequency quality must remain constant.

// MARK: - Quantization SIMD

/// Cap for the DC (qLow) quantizer step, in Q4 units (192 = real step 12).
/// Was 256 (real step 16): at deeply saturated rates that step turns slow
/// global luminance ramps (fades) into visible 32×32px DC plateaus at full
/// resolution — the enhancement layers cannot repair them because the
/// correction falls inside the qMid dead zones. Step 12 is below the banding
/// visibility threshold on measured content and, with rate-control feedback,
/// is net-free (miko1 500k: size −0.2%, worst-frame SSIM +0.010). Inactive
/// at normal rates (qLow = baseStep/8 stays below the cap).
let qLowCapQ4: Int = 192

@inline(__always)
internal func quantizeDPCM(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<4 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            ptr[i] = Int16(clamping: res)
        }
    }
}

@inline(__always)
internal func quantize4(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<4 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<4 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            let v = Int16(clamping: res)
            ptr[i] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
        }
    }
}

@inline(__always)
internal func quantize8(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<8 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<8 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            let v = Int16(clamping: res)
            ptr[i] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
        }
    }
}

@inline(__always)
internal func quantize16(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<16 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<16 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            let v = Int16(clamping: res)
            ptr[i] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
        }
    }
}

@inline(__always)
internal func quantize32(_ block: BlockView, q: Quantizer) {
    let mul = q.mul
    let shift = Int32(q.shift)
    let bias = q.bias
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        for i in 0..<32 {
            let val = Int32(ptr[i])
            let signMask = val &>> 31
            let absVal = (val ^ signMask) &- signMask
            let qVal = max(0, (((absVal &* mul) &+ bias) &>> shift))
            let res = (qVal ^ signMask) &- signMask
            let v = Int16(clamping: res)
            ptr[i] = Int16(bitPattern: UInt16(bitPattern: ((v &<< 1) ^ (v &>> 15))))
        }
    }
}

// MARK: - Dequantization SIMD

@inline(__always)
internal func dequantizeDPCM(ptr: UnsafeMutablePointer<Int16>, stride: Int, q: Quantizer) {
    let step = Int32(q.step)
    var rowPtr = ptr
    for _ in 0..<4 {
        for i in 0..<4 {
            rowPtr[i] = Int16(clamping: (Int32(rowPtr[i]) &* step &+ 8) >> 4)
        }
        rowPtr = rowPtr.advanced(by: stride)
    }
}

/// Dequantization centroid offset for AC coefficients, in 1/16-step units,
/// applied away from zero. The nominal k·Δ reconstruction sits at the bin's
/// lower edge (dead-zone bins), while the Laplacian bin centroid lies
/// ~0.3–0.9 steps above it (fit per subband by the offline `vevc-training
/// offsets` tool, removed 2026-08-17 — in git history). 3/16 measured best end-to-end on live-stream
/// content: miko1 +0.20〜0.40 dB PSNR at equal-or-smaller size across
/// 800k–4000k with worst-frame SSIM intact; film content (ToS) is metric-
/// neutral (−0.02 dB avg) and visually indistinguishable. Shared by the
/// encoder reconstruction loop and the decoder, so both stay in sync.
let dequantOffsetNumQ4: Int32 = 3

@inline(__always)
internal func dequantize4(ptr: UnsafeMutablePointer<Int16>, stride: Int, q: Quantizer) {
    let step = Int32(q.step)
    let offQ = q.centroidOffset ? (dequantOffsetNumQ4 &* step) >> 4 : 0
    var rowPtr = ptr
    for _ in 0..<4 {
        let v = UnsafeRawPointer(rowPtr).loadUnaligned(as: SIMD4<UInt16>.self)
        let decodedUInt = ((v &>> 1) ^ (.zero &- (v & 1)))
        let v16 = SIMD4<Int16>(truncatingIfNeeded: decodedUInt)
        @inline(__always) func deq(_ s: Int16) -> Int16 {
            let adj: Int32 = s == 0 ? 0 : (0 < s ? offQ : -offQ)
            return Int16(clamping: (Int32(s) &* step &+ adj &+ 8) >> 4)
        }
        let res16 = SIMD4<Int16>(deq(v16[0]), deq(v16[1]), deq(v16[2]), deq(v16[3]))
        UnsafeMutableRawPointer(rowPtr).storeBytes(of: res16, as: SIMD4<Int16>.self)
        rowPtr = rowPtr.advanced(by: stride)
    }
}

@inline(__always)
internal func dequantize8(ptr: UnsafeMutablePointer<Int16>, stride: Int, q: Quantizer) {
    let step = Int32(q.step)
    let offQ = q.centroidOffset ? (dequantOffsetNumQ4 &* step) >> 4 : 0
    var rowPtr = ptr
    for _ in 0..<8 {
        let v = UnsafeRawPointer(rowPtr).loadUnaligned(as: SIMD8<UInt16>.self)
        let decodedUInt = ((v &>> 1) ^ (.zero &- (v & 1)))
        let v16 = SIMD8<Int16>(truncatingIfNeeded: decodedUInt)
        @inline(__always) func deq(_ s: Int16) -> Int16 {
            let adj: Int32 = s == 0 ? 0 : (0 < s ? offQ : -offQ)
            return Int16(clamping: (Int32(s) &* step &+ adj &+ 8) >> 4)
        }
        let res16 = SIMD8<Int16>(deq(v16[0]), deq(v16[1]), deq(v16[2]), deq(v16[3]), deq(v16[4]), deq(v16[5]), deq(v16[6]), deq(v16[7]))
        UnsafeMutableRawPointer(rowPtr).storeBytes(of: res16, as: SIMD8<Int16>.self)
        rowPtr = rowPtr.advanced(by: stride)
    }
}

@inline(__always)
internal func dequantize16(ptr: UnsafeMutablePointer<Int16>, stride: Int, q: Quantizer) {
    let step = Int32(q.step)
    let vStep = SIMD8<Int32>(repeating: step)
    let v8 = SIMD8<Int32>(repeating: 8)
    let offQ = q.centroidOffset ? (dequantOffsetNumQ4 &* step) >> 4 : 0
    let vOffPos = SIMD8<Int32>(repeating: offQ)
    let vOffNeg = SIMD8<Int32>(repeating: 0 &- offQ)
    let vZero32 = SIMD8<Int32>(repeating: 0)
    var rowPtr = ptr
    for _ in 0..<16 {
        let v = UnsafeRawPointer(rowPtr).loadUnaligned(as: SIMD16<UInt16>.self)
        let decodedUInt = ((v &>> 1) ^ (.zero &- (v & 1)))
        let v16 = SIMD16<Int16>(truncatingIfNeeded: decodedUInt)

        let low8 = v16.lowHalf
        let high8 = v16.highHalf

        let l32 = SIMD8<Int32>(truncatingIfNeeded: low8)
        let h32 = SIMD8<Int32>(truncatingIfNeeded: high8)

        var offL = vZero32
        offL.replace(with: vOffPos, where: vZero32 .< l32)
        offL.replace(with: vOffNeg, where: l32 .< vZero32)
        var offH = vZero32
        offH.replace(with: vOffPos, where: vZero32 .< h32)
        offH.replace(with: vOffNeg, where: h32 .< vZero32)

        let resLow8 = (l32 &* vStep &+ offL &+ v8) &>> 4
        let resHigh8 = (h32 &* vStep &+ offH &+ v8) &>> 4
        
        let cLow8 = SIMD8<Int16>(truncatingIfNeeded: resLow8.clamped(lowerBound: SIMD8(repeating: -32768), upperBound: SIMD8(repeating: 32767)))
        let cHigh8 = SIMD8<Int16>(truncatingIfNeeded: resHigh8.clamped(lowerBound: SIMD8(repeating: -32768), upperBound: SIMD8(repeating: 32767)))
        
        let res16 = SIMD16<Int16>(lowHalf: cLow8, highHalf: cHigh8)
        UnsafeMutableRawPointer(rowPtr).storeBytes(of: res16, as: SIMD16<Int16>.self)
        rowPtr = rowPtr.advanced(by: stride)
    }
}

@inline(__always)
internal func dequantize32(_ block: BlockView, q: Quantizer) {
    let step = Int32(q.step)
    let vStep = SIMD8<Int32>(repeating: step)
    let v8 = SIMD8<Int32>(repeating: 8)
    for y in 0..<32 {
        let ptr = block.rowPointer(y: y)
        let ptrRaw0 = UnsafeRawPointer(ptr)
        let ptrRaw1 = UnsafeRawPointer(ptr.advanced(by: 16))
        let v0 = ptrRaw0.loadUnaligned(as: SIMD16<UInt16>.self)
        let v1 = ptrRaw1.loadUnaligned(as: SIMD16<UInt16>.self)
        let decodedUInt0 = ((v0 &>> 1) ^ (.zero &- (v0 & 1)))
        let decodedUInt1 = ((v1 &>> 1) ^ (.zero &- (v1 & 1)))
        let v16_0 = SIMD16<Int16>(truncatingIfNeeded: decodedUInt0)
        let v16_1 = SIMD16<Int16>(truncatingIfNeeded: decodedUInt1)
        
        let low8_0 = v16_0.lowHalf
        let high8_0 = v16_0.highHalf
        
        let low8_1 = v16_1.lowHalf
        let high8_1 = v16_1.highHalf
        
        let l32_0 = SIMD8<Int32>(truncatingIfNeeded: low8_0)
        let h32_0 = SIMD8<Int32>(truncatingIfNeeded: high8_0)
        let l32_1 = SIMD8<Int32>(truncatingIfNeeded: low8_1)
        let h32_1 = SIMD8<Int32>(truncatingIfNeeded: high8_1)
        
        let resLow8_0 = (l32_0 &* vStep &+ v8) &>> 4
        let resHigh8_0 = (h32_0 &* vStep &+ v8) &>> 4
        let resLow8_1 = (l32_1 &* vStep &+ v8) &>> 4
        let resHigh8_1 = (h32_1 &* vStep &+ v8) &>> 4
        
        let limitMin = SIMD8<Int32>(repeating: -32768)
        let limitMax = SIMD8<Int32>(repeating: 32767)
        
        let cLow8_0 = SIMD8<Int16>(truncatingIfNeeded: resLow8_0.clamped(lowerBound: limitMin, upperBound: limitMax))
        let cHigh8_0 = SIMD8<Int16>(truncatingIfNeeded: resHigh8_0.clamped(lowerBound: limitMin, upperBound: limitMax))
        let cLow8_1 = SIMD8<Int16>(truncatingIfNeeded: resLow8_1.clamped(lowerBound: limitMin, upperBound: limitMax))
        let cHigh8_1 = SIMD8<Int16>(truncatingIfNeeded: resHigh8_1.clamped(lowerBound: limitMin, upperBound: limitMax))
        
        let res16_0 = SIMD16<Int16>(lowHalf: cLow8_0, highHalf: cHigh8_0)
        let res16_1 = SIMD16<Int16>(lowHalf: cLow8_1, highHalf: cHigh8_1)
        
        UnsafeMutableRawPointer(ptr).storeBytes(of: res16_0, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(ptr.advanced(by: 16)).storeBytes(of: res16_1, as: SIMD16<Int16>.self)
    }
}


