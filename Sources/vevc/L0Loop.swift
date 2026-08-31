// MARK: - Layer0 closed loop (Profile 0x02, One-Pyramid design §4)
//
// Base8 codes r0 = LL2(source) − MC_L0(L0_ref) instead of LL2(residual).
// The L0-only decoder is unchanged (deq(r0) + MC_L0 is exactly its existing
// pipeline), which makes the quarter-resolution reconstruction bit-exact
// between encoder, full decoder, and L0-only decoder. The full decoder
// recovers the LL2 coefficient slot as L0_recon − LL2(P) where P is the
// full-resolution prediction, analyzed with the exact encoder DWT chain.
//
// Invariants (docs/profile2-update-design.md, memory: profile2-update-design):
// - The L0 chain closes over the bitstream only: references are dequantized
//   Base8 output, never a resampled full-resolution reconstruction.
// - Every operator (LL analysis, MC, clamp, deblock, skip copy) is shared
//   code invoked with identical parameters on both sides.
import Foundation

/// L0 reference planes (quarter resolution, sub1 domain) threaded through the
/// encode/decode chain. `prev` is the previous frame's L0 reconstruction,
/// `ltr` the GOP-first (I-frame) one. Owned by one actor; frames within a GOP
/// are processed sequentially, so unsynchronized access is safe.
final class L0RefState: @unchecked Sendable {
    var prev: PlaneData420?
    var ltr: PlaneData420?
    init() {}
}

/// One full-block LeGall 5/3 analysis level, returning only the gathered LL
/// plane. Identical code path to the encoder's extractSingleTransformBlocks*
/// LL gather (same edge padding via Int16Reader, same dwt2DBlock kernels).
@inline(__always)
func llAnalyzeLevel(_ plane: [Int16], w: Int, h: Int, blockSize: Int) -> [Int16] {
    let q = blockSize / 2
    let colCount = (w + blockSize - 1) / blockSize
    let rowCount = (h + blockSize - 1) / blockSize
    let llW = (w + 1) / 2
    let llH = (h + 1) / 2
    var ll = [Int16](repeating: 0, count: llW * llH)
    var scratch = [Int16](repeating: 0, count: blockSize * blockSize)
    let reader = Int16Reader(data: plane, width: w, height: h)
    withUnsafePointers(mut: &scratch, mut: &ll) { sBase, dBase in
        let view = BlockView(base: sBase, width: blockSize, height: blockSize, stride: blockSize)
        for r in 0..<rowCount {
            for c in 0..<colCount {
                reader.readBlock(x: c * blockSize, y: r * blockSize, width: blockSize, height: blockSize, into: view)
                // LL-only forward lifting: bit-identical LL bytes, the other
                // quadrants (never gathered here) stay unspecified.
                switch blockSize {
                case 32: dwt2DBlock32LL(view)
                case 16: dwt2DBlock16LL(view)
                default: dwt2DBlock8(view)
                }
                let dx0 = c * q
                let dy0 = r * q
                let copyW = min(q, llW - dx0)
                if copyW <= 0 { continue }
                for y in 0..<q {
                    let dy = dy0 + y
                    if dy < llH {
                        let src = sBase.advanced(by: y * blockSize)
                        dBase.advanced(by: dy * llW + dx0).update(from: src, count: copyW)
                    }
                }
            }
        }
    }
    return ll
}

/// Two-level LL analysis (32-block then 16-block) of a full-resolution
/// PlaneData420, producing the quarter-resolution sub1-domain planes — the
/// exact transform T whose output domain Base8 codes.
@inline(__always)
func analyzeLL2(pd: PlaneData420) -> PlaneData420 {
    let dx = pd.width
    let dy = pd.height
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let y1 = llAnalyzeLevel(pd.y, w: dx, h: dy, blockSize: 32)
    let y0 = llAnalyzeLevel(y1, w: l1dx, h: l1dy, blockSize: 16)
    let cb1 = llAnalyzeLevel(pd.cb, w: cbDx, h: cbDy, blockSize: 32)
    let cb0 = llAnalyzeLevel(cb1, w: (cbDx + 1) / 2, h: (cbDy + 1) / 2, blockSize: 16)
    let cr1 = llAnalyzeLevel(pd.cr, w: cbDx, h: cbDy, blockSize: 32)
    let cr0 = llAnalyzeLevel(cr1, w: (cbDx + 1) / 2, h: (cbDy + 1) / 2, blockSize: 16)
    return PlaneData420(width: (l1dx + 1) / 2, height: (l1dy + 1) / 2, y: y0, cb: cb0, cr: cr0)
}

/// Adds the L0-resolution motion-compensated prediction into `img` — the
/// exact call sequence of the decoder's layer0 MC path (8x8 luma blocks
/// mvShift 2, 4x4 chroma blocks mvShift 1).
@inline(__always)
func applyL0MotionCompensation(img: inout Image16, prevPd: PlaneData420, ltrPd: PlaneData420?, mvs: MotionVectors, refDirs: [Bool]?, skipMap: [BlockMode]?, roundOffset: Int) async {
    let l0dx = img.width
    let l0dy = img.height
    let cbDx0 = (l0dx + 1) / 2
    let cbDy0 = (l0dy + 1) / 2
    if let tNext = ltrPd, let dirs = refDirs {
        await applyScaledBidirectionalMotionCompensationLuma(plane: &img.y, prevPlane: prevPd.y, nextPlane: tNext.y, mvs: mvs, refDirs: dirs, skipMap: skipMap, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
        await applyScaledBidirectionalMotionCompensationChroma(plane: &img.cb, prevPlane: prevPd.cb, nextPlane: tNext.cb, mvs: mvs, refDirs: dirs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
        await applyScaledBidirectionalMotionCompensationChroma(plane: &img.cr, prevPlane: prevPd.cr, nextPlane: tNext.cr, mvs: mvs, refDirs: dirs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
    } else {
        await applyScaledMotionCompensationLuma(plane: &img.y, prevPlane: prevPd.y, mvs: mvs, skipMap: skipMap, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
        await applyScaledMotionCompensationChroma(plane: &img.cb, prevPlane: prevPd.cb, mvs: mvs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
        await applyScaledMotionCompensationChroma(plane: &img.cr, prevPlane: prevPd.cr, mvs: mvs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
    }
}

/// Clamp + deblock of the L0 reconstruction — the exact post-MC sequence of
/// the decoder's layer0 path. qtYStepQ4/qtCStepQ4 are the raw Q4 steps as
/// returned by decodeBase8 (Int(qt.step)).
@inline(__always)
func finishL0Reconstruction(img: inout Image16, qtYStepQ4: Int, qtCStepQ4: Int) {
    clampPlaneToPixelRange(plane: &img.y)
    clampPlaneToPixelRange(plane: &img.cb)
    clampPlaneToPixelRange(plane: &img.cr)
    let l0dx = img.width
    let l0dy = img.height
    let cbDx0 = (l0dx + 1) / 2
    let cbDy0 = (l0dy + 1) / 2
    applyDeblockingFilterN(plane: &img.y, width: l0dx, height: l0dy, qStep: qtYStepQ4, blockSize: 8)
    let cStep = min(qtCStepQ4 * 2, 255)
    applyDeblockingFilterN(plane: &img.cb, width: cbDx0, height: cbDy0, qStep: cStep, blockSize: 4)
    applyDeblockingFilterN(plane: &img.cr, width: cbDx0, height: cbDy0, qStep: cStep, blockSize: 4)
}

/// Skip-block copy into the L0 reconstruction from the layer-matched
/// reference planes — the exact geometry of the decoder's profile-0x02 skip
/// copy at layer0 (scale 4, 8x8 luma / 4x4 chroma blocks).
@inline(__always)
func applyL0SkipCopy(img: inout Image16, prevPd: PlaneData420, ltrPd: PlaneData420?, skipMap: [BlockMode], fullDx: Int) {
    let bw = (fullDx + 31) / 32
    let targetDx = img.width
    let targetDy = img.height
    let targetCbDx = (targetDx + 1) / 2
    let targetCbDy = (targetDy + 1) / 2
    let lPd = ltrPd ?? prevPd
    withUnsafePointers(
        lPd.y, lPd.cb, lPd.cr,
        prevPd.y, prevPd.cb, prevPd.cr,
        mut: &img.y, mut: &img.cb, mut: &img.cr
    ) { ltrYPtr, ltrCbPtr, ltrCrPtr, prevYPtr, prevCbPtr, prevCrPtr, currYPtr, currCbPtr, currCrPtr in
        for i in 0..<skipMap.count {
            let mode = skipMap[i]
            if mode != .inter {
                let bx = ((i % bw) * 32) / 4
                let by = ((i / bw) * 32) / 4
                if bx + 8 <= targetDx && by + 8 <= targetDy {
                    switch mode {
                    case .skip_ltr where ltrPd != nil:
                        copyBlock8Pointer(from: ltrYPtr, to: currYPtr, bx: bx, by: by, stride: targetDx)
                        copyBlock4Pointer(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                        copyBlock4Pointer(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                    case .skip_prev:
                        copyBlock8Pointer(from: prevYPtr, to: currYPtr, bx: bx, by: by, stride: targetDx)
                        copyBlock4Pointer(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                        copyBlock4Pointer(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                    default: break
                    }
                } else {
                    switch mode {
                    case .skip_ltr where ltrPd != nil:
                        copyBlockSafe(from: ltrYPtr, to: currYPtr, bx: bx, by: by, width: targetDx, height: targetDy, blockSize: 8)
                        copyBlockSafe(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 4)
                        copyBlockSafe(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 4)
                    case .skip_prev:
                        copyBlockSafe(from: prevYPtr, to: currYPtr, bx: bx, by: by, width: targetDx, height: targetDy, blockSize: 8)
                        copyBlockSafe(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 4)
                        copyBlockSafe(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 4)
                    default: break
                    }
                }
            }
        }
    }
}

/// Pixel-range clamp shared by encoder and decoder L0 loops (identical to the
/// decoder's internal plane clamp).
@inline(__always)
func clampPlaneToPixelRange(plane: inout [Int16]) {
    plane.withUnsafeMutableBufferPointer { ptr in
        let base = ptr.baseAddress!
        var x = 0
        let c = ptr.count
        let vMin = SIMD16<Int16>(repeating: -128)
        let vMax = SIMD16<Int16>(repeating: 127)
        while x < c - 15 {
            let p = base.advanced(by: x)
            let v = UnsafeRawPointer(p).loadUnaligned(as: SIMD16<Int16>.self)
            let clampedMin = v.replacing(with: vMin, where: v .< vMin)
            let clamped = clampedMin.replacing(with: vMax, where: clampedMin .> vMax)
            UnsafeMutableRawPointer(p).storeBytes(of: clamped, as: SIMD16<Int16>.self)
            x &+= 16
        }
        while x < c {
            let v = ptr[x]
            switch true {
            case v < -128: ptr[x] = -128
            case 127 < v: ptr[x] = 127
            default: break
            }
            x &+= 1
        }
    }
}

/// Builds the full-resolution prediction plane P = MC_full(refs) by applying
/// the layer2 MC (32-block, mvShift 0) into zeroed planes. Both sides derive
/// LL2(P) from this, so it must be produced by the identical call sequence
/// the decoder uses when adding prediction at layer2.
@inline(__always)
func buildFullResolutionPrediction(dx: Int, dy: Int, prevPd: PlaneData420, ltrPd: PlaneData420?, mvs: MotionVectors, refDirs: [Bool]?, skipMap: [BlockMode]?, roundOffset: Int) async -> PlaneData420 {
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    var y = [Int16](repeating: 0, count: dx * dy)
    var cb = [Int16](repeating: 0, count: cbDx * cbDy)
    var cr = [Int16](repeating: 0, count: cbDx * cbDy)
    if let tNext = ltrPd, let dirs = refDirs {
        await applyScaledBidirectionalMotionCompensationLuma(plane: &y, prevPlane: prevPd.y, nextPlane: tNext.y, mvs: mvs, refDirs: dirs, skipMap: skipMap, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
        await applyScaledBidirectionalMotionCompensationChroma(plane: &cb, prevPlane: prevPd.cb, nextPlane: tNext.cb, mvs: mvs, refDirs: dirs, skipMap: skipMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        await applyScaledBidirectionalMotionCompensationChroma(plane: &cr, prevPlane: prevPd.cr, nextPlane: tNext.cr, mvs: mvs, refDirs: dirs, skipMap: skipMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    } else {
        await applyScaledMotionCompensationLuma(plane: &y, prevPlane: prevPd.y, mvs: mvs, skipMap: skipMap, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
        await applyScaledMotionCompensationChroma(plane: &cb, prevPlane: prevPd.cb, mvs: mvs, skipMap: skipMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        await applyScaledMotionCompensationChroma(plane: &cr, prevPlane: prevPd.cr, mvs: mvs, skipMap: skipMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    }
    return PlaneData420(width: dx, height: dy, y: y, cb: cb, cr: cr)
}

/// Builds the layer1-resolution prediction by applying the decoder's layer1
/// MC (16-block luma / 8-block chroma, mvShift 1) into zeroed planes. Used
/// when layer2 is absent (splitter-truncated stream decoded at layer1).
@inline(__always)
func buildL1Prediction(l1dx: Int, l1dy: Int, prevPd: PlaneData420, ltrPd: PlaneData420?, mvs: MotionVectors, refDirs: [Bool]?, skipMap: [BlockMode]?, roundOffset: Int) async -> PlaneData420 {
    let cbDx1 = (l1dx + 1) / 2
    let cbDy1 = (l1dy + 1) / 2
    var y = [Int16](repeating: 0, count: l1dx * l1dy)
    var cb = [Int16](repeating: 0, count: cbDx1 * cbDy1)
    var cr = [Int16](repeating: 0, count: cbDx1 * cbDy1)
    if let tNext = ltrPd, let dirs = refDirs {
        await applyScaledBidirectionalMotionCompensationLuma(plane: &y, prevPlane: prevPd.y, nextPlane: tNext.y, mvs: mvs, refDirs: dirs, skipMap: skipMap, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
        await applyScaledBidirectionalMotionCompensationChroma(plane: &cb, prevPlane: prevPd.cb, nextPlane: tNext.cb, mvs: mvs, refDirs: dirs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
        await applyScaledBidirectionalMotionCompensationChroma(plane: &cr, prevPlane: prevPd.cr, nextPlane: tNext.cr, mvs: mvs, refDirs: dirs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
    } else {
        await applyScaledMotionCompensationLuma(plane: &y, prevPlane: prevPd.y, mvs: mvs, skipMap: skipMap, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
        await applyScaledMotionCompensationChroma(plane: &cb, prevPlane: prevPd.cb, mvs: mvs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
        await applyScaledMotionCompensationChroma(plane: &cr, prevPlane: prevPd.cr, mvs: mvs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
    }
    return PlaneData420(width: l1dx, height: l1dy, y: y, cb: cb, cr: cr)
}

/// One-level LL analysis (16-block) of a layer1-resolution PlaneData420 —
/// the encoder chain's half-res → quarter-res step.
@inline(__always)
func analyzeLL1(pd: PlaneData420) -> PlaneData420 {
    let dx = pd.width
    let dy = pd.height
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let y0 = llAnalyzeLevel(pd.y, w: dx, h: dy, blockSize: 16)
    let cb0 = llAnalyzeLevel(pd.cb, w: cbDx, h: cbDy, blockSize: 16)
    let cr0 = llAnalyzeLevel(pd.cr, w: cbDx, h: cbDy, blockSize: 16)
    return PlaneData420(width: (dx + 1) / 2, height: (dy + 1) / 2, y: y0, cb: cb0, cr: cr0)
}

// MARK: - Weighted prediction (per-plane offsets, #21)
//
// P′ = P + offset on INTER blocks only: intra blocks carry no prediction and
// skip blocks are raw reference copies (spec-pinned verbatim). The same call
// runs on both sides at every point a prediction plane is formed — full
// resolution, the L1 prediction, and the quarter-resolution L0 chain (the
// LeGall lowpass has DC gain 1, so the full-resolution offset is also the
// correct LL-domain offset). Every plane's block grid at every layer is 1:1
// with the skip map (luma 32/16/8, chroma 16/8/4); intra blocks are marked
// by the MV sentinel 32767.

@inline(__always)
func applyPredictionOffset32(plane: inout [Int16], offset: Int, mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode], width: Int, height: Int) {
    applyPredictionOffset(plane: &plane, offset: offset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: width, height: height, blockPx: 32)
}

@inline(__always)
func applyPredictionOffset16(plane: inout [Int16], offset: Int, mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode], width: Int, height: Int) {
    applyPredictionOffset(plane: &plane, offset: offset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: width, height: height, blockPx: 16)
}

@inline(__always)
func applyPredictionOffset8(plane: inout [Int16], offset: Int, mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode], width: Int, height: Int) {
    applyPredictionOffset(plane: &plane, offset: offset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: width, height: height, blockPx: 8)
}

@inline(__always)
func applyPredictionOffset4(plane: inout [Int16], offset: Int, mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode], width: Int, height: Int) {
    applyPredictionOffset(plane: &plane, offset: offset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: width, height: height, blockPx: 4)
}

/// Frame-level offset application for a full-resolution prediction plane
/// set (32px luma / 16px chroma blocks).
@inline(__always)
func applyPredictionOffsetsL2(pd: inout PlaneData420, lumaOffset: Int, chromaOffset: Int, mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode]) {
    if lumaOffset != 0 {
        applyPredictionOffset32(plane: &pd.y, offset: lumaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: pd.width, height: pd.height)
    }
    if chromaOffset != 0 {
        let cw = (pd.width + 1) / 2
        let ch = (pd.height + 1) / 2
        applyPredictionOffset16(plane: &pd.cb, offset: chromaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: cw, height: ch)
        applyPredictionOffset16(plane: &pd.cr, offset: chromaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: cw, height: ch)
    }
}

/// Frame-level offset application for a full-resolution reconstruction image
/// (32px luma / 16px chroma blocks).
@inline(__always)
func applyPredictionOffsetsL2(img: inout Image16, lumaOffset: Int, chromaOffset: Int, mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode]) {
    if lumaOffset != 0 {
        applyPredictionOffset32(plane: &img.y, offset: lumaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: img.width, height: img.height)
    }
    if chromaOffset != 0 {
        let cw = (img.width + 1) / 2
        let ch = (img.height + 1) / 2
        applyPredictionOffset16(plane: &img.cb, offset: chromaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: cw, height: ch)
        applyPredictionOffset16(plane: &img.cr, offset: chromaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: cw, height: ch)
    }
}

/// Half-resolution variant (16px luma / 8px chroma blocks) for the L1
/// prediction and the layer1 display path.
@inline(__always)
func applyPredictionOffsetsL1(pd: inout PlaneData420, lumaOffset: Int, chromaOffset: Int, mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode]) {
    if lumaOffset != 0 {
        applyPredictionOffset16(plane: &pd.y, offset: lumaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: pd.width, height: pd.height)
    }
    if chromaOffset != 0 {
        let cw = (pd.width + 1) / 2
        let ch = (pd.height + 1) / 2
        applyPredictionOffset8(plane: &pd.cb, offset: chromaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: cw, height: ch)
        applyPredictionOffset8(plane: &pd.cr, offset: chromaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: cw, height: ch)
    }
}

@inline(__always)
func applyPredictionOffsetsL1(img: inout Image16, lumaOffset: Int, chromaOffset: Int, mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode]) {
    if lumaOffset != 0 {
        applyPredictionOffset16(plane: &img.y, offset: lumaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: img.width, height: img.height)
    }
    if chromaOffset != 0 {
        let cw = (img.width + 1) / 2
        let ch = (img.height + 1) / 2
        applyPredictionOffset8(plane: &img.cb, offset: chromaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: cw, height: ch)
        applyPredictionOffset8(plane: &img.cr, offset: chromaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: cw, height: ch)
    }
}

/// Quarter-resolution variant (8px luma / 4px chroma blocks) for the L0
/// chain predictions and the layer0 display path.
@inline(__always)
func applyPredictionOffsetsL0(img: inout Image16, lumaOffset: Int, chromaOffset: Int, mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode]) {
    if lumaOffset != 0 {
        applyPredictionOffset8(plane: &img.y, offset: lumaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: img.width, height: img.height)
    }
    if chromaOffset != 0 {
        let cw = (img.width + 1) / 2
        let ch = (img.height + 1) / 2
        applyPredictionOffset4(plane: &img.cb, offset: chromaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: cw, height: ch)
        applyPredictionOffset4(plane: &img.cr, offset: chromaOffset, mvs: mvs, refDirs: refDirs, skipMap: skipMap, width: cw, height: ch)
    }
}

@inline(__always)
private func applyPredictionOffset(plane: inout [Int16], offset: Int, mvs: MotionVectors, refDirs: [Bool], skipMap: [BlockMode], width: Int, height: Int, blockPx: Int) {
    let bw = (width + blockPx - 1) / blockPx
    let bh = (height + blockPx - 1) / blockPx
    let o = Int16(clamping: offset)
    let mvCount = mvs.dx.count
    let dirCount = refDirs.count
    withUnsafePointers(mut: &plane) { base in
        for by in 0..<bh {
            let y0 = by * blockPx
            let h = min(blockPx, height - y0)
            for bx in 0..<bw {
                let idx = by * bw + bx
                if skipMap[idx] != .inter {
                    continue
                }
                if idx < mvCount && mvs.dx[idx] == 32767 {
                    continue
                }
                // The offset is estimated against the prev reference; blocks
                // referencing the LTR (refDir true) keep P unchanged.
                if idx < dirCount && refDirs[idx] {
                    continue
                }
                let x0 = bx * blockPx
                let w = min(blockPx, width - x0)
                for y in 0..<h {
                    let row = base.advanced(by: (y0 + y) * width + x0)
                    for x in 0..<w {
                        row[x] &+= o
                    }
                }
            }
        }
    }
}

// MARK: - Prediction plane fusion
//
// The L0 chain builds the prediction plane P with the exact MC call sequence
// the reconstruction would use. Fusing P into the reconstruction is
// bit-identical to re-running the MC apply pass:
// - inter blocks: the apply adds a value independent of the destination, and
//   P holds exactly that value (added into a zero plane) → recon &+= P.
// - skip blocks: the apply REPLACES the region with a raw reference copy,
//   and P holds exactly that copy → recon = P. (Replacement matters: the
//   layer2 chroma reconstruction leaves small nonzero values in some skip
//   regions — its skip test indexes the luma-geometry skip map with the
//   chroma block grid — and the apply pass discards them, so adding would
//   diverge.)
// - intra blocks: the apply adds nothing and P is zero there → recon &+= 0.
// One function per block size (the per-plane block grid is 1:1 with the
// skip-map grid at every size: ceil(ceil(dx/2^k)/2^(5-k)) == ceil(dx/32)).

@inline(__always)
func fusePredictionPlane32(recon: inout [Int16], p: [Int16], skipMap: [BlockMode], width: Int, height: Int) {
    fusePredictionPlane(recon: &recon, p: p, skipMap: skipMap, width: width, height: height, blockSize: 32)
}

@inline(__always)
func fusePredictionPlane16(recon: inout [Int16], p: [Int16], skipMap: [BlockMode], width: Int, height: Int) {
    fusePredictionPlane(recon: &recon, p: p, skipMap: skipMap, width: width, height: height, blockSize: 16)
}

@inline(__always)
func fusePredictionPlane8(recon: inout [Int16], p: [Int16], skipMap: [BlockMode], width: Int, height: Int) {
    fusePredictionPlane(recon: &recon, p: p, skipMap: skipMap, width: width, height: height, blockSize: 8)
}

@inline(__always)
func fusePredictionPlane4(recon: inout [Int16], p: [Int16], skipMap: [BlockMode], width: Int, height: Int) {
    fusePredictionPlane(recon: &recon, p: p, skipMap: skipMap, width: width, height: height, blockSize: 4)
}

@inline(__always)
private func fusePredictionPlane(recon: inout [Int16], p: [Int16], skipMap: [BlockMode], width: Int, height: Int, blockSize: Int) {
    let bw = (width + blockSize - 1) / blockSize
    let bh = (height + blockSize - 1) / blockSize
    withUnsafePointers(p, mut: &recon) { pBase, rBase in
        for by in 0..<bh {
            let y0 = by * blockSize
            let h = min(blockSize, height - y0)
            for bx in 0..<bw {
                let x0 = bx * blockSize
                let w = min(blockSize, width - x0)
                if skipMap[by * bw + bx] != .inter {
                    for y in 0..<h {
                        let off = (y0 + y) * width + x0
                        rBase.advanced(by: off).update(from: pBase.advanced(by: off), count: w)
                    }
                } else {
                    for y in 0..<h {
                        let off = (y0 + y) * width + x0
                        let r = rBase.advanced(by: off)
                        let pp = pBase.advanced(by: off)
                        for x in 0..<w {
                            r[x] &+= pp[x]
                        }
                    }
                }
            }
        }
    }
}

/// Elementwise a −= b over all three planes (dimensions must match).
@inline(__always)
func subtractPlanes(_ a: inout Image16, _ b: PlaneData420) {
    subtractInt16(&a.y, b.y)
    subtractInt16(&a.cb, b.cb)
    subtractInt16(&a.cr, b.cr)
}

@inline(__always)
private func subtractInt16(_ a: inout [Int16], _ b: [Int16]) {
    let count = min(a.count, b.count)
    withUnsafePointers(mut: &a, b) { aBase, bBase in
        for i in 0..<count {
            aBase[i] &-= bBase[i]
        }
    }
}

/// Deep copy of an Image16 into fresh (non-pool) PlaneData420 storage, safe to
/// retain across frames. Pool-backed buffers may be recycled by their release
/// closures, so the L0 reference chain always owns plain arrays.
@inline(__always)
func copyImageToPlaneData420(_ img: Image16) -> PlaneData420 {
    var y = [Int16](repeating: 0, count: img.y.count)
    var cb = [Int16](repeating: 0, count: img.cb.count)
    var cr = [Int16](repeating: 0, count: img.cr.count)
    copyPlaneBuffers(y: img.y, cb: img.cb, cr: img.cr, intoY: &y, cb: &cb, cr: &cr)
    return PlaneData420(width: img.width, height: img.height, y: y, cb: cb, cr: cr)
}

/// Zeroes r0 in skip blocks (8x8 luma / 4x4 chroma at quarter resolution).
/// The decoder's skip copy overwrites these regions, so coding them is pure
/// rate waste; both reconstructions are driven by the skip copy, not r0.
@inline(__always)
func clearL0SkipResidual(img: inout Image16, skipMap: [BlockMode], fullDx: Int) {
    let bw = (fullDx + 31) / 32
    let dx = img.width
    let dy = img.height
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    withUnsafePointers(mut: &img.y, mut: &img.cb, mut: &img.cr) { yBuf, cbBuf, crBuf in
        for i in 0..<skipMap.count {
            if skipMap[i] != .inter {
                let bx = ((i % bw) * 32) / 4
                let by = ((i / bw) * 32) / 4
                for yy in by..<min(by + 8, dy) {
                    let off = yy * dx + bx
                    for xx in 0..<min(8, dx - bx) { yBuf[off + xx] = 0 }
                }
                let cx = bx / 2
                let cy = by / 2
                for yy in cy..<min(cy + 4, cbDy) {
                    let off = yy * cbDx + cx
                    for xx in 0..<min(4, cbDx - cx) {
                        cbBuf[off + xx] = 0
                        crBuf[off + xx] = 0
                    }
                }
            }
        }
    }
}
