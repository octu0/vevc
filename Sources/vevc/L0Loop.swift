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
func llAnalyzeLevel(_ plane: [Int16], w: Int, h: Int, blockSize: Int) -> [Int16] {
    let q = blockSize / 2
    let colCount = (w + blockSize - 1) / blockSize
    let rowCount = (h + blockSize - 1) / blockSize
    let llW = (w + 1) / 2
    let llH = (h + 1) / 2
    var ll = [Int16](repeating: 0, count: llW * llH)
    var scratch = [Int16](repeating: 0, count: blockSize * blockSize)
    let reader = Int16Reader(data: plane, width: w, height: h)
    for r in 0..<rowCount {
        for c in 0..<colCount {
            scratch.withUnsafeMutableBufferPointer { sb in
                let sBase = sb.baseAddress!
                let view = BlockView(base: sBase, width: blockSize, height: blockSize, stride: blockSize)
                reader.readBlock(x: c * blockSize, y: r * blockSize, width: blockSize, height: blockSize, into: view)
                switch blockSize {
                case 32: dwt2DBlock32(view)
                case 16: dwt2DBlock16(view)
                default: dwt2DBlock8(view)
                }
                ll.withUnsafeMutableBufferPointer { dst in
                    let dBase = dst.baseAddress!
                    let dx0 = c * q
                    let dy0 = r * q
                    let copyW = min(q, llW - dx0)
                    if 0 < copyW {
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
        }
    }
    return ll
}

/// Two-level LL analysis (32-block then 16-block) of a full-resolution
/// PlaneData420, producing the quarter-resolution sub1-domain planes — the
/// exact transform T whose output domain Base8 codes.
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
                        copyBlockPointer(from: ltrYPtr, to: currYPtr, bx: bx, by: by, stride: targetDx, blockSize: 8)
                        copyBlockPointer(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: 4)
                        copyBlockPointer(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: 4)
                    case .skip_prev:
                        copyBlockPointer(from: prevYPtr, to: currYPtr, bx: bx, by: by, stride: targetDx, blockSize: 8)
                        copyBlockPointer(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: 4)
                        copyBlockPointer(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: 4)
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
func clampPlaneToPixelRange(plane: inout [Int16]) {
    plane.withUnsafeMutableBufferPointer { ptr in
        guard let base = ptr.baseAddress else { return }
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

/// Elementwise a −= b over all three planes (dimensions must match).
func subtractPlanes(_ a: inout Image16, _ b: PlaneData420) {
    subtractInt16(&a.y, b.y)
    subtractInt16(&a.cb, b.cb)
    subtractInt16(&a.cr, b.cr)
}

@inline(__always)
private func subtractInt16(_ a: inout [Int16], _ b: [Int16]) {
    a.withUnsafeMutableBufferPointer { aBuf in
        b.withUnsafeBufferPointer { bBuf in
            let aBase = aBuf.baseAddress!
            let bBase = bBuf.baseAddress!
            for i in 0..<min(aBuf.count, bBuf.count) {
                aBase[i] &-= bBase[i]
            }
        }
    }
}

/// Deep copy into fresh (non-pool) storage, safe to retain across frames.
/// Pool-backed buffers may be recycled by their release closures, so the L0
/// reference chain always owns plain arrays.
func freshCopy(_ img: Image16) -> PlaneData420 {
    var y = [Int16](repeating: 0, count: img.y.count)
    var cb = [Int16](repeating: 0, count: img.cb.count)
    var cr = [Int16](repeating: 0, count: img.cr.count)
    y.withUnsafeMutableBufferPointer { dst in img.y.withUnsafeBufferPointer { dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) } }
    cb.withUnsafeMutableBufferPointer { dst in img.cb.withUnsafeBufferPointer { dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) } }
    cr.withUnsafeMutableBufferPointer { dst in img.cr.withUnsafeBufferPointer { dst.baseAddress!.update(from: $0.baseAddress!, count: $0.count) } }
    return PlaneData420(width: img.width, height: img.height, y: y, cb: cb, cr: cr)
}

/// Zeroes r0 in skip blocks (8x8 luma / 4x4 chroma at quarter resolution).
/// The decoder's skip copy overwrites these regions, so coding them is pure
/// rate waste; both reconstructions are driven by the skip copy, not r0.
func clearL0SkipResidual(img: inout Image16, skipMap: [BlockMode], fullDx: Int) {
    let bw = (fullDx + 31) / 32
    let dx = img.width
    let dy = img.height
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    img.y.withUnsafeMutableBufferPointer { yBuf in
        img.cb.withUnsafeMutableBufferPointer { cbBuf in
            img.cr.withUnsafeMutableBufferPointer { crBuf in
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
    }
}
