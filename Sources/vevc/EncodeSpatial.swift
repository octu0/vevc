// MARK: - Encode Spatial
import Foundation

@inline(__always)
func zeroBlockSubbands32(view: BlockView) {
    let base = view.base
    clearBlockRegion(base: base.advanced(by: 16), width: 16, height: 16, stride: 32)
    clearBlockRegion(base: base.advanced(by: 16 * 32), width: 16, height: 16, stride: 32)
    clearBlockRegion(base: base.advanced(by: 16 * 32 + 16), width: 16, height: 16, stride: 32)
}

@inline(__always)
func zeroBlocksSubbands32(blocks: inout [BlockView]) {
    for i in 0..<blocks.count {
        zeroBlockSubbands32(view: blocks[i])
    }
}

@inline(__always)
func zeroBlockSubbands16(view: BlockView) {
    let base = view.base
    clearBlockRegion(base: base.advanced(by: 8), width: 8, height: 8, stride: 16)
    clearBlockRegion(base: base.advanced(by: 8 * 16), width: 8, height: 8, stride: 16)
    clearBlockRegion(base: base.advanced(by: 8 * 16 + 8), width: 8, height: 8, stride: 16)
}

@inline(__always)
func zeroBlocksSubbands16(blocks: inout [BlockView]) {
    for i in 0..<blocks.count {
        zeroBlockSubbands16(view: blocks[i])
    }
}

/// Base8 blocks are a single 8x8 DWT stage; temporal thinning codes the
/// whole block as all-zero coefficients (legal like the L1/L2 cadence:
/// dequant(zero) == 0 keeps encoder/decoder reconstructions identical).
@inline(__always)
func zeroBlockSubbandsBase8(view: BlockView) {
    clearBlockRegion(base: view.base, width: 8, height: 8, stride: 8)
}

@inline(__always)
func zeroBlocksSubbandsBase8(blocks: inout [BlockView]) {
    for i in 0..<blocks.count {
        zeroBlockSubbandsBase8(view: blocks[i])
    }
}

@inline(__always)
func shouldZeroCadence(cadence: Int, gopPosition: Int) -> Bool {
    switch cadence {
    case 0:
        return true
    case let n where 2 <= n:
        return gopPosition % n != 0
    default:
        return false
    }
}

/// I-frame encode, profile 0x01: parent-conditioned entropy contexts with
/// the shipped static tables.
@inline(__always)
func encodeSpatialLayersIntra(pd: PlaneData420, pool: BlockViewPool, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int) async throws -> ([UInt8], PlaneData420, MotionVectors, [Int], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height

    let qtY2 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 2)
    let qtC2 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 2)
    let qtY1 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 1)
    let qtC1 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 1)
    let qtY0 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 0)
    let qtC0 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 0)

    let resPd = PlaneData420(width: dx, height: dy, y: pd.y, cb: pd.cb, cr: pd.cr)

    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks, releaseL2) = await preparePlaneLayer32(pd: resPd, pool: pool, qtY: qtY2, qtC: qtC2)
    defer { releaseL2() }
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks, releaseL1) = await preparePlaneLayer16(pd: sub2, pool: pool, qtY: qtY1, qtC: qtC1)
    defer { releaseL1() }
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks, releaseBase) = await encodePlaneBase8IntraProfile1(pd: sub1, pool: pool, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold, selectModel: unifiedSelectModel)
    defer { releaseBase() }

    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)

    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = encodeLayer16Payload(dx: sub2.width, dy: sub2.height, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks, histories: nil, selectModel: unifiedSelectModel, updateHistory: true)

    let (mutReconL1Y, r1Y) = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let (mutReconL1Cb, r1Cb) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let (mutReconL1Cr, r1Cr) = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    defer { r1Y(); r1Cb(); r1Cr() }

    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    let layer2 = encodeLayer32Payload(dx: pd.width, dy: pd.height, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks, histories: nil, selectModel: unifiedSelectModel, updateHistory: true)

    let (reconstructed, releaseRecon) = finishIntraReconstruction(pd: pd, pool: pool, l1Img: l1Img, l2yBlocks: l2yBlocks, l2cbBlocks: l2cbBlocks, l2crBlocks: l2crBlocks, qtY2: qtY2, qtC2: qtC2)

    debugLog({
        return "  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes"
    }())

    let out = serializeIntraFrame(profile: 0x01, layer0: layer0, layer1: layer1, layer2: layer2)
    return (out, reconstructed, MotionVectors.empty, [], releaseRecon)
}

/// I-frame encode, profile 0x02: parent-free entropy contexts against the
/// parent-free static tables (EntropyCodec.swift), and the L0 closed-loop
/// chain starts here — the I-frame L0 reference is the dequantized Base8
/// output, identical on both sides by construction (One-Pyramid §4).
@inline(__always)
func encodeSpatialLayersIntraForProfile2(pd: PlaneData420, pool: BlockViewPool, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, l0State: L0RefState) async throws -> ([UInt8], PlaneData420, MotionVectors, [Int], @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height

    let qtY2 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 2)
    let qtC2 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 2)
    let qtY1 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 1)
    let qtC1 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 1)
    let qtY0 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 0)
    let qtC0 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 0)

    let resPd = PlaneData420(width: dx, height: dy, y: pd.y, cb: pd.cb, cr: pd.cr)

    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks, releaseL2) = await preparePlaneLayer32(pd: resPd, pool: pool, qtY: qtY2, qtC: qtC2)
    defer { releaseL2() }
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks, releaseL1) = await preparePlaneLayer16(pd: sub2, pool: pool, qtY: qtY1, qtC: qtC1)
    defer { releaseL1() }
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks, releaseBase) = await encodePlaneBase8IntraProfile2(pd: sub1, pool: pool, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold, selectModel: unifiedSelectModelParentFree)
    defer { releaseBase() }

    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)

    // The I-frame L0 reference: dequantized Base8 output.
    let ref = copyImageToPlaneData420(baseImg)
    l0State.prev = ref
    l0State.ltr = ref

    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = encodeLayer16Payload(dx: sub2.width, dy: sub2.height, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: parentFreeParents8(count: base8YBlocks.count), parentCbBlocks: parentFreeParents8(count: base8CbBlocks.count), parentCrBlocks: parentFreeParents8(count: base8CrBlocks.count), histories: nil, selectModel: unifiedSelectModelParentFree, updateHistory: true)

    let (mutReconL1Y, r1Y) = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let (mutReconL1Cb, r1Cb) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let (mutReconL1Cr, r1Cr) = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    defer { r1Y(); r1Cb(); r1Cr() }

    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    let layer2 = encodeLayer32Payload(dx: pd.width, dy: pd.height, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: parentFreeParents16(count: l1yBlocks.count), parentCbBlocks: parentFreeParents16(count: l1cbBlocks.count), parentCrBlocks: parentFreeParents16(count: l1crBlocks.count), histories: nil, selectModel: unifiedSelectModelParentFree, updateHistory: true)

    let (reconstructed, releaseRecon) = finishIntraReconstruction(pd: pd, pool: pool, l1Img: l1Img, l2yBlocks: l2yBlocks, l2cbBlocks: l2cbBlocks, l2crBlocks: l2crBlocks, qtY2: qtY2, qtC2: qtC2)

    debugLog({
        return "  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes"
    }())

    let out = serializeIntraFrame(profile: 0x02, layer0: layer0, layer1: layer1, layer2: layer2)
    return (out, reconstructed, MotionVectors.empty, [], releaseRecon)
}

/// Layer2 reconstruction + the decoder's exact I-frame deblock sequence (its
/// mvs==nil branch): plain luma + applyDeblockingFilter16 chroma. The encoder
/// previously used applyDeblockingFilterChroma16 with empty mvs, which
/// differs by ±1 at some block edges; with the L0 closed loop that drift
/// reaches the entropy contexts (LL2 slot couples P into the parent blocks)
/// in P frames and breaks decoder-exact tracking.
@inline(__always)
private func finishIntraReconstruction(
    pd: PlaneData420,
    pool: BlockViewPool,
    l1Img: Image16,
    l2yBlocks: [BlockView],
    l2cbBlocks: [BlockView],
    l2crBlocks: [BlockView],
    qtY2: QuantizationTable,
    qtC2: QuantizationTable
) -> (PlaneData420, @Sendable () -> Void) {
    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    let (reconL2Y, r2Y) = reconstructPlaneLayer32YWithoutSkipMap(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool)
    var mutReconL2Y = reconL2Y
    let (reconL2Cb, r2Cb) = reconstructPlaneLayer32CbWithoutSkipMap(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cb = reconL2Cb
    let (reconL2Cr, r2Cr) = reconstructPlaneLayer32CrWithoutSkipMap(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cr = reconL2Cr

    applyDeblockingFilter32WithoutMVs(plane: &mutReconL2Y, width: dx, height: dy, qStep: (Int(qtY2.step) + 8) >> 4)
    applyDeblockingFilter16(plane: &mutReconL2Cb, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4)
    applyDeblockingFilter16(plane: &mutReconL2Cr, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4)

    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)
    return (reconstructed, { r2Y(); r2Cb(); r2Cr() })
}

@inline(__always)
private func serializeIntraFrame(profile: UInt8, layer0: [UInt8], layer1: [UInt8], layer2: [UInt8]) -> [UInt8] {
    var out: [UInt8] = []
    let frameHeader = VEVCFrameHeader(frameType: .iFrame, hasRefDir: false, skipMapSize: 0, mvsSize: 0, refDirSize: 0, lumaOffset: 0, chromaOffset: 0, layer0Size: layer0.count, layer1Size: layer1.count, layer2Size: layer2.count)
    out.append(contentsOf: frameHeader.serialize(profile: profile))
    out.append(contentsOf: layer0)
    out.append(contentsOf: layer1)
    out.append(contentsOf: layer2)
    return out
}

/// Profile 0x02 skip decision: skip_ltr for blocks static since the GOP head
/// that match the LTR input, skip_prev for blocks static for 4+ frames whose
/// previous reconstruction also matches the current input. Both skip modes
/// require the reconstruction the decoder would copy to match the current
/// input — an input-only match would freeze a stale reconstruction (vanished
/// subtitles, half-repaired blocks) until the next I frame. Updates
/// staticCounters in place.
@inline(__always)
func computeProfile2SkipMap(pd: PlaneData420, prevInput: PlaneData420, ltrInput: PlaneData420, predictedPd: PlaneData420, ltrRecon: PlaneData420, gopPosition: Int, ltrAge: Int, skipThreshold: Int, reconThresholdScale: Int, staticCounters: inout [Int]) async -> [BlockMode] {
    let dx = pd.width
    let dy = pd.height
    let bw = (dx + 31) / 32
    let bh = (dy + 31) / 32
    let blockCount = bw * bh
    var skipMap = [BlockMode](repeating: .inter, count: blockCount)
    let skipThresholdPerPixel = skipThreshold
    let prevStaticCounters = staticCounters

    let curYWrapper = UnsafePointerWrapper(base: pd.y.withUnsafeBufferPointer { $0.baseAddress! })
    let curCbWrapper = UnsafePointerWrapper(base: pd.cb.withUnsafeBufferPointer { $0.baseAddress! })
    let curCrWrapper = UnsafePointerWrapper(base: pd.cr.withUnsafeBufferPointer { $0.baseAddress! })

    let prevInYWrapper = UnsafePointerWrapper(base: prevInput.y.withUnsafeBufferPointer { $0.baseAddress! })
    let prevInCbWrapper = UnsafePointerWrapper(base: prevInput.cb.withUnsafeBufferPointer { $0.baseAddress! })
    let prevInCrWrapper = UnsafePointerWrapper(base: prevInput.cr.withUnsafeBufferPointer { $0.baseAddress! })

    let ltrInYWrapper = UnsafePointerWrapper(base: ltrInput.y.withUnsafeBufferPointer { $0.baseAddress! })
    let ltrInCbWrapper = UnsafePointerWrapper(base: ltrInput.cb.withUnsafeBufferPointer { $0.baseAddress! })
    let ltrInCrWrapper = UnsafePointerWrapper(base: ltrInput.cr.withUnsafeBufferPointer { $0.baseAddress! })

    let prevReconYWrapper = UnsafePointerWrapper(base: predictedPd.y.withUnsafeBufferPointer { $0.baseAddress! })
    let prevReconCbWrapper = UnsafePointerWrapper(base: predictedPd.cb.withUnsafeBufferPointer { $0.baseAddress! })
    let prevReconCrWrapper = UnsafePointerWrapper(base: predictedPd.cr.withUnsafeBufferPointer { $0.baseAddress! })

    let ltrReconYWrapper = UnsafePointerWrapper(base: ltrRecon.y.withUnsafeBufferPointer { $0.baseAddress! })
    let ltrReconCbWrapper = UnsafePointerWrapper(base: ltrRecon.cb.withUnsafeBufferPointer { $0.baseAddress! })
    let ltrReconCrWrapper = UnsafePointerWrapper(base: ltrRecon.cr.withUnsafeBufferPointer { $0.baseAddress! })

    let matchResults = await withTaskGroup(of: [(Int, Bool, Bool, Bool)].self) { group in
        let batchSize = 128
        for batchStart in stride(from: 0, to: blockCount, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, blockCount)
            group.addTask {
                var results = [(Int, Bool, Bool, Bool)]()
                results.reserveCapacity(batchEnd - batchStart)

                let curY = curYWrapper.base, curCb = curCbWrapper.base, curCr = curCrWrapper.base
                let prevInY = prevInYWrapper.base, prevInCb = prevInCbWrapper.base, prevInCr = prevInCrWrapper.base
                let ltrInY = ltrInYWrapper.base, ltrInCb = ltrInCbWrapper.base, ltrInCr = ltrInCrWrapper.base
                let prevReconY = prevReconYWrapper.base, prevReconCb = prevReconCbWrapper.base, prevReconCr = prevReconCrWrapper.base
                let ltrReconY = ltrReconYWrapper.base, ltrReconCb = ltrReconCbWrapper.base, ltrReconCr = ltrReconCrWrapper.base

                for i in batchStart..<batchEnd {
                    let bx = (i % bw) * 32
                    let by = (i / bw) * 32
                    let prevCount = prevStaticCounters[i]

                    var allSubBlocksMatchPrev = true
                    var allSubBlocksMatchLtr = false
                    var allSubBlocksMatchPrevRecon = false

                    for sy in 0..<2 {
                        for sx in 0..<2 {
                            let subX = bx + sx * 16
                            let subY = by + sy * 16
                            let mw = min(16, dx - subX)
                            let mh = min(16, dy - subY)
                            if mw <= 0 || mh <= 0 { continue }
                            let mwc = ((mw + 1) / 2)
                            let mhc = ((mh + 1) / 2)

                            let area = mw * mh + mwc * mhc * 2
                            let blockThreshold = skipThresholdPerPixel * area

                            let sadPrevIn: Int
                            if mw == 16 && mh == 16 && mwc == 8 && mhc == 8 {
                                sadPrevIn = computeZeroSAD16x16(cY: curY, rY: prevInY, cCb: curCb, rCb: prevInCb, cCr: curCr, rCr: prevInCr, bx: subX, by: subY, width: dx, limit: blockThreshold)
                            } else {
                                sadPrevIn = computeZeroSADSubBlock(cY: curY, rY: prevInY, cCb: curCb, rCb: prevInCb, cCr: curCr, rCr: prevInCr, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc, limit: blockThreshold)
                            }

                            if blockThreshold < sadPrevIn {
                                allSubBlocksMatchPrev = false
                                break
                            }
                        }
                        if allSubBlocksMatchPrev != true { break }
                    }

                    let nextStaticCount = if allSubBlocksMatchPrev { prevCount + 1 } else { 0 }

                    if nextStaticCount == ltrAge {
                        allSubBlocksMatchLtr = true
                        for sy in 0..<2 {
                            for sx in 0..<2 {
                                let subX = bx + sx * 16
                                let subY = by + sy * 16
                                let mw = min(16, dx - subX)
                                let mh = min(16, dy - subY)
                                if mw <= 0 || mh <= 0 { continue }
                                let mwc = ((mw + 1) / 2)
                                let mhc = ((mh + 1) / 2)

                                let area = mw * mh + mwc * mhc * 2
                                let blockThreshold = skipThresholdPerPixel * area

                                let sadLtrIn: Int
                                if mw == 16 && mh == 16 && mwc == 8 && mhc == 8 {
                                    sadLtrIn = computeZeroSAD16x16(cY: curY, rY: ltrInY, cCb: curCb, rCb: ltrInCb, cCr: curCr, rCr: ltrInCr, bx: subX, by: subY, width: dx, limit: blockThreshold)
                                } else {
                                    sadLtrIn = computeZeroSADSubBlock(cY: curY, rY: ltrInY, cCb: curCb, rCb: ltrInCb, cCr: curCr, rCr: ltrInCr, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc, limit: blockThreshold)
                                }

                                if blockThreshold < sadLtrIn {
                                    allSubBlocksMatchLtr = false
                                    break
                                }

                                // The decoder copies the LTR *reconstruction*;
                                // an input-only match would freeze whatever
                                // stale content it carries (skip_prev has the
                                // same guard against its own copy source).
                                let reconBlockThreshold = blockThreshold * reconThresholdScale
                                let sadLtrRecon: Int
                                if mw == 16 && mh == 16 && mwc == 8 && mhc == 8 {
                                    sadLtrRecon = computeZeroSAD16x16(cY: curY, rY: ltrReconY, cCb: curCb, rCb: ltrReconCb, cCr: curCr, rCr: ltrReconCr, bx: subX, by: subY, width: dx, limit: reconBlockThreshold)
                                } else {
                                    sadLtrRecon = computeZeroSADSubBlock(cY: curY, rY: ltrReconY, cCb: curCb, rCb: ltrReconCb, cCr: curCr, rCr: ltrReconCr, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc, limit: reconBlockThreshold)
                                }

                                if reconBlockThreshold < sadLtrRecon {
                                    allSubBlocksMatchLtr = false
                                    break
                                }
                            }
                            if allSubBlocksMatchLtr != true { break }
                        }
                    }

                    let isLtrMatch = allSubBlocksMatchLtr && (nextStaticCount == ltrAge)
                    if isLtrMatch != true && allSubBlocksMatchPrev && (3 < nextStaticCount) {
                        allSubBlocksMatchPrevRecon = true
                        for sy in 0..<2 {
                            for sx in 0..<2 {
                                let subX = bx + sx * 16
                                let subY = by + sy * 16
                                let mw = min(16, dx - subX)
                                let mh = min(16, dy - subY)
                                if mw <= 0 || mh <= 0 { continue }
                                let mwc = ((mw + 1) / 2)
                                let mhc = ((mh + 1) / 2)

                                let area = mw * mh + mwc * mhc * 2
                                let blockThreshold = skipThresholdPerPixel * area
                                let reconBlockThreshold = blockThreshold * reconThresholdScale

                                let sadPrevRecon: Int
                                if mw == 16 && mh == 16 && mwc == 8 && mhc == 8 {
                                    sadPrevRecon = computeZeroSAD16x16(cY: curY, rY: prevReconY, cCb: curCb, rCb: prevReconCb, cCr: curCr, rCr: prevReconCr, bx: subX, by: subY, width: dx, limit: reconBlockThreshold)
                                } else {
                                    sadPrevRecon = computeZeroSADSubBlock(cY: curY, rY: prevReconY, cCb: curCb, rCb: prevReconCb, cCr: curCr, rCr: prevReconCr, bx: subX, by: subY, width: dx, height: dy, subWidth: mw, subHeight: mh, subWc: mwc, subHc: mhc, limit: reconBlockThreshold)
                                }

                                if reconBlockThreshold < sadPrevRecon {
                                    allSubBlocksMatchPrevRecon = false
                                    break
                                }
                            }
                            if allSubBlocksMatchPrevRecon != true { break }
                        }
                    }

                    results.append((i, allSubBlocksMatchPrev, allSubBlocksMatchLtr, allSubBlocksMatchPrevRecon))
                }
                return results
            }
        }

        var allResults = [(Int, Bool, Bool, Bool)]()
        for await batch in group {
            allResults.append(contentsOf: batch)
        }
        return allResults
    }

    for (i, allSubBlocksMatchPrev, allSubBlocksMatchLtr, allSubBlocksMatchPrevRecon) in matchResults {
        if allSubBlocksMatchPrev {
            staticCounters[i] += 1
        } else {
            staticCounters[i] = 0
        }

        switch true {
        case allSubBlocksMatchLtr && staticCounters[i] == ltrAge:
            skipMap[i] = .skip_ltr
        case allSubBlocksMatchPrev && allSubBlocksMatchPrevRecon && (3 < staticCounters[i]):
            skipMap[i] = .skip_prev
        default:
            skipMap[i] = .inter
        }
    }

    return skipMap
}

/// P-frame encode, profile 0x01: bidirectional MC residual coding without
/// skip blocks. Profile 0x02 lives in encodeSpatialLayersForProfile2 — the
/// caller selects the pipeline, keeping each one branch-free.
@inline(__always)
func encodeSpatialLayers(pd: PlaneData420, pool: BlockViewPool, predictedPd: PlaneData420, nextPd: PlaneData420, prevInput: PlaneData420, ltrInput: PlaneData420, prevMVs: MotionVectors?, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int, gopPosition: Int, cachedNextSub2: [Int16]?, cachedNextSub1: [Int16]?) async throws -> ([UInt8], PlaneData420, MotionVectors, [Int], @Sendable () -> Void, [Int16], [Int16]) {
    let pPd = predictedPd
    let nPd = nextPd

    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    let qtY2 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 2)
    let qtC2 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 2)
    let qtY1 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 1)
    let qtC1 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 1)
    let qtY0 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 0)
    let qtC0 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 0)

    let (mvs, sads, refDirs, _, nextSub2Res, nextSub1Res) = await computeBidirectionalMotionVectors(curr: pd, prev: pPd, next: nPd, prevMVs: prevMVs ?? MotionVectors.empty, pool: pool, roundOffset: roundOffset, gopPosition: gopPosition, skipMap: [], cachedNextSub2: cachedNextSub2, cachedNextSub1: cachedNextSub1, dualOut: nil)

    var mutPdY = pool.getInt16(count: pd.y.count)
    var mutPdCb = pool.getInt16(count: pd.cb.count)
    var mutPdCr = pool.getInt16(count: pd.cr.count)

    copyPlaneBuffers(y: pd.y, cb: pd.cb, cr: pd.cr, intoY: &mutPdY, cb: &mutPdCb, cr: &mutPdCr)
    let mvsConst = mvs
    let refDirsConst = refDirs
    async let tY = { [mvsConst, refDirsConst] () -> [Int16] in
        var y = mutPdY
        subtractScaledBidirectionalMotionCompensationLumaWithoutSkipMap(plane: &y, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvsConst, refDirs: refDirsConst, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
        return y
    }()
    async let tCb = { [mvsConst, refDirsConst] () -> [Int16] in
        var cb = mutPdCb
        subtractScaledBidirectionalMotionCompensationChromaWithoutSkipMap(plane: &cb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvsConst, refDirs: refDirsConst, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        return cb
    }()
    async let tCr = { [mvsConst, refDirsConst] () -> [Int16] in
        var cr = mutPdCr
        subtractScaledBidirectionalMotionCompensationChromaWithoutSkipMap(plane: &cr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvsConst, refDirs: refDirsConst, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        return cr
    }()

    let resY = await tY
    let resCb = await tCb
    let resCr = await tCr

    let resPd = PlaneData420(width: dx, height: dy, y: resY, cb: resCb, cr: resCr)

    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks, releaseL2) = await preparePlaneLayer32(pd: resPd, pool: pool, qtY: qtY2, qtC: qtC2)
    defer { releaseL2() }
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks, releaseL1) = await preparePlaneLayer16(pd: sub2, pool: pool, qtY: qtY1, qtC: qtC1)
    defer { releaseL1() }
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks, releaseBase) = await encodePlaneBase8PFrame(pd: sub1, pool: pool, sads: sads, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold)
    defer { releaseBase() }

    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)

    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = encodeLayer16Payload(dx: sub2.width, dy: sub2.height, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks, histories: nil, selectModel: unifiedSelectModel, updateHistory: true)

    let (mutReconL1Y, r1Y) = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let (mutReconL1Cb, r1Cb) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let (mutReconL1Cr, r1Cr) = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    defer { r1Y(); r1Cb(); r1Cr() }

    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    let layer2 = encodeLayer32Payload(dx: pd.width, dy: pd.height, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks, histories: nil, selectModel: unifiedSelectModel, updateHistory: true)

    let (reconL2Y, r2Y) = reconstructPlaneLayer32YWithoutSkipMap(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool)
    var mutReconL2Y = reconL2Y
    let (reconL2Cb, r2Cb) = reconstructPlaneLayer32CbWithoutSkipMap(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cb = reconL2Cb
    let (reconL2Cr, r2Cr) = reconstructPlaneLayer32CrWithoutSkipMap(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cr = reconL2Cr

    await applyScaledBidirectionalMotionCompensationLumaWithoutSkipMap(plane: &mutReconL2Y, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvs, refDirs: refDirs, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
    await applyScaledBidirectionalMotionCompensationChromaWithoutSkipMap(plane: &mutReconL2Cb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    await applyScaledBidirectionalMotionCompensationChromaWithoutSkipMap(plane: &mutReconL2Cr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)

    let mvData = encodeMVsProfile1(mvs: mvs)

    let refDirBuf = encodeRefDirsProfile1(refDirs: refDirs)

    // Must match the decoder's deblock invocation exactly (mvs + skipMap
    // variants) — see the profile-2 pipeline for the asymmetry history.
    applyDeblockingFilter32WithMVs(plane: &mutReconL2Y, width: dx, height: dy, qStep: (Int(qtY2.step) + 8) >> 4, mvs: mvs)
    applyDeblockingFilterChroma16WithMVs(plane: &mutReconL2Cb, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvs)
    applyDeblockingFilterChroma16WithMVs(plane: &mutReconL2Cr, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvs)

    let releaseY: @Sendable () -> Void = { r2Y() }
    let releaseCb: @Sendable () -> Void = { r2Cb() }
    let releaseCr: @Sendable () -> Void = { r2Cr() }

    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)

    debugLog({
        return "  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes"
    }())

    var out: [UInt8] = []
    let frameHeader = VEVCFrameHeader(frameType: .pFrame, hasRefDir: true, skipMapSize: 0, mvsSize: mvData.count, refDirSize: refDirBuf.count, lumaOffset: 0, chromaOffset: 0, layer0Size: layer0.count, layer1Size: layer1.count, layer2Size: layer2.count)
    out.append(contentsOf: frameHeader.serialize(profile: 0x01))
    out.append(contentsOf: mvData)
    out.append(contentsOf: refDirBuf)
    out.append(contentsOf: layer0)
    out.append(contentsOf: layer1)
    out.append(contentsOf: layer2)

    return (out, reconstructed, mvs, sads, { releaseY(); releaseCb(); releaseCr() }, nextSub2Res, nextSub1Res)
}

@inline(__always)
func encodeSpatialLayersWithDumpWriter(pd: PlaneData420, pool: BlockViewPool, predictedPd: PlaneData420, nextPd: PlaneData420, prevInput: PlaneData420, ltrInput: PlaneData420, prevMVs: MotionVectors?, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int, gopPosition: Int, cachedNextSub2: [Int16]?, cachedNextSub1: [Int16]?, dumpWriter: CoeffDumpWriter) async throws -> ([UInt8], PlaneData420, MotionVectors, [Int], @Sendable () -> Void, [Int16], [Int16]) {
    let pPd = predictedPd
    let nPd = nextPd

    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    let qtY2 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 2)
    let qtC2 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 2)
    let qtY1 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 1)
    let qtC1 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 1)
    let qtY0 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 0)
    let qtC0 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 0)

    let (mvs, sads, refDirs, _, nextSub2Res, nextSub1Res) = await computeBidirectionalMotionVectors(curr: pd, prev: pPd, next: nPd, prevMVs: prevMVs ?? MotionVectors.empty, pool: pool, roundOffset: roundOffset, gopPosition: gopPosition, skipMap: [], cachedNextSub2: cachedNextSub2, cachedNextSub1: cachedNextSub1, dualOut: nil)

    var mutPdY = pool.getInt16(count: pd.y.count)
    var mutPdCb = pool.getInt16(count: pd.cb.count)
    var mutPdCr = pool.getInt16(count: pd.cr.count)

    copyPlaneBuffers(y: pd.y, cb: pd.cb, cr: pd.cr, intoY: &mutPdY, cb: &mutPdCb, cr: &mutPdCr)
    let mvsConst = mvs
    let refDirsConst = refDirs
    async let tY = { [mvsConst, refDirsConst] () -> [Int16] in
        var y = mutPdY
        subtractScaledBidirectionalMotionCompensationLumaWithoutSkipMap(plane: &y, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvsConst, refDirs: refDirsConst, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
        return y
    }()
    async let tCb = { [mvsConst, refDirsConst] () -> [Int16] in
        var cb = mutPdCb
        subtractScaledBidirectionalMotionCompensationChromaWithoutSkipMap(plane: &cb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvsConst, refDirs: refDirsConst, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        return cb
    }()
    async let tCr = { [mvsConst, refDirsConst] () -> [Int16] in
        var cr = mutPdCr
        subtractScaledBidirectionalMotionCompensationChromaWithoutSkipMap(plane: &cr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvsConst, refDirs: refDirsConst, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        return cr
    }()

    let resY = await tY
    let resCb = await tCb
    let resCr = await tCr

    let resPd = PlaneData420(width: dx, height: dy, y: resY, cb: resCb, cr: resCr)

    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks, releaseL2) = await preparePlaneLayer32(pd: resPd, pool: pool, qtY: qtY2, qtC: qtC2)
    defer { releaseL2() }
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks, releaseL1) = await preparePlaneLayer16(pd: sub2, pool: pool, qtY: qtY1, qtC: qtC1)
    defer { releaseL1() }
    let (layer0, baseRecon, base8YBlocks, base8CbBlocks, base8CrBlocks, releaseBase) = await encodePlaneBase8PFrame(pd: sub1, pool: pool, sads: sads, qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold)
    defer { releaseBase() }

    let baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)

    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)
    let layer1 = encodeLayer16Payload(dx: sub2.width, dy: sub2.height, qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold, yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks, histories: nil, selectModel: unifiedSelectModel, updateHistory: true)

    let (mutReconL1Y, r1Y) = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let (mutReconL1Cb, r1Cb) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let (mutReconL1Cr, r1Cr) = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    defer { r1Y(); r1Cb(); r1Cr() }

    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    let layer2 = encodeLayer32Payload(dx: pd.width, dy: pd.height, qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold, yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks, parentYBlocks: l1yBlocks, parentCbBlocks: l1cbBlocks, parentCrBlocks: l1crBlocks, histories: nil, selectModel: unifiedSelectModel, updateHistory: true)

    dumpWriter.stash("L0Y", blocks: base8YBlocks, planeW: sub1.width, planeH: sub1.height, blockSize: 8, includeLL: true)
    dumpWriter.stash("L0Cb", blocks: base8CbBlocks, planeW: (sub1.width + 1) / 2, planeH: (sub1.height + 1) / 2, blockSize: 8, includeLL: true)
    dumpWriter.stash("L0Cr", blocks: base8CrBlocks, planeW: (sub1.width + 1) / 2, planeH: (sub1.height + 1) / 2, blockSize: 8, includeLL: true)

    dumpWriter.stash("L1Y", blocks: l1yBlocks, planeW: sub2.width, planeH: sub2.height, blockSize: 16, includeLL: false)
    dumpWriter.stash("L1Cb", blocks: l1cbBlocks, planeW: l1cbDx, planeH: l1cbDy, blockSize: 16, includeLL: false)
    dumpWriter.stash("L1Cr", blocks: l1crBlocks, planeW: l1cbDx, planeH: l1cbDy, blockSize: 16, includeLL: false)

    dumpWriter.stash("L2Y", blocks: l2yBlocks, planeW: dx, planeH: dy, blockSize: 32, includeLL: false)
    dumpWriter.stash("L2Cb", blocks: l2cbBlocks, planeW: cbDx, planeH: cbDy, blockSize: 32, includeLL: false)
    dumpWriter.stash("L2Cr", blocks: l2crBlocks, planeW: cbDx, planeH: cbDy, blockSize: 32, includeLL: false)

    dumpWriter.finalizePFrame(
        gopPosition: gopPosition, width: dx, height: dy, predictedPd: pPd,
        l1yBlocks: l1yBlocks, l1cbBlocks: l1cbBlocks, l1crBlocks: l1crBlocks,
        b8yBlocks: base8YBlocks, b8cbBlocks: base8CbBlocks, b8crBlocks: base8CrBlocks,
        sub2W: sub2.width, sub2H: sub2.height, sub1W: sub1.width, sub1H: sub1.height,
        qtY2: qtY2, qtC2: qtC2, qtY1: qtY1, qtC1: qtC1, qtY0: qtY0, qtC0: qtC0,
        layer0Bytes: layer0.count, layer1Bytes: layer1.count, layer2Bytes: layer2.count
    )

    let (reconL2Y, r2Y) = reconstructPlaneLayer32YWithoutSkipMap(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool)
    var mutReconL2Y = reconL2Y
    let (reconL2Cb, r2Cb) = reconstructPlaneLayer32CbWithoutSkipMap(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cb = reconL2Cb
    let (reconL2Cr, r2Cr) = reconstructPlaneLayer32CrWithoutSkipMap(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool)
    var mutReconL2Cr = reconL2Cr

    await applyScaledBidirectionalMotionCompensationLumaWithoutSkipMap(plane: &mutReconL2Y, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvs, refDirs: refDirs, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
    await applyScaledBidirectionalMotionCompensationChromaWithoutSkipMap(plane: &mutReconL2Cb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
    await applyScaledBidirectionalMotionCompensationChromaWithoutSkipMap(plane: &mutReconL2Cr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvs, refDirs: refDirs, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)

    let mvData = encodeMVsProfile1(mvs: mvs)

    let refDirBuf = encodeRefDirsProfile1(refDirs: refDirs)

    // Must match the decoder's deblock invocation exactly (mvs + skipMap
    // variants) — see the profile-2 pipeline for the asymmetry history.
    applyDeblockingFilter32WithMVs(plane: &mutReconL2Y, width: dx, height: dy, qStep: (Int(qtY2.step) + 8) >> 4, mvs: mvs)
    applyDeblockingFilterChroma16WithMVs(plane: &mutReconL2Cb, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvs)
    applyDeblockingFilterChroma16WithMVs(plane: &mutReconL2Cr, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvs)

    let releaseY: @Sendable () -> Void = { r2Y() }
    let releaseCb: @Sendable () -> Void = { r2Cb() }
    let releaseCr: @Sendable () -> Void = { r2Cr() }

    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)

    debugLog({
        return "  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes"
    }())

    var out: [UInt8] = []
    let frameHeader = VEVCFrameHeader(frameType: .pFrame, hasRefDir: true, skipMapSize: 0, mvsSize: mvData.count, refDirSize: refDirBuf.count, lumaOffset: 0, chromaOffset: 0, layer0Size: layer0.count, layer1Size: layer1.count, layer2Size: layer2.count)
    out.append(contentsOf: frameHeader.serialize(profile: 0x01))
    out.append(contentsOf: mvData)
    out.append(contentsOf: refDirBuf)
    out.append(contentsOf: layer0)
    out.append(contentsOf: layer1)
    out.append(contentsOf: layer2)

    return (out, reconstructed, mvs, sads, { releaseY(); releaseCb(); releaseCr() }, nextSub2Res, nextSub1Res)
}

/// Restricts motion-masking detail omission to frames with deep quantization (rate pressure). At comfortable bitrates, details are preserved to maintain legibility for pursuit-eye-movement content.
let motionMaskingMinQStep: Int = 2048

/// P-frame encode, profile 0x02: the profile-1 pipeline plus the skip map
/// (skip_prev / skip_ltr block copies), the L0 closed loop when an l0State
/// chain is attached, and backward-adaptive entropy histories.
@inline(__always)
func encodeSpatialLayersForProfile2(pd: PlaneData420, pool: BlockViewPool, predictedPd: PlaneData420, nextPd: PlaneData420, prevInput: PlaneData420, ltrInput: PlaneData420, prevMVs: MotionVectors?, maxbitrate: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, roundOffset: Int, gopPosition: Int, ltrAge: Int, skipThreshold: Int, reconThresholdScale: Int, staticCounters: inout [Int], cachedNextSub2: [Int16]?, cachedNextSub1: [Int16]?, entropyHistories: FrameEntropyHistories?, syntaxContext: SyntaxContextModels, l0State: L0RefState, l2Cadence: Int, l1Cadence: Int, l0Cadence: Int, framerate: Int, motionMaskingPx: Int, adjustedStep: Int, smooth: Int, updateL0Prev: Bool, skipModel: SkipDecider?, ransContextWorkspace: rANSContextWorkspace?, dumpWriter: CoeffDumpWriter?) async throws -> ([UInt8], PlaneData420, MotionVectors, [Int], @Sendable () -> Void, [Int16], [Int16], [BlockMode]) {
    let pPd = predictedPd
    let nPd = nextPd

    let dx = pd.width
    let dy = pd.height
    let cbDx = ((dx + 1) / 2)
    let cbDy = ((dy + 1) / 2)

    let qtY2 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 2)
    let qtC2 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 2)
    let qtY1 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 1)
    let qtC1 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 1)
    let qtY0 = QuantizationTable(baseStep: Int(qtY.step), isChroma: false, layerIndex: 0)
    let qtC0 = QuantizationTable(baseStep: Int(qtC.step), isChroma: true, layerIndex: 0)

    var skipMap = await computeProfile2SkipMap(pd: pd, prevInput: prevInput, ltrInput: ltrInput, predictedPd: predictedPd, ltrRecon: nPd, gopPosition: gopPosition, ltrAge: ltrAge, skipThreshold: skipThreshold, reconThresholdScale: reconThresholdScale, staticCounters: &staticCounters)

    // The skip model prices both reference directions per block, which the
    // coding path does not keep; ask the search to export them when it runs.
    let dualSink: DualMVSink? = skipModel != nil ? DualMVSink() : nil
    let (mvs_original, sads, refDirs_original, _, nextSub2Res, nextSub1Res) = await computeBidirectionalMotionVectors(curr: pd, prev: pPd, next: nPd, prevMVs: prevMVs ?? MotionVectors.empty, pool: pool, roundOffset: roundOffset, gopPosition: gopPosition, skipMap: skipMap, cachedNextSub2: cachedNextSub2, cachedNextSub1: cachedNextSub1, dualOut: dualSink)
    var mvs = mvs_original
    var refDirs = refDirs_original
    for i in 0..<skipMap.count {
        switch skipMap[i] {
        case .skip_ltr:
            mvs.dx[i] = 0
            mvs.dy[i] = 0
            refDirs[i] = true
        case .skip_prev:
            mvs.dx[i] = 0
            mvs.dy[i] = 0
            refDirs[i] = false
        default: break
        }
    }

    // Learned skip decision. It runs here, before any consumer reads the map,
    // so coding (skip map / MVs / ref dirs / layer preparation / zero flags)
    // and reconstruction (MC subtract, L0 loop, deblock, skip copy) all
    // describe the same blocks. Its activity classes are the ones the AQ stage
    // would compute anyway, so they are handed forward instead of recomputed.
    var earlyActivity: [BlockActivityClass]? = nil
    if let model = skipModel, let dual = dualSink {
        let variances = computeBlockActivityMap(source: pd.y, width: dx, height: dy)
        let classes: [BlockActivityClass]
        if smooth == 1 {
            classes = classifyBlockActivityWithCoherence(
                varianceMap: variances,
                flatVarianceMax: EncoderTuning.shared.aqFlatVarianceMax,
                texturedVarianceMin: EncoderTuning.shared.aqTexturedVarianceMin,
                source: pd.y,
                width: dx,
                height: dy
            )
        } else {
            classes = classifyBlockActivity(
                varianceMap: variances,
                flatVarianceMax: EncoderTuning.shared.aqFlatVarianceMax,
                texturedVarianceMin: EncoderTuning.shared.aqTexturedVarianceMin
            )
        }
        earlyActivity = classes
        model.apply(
            skipMap: &skipMap,
            mvs: &mvs,
            refDirs: &refDirs,
            inputs: SkipDeciderFrameInputs(
                mvsPrev: dual.prevMVs,
                mvsLtr: dual.ltrMVs,
                meSadPrev: dual.prevSADs,
                meSadLtr: dual.ltrSADs,
                variance: variances,
                activityClass: classes,
                staticCounters: staticCounters,
                adjustedStep: adjustedStep,
                gopPosition: gopPosition,
                ltrAge: ltrAge
            ),
            source: pd,
            prevRef: pPd,
            ltrRef: nPd
        )
    }

    var mutPdY = pool.getInt16(count: pd.y.count)
    var mutPdCb = pool.getInt16(count: pd.cb.count)
    var mutPdCr = pool.getInt16(count: pd.cr.count)

    copyPlaneBuffers(y: pd.y, cb: pd.cb, cr: pd.cr, intoY: &mutPdY, cb: &mutPdCb, cr: &mutPdCr)

    let sMap = skipMap
    // Weighted prediction (#21): global luma offset of this frame against
    // the prediction reference, signaled in the frame header and applied as
    // P′ = P + offset on inter blocks at every prediction site (full
    // resolution, LL2 slot, L0 chain). The chroma offset is signaled for
    // symmetry but not yet estimated. Estimated concurrently with the
    // MC-subtract tasks; the value is first needed after tY resolves.
    async let tWp = { [pdY = pd.y, pPdY = pPd.y] () -> Int in
        estimateLumaOffset(source: pdY, reference: pPdY, width: dx, height: dy)
    }()
    // σ-normalized AQ: per-block source-luma activity classes select the
    // dead-zone variants during layer 2/1 luma quantization (SAD.swift,
    // Quant.swift). Computed concurrently with the MC-subtract tasks.
    async let tAq = { [pdY = pd.y, earlyActivity] () -> [BlockActivityClass] in
        if let a = earlyActivity { return a }
        let variances = computeBlockActivityMap(source: pdY, width: dx, height: dy)
        if smooth == 1 {
            return classifyBlockActivityWithCoherence(
                varianceMap: variances,
                flatVarianceMax: EncoderTuning.shared.aqFlatVarianceMax,
                texturedVarianceMin: EncoderTuning.shared.aqTexturedVarianceMin,
                source: pdY,
                width: dx,
                height: dy
            )
        }
        return classifyBlockActivity(
            varianceMap: variances,
            flatVarianceMax: EncoderTuning.shared.aqFlatVarianceMax,
            texturedVarianceMin: EncoderTuning.shared.aqTexturedVarianceMin
        )
    }()
    let mvsConst = mvs
    let refDirsConst = refDirs
    async let tY = { [mvsConst, refDirsConst, sMap] () -> [Int16] in
        var y = mutPdY
        subtractScaledBidirectionalMotionCompensationLumaWithSkipMap(plane: &y, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvsConst, refDirs: refDirsConst, skipMap: sMap, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
        return y
    }()
    async let tCb = { [mvsConst, refDirsConst, sMap] () -> [Int16] in
        var cb = mutPdCb
        subtractScaledBidirectionalMotionCompensationChromaWithSkipMap(plane: &cb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvsConst, refDirs: refDirsConst, skipMap: sMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        return cb
    }()
    async let tCr = { [mvsConst, refDirsConst, sMap] () -> [Int16] in
        var cr = mutPdCr
        subtractScaledBidirectionalMotionCompensationChromaWithSkipMap(plane: &cr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvsConst, refDirs: refDirsConst, skipMap: sMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        return cr
    }()

    var resY = await tY
    let resCb = await tCb
    let resCr = await tCr
    let wpLuma = await tWp

    if wpLuma != 0 {
        applyPredictionOffset32(plane: &resY, offset: -wpLuma, mvs: mvs, refDirs: refDirs, skipMap: sMap, width: dx, height: dy)
    }

    let activityMap = await tAq

    if smooth == 1 {
        if motionMaskingMinQStep <= adjustedStep {
            var tempY = pool.getInt16(count: resY.count)
            var smoothedY = pool.getInt16(count: resY.count)
            withUnsafePointers(resY, mut: &smoothedY, mut: &tempY) { srcPtr, dstPtr, tmpPtr in
                smoothResidualPlaneContinuous(
                    src: srcPtr, dst: dstPtr, temp: tmpPtr,
                    width: dx, height: dy, activityMap: activityMap, stride: dx,
                    mvs: mvs, skipMap: skipMap, framerate: framerate
                )
            }
            pool.putInt16(tempY)
            pool.putInt16(resY)
            resY = smoothedY
        }
    }

    let resPd = PlaneData420(width: dx, height: dy, y: resY, cb: resCb, cr: resCr)

    // Skip blocks bypass read/DWT/quant in the extracts (One-Pyramid §5) —
    // their residual is already zero, so the coded streams are unchanged.
    let skipBw = (dx + 31) / 32
    let skipBh = (dy + 31) / 32

    var (sub2, l2yBlocks, l2cbBlocks, l2crBlocks, releaseL2) = await preparePlaneLayer32WithSkipMapAndActivity(pd: resPd, pool: pool, qtY: qtY2, qtC: qtC2, skipMap: skipMap, skipMapWidth: skipBw, activity: activityMap)
    defer { releaseL2() }
    var (sub1, l1yBlocks, l1cbBlocks, l1crBlocks, releaseL1) = await preparePlaneLayer16WithSkipMapAndActivity(pd: sub2, pool: pool, qtY: qtY1, qtC: qtC1, skipMap: skipMap, skipMapWidth: skipBw, activity: activityMap)
    defer { releaseL1() }

    var effectiveMvtQ = 0
    if 0 < motionMaskingPx {
        effectiveMvtQ = (motionMaskingPx * 4 * 60) / max(1, framerate)
    }

    if shouldZeroCadence(cadence: l2Cadence, gopPosition: gopPosition) {
        zeroBlocksSubbands32(blocks: &l2yBlocks)
        zeroBlocksSubbands32(blocks: &l2cbBlocks)
        zeroBlocksSubbands32(blocks: &l2crBlocks)
    } else {
        if 0 < motionMaskingPx {
            if motionMaskingMinQStep <= adjustedStep {
                let blockCount = min(l2yBlocks.count, min(skipMap.count, min(activityMap.count, min(mvs.dx.count, mvs.dy.count))))
                for i in 0..<blockCount {
                    switch skipMap[i] {
                    case .inter:
                        if activityMap[i] != .textured {
                            let currDx = abs(Int(mvs.dx[i]))
                            let currDy = abs(Int(mvs.dy[i]))
                            let currMvMag = max(currDx, currDy)
                            if effectiveMvtQ <= currMvMag {
                                zeroBlockSubbands32(view: l2yBlocks[i])
                            }
                        }
                    default:
                        break
                    }
                }
            }
        }
    }

    if shouldZeroCadence(cadence: l1Cadence, gopPosition: gopPosition) {
        zeroBlocksSubbands16(blocks: &l1yBlocks)
        zeroBlocksSubbands16(blocks: &l1cbBlocks)
        zeroBlocksSubbands16(blocks: &l1crBlocks)
    }

    // L0 closed loop (One-Pyramid §4): Base8 codes r0 = LL2(source) −
    // MC_L0(L0_ref) instead of LL2(residual), so the quarter-resolution
    // reconstruction closes over the bitstream alone (bit-exact with the
    // decoder's layer0 chain). Requires an L0 reference from a preceding
    // I-frame; without one the legacy LL2(residual) semantics apply.
    var base8Input = sub1
    // The full-resolution prediction built for the LL2 slot, reused for the
    // layer2 reconstruction (fusePredictionPlane* is bit-identical to a
    // second MC apply pass).
    var fullPForRecon: PlaneData420? = nil
    if let l0Prev = l0State.prev {
        let tSrc = analyzeLL2(pd: pd)
        var r0 = Image16(width: tSrc.width, height: tSrc.height, y: tSrc.y, cb: tSrc.cb, cr: tSrc.cr)
        var pred0 = Image16(
            width: tSrc.width, height: tSrc.height,
            y: [Int16](repeating: 0, count: tSrc.y.count),
            cb: [Int16](repeating: 0, count: tSrc.cb.count),
            cr: [Int16](repeating: 0, count: tSrc.cr.count)
        )
        await applyL0MotionCompensation(img: &pred0, prevPd: l0Prev, ltrPd: l0State.ltr, mvs: mvs, refDirs: refDirs, skipMap: sMap, roundOffset: roundOffset)
        if wpLuma != 0 {
            applyPredictionOffset8(plane: &pred0.y, offset: wpLuma, mvs: mvs, refDirs: refDirs, skipMap: sMap, width: pred0.width, height: pred0.height)
        }
        subtractPlanes(&r0, PlaneData420(img16: pred0))
        clearL0SkipResidual(img: &r0, skipMap: sMap, fullDx: dx)
        base8Input = PlaneData420(img16: r0)
    }

    var smoothL0Flags: [Bool]? = nil
    if smooth == 1 {
        let effective8PxQ = (8 * 4 * 60) / max(1, framerate)
        let blockCount = min(skipMap.count, min(activityMap.count, min(mvs.dx.count, mvs.dy.count)))
        var flags = [Bool](repeating: false, count: blockCount)
        for i in 0..<blockCount {
            switch skipMap[i] {
            case .inter:
                if activityMap[i] != .textured {
                    let currDx = abs(Int(mvs.dx[i]))
                    let currDy = abs(Int(mvs.dy[i]))
                    let currMvMag = max(currDx, currDy)
                    if effective8PxQ <= currMvMag {
                        flags[i] = true
                    }
                }
            default:
                break
            }
        }
        smoothL0Flags = flags
    }

    var (base8YBlocks, base8CbBlocks, base8CrBlocks, releaseBaseBlocks) = await preparePlaneBase8WithSkipMapAndActivity(
        pd: base8Input, pool: pool, sads: sads,
        qtY: qtY0, qtC: qtC0,
        skipMap: skipMap, skipMapWidth: skipBw,
        activity: activityMap,
        smoothFlags: smoothL0Flags
    )
    defer { releaseBaseBlocks() }

    // L0 temporal thinning (experimental -l0cadence): zero the whole Base8
    // residual before flag computation so treeMap aggregation also sees the
    // all-zero trees. The L0 closed loop reconstructs MC-only prediction on
    // both sides (serializePlaneBase8PFrameWithSkipMap codes zeros; dequant
    // maps them back to zero), keeping the chain bit-exact.
    if shouldZeroCadence(cadence: l0Cadence, gopPosition: gopPosition) {
        zeroBlocksSubbandsBase8(blocks: &base8YBlocks)
        zeroBlocksSubbandsBase8(blocks: &base8CbBlocks)
        zeroBlocksSubbandsBase8(blocks: &base8CrBlocks)
    }
    let l1dx = sub2.width
    let l1dy = sub2.height
    let l1cbDx = ((l1dx + 1) / 2)
    let l1cbDy = ((l1dy + 1) / 2)

    let l2ySkip = lumaSkipFlags(skipMap: skipMap, mapWidth: skipBw, rowCount: (dy + 31) / 32, colCount: (dx + 31) / 32)
    let l2cSkip = chromaSkipFlags(skipMap: skipMap, mapWidth: skipBw, rowCount: (cbDy + 31) / 32, colCount: (cbDx + 31) / 32)
    let l1ySkip = lumaSkipFlags(skipMap: skipMap, mapWidth: skipBw, rowCount: (l1dy + 15) / 16, colCount: (l1dx + 15) / 16)
    let l1cSkip = chromaSkipFlags(skipMap: skipMap, mapWidth: skipBw, rowCount: (l1cbDy + 15) / 16, colCount: (l1cbDx + 15) / 16)

    // Compute zero flags for each plane and layer to identify zero trees (treez)
    let safeThresholdY2 = min(3, min(zeroThreshold, max(0, Int(qtY2.step) / 64)))
    let colCountY2 = (dx + 31) / 32
    let rowCountY2 = (dy + 31) / 32
    let l2yZeros = computeZeroFlags32(blocks: &l2yBlocks, zeroThreshold: safeThresholdY2, colCount: colCountY2, rowCount: rowCountY2, isSkip: l2ySkip)

    let safeThresholdC2 = min(8, min(zeroThreshold, max(0, Int(qtC2.step) / 64)))
    let colCountC2 = (cbDx + 31) / 32
    let rowCountC2 = (cbDy + 31) / 32
    let l2cbZeros = computeZeroFlags32(blocks: &l2cbBlocks, zeroThreshold: safeThresholdC2, colCount: colCountC2, rowCount: rowCountC2, isSkip: l2cSkip)
    let l2crZeros = computeZeroFlags32(blocks: &l2crBlocks, zeroThreshold: safeThresholdC2, colCount: colCountC2, rowCount: rowCountC2, isSkip: l2cSkip)

    let safeThresholdY1 = min(2, min(zeroThreshold, max(0, Int(qtY1.step) / 64)))
    let colCountY1 = (l1dx + 15) / 16
    let rowCountY1 = (l1dy + 15) / 16
    let l1yZeros = computeZeroFlags16(blocks: &l1yBlocks, zeroThreshold: safeThresholdY1, colCount: colCountY1, rowCount: rowCountY1, isSkip: l1ySkip)

    let safeThresholdC1 = min(8, min(zeroThreshold, max(0, Int(qtC1.step) / 64)))
    let colCountC1 = (l1cbDx + 15) / 16
    let rowCountC1 = (l1cbDy + 15) / 16
    let l1cbZeros = computeZeroFlags16(blocks: &l1cbBlocks, zeroThreshold: safeThresholdC1, colCount: colCountC1, rowCount: rowCountC1, isSkip: l1cSkip)
    let l1crZeros = computeZeroFlags16(blocks: &l1crBlocks, zeroThreshold: safeThresholdC1, colCount: colCountC1, rowCount: rowCountC1, isSkip: l1cSkip)

    let safeThresholdY0 = min(1, min(zeroThreshold, max(0, Int(qtY0.step) / 64)))
    let l0yZeros = computeZeroFlagsBase8(blocks: base8YBlocks, zeroThreshold: safeThresholdY0, isSkip: l2ySkip)

    let safeThresholdC0 = min(8, max(0, (zeroThreshold / 8) - (Int(qtC0.step) / 32)))
    let l0cbZeros = computeZeroFlagsBase8(blocks: base8CbBlocks, zeroThreshold: safeThresholdC0, isSkip: l2cSkip)
    let l0crZeros = computeZeroFlagsBase8(blocks: base8CrBlocks, zeroThreshold: safeThresholdC0, isSkip: l2cSkip)

    var isTreezY = [Bool](repeating: false, count: l2yBlocks.count)
    for i in 0..<l2yBlocks.count {
        if l2ySkip[i] != true {
            if l0yZeros[i] {
                if l1yZeros[i] {
                    if l2yZeros[i] {
                        isTreezY[i] = true
                    }
                }
            }
        }
    }
    var isTreezCb = [Bool](repeating: false, count: l2cbBlocks.count)
    for i in 0..<l2cbBlocks.count {
        if l2cSkip[i] != true {
            if l0cbZeros[i] {
                if l1cbZeros[i] {
                    if l2cbZeros[i] {
                        isTreezCb[i] = true
                    }
                }
            }
        }
    }

    var isTreezCr = [Bool](repeating: false, count: l2crBlocks.count)
    for i in 0..<l2crBlocks.count {
        if l2cSkip[i] != true {
            if l0crZeros[i] {
                if l1crZeros[i] {
                    if l2crZeros[i] {
                        isTreezCr[i] = true
                    }
                }
            }
        }
    }

    let treeMapBuf = encodeTreeMapContextProfile2(
        isTreezY: isTreezY, ySkip: l2ySkip, colsY: (dx + 31) / 32,
        isTreezCb: isTreezCb, cbSkip: l2cSkip,
        isTreezCr: isTreezCr, crSkip: l2cSkip, colsC: (cbDx + 31) / 32
    )

    let (layer0, baseRecon, releaseBaseRecon, _, _, _, hasRANSContext) = serializePlaneBase8PFrameWithSkipMap(
        pd: base8Input, pool: pool,
        qtY: qtY0, qtC: qtC0, zeroThreshold: zeroThreshold,
        base8YBlocks: &base8YBlocks, base8CbBlocks: &base8CbBlocks, base8CrBlocks: &base8CrBlocks,
        skipMap: skipMap, skipMapWidth: skipBw,
        isTreezY: isTreezY, isTreezCb: isTreezCb, isTreezCr: isTreezCr,
        histories: entropyHistories?.streams[0],
        updateHistory: updateL0Prev,
        ransContextWorkspace: ransContextWorkspace
    )
    defer { releaseBaseRecon() }

    var baseImg = Image16(width: baseRecon.width, height: baseRecon.height, y: baseRecon.y, cb: baseRecon.cb, cr: baseRecon.cr)

    if let l0Prev = l0State.prev {
        // L0 reconstruction — the decoder's exact layer0 pipeline:
        // deq(r0) + MC_L0, clamp, deblock, skip copy.
        let baseCopy = copyImageToPlaneData420(baseImg)
        var l0Cur = Image16(width: baseRecon.width, height: baseRecon.height, y: baseCopy.y, cb: baseCopy.cb, cr: baseCopy.cr)
        await applyL0MotionCompensation(img: &l0Cur, prevPd: l0Prev, ltrPd: l0State.ltr, mvs: mvs, refDirs: refDirs, skipMap: sMap, roundOffset: roundOffset)
        if wpLuma != 0 {
            applyPredictionOffset8(plane: &l0Cur.y, offset: wpLuma, mvs: mvs, refDirs: refDirs, skipMap: sMap, width: l0Cur.width, height: l0Cur.height)
        }
        finishL0Reconstruction(img: &l0Cur, qtYStepQ4: Int(qtY0.step), qtCStepQ4: Int(qtC0.step))
        applyL0SkipCopy(img: &l0Cur, prevPd: l0Prev, ltrPd: l0State.ltr, skipMap: sMap, fullDx: dx)
        let newRef = PlaneData420(img16: l0Cur)

        // The full loop's LL2 coefficient slot is L0_recon − LL2(P), with P
        // built by the identical MC call sequence the decoder uses.
        var fullP = await buildFullResolutionPrediction(dx: dx, dy: dy, prevPd: pPd, ltrPd: nPd, mvs: mvs, refDirs: refDirs, skipMap: sMap, roundOffset: roundOffset)
        if wpLuma != 0 {
            // Baking the offset into P makes the LL2 slot (analyzeLL2) and
            // the layer2 reconstruction (fusePredictionPlane) both see P′.
            applyPredictionOffset32(plane: &fullP.y, offset: wpLuma, mvs: mvs, refDirs: refDirs, skipMap: sMap, width: dx, height: dy)
        }
        fullPForRecon = fullP
        let tP = analyzeLL2(pd: fullP)
        var slot = Image16(width: newRef.width, height: newRef.height, y: newRef.y, cb: newRef.cb, cr: newRef.cr)
        subtractPlanes(&slot, tP)
        baseImg = slot

        if updateL0Prev {
            l0State.prev = newRef
        }
    }

    let (layer1, _, _, _) = encodeLayer16PayloadWithSkipMap(
        dx: sub2.width, dy: sub2.height,
        qtY: qtY1, qtC: qtC1, zeroThreshold: zeroThreshold,
        yBlocks: &l1yBlocks, cbBlocks: &l1cbBlocks, crBlocks: &l1crBlocks,
        parentYBlocks: parentFreeParents8(count: base8YBlocks.count),
        parentCbBlocks: parentFreeParents8(count: base8CbBlocks.count),
        parentCrBlocks: parentFreeParents8(count: base8CbBlocks.count),
        ySkip: l1ySkip, cSkip: l1cSkip,
        isTreezY: isTreezY, isTreezCb: isTreezCb, isTreezCr: isTreezCr,
        histories: entropyHistories?.streams[1],
        selectModel: unifiedSelectModelParentFree,
        updateHistory: updateL0Prev
    )

    let (mutReconL1Y, r1Y) = reconstructPlaneLayer16Y(blocks: l1yBlocks, prevImg: baseImg, width: l1dx, height: l1dy, qt: qtY1, pool: pool)
    let (mutReconL1Cb, r1Cb) = reconstructPlaneLayer16Cb(blocks: l1cbBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    let (mutReconL1Cr, r1Cr) = reconstructPlaneLayer16Cr(blocks: l1crBlocks, prevImg: baseImg, width: l1cbDx, height: l1cbDy, qt: qtC1, pool: pool)
    defer { r1Y(); r1Cb(); r1Cr() }

    let l1Img = Image16(width: l1dx, height: l1dy, y: mutReconL1Y, cb: mutReconL1Cb, cr: mutReconL1Cr)
    let (layer2, _, _, _) = encodeLayer32PayloadWithSkipMap(
        dx: pd.width, dy: pd.height,
        qtY: qtY2, qtC: qtC2, zeroThreshold: zeroThreshold,
        yBlocks: &l2yBlocks, cbBlocks: &l2cbBlocks, crBlocks: &l2crBlocks,
        parentYBlocks: parentFreeParents16(count: l1yBlocks.count),
        parentCbBlocks: parentFreeParents16(count: l1cbBlocks.count),
        parentCrBlocks: parentFreeParents16(count: l1cbBlocks.count),
        ySkip: l2ySkip, cSkip: l2cSkip,
        isTreezY: isTreezY, isTreezCb: isTreezCb, isTreezCr: isTreezCr,
        histories: entropyHistories?.streams[2],
        selectModel: unifiedSelectModelParentFree,
        updateHistory: updateL0Prev
    )

    if let dumper = dumpWriter {
        dumper.stash("L0Y", blocks: base8YBlocks, planeW: sub1.width, planeH: sub1.height, blockSize: 8, includeLL: true)
        dumper.stash("L0Cb", blocks: base8CbBlocks, planeW: (sub1.width + 1) / 2, planeH: (sub1.height + 1) / 2, blockSize: 8, includeLL: true)
        dumper.stash("L0Cr", blocks: base8CrBlocks, planeW: (sub1.width + 1) / 2, planeH: (sub1.height + 1) / 2, blockSize: 8, includeLL: true)

        dumper.stash("L1Y", blocks: l1yBlocks, planeW: sub2.width, planeH: sub2.height, blockSize: 16, includeLL: false)
        dumper.stash("L1Cb", blocks: l1cbBlocks, planeW: l1cbDx, planeH: l1cbDy, blockSize: 16, includeLL: false)
        dumper.stash("L1Cr", blocks: l1crBlocks, planeW: l1cbDx, planeH: l1cbDy, blockSize: 16, includeLL: false)

        dumper.stash("L2Y", blocks: l2yBlocks, planeW: dx, planeH: dy, blockSize: 32, includeLL: false)
        dumper.stash("L2Cb", blocks: l2cbBlocks, planeW: cbDx, planeH: cbDy, blockSize: 32, includeLL: false)
        dumper.stash("L2Cr", blocks: l2crBlocks, planeW: cbDx, planeH: cbDy, blockSize: 32, includeLL: false)

        dumper.finalizePFrame(
            gopPosition: gopPosition, width: dx, height: dy, predictedPd: pPd,
            l1yBlocks: l1yBlocks, l1cbBlocks: l1cbBlocks, l1crBlocks: l1crBlocks,
            b8yBlocks: base8YBlocks, b8cbBlocks: base8CbBlocks, b8crBlocks: base8CrBlocks,
            sub2W: sub2.width, sub2H: sub2.height, sub1W: sub1.width, sub1H: sub1.height,
            qtY2: qtY2, qtC2: qtC2, qtY1: qtY1, qtC1: qtC1, qtY0: qtY0, qtC0: qtC0,
            layer0Bytes: layer0.count, layer1Bytes: layer1.count, layer2Bytes: layer2.count
        )
    }

    // skipMap must be passed here: the decoder's layer2 reconstruction skips
    // skip blocks entirely (they stay zero until the final skip copy), and
    // the deblocking filter runs BEFORE that copy, reading pixels from skip
    // regions at mixed inter|skip block edges. Reconstructing those regions
    // on the encoder only (as before) diverges the reconstructions by ±1 at
    // such edges — harmless historically, but the L0 closed loop couples the
    // reconstruction into the entropy contexts where any divergence desyncs
    // the backward-adaptive tables.
    let (reconL2Y, r2Y) = reconstructPlaneLayer32YWithSkipMap(blocks: l2yBlocks, prevImg: l1Img, width: dx, height: dy, qt: qtY2, pool: pool, skipMap: sMap)
    var mutReconL2Y = reconL2Y
    let (reconL2Cb, r2Cb) = reconstructPlaneLayer32CbWithSkipMap(blocks: l2cbBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool, skipMap: sMap, skipBw: skipBw, skipBh: skipBh)
    var mutReconL2Cb = reconL2Cb
    let (reconL2Cr, r2Cr) = reconstructPlaneLayer32CrWithSkipMap(blocks: l2crBlocks, prevImg: l1Img, width: cbDx, height: cbDy, qt: qtC2, pool: pool, skipMap: sMap, skipBw: skipBw, skipBh: skipBh)
    var mutReconL2Cr = reconL2Cr

    // With the L0 loop active, the full-resolution prediction plane was
    // already built by the identical MC call sequence — fuse it in (add for
    // inter blocks, replace for skip blocks) instead of a second MC pass.
    if let fullP = fullPForRecon {
        fusePredictionPlane32(recon: &mutReconL2Y, p: fullP.y, skipMap: sMap, width: dx, height: dy)
        fusePredictionPlane16(recon: &mutReconL2Cb, p: fullP.cb, skipMap: sMap, width: cbDx, height: cbDy)
        fusePredictionPlane16(recon: &mutReconL2Cr, p: fullP.cr, skipMap: sMap, width: cbDx, height: cbDy)
    } else {
        await applyScaledBidirectionalMotionCompensationLumaWithSkipMap(plane: &mutReconL2Y, prevPlane: pPd.y, nextPlane: nPd.y, mvs: mvs, refDirs: refDirs, skipMap: sMap, width: dx, height: dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
        await applyScaledBidirectionalMotionCompensationChromaWithSkipMap(plane: &mutReconL2Cb, prevPlane: pPd.cb, nextPlane: nPd.cb, mvs: mvs, refDirs: refDirs, skipMap: sMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        await applyScaledBidirectionalMotionCompensationChromaWithSkipMap(plane: &mutReconL2Cr, prevPlane: pPd.cr, nextPlane: nPd.cr, mvs: mvs, refDirs: refDirs, skipMap: sMap, width: cbDx, height: cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        if wpLuma != 0 {
            applyPredictionOffset32(plane: &mutReconL2Y, offset: wpLuma, mvs: mvs, refDirs: refDirs, skipMap: sMap, width: dx, height: dy)
        }
    }

    // #36: context-conditioned adaptive rANS replaces the RLE + per-frame
    // tables. The state is per-GOP and reset at every coded I frame.
    let skipMapData = encodeSkipMapContext(map: skipMap, cols: skipBw, state: syntaxContext)
    let mvData = encodeMVs(
        mvs: mvs,
        skipMap: skipMap,
        cols: deriveMVColumns(width: dx),
        profile: 0x02,
        prevMVs: prevMVs,
        syntax: syntaxContext,
        updateHistory: updateL0Prev
    )

    let refDirBuf = encodeRefDirsContextProfile2(refDirs: refDirs, skipMap: skipMap)

    // Must match the decoder's deblock invocation exactly (mvs + skipMap
    // variants): the encoder previously filtered its reconstruction with the
    // plain/partial variants while the decoder used the intra-boundary
    // enhanced + skip-gated ones, so the two reconstructions diverged and the
    // P-chain accumulated the difference into chroma-heavy smears in
    // intra-dense motion regions (grew with GOP position, immune to bitrate).
    applyDeblockingFilter32WithMVsAndSkipMap(plane: &mutReconL2Y, width: dx, height: dy, qStep: (Int(qtY2.step) + 8) >> 4, mvs: mvs, skipMap: sMap)
    applyDeblockingFilterChroma16WithMVsAndSkipMap(plane: &mutReconL2Cb, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvs, skipMap: sMap)
    applyDeblockingFilterChroma16WithMVsAndSkipMap(plane: &mutReconL2Cr, width: cbDx, height: cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: mvs, skipMap: sMap)

    applyProfile2SkipCopy(skipMap: skipMap, ltrPd: nPd, prevPd: pPd, y: &mutReconL2Y, cb: &mutReconL2Cb, cr: &mutReconL2Cr, dx: dx, dy: dy)

    let releaseY: @Sendable () -> Void = { r2Y() }
    let releaseCb: @Sendable () -> Void = { r2Cb() }
    let releaseCr: @Sendable () -> Void = { r2Cr() }

    let reconstructed = PlaneData420(width: dx, height: dy, y: mutReconL2Y, cb: mutReconL2Cb, cr: mutReconL2Cr)

    debugLog({
        return "  [Summary] Layer0=\(layer0.count) Layer1=\(layer1.count) Layer2=\(layer2.count) total=\(layer0.count + layer1.count + layer2.count) bytes"
    }())

    var out: [UInt8] = []
    let frameHeader = VEVCFrameHeader(frameType: .pFrame, hasRefDir: true, hasRANSContext: hasRANSContext, skipMapSize: skipMapData.count, mvsSize: mvData.count, refDirSize: refDirBuf.count, treeMapSize: treeMapBuf.count, lumaOffset: wpLuma, chromaOffset: 0, layer0Size: layer0.count, layer1Size: layer1.count, layer2Size: layer2.count)
    out.append(contentsOf: frameHeader.serialize(profile: 0x02))
    out.append(contentsOf: skipMapData)
    out.append(contentsOf: mvData)
    out.append(contentsOf: refDirBuf)
    out.append(contentsOf: treeMapBuf)
    out.append(contentsOf: layer0)
    out.append(contentsOf: layer1)
    out.append(contentsOf: layer2)

    return (out, reconstructed, mvs, sads, { releaseY(); releaseCb(); releaseCr() }, nextSub2Res, nextSub1Res, skipMap)
}

/// Final skip-block copy into the full-resolution reconstruction from the
/// layer-matched references (skip_ltr ← LTR recon, skip_prev ← previous
/// recon), mirroring the decoder's copy at layer2.
@inline(__always)
private func applyProfile2SkipCopy(skipMap: [BlockMode], ltrPd: PlaneData420, prevPd: PlaneData420, y: inout [Int16], cb: inout [Int16], cr: inout [Int16], dx: Int, dy: Int) {
    let bw = (dx + 31) / 32
    let targetCbDx = (dx + 1) / 2
    let targetCbDy = (dy + 1) / 2
    let targetBSize = 32
    let tCbSize = 16

    withUnsafePointers(
        ltrPd.y, ltrPd.cb, ltrPd.cr,
        prevPd.y, prevPd.cb, prevPd.cr,
        mut: &y, mut: &cb, mut: &cr
    ) { ltrYPtr, ltrCbPtr, ltrCrPtr, prevYPtr, prevCbPtr, prevCrPtr, currYPtr, currCbPtr, currCrPtr in
        for i in 0..<skipMap.count {
            let mode = skipMap[i]
            if mode != .inter {
                let bx = (i % bw) * 32
                let by = (i / bw) * 32

                if bx + targetBSize <= dx && by + targetBSize <= dy {
                    switch mode {
                    case .skip_ltr:
                        copyBlock32Pointer(from: ltrYPtr, to: currYPtr, bx: bx, by: by, stride: dx)
                        copyBlock16Pointer(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                        copyBlock16Pointer(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                    case .skip_prev:
                        copyBlock32Pointer(from: prevYPtr, to: currYPtr, bx: bx, by: by, stride: dx)
                        copyBlock16Pointer(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                        copyBlock16Pointer(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                    default: break
                    }
                } else {
                    switch mode {
                    case .skip_ltr:
                        copyBlockSafe(from: ltrYPtr, to: currYPtr, bx: bx, by: by, width: dx, height: dy, blockSize: targetBSize)
                        copyBlockSafe(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                        copyBlockSafe(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                    case .skip_prev:
                        copyBlockSafe(from: prevYPtr, to: currYPtr, bx: bx, by: by, width: dx, height: dy, blockSize: targetBSize)
                        copyBlockSafe(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                        copyBlockSafe(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                    default: break
                    }
                }
            }
        }
    }
}

@inline(__always)
func encodeRefDirsProfile1(refDirs: [Bool]) -> [UInt8] {
    let refDirByteCount = (refDirs.count + 7) / 8
    var refDirBuf = [UInt8](repeating: 0, count: refDirByteCount)
    for i in refDirs.indices {
        if refDirs[i] {
            refDirBuf[i / 8] |= UInt8(1 << (i % 8))
        }
    }
    return refDirBuf
}
