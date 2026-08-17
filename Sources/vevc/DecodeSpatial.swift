// MARK: - Decode Spatial

/// Shared profile-0x01 frame prelude: header, motion vectors, reference
/// directions, and the layer0 payload slice — everything before the layer
/// pipelines diverge by maxLayer.
private struct Profile1Prelude {
    let frameHeader: VEVCFrameHeader
    let mvs: MotionVectors?
    let refDirs: [Bool]?
    let layer0Data: [UInt8]
    /// Byte offset of the layer1 payload within the frame chunk.
    let layer1Offset: Int
}

@inline(__always)
private func parseProfile1Frame(r: [UInt8], dx: Int, dy: Int, nextPd: PlaneData420?) throws -> Profile1Prelude {
    var offset = 0
    let frameHeader = try VEVCFrameHeader.deserialize(from: r, offset: &offset, profile: 0x01)
    if frameHeader.isCopyFrame {
        throw DecodeError.insufficientDataContext("decodeSpatialLayers passed copy frame")
    }

    var mvs: MotionVectors? = nil
    var refDirs: [Bool]? = nil

    let mvsCount = deriveMVCount(width: dx, height: dy)

    if frameHeader.isIFrame != true && 0 < frameHeader.mvsSize {
        guard (offset + frameHeader.mvsSize) <= r.count else { throw DecodeError.insufficientData }
        mvs = try decodeMVs(data: Array(r[offset..<(offset + frameHeader.mvsSize)]), count: mvsCount, skipMap: nil, profile: 0x01)
        offset += frameHeader.mvsSize
    }

    // Direction flag only exists for bidirectional prediction frames.
    // Indicates whether each block uses forward (prev) or backward (next) reference.
    if frameHeader.hasRefDir {
        let refDirByteCount = frameHeader.refDirSize
        guard (offset + refDirByteCount) <= r.count else { throw DecodeError.insufficientData }
        let refDirBuf = Array(r[offset..<(offset + refDirByteCount)])
        offset += refDirByteCount

        if nextPd != nil {
            var dirs = [Bool]()
            dirs.reserveCapacity(mvsCount)
            for i in 0..<mvsCount {
                let byteIdx = i / 8
                let bitIdx = i % 8
                let isBackward = (byteIdx < refDirBuf.count) && ((refDirBuf[byteIdx] & UInt8(1 << bitIdx)) != 0)
                dirs.append(isBackward)
            }
            refDirs = dirs
        }
    }

    guard (offset + frameHeader.layer0Size) <= r.count else { throw DecodeError.insufficientData }
    let layer0Data = Array(r[offset..<(offset + frameHeader.layer0Size)])
    offset += frameHeader.layer0Size

    return Profile1Prelude(frameHeader: frameHeader, mvs: mvs, refDirs: refDirs, layer0Data: layer0Data, layer1Offset: offset)
}

/// Layer2 data absent: MC + clamp + deblock at layer1 resolution
/// (16x16 blocks, mvShift=1). No-op on I-frames (no MVs / no prediction).
@inline(__always)
private func finishProfile1AtLayer1(current: inout Image16, l1dx: Int, l1dy: Int, mvs: MotionVectors?, refDirs: [Bool]?, predictedPd: PlaneData420?, nextPd: PlaneData420?, roundOffset: Int, qtYStep: Int, qtCStep: Int) async {
    guard let tMVs = mvs, let tPrev = predictedPd else { return }
    let cbDx1 = (l1dx + 1) / 2
    let cbDy1 = (l1dy + 1) / 2
    if let tNext = nextPd, let dirs = refDirs {
        await applyScaledBidirectionalMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMVs, refDirs: dirs, skipMap: nil, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
        await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMVs, refDirs: dirs, skipMap: nil, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
        await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMVs, refDirs: dirs, skipMap: nil, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
    } else {
        await applyScaledMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, mvs: tMVs, skipMap: nil, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
        await applyScaledMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, mvs: tMVs, skipMap: nil, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
        await applyScaledMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, mvs: tMVs, skipMap: nil, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
    }

    clampPlane(plane: &current.y)
    clampPlane(plane: &current.cb)
    clampPlane(plane: &current.cr)

    applyDeblockingFilterN(plane: &current.y, width: l1dx, height: l1dy, qStep: qtYStep, blockSize: 16)
    let cStep = min(qtCStep * 2, 255)
    applyDeblockingFilterN(plane: &current.cb, width: cbDx1, height: cbDy1, qStep: cStep, blockSize: 8)
    applyDeblockingFilterN(plane: &current.cr, width: cbDx1, height: cbDy1, qStep: cStep, blockSize: 8)
}

/// Upper layers absent: MC + clamp + deblock at layer0 resolution
/// (8x8 blocks, mvShift=2 for luma, mvShift=1 for chroma). No-op on
/// I-frames (no MVs / no prediction).
@inline(__always)
private func finishProfile1AtLayer0(current: inout Image16, l0dx: Int, l0dy: Int, mvs: MotionVectors?, refDirs: [Bool]?, predictedPd: PlaneData420?, nextPd: PlaneData420?, roundOffset: Int, qtYStep: Int, qtCStep: Int) async {
    guard let tMVs = mvs, let tPrev = predictedPd else { return }
    let cbDx0 = (l0dx + 1) / 2
    let cbDy0 = (l0dy + 1) / 2
    if let tNext = nextPd, let dirs = refDirs {
        await applyScaledBidirectionalMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMVs, refDirs: dirs, skipMap: nil, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
        await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMVs, refDirs: dirs, skipMap: nil, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
        await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMVs, refDirs: dirs, skipMap: nil, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
    } else {
        await applyScaledMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, mvs: tMVs, skipMap: nil, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
        await applyScaledMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, mvs: tMVs, skipMap: nil, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
        await applyScaledMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, mvs: tMVs, skipMap: nil, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
    }

    clampPlane(plane: &current.y)
    clampPlane(plane: &current.cb)
    clampPlane(plane: &current.cr)

    applyDeblockingFilterN(plane: &current.y, width: l0dx, height: l0dy, qStep: qtYStep, blockSize: 8)
    let cStep = min(qtCStep * 2, 255)
    applyDeblockingFilterN(plane: &current.cb, width: cbDx0, height: cbDy0, qStep: cStep, blockSize: 4)
    applyDeblockingFilterN(plane: &current.cr, width: cbDx0, height: cbDy0, qStep: cStep, blockSize: 4)
}

/// Frame decode, profile 0x01, maxLayer == 2 with full payloads — the
/// straight-line production pipeline (parent-conditioned entropy contexts,
/// so the three layers decode sequentially). Streams whose upper layers
/// were stripped by the splitter (layer sizes 0) delegate to the generic
/// decodeSpatialLayers.
@inline(__always)
func decodeSpatialLayersFull(r: [UInt8], pool: BlockViewPool, dx: Int, dy: Int, predictedPd: PlaneData420?, nextPd: PlaneData420?, roundOffset: Int, entropyHistories: FrameEntropyHistories?) async throws -> Image16 {
    let p = try parseProfile1Frame(r: r, dx: dx, dy: dy, nextPd: nextPd)
    guard 0 < p.frameHeader.layer1Size, 0 < p.frameHeader.layer2Size else {
        return try await decodeSpatialLayers(
            r: r, pool: pool, maxLayer: 2, dx: dx, dy: dy,
            predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffset, entropyHistories: entropyHistories
        )
    }

    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    // Backward-adaptive tables apply to profile 0x02 only.
    let histories: FrameEntropyHistories? = p.frameHeader.isIFrame ? nil : entropyHistories

    let (baseImg, base8YBlocks, base8CbBlocks, base8CrBlocks, _, _) = try await decodeBase8(r: p.layer0Data, pool: pool, layer: 0, dx: l0dx, dy: l0dy, isIFrame: p.frameHeader.isIFrame, histories: histories?.streams[0])

    var offset = p.layer1Offset
    guard (offset + p.frameHeader.layer1Size) <= r.count else { throw DecodeError.insufficientData }
    let layer1Data = Array(r[offset..<(offset + p.frameHeader.layer1Size)])
    offset += p.frameHeader.layer1Size

    let (l16Img, l16YBlocks, l16CbBlocks, l16CrBlocks) = try await decodeLayer16(r: layer1Data, pool: pool, layer: 1, dx: l1dx, dy: l1dy, prev: baseImg, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks, histories: histories?.streams[1])

    pool.putBlockViewArray64(base8YBlocks)
    pool.putBlockViewArray64(base8CbBlocks)
    pool.putBlockViewArray64(base8CrBlocks)

    guard (offset + p.frameHeader.layer2Size) <= r.count else { throw DecodeError.insufficientData }
    let layer2Data = Array(r[offset..<(offset + p.frameHeader.layer2Size)])

    let result = try await decodeLayer32(r: layer2Data, pool: pool, layer: 2, dx: dx, dy: dy, prev: l16Img, parentYBlocks: l16YBlocks, parentCbBlocks: l16CbBlocks, parentCrBlocks: l16CrBlocks, predictedPd: predictedPd, nextPd: nextPd, mvs: p.mvs, refDirs: p.refDirs, roundOffset: roundOffset, skipMap: nil, histories: histories?.streams[2])

    pool.putBlockViewArray256(l16YBlocks)
    pool.putBlockViewArray256(l16CbBlocks)
    pool.putBlockViewArray256(l16CrBlocks)

    return result
}

/// Frame decode, profile 0x01, maxLayer == 1: Base8 + Layer1, MC at half
/// resolution. A stripped layer1 (size 0) delegates to the generic
/// decodeSpatialLayers.
@inline(__always)
func decodeSpatialLayersWithLayer1(r: [UInt8], pool: BlockViewPool, dx: Int, dy: Int, predictedPd: PlaneData420?, nextPd: PlaneData420?, roundOffset: Int, entropyHistories: FrameEntropyHistories?) async throws -> Image16 {
    let p = try parseProfile1Frame(r: r, dx: dx, dy: dy, nextPd: nextPd)
    guard 0 < p.frameHeader.layer1Size else {
        return try await decodeSpatialLayers(
            r: r, pool: pool, maxLayer: 1, dx: dx, dy: dy,
            predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffset, entropyHistories: entropyHistories
        )
    }

    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    let histories: FrameEntropyHistories? = p.frameHeader.isIFrame ? nil : entropyHistories

    let (baseImg, base8YBlocks, base8CbBlocks, base8CrBlocks, qtYStep, qtCStep) = try await decodeBase8(r: p.layer0Data, pool: pool, layer: 0, dx: l0dx, dy: l0dy, isIFrame: p.frameHeader.isIFrame, histories: histories?.streams[0])

    let offset = p.layer1Offset
    guard (offset + p.frameHeader.layer1Size) <= r.count else { throw DecodeError.insufficientData }
    let layer1Data = Array(r[offset..<(offset + p.frameHeader.layer1Size)])

    let (l16Img, l16YBlocks, l16CbBlocks, l16CrBlocks) = try await decodeLayer16(r: layer1Data, pool: pool, layer: 1, dx: l1dx, dy: l1dy, prev: baseImg, parentYBlocks: base8YBlocks, parentCbBlocks: base8CbBlocks, parentCrBlocks: base8CrBlocks, histories: histories?.streams[1])

    pool.putBlockViewArray64(base8YBlocks)
    pool.putBlockViewArray64(base8CbBlocks)
    pool.putBlockViewArray64(base8CrBlocks)

    var current = l16Img
    await finishProfile1AtLayer1(current: &current, l1dx: l1dx, l1dy: l1dy, mvs: p.mvs, refDirs: p.refDirs, predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffset, qtYStep: qtYStep, qtCStep: qtCStep)

    pool.putBlockViewArray256(l16YBlocks)
    pool.putBlockViewArray256(l16CbBlocks)
    pool.putBlockViewArray256(l16CrBlocks)

    return current
}

/// Frame decode, profile 0x01, maxLayer == 0: Base8 + MC at quarter
/// resolution. Upper-layer payloads are never read, so no stripped-stream
/// delegation is needed.
@inline(__always)
func decodeSpatialLayersBase8Only(r: [UInt8], pool: BlockViewPool, dx: Int, dy: Int, predictedPd: PlaneData420?, nextPd: PlaneData420?, roundOffset: Int, entropyHistories: FrameEntropyHistories?) async throws -> Image16 {
    let p = try parseProfile1Frame(r: r, dx: dx, dy: dy, nextPd: nextPd)

    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    let histories: FrameEntropyHistories? = p.frameHeader.isIFrame ? nil : entropyHistories

    let (baseImg, base8YBlocks, base8CbBlocks, base8CrBlocks, qtYStep, qtCStep) = try await decodeBase8(r: p.layer0Data, pool: pool, layer: 0, dx: l0dx, dy: l0dy, isIFrame: p.frameHeader.isIFrame, histories: histories?.streams[0])

    pool.putBlockViewArray64(base8YBlocks)
    pool.putBlockViewArray64(base8CbBlocks)
    pool.putBlockViewArray64(base8CrBlocks)

    var current = baseImg
    await finishProfile1AtLayer0(current: &current, l0dx: l0dx, l0dy: l0dy, mvs: p.mvs, refDirs: p.refDirs, predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffset, qtYStep: qtYStep, qtCStep: qtCStep)

    return current
}

/// Frame decode, profile 0x01, generic pipeline: handles streams whose
/// upper layers were stripped by the splitter (layer sizes 0), where the
/// effective top layer is only known from the frame header. Production
/// decoding uses the straight-line variants above, which delegate here
/// only in those stripped cases.
@inline(__always)
func decodeSpatialLayers(r: [UInt8], pool: BlockViewPool, maxLayer: Int, dx: Int, dy: Int, predictedPd: PlaneData420?, nextPd: PlaneData420?, roundOffset: Int, entropyHistories: FrameEntropyHistories?) async throws -> Image16 {
    // Compute per-layer dimensions matching encoder DWT subband sizes:
    // Layer2 (32x32): original size
    // Layer1 (16x16): DWT LL subband of Layer2 = (dx+1)/2 × (dy+1)/2
    // Layer0 (Base8): DWT LL subband of Layer1 = ((dx+1)/2+1)/2 × ((dy+1)/2+1)/2
    let l2dx = dx
    let l2dy = dy
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2

    let p = try parseProfile1Frame(r: r, dx: dx, dy: dy, nextPd: nextPd)
    let frameHeader = p.frameHeader
    let mvs = p.mvs
    let refDirs = p.refDirs
    var offset = p.layer1Offset

    // Backward-adaptive tables apply to profile 0x02 only.
    let histories: FrameEntropyHistories? = frameHeader.isIFrame ? nil : entropyHistories

    // Base layer (layer 0) is always Base8
    let (baseImg, base8YBlocks, base8CbBlocks, base8CrBlocks, qtYStep, qtCStep) = try await decodeBase8(r: p.layer0Data, pool: pool, layer: 0, dx: l0dx, dy: l0dy, isIFrame: frameHeader.isIFrame, histories: histories?.streams[0])
    var current = baseImg
    var parentYBlocks: [BlockView]? = base8YBlocks
    var parentCbBlocks: [BlockView]? = base8CbBlocks
    var parentCrBlocks: [BlockView]? = base8CrBlocks

    // Determine the effective layer based on maxLayer AND available data.
    // When the splitter strips upper layers, their size becomes 0.
    // MC must be applied at the highest available layer's resolution.
    let hasLayer1 = (1 <= maxLayer && 0 < frameHeader.layer1Size)
    let hasLayer2 = (2 <= maxLayer && 0 < frameHeader.layer2Size)

    // Decode Layer1 if data is present
    if hasLayer1 {
        guard (offset + frameHeader.layer1Size) <= r.count else { throw DecodeError.insufficientData }
        let layer1Data = Array(r[offset..<(offset + frameHeader.layer1Size)])
        offset += frameHeader.layer1Size

        let (l16Img, l16YBlocks, l16CbBlocks, l16CrBlocks) = try await decodeLayer16(r: layer1Data, pool: pool, layer: 1, dx: l1dx, dy: l1dy, prev: current, parentYBlocks: parentYBlocks, parentCbBlocks: parentCbBlocks, parentCrBlocks: parentCrBlocks, histories: histories?.streams[1])

        if let y = parentYBlocks { pool.putBlockViewArray64(y) }
        if let cb = parentCbBlocks { pool.putBlockViewArray64(cb) }
        if let cr = parentCrBlocks { pool.putBlockViewArray64(cr) }

        current = l16Img
        parentYBlocks = l16YBlocks
        parentCbBlocks = l16CbBlocks
        parentCrBlocks = l16CrBlocks
    } else {
        offset += frameHeader.layer1Size
    }

    // Decode Layer2 if data is present (MC is applied inside decodeLayer32)
    if hasLayer2 {
        guard (offset + frameHeader.layer2Size) <= r.count else { throw DecodeError.insufficientData }
        let layer2Data = Array(r[offset..<(offset + frameHeader.layer2Size)])
        offset += frameHeader.layer2Size

        current = try await decodeLayer32(r: layer2Data, pool: pool, layer: 2, dx: l2dx, dy: l2dy, prev: current, parentYBlocks: parentYBlocks, parentCbBlocks: parentCbBlocks, parentCrBlocks: parentCrBlocks, predictedPd: predictedPd, nextPd: nextPd, mvs: mvs, refDirs: refDirs, roundOffset: roundOffset, skipMap: nil, histories: histories?.streams[2])

        if let y = parentYBlocks {
            if hasLayer1 { pool.putBlockViewArray256(y) }
            else { pool.putBlockViewArray64(y) }
        }
        if let cb = parentCbBlocks {
            if hasLayer1 { pool.putBlockViewArray256(cb) }
            else { pool.putBlockViewArray64(cb) }
        }
        if let cr = parentCrBlocks {
            if hasLayer1 { pool.putBlockViewArray256(cr) }
            else { pool.putBlockViewArray64(cr) }
        }
        parentYBlocks = nil
        parentCbBlocks = nil
        parentCrBlocks = nil
    } else {
        offset += frameHeader.layer2Size

        // Layer2 data is absent: apply MC at the highest available layer's resolution.
        // This handles the case where the splitter stripped upper layers.
        if hasLayer1 {
            await finishProfile1AtLayer1(current: &current, l1dx: l1dx, l1dy: l1dy, mvs: mvs, refDirs: refDirs, predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffset, qtYStep: qtYStep, qtCStep: qtCStep)
        } else {
            await finishProfile1AtLayer0(current: &current, l0dx: l0dx, l0dy: l0dy, mvs: mvs, refDirs: refDirs, predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffset, qtYStep: qtYStep, qtCStep: qtCStep)
        }
    }

    if let y = parentYBlocks {
        if hasLayer1 { pool.putBlockViewArray256(y) }
        else { pool.putBlockViewArray64(y) }
    }
    if let cb = parentCbBlocks {
        if hasLayer1 { pool.putBlockViewArray256(cb) }
        else { pool.putBlockViewArray64(cb) }
    }
    if let cr = parentCrBlocks {
        if hasLayer1 { pool.putBlockViewArray256(cr) }
        else { pool.putBlockViewArray64(cr) }
    }

    return current
}

/// Shared profile-0x02 frame prelude: header, skip map, motion vectors,
/// reference directions, and the layer0 payload slice — everything before
/// the layer pipelines diverge by maxLayer.
private struct Profile2Prelude {
    let frameHeader: VEVCFrameHeader
    let skipMap: [BlockMode]?
    let mvs: MotionVectors?
    let refDirs: [Bool]?
    let layer0Data: [UInt8]
    /// Byte offset of the layer1 payload within the frame chunk.
    let layer1Offset: Int
}

@inline(__always)
private func parseProfile2Frame(r: [UInt8], dx: Int, dy: Int, nextPd: PlaneData420?) throws -> Profile2Prelude {
    var offset = 0
    let frameHeader = try VEVCFrameHeader.deserialize(from: r, offset: &offset, profile: 0x02)
    if frameHeader.isCopyFrame {
        throw DecodeError.insufficientDataContext("decodeSpatialLayersForProfile2 passed copy frame")
    }

    var mvs: MotionVectors? = nil
    var refDirs: [Bool]? = nil

    let mvsCount = deriveMVCount(width: dx, height: dy)

    var skipMap: [BlockMode]? = nil
    if frameHeader.isIFrame != true && frameHeader.skipMapSize > 0 {
        guard (offset + frameHeader.skipMapSize) <= r.count else { throw DecodeError.insufficientData }
        let smData = Array(r[offset..<(offset + frameHeader.skipMapSize)])
        skipMap = try decodeSkipMap(data: smData, count: mvsCount)
        offset += frameHeader.skipMapSize
    }

    if frameHeader.isIFrame != true && 0 < frameHeader.mvsSize {
        guard (offset + frameHeader.mvsSize) <= r.count else { throw DecodeError.insufficientData }
        mvs = try decodeMVs(data: Array(r[offset..<(offset + frameHeader.mvsSize)]), count: mvsCount, skipMap: skipMap, profile: 0x02)
        offset += frameHeader.mvsSize
    }

    if frameHeader.hasRefDir {
        let refDirByteCount = frameHeader.refDirSize
        guard (offset + refDirByteCount) <= r.count else { throw DecodeError.insufficientData }
        let refDirBuf = Array(r[offset..<(offset + refDirByteCount)])
        offset += refDirByteCount

        if nextPd != nil {
            var dirs = [Bool]()
            dirs.reserveCapacity(mvsCount)
            for i in 0..<mvsCount {
                let byteIdx = i / 8
                let bitIdx = i % 8
                let isBackward = (byteIdx < refDirBuf.count) && ((refDirBuf[byteIdx] & UInt8(1 << bitIdx)) != 0)
                dirs.append(isBackward)
            }
            refDirs = dirs
        }
    }

    guard (offset + frameHeader.layer0Size) <= r.count else { throw DecodeError.insufficientData }
    let layer0Data = Array(r[offset..<(offset + frameHeader.layer0Size)])
    offset += frameHeader.layer0Size

    return Profile2Prelude(frameHeader: frameHeader, skipMap: skipMap, mvs: mvs, refDirs: refDirs, layer0Data: layer0Data, layer1Offset: offset)
}

/// Layer0 reconstruction shared by every profile-0x02 pipeline: dequant +
/// IDWT of the Base8 blocks (skip blocks bypass both — bit-exact, their
/// coefficients are all zero by construction, One-Pyramid §5), then the
/// block views return to the pool.
@inline(__always)
private func reconstructProfile2Base8(pool: BlockViewPool, l0dx: Int, l0dy: Int, yBlocks: [BlockView], cbBlocks: [BlockView], crBlocks: [BlockView], qtY0: QuantizationTable, qtC0: QuantizationTable, skipMap: [BlockMode]?, fullDx: Int, fullDy: Int) async -> Image16 {
    let l0cbDx = (l0dx + 1) / 2
    let l0cbDy = (l0dy + 1) / 2
    var baseImg = Image16(width: l0dx, height: l0dy, pool: pool, zeroed: false)
    let rc0Y = (l0dy + 7) / 8
    let cc0Y = (l0dx + 7) / 8
    let rc0C = (l0cbDy + 7) / 8
    let cc0C = (l0cbDx + 7) / 8
    if let sMap = skipMap {
        let bw = (fullDx + 31) / 32
        let bh = (fullDy + 31) / 32
        await decodeBase8ProcessYWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc0Y, rowCount: rc0Y, dx: l0dx, colCount: cc0Y, blocks: yBlocks, qt: qtY0, skipMap: sMap, sub: &baseImg)
        await decodeBase8ProcessCbWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc0C, rowCount: rc0C, dx: l0cbDx, colCount: cc0C, blocks: cbBlocks, qt: qtC0, skipMap: sMap, skipBw: bw, skipBh: bh, sub: &baseImg)
        await decodeBase8ProcessCrWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc0C, rowCount: rc0C, dx: l0cbDx, colCount: cc0C, blocks: crBlocks, qt: qtC0, skipMap: sMap, skipBw: bw, skipBh: bh, sub: &baseImg)
    } else {
        await decodeBase8ProcessY(pool: pool, taskIdx: 0, chunkSize: rc0Y, rowCount: rc0Y, dx: l0dx, colCount: cc0Y, blocks: yBlocks, qt: qtY0, sub: &baseImg)
        await decodeBase8ProcessCb(pool: pool, taskIdx: 0, chunkSize: rc0C, rowCount: rc0C, dx: l0cbDx, colCount: cc0C, blocks: cbBlocks, qt: qtC0, sub: &baseImg)
        await decodeBase8ProcessCr(pool: pool, taskIdx: 0, chunkSize: rc0C, rowCount: rc0C, dx: l0cbDx, colCount: cc0C, blocks: crBlocks, qt: qtC0, sub: &baseImg)
    }
    pool.putBlockViewArray64(yBlocks)
    pool.putBlockViewArray64(cbBlocks)
    pool.putBlockViewArray64(crBlocks)
    return baseImg
}

/// Frame decode, profile 0x02, maxLayer == 2 with full payloads — the
/// straight-line production pipeline. The parent-free entropy contexts
/// (EntropyCodec.swift) make the 9 coefficient streams (3 layers × 3
/// planes) independent; with parallelEntropy they entropy-decode
/// concurrently up front, otherwise sequentially — the outputs are
/// bit-identical either way. Parallel wins per-frame latency (single-stream
/// / low-latency playback); under GOP-parallel throughput decoding the cores
/// are already saturated and the fan-out only adds task and pool-contention
/// overhead, so the GOP-parallel Decoder requests the sequential path.
/// Streams whose upper layers were stripped by the splitter (layer sizes 0)
/// delegate to the generic decodeSpatialLayersForProfile2.
@inline(__always)
func decodeSpatialLayersForProfile2Full(r: [UInt8], pool: BlockViewPool, dx: Int, dy: Int, predictedPd: PlaneData420?, nextPd: PlaneData420?, roundOffset: Int, entropyHistories: FrameEntropyHistories?, l0State: L0RefState, parallelEntropy: Bool) async throws -> Image16 {
    let p = try parseProfile2Frame(r: r, dx: dx, dy: dy, nextPd: nextPd)
    guard 0 < p.frameHeader.layer1Size, 0 < p.frameHeader.layer2Size else {
        return try await decodeSpatialLayersForProfile2(
            r: r, pool: pool, maxLayer: 2, dx: dx, dy: dy,
            predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffset,
            entropyHistories: entropyHistories, l0State: l0State, parallelEntropy: parallelEntropy
        )
    }

    let l2dx = dx
    let l2dy = dy
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    let skipMap = p.skipMap
    let mvs = p.mvs
    let refDirs = p.refDirs
    let isIFrame = p.frameHeader.isIFrame
    // Backward-adaptive tables apply to P-frames only; the encoder resets the
    // state at every I-frame and never uses it there.
    let histories: FrameEntropyHistories? = isIFrame ? nil : entropyHistories

    var offset = p.layer1Offset
    guard (offset + p.frameHeader.layer1Size) <= r.count else { throw DecodeError.insufficientData }
    let layer1Data = Array(r[offset..<(offset + p.frameHeader.layer1Size)])
    offset += p.frameHeader.layer1Size
    guard (offset + p.frameHeader.layer2Size) <= r.count else { throw DecodeError.insufficientData }
    let layer2Data = Array(r[offset..<(offset + p.frameHeader.layer2Size)])

    // --- Stage 1: all 9 coefficient streams entropy-decode ------------------
    let (qtY0, qtC0, b0Y, b0Cb, b0Cr) = try VEVCLayerData.deserialize(from: p.layer0Data, layer: 0, layerLabel: "Base8")
    let l0cbDx = (l0dx + 1) / 2
    let l0cbDy = (l0dy + 1) / 2
    let n0Y = ((l0dy + 7) / 8) * ((l0dx + 7) / 8)
    let n0C = ((l0cbDy + 7) / 8) * ((l0cbDx + 7) / 8)

    let (qtY1, qtC1, b1Y, b1Cb, b1Cr) = try VEVCLayerData.deserialize(from: layer1Data, layer: 1, layerLabel: "Layer16")
    let l1cbDx = (l1dx + 1) / 2
    let l1cbDy = (l1dy + 1) / 2
    let n1Y = ((l1dy + 15) / 16) * ((l1dx + 15) / 16)
    let n1C = ((l1cbDy + 15) / 16) * ((l1cbDx + 15) / 16)

    let (qtY2, qtC2, b2Y, b2Cb, b2Cr) = try VEVCLayerData.deserialize(from: layer2Data, layer: 2, layerLabel: "Layer32")
    let l2cbDx = (l2dx + 1) / 2
    let l2cbDy = (l2dy + 1) / 2
    let n2Y = ((l2dy + 31) / 32) * ((l2dx + 31) / 32)
    let n2C = ((l2cbDy + 31) / 32) * ((l2cbDx + 31) / 32)

    var blocksBySlot = [[BlockView]?](repeating: nil, count: 9)
    let h0 = histories?.streams[0]
    let h1 = histories?.streams[1]
    let h2 = histories?.streams[2]
    if parallelEntropy {
        try await withThrowingTaskGroup(of: (Int, [BlockView]).self) { group in
            group.addTask { (0, try decodePlaneBaseSubbands8(data: b0Y, pool: pool, blockCount: n0Y, isIFrame: isIFrame, history: h0?[0], parentFreeStatics: true)) }
            group.addTask { (1, try decodePlaneBaseSubbands8(data: b0Cb, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[1], parentFreeStatics: true)) }
            group.addTask { (2, try decodePlaneBaseSubbands8(data: b0Cr, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[2], parentFreeStatics: true)) }
            group.addTask { (3, try decodePlaneSubbands16WithParentBlocks(data: b1Y, pool: pool, blockCount: n1Y, parentBlocks: parentFreeParents8(count: n1Y), history: h1?[0], parentFreeStatics: true)) }
            group.addTask { (4, try decodePlaneSubbands16WithParentBlocks(data: b1Cb, pool: pool, blockCount: n1C, parentBlocks: parentFreeParents8(count: n1C), history: h1?[1], parentFreeStatics: true)) }
            group.addTask { (5, try decodePlaneSubbands16WithParentBlocks(data: b1Cr, pool: pool, blockCount: n1C, parentBlocks: parentFreeParents8(count: n1C), history: h1?[2], parentFreeStatics: true)) }
            group.addTask { (6, try decodePlaneSubbands32WithParentBlocks(data: b2Y, pool: pool, blockCount: n2Y, parentBlocks: parentFreeParents16(count: n2Y), history: h2?[0], parentFreeStatics: true)) }
            group.addTask { (7, try decodePlaneSubbands32WithParentBlocks(data: b2Cb, pool: pool, blockCount: n2C, parentBlocks: parentFreeParents16(count: n2C), history: h2?[1], parentFreeStatics: true)) }
            group.addTask { (8, try decodePlaneSubbands32WithParentBlocks(data: b2Cr, pool: pool, blockCount: n2C, parentBlocks: parentFreeParents16(count: n2C), history: h2?[2], parentFreeStatics: true)) }
            for try await (slot, blocks) in group {
                blocksBySlot[slot] = blocks
            }
        }
    } else {
        blocksBySlot[0] = try decodePlaneBaseSubbands8(data: b0Y, pool: pool, blockCount: n0Y, isIFrame: isIFrame, history: h0?[0], parentFreeStatics: true)
        blocksBySlot[1] = try decodePlaneBaseSubbands8(data: b0Cb, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[1], parentFreeStatics: true)
        blocksBySlot[2] = try decodePlaneBaseSubbands8(data: b0Cr, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[2], parentFreeStatics: true)
        blocksBySlot[3] = try decodePlaneSubbands16WithParentBlocks(data: b1Y, pool: pool, blockCount: n1Y, parentBlocks: parentFreeParents8(count: n1Y), history: h1?[0], parentFreeStatics: true)
        blocksBySlot[4] = try decodePlaneSubbands16WithParentBlocks(data: b1Cb, pool: pool, blockCount: n1C, parentBlocks: parentFreeParents8(count: n1C), history: h1?[1], parentFreeStatics: true)
        blocksBySlot[5] = try decodePlaneSubbands16WithParentBlocks(data: b1Cr, pool: pool, blockCount: n1C, parentBlocks: parentFreeParents8(count: n1C), history: h1?[2], parentFreeStatics: true)
        blocksBySlot[6] = try decodePlaneSubbands32WithParentBlocks(data: b2Y, pool: pool, blockCount: n2Y, parentBlocks: parentFreeParents16(count: n2Y), history: h2?[0], parentFreeStatics: true)
        blocksBySlot[7] = try decodePlaneSubbands32WithParentBlocks(data: b2Cb, pool: pool, blockCount: n2C, parentBlocks: parentFreeParents16(count: n2C), history: h2?[1], parentFreeStatics: true)
        blocksBySlot[8] = try decodePlaneSubbands32WithParentBlocks(data: b2Cr, pool: pool, blockCount: n2C, parentBlocks: parentFreeParents16(count: n2C), history: h2?[2], parentFreeStatics: true)
    }

    // --- Stage 2: layer 0 reconstruction + L0 reference chain ---------------
    let baseImg = await reconstructProfile2Base8(pool: pool, l0dx: l0dx, l0dy: l0dy, yBlocks: blocksBySlot[0]!, cbBlocks: blocksBySlot[1]!, crBlocks: blocksBySlot[2]!, qtY0: qtY0, qtC0: qtC0, skipMap: skipMap, fullDx: dx, fullDy: dy)
    let qtYStep = Int(qtY0.step)
    let qtCStep = Int(qtC0.step)
    var current = baseImg

    // L0 closed loop (One-Pyramid §4): Base8 carries
    // r0 = LL2(source) − MC_L0(L0_ref); maintain the quarter-resolution
    // reference chain and substitute the LL2 coefficient slot
    // (L0_recon − LL2(P)) before the detail layers reconstruct.
    if isIFrame {
        let ref = freshCopy(baseImg)
        l0State.prev = ref
        l0State.ltr = ref
    } else if let tMVs = mvs, let l0Prev = l0State.prev {
        let baseCopy = freshCopy(baseImg)
        var l0Cur = Image16(width: baseImg.width, height: baseImg.height, y: baseCopy.y, cb: baseCopy.cb, cr: baseCopy.cr)
        await applyL0MotionCompensation(img: &l0Cur, prevPd: l0Prev, ltrPd: l0State.ltr, mvs: tMVs, refDirs: refDirs, skipMap: skipMap, roundOffset: roundOffset)
        finishL0Reconstruction(img: &l0Cur, qtYStepQ4: qtYStep, qtCStepQ4: qtCStep)
        if let map = skipMap {
            applyL0SkipCopy(img: &l0Cur, prevPd: l0Prev, ltrPd: l0State.ltr, skipMap: map, fullDx: dx)
        }
        let newRef = PlaneData420(img16: l0Cur)

        if let tPrev = predictedPd {
            let fullP = await buildFullResolutionPrediction(dx: dx, dy: dy, prevPd: tPrev, ltrPd: nextPd, mvs: tMVs, refDirs: refDirs, skipMap: skipMap, roundOffset: roundOffset)
            let tP = analyzeLL2(pd: fullP)
            var slot = Image16(width: newRef.width, height: newRef.height, y: newRef.y, cb: newRef.cb, cr: newRef.cr)
            subtractPlanes(&slot, tP)
            current = slot
        }

        l0State.prev = newRef
    }

    // --- Stage 3: layer 1 reconstruction -------------------------------------
    let l1YBlocks = blocksBySlot[3]!
    let l1CbBlocks = blocksBySlot[4]!
    let l1CrBlocks = blocksBySlot[5]!
    var l1Img = Image16(width: l1dx, height: l1dy, pool: pool, zeroed: false)
    let rc1Y = (l1dy + 15) / 16
    let cc1Y = (l1dx + 15) / 16
    let rc1C = (l1cbDy + 15) / 16
    let cc1C = (l1cbDx + 15) / 16
    // Layer2 skips the same block indices, so layer1 skip blocks are never
    // read — bypass their reconstruction entirely (One-Pyramid §5).
    if let sMap = skipMap {
        await decodeLayer16ProcessYWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc1Y, rowCount: rc1Y, dx: l1dx, colCount: cc1Y, blocks: l1YBlocks, prev: current, qt: qtY1, skipMap: sMap, sub: &l1Img)
        await decodeLayer16ProcessCbWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc1C, rowCount: rc1C, dx: l1cbDx, colCount: cc1C, blocks: l1CbBlocks, prev: current, qt: qtC1, skipMap: sMap, sub: &l1Img)
        await decodeLayer16ProcessCrWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc1C, rowCount: rc1C, dx: l1cbDx, colCount: cc1C, blocks: l1CrBlocks, prev: current, qt: qtC1, skipMap: sMap, sub: &l1Img)
    } else {
        await decodeLayer16ProcessY(pool: pool, taskIdx: 0, chunkSize: rc1Y, rowCount: rc1Y, dx: l1dx, colCount: cc1Y, blocks: l1YBlocks, prev: current, qt: qtY1, sub: &l1Img)
        await decodeLayer16ProcessCb(pool: pool, taskIdx: 0, chunkSize: rc1C, rowCount: rc1C, dx: l1cbDx, colCount: cc1C, blocks: l1CbBlocks, prev: current, qt: qtC1, sub: &l1Img)
        await decodeLayer16ProcessCr(pool: pool, taskIdx: 0, chunkSize: rc1C, rowCount: rc1C, dx: l1cbDx, colCount: cc1C, blocks: l1CrBlocks, prev: current, qt: qtC1, sub: &l1Img)
    }
    pool.putBlockViewArray256(l1YBlocks)
    pool.putBlockViewArray256(l1CbBlocks)
    pool.putBlockViewArray256(l1CrBlocks)
    current = l1Img

    // --- Stage 4: layer 2 reconstruction + MC + deblock ----------------------
    let l2YBlocks = blocksBySlot[6]!
    let l2CbBlocks = blocksBySlot[7]!
    let l2CrBlocks = blocksBySlot[8]!
    var l2Img = Image16(width: l2dx, height: l2dy, pool: pool)
    let rc2Y = (l2dy + 31) / 32
    let cc2Y = (l2dx + 31) / 32
    let rc2C = (l2cbDy + 31) / 32
    let cc2C = (l2cbDx + 31) / 32
    if let sMap = skipMap {
        await decodeLayer32ProcessYWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc2Y, rowCount: rc2Y, dx: l2dx, colCount: cc2Y, blocks: l2YBlocks, prev: current, qt: qtY2, skipMap: sMap, sub: &l2Img)
        await decodeLayer32ProcessCbWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc2C, rowCount: rc2C, dx: l2cbDx, colCount: cc2C, blocks: l2CbBlocks, prev: current, qt: qtC2, skipMap: sMap, sub: &l2Img)
        await decodeLayer32ProcessCrWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc2C, rowCount: rc2C, dx: l2cbDx, colCount: cc2C, blocks: l2CrBlocks, prev: current, qt: qtC2, skipMap: sMap, sub: &l2Img)
    } else {
        await decodeLayer32ProcessY(pool: pool, taskIdx: 0, chunkSize: rc2Y, rowCount: rc2Y, dx: l2dx, colCount: cc2Y, blocks: l2YBlocks, prev: current, qt: qtY2, sub: &l2Img)
        await decodeLayer32ProcessCb(pool: pool, taskIdx: 0, chunkSize: rc2C, rowCount: rc2C, dx: l2cbDx, colCount: cc2C, blocks: l2CbBlocks, prev: current, qt: qtC2, sub: &l2Img)
        await decodeLayer32ProcessCr(pool: pool, taskIdx: 0, chunkSize: rc2C, rowCount: rc2C, dx: l2cbDx, colCount: cc2C, blocks: l2CrBlocks, prev: current, qt: qtC2, sub: &l2Img)
    }
    pool.putBlockViewArray1024(l2YBlocks)
    pool.putBlockViewArray1024(l2CbBlocks)
    pool.putBlockViewArray1024(l2CrBlocks)

    // MC: MV is layer0 precision -> layer2 (full resolution) mvScale=4
    if let tPrev = predictedPd, let tMVs = mvs {
        if let tNext = nextPd, let dirs = refDirs {
            await applyScaledBidirectionalMotionCompensationLuma(plane: &l2Img.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: l2dx, height: l2dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
            await applyScaledBidirectionalMotionCompensationChroma(plane: &l2Img.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: l2cbDx, height: l2cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
            await applyScaledBidirectionalMotionCompensationChroma(plane: &l2Img.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: l2cbDx, height: l2cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        } else {
            await applyScaledMotionCompensationLuma(plane: &l2Img.y, prevPlane: tPrev.y, mvs: tMVs, skipMap: skipMap, width: l2dx, height: l2dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
            await applyScaledMotionCompensationChroma(plane: &l2Img.cb, prevPlane: tPrev.cb, mvs: tMVs, skipMap: skipMap, width: l2cbDx, height: l2cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
            await applyScaledMotionCompensationChroma(plane: &l2Img.cr, prevPlane: tPrev.cr, mvs: tMVs, skipMap: skipMap, width: l2cbDx, height: l2cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
        }
    }

    if let tMVs = mvs, tMVs.isEmpty != true {
        applyDeblockingFilter32(plane: &l2Img.y, width: l2dx, height: l2dy, qStep: (Int(qtY2.step) + 8) >> 4, mvs: tMVs, skipMap: skipMap)
        applyDeblockingFilterChroma16(plane: &l2Img.cb, width: l2cbDx, height: l2cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: tMVs, skipMap: skipMap)
        applyDeblockingFilterChroma16(plane: &l2Img.cr, width: l2cbDx, height: l2cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: tMVs, skipMap: skipMap)
    } else {
        applyDeblockingFilter32(plane: &l2Img.y, width: l2dx, height: l2dy, qStep: (Int(qtY2.step) + 8) >> 4)
        applyDeblockingFilter16(plane: &l2Img.cb, width: l2cbDx, height: l2cbDy, qStep: (Int(qtC2.step) + 8) >> 4)
        applyDeblockingFilter16(plane: &l2Img.cr, width: l2cbDx, height: l2cbDy, qStep: (Int(qtC2.step) + 8) >> 4)
    }

    current = l2Img

    // --- Stage 5: skip-block copies at full resolution -----------------------
    if let map = skipMap {
        let bw = (dx + 31) / 32
        let targetCbDx = (l2dx + 1) / 2
        let targetCbDy = (l2dy + 1) / 2
        let pPd = predictedPd ?? PlaneData420(width: dx, height: dy, y: [], cb: [], cr: [])
        let lPd = nextPd ?? pPd

        withUnsafePointers(
            lPd.y, lPd.cb, lPd.cr,
            pPd.y, pPd.cb, pPd.cr,
            mut: &current.y, mut: &current.cb, mut: &current.cr
        ) { ltrYPtr, ltrCbPtr, ltrCrPtr, prevYPtr, prevCbPtr, prevCrPtr, currYPtr, currCbPtr, currCrPtr in
            for i in 0..<map.count {
                let mode = map[i]
                if mode != .inter {
                    let bx = (i % bw) * 32
                    let by = (i / bw) * 32

                    if bx + 32 <= l2dx && by + 32 <= l2dy {
                        switch mode {
                        case .skip_ltr where nextPd != nil:
                            copyBlock32Pointer(from: ltrYPtr, to: currYPtr, bx: bx, by: by, stride: l2dx)
                            copyBlock16Pointer(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                            copyBlock16Pointer(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                        case .skip_prev:
                            copyBlock32Pointer(from: prevYPtr, to: currYPtr, bx: bx, by: by, stride: l2dx)
                            copyBlock16Pointer(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                            copyBlock16Pointer(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                        default: break
                        }
                    } else {
                        switch mode {
                        case .skip_ltr where nextPd != nil:
                            copyBlockSafe(from: ltrYPtr, to: currYPtr, bx: bx, by: by, width: l2dx, height: l2dy, blockSize: 32)
                            copyBlockSafe(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 16)
                            copyBlockSafe(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 16)
                        case .skip_prev:
                            copyBlockSafe(from: prevYPtr, to: currYPtr, bx: bx, by: by, width: l2dx, height: l2dy, blockSize: 32)
                            copyBlockSafe(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 16)
                            copyBlockSafe(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 16)
                        default: break
                        }
                    }
                }
            }
        }
    }

    return current
}

/// Frame decode, profile 0x02, maxLayer == 1: Base8 + Layer1, MC at half
/// resolution. Layer1 is the display output, so every block reconstructs
/// (no skip bypass). A stripped layer1 (size 0) delegates to the generic
/// decodeSpatialLayersForProfile2.
@inline(__always)
func decodeSpatialLayersForProfile2WithLayer1(r: [UInt8], pool: BlockViewPool, dx: Int, dy: Int, predictedPd: PlaneData420?, nextPd: PlaneData420?, roundOffset: Int, entropyHistories: FrameEntropyHistories?, l0State: L0RefState, parallelEntropy: Bool) async throws -> Image16 {
    let p = try parseProfile2Frame(r: r, dx: dx, dy: dy, nextPd: nextPd)
    guard 0 < p.frameHeader.layer1Size else {
        return try await decodeSpatialLayersForProfile2(
            r: r, pool: pool, maxLayer: 1, dx: dx, dy: dy,
            predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffset,
            entropyHistories: entropyHistories, l0State: l0State, parallelEntropy: parallelEntropy
        )
    }

    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    let skipMap = p.skipMap
    let mvs = p.mvs
    let refDirs = p.refDirs
    let isIFrame = p.frameHeader.isIFrame
    let histories: FrameEntropyHistories? = isIFrame ? nil : entropyHistories

    let offset = p.layer1Offset
    guard (offset + p.frameHeader.layer1Size) <= r.count else { throw DecodeError.insufficientData }
    let layer1Data = Array(r[offset..<(offset + p.frameHeader.layer1Size)])

    // --- Stage 1: the 6 coefficient streams entropy-decode ------------------
    let (qtY0, qtC0, b0Y, b0Cb, b0Cr) = try VEVCLayerData.deserialize(from: p.layer0Data, layer: 0, layerLabel: "Base8")
    let l0cbDx = (l0dx + 1) / 2
    let l0cbDy = (l0dy + 1) / 2
    let n0Y = ((l0dy + 7) / 8) * ((l0dx + 7) / 8)
    let n0C = ((l0cbDy + 7) / 8) * ((l0cbDx + 7) / 8)

    let (qtY1, qtC1, b1Y, b1Cb, b1Cr) = try VEVCLayerData.deserialize(from: layer1Data, layer: 1, layerLabel: "Layer16")
    let l1cbDx = (l1dx + 1) / 2
    let l1cbDy = (l1dy + 1) / 2
    let n1Y = ((l1dy + 15) / 16) * ((l1dx + 15) / 16)
    let n1C = ((l1cbDy + 15) / 16) * ((l1cbDx + 15) / 16)

    var blocksBySlot = [[BlockView]?](repeating: nil, count: 6)
    let h0 = histories?.streams[0]
    let h1 = histories?.streams[1]
    if parallelEntropy {
        try await withThrowingTaskGroup(of: (Int, [BlockView]).self) { group in
            group.addTask { (0, try decodePlaneBaseSubbands8(data: b0Y, pool: pool, blockCount: n0Y, isIFrame: isIFrame, history: h0?[0], parentFreeStatics: true)) }
            group.addTask { (1, try decodePlaneBaseSubbands8(data: b0Cb, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[1], parentFreeStatics: true)) }
            group.addTask { (2, try decodePlaneBaseSubbands8(data: b0Cr, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[2], parentFreeStatics: true)) }
            group.addTask { (3, try decodePlaneSubbands16WithParentBlocks(data: b1Y, pool: pool, blockCount: n1Y, parentBlocks: parentFreeParents8(count: n1Y), history: h1?[0], parentFreeStatics: true)) }
            group.addTask { (4, try decodePlaneSubbands16WithParentBlocks(data: b1Cb, pool: pool, blockCount: n1C, parentBlocks: parentFreeParents8(count: n1C), history: h1?[1], parentFreeStatics: true)) }
            group.addTask { (5, try decodePlaneSubbands16WithParentBlocks(data: b1Cr, pool: pool, blockCount: n1C, parentBlocks: parentFreeParents8(count: n1C), history: h1?[2], parentFreeStatics: true)) }
            for try await (slot, blocks) in group {
                blocksBySlot[slot] = blocks
            }
        }
    } else {
        blocksBySlot[0] = try decodePlaneBaseSubbands8(data: b0Y, pool: pool, blockCount: n0Y, isIFrame: isIFrame, history: h0?[0], parentFreeStatics: true)
        blocksBySlot[1] = try decodePlaneBaseSubbands8(data: b0Cb, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[1], parentFreeStatics: true)
        blocksBySlot[2] = try decodePlaneBaseSubbands8(data: b0Cr, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[2], parentFreeStatics: true)
        blocksBySlot[3] = try decodePlaneSubbands16WithParentBlocks(data: b1Y, pool: pool, blockCount: n1Y, parentBlocks: parentFreeParents8(count: n1Y), history: h1?[0], parentFreeStatics: true)
        blocksBySlot[4] = try decodePlaneSubbands16WithParentBlocks(data: b1Cb, pool: pool, blockCount: n1C, parentBlocks: parentFreeParents8(count: n1C), history: h1?[1], parentFreeStatics: true)
        blocksBySlot[5] = try decodePlaneSubbands16WithParentBlocks(data: b1Cr, pool: pool, blockCount: n1C, parentBlocks: parentFreeParents8(count: n1C), history: h1?[2], parentFreeStatics: true)
    }

    // --- Stage 2: layer 0 reconstruction + L0 reference chain ---------------
    let baseImg = await reconstructProfile2Base8(pool: pool, l0dx: l0dx, l0dy: l0dy, yBlocks: blocksBySlot[0]!, cbBlocks: blocksBySlot[1]!, crBlocks: blocksBySlot[2]!, qtY0: qtY0, qtC0: qtC0, skipMap: skipMap, fullDx: dx, fullDy: dy)
    let qtYStep = Int(qtY0.step)
    let qtCStep = Int(qtC0.step)
    var current = baseImg

    // L0 closed loop (One-Pyramid §4): same chain as the full pipeline, with
    // the prediction analyzed at layer1 resolution (T = LL of the L1
    // prediction) because no full-resolution plane exists at this maxLayer.
    if isIFrame {
        let ref = freshCopy(baseImg)
        l0State.prev = ref
        l0State.ltr = ref
    } else if let tMVs = mvs, let l0Prev = l0State.prev {
        let baseCopy = freshCopy(baseImg)
        var l0Cur = Image16(width: baseImg.width, height: baseImg.height, y: baseCopy.y, cb: baseCopy.cb, cr: baseCopy.cr)
        await applyL0MotionCompensation(img: &l0Cur, prevPd: l0Prev, ltrPd: l0State.ltr, mvs: tMVs, refDirs: refDirs, skipMap: skipMap, roundOffset: roundOffset)
        finishL0Reconstruction(img: &l0Cur, qtYStepQ4: qtYStep, qtCStepQ4: qtCStep)
        if let map = skipMap {
            applyL0SkipCopy(img: &l0Cur, prevPd: l0Prev, ltrPd: l0State.ltr, skipMap: map, fullDx: dx)
        }
        let newRef = PlaneData420(img16: l0Cur)

        if let tPrev = predictedPd {
            let l1P = await buildL1Prediction(l1dx: l1dx, l1dy: l1dy, prevPd: tPrev, ltrPd: nextPd, mvs: tMVs, refDirs: refDirs, skipMap: skipMap, roundOffset: roundOffset)
            let tP = analyzeLL1(pd: l1P)
            var slot = Image16(width: newRef.width, height: newRef.height, y: newRef.y, cb: newRef.cb, cr: newRef.cr)
            subtractPlanes(&slot, tP)
            current = slot
        }

        l0State.prev = newRef
    }

    // --- Stage 3: layer 1 reconstruction (display output — every block) -----
    let l1YBlocks = blocksBySlot[3]!
    let l1CbBlocks = blocksBySlot[4]!
    let l1CrBlocks = blocksBySlot[5]!
    var l1Img = Image16(width: l1dx, height: l1dy, pool: pool, zeroed: false)
    let rc1Y = (l1dy + 15) / 16
    let cc1Y = (l1dx + 15) / 16
    let rc1C = (l1cbDy + 15) / 16
    let cc1C = (l1cbDx + 15) / 16
    await decodeLayer16ProcessY(pool: pool, taskIdx: 0, chunkSize: rc1Y, rowCount: rc1Y, dx: l1dx, colCount: cc1Y, blocks: l1YBlocks, prev: current, qt: qtY1, sub: &l1Img)
    await decodeLayer16ProcessCb(pool: pool, taskIdx: 0, chunkSize: rc1C, rowCount: rc1C, dx: l1cbDx, colCount: cc1C, blocks: l1CbBlocks, prev: current, qt: qtC1, sub: &l1Img)
    await decodeLayer16ProcessCr(pool: pool, taskIdx: 0, chunkSize: rc1C, rowCount: rc1C, dx: l1cbDx, colCount: cc1C, blocks: l1CrBlocks, prev: current, qt: qtC1, sub: &l1Img)
    pool.putBlockViewArray256(l1YBlocks)
    pool.putBlockViewArray256(l1CbBlocks)
    pool.putBlockViewArray256(l1CrBlocks)
    current = l1Img

    // --- Stage 4: MC at layer1 resolution + clamp + deblock ------------------
    if let tMVs = mvs, let tPrev = predictedPd {
        let cbDx1 = (l1dx + 1) / 2
        let cbDy1 = (l1dy + 1) / 2
        if let tNext = nextPd, let dirs = refDirs {
            await applyScaledBidirectionalMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
            await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
            await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
        } else {
            await applyScaledMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, mvs: tMVs, skipMap: skipMap, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
            await applyScaledMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, mvs: tMVs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
            await applyScaledMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, mvs: tMVs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
        }

        clampPlane(plane: &current.y)
        clampPlane(plane: &current.cb)
        clampPlane(plane: &current.cr)

        applyDeblockingFilterN(plane: &current.y, width: l1dx, height: l1dy, qStep: qtYStep, blockSize: 16)
        let cStep = min(qtCStep * 2, 255)
        applyDeblockingFilterN(plane: &current.cb, width: (l1dx + 1) / 2, height: (l1dy + 1) / 2, qStep: cStep, blockSize: 8)
        applyDeblockingFilterN(plane: &current.cr, width: (l1dx + 1) / 2, height: (l1dy + 1) / 2, qStep: cStep, blockSize: 8)
    }

    // --- Stage 5: skip-block copies at layer1 resolution ---------------------
    if let map = skipMap {
        let bw = (dx + 31) / 32
        let targetCbDx = (l1dx + 1) / 2
        let targetCbDy = (l1dy + 1) / 2
        let pPd = predictedPd ?? PlaneData420(width: dx, height: dy, y: [], cb: [], cr: [])
        let lPd = nextPd ?? pPd

        withUnsafePointers(
            lPd.y, lPd.cb, lPd.cr,
            pPd.y, pPd.cb, pPd.cr,
            mut: &current.y, mut: &current.cb, mut: &current.cr
        ) { ltrYPtr, ltrCbPtr, ltrCrPtr, prevYPtr, prevCbPtr, prevCrPtr, currYPtr, currCbPtr, currCrPtr in
            for i in 0..<map.count {
                let mode = map[i]
                if mode != .inter {
                    let bx = ((i % bw) * 32) / 2
                    let by = ((i / bw) * 32) / 2

                    if bx + 16 <= l1dx && by + 16 <= l1dy {
                        switch mode {
                        case .skip_ltr where nextPd != nil:
                            copyBlock16Pointer(from: ltrYPtr, to: currYPtr, bx: bx, by: by, stride: l1dx)
                            copyBlock8Pointer(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                            copyBlock8Pointer(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                        case .skip_prev:
                            copyBlock16Pointer(from: prevYPtr, to: currYPtr, bx: bx, by: by, stride: l1dx)
                            copyBlock8Pointer(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                            copyBlock8Pointer(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                        default: break
                        }
                    } else {
                        switch mode {
                        case .skip_ltr where nextPd != nil:
                            copyBlockSafe(from: ltrYPtr, to: currYPtr, bx: bx, by: by, width: l1dx, height: l1dy, blockSize: 16)
                            copyBlockSafe(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 8)
                            copyBlockSafe(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 8)
                        case .skip_prev:
                            copyBlockSafe(from: prevYPtr, to: currYPtr, bx: bx, by: by, width: l1dx, height: l1dy, blockSize: 16)
                            copyBlockSafe(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 8)
                            copyBlockSafe(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 8)
                        default: break
                        }
                    }
                }
            }
        }
    }

    return current
}

/// Frame decode, profile 0x02, maxLayer == 0: Base8 + MC at quarter
/// resolution on the layer-matched chain. This pipeline IS the L0 loop's
/// decode side (deq(r0) + MC_L0), so it needs no separate L0RefState and
/// never reads the upper-layer payloads.
@inline(__always)
func decodeSpatialLayersForProfile2Base8Only(r: [UInt8], pool: BlockViewPool, dx: Int, dy: Int, predictedPd: PlaneData420?, nextPd: PlaneData420?, roundOffset: Int, entropyHistories: FrameEntropyHistories?, parallelEntropy: Bool) async throws -> Image16 {
    let p = try parseProfile2Frame(r: r, dx: dx, dy: dy, nextPd: nextPd)

    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2
    let skipMap = p.skipMap
    let mvs = p.mvs
    let refDirs = p.refDirs
    let isIFrame = p.frameHeader.isIFrame
    let histories: FrameEntropyHistories? = isIFrame ? nil : entropyHistories

    // --- Stage 1: the 3 Base8 coefficient streams entropy-decode ------------
    let (qtY0, qtC0, b0Y, b0Cb, b0Cr) = try VEVCLayerData.deserialize(from: p.layer0Data, layer: 0, layerLabel: "Base8")
    let l0cbDx = (l0dx + 1) / 2
    let l0cbDy = (l0dy + 1) / 2
    let n0Y = ((l0dy + 7) / 8) * ((l0dx + 7) / 8)
    let n0C = ((l0cbDy + 7) / 8) * ((l0cbDx + 7) / 8)

    var blocksBySlot = [[BlockView]?](repeating: nil, count: 3)
    let h0 = histories?.streams[0]
    if parallelEntropy {
        try await withThrowingTaskGroup(of: (Int, [BlockView]).self) { group in
            group.addTask { (0, try decodePlaneBaseSubbands8(data: b0Y, pool: pool, blockCount: n0Y, isIFrame: isIFrame, history: h0?[0], parentFreeStatics: true)) }
            group.addTask { (1, try decodePlaneBaseSubbands8(data: b0Cb, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[1], parentFreeStatics: true)) }
            group.addTask { (2, try decodePlaneBaseSubbands8(data: b0Cr, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[2], parentFreeStatics: true)) }
            for try await (slot, blocks) in group {
                blocksBySlot[slot] = blocks
            }
        }
    } else {
        blocksBySlot[0] = try decodePlaneBaseSubbands8(data: b0Y, pool: pool, blockCount: n0Y, isIFrame: isIFrame, history: h0?[0], parentFreeStatics: true)
        blocksBySlot[1] = try decodePlaneBaseSubbands8(data: b0Cb, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[1], parentFreeStatics: true)
        blocksBySlot[2] = try decodePlaneBaseSubbands8(data: b0Cr, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[2], parentFreeStatics: true)
    }

    // --- Stage 2: layer 0 reconstruction -------------------------------------
    var current = await reconstructProfile2Base8(pool: pool, l0dx: l0dx, l0dy: l0dy, yBlocks: blocksBySlot[0]!, cbBlocks: blocksBySlot[1]!, crBlocks: blocksBySlot[2]!, qtY0: qtY0, qtC0: qtC0, skipMap: skipMap, fullDx: dx, fullDy: dy)
    let qtYStep = Int(qtY0.step)
    let qtCStep = Int(qtC0.step)

    // --- Stage 4: MC at layer0 resolution + clamp + deblock ------------------
    if let tMVs = mvs, let tPrev = predictedPd {
        let cbDx0 = (l0dx + 1) / 2
        let cbDy0 = (l0dy + 1) / 2
        if let tNext = nextPd, let dirs = refDirs {
            await applyScaledBidirectionalMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
            await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
            await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
        } else {
            await applyScaledMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, mvs: tMVs, skipMap: skipMap, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
            await applyScaledMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, mvs: tMVs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
            await applyScaledMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, mvs: tMVs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
        }

        clampPlane(plane: &current.y)
        clampPlane(plane: &current.cb)
        clampPlane(plane: &current.cr)

        applyDeblockingFilterN(plane: &current.y, width: l0dx, height: l0dy, qStep: qtYStep, blockSize: 8)
        let cStep = min(qtCStep * 2, 255)
        applyDeblockingFilterN(plane: &current.cb, width: cbDx0, height: cbDy0, qStep: cStep, blockSize: 4)
        applyDeblockingFilterN(plane: &current.cr, width: cbDx0, height: cbDy0, qStep: cStep, blockSize: 4)
    }

    // --- Stage 5: skip-block copies at layer0 resolution ---------------------
    if let map = skipMap {
        let bw = (dx + 31) / 32
        let targetCbDx = (l0dx + 1) / 2
        let targetCbDy = (l0dy + 1) / 2
        let pPd = predictedPd ?? PlaneData420(width: dx, height: dy, y: [], cb: [], cr: [])
        let lPd = nextPd ?? pPd

        withUnsafePointers(
            lPd.y, lPd.cb, lPd.cr,
            pPd.y, pPd.cb, pPd.cr,
            mut: &current.y, mut: &current.cb, mut: &current.cr
        ) { ltrYPtr, ltrCbPtr, ltrCrPtr, prevYPtr, prevCbPtr, prevCrPtr, currYPtr, currCbPtr, currCrPtr in
            for i in 0..<map.count {
                let mode = map[i]
                if mode != .inter {
                    let bx = ((i % bw) * 32) / 4
                    let by = ((i / bw) * 32) / 4

                    if bx + 8 <= l0dx && by + 8 <= l0dy {
                        switch mode {
                        case .skip_ltr where nextPd != nil:
                            copyBlock8Pointer(from: ltrYPtr, to: currYPtr, bx: bx, by: by, stride: l0dx)
                            copyBlock4Pointer(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                            copyBlock4Pointer(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                        case .skip_prev:
                            copyBlock8Pointer(from: prevYPtr, to: currYPtr, bx: bx, by: by, stride: l0dx)
                            copyBlock4Pointer(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                            copyBlock4Pointer(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx)
                        default: break
                        }
                    } else {
                        switch mode {
                        case .skip_ltr where nextPd != nil:
                            copyBlockSafe(from: ltrYPtr, to: currYPtr, bx: bx, by: by, width: l0dx, height: l0dy, blockSize: 8)
                            copyBlockSafe(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 4)
                            copyBlockSafe(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 4)
                        case .skip_prev:
                            copyBlockSafe(from: prevYPtr, to: currYPtr, bx: bx, by: by, width: l0dx, height: l0dy, blockSize: 8)
                            copyBlockSafe(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 4)
                            copyBlockSafe(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: 4)
                        default: break
                        }
                    }
                }
            }
        }
    }

    return current
}

/// Frame decode, profile 0x02, generic pipeline: handles streams whose
/// upper layers were stripped by the splitter (layer sizes 0), where the
/// effective top layer is only known from the frame header. Production
/// decoding uses the straight-line variants above, which delegate here
/// only in those stripped cases.
@inline(__always)
func decodeSpatialLayersForProfile2(r: [UInt8], pool: BlockViewPool, maxLayer: Int, dx: Int, dy: Int, predictedPd: PlaneData420?, nextPd: PlaneData420?, roundOffset: Int, entropyHistories: FrameEntropyHistories?, l0State: L0RefState?, parallelEntropy: Bool) async throws -> Image16 {
    let l2dx = dx
    let l2dy = dy
    let l1dx = (dx + 1) / 2
    let l1dy = (dy + 1) / 2
    let l0dx = (l1dx + 1) / 2
    let l0dy = (l1dy + 1) / 2

    let p = try parseProfile2Frame(r: r, dx: dx, dy: dy, nextPd: nextPd)
    let frameHeader = p.frameHeader
    let skipMap = p.skipMap
    let mvs = p.mvs
    let refDirs = p.refDirs
    let layer0Data = p.layer0Data
    var offset = p.layer1Offset

    let hasLayer1 = (1 <= maxLayer && 0 < frameHeader.layer1Size)
    let hasLayer2 = (2 <= maxLayer && 0 < frameHeader.layer2Size)

    var layer1Data: [UInt8] = []
    if hasLayer1 {
        guard (offset + frameHeader.layer1Size) <= r.count else { throw DecodeError.insufficientData }
        layer1Data = Array(r[offset..<(offset + frameHeader.layer1Size)])
    }
    offset += frameHeader.layer1Size

    var layer2Data: [UInt8] = []
    if hasLayer2 {
        guard (offset + frameHeader.layer2Size) <= r.count else { throw DecodeError.insufficientData }
        layer2Data = Array(r[offset..<(offset + frameHeader.layer2Size)])
    }
    offset += frameHeader.layer2Size

    // Backward-adaptive tables apply to P-frames only; the encoder resets the
    // state at every I-frame and never uses it there.
    let histories: FrameEntropyHistories? = frameHeader.isIFrame ? nil : entropyHistories
    let isIFrame = frameHeader.isIFrame

    // --- Stage 1: all coefficient streams entropy-decode concurrently -------
    let (qtY0, qtC0, b0Y, b0Cb, b0Cr) = try VEVCLayerData.deserialize(from: layer0Data, layer: 0, layerLabel: "Base8")

    let l0cbDx = (l0dx + 1) / 2
    let l0cbDy = (l0dy + 1) / 2
    let n0Y = ((l0dy + 7) / 8) * ((l0dx + 7) / 8)
    let n0C = ((l0cbDy + 7) / 8) * ((l0cbDx + 7) / 8)

    let (qtY1, qtC1, b1Y, b1Cb, b1Cr): (QuantizationTable, QuantizationTable, ArraySlice<UInt8>, ArraySlice<UInt8>, ArraySlice<UInt8>)
    if hasLayer1 {
        (qtY1, qtC1, b1Y, b1Cb, b1Cr) = try VEVCLayerData.deserialize(from: layer1Data, layer: 1, layerLabel: "Layer16")
    } else {
        (qtY1, qtC1, b1Y, b1Cb, b1Cr) = (qtY0, qtC0, ArraySlice<UInt8>(), ArraySlice<UInt8>(), ArraySlice<UInt8>())
    }
    let l1cbDx = (l1dx + 1) / 2
    let l1cbDy = (l1dy + 1) / 2
    let n1Y = ((l1dy + 15) / 16) * ((l1dx + 15) / 16)
    let n1C = ((l1cbDy + 15) / 16) * ((l1cbDx + 15) / 16)

    let (qtY2, qtC2, b2Y, b2Cb, b2Cr): (QuantizationTable, QuantizationTable, ArraySlice<UInt8>, ArraySlice<UInt8>, ArraySlice<UInt8>)
    if hasLayer2 {
        (qtY2, qtC2, b2Y, b2Cb, b2Cr) = try VEVCLayerData.deserialize(from: layer2Data, layer: 2, layerLabel: "Layer32")
    } else {
        (qtY2, qtC2, b2Y, b2Cb, b2Cr) = (qtY0, qtC0, ArraySlice<UInt8>(), ArraySlice<UInt8>(), ArraySlice<UInt8>())
    }
    let l2cbDx = (l2dx + 1) / 2
    let l2cbDy = (l2dy + 1) / 2
    let n2Y = ((l2dy + 31) / 32) * ((l2dx + 31) / 32)
    let n2C = ((l2cbDy + 31) / 32) * ((l2cbDx + 31) / 32)

    var blocksBySlot = [[BlockView]?](repeating: nil, count: 9)
    let h0 = histories?.streams[0]
    let h1 = histories?.streams[1]
    let h2 = histories?.streams[2]
    if parallelEntropy {
        try await withThrowingTaskGroup(of: (Int, [BlockView]).self) { group in
            group.addTask { (0, try decodePlaneBaseSubbands8(data: b0Y, pool: pool, blockCount: n0Y, isIFrame: isIFrame, history: h0?[0], parentFreeStatics: true)) }
            group.addTask { (1, try decodePlaneBaseSubbands8(data: b0Cb, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[1], parentFreeStatics: true)) }
            group.addTask { (2, try decodePlaneBaseSubbands8(data: b0Cr, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[2], parentFreeStatics: true)) }
            if hasLayer1 {
                group.addTask { (3, try decodePlaneSubbands16WithParentBlocks(data: b1Y, pool: pool, blockCount: n1Y, parentBlocks: parentFreeParents8(count: n1Y), history: h1?[0], parentFreeStatics: true)) }
                group.addTask { (4, try decodePlaneSubbands16WithParentBlocks(data: b1Cb, pool: pool, blockCount: n1C, parentBlocks: parentFreeParents8(count: n1C), history: h1?[1], parentFreeStatics: true)) }
                group.addTask { (5, try decodePlaneSubbands16WithParentBlocks(data: b1Cr, pool: pool, blockCount: n1C, parentBlocks: parentFreeParents8(count: n1C), history: h1?[2], parentFreeStatics: true)) }
            }
            if hasLayer2 {
                group.addTask { (6, try decodePlaneSubbands32WithParentBlocks(data: b2Y, pool: pool, blockCount: n2Y, parentBlocks: parentFreeParents16(count: n2Y), history: h2?[0], parentFreeStatics: true)) }
                group.addTask { (7, try decodePlaneSubbands32WithParentBlocks(data: b2Cb, pool: pool, blockCount: n2C, parentBlocks: parentFreeParents16(count: n2C), history: h2?[1], parentFreeStatics: true)) }
                group.addTask { (8, try decodePlaneSubbands32WithParentBlocks(data: b2Cr, pool: pool, blockCount: n2C, parentBlocks: parentFreeParents16(count: n2C), history: h2?[2], parentFreeStatics: true)) }
            }
            for try await (slot, blocks) in group {
                blocksBySlot[slot] = blocks
            }
        }
    } else {
        blocksBySlot[0] = try decodePlaneBaseSubbands8(data: b0Y, pool: pool, blockCount: n0Y, isIFrame: isIFrame, history: h0?[0], parentFreeStatics: true)
        blocksBySlot[1] = try decodePlaneBaseSubbands8(data: b0Cb, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[1], parentFreeStatics: true)
        blocksBySlot[2] = try decodePlaneBaseSubbands8(data: b0Cr, pool: pool, blockCount: n0C, isIFrame: isIFrame, history: h0?[2], parentFreeStatics: true)
        if hasLayer1 {
            blocksBySlot[3] = try decodePlaneSubbands16WithParentBlocks(data: b1Y, pool: pool, blockCount: n1Y, parentBlocks: parentFreeParents8(count: n1Y), history: h1?[0], parentFreeStatics: true)
            blocksBySlot[4] = try decodePlaneSubbands16WithParentBlocks(data: b1Cb, pool: pool, blockCount: n1C, parentBlocks: parentFreeParents8(count: n1C), history: h1?[1], parentFreeStatics: true)
            blocksBySlot[5] = try decodePlaneSubbands16WithParentBlocks(data: b1Cr, pool: pool, blockCount: n1C, parentBlocks: parentFreeParents8(count: n1C), history: h1?[2], parentFreeStatics: true)
        }
        if hasLayer2 {
            blocksBySlot[6] = try decodePlaneSubbands32WithParentBlocks(data: b2Y, pool: pool, blockCount: n2Y, parentBlocks: parentFreeParents16(count: n2Y), history: h2?[0], parentFreeStatics: true)
            blocksBySlot[7] = try decodePlaneSubbands32WithParentBlocks(data: b2Cb, pool: pool, blockCount: n2C, parentBlocks: parentFreeParents16(count: n2C), history: h2?[1], parentFreeStatics: true)
            blocksBySlot[8] = try decodePlaneSubbands32WithParentBlocks(data: b2Cr, pool: pool, blockCount: n2C, parentBlocks: parentFreeParents16(count: n2C), history: h2?[2], parentFreeStatics: true)
        }
    }
    let base8YBlocks = blocksBySlot[0]!
    let base8CbBlocks = blocksBySlot[1]!
    let base8CrBlocks = blocksBySlot[2]!

    // --- Stage 2: layer 0 reconstruction + L0 reference chain ---------------
    // Skip blocks bypass dequant/IDWT (One-Pyramid §5, DecodeSkipBypass.swift)
    // — bit-exact, their coefficients are all zero by construction.
    var baseImg = Image16(width: l0dx, height: l0dy, pool: pool, zeroed: false)
    let rc0Y = (l0dy + 7) / 8
    let cc0Y = (l0dx + 7) / 8
    let rc0C = (l0cbDy + 7) / 8
    let cc0C = (l0cbDx + 7) / 8
    if let sMap = skipMap {
        let bw = (dx + 31) / 32
        let bh = (dy + 31) / 32
        await decodeBase8ProcessYWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc0Y, rowCount: rc0Y, dx: l0dx, colCount: cc0Y, blocks: base8YBlocks, qt: qtY0, skipMap: sMap, sub: &baseImg)
        await decodeBase8ProcessCbWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc0C, rowCount: rc0C, dx: l0cbDx, colCount: cc0C, blocks: base8CbBlocks, qt: qtC0, skipMap: sMap, skipBw: bw, skipBh: bh, sub: &baseImg)
        await decodeBase8ProcessCrWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc0C, rowCount: rc0C, dx: l0cbDx, colCount: cc0C, blocks: base8CrBlocks, qt: qtC0, skipMap: sMap, skipBw: bw, skipBh: bh, sub: &baseImg)
    } else {
        await decodeBase8ProcessY(pool: pool, taskIdx: 0, chunkSize: rc0Y, rowCount: rc0Y, dx: l0dx, colCount: cc0Y, blocks: base8YBlocks, qt: qtY0, sub: &baseImg)
        await decodeBase8ProcessCb(pool: pool, taskIdx: 0, chunkSize: rc0C, rowCount: rc0C, dx: l0cbDx, colCount: cc0C, blocks: base8CbBlocks, qt: qtC0, sub: &baseImg)
        await decodeBase8ProcessCr(pool: pool, taskIdx: 0, chunkSize: rc0C, rowCount: rc0C, dx: l0cbDx, colCount: cc0C, blocks: base8CrBlocks, qt: qtC0, sub: &baseImg)
    }
    pool.putBlockViewArray64(base8YBlocks)
    pool.putBlockViewArray64(base8CbBlocks)
    pool.putBlockViewArray64(base8CrBlocks)

    let qtYStep = Int(qtY0.step)
    let qtCStep = Int(qtC0.step)
    var current = baseImg

    // L0 closed loop (One-Pyramid §4, with an l0State chain). Base8 carries
    // r0 = LL2(source) − MC_L0(L0_ref); maintain the quarter-resolution
    // reference chain and substitute the LL2 coefficient slot
    // (L0_recon − LL2(P)) before the detail layers reconstruct. The
    // maxLayer==0 path needs none of this: its existing pipeline is exactly
    // deq(r0) + MC_L0 on its own layer-matched chain.
    if let l0s = l0State {
        switch true {
        case frameHeader.isIFrame:
            let ref = freshCopy(baseImg)
            l0s.prev = ref
            l0s.ltr = ref
        case (hasLayer1 || hasLayer2):
            guard let tMVs = mvs, let l0Prev = l0s.prev else { break }
            let baseCopy = freshCopy(baseImg)
            var l0Cur = Image16(width: baseImg.width, height: baseImg.height, y: baseCopy.y, cb: baseCopy.cb, cr: baseCopy.cr)
            await applyL0MotionCompensation(img: &l0Cur, prevPd: l0Prev, ltrPd: l0s.ltr, mvs: tMVs, refDirs: refDirs, skipMap: skipMap, roundOffset: roundOffset)
            finishL0Reconstruction(img: &l0Cur, qtYStepQ4: qtYStep, qtCStepQ4: qtCStep)
            if let map = skipMap {
                applyL0SkipCopy(img: &l0Cur, prevPd: l0Prev, ltrPd: l0s.ltr, skipMap: map, fullDx: dx)
            }
            let newRef = PlaneData420(img16: l0Cur)

            if let tPrev = predictedPd {
                let tP: PlaneData420
                if hasLayer2 {
                    let fullP = await buildFullResolutionPrediction(dx: dx, dy: dy, prevPd: tPrev, ltrPd: nextPd, mvs: tMVs, refDirs: refDirs, skipMap: skipMap, roundOffset: roundOffset)
                    tP = analyzeLL2(pd: fullP)
                } else {
                    let l1P = await buildL1Prediction(l1dx: l1dx, l1dy: l1dy, prevPd: tPrev, ltrPd: nextPd, mvs: tMVs, refDirs: refDirs, skipMap: skipMap, roundOffset: roundOffset)
                    tP = analyzeLL1(pd: l1P)
                }
                var slot = Image16(width: newRef.width, height: newRef.height, y: newRef.y, cb: newRef.cb, cr: newRef.cr)
                subtractPlanes(&slot, tP)
                current = slot
            }

            l0s.prev = newRef
        default:
            break
        }
    }

    // --- Stage 3: layer 1 reconstruction -------------------------------------
    if hasLayer1 {
        let l1YBlocks = blocksBySlot[3]!
        let l1CbBlocks = blocksBySlot[4]!
        let l1CrBlocks = blocksBySlot[5]!
        var l1Img = Image16(width: l1dx, height: l1dy, pool: pool, zeroed: false)
        let rc1Y = (l1dy + 15) / 16
        let cc1Y = (l1dx + 15) / 16
        let rc1C = (l1cbDy + 15) / 16
        let cc1C = (l1cbDx + 15) / 16
        // With layer2 present, layer2 skips the same block indices, so layer1
        // skip blocks are never read — bypass their reconstruction entirely
        // (One-Pyramid §5). Without layer2 this plane is the display output,
        // so every block reconstructs.
        if hasLayer2, let sMap = skipMap {
            await decodeLayer16ProcessYWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc1Y, rowCount: rc1Y, dx: l1dx, colCount: cc1Y, blocks: l1YBlocks, prev: current, qt: qtY1, skipMap: sMap, sub: &l1Img)
            await decodeLayer16ProcessCbWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc1C, rowCount: rc1C, dx: l1cbDx, colCount: cc1C, blocks: l1CbBlocks, prev: current, qt: qtC1, skipMap: sMap, sub: &l1Img)
            await decodeLayer16ProcessCrWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc1C, rowCount: rc1C, dx: l1cbDx, colCount: cc1C, blocks: l1CrBlocks, prev: current, qt: qtC1, skipMap: sMap, sub: &l1Img)
        } else {
            await decodeLayer16ProcessY(pool: pool, taskIdx: 0, chunkSize: rc1Y, rowCount: rc1Y, dx: l1dx, colCount: cc1Y, blocks: l1YBlocks, prev: current, qt: qtY1, sub: &l1Img)
            await decodeLayer16ProcessCb(pool: pool, taskIdx: 0, chunkSize: rc1C, rowCount: rc1C, dx: l1cbDx, colCount: cc1C, blocks: l1CbBlocks, prev: current, qt: qtC1, sub: &l1Img)
            await decodeLayer16ProcessCr(pool: pool, taskIdx: 0, chunkSize: rc1C, rowCount: rc1C, dx: l1cbDx, colCount: cc1C, blocks: l1CrBlocks, prev: current, qt: qtC1, sub: &l1Img)
        }
        pool.putBlockViewArray256(l1YBlocks)
        pool.putBlockViewArray256(l1CbBlocks)
        pool.putBlockViewArray256(l1CrBlocks)
        current = l1Img
    }

    // --- Stage 4: layer 2 reconstruction + MC + deblock ----------------------
    if hasLayer2 {
        let l2YBlocks = blocksBySlot[6]!
        let l2CbBlocks = blocksBySlot[7]!
        let l2CrBlocks = blocksBySlot[8]!
        var l2Img = Image16(width: l2dx, height: l2dy, pool: pool)
        let rc2Y = (l2dy + 31) / 32
        let cc2Y = (l2dx + 31) / 32
        let rc2C = (l2cbDy + 31) / 32
        let cc2C = (l2cbDx + 31) / 32
        if let sMap = skipMap {
            await decodeLayer32ProcessYWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc2Y, rowCount: rc2Y, dx: l2dx, colCount: cc2Y, blocks: l2YBlocks, prev: current, qt: qtY2, skipMap: sMap, sub: &l2Img)
            await decodeLayer32ProcessCbWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc2C, rowCount: rc2C, dx: l2cbDx, colCount: cc2C, blocks: l2CbBlocks, prev: current, qt: qtC2, skipMap: sMap, sub: &l2Img)
            await decodeLayer32ProcessCrWithSkipMap(pool: pool, taskIdx: 0, chunkSize: rc2C, rowCount: rc2C, dx: l2cbDx, colCount: cc2C, blocks: l2CrBlocks, prev: current, qt: qtC2, skipMap: sMap, sub: &l2Img)
        } else {
            await decodeLayer32ProcessY(pool: pool, taskIdx: 0, chunkSize: rc2Y, rowCount: rc2Y, dx: l2dx, colCount: cc2Y, blocks: l2YBlocks, prev: current, qt: qtY2, sub: &l2Img)
            await decodeLayer32ProcessCb(pool: pool, taskIdx: 0, chunkSize: rc2C, rowCount: rc2C, dx: l2cbDx, colCount: cc2C, blocks: l2CbBlocks, prev: current, qt: qtC2, sub: &l2Img)
            await decodeLayer32ProcessCr(pool: pool, taskIdx: 0, chunkSize: rc2C, rowCount: rc2C, dx: l2cbDx, colCount: cc2C, blocks: l2CrBlocks, prev: current, qt: qtC2, sub: &l2Img)
        }
        pool.putBlockViewArray1024(l2YBlocks)
        pool.putBlockViewArray1024(l2CbBlocks)
        pool.putBlockViewArray1024(l2CrBlocks)

        // MC: MV is layer0 precision -> layer2 (full resolution) mvScale=4
        if let tPrev = predictedPd, let tMVs = mvs {
            if let tNext = nextPd, let dirs = refDirs {
                await applyScaledBidirectionalMotionCompensationLuma(plane: &l2Img.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: l2dx, height: l2dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
                await applyScaledBidirectionalMotionCompensationChroma(plane: &l2Img.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: l2cbDx, height: l2cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
                await applyScaledBidirectionalMotionCompensationChroma(plane: &l2Img.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: l2cbDx, height: l2cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
            } else {
                await applyScaledMotionCompensationLuma(plane: &l2Img.y, prevPlane: tPrev.y, mvs: tMVs, skipMap: skipMap, width: l2dx, height: l2dy, lumaBlockSize: 32, mvShift: 0, roundOffset: roundOffset)
                await applyScaledMotionCompensationChroma(plane: &l2Img.cb, prevPlane: tPrev.cb, mvs: tMVs, skipMap: skipMap, width: l2cbDx, height: l2cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
                await applyScaledMotionCompensationChroma(plane: &l2Img.cr, prevPlane: tPrev.cr, mvs: tMVs, skipMap: skipMap, width: l2cbDx, height: l2cbDy, chromaBlockSize: 16, mvShift: 0, roundOffset: roundOffset)
            }
        }

        if let tMVs = mvs, tMVs.isEmpty != true {
            applyDeblockingFilter32(plane: &l2Img.y, width: l2dx, height: l2dy, qStep: (Int(qtY2.step) + 8) >> 4, mvs: tMVs, skipMap: skipMap)
            applyDeblockingFilterChroma16(plane: &l2Img.cb, width: l2cbDx, height: l2cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: tMVs, skipMap: skipMap)
            applyDeblockingFilterChroma16(plane: &l2Img.cr, width: l2cbDx, height: l2cbDy, qStep: (Int(qtC2.step) + 8) >> 4, mvs: tMVs, skipMap: skipMap)
        } else {
            applyDeblockingFilter32(plane: &l2Img.y, width: l2dx, height: l2dy, qStep: (Int(qtY2.step) + 8) >> 4)
            applyDeblockingFilter16(plane: &l2Img.cb, width: l2cbDx, height: l2cbDy, qStep: (Int(qtC2.step) + 8) >> 4)
            applyDeblockingFilter16(plane: &l2Img.cr, width: l2cbDx, height: l2cbDy, qStep: (Int(qtC2.step) + 8) >> 4)
        }

        current = l2Img
    } else if hasLayer1 {
        // Layer2 absent (splitter-truncated): MC at layer1 resolution.
        if let tMVs = mvs, let tPrev = predictedPd {
            let cbDx1 = (l1dx + 1) / 2
            let cbDy1 = (l1dy + 1) / 2
            if let tNext = nextPd, let dirs = refDirs {
                await applyScaledBidirectionalMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
                await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
            } else {
                await applyScaledMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, mvs: tMVs, skipMap: skipMap, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
                await applyScaledMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, mvs: tMVs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                await applyScaledMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, mvs: tMVs, skipMap: skipMap, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
            }

            clampPlane(plane: &current.y)
            clampPlane(plane: &current.cb)
            clampPlane(plane: &current.cr)

            applyDeblockingFilterN(plane: &current.y, width: l1dx, height: l1dy, qStep: qtYStep, blockSize: 16)
            let cStep = min(qtCStep * 2, 255)
            applyDeblockingFilterN(plane: &current.cb, width: (l1dx + 1) / 2, height: (l1dy + 1) / 2, qStep: cStep, blockSize: 8)
            applyDeblockingFilterN(plane: &current.cr, width: (l1dx + 1) / 2, height: (l1dy + 1) / 2, qStep: cStep, blockSize: 8)
        }
    } else {
        // Layer0-only: MC at layer0 resolution (8x8 blocks, mvShift=2 luma,
        // mvShift=1 chroma).
        if let tMVs = mvs, let tPrev = predictedPd {
            let cbDx0 = (l0dx + 1) / 2
            let cbDy0 = (l0dy + 1) / 2
            if let tNext = nextPd, let dirs = refDirs {
                await applyScaledBidirectionalMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
                await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
                await applyScaledBidirectionalMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMVs, refDirs: dirs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
            } else {
                await applyScaledMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, mvs: tMVs, skipMap: skipMap, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
                await applyScaledMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, mvs: tMVs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
                await applyScaledMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, mvs: tMVs, skipMap: skipMap, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 1, roundOffset: roundOffset)
            }

            clampPlane(plane: &current.y)
            clampPlane(plane: &current.cb)
            clampPlane(plane: &current.cr)

            applyDeblockingFilterN(plane: &current.y, width: l0dx, height: l0dy, qStep: qtYStep, blockSize: 8)
            let cStep = min(qtCStep * 2, 255)
            applyDeblockingFilterN(plane: &current.cb, width: cbDx0, height: cbDy0, qStep: cStep, blockSize: 4)
            applyDeblockingFilterN(plane: &current.cr, width: cbDx0, height: cbDy0, qStep: cStep, blockSize: 4)
        }
    }

    // --- Stage 5: skip-block copies from the layer-matched references --------
    let targetDx = hasLayer2 ? l2dx : (hasLayer1 ? l1dx : l0dx)
    let targetDy = hasLayer2 ? l2dy : (hasLayer1 ? l1dy : l0dy)
    let targetBSize = hasLayer2 ? 32 : (hasLayer1 ? 16 : 8)

    if let map = skipMap {
        let bw = (dx + 31) / 32
        let targetCbDx = (targetDx + 1) / 2
        let targetCbDy = (targetDy + 1) / 2
        let scale = hasLayer2 ? 1 : (hasLayer1 ? 2 : 4)

        let tCbSize = targetBSize / 2
        let pPd = predictedPd ?? PlaneData420(width: dx, height: dy, y: [], cb: [], cr: [])
        let lPd = nextPd ?? pPd

        withUnsafePointers(
            lPd.y, lPd.cb, lPd.cr,
            pPd.y, pPd.cb, pPd.cr,
            mut: &current.y, mut: &current.cb, mut: &current.cr
        ) { ltrYPtr, ltrCbPtr, ltrCrPtr, prevYPtr, prevCbPtr, prevCrPtr, currYPtr, currCbPtr, currCrPtr in
            for i in 0..<map.count {
                let mode = map[i]
                if mode != .inter {
                    let bxL2 = (i % bw) * 32
                    let byL2 = (i / bw) * 32

                    let bx = bxL2 / scale
                    let by = byL2 / scale

                    if bx + targetBSize <= targetDx && by + targetBSize <= targetDy {
                        switch mode {
                        case .skip_ltr where nextPd != nil:
                            copyBlockPointer(from: ltrYPtr, to: currYPtr, bx: bx, by: by, stride: targetDx, blockSize: targetBSize)
                            copyBlockPointer(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: tCbSize)
                            copyBlockPointer(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: tCbSize)
                        case .skip_prev:
                            copyBlockPointer(from: prevYPtr, to: currYPtr, bx: bx, by: by, stride: targetDx, blockSize: targetBSize)
                            copyBlockPointer(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: tCbSize)
                            copyBlockPointer(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, stride: targetCbDx, blockSize: tCbSize)
                        default: break
                        }
                    } else {
                        switch mode {
                        case .skip_ltr where nextPd != nil:
                            copyBlockSafe(from: ltrYPtr, to: currYPtr, bx: bx, by: by, width: targetDx, height: targetDy, blockSize: targetBSize)
                            copyBlockSafe(from: ltrCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                            copyBlockSafe(from: ltrCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                        case .skip_prev:
                            copyBlockSafe(from: prevYPtr, to: currYPtr, bx: bx, by: by, width: targetDx, height: targetDy, blockSize: targetBSize)
                            copyBlockSafe(from: prevCbPtr, to: currCbPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                            copyBlockSafe(from: prevCrPtr, to: currCrPtr, bx: bx/2, by: by/2, width: targetCbDx, height: targetCbDy, blockSize: tCbSize)
                        default: break
                        }
                    }
                }
            }
        }
    }

    return current
}

@inline(__always)
fileprivate func clampPlane(plane: inout [Int16]) {
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


// Full-block reference copies, one function per block size — every caller
// knows its block size statically, so there is no size dispatch at runtime.

@inline(__always)
func copyBlock32Pointer(from src: UnsafePointer<Int16>, to dst: UnsafeMutablePointer<Int16>, bx: Int, by: Int, stride: Int) {
    for y in 0..<32 {
        let offset = (by + y) * stride + bx
        let dstPtr = UnsafeMutableRawPointer(dst.advanced(by: offset))
        let srcPtr = UnsafeRawPointer(src.advanced(by: offset))
        dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
        dstPtr.advanced(by: 32).storeBytes(of: srcPtr.advanced(by: 32).loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
    }
}

@inline(__always)
func copyBlock16Pointer(from src: UnsafePointer<Int16>, to dst: UnsafeMutablePointer<Int16>, bx: Int, by: Int, stride: Int) {
    for y in 0..<16 {
        let offset = (by + y) * stride + bx
        let dstPtr = UnsafeMutableRawPointer(dst.advanced(by: offset))
        let srcPtr = UnsafeRawPointer(src.advanced(by: offset))
        dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
    }
}

@inline(__always)
func copyBlock8Pointer(from src: UnsafePointer<Int16>, to dst: UnsafeMutablePointer<Int16>, bx: Int, by: Int, stride: Int) {
    for y in 0..<8 {
        let offset = (by + y) * stride + bx
        let dstPtr = UnsafeMutableRawPointer(dst.advanced(by: offset))
        let srcPtr = UnsafeRawPointer(src.advanced(by: offset))
        dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD8<Int16>.self), as: SIMD8<Int16>.self)
    }
}

@inline(__always)
func copyBlock4Pointer(from src: UnsafePointer<Int16>, to dst: UnsafeMutablePointer<Int16>, bx: Int, by: Int, stride: Int) {
    for y in 0..<4 {
        let offset = (by + y) * stride + bx
        dst.advanced(by: offset).update(from: src.advanced(by: offset), count: 4)
    }
}

/// Size dispatch for the generic stripped-stream pipeline only, where the
/// block size is not known until the frame header is read — the production
/// pipelines call the sized copies directly.
@inline(__always)
private func copyBlockPointer(from src: UnsafePointer<Int16>, to dst: UnsafeMutablePointer<Int16>, bx: Int, by: Int, stride: Int, blockSize: Int) {
    switch blockSize {
    case 32: copyBlock32Pointer(from: src, to: dst, bx: bx, by: by, stride: stride)
    case 16: copyBlock16Pointer(from: src, to: dst, bx: bx, by: by, stride: stride)
    case 8: copyBlock8Pointer(from: src, to: dst, bx: bx, by: by, stride: stride)
    default: copyBlock4Pointer(from: src, to: dst, bx: bx, by: by, stride: stride)
    }
}

@inline(__always)
func copyBlockSafe(from src: UnsafePointer<Int16>, to dst: UnsafeMutablePointer<Int16>, bx: Int, by: Int, width: Int, height: Int, blockSize: Int) {
    let maxY = min(by + blockSize, height)
    let maxX = min(bx + blockSize, width)
    let copyCount = maxX - bx
    if copyCount <= 0 { return }

    switch copyCount {
    case 32:
        for y in by..<maxY {
            let offset = y * width + bx
            let dstPtr = UnsafeMutableRawPointer(dst.advanced(by: offset))
            let srcPtr = UnsafeRawPointer(src.advanced(by: offset))
            dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
            dstPtr.advanced(by: 32).storeBytes(of: srcPtr.advanced(by: 32).loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
        }
    case 16:
        for y in by..<maxY {
            let offset = y * width + bx
            let dstPtr = UnsafeMutableRawPointer(dst.advanced(by: offset))
            let srcPtr = UnsafeRawPointer(src.advanced(by: offset))
            dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD16<Int16>.self), as: SIMD16<Int16>.self)
        }
    case 8:
        for y in by..<maxY {
            let offset = y * width + bx
            let dstPtr = UnsafeMutableRawPointer(dst.advanced(by: offset))
            let srcPtr = UnsafeRawPointer(src.advanced(by: offset))
            dstPtr.storeBytes(of: srcPtr.loadUnaligned(as: SIMD8<Int16>.self), as: SIMD8<Int16>.self)
        }
    default:
        for y in by..<maxY {
            let offset = y * width + bx
            dst.advanced(by: offset).update(from: src.advanced(by: offset), count: copyCount)
        }
    }
}
