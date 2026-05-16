// MARK: - Decode Spatial

@inline(__always)
func decodeSpatialLayers(r: [UInt8], pool: BlockViewPool, maxLayer: Int, dx: Int, dy: Int, predictedPd: PlaneData420? = nil, nextPd: PlaneData420? = nil, roundOffset: Int) async throws -> Image16 {
    var offset = 0

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

    let frameHeader = try VEVCFrameHeader.deserialize(from: r, offset: &offset)
    if frameHeader.isCopyFrame {
        throw DecodeError.insufficientDataContext("decodeSpatialLayers passed copy frame")
    }
    
    var mvs: [MotionVector]? = nil
    var refDirs: [Bool]? = nil
    
    if 0 < frameHeader.mvsCount && 0 < frameHeader.mvsSize {
        guard (offset + frameHeader.mvsSize) <= r.count else { throw DecodeError.insufficientData }
        mvs = try decodeMVs(data: Array(r[offset..<(offset + frameHeader.mvsSize)]), count: frameHeader.mvsCount)
        offset += frameHeader.mvsSize
    }
    
    // Direction flag only exists for bidirectional prediction frames.
    // Indicates whether each block uses forward (prev) or backward (next) reference.
    if 0 < frameHeader.mvsCount && nextPd != nil && 0 < frameHeader.refDirSize {
        guard (offset + frameHeader.refDirSize) <= r.count else { throw DecodeError.insufficientData }
        let refDirBuf = Array(r[offset..<(offset + frameHeader.refDirSize)])
        offset += frameHeader.refDirSize
        
        var dirs = [Bool]()
        dirs.reserveCapacity(frameHeader.mvsCount)
        for i in 0..<frameHeader.mvsCount {
            let byteIdx = i / 8
            let bitIdx = i % 8
            let isBackward = (byteIdx < refDirBuf.count) && ((refDirBuf[byteIdx] & UInt8(1 << bitIdx)) != 0)
            dirs.append(isBackward)
        }
        refDirs = dirs
    }
    
    guard (offset + frameHeader.layer0Size) <= r.count else { throw DecodeError.insufficientData }
    let layer0Data = Array(r[offset..<(offset + frameHeader.layer0Size)])
    offset += frameHeader.layer0Size
    
    // Base layer (layer 0) is always Base8
    let (baseImg, base8YBlocks, base8CbBlocks, base8CrBlocks) = try await decodeBase8(r: layer0Data, pool: pool, layer: 0, dx: l0dx, dy: l0dy, isIFrame: (frameHeader.mvsCount == 0))
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
        
        let (l16Img, l16YBlocks, l16CbBlocks, l16CrBlocks) = try await decodeLayer16(r: layer1Data, pool: pool, layer: 1, dx: l1dx, dy: l1dy, prev: current, parentYBlocks: parentYBlocks, parentCbBlocks: parentCbBlocks, parentCrBlocks: parentCrBlocks)
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
        
        current = try await decodeLayer32(r: layer2Data, pool: pool, layer: 2, dx: l2dx, dy: l2dy, prev: current, parentYBlocks: parentYBlocks, parentCbBlocks: parentCbBlocks, parentCrBlocks: parentCrBlocks, predictedPd: predictedPd, nextPd: nextPd, mvs: mvs, refDirs: refDirs, roundOffset: roundOffset)
    } else {
        offset += frameHeader.layer2Size
        
        // Layer2 data is absent: apply MC at the highest available layer's resolution.
        // This handles the case where the splitter stripped upper layers.
        if hasLayer1 {
            // Apply MC at layer1 resolution (16x16 blocks, mvShift=1)
            if let tMvs = mvs, let tPrev = predictedPd {
                let cbDx1 = (l1dx + 1) / 2
                let cbDy1 = (l1dy + 1) / 2
                if let tNext = nextPd, let dirs = refDirs {
                    applyScaledBidirectionalMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMvs, refDirs: dirs, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
                    applyScaledBidirectionalMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMvs, refDirs: dirs, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                    applyScaledBidirectionalMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMvs, refDirs: dirs, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                } else {
                    applyScaledMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, mvs: tMvs, width: l1dx, height: l1dy, lumaBlockSize: 16, mvShift: 1, roundOffset: roundOffset)
                    applyScaledMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, mvs: tMvs, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                    applyScaledMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, mvs: tMvs, width: cbDx1, height: cbDy1, chromaBlockSize: 8, mvShift: 1, roundOffset: roundOffset)
                }
                applyDeblockingFilter16(plane: &current.y, width: l1dx, height: l1dy, qStep: 1)
                applyDeblockingFilter16(plane: &current.cb, width: cbDx1, height: cbDy1, qStep: 1)
                applyDeblockingFilter16(plane: &current.cr, width: cbDx1, height: cbDy1, qStep: 1)
            }
        } else {
            // Apply MC at layer0 resolution (8x8 blocks, mvShift=2)
            if let tMvs = mvs, let tPrev = predictedPd {
                let cbDx0 = (l0dx + 1) / 2
                let cbDy0 = (l0dy + 1) / 2
                if let tNext = nextPd, let dirs = refDirs {
                    applyScaledBidirectionalMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, nextPlane: tNext.y, mvs: tMvs, refDirs: dirs, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
                    applyScaledBidirectionalMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, nextPlane: tNext.cb, mvs: tMvs, refDirs: dirs, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 2, roundOffset: roundOffset)
                    applyScaledBidirectionalMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, nextPlane: tNext.cr, mvs: tMvs, refDirs: dirs, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 2, roundOffset: roundOffset)
                } else {
                    applyScaledMotionCompensationLuma(plane: &current.y, prevPlane: tPrev.y, mvs: tMvs, width: l0dx, height: l0dy, lumaBlockSize: 8, mvShift: 2, roundOffset: roundOffset)
                    applyScaledMotionCompensationChroma(plane: &current.cb, prevPlane: tPrev.cb, mvs: tMvs, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 2, roundOffset: roundOffset)
                    applyScaledMotionCompensationChroma(plane: &current.cr, prevPlane: tPrev.cr, mvs: tMvs, width: cbDx0, height: cbDy0, chromaBlockSize: 4, mvShift: 2, roundOffset: roundOffset)
                }
                applyDeblockingFilter32(plane: &current.y, width: l0dx, height: l0dy, qStep: 1)
                applyDeblockingFilter32(plane: &current.cb, width: cbDx0, height: cbDy0, qStep: 1)
                applyDeblockingFilter32(plane: &current.cr, width: cbDx0, height: cbDy0, qStep: 1)
            }
        }
    }
    
    return current
}
