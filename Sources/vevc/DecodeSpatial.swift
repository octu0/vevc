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

    // encodeSpatialLayers appends layer0, then layer1, then layer2.
    var mvs: [MotionVector]? = nil
    var refDirs: [Bool]? = nil
    let mvCount = Int(try readUInt32BEFromBytes(r, offset: &offset))
    let mvDataLen = Int(try readUInt32BEFromBytes(r, offset: &offset))
    
    if 0 < mvCount && 0 < mvDataLen {
        guard (offset + mvDataLen) <= r.count else { throw DecodeError.insufficientData }
        
        mvs = try decodeMVs(data: Array(r[offset..<(offset + mvDataLen)]), count: mvCount)
        offset += mvDataLen
    }
    
    // direction flag only exists for bidirectional prediction frames
    // indicates whether each block uses forward (prev) or backward (next) reference
    if 0 < mvCount && nextPd != nil {
        let refDirByteCount = Int(try readUInt32BEFromBytes(r, offset: &offset))
        if 0 < refDirByteCount {
            guard (offset + refDirByteCount) <= r.count else { throw DecodeError.insufficientData }
            let refDirBuf = Array(r[offset..<(offset + refDirByteCount)])
            offset += refDirByteCount
            
            var dirs = [Bool]()
            dirs.reserveCapacity(mvCount)
            for i in 0..<mvCount {
                let byteIdx = i / 8
                let bitIdx = i % 8
                let isBackward = (byteIdx < refDirBuf.count) && ((refDirBuf[byteIdx] & UInt8(1 << bitIdx)) != 0)
                dirs.append(isBackward)
            }
            refDirs = dirs
        }
    }
    
    let len0 = try readUInt32BEFromBytes(r, offset: &offset)
    guard (offset + Int(len0)) <= r.count else { throw DecodeError.insufficientData }
    let layer0Data = Array(r[offset..<(offset + Int(len0))])
    offset += Int(len0)
    
    // Base layer (layer 0) is always Base8
    let (baseImg, base8YBlocks, base8CbBlocks, base8CrBlocks) = try await decodeBase8(r: layer0Data, pool: pool, layer: 0, dx: l0dx, dy: l0dy, isIFrame: (mvCount == 0))
    var current = baseImg
    var parentYBlocks: [BlockView]? = base8YBlocks
    var parentCbBlocks: [BlockView]? = base8CbBlocks
    var parentCrBlocks: [BlockView]? = base8CrBlocks
    
    if 1 <= maxLayer {
        let len1 = try readUInt32BEFromBytes(r, offset: &offset)
        guard (offset + Int(len1)) <= r.count else { throw DecodeError.insufficientData }
        let layer1Data = Array(r[offset..<(offset + Int(len1))])
        offset += Int(len1)
        
        let (l16Img, l16YBlocks, l16CbBlocks, l16CrBlocks) = try await decodeLayer16(r: layer1Data, pool: pool, layer: 1, dx: l1dx, dy: l1dy, prev: current, parentYBlocks: parentYBlocks, parentCbBlocks: parentCbBlocks, parentCrBlocks: parentCrBlocks)
        current = l16Img
        parentYBlocks = l16YBlocks
        parentCbBlocks = l16CbBlocks
        parentCrBlocks = l16CrBlocks
    }
    
    if 2 <= maxLayer {
        let len2 = try readUInt32BEFromBytes(r, offset: &offset)
        guard (offset + Int(len2)) <= r.count else { throw DecodeError.insufficientData }
        let layer2Data = Array(r[offset..<(offset + Int(len2))])
        offset += Int(len2)
        
        current = try await decodeLayer32(r: layer2Data, pool: pool, layer: 2, dx: l2dx, dy: l2dy, prev: current, parentYBlocks: parentYBlocks, parentCbBlocks: parentCbBlocks, parentCrBlocks: parentCrBlocks, predictedPd: predictedPd, nextPd: nextPd, mvs: mvs, refDirs: refDirs, roundOffset: roundOffset)
    }
    
    return current
}
