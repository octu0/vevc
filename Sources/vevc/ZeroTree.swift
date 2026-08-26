// MARK: - Zero-Tree (TreeMap) Subband Pruning (Profile 0x02)
//
// In Profile 0x02 P-frames, when all subbands of a block across all wavelet
// pyramid levels (Layer0 Base8, Layer1 16x16, Layer2 32x32) are determined
// to be effectively zero (or within safe perceptual zero thresholds), the entire
// spatial subband hierarchy is pruned as a "Zero Tree" (treez).
//
// Tree flags (1 bit per inter/non-skip block) are bit-packed into the frame
// header's treeMap area. Both encoder and decoder use this map to bypass subband
// entropy coding and reconstruction, clearing high frequencies to zero.
//
// Invariants:
// - Tree detection operates on non-skip (Inter) blocks only; skip blocks are
//   already zero-residual and excluded from treeMap bit-packing.
// - All three pyramid layers (L0, L1, L2) must be zero for treez = true.
// - Bitstream serialization order: Y plane -> Cb plane -> Cr plane (1 bit / inter block, LSB first).
import Foundation

// MARK: - Zero Detection & Pruning Functions

/// Checks if high-frequency subbands in a 32x32 block (HL, LH, HH) are effectively zero
/// within the specified threshold. If zero, clears all subband high-frequency regions.
@inline(__always)
func isEffectivelyZero32(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    let th = Int16(threshold)
    let thPos = SIMD16<Int16>(repeating: th)
    let thNeg = SIMD16<Int16>(repeating: -th)

    let lowerHalfBase = base + 16 * 32
    for i in stride(from: 0, to: 512, by: 16) {
        let vec: SIMD16<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD16<Int16>.self)
        let overPos = thPos .< vec
        let underNeg = vec .< thNeg
        if any(overPos .| underNeg) { return false }
    }
    for y in 0..<16 {
        let ptr = base + y * 32 + 16
        let vec: SIMD16<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD16<Int16>.self)
        let overPos = thPos .< vec
        let underNeg = vec .< thNeg
        if any(overPos .| underNeg) { return false }
    }

    let zeroVec = SIMD16<Int16>(repeating: 0)
    for i in stride(from: 0, to: 512, by: 16) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD16<Int16>.self)
        ptr.pointee = zeroVec
    }
    for y in 0..<16 {
        let ptr = UnsafeMutableRawPointer(base + y * 32 + 16).assumingMemoryBound(to: SIMD16<Int16>.self)
        ptr.pointee = zeroVec
    }
    return true
}

/// Checks if high-frequency subbands in a 16x16 block (HL, LH, HH) are effectively zero
/// within the specified threshold. If zero, clears all subband high-frequency regions.
@inline(__always)
func isEffectivelyZero16(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    let th = Int16(threshold)
    let thPos = SIMD8<Int16>(repeating: th)
    let thNeg = SIMD8<Int16>(repeating: -th)

    let lowerHalfBase = base + 8 * 16
    for i in stride(from: 0, to: 128, by: 8) {
        let vec: SIMD8<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD8<Int16>.self)
        let overPos = thPos .< vec
        let underNeg = vec .< thNeg
        if any(overPos .| underNeg) { return false }
    }
    for y in 0..<8 {
        let ptr = base + y * 16 + 8
        let vec: SIMD8<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        let overPos = thPos .< vec
        let underNeg = vec .< thNeg
        if any(overPos .| underNeg) { return false }
    }

    let zeroVec = SIMD8<Int16>(repeating: 0)
    for i in stride(from: 0, to: 128, by: 8) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD8<Int16>.self)
        ptr.pointee = zeroVec
    }
    for y in 0..<8 {
        let ptr = UnsafeMutableRawPointer(base + y * 16 + 8).assumingMemoryBound(to: SIMD8<Int16>.self)
        ptr.pointee = zeroVec
    }
    return true
}

/// Checks if an 8x8 base block has exact-zero LL and thresholded high-frequency bands (HL, LH, HH).
@inline(__always)
func isEffectivelyZeroBase4(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    let th = Int16(threshold)
    let thPos = SIMD4<Int16>(repeating: th)
    let thNeg = SIMD4<Int16>(repeating: -th)
    
    // Check LL band (top-left 4x4) must be exactly zero
    for y in 0..<4 {
        let ptr = base.advanced(by: y * 8)
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
        if vec[0] != 0 || vec[1] != 0 || vec[2] != 0 || vec[3] != 0 {
            return false
        }
    }
    
    // Check top-right 4x4 (HL)
    for y in 0..<4 {
        let ptr = base.advanced(by: y * 8 + 4)
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if any(mask) {
            return false
        }
    }
    
    // Check bottom half 8x4 (LH and HH)
    for y in 4..<8 {
        let ptr = base.advanced(by: y * 8)
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        let mask = (vec .> SIMD8<Int16>(repeating: th)) .| (vec .< SIMD8<Int16>(repeating: -th))
        if any(mask) {
            return false
        }
    }
    return true
}

/// Checks if an 8x8 base block in a P-frame has exact-zero LL and thresholded high-frequency bands (HL, LH, HH).
@inline(__always)
func isEffectivelyZeroBase4PFrame(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    let safeThreshold = min(8, max(0, threshold))
    let th = Int16(safeThreshold)
    let thPos = SIMD4<Int16>(repeating: th)
    let thNeg = SIMD4<Int16>(repeating: -th)
    
    // Check LL band (top-left 4x4) must be EXACTLY ZERO
    // (Because any non-zero quantized LL value carries critical base color/luma)
    for y in 0..<4 {
        let ptr = base.advanced(by: y * 8)
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
        if vec[0] != 0 || vec[1] != 0 || vec[2] != 0 || vec[3] != 0 {
            return false
        }
    }
    
    // Check top-right 4x4 (HL) with threshold
    for y in 0..<4 {
        let ptr = base.advanced(by: y * 8 + 4)
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD4<Int16>.self)
        let mask = (vec .> thPos) .| (vec .< thNeg)
        if any(mask) {
            return false
        }
    }
    
    // Check bottom half 8x4 (LH and HH) with threshold
    for y in 4..<8 {
        let ptr = base.advanced(by: y * 8)
        let vec = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD8<Int16>.self)
        let mask = (vec .> SIMD8<Int16>(repeating: th)) .| (vec .< SIMD8<Int16>(repeating: -th))
        if any(mask) {
            return false
        }
    }
    return true
}

/// Checks if a 32x32 block has exact-zero LL and thresholded subbands.
@inline(__always)
func isEffectivelyZeroBase32(data base: UnsafeMutablePointer<Int16>, threshold: Int) -> Bool {
    // Check LL
    let zeroVec16 = SIMD16<Int16>(repeating: 0)
    for y in 0..<16 {
        let ptr = base + y * 32
        let vec: SIMD16<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD16<Int16>.self)
        let mask = vec .!= zeroVec16
        if any(mask) {
            return false
        }
    }
    
    // Check Subbands
    let th = Int16(threshold)
    let thPos = SIMD16<Int16>(repeating: th)
    let thNeg = SIMD16<Int16>(repeating: -th)

    let lowerHalfBase = base + 16 * 32
    for i in stride(from: 0, to: 512, by: 16) {
        let vec: SIMD16<Int16> = UnsafeRawPointer(lowerHalfBase + i).loadUnaligned(as: SIMD16<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) {
            return false
        }
    }
    for y in 0..<16 {
        let ptr = base + y * 32 + 16
        let vec: SIMD16<Int16> = UnsafeRawPointer(ptr).loadUnaligned(as: SIMD16<Int16>.self)
        let overPos = vec .> thPos
        let underNeg = vec .< thNeg
        let mask = overPos .| underNeg
        if any(mask) {
            return false
        }
    }

    for i in stride(from: 0, to: 512, by: 16) {
        let ptr = UnsafeMutableRawPointer(lowerHalfBase + i).assumingMemoryBound(to: SIMD16<Int16>.self)
        ptr.pointee = zeroVec16
    }
    for y in 0..<16 {
        let ptr = UnsafeMutableRawPointer(base + y * 32 + 16).assumingMemoryBound(to: SIMD16<Int16>.self)
        ptr.pointee = zeroVec16
    }
    return true
}

// MARK: - Zero Flags Computation

/// Computes zero flags for an array of 32x32 blocks taking spatial weighting and skip maps into account.
@inline(__always)
func computeZeroFlags32(blocks: inout [BlockView], zeroThreshold: Int, colCount: Int, rowCount: Int, isSkip: [Bool]) -> [Bool] {
    let useSpatialWeight = 1 < colCount && 1 < rowCount
    var isZeroFlags = [Bool](repeating: true, count: blocks.count)
    for i in blocks.indices {
        if isSkip[i] {
            isZeroFlags[i] = true
            continue
        }
        let blockThreshold: Int
        if useSpatialWeight {
            let col = i % colCount
            let row = i / colCount
            let weight = spatialWeight(blockCol: col, blockRow: row, colCount: colCount, rowCount: rowCount)
            switch zeroThreshold == 0 {
            case true:
                blockThreshold = 0
            case false:
                blockThreshold = (zeroThreshold * weight) / 1024
            }
        } else {
            blockThreshold = zeroThreshold
        }
        isZeroFlags[i] = isEffectivelyZero32(data: blocks[i].base, threshold: blockThreshold)
    }
    return isZeroFlags
}

/// Computes zero flags for an array of 16x16 blocks taking spatial weighting and skip maps into account.
@inline(__always)
func computeZeroFlags16(blocks: inout [BlockView], zeroThreshold: Int, colCount: Int, rowCount: Int, isSkip: [Bool]) -> [Bool] {
    let useSpatialWeight = 1 < colCount && 1 < rowCount
    var isZeroFlags = [Bool](repeating: true, count: blocks.count)
    for i in blocks.indices {
        if isSkip[i] {
            isZeroFlags[i] = true
            continue
        }
        let blockThreshold: Int
        if useSpatialWeight {
            let col = i % colCount
            let row = i / colCount
            let weight = spatialWeight(blockCol: col, blockRow: row, colCount: colCount, rowCount: rowCount)
            switch zeroThreshold == 0 {
            case true:
                blockThreshold = 0
            case false:
                blockThreshold = (zeroThreshold * weight) / 1024
            }
        } else {
            blockThreshold = zeroThreshold
        }
        isZeroFlags[i] = isEffectivelyZero16(data: blocks[i].base, threshold: blockThreshold)
    }
    return isZeroFlags
}

/// Computes zero flags for an array of Base8 blocks taking skip maps into account.
@inline(__always)
func computeZeroFlagsBase8(blocks: [BlockView], zeroThreshold: Int, isSkip: [Bool]) -> [Bool] {
    var isZeroFlags = [Bool](repeating: true, count: blocks.count)
    for i in blocks.indices {
        if isSkip[i] {
            isZeroFlags[i] = true
            continue
        }
        isZeroFlags[i] = isEffectivelyZeroBase4PFrame(data: blocks[i].base, threshold: zeroThreshold)
    }
    return isZeroFlags
}

// MARK: - TreeMap Bitpacking (Encoder)

/// Packs zero-tree boolean flags for non-skip blocks of a single plane into a compact byte array.
@inline(__always)
func packPlaneTreeMap(isTreez: [Bool], isSkip: [Bool]) -> [UInt8] {
    var interCount = 0
    for i in 0..<isSkip.count {
        if isSkip[i] != true {
            interCount += 1
        }
    }
    if interCount == 0 {
        return []
    }
    let byteCount = (interCount + 7) / 8
    var buf = [UInt8](repeating: 0, count: byteCount)
    var bitIndex = 0
    for i in 0..<isSkip.count {
        if isSkip[i] != true {
            if isTreez[i] {
                buf[bitIndex / 8] |= UInt8(1 << (bitIndex % 8))
            }
            bitIndex += 1
        }
    }
    return buf
}

/// Serializes treeMap data for Y, Cb, and Cr planes into a single byte array for Profile 0x02 frame header.
@inline(__always)
func encodeTreeMapProfile2(
    isTreezY: [Bool], ySkip: [Bool],
    isTreezCb: [Bool], cbSkip: [Bool],
    isTreezCr: [Bool], crSkip: [Bool]
) -> [UInt8] {
    let bufY = packPlaneTreeMap(isTreez: isTreezY, isSkip: ySkip)
    let bufCb = packPlaneTreeMap(isTreez: isTreezCb, isSkip: cbSkip)
    let bufCr = packPlaneTreeMap(isTreez: isTreezCr, isSkip: crSkip)
    var out: [UInt8] = []
    out.reserveCapacity(bufY.count + bufCb.count + bufCr.count)
    out.append(contentsOf: bufY)
    out.append(contentsOf: bufCb)
    out.append(contentsOf: bufCr)
    return out
}

// MARK: - TreeMap Unpacking (Decoder)

/// Unpacks zero-tree boolean flags for non-skip blocks of a single plane from the frame header buffer.
@inline(__always)
func unpackPlaneTreeMap(buf: ArraySlice<UInt8>, offset: inout Int, count: Int, isSkip: [Bool]?) -> [Bool] {
    var isTreez = [Bool](repeating: false, count: count)
    guard let isSkip = isSkip else {
        return isTreez
    }
    var interCount = 0
    for i in 0..<count {
        if isSkip[i] != true {
            interCount += 1
        }
    }
    if interCount == 0 {
        return isTreez
    }
    let byteCount = (interCount + 7) / 8
    let endOffset = offset + byteCount
    var bitIndex = 0
    let startIdx = buf.startIndex
    for i in 0..<count {
        if isSkip[i] != true {
            let byteIdx = offset + (bitIndex / 8)
            let bitIdx = bitIndex % 8
            if byteIdx < buf.count {
                if (buf[startIdx + byteIdx] & UInt8(1 << bitIdx)) != 0 {
                    isTreez[i] = true
                }
            }
            bitIndex += 1
        }
    }
    offset = endOffset
    return isTreez
}

@inline(__always)
func unpackPlaneTreeMap(buf: [UInt8], offset: inout Int, count: Int, isSkip: [Bool]?) -> [Bool] {
    return unpackPlaneTreeMap(buf: buf[...], offset: &offset, count: count, isSkip: isSkip)
}

/// Decodes treeMap data for Y, Cb, and Cr planes from the frame header buffer for Profile 0x02.
@inline(__always)
func decodeTreeMapProfile2(
    buf: ArraySlice<UInt8>,
    yCount: Int, ySkip: [Bool]?,
    cbCount: Int, cbSkip: [Bool]?,
    crCount: Int, crSkip: [Bool]?
) -> (isTreezY: [Bool], isTreezCb: [Bool], isTreezCr: [Bool]) {
    var offset = 0
    let tzY = unpackPlaneTreeMap(buf: buf, offset: &offset, count: yCount, isSkip: ySkip)
    let tzCb = unpackPlaneTreeMap(buf: buf, offset: &offset, count: cbCount, isSkip: cbSkip)
    let tzCr = unpackPlaneTreeMap(buf: buf, offset: &offset, count: crCount, isSkip: crSkip)
    return (tzY, tzCb, tzCr)
}

@inline(__always)
func decodeTreeMapProfile2(
    buf: [UInt8],
    yCount: Int, ySkip: [Bool]?,
    cbCount: Int, cbSkip: [Bool]?,
    crCount: Int, crSkip: [Bool]?
) -> (isTreezY: [Bool], isTreezCb: [Bool], isTreezCr: [Bool]) {
    return decodeTreeMapProfile2(buf: buf[...], yCount: yCount, ySkip: ySkip, cbCount: cbCount, cbSkip: cbSkip, crCount: crCount, crSkip: crSkip)
}
