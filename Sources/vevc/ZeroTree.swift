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
            if zeroThreshold == 0 {
                blockThreshold = 0
            } else {
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
            if zeroThreshold == 0 {
                blockThreshold = 0
            } else {
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
