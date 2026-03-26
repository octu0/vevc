// MARK: - Temporal DWT
// Applies LeGall 5/3 wavelet transform along the temporal axis (across frames).
// Each pixel position (x,y) is treated as a 1D signal across GOP frames.
// SIMD8-optimized: processes 8 pixels simultaneously for maximum throughput.

import Foundation

// MARK: - Error

enum TemporalDWTError: Error {
    case invalidFrameCount(expected: Int, actual: Int)
    case invalidSubbandCount(expected: Int, actualLow: Int, actualHigh: Int)
}

/// Result of temporal forward DWT: temporal low-frequency and high-frequency frames.
struct TemporalSubbands {
    let low: [PlaneData420]   // N/2 temporal low-frequency frames
    let high: [PlaneData420]  // N/2 temporal high-frequency frames
}

// MARK: - SIMD8 Load/Store helpers

@inline(__always)
private func loadSIMD8(_ ptr: UnsafePointer<Int16>, offset: Int) -> SIMD8<Int16> {
    return UnsafeRawPointer(ptr + offset).load(as: SIMD8<Int16>.self)
}

@inline(__always)
private func storeSIMD8(_ ptr: UnsafeMutablePointer<Int16>, offset: Int, value: SIMD8<Int16>) {
    UnsafeMutableRawPointer(ptr + offset).storeBytes(of: value, as: SIMD8<Int16>.self)
}

// MARK: - SIMD8 Temporal Lifting (Forward)

/// SIMD8-optimized forward temporal lift for a single plane.
@inline(__always)
private func temporalLift4PlaneSIMD(
    src0: UnsafePointer<Int16>, src1: UnsafePointer<Int16>,
    src2: UnsafePointer<Int16>, src3: UnsafePointer<Int16>,
    lo0: UnsafeMutablePointer<Int16>, lo1: UnsafeMutablePointer<Int16>,
    hi0: UnsafeMutablePointer<Int16>, hi1: UnsafeMutablePointer<Int16>,
    count: Int,
) {
    let simdCount = count & ~7  // round down to multiple of 8
    let two = SIMD8<Int16>(repeating: 2)
    
    // SIMD8 main loop: 8 pixels per iteration
    var i = 0
    while i < simdCount {
        let even0 = loadSIMD8(src0, offset: i)  // frame 0
        let odd0  = loadSIMD8(src1, offset: i)  // frame 1
        let even1 = loadSIMD8(src2, offset: i)  // frame 2
        let odd1  = loadSIMD8(src3, offset: i)  // frame 3
        
        // Predict: H[n] -= (L[n] + L[n+1]) >> 1
        let h0 = odd0 &- ((even0 &+ even1) &>> 1)
        let h1 = odd1 &- ((even1 &+ even1) &>> 1)  // boundary mirror
        
        // Update: L[n] += (H[n-1] + H[n] + 2) >> 2
        let l0 = even0 &+ ((h0 &+ h0 &+ two) &>> 2)  // boundary mirror
        let l1 = even1 &+ ((h0 &+ h1 &+ two) &>> 2)
        
        storeSIMD8(lo0, offset: i, value: l0)
        storeSIMD8(lo1, offset: i, value: l1)
        storeSIMD8(hi0, offset: i, value: h0)
        storeSIMD8(hi1, offset: i, value: h1)
        i += 8
    }
    
    // Scalar tail for remaining pixels
    while i < count {
        let even0 = src0[i]
        let odd0  = src1[i]
        let even1 = src2[i]
        let odd1  = src3[i]
        
        let high0Val = odd0 &- ((even0 &+ even1) &>> 1)
        let high1Val = odd1 &- ((even1 &+ even1) &>> 1)
        
        lo0[i] = even0 &+ ((high0Val &+ high0Val &+ 2) &>> 2)
        lo1[i] = even1 &+ ((high0Val &+ high1Val &+ 2) &>> 2)
        hi0[i] = high0Val
        hi1[i] = high1Val
        i += 1
    }
}

// MARK: - SIMD8 Temporal Lifting (Inverse)

/// SIMD8-optimized inverse temporal lift for a single plane.
@inline(__always)
private func temporalInvLift4PlaneSIMD(
    lo0: UnsafePointer<Int16>, lo1: UnsafePointer<Int16>,
    hi0: UnsafePointer<Int16>, hi1: UnsafePointer<Int16>,
    out0: UnsafeMutablePointer<Int16>, out1: UnsafeMutablePointer<Int16>,
    out2: UnsafeMutablePointer<Int16>, out3: UnsafeMutablePointer<Int16>,
    count: Int,
) {
    let simdCount = count & ~7
    let two = SIMD8<Int16>(repeating: 2)
    
    var i = 0
    while i < simdCount {
        var low0 = loadSIMD8(lo0, offset: i)
        var low1 = loadSIMD8(lo1, offset: i)
        let high0 = loadSIMD8(hi0, offset: i)
        let high1 = loadSIMD8(hi1, offset: i)
        
        // Inverse Update: L[n] -= (H[n-1] + H[n] + 2) >> 2
        low0 &-= ((high0 &+ high0 &+ two) &>> 2)  // boundary mirror
        low1 &-= ((high0 &+ high1 &+ two) &>> 2)
        
        // Inverse Predict: H[n] += (L[n] + L[n+1]) >> 1
        let odd0 = high0 &+ ((low0 &+ low1) &>> 1)
        let odd1 = high1 &+ ((low1 &+ low1) &>> 1)  // boundary mirror
        
        // Store interleaved: [even0, odd0, even1, odd1]
        storeSIMD8(out0, offset: i, value: low0)
        storeSIMD8(out1, offset: i, value: odd0)
        storeSIMD8(out2, offset: i, value: low1)
        storeSIMD8(out3, offset: i, value: odd1)
        i += 8
    }
    
    // Scalar tail
    while i < count {
        var low0Val = lo0[i]
        var low1Val = lo1[i]
        let high0Val = hi0[i]
        let high1Val = hi1[i]
        
        low0Val &-= ((high0Val &+ high0Val &+ 2) &>> 2)
        low1Val &-= ((high0Val &+ high1Val &+ 2) &>> 2)
        
        out0[i] = low0Val
        out1[i] = high0Val &+ ((low0Val &+ low1Val) &>> 1)
        out2[i] = low1Val
        out3[i] = high1Val &+ ((low1Val &+ low1Val) &>> 1)
        i += 1
    }
}

// MARK: - Plane-level wrapper

/// Forward temporal DWT for a single plane using SIMD8.
@inline(__always)
private func temporalForwardPlane(
    src0: [Int16], src1: [Int16], src2: [Int16], src3: [Int16],
    size: Int,
) -> (lo0: [Int16], lo1: [Int16], hi0: [Int16], hi1: [Int16]) {
    var lo0 = [Int16](repeating: 0, count: size)
    var lo1 = [Int16](repeating: 0, count: size)
    var hi0 = [Int16](repeating: 0, count: size)
    var hi1 = [Int16](repeating: 0, count: size)
    
    withUnsafePointers(
        src0, src1, src2, src3,
        mut: &lo0, mut: &lo1, mut: &hi0, mut: &hi1,
    ) { pSrc0, pSrc1, pSrc2, pSrc3, pLo0, pLo1, pHi0, pHi1 in
        temporalLift4PlaneSIMD(
            src0: pSrc0, src1: pSrc1,
            src2: pSrc2, src3: pSrc3,
            lo0: pLo0, lo1: pLo1,
            hi0: pHi0, hi1: pHi1,
            count: size,
        )
    }
    return (lo0, lo1, hi0, hi1)
}

/// Inverse temporal DWT for a single plane using SIMD8.
@inline(__always)
private func temporalInversePlane(
    lo0: [Int16], lo1: [Int16], hi0: [Int16], hi1: [Int16],
    size: Int,
) -> (out0: [Int16], out1: [Int16], out2: [Int16], out3: [Int16]) {
    var out0 = [Int16](repeating: 0, count: size)
    var out1 = [Int16](repeating: 0, count: size)
    var out2 = [Int16](repeating: 0, count: size)
    var out3 = [Int16](repeating: 0, count: size)
    
    withUnsafePointers(
        lo0, lo1, hi0, hi1,
        mut: &out0, mut: &out1, mut: &out2, mut: &out3,
    ) { pLo0, pLo1, pHi0, pHi1, pOut0, pOut1, pOut2, pOut3 in
        temporalInvLift4PlaneSIMD(
            lo0: pLo0, lo1: pLo1,
            hi0: pHi0, hi1: pHi1,
            out0: pOut0, out1: pOut1,
            out2: pOut2, out3: pOut3,
            count: size,
        )
    }
    return (out0, out1, out2, out3)
}

// MARK: - Public API

/// Forward temporal DWT on 4 frames using LeGall 5/3.
func temporalForwardDWT4(frames: [PlaneData420]) throws -> TemporalSubbands {
    guard frames.count == 4 else {
        throw TemporalDWTError.invalidFrameCount(expected: 4, actual: frames.count)
    }
    
    let width = frames[0].width
    let height = frames[0].height
    let chromaWidth = (width + 1) / 2
    let chromaHeight = (height + 1) / 2
    let ySize = width * height
    let cSize = chromaWidth * chromaHeight
    
    // Y plane
    let (yL0, yL1, yH0, yH1) = temporalForwardPlane(
        src0: frames[0].y, src1: frames[1].y,
        src2: frames[2].y, src3: frames[3].y,
        size: ySize,
    )
    
    // Cb plane
    let (cbL0, cbL1, cbH0, cbH1) = temporalForwardPlane(
        src0: frames[0].cb, src1: frames[1].cb,
        src2: frames[2].cb, src3: frames[3].cb,
        size: cSize,
    )
    
    // Cr plane
    let (crL0, crL1, crH0, crH1) = temporalForwardPlane(
        src0: frames[0].cr, src1: frames[1].cr,
        src2: frames[2].cr, src3: frames[3].cr,
        size: cSize,
    )
    
    let lowFrames = [
        PlaneData420(width: width, height: height, y: yL0, cb: cbL0, cr: crL0),
        PlaneData420(width: width, height: height, y: yL1, cb: cbL1, cr: crL1),
    ]
    let highFrames = [
        PlaneData420(width: width, height: height, y: yH0, cb: cbH0, cr: crH0),
        PlaneData420(width: width, height: height, y: yH1, cb: cbH1, cr: crH1),
    ]
    
    return TemporalSubbands(low: lowFrames, high: highFrames)
}

/// Inverse temporal DWT on 2 low + 2 high frames to reconstruct 4 original frames.
func temporalInverseDWT4(subbands: TemporalSubbands) throws -> [PlaneData420] {
    guard subbands.low.count == 2, subbands.high.count == 2 else {
        throw TemporalDWTError.invalidSubbandCount(
            expected: 2,
            actualLow: subbands.low.count,
            actualHigh: subbands.high.count,
        )
    }
    
    let width = subbands.low[0].width
    let height = subbands.low[0].height
    let chromaWidth = (width + 1) / 2
    let chromaHeight = (height + 1) / 2
    let ySize = width * height
    let cSize = chromaWidth * chromaHeight
    
    // Y plane
    let (yO0, yO1, yO2, yO3) = temporalInversePlane(
        lo0: subbands.low[0].y, lo1: subbands.low[1].y,
        hi0: subbands.high[0].y, hi1: subbands.high[1].y,
        size: ySize,
    )
    
    // Cb plane
    let (cbO0, cbO1, cbO2, cbO3) = temporalInversePlane(
        lo0: subbands.low[0].cb, lo1: subbands.low[1].cb,
        hi0: subbands.high[0].cb, hi1: subbands.high[1].cb,
        size: cSize,
    )
    
    // Cr plane
    let (crO0, crO1, crO2, crO3) = temporalInversePlane(
        lo0: subbands.low[0].cr, lo1: subbands.low[1].cr,
        hi0: subbands.high[0].cr, hi1: subbands.high[1].cr,
        size: cSize,
    )
    
    return [
        PlaneData420(width: width, height: height, y: yO0, cb: cbO0, cr: crO0),
        PlaneData420(width: width, height: height, y: yO1, cb: cbO1, cr: crO1),
        PlaneData420(width: width, height: height, y: yO2, cb: cbO2, cr: crO2),
        PlaneData420(width: width, height: height, y: yO3, cb: cbO3, cr: crO3),
    ]
}
