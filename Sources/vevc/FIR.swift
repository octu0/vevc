@usableFromInline
let FIRLUMACoeffs: [[Int]] = [
    [0, 8, 0, 0],
    [-1, 7, 2, 0],
    [-1, 5, 5, -1],
    [0, 2, 7, -1]
]

// why: SIMD8<Int32> horizontal FIR helper for Luma 4-tap filter
// Loads 4 shifted SIMD8<Int16> vectors from row pointer, widens to Int32, multiplies by coefficients and sums
@inline(__always)
func horizontalFIRLuma8(
    _ row: UnsafePointer<Int16>, _ offset: Int,
    _ vcX0: SIMD8<Int32>, _ vcX1: SIMD8<Int32>, _ vcX2: SIMD8<Int32>, _ vcX3: SIMD8<Int32>
) -> SIMD8<Int32> {
    let s0 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(row.advanced(by: offset - 1)).loadUnaligned(as: SIMD8<Int16>.self))
    let s1 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(row.advanced(by: offset)).loadUnaligned(as: SIMD8<Int16>.self))
    let s2 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(row.advanced(by: offset + 1)).loadUnaligned(as: SIMD8<Int16>.self))
    let s3 = SIMD8<Int32>(truncatingIfNeeded: UnsafeRawPointer(row.advanced(by: offset + 2)).loadUnaligned(as: SIMD8<Int16>.self))
    return vcX0 &* s0 &+ vcX1 &* s1 &+ vcX2 &* s2 &+ vcX3 &* s3
}
