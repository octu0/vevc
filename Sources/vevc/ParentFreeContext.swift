// MARK: - Parent-free entropy contexts (Profile 0x02, One-Pyramid §6)
//
// NORMATIVE for profile 0x02: AC coefficient contexts do not read the parent
// layer. Measured against the shipped backward-adaptive tables on
// miko1/ToS @500k/2500k, dropping the parent term SHRINKS the coefficient
// streams on every configuration (+0.17%…+1.52%): with header-free adaptive
// tables, fewer contexts warm up faster and the parent bit is redundant with
// the previous-coefficient feature. Removing the dependency also frees the
// three layers to entropy-decode in parallel and lets skip blocks bypass
// upper layers entirely (design doc §5/§6) — parents are no longer read.
//
// Implementation: getContext(prevVal:isParentZero:) collapses to its
// isParentZero==false half (contexts {0,1}) when every parent lookup sees a
// nonzero value. Profile 0x02 therefore feeds the shared subband coders these
// all-ones dummy parent blocks instead of real parent blocks; profile 0x01
// keeps real parents and its bitstreams are untouched. The parent plumbing
// itself is removed together with the 9-stream restructure (Wave 2), which
// rewrites these call paths anyway.

private struct ParentFreeDummies: @unchecked Sendable {
    let block16: BlockView
    let block8: BlockView
}

private let parentFreeDummies: ParentFreeDummies = {
    let b16 = BlockView.allocate(width: 16, height: 16)
    let b8 = BlockView.allocate(width: 8, height: 8)
    b16.base.update(repeating: 1, count: 16 * b16.stride)
    b8.base.update(repeating: 1, count: 8 * b8.stride)
    return ParentFreeDummies(block16: b16, block8: b8)
}()

/// Dummy parents for layer2 coding (replaces the layer1 16x16 blocks).
@inline(__always)
func parentFreeParents16(count: Int) -> [BlockView] {
    [BlockView](repeating: parentFreeDummies.block16, count: count)
}

/// Dummy parents for layer1 coding (replaces the Base8 8x8 blocks).
@inline(__always)
func parentFreeParents8(count: Int) -> [BlockView] {
    [BlockView](repeating: parentFreeDummies.block8, count: count)
}
