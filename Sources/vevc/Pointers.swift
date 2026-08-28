/// Borrowed Y/Cb/Cr base pointers of one PlaneData420. Only valid inside the
/// withUnsafePlanePointers call that produced it.
internal struct PlanePointers {
    internal let y: UnsafePointer<Int16>
    internal let cb: UnsafePointer<Int16>
    internal let cr: UnsafePointer<Int16>
}

@inline(__always)
internal func withUnsafePlanePointers<R>(
    _ a: PlaneData420,
    _ body: (PlanePointers) throws -> R
) rethrows -> R {
    try a.y.withUnsafeBufferPointer { aY in
        try a.cb.withUnsafeBufferPointer { aCb in
            try a.cr.withUnsafeBufferPointer { aCr in
                try body(PlanePointers(y: aY.baseAddress!, cb: aCb.baseAddress!, cr: aCr.baseAddress!))
            }
        }
    }
}

@inline(__always)
internal func withUnsafePlanePointers<R>(
    _ a: PlaneData420, _ b: PlaneData420,
    _ body: (PlanePointers, PlanePointers) throws -> R
) rethrows -> R {
    try withUnsafePlanePointers(a) { pA in
        try withUnsafePlanePointers(b) { pB in
            try body(pA, pB)
        }
    }
}

@inline(__always)
internal func withUnsafePlanePointers<R>(
    _ a: PlaneData420, _ b: PlaneData420, _ c: PlaneData420,
    _ body: (PlanePointers, PlanePointers, PlanePointers) throws -> R
) rethrows -> R {
    try withUnsafePlanePointers(a) { pA in
        try withUnsafePlanePointers(b) { pB in
            try withUnsafePlanePointers(c) { pC in
                try body(pA, pB, pC)
            }
        }
    }
}

@inline(__always)
internal func withUnsafePlanePointers<R>(
    _ a: PlaneData420, _ b: PlaneData420, _ c: PlaneData420, _ d: PlaneData420,
    _ body: (PlanePointers, PlanePointers, PlanePointers, PlanePointers) throws -> R
) rethrows -> R {
    try withUnsafePlanePointers(a) { pA in
        try withUnsafePlanePointers(b) { pB in
            try withUnsafePlanePointers(c) { pC in
                try withUnsafePlanePointers(d) { pD in
                    try body(pA, pB, pC, pD)
                }
            }
        }
    }
}

internal struct UnsafeSendablePointer<T>: @unchecked Sendable {
    internal let ptr: UnsafePointer<T>
}

internal struct UnsafeSendableMutablePointer<T>: @unchecked Sendable {
    internal let ptr: UnsafeMutablePointer<T>
}

/// Escaped array view for @Sendable hot-loop closures: element access through
/// a captured Array would emit retain/release traffic per subscript; through
/// this buffer it is a plain load. The array must outlive all uses (function
/// parameters are kept alive for the whole call).
internal struct UnsafeSendableBufferPointer<T>: @unchecked Sendable {
    internal let ptr: UnsafeBufferPointer<T>
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T],
    _ body: (UnsafePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try body(pA.baseAddress!)
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    mut a: inout [T],
    _ body: (UnsafeMutablePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeMutableBufferPointer { pA in
        try body(pA.baseAddress!)
    }
}

@inline(__always)
internal func withUnsafePointers<T1, T2, R>(
    mut a: inout [T1], mut b: inout [T2],
    _ body: (UnsafeMutablePointer<T1>, UnsafeMutablePointer<T2>) throws -> R
) rethrows -> R {
    try a.withUnsafeMutableBufferPointer { pA in
        try b.withUnsafeMutableBufferPointer { pB in
            try body(pA.baseAddress!, pB.baseAddress!)
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    mut a: inout [T], mut b: inout [T], mut c: inout [T],
    _ body: (UnsafeMutablePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeMutableBufferPointer { pA in
        try b.withUnsafeMutableBufferPointer { pB in
            try c.withUnsafeMutableBufferPointer { pC in
                try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!)
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T], _ b: [T],
    _ body: (UnsafePointer<T>, UnsafePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try body(pA.baseAddress!, pB.baseAddress!)
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T1, T2, T3, R>(
    _ a: [T1], _ b: [T2], _ c: [T3],
    _ body: (UnsafePointer<T1>, UnsafePointer<T2>, UnsafePointer<T3>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!)
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T], _ b: [T], _ c: [T], _ d: [T],
    _ body: (UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try d.withUnsafeBufferPointer { pD in
                    try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!)
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T],
    mut b: inout [T],
    _ body: (UnsafePointer<T>, UnsafeMutablePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeMutableBufferPointer { pB in
            try body(pA.baseAddress!, pB.baseAddress!)
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T],
    mut b: inout [T], mut c: inout [T],
    _ body: (UnsafePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeMutableBufferPointer { pB in
            try c.withUnsafeMutableBufferPointer { pC in
                try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!)
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T],
    mut b: inout [T], mut c: inout [T], mut d: inout [T],
    _ body: (UnsafePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeMutableBufferPointer { pB in
            try c.withUnsafeMutableBufferPointer { pC in
                try d.withUnsafeMutableBufferPointer { pD in
                    try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!)
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T], _ b: [T],
    mut c: inout [T], mut d: inout [T],
    _ body: (UnsafePointer<T>, UnsafePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeMutableBufferPointer { pC in
                try d.withUnsafeMutableBufferPointer { pD in
                    try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!)
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T], _ b: [T], _ c: [T], _ d: [T],
    mut e: inout [T], mut f: inout [T],
    _ body: (UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try d.withUnsafeBufferPointer { pD in
                    try e.withUnsafeMutableBufferPointer { pE in
                        try f.withUnsafeMutableBufferPointer { pF in
                            try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!, pE.baseAddress!, pF.baseAddress!)
                        }
                    }
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T], _ b: [T], _ c: [T],
    mut d: inout [T], mut e: inout [T], mut f: inout [T], mut g: inout [T],
    _ body: (UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try d.withUnsafeMutableBufferPointer { pD in
                    try e.withUnsafeMutableBufferPointer { pE in
                        try f.withUnsafeMutableBufferPointer { pF in
                            try g.withUnsafeMutableBufferPointer { pG in
                                try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!, pE.baseAddress!, pF.baseAddress!, pG.baseAddress!)
                            }
                        }
                    }
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T], _ b: [T], _ c: [T],
    mut d: inout [T],
    _ body: (UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>, UnsafeMutablePointer<T>) throws -> R,
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try d.withUnsafeMutableBufferPointer { pD in
                    try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!)
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T], _ b: [T], _ c: [T], _ d: [T],
    mut e: inout [T], mut f: inout [T], mut g: inout [T], mut h: inout [T],
    _ body: (
        UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>,
        UnsafeMutablePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>
    ) throws -> R,
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try d.withUnsafeBufferPointer { pD in
                    try e.withUnsafeMutableBufferPointer { pE in
                        try f.withUnsafeMutableBufferPointer { pF in
                            try g.withUnsafeMutableBufferPointer { pG in
                                try h.withUnsafeMutableBufferPointer { pH in
                                    try body(
                                        pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!,
                                        pE.baseAddress!, pF.baseAddress!, pG.baseAddress!, pH.baseAddress!,
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T], _ b: [T],
    mut c: inout [T],
    _ body: (UnsafePointer<T>, UnsafePointer<T>, UnsafeMutablePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeMutableBufferPointer { pC in
                try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!)
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T1, T2, T3, R>(
    mut a: inout [T1], mut b: inout [T2], mut c: inout [T3],
    _ body: (UnsafeMutablePointer<T1>, UnsafeMutablePointer<T2>, UnsafeMutablePointer<T3>) throws -> R
) rethrows -> R {
    try a.withUnsafeMutableBufferPointer { pA in
        try b.withUnsafeMutableBufferPointer { pB in
            try c.withUnsafeMutableBufferPointer { pC in
                try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!)
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T1, T2, R>(
    _ a: [T1],
    mut b: inout [T2],
    _ body: (UnsafePointer<T1>, UnsafeMutablePointer<T2>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeMutableBufferPointer { pB in
            try body(pA.baseAddress!, pB.baseAddress!)
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T1, T2, R>(
    mut a: inout [T1],
    _ b: [T2],
    _ body: (UnsafeMutablePointer<T1>, UnsafePointer<T2>) throws -> R
) rethrows -> R {
    try a.withUnsafeMutableBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try body(pA.baseAddress!, pB.baseAddress!)
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T1, T2, T3, R>(
    _ a: [T1], _ b: [T2],
    mut c: inout [T3],
    _ body: (UnsafePointer<T1>, UnsafePointer<T2>, UnsafeMutablePointer<T3>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeMutableBufferPointer { pC in
                try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!)
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T1, T2, T3, T4, R>(
    mut a: inout [T1], mut b: inout [T2],
    _ c: [T3], _ d: [T4],
    _ body: (UnsafeMutablePointer<T1>, UnsafeMutablePointer<T2>, UnsafePointer<T3>, UnsafePointer<T4>) throws -> R
) rethrows -> R {
    try a.withUnsafeMutableBufferPointer { pA in
        try b.withUnsafeMutableBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try d.withUnsafeBufferPointer { pD in
                    try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!)
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T1, T2, T3, T4, R>(
    mut a: inout [T1], mut b: inout [T2], mut c: inout [T3], mut d: inout [T4],
    _ body: (UnsafeMutablePointer<T1>, UnsafeMutablePointer<T2>, UnsafeMutablePointer<T3>, UnsafeMutablePointer<T4>) throws -> R
) rethrows -> R {
    try a.withUnsafeMutableBufferPointer { pA in
        try b.withUnsafeMutableBufferPointer { pB in
            try c.withUnsafeMutableBufferPointer { pC in
                try d.withUnsafeMutableBufferPointer { pD in
                    try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!)
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T], _ b: [T], _ c: [T], _ d: [T], _ e: [T], _ f: [T],
    _ body: (UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try d.withUnsafeBufferPointer { pD in
                    try e.withUnsafeBufferPointer { pE in
                        try f.withUnsafeBufferPointer { pF in
                            try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!, pE.baseAddress!, pF.baseAddress!)
                        }
                    }
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T], _ b: [T], _ c: [T],
    mut d: inout [T], mut e: inout [T], mut f: inout [T],
    _ body: (UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try d.withUnsafeMutableBufferPointer { pD in
                    try e.withUnsafeMutableBufferPointer { pE in
                        try f.withUnsafeMutableBufferPointer { pF in
                            try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!, pE.baseAddress!, pF.baseAddress!)
                        }
                    }
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T1, T2, T3, T4, R>(
    mut a: inout [T1], _ b: [T2], _ c: [T3], _ d: [T4],
    _ body: (UnsafeMutablePointer<T1>, UnsafePointer<T2>, UnsafePointer<T3>, UnsafePointer<T4>) throws -> R
) rethrows -> R {
    try a.withUnsafeMutableBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try d.withUnsafeBufferPointer { pD in
                    try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!)
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T1, T2, T3, T4, R>(
    _ a: [T1], mut b: inout [T2], _ c: [T3], _ d: [T4],
    _ body: (UnsafePointer<T1>, UnsafeMutablePointer<T2>, UnsafePointer<T3>, UnsafePointer<T4>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeMutableBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try d.withUnsafeBufferPointer { pD in
                    try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!)
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T, R>(
    _ a: [T], _ b: [T], _ c: [T], _ d: [T], _ e: [T], _ f: [T],
    mut g: inout [T], mut h: inout [T], mut i: inout [T],
    _ body: (
        UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>,
        UnsafePointer<T>, UnsafePointer<T>, UnsafePointer<T>,
        UnsafeMutablePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>
    ) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try d.withUnsafeBufferPointer { pD in
                    try e.withUnsafeBufferPointer { pE in
                        try f.withUnsafeBufferPointer { pF in
                            try g.withUnsafeMutableBufferPointer { pG in
                                try h.withUnsafeMutableBufferPointer { pH in
                                    try i.withUnsafeMutableBufferPointer { pI in
                                        try body(
                                            pA.baseAddress!, pB.baseAddress!, pC.baseAddress!,
                                            pD.baseAddress!, pE.baseAddress!, pF.baseAddress!,
                                            pG.baseAddress!, pH.baseAddress!, pI.baseAddress!
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T1, T2, T3, T4, T5, T6, R>(
    mut a: inout [T1], _ b: [T2], _ c: [T3], _ d: [T4], _ e: [T5], _ f: [T6],
    _ body: (UnsafeMutablePointer<T1>, UnsafePointer<T2>, UnsafePointer<T3>, UnsafePointer<T4>, UnsafePointer<T5>, UnsafePointer<T6>) throws -> R
) rethrows -> R {
    try a.withUnsafeMutableBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeBufferPointer { pC in
                try d.withUnsafeBufferPointer { pD in
                    try e.withUnsafeBufferPointer { pE in
                        try f.withUnsafeBufferPointer { pF in
                            try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!, pE.baseAddress!, pF.baseAddress!)
                        }
                    }
                }
            }
        }
    }
}

@inline(__always)
internal func withUnsafePointers<T1, T2, T3, T4, T5, T6, R>(
    _ a: [T1], _ b: [T2], mut c: inout [T3], _ d: [T4], _ e: [T5], _ f: [T6],
    _ body: (UnsafePointer<T1>, UnsafePointer<T2>, UnsafeMutablePointer<T3>, UnsafePointer<T4>, UnsafePointer<T5>, UnsafePointer<T6>) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try b.withUnsafeBufferPointer { pB in
            try c.withUnsafeMutableBufferPointer { pC in
                try d.withUnsafeBufferPointer { pD in
                    try e.withUnsafeBufferPointer { pE in
                        try f.withUnsafeBufferPointer { pF in
                            try body(pA.baseAddress!, pB.baseAddress!, pC.baseAddress!, pD.baseAddress!, pE.baseAddress!, pF.baseAddress!)
                        }
                    }
                }
            }
        }
    }
}

/// Read-only pointers for the CtxRans context predictor: the block's own
/// coefficients plus the top / left neighbour buffers, where either neighbour
/// may be unavailable at a plane edge. All three pointers stay inside `body`,
/// so the availability test happens here rather than at the call site — hoisting
/// `baseAddress` out of the closure to build the optional would escape the
/// pointer past its valid region.
@inline(__always)
internal func withUnsafeNeighborPointers<T, R>(
    _ a: [T], top: [T], hasTop: Bool, left: [T], hasLeft: Bool,
    _ body: (UnsafePointer<T>, UnsafePointer<T>?, UnsafePointer<T>?) throws -> R
) rethrows -> R {
    try a.withUnsafeBufferPointer { pA in
        try top.withUnsafeBufferPointer { pTop in
            try left.withUnsafeBufferPointer { pLeft in
                let topPtr: UnsafePointer<T>?
                switch hasTop {
                case true:
                    topPtr = pTop.baseAddress!
                case false:
                    topPtr = nil
                }
                let leftPtr: UnsafePointer<T>?
                switch hasLeft {
                case true:
                    leftPtr = pLeft.baseAddress!
                case false:
                    leftPtr = nil
                }
                return try body(pA.baseAddress!, topPtr, leftPtr)
            }
        }
    }
}
