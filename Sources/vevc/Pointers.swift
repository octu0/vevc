import Foundation

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

