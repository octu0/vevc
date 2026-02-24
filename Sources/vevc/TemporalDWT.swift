@inline(__always)
private func lift53TemporalLevel1(
    f0: UnsafePointer<Int16>,
    f1: UnsafePointer<Int16>,
    f2: UnsafePointer<Int16>,
    f3: UnsafePointer<Int16>,
    count: Int,
    outL0: UnsafeMutablePointer<Int16>,
    outL1: UnsafeMutablePointer<Int16>,
    outH0: UnsafeMutablePointer<Int16>,
    outH1: UnsafeMutablePointer<Int16>
) {
    assert(count % 8 == 0, "count must be a multiple of 8")
    var i = 0
    
    // SIMD16 main loop
    while (i + 16) <= count {
        let v0 = SIMD16<Int16>(
            f0[i], f0[i+1], f0[i+2], f0[i+3], f0[i+4], f0[i+5], f0[i+6], f0[i+7],
            f0[i+8], f0[i+9], f0[i+10], f0[i+11], f0[i+12], f0[i+13], f0[i+14], f0[i+15]
        )
        let v1 = SIMD16<Int16>(
            f1[i], f1[i+1], f1[i+2], f1[i+3], f1[i+4], f1[i+5], f1[i+6], f1[i+7],
            f1[i+8], f1[i+9], f1[i+10], f1[i+11], f1[i+12], f1[i+13], f1[i+14], f1[i+15]
        )
        let v2 = SIMD16<Int16>(
            f2[i], f2[i+1], f2[i+2], f2[i+3], f2[i+4], f2[i+5], f2[i+6], f2[i+7],
            f2[i+8], f2[i+9], f2[i+10], f2[i+11], f2[i+12], f2[i+13], f2[i+14], f2[i+15]
        )
        let v3 = SIMD16<Int16>(
            f3[i], f3[i+1], f3[i+2], f3[i+3], f3[i+4], f3[i+5], f3[i+6], f3[i+7],
            f3[i+8], f3[i+9], f3[i+10], f3[i+11], f3[i+12], f3[i+13], f3[i+14], f3[i+15]
        )
        
        let h0 = v1 &- ((v0 &+ v2) &>> 1)
        let h1 = v3 &- v2
        let l0 = v0 &+ ((h0 &+ h0 &+ 2) &>> 2)
        let l1 = v2 &+ ((h0 &+ h1 &+ 2) &>> 2)
        
        UnsafeMutableRawPointer(outL0.advanced(by: i)).storeBytes(of: l0, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(outL1.advanced(by: i)).storeBytes(of: l1, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(outH0.advanced(by: i)).storeBytes(of: h0, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(outH1.advanced(by: i)).storeBytes(of: h1, as: SIMD16<Int16>.self)
        
        i += 16
    }
    
    // SIMD8 remainder
    while i < count {
        let v0 = SIMD8<Int16>(
            f0[i], f0[i+1], f0[i+2], f0[i+3], f0[i+4], f0[i+5], f0[i+6], f0[i+7]
        )
        let v1 = SIMD8<Int16>(
            f1[i], f1[i+1], f1[i+2], f1[i+3], f1[i+4], f1[i+5], f1[i+6], f1[i+7]
        )
        let v2 = SIMD8<Int16>(
            f2[i], f2[i+1], f2[i+2], f2[i+3], f2[i+4], f2[i+5], f2[i+6], f2[i+7]
        )
        let v3 = SIMD8<Int16>(
            f3[i], f3[i+1], f3[i+2], f3[i+3], f3[i+4], f3[i+5], f3[i+6], f3[i+7]
        )
        
        let h0 = v1 &- ((v0 &+ v2) &>> 1)
        let h1 = v3 &- v2
        let l0 = v0 &+ ((h0 &+ h0 &+ 2) &>> 2)
        let l1 = v2 &+ ((h0 &+ h1 &+ 2) &>> 2)
        
        UnsafeMutableRawPointer(outL0.advanced(by: i)).storeBytes(of: l0, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outL1.advanced(by: i)).storeBytes(of: l1, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outH0.advanced(by: i)).storeBytes(of: h0, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outH1.advanced(by: i)).storeBytes(of: h1, as: SIMD8<Int16>.self)
        
        i += 8
    }
}

@inline(__always)
private func lift53TemporalLevel2(
    l0: UnsafePointer<Int16>,
    l1: UnsafePointer<Int16>,
    count: Int,
    outLL: UnsafeMutablePointer<Int16>,
    outLH: UnsafeMutablePointer<Int16>
) {
    assert(count % 8 == 0, "count must be a multiple of 8")
    var i = 0
    
    // SIMD16 main loop
    while (i + 16) <= count {
        let v0 = SIMD16<Int16>(
            l0[i], l0[i+1], l0[i+2], l0[i+3], l0[i+4], l0[i+5], l0[i+6], l0[i+7],
            l0[i+8], l0[i+9], l0[i+10], l0[i+11], l0[i+12], l0[i+13], l0[i+14], l0[i+15]
        )
        let v1 = SIMD16<Int16>(
            l1[i], l1[i+1], l1[i+2], l1[i+3], l1[i+4], l1[i+5], l1[i+6], l1[i+7],
            l1[i+8], l1[i+9], l1[i+10], l1[i+11], l1[i+12], l1[i+13], l1[i+14], l1[i+15]
        )
        
        let lh = v1 &- v0
        let ll = v0 &+ ((lh &+ lh &+ 2) &>> 2)
        
        UnsafeMutableRawPointer(outLL.advanced(by: i)).storeBytes(of: ll, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(outLH.advanced(by: i)).storeBytes(of: lh, as: SIMD16<Int16>.self)
        
        i += 16
    }
    
    // SIMD8 remainder
    while i < count {
        let v0 = SIMD8<Int16>(
            l0[i], l0[i+1], l0[i+2], l0[i+3], l0[i+4], l0[i+5], l0[i+6], l0[i+7]
        )
        let v1 = SIMD8<Int16>(
            l1[i], l1[i+1], l1[i+2], l1[i+3], l1[i+4], l1[i+5], l1[i+6], l1[i+7]
        )
        
        let lh = v1 &- v0
        let ll = v0 &+ ((lh &+ lh &+ 2) &>> 2)
        
        UnsafeMutableRawPointer(outLL.advanced(by: i)).storeBytes(of: ll, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outLH.advanced(by: i)).storeBytes(of: lh, as: SIMD8<Int16>.self)
        
        i += 8
    }
}

@inline(__always)
private func invLift53TemporalLevel1(
    inL0: UnsafePointer<Int16>,
    inL1: UnsafePointer<Int16>,
    inH0: UnsafePointer<Int16>,
    inH1: UnsafePointer<Int16>,
    count: Int,
    outF0: UnsafeMutablePointer<Int16>,
    outF1: UnsafeMutablePointer<Int16>,
    outF2: UnsafeMutablePointer<Int16>,
    outF3: UnsafeMutablePointer<Int16>
) {
    assert(count % 8 == 0, "count must be a multiple of 8")
    var i = 0
    
    // SIMD16 main loop
    while (i + 16) <= count {
        let l0 = SIMD16<Int16>(
            inL0[i], inL0[i+1], inL0[i+2], inL0[i+3], inL0[i+4], inL0[i+5], inL0[i+6], inL0[i+7],
            inL0[i+8], inL0[i+9], inL0[i+10], inL0[i+11], inL0[i+12], inL0[i+13], inL0[i+14], inL0[i+15]
        )
        let l1 = SIMD16<Int16>(
            inL1[i], inL1[i+1], inL1[i+2], inL1[i+3], inL1[i+4], inL1[i+5], inL1[i+6], inL1[i+7],
            inL1[i+8], inL1[i+9], inL1[i+10], inL1[i+11], inL1[i+12], inL1[i+13], inL1[i+14], inL1[i+15]
        )
        let h0 = SIMD16<Int16>(
            inH0[i], inH0[i+1], inH0[i+2], inH0[i+3], inH0[i+4], inH0[i+5], inH0[i+6], inH0[i+7],
            inH0[i+8], inH0[i+9], inH0[i+10], inH0[i+11], inH0[i+12], inH0[i+13], inH0[i+14], inH0[i+15]
        )
        let h1 = SIMD16<Int16>(
            inH1[i], inH1[i+1], inH1[i+2], inH1[i+3], inH1[i+4], inH1[i+5], inH1[i+6], inH1[i+7],
            inH1[i+8], inH1[i+9], inH1[i+10], inH1[i+11], inH1[i+12], inH1[i+13], inH1[i+14], inH1[i+15]
        )
        
        let f0 = l0 &- ((h0 &+ h0 &+ 2) &>> 2)
        let f2 = l1 &- ((h0 &+ h1 &+ 2) &>> 2)
        let f1 = h0 &+ ((f0 &+ f2) &>> 1)
        let f3 = h1 &+ f2
        
        UnsafeMutableRawPointer(outF0.advanced(by: i)).storeBytes(of: f0, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(outF1.advanced(by: i)).storeBytes(of: f1, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(outF2.advanced(by: i)).storeBytes(of: f2, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(outF3.advanced(by: i)).storeBytes(of: f3, as: SIMD16<Int16>.self)
        
        i += 16
    }
    
    // SIMD8 remainder
    while i < count {
        let l0 = SIMD8<Int16>(
            inL0[i], inL0[i+1], inL0[i+2], inL0[i+3], inL0[i+4], inL0[i+5], inL0[i+6], inL0[i+7]
        )
        let l1 = SIMD8<Int16>(
            inL1[i], inL1[i+1], inL1[i+2], inL1[i+3], inL1[i+4], inL1[i+5], inL1[i+6], inL1[i+7]
        )
        let h0 = SIMD8<Int16>(
            inH0[i], inH0[i+1], inH0[i+2], inH0[i+3], inH0[i+4], inH0[i+5], inH0[i+6], inH0[i+7]
        )
        let h1 = SIMD8<Int16>(
            inH1[i], inH1[i+1], inH1[i+2], inH1[i+3], inH1[i+4], inH1[i+5], inH1[i+6], inH1[i+7]
        )
        
        let f0 = l0 &- ((h0 &+ h0 &+ 2) &>> 2)
        let f2 = l1 &- ((h0 &+ h1 &+ 2) &>> 2)
        let f1 = h0 &+ ((f0 &+ f2) &>> 1)
        let f3 = h1 &+ f2
        
        UnsafeMutableRawPointer(outF0.advanced(by: i)).storeBytes(of: f0, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outF1.advanced(by: i)).storeBytes(of: f1, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outF2.advanced(by: i)).storeBytes(of: f2, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outF3.advanced(by: i)).storeBytes(of: f3, as: SIMD8<Int16>.self)
        
        i += 8
    }
}

@inline(__always)
private func invLift53TemporalLevel2(
    inLL: UnsafePointer<Int16>,
    inLH: UnsafePointer<Int16>,
    count: Int,
    outL0: UnsafeMutablePointer<Int16>,
    outL1: UnsafeMutablePointer<Int16>
) {
    assert(count % 8 == 0, "count must be a multiple of 8")
    var i = 0
    
    // SIMD16 main loop
    while (i + 16) <= count {
        let ll = SIMD16<Int16>(
            inLL[i], inLL[i+1], inLL[i+2], inLL[i+3], inLL[i+4], inLL[i+5], inLL[i+6], inLL[i+7],
            inLL[i+8], inLL[i+9], inLL[i+10], inLL[i+11], inLL[i+12], inLL[i+13], inLL[i+14], inLL[i+15]
        )
        let lh = SIMD16<Int16>(
            inLH[i], inLH[i+1], inLH[i+2], inLH[i+3], inLH[i+4], inLH[i+5], inLH[i+6], inLH[i+7],
            inLH[i+8], inLH[i+9], inLH[i+10], inLH[i+11], inLH[i+12], inLH[i+13], inLH[i+14], inLH[i+15]
        )
        
        let l0 = ll &- ((lh &+ lh &+ 2) &>> 2)
        let l1 = lh &+ l0
        
        UnsafeMutableRawPointer(outL0.advanced(by: i)).storeBytes(of: l0, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(outL1.advanced(by: i)).storeBytes(of: l1, as: SIMD16<Int16>.self)
        
        i += 16
    }
    
    // SIMD8 remainder
    while i < count {
        let ll = SIMD8<Int16>(
            inLL[i], inLL[i+1], inLL[i+2], inLL[i+3], inLL[i+4], inLL[i+5], inLL[i+6], inLL[i+7]
        )
        let lh = SIMD8<Int16>(
            inLH[i], inLH[i+1], inLH[i+2], inLH[i+3], inLH[i+4], inLH[i+5], inLH[i+6], inLH[i+7]
        )
        
        let l0 = ll &- ((lh &+ lh &+ 2) &>> 2)
        let l1 = lh &+ l0
        
        UnsafeMutableRawPointer(outL0.advanced(by: i)).storeBytes(of: l0, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outL1.advanced(by: i)).storeBytes(of: l1, as: SIMD8<Int16>.self)
        
        i += 8
    }
}

@inline(__always)
func temporalDWT(
    f0: UnsafePointer<Int16>,
    f1: UnsafePointer<Int16>,
    f2: UnsafePointer<Int16>,
    f3: UnsafePointer<Int16>,
    count: Int,
    outLL: UnsafeMutablePointer<Int16>, 
    outLH: UnsafeMutablePointer<Int16>,
    outH0: UnsafeMutablePointer<Int16>,
    outH1: UnsafeMutablePointer<Int16>,
    tempL0: UnsafeMutablePointer<Int16>, 
    tempL1: UnsafeMutablePointer<Int16>
) {
    // --- Level 1 ---
    // f0, f1, f2, f3 -> L0, L1, H0, H1
    lift53TemporalLevel1(
        f0: f0,
        f1: f1,
        f2: f2,
        f3: f3,
        count: count,
        outL0: tempL0,
        outL1: tempL1,
        outH0: outH0,
        outH1: outH1
    )
    
    // --- Level 2 ---
    // L0, L1 -> LL, LH
    lift53TemporalLevel2(
        l0: tempL0,
        l1: tempL1,
        count: count,
        outLL: outLL,
        outLH: outLH
    )
}

@inline(__always)
func invTemporalDWT(
    inLL: UnsafePointer<Int16>,
    inLH: UnsafePointer<Int16>,
    inH0: UnsafePointer<Int16>,
    inH1: UnsafePointer<Int16>,
    count: Int,
    outF0: UnsafeMutablePointer<Int16>,
    outF1: UnsafeMutablePointer<Int16>,
    outF2: UnsafeMutablePointer<Int16>,
    outF3: UnsafeMutablePointer<Int16>,
    tempL0: UnsafeMutablePointer<Int16>,
    tempL1: UnsafeMutablePointer<Int16>
) {
    // --- inv Level 2 ---
    // LL, LH -> L0, L1
    invLift53TemporalLevel2(
        inLL: inLL,
        inLH: inLH,
        count: count,
        outL0: tempL0,
        outL1: tempL1
    )
    
    // --- inv Level 1 ---
    // L0, L1, H0, H1 -> F0, F1, F2, F3
    invLift53TemporalLevel1(
        inL0: tempL0,
        inL1: tempL1,
        inH0: inH0,
        inH1: inH1,
        count: count,
        outF0: outF0,
        outF1: outF1,
        outF2: outF2,
        outF3: outF3
    )
}