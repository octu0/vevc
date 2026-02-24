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
        
        // Predict (High)
        let h0 = v1 &- ((v0 &+ v2) &>> 1)
        let h1 = v3 &- v2
        
        // Update (Low)
        let l0 = v0 &+ ((h0 &+ h0 &+ 2) &>> 2)
        let l1 = v2 &+ ((h0 &+ h1 &+ 2) &>> 2)
        
        UnsafeMutableRawPointer(outL0.advanced(by: i)).storeBytes(of: l0, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outL1.advanced(by: i)).storeBytes(of: l1, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outH0.advanced(by: i)).storeBytes(of: h0, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outH1.advanced(by: i)).storeBytes(of: h1, as: SIMD8<Int16>.self)
        
        i += 8
    }
}

private func lift53TemporalLevel2(
    l0: UnsafePointer<Int16>,
    l1: UnsafePointer<Int16>,
    count: Int,
    outLL: UnsafeMutablePointer<Int16>,
    outLH: UnsafeMutablePointer<Int16>
) {
    assert(count % 8 == 0, "count must be a multiple of 8")
    var i = 0
    while i < count {
        let v0 = SIMD8<Int16>(
            l0[i], l0[i+1], l0[i+2], l0[i+3], l0[i+4], l0[i+5], l0[i+6], l0[i+7]
        )
        let v1 = SIMD8<Int16>(
            l1[i], l1[i+1], l1[i+2], l1[i+3], l1[i+4], l1[i+5], l1[i+6], l1[i+7]
        )
        
        // Predict
        let lh = v1 &- v0
        
        // Update
        let ll = v0 &+ ((lh &+ lh &+ 2) &>> 2)
        
        UnsafeMutableRawPointer(outLL.advanced(by: i)).storeBytes(of: ll, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outLH.advanced(by: i)).storeBytes(of: lh, as: SIMD8<Int16>.self)
        
        i += 8
    }
}

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
        
        // Inv Update
        let f0 = l0 &- ((h0 &+ h0 &+ 2) &>> 2)
        let f2 = l1 &- ((h0 &+ h1 &+ 2) &>> 2)
        
        // Inv Predict
        let f1 = h0 &+ ((f0 &+ f2) &>> 1)
        let f3 = h1 &+ f2
        
        UnsafeMutableRawPointer(outF0.advanced(by: i)).storeBytes(of: f0, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outF1.advanced(by: i)).storeBytes(of: f1, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outF2.advanced(by: i)).storeBytes(of: f2, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outF3.advanced(by: i)).storeBytes(of: f3, as: SIMD8<Int16>.self)
        
        i += 8
    }
}

private func invLift53TemporalLevel2(
    inLL: UnsafePointer<Int16>,
    inLH: UnsafePointer<Int16>,
    count: Int,
    outL0: UnsafeMutablePointer<Int16>,
    outL1: UnsafeMutablePointer<Int16>
) {
    assert(count % 8 == 0, "count must be a multiple of 8")
    var i = 0
    while i < count {
        let ll = SIMD8<Int16>(
            inLL[i], inLL[i+1], inLL[i+2], inLL[i+3], inLL[i+4], inLL[i+5], inLL[i+6], inLL[i+7]
        )
        let lh = SIMD8<Int16>(
            inLH[i], inLH[i+1], inLH[i+2], inLH[i+3], inLH[i+4], inLH[i+5], inLH[i+6], inLH[i+7]
        )
        
        // Inv Update
        let l0 = ll &- ((lh &+ lh &+ 2) &>> 2)
        
        // Inv Predict
        let l1 = lh &+ l0
        
        UnsafeMutableRawPointer(outL0.advanced(by: i)).storeBytes(of: l0, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outL1.advanced(by: i)).storeBytes(of: l1, as: SIMD8<Int16>.self)
        
        i += 8
    }
}

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