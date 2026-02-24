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
        let v0 = UnsafeRawPointer(f0.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        let v1 = UnsafeRawPointer(f1.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        let v2 = UnsafeRawPointer(f2.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        let v3 = UnsafeRawPointer(f3.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        
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
        let v0 = UnsafeRawPointer(f0.advanced(by: i)).loadUnaligned(as: SIMD8<Int16>.self)
        let v1 = UnsafeRawPointer(f1.advanced(by: i)).loadUnaligned(as: SIMD8<Int16>.self)
        let v2 = UnsafeRawPointer(f2.advanced(by: i)).loadUnaligned(as: SIMD8<Int16>.self)
        let v3 = UnsafeRawPointer(f3.advanced(by: i)).loadUnaligned(as: SIMD8<Int16>.self)
        
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
        let v0 = UnsafeRawPointer(l0.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        let v1 = UnsafeRawPointer(l1.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        
        let lh = v1 &- v0
        let ll = v0 &+ ((lh &+ lh &+ 2) &>> 2)
        
        UnsafeMutableRawPointer(outLL.advanced(by: i)).storeBytes(of: ll, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(outLH.advanced(by: i)).storeBytes(of: lh, as: SIMD16<Int16>.self)
        
        i += 16
    }
    
    // SIMD8 remainder
    while i < count {
        let v0 = UnsafeRawPointer(l0.advanced(by: i)).loadUnaligned(as: SIMD8<Int16>.self)
        let v1 = UnsafeRawPointer(l1.advanced(by: i)).loadUnaligned(as: SIMD8<Int16>.self)
        
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
        let l0 = UnsafeRawPointer(inL0.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        let l1 = UnsafeRawPointer(inL1.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        let h0 = UnsafeRawPointer(inH0.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        let h1 = UnsafeRawPointer(inH1.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        
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
        let l0 = UnsafeRawPointer(inL0.advanced(by: i)).loadUnaligned(as: SIMD8<Int16>.self)
        let l1 = UnsafeRawPointer(inL1.advanced(by: i)).loadUnaligned(as: SIMD8<Int16>.self)
        let h0 = UnsafeRawPointer(inH0.advanced(by: i)).loadUnaligned(as: SIMD8<Int16>.self)
        let h1 = UnsafeRawPointer(inH1.advanced(by: i)).loadUnaligned(as: SIMD8<Int16>.self)
        
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
        let ll = UnsafeRawPointer(inLL.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        let lh = UnsafeRawPointer(inLH.advanced(by: i)).loadUnaligned(as: SIMD16<Int16>.self)
        
        let l0 = ll &- ((lh &+ lh &+ 2) &>> 2)
        let l1 = lh &+ l0
        
        UnsafeMutableRawPointer(outL0.advanced(by: i)).storeBytes(of: l0, as: SIMD16<Int16>.self)
        UnsafeMutableRawPointer(outL1.advanced(by: i)).storeBytes(of: l1, as: SIMD16<Int16>.self)
        
        i += 16
    }
    
    // SIMD8 remainder
    while i < count {
        let ll = UnsafeRawPointer(inLL.advanced(by: i)).loadUnaligned(as: SIMD8<Int16>.self)
        let lh = UnsafeRawPointer(inLH.advanced(by: i)).loadUnaligned(as: SIMD8<Int16>.self)
        
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