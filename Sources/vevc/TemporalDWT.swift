private func haar(
    f0: UnsafePointer<Int16>,
    f1: UnsafePointer<Int16>,
    count: Int,
    outL: UnsafeMutablePointer<Int16>,
    outH: UnsafeMutablePointer<Int16>
) {
    assert(count % 8 == 0, "count must be a multiple of 8")
    var i = 0
    
    while i < count {
        let v0 = SIMD8<Int16>(
            f0[i], f0[i+1], f0[i+2], f0[i+3],
            f0[i+4], f0[i+5], f0[i+6], f0[i+7]
        )
        let v1 = SIMD8<Int16>(
            f1[i], f1[i+1], f1[i+2], f1[i+3],
            f1[i+4], f1[i+5], f1[i+6], f1[i+7]
        )
        
        // H (High): f0 - f1
        let h = v0 &- v1
        
        // L (Low): f0 - (h / 2)
        // using lifting step to ensure perfect reconstruction
        let l = v0 &- (h &>> 1)
        
        UnsafeMutableRawPointer(outL.advanced(by: i)).storeBytes(of: l, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outH.advanced(by: i)).storeBytes(of: h, as: SIMD8<Int16>.self)
        
        i += 8
    }
}

func invHaar(
    inL: UnsafePointer<Int16>,
    inH: UnsafePointer<Int16>,
    count: Int,
    outF0: UnsafeMutablePointer<Int16>,
    outF1: UnsafeMutablePointer<Int16>
) {
    assert(count % 8 == 0, "count must be a multiple of 8")
    var i = 0
    while i < count {
        let l = SIMD8<Int16>(
            inL[i], inL[i+1], inL[i+2], inL[i+3],
            inL[i+4], inL[i+5], inL[i+6], inL[i+7]
        )
        let h = SIMD8<Int16>(
            inH[i], inH[i+1], inH[i+2], inH[i+3],
            inH[i+4], inH[i+5], inH[i+6], inH[i+7]
        )
        
        // f0 = l + (h / 2)
        // using lifting step to ensure perfect reconstruction
        let f0 = l &+ (h &>> 1)
        
        // f1 = f0 - h
        let f1 = f0 &- h
        
        UnsafeMutableRawPointer(outF0.advanced(by: i)).storeBytes(of: f0, as: SIMD8<Int16>.self)
        UnsafeMutableRawPointer(outF1.advanced(by: i)).storeBytes(of: f1, as: SIMD8<Int16>.self)
        
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
    // gen: L0, H0
    haar(f0: f0, f1: f1, count: count, outL: tempL0, outH: outH0)
    
    // gen: L1, H1
    haar(f0: f2, f1: f3, count: count, outL: tempL1, outH: outH1)
    
    // --- Level 2 ---
    // L0,L1 -> LL, LH
    haar(f0: tempL0, f1: tempL1, count: count, outL: outLL, outH: outLH)
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
    invHaar(inL: inLL, inH: inLH, count: count, outF0: tempL0, outF1: tempL1)
    
    // --- inv Level 1 ---
    // L0, H0 -> F0, F1
    invHaar(inL: tempL0, inH: inH0, count: count, outF0: outF0, outF1: outF1)
    
    // L1, H1 -> F2, F3
    invHaar(inL: tempL1, inH: inH1, count: count, outF0: outF2, outF1: outF3)
}