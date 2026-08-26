// MARK: - SNN Neural Loop Filter (NLF) Inference Engine
// Pure integer, SIMD16 vectorization, deterministic bit-exact inference engine.

import Foundation

public enum PlaneType: Sendable {
    case y
    case cb
    case cr
}

public struct SNNLoopFilter {
    @inline(__always)
    public static func getBlockSize(for planeType: PlaneType) -> Int {
        switch planeType {
        case .y:
            return 32
        case .cb, .cr:
            return 16
        }
    }
}

private let kDistX32_0 = SIMD16<Int16>(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
private let kDistX32_16 = SIMD16<Int16>(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
private let kDistX16 = SIMD16<Int16>(0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1)

struct InputKernelWeightsScalar {
    let w0: Int32
    let w1: Int32
    let w2: Int32
    let w3: Int32
    let w4: Int32
    let w5: Int32
    let w6: Int32
    let w7: Int32
}

struct StaticKernelWeightsScalar {
    let w0: Int32
    let w1: Int32
    let w2: Int32
    let w3: Int32
    let w4: Int32
    let w5: Int32
    let w6: Int32
    let w7: Int32
}

final class StaticWeightsTable: @unchecked Sendable {
    static let shared = StaticWeightsTable()

    let conv1InputWeights: [StaticKernelWeightsScalar]
    let conv1B16: [SIMD16<Int16>]
    let conv2B16: [SIMD16<Int16>]
    let conv2W16: [SIMD16<Int16>]
    let outW16: [SIMD16<Int16>]
    let outB16Double: SIMD16<Int16>

    let staticSliceWorkspace: UnsafeMutablePointer<Int16>
    let staticOutputBuffer: UnsafeMutablePointer<Int16>
    let maxSlices: Int
    let sliceBufSize: Int
    let lock = NSLock()

    private init() {
        var inW = [StaticKernelWeightsScalar]()
        inW.reserveCapacity(36)
        for k in 0..<36 {
            let w0 = Int32(SNNWeights.conv1Weights[k])
            let w1 = Int32(SNNWeights.conv1Weights[36 + k])
            let w2 = Int32(SNNWeights.conv1Weights[72 + k])
            let w3 = Int32(SNNWeights.conv1Weights[108 + k])
            let w4 = Int32(SNNWeights.conv1Weights[144 + k])
            let w5 = Int32(SNNWeights.conv1Weights[180 + k])
            let w6 = Int32(SNNWeights.conv1Weights[216 + k])
            let w7 = Int32(SNNWeights.conv1Weights[252 + k])
            inW.append(StaticKernelWeightsScalar(w0: w0, w1: w1, w2: w2, w3: w3, w4: w4, w5: w5, w6: w6, w7: w7))
        }
        self.conv1InputWeights = inW

        var c1B = [SIMD16<Int16>]()
        c1B.reserveCapacity(8)
        for i in 0..<8 {
            c1B.append(SIMD16<Int16>(repeating: SNNWeights.conv1Biases[i]))
        }
        self.conv1B16 = c1B

        var c2B = [SIMD16<Int16>]()
        c2B.reserveCapacity(8)
        for i in 0..<8 {
            c2B.append(SIMD16<Int16>(repeating: SNNWeights.conv2Biases[i]))
        }
        self.conv2B16 = c2B

        var c2W = [SIMD16<Int16>]()
        c2W.reserveCapacity(64)
        for i in 0..<64 {
            c2W.append(SIMD16<Int16>(repeating: Int16(SNNWeights.conv2Weights[i]) &<< 4))
        }
        self.conv2W16 = c2W

        var oW = [SIMD16<Int16>]()
        oW.reserveCapacity(8)
        for i in 0..<8 {
            oW.append(SIMD16<Int16>(repeating: SNNWeights.outWeights[i]))
        }
        self.outW16 = oW

        self.outB16Double = SIMD16<Int16>(repeating: SNNWeights.outBias &* 2)

        let maxW = 1920
        self.maxSlices = 36
        let maxRowsPerSlice = 40
        let chStride = maxW + 32
        let sliceRowStride = 4 * chStride
        self.sliceBufSize = maxRowsPerSlice * sliceRowStride
        let totalWorkspaceCapacity = maxSlices * sliceBufSize
        self.staticSliceWorkspace = UnsafeMutablePointer<Int16>.allocate(capacity: totalWorkspaceCapacity)

        let outputBufferCapacity = maxW * 1080
        self.staticOutputBuffer = UnsafeMutablePointer<Int16>.allocate(capacity: outputBufferCapacity)
    }

    deinit {
        staticSliceWorkspace.deallocate()
        staticOutputBuffer.deallocate()
    }
}

/// Applies the SNN Neural Loop Filter to the given plane in-place.
public func applySNNNeuralLoopFilter(
    plane: inout [Int16],
    width: Int,
    height: Int,
    planeType: PlaneType,
    skipMask: [UInt8]? = nil
) {
    guard 0 < width && 0 < height else { return }

    let blockSize: Int
    switch planeType {
    case .y:
        blockSize = 32
    case .cb, .cr:
        blockSize = 16
    }

    let numBlocksX = (width + blockSize - 1) / blockSize
    let numBlocksY = (height + blockSize - 1) / blockSize
    let totalBlocks = numBlocksX * numBlocksY

    let count = width * height
    guard plane.count == count else { return }

    if let sm = skipMask, sm.count == totalBlocks {
        var allSkipped = true
        for i in 0..<totalBlocks {
            if sm[i] == 0 {
                allSkipped = false
                break
            }
        }
        if allSkipped {
            return
        }
    }

    let weights = StaticWeightsTable.shared
    weights.lock.lock()
    defer { weights.lock.unlock() }

    let numSlices = min(36, max(4, (height + 31) / 32))
    let sliceHeight = (height + numSlices - 1) / numSlices

    let pSkipPtr: UnsafeSendablePointer<UInt8>?
    if let sm = skipMask, sm.count == totalBlocks {
        pSkipPtr = sm.withUnsafeBufferPointer { UnsafeSendablePointer(ptr: $0.baseAddress!) }
    } else {
        pSkipPtr = nil
    }

    let pad = 16
    let chStride = width + 32
    let sliceRowStride = 4 * chStride
    let sliceBufSize = weights.sliceBufSize

    let workspacePtr = UnsafeSendableMutablePointer(ptr: weights.staticSliceWorkspace)
    let dstPtr = UnsafeSendableMutablePointer(ptr: weights.staticOutputBuffer)

    weights.conv1InputWeights.withUnsafeBufferPointer { pC1W in
        weights.conv1B16.withUnsafeBufferPointer { pC1B in
            weights.conv2B16.withUnsafeBufferPointer { pC2B in
                weights.conv2W16.withUnsafeBufferPointer { pC2W in
                    weights.outW16.withUnsafeBufferPointer { pOW in
                        plane.withUnsafeBufferPointer { srcBuf in
                            let srcPtr = UnsafeSendablePointer(ptr: srcBuf.baseAddress!)
                            let c1WPtr = UnsafeSendablePointer(ptr: pC1W.baseAddress!)
                            let c1BPtr = UnsafeSendablePointer(ptr: pC1B.baseAddress!)
                            let c2BPtr = UnsafeSendablePointer(ptr: pC2B.baseAddress!)
                            let c2WPtr = UnsafeSendablePointer(ptr: pC2W.baseAddress!)
                            let oWPtr = UnsafeSendablePointer(ptr: pOW.baseAddress!)

                            // Single-pass per-slice execution: 100% cache resident
                            DispatchQueue.concurrentPerform(iterations: numSlices) { sliceIdx in
                                let startY = sliceIdx * sliceHeight
                                let endY = min(height, startY + sliceHeight)
                                guard startY < endY else { return }

                                let sliceBuf = workspacePtr.ptr.advanced(by: sliceIdx * sliceBufSize)

                                // 1. Compute local feature rows: from y = max(0, startY - 1) to min(height - 1, endY)
                                let fetchStartY = max(0, startY - 1)
                                let fetchEndY = min(height - 1, endY)

                                for y in fetchStartY...fetchEndY {
                                    let localRowIdx = y - startY + 1
                                    computeFeatureRow(
                                        src: srcPtr.ptr,
                                        width: width,
                                        height: height,
                                        y: y,
                                        blockSize: blockSize,
                                        dstBase: sliceBuf.advanced(by: localRowIdx * sliceRowStride),
                                        chStride: chStride,
                                        pad: pad
                                    )
                                }

                                if startY == 0 {
                                    sliceBuf.advanced(by: 0).update(from: sliceBuf.advanced(by: sliceRowStride), count: sliceRowStride)
                                }
                                if endY == height {
                                    let lastLocal = endY - startY
                                    sliceBuf.advanced(by: (lastLocal + 1) * sliceRowStride).update(from: sliceBuf.advanced(by: lastLocal * sliceRowStride), count: sliceRowStride)
                                }

                                // 2. Process SNN Convolutions directly from local cache
                                processSliceSIMD16Direct(
                                    src: srcPtr.ptr,
                                    dst: dstPtr.ptr,
                                    featBase: sliceBuf,
                                    width: width,
                                    height: height,
                                    startY: startY,
                                    endY: endY,
                                    blockSize: blockSize,
                                    rowStride: sliceRowStride,
                                    chStride: chStride,
                                    pad: pad,
                                    skipMask: pSkipPtr?.ptr,
                                    numBlocksX: numBlocksX,
                                    c1InW: c1WPtr.ptr,
                                    c1B: c1BPtr.ptr,
                                    c2B: c2BPtr.ptr,
                                    c2W: c2WPtr.ptr,
                                    oW: oWPtr.ptr,
                                    outBDouble: weights.outB16Double
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    plane.withUnsafeMutableBufferPointer { pDst in
        pDst.baseAddress!.update(from: weights.staticOutputBuffer, count: count)
    }
}

// MARK: - SIMD16 Feature Extraction

@inline(__always)
private func computeFeatureRow(
    src: UnsafePointer<Int16>,
    width: Int,
    height: Int,
    y: Int,
    blockSize: Int,
    dstBase: UnsafeMutablePointer<Int16>,
    chStride: Int,
    pad: Int
) {
    let topY = max(0, y - 1)
    let botY = min(height - 1, y + 1)
    let topRow = src.advanced(by: topY * width)
    let currRow = src.advanced(by: y * width)
    let botRow = src.advanced(by: botY * width)

    let ch0 = dstBase.advanced(by: pad)
    let ch1 = dstBase.advanced(by: pad + chStride)
    let ch2 = dstBase.advanced(by: pad + 2 * chStride)
    let ch3 = dstBase.advanced(by: pad + 3 * chStride)

    let vecZero = SIMD16<Int16>.zero
    let shift2 = SIMD16<Int16>(repeating: 2)

    let yMod = y % blockSize
    let distY = Int16(min(yMod, blockSize - yMod))

    var x = 0
    let wFast16 = (width / 16) * 16

    var prevScalar = currRow[0]
    while x < wFast16 {
        let pCurr = UnsafeRawPointer(currRow.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
        let pTop = UnsafeRawPointer(topRow.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)
        let pBot = UnsafeRawPointer(botRow.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)

        let pLeft: SIMD16<Int16>
        let pRight: SIMD16<Int16>
        if 0 < x {
            pLeft = UnsafeRawPointer(currRow.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self)
        } else {
            pLeft = SIMD16<Int16>(
                prevScalar, pCurr[0], pCurr[1], pCurr[2],
                pCurr[3], pCurr[4], pCurr[5], pCurr[6],
                pCurr[7], pCurr[8], pCurr[9], pCurr[10],
                pCurr[11], pCurr[12], pCurr[13], pCurr[14]
            )
        }
        if x + 16 < width {
            pRight = UnsafeRawPointer(currRow.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self)
        } else {
            pRight = SIMD16<Int16>(
                pCurr[1], pCurr[2], pCurr[3], pCurr[4],
                pCurr[5], pCurr[6], pCurr[7], pCurr[8],
                pCurr[9], pCurr[10], pCurr[11], pCurr[12],
                pCurr[13], pCurr[14], pCurr[15], pCurr[15]
            )
        }
        prevScalar = pCurr[15]

        // Ch 0: Centered pixel
        UnsafeMutableRawPointer(ch0.advanced(by: x)).storeBytes(of: pCurr, as: SIMD16<Int16>.self)

        // Ch 1: 2nd-order Laplacian
        let p4 = pCurr &<< shift2
        let pTB = pTop &+ pBot
        let pLR = pLeft &+ pRight
        let f1 = p4 &- pTB &- pLR
        UnsafeMutableRawPointer(ch1.advanced(by: x)).storeBytes(of: f1, as: SIMD16<Int16>.self)

        // Ch 2: Local gradient magnitude
        let diffH = pRight &- pLeft
        let absH = diffH.replacing(with: vecZero &- diffH, where: diffH .< vecZero)
        let diffV = pBot &- pTop
        let absV = diffV.replacing(with: vecZero &- diffV, where: diffV .< vecZero)
        let f2 = absH &+ absV
        UnsafeMutableRawPointer(ch2.advanced(by: x)).storeBytes(of: f2, as: SIMD16<Int16>.self)

        // Ch 3: Boundary distance
        for xi in 0..<16 {
            let px = x + xi
            let xMod = px % blockSize
            let distX = Int16(min(xMod, blockSize - xMod))
            ch3[px] = (distX &+ distY) &<< 2
        }

        x &+= 16
    }

    while x < width {
        let xPrev = max(0, x - 1)
        let xNext = min(width - 1, x + 1)

        let centerVal = currRow[x]
        let topVal = topRow[x]
        let bottomVal = botRow[x]
        let leftVal = currRow[xPrev]
        let rightVal = currRow[xNext]

        ch0[x] = centerVal
        ch1[x] = (4 &* centerVal) &- topVal &- bottomVal &- leftVal &- rightVal

        let diffH = rightVal &- leftVal
        let absH: Int16 = diffH < 0 ? (0 &- diffH) : diffH
        let diffV = bottomVal &- topVal
        let absV: Int16 = diffV < 0 ? (0 &- diffV) : diffV
        ch2[x] = absH &+ absV

        let xMod = x % blockSize
        let distX = Int16(min(xMod, blockSize - xMod))
        ch3[x] = (distX &+ distY) &<< 2

        x &+= 1
    }

    let padLeft = pad
    let padRight = pad
    let first0 = ch0[0], first1 = ch1[0], first2 = ch2[0], first3 = ch3[0]
    let last0 = ch0[width - 1], last1 = ch1[width - 1], last2 = ch2[width - 1], last3 = ch3[width - 1]
    for p in 1...padLeft {
        ch0[-p] = first0
        ch1[-p] = first1
        ch2[-p] = first2
        ch3[-p] = first3
    }
    for p in 0..<padRight {
        ch0[width + p] = last0
        ch1[width + p] = last1
        ch2[width + p] = last2
        ch3[width + p] = last3
    }
}

// MARK: - SIMD16 Direct Slice Processing

@inline(__always)
private func processSliceSIMD16Direct(
    src: UnsafePointer<Int16>,
    dst: UnsafeMutablePointer<Int16>,
    featBase: UnsafePointer<Int16>,
    width: Int,
    height: Int,
    startY: Int,
    endY: Int,
    blockSize: Int,
    rowStride: Int,
    chStride: Int,
    pad: Int,
    skipMask: UnsafePointer<UInt8>? = nil,
    numBlocksX: Int,
    c1InW: UnsafePointer<StaticKernelWeightsScalar>,
    c1B: UnsafePointer<SIMD16<Int16>>,
    c2B: UnsafePointer<SIMD16<Int16>>,
    c2W: UnsafePointer<SIMD16<Int16>>,
    oW: UnsafePointer<SIMD16<Int16>>,
    outBDouble: SIMD16<Int16>
) {
    let wFast16 = (width / 16) * 16
    let vThresh1Vec = SIMD16<Int16>(repeating: SNNWeights.vThresh1)
    let vThresh2Vec = SIMD16<Int16>(repeating: SNNWeights.vThresh2)
    let leakShiftVec = SIMD16<Int16>(repeating: Int16(SNNWeights.leakShift))
    let shift3Vec32 = SIMD16<Int32>(repeating: 3)
    let fourVec32 = SIMD16<Int32>(repeating: 4)

    let cb1_0 = c1B[0], cb1_1 = c1B[1], cb1_2 = c1B[2], cb1_3 = c1B[3]
    let cb1_4 = c1B[4], cb1_5 = c1B[5], cb1_6 = c1B[6], cb1_7 = c1B[7]

    let cb2_0 = c2B[0], cb2_1 = c2B[1], cb2_2 = c2B[2], cb2_3 = c2B[3]
    let cb2_4 = c2B[4], cb2_5 = c2B[5], cb2_6 = c2B[6], cb2_7 = c2B[7]

    let shift4Vec16 = SIMD16<Int16>(repeating: 4)
    let eightVec16 = SIMD16<Int16>(repeating: 8)
    let minDelta = SIMD16<Int16>(repeating: -16)
    let maxDelta = SIMD16<Int16>(repeating: 16)
    let minPixel = SIMD16<Int16>(repeating: -128)
    let maxPixel = SIMD16<Int16>(repeating: 127)
    let zVec = SIMD16<Int16>.zero

    var y = startY
    while y < endY {
        let currRowSrc = src.advanced(by: y * width)
        let currRowDst = dst.advanced(by: y * width)

        let by = y / blockSize
        let byOffset = by * numBlocksX

        let topBase = featBase.advanced(by: (y - startY) * rowStride + pad)
        let midBase = featBase.advanced(by: (y - startY + 1) * rowStride + pad)
        let botBase = featBase.advanced(by: (y - startY + 2) * rowStride + pad)

        let topCh0 = topBase
        let topCh1 = topBase.advanced(by: chStride)
        let topCh2 = topBase.advanced(by: 2 * chStride)
        let topCh3 = topBase.advanced(by: 3 * chStride)

        let midCh0 = midBase
        let midCh1 = midBase.advanced(by: chStride)
        let midCh2 = midBase.advanced(by: 2 * chStride)
        let midCh3 = midBase.advanced(by: 3 * chStride)

        let botCh0 = botBase
        let botCh1 = botBase.advanced(by: chStride)
        let botCh2 = botBase.advanced(by: 2 * chStride)
        let botCh3 = botBase.advanced(by: 3 * chStride)

        var x = 0
        while x < wFast16 {
            let pCurr = UnsafeRawPointer(currRowSrc.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self)

            if let sm = skipMask {
                let bx0 = x / blockSize
                let bx1 = (x + 15) / blockSize
                let isSkipped0 = sm[byOffset + bx0] != 0
                let isSkipped1 = sm[byOffset + bx1] != 0

                if isSkipped0 && isSkipped1 {
                    UnsafeMutableRawPointer(currRowDst.advanced(by: x)).storeBytes(of: pCurr, as: SIMD16<Int16>.self)
                    x &+= 16
                    continue
                }
            }

            // Layer 1: Conv 3x3 (4ch -> 8ch) + LIF Forward (T=2)
            var sum0 = SIMD16<Int32>.zero
            var sum1 = SIMD16<Int32>.zero
            var sum2 = SIMD16<Int32>.zero
            var sum3 = SIMD16<Int32>.zero
            var sum4 = SIMD16<Int32>.zero
            var sum5 = SIMD16<Int32>.zero
            var sum6 = SIMD16<Int32>.zero
            var sum7 = SIMD16<Int32>.zero

            @inline(__always)
            func accumCh(
                _ in_0: SIMD16<Int16>, _ in_1: SIMD16<Int16>, _ in_2: SIMD16<Int16>,
                _ in_3: SIMD16<Int16>, _ in_4: SIMD16<Int16>, _ in_5: SIMD16<Int16>,
                _ in_6: SIMD16<Int16>, _ in_7: SIMD16<Int16>, _ in_8: SIMD16<Int16>,
                baseIdx: Int
            ) {
                let maskOr = in_0 | in_1 | in_2 | in_3 | in_4 | in_5 | in_6 | in_7 | in_8
                if maskOr == SIMD16<Int16>.zero {
                    return
                }

                let k0 = c1InW[baseIdx + 0]
                let k1 = c1InW[baseIdx + 1]
                let k2 = c1InW[baseIdx + 2]
                let k3 = c1InW[baseIdx + 3]
                let k4 = c1InW[baseIdx + 4]
                let k5 = c1InW[baseIdx + 5]
                let k6 = c1InW[baseIdx + 6]
                let k7 = c1InW[baseIdx + 7]
                let k8 = c1InW[baseIdx + 8]

                let v0 = SIMD16<Int32>(truncatingIfNeeded: in_0)
                let v1 = SIMD16<Int32>(truncatingIfNeeded: in_1)
                let v2 = SIMD16<Int32>(truncatingIfNeeded: in_2)
                let v3 = SIMD16<Int32>(truncatingIfNeeded: in_3)
                let v4 = SIMD16<Int32>(truncatingIfNeeded: in_4)
                let v5 = SIMD16<Int32>(truncatingIfNeeded: in_5)
                let v6 = SIMD16<Int32>(truncatingIfNeeded: in_6)
                let v7 = SIMD16<Int32>(truncatingIfNeeded: in_7)
                let v8 = SIMD16<Int32>(truncatingIfNeeded: in_8)

                sum0 &+= v0 &* k0.w0 &+ v1 &* k1.w0 &+ v2 &* k2.w0 &+ v3 &* k3.w0 &+ v4 &* k4.w0 &+ v5 &* k5.w0 &+ v6 &* k6.w0 &+ v7 &* k7.w0 &+ v8 &* k8.w0
                sum1 &+= v0 &* k0.w1 &+ v1 &* k1.w1 &+ v2 &* k2.w1 &+ v3 &* k3.w1 &+ v4 &* k4.w1 &+ v5 &* k5.w1 &+ v6 &* k6.w1 &+ v7 &* k7.w1 &+ v8 &* k8.w1
                sum2 &+= v0 &* k0.w2 &+ v1 &* k1.w2 &+ v2 &* k2.w2 &+ v3 &* k3.w2 &+ v4 &* k4.w2 &+ v5 &* k5.w2 &+ v6 &* k6.w2 &+ v7 &* k7.w2 &+ v8 &* k8.w2
                sum3 &+= v0 &* k0.w3 &+ v1 &* k1.w3 &+ v2 &* k2.w3 &+ v3 &* k3.w3 &+ v4 &* k4.w3 &+ v5 &* k5.w3 &+ v6 &* k6.w3 &+ v7 &* k7.w3 &+ v8 &* k8.w3
                sum4 &+= v0 &* k0.w4 &+ v1 &* k1.w4 &+ v2 &* k2.w4 &+ v3 &* k3.w4 &+ v4 &* k4.w4 &+ v5 &* k5.w4 &+ v6 &* k6.w4 &+ v7 &* k7.w4 &+ v8 &* k8.w4
                sum5 &+= v0 &* k0.w5 &+ v1 &* k1.w5 &+ v2 &* k2.w5 &+ v3 &* k3.w5 &+ v4 &* k4.w5 &+ v5 &* k5.w5 &+ v6 &* k6.w5 &+ v7 &* k7.w5 &+ v8 &* k8.w5
                sum6 &+= v0 &* k0.w6 &+ v1 &* k1.w6 &+ v2 &* k2.w6 &+ v3 &* k3.w6 &+ v4 &* k4.w6 &+ v5 &* k5.w6 &+ v6 &* k6.w6 &+ v7 &* k7.w6 &+ v8 &* k8.w6
                sum7 &+= v0 &* k0.w7 &+ v1 &* k1.w7 &+ v2 &* k2.w7 &+ v3 &* k3.w7 &+ v4 &* k4.w7 &+ v5 &* k5.w7 &+ v6 &* k6.w7 &+ v7 &* k7.w7 &+ v8 &* k8.w7
            }

            // Channel 0: Centered pixel
            accumCh(
                UnsafeRawPointer(topCh0.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(topCh0.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(topCh0.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(midCh0.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(midCh0.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(midCh0.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(botCh0.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(botCh0.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(botCh0.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self),
                baseIdx: 0
            )

            // Channel 1: 2nd-order Laplacian
            accumCh(
                UnsafeRawPointer(topCh1.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(topCh1.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(topCh1.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(midCh1.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(midCh1.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(midCh1.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(botCh1.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(botCh1.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(botCh1.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self),
                baseIdx: 9
            )

            // Channel 2: Local gradient magnitude
            accumCh(
                UnsafeRawPointer(topCh2.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(topCh2.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(topCh2.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(midCh2.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(midCh2.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(midCh2.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(botCh2.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(botCh2.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(botCh2.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self),
                baseIdx: 18
            )

            // Channel 3: Boundary distance
            accumCh(
                UnsafeRawPointer(topCh3.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(topCh3.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(topCh3.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(midCh3.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(midCh3.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(midCh3.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(botCh3.advanced(by: x - 1)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(botCh3.advanced(by: x)).loadUnaligned(as: SIMD16<Int16>.self),
                UnsafeRawPointer(botCh3.advanced(by: x + 1)).loadUnaligned(as: SIMD16<Int16>.self),
                baseIdx: 27
            )

            // Layer 1 LIF Forward
            let i1_0 = SIMD16<Int16>(truncatingIfNeeded: (sum0 &+ fourVec32) &>> shift3Vec32) &+ cb1_0
            let spk1_0_0 = vThresh1Vec .<= i1_0
            let u1Reset0_0 = i1_0.replacing(with: i1_0 &- vThresh1Vec, where: spk1_0_0)
            let spk1_1_0 = vThresh1Vec .<= (u1Reset0_0 &- (u1Reset0_0 &>> leakShiftVec) &+ i1_0)

            let i1_1 = SIMD16<Int16>(truncatingIfNeeded: (sum1 &+ fourVec32) &>> shift3Vec32) &+ cb1_1
            let spk1_0_1 = vThresh1Vec .<= i1_1
            let u1Reset0_1 = i1_1.replacing(with: i1_1 &- vThresh1Vec, where: spk1_0_1)
            let spk1_1_1 = vThresh1Vec .<= (u1Reset0_1 &- (u1Reset0_1 &>> leakShiftVec) &+ i1_1)

            let i1_2 = SIMD16<Int16>(truncatingIfNeeded: (sum2 &+ fourVec32) &>> shift3Vec32) &+ cb1_2
            let spk1_0_2 = vThresh1Vec .<= i1_2
            let u1Reset0_2 = i1_2.replacing(with: i1_2 &- vThresh1Vec, where: spk1_0_2)
            let spk1_1_2 = vThresh1Vec .<= (u1Reset0_2 &- (u1Reset0_2 &>> leakShiftVec) &+ i1_2)

            let i1_3 = SIMD16<Int16>(truncatingIfNeeded: (sum3 &+ fourVec32) &>> shift3Vec32) &+ cb1_3
            let spk1_0_3 = vThresh1Vec .<= i1_3
            let u1Reset0_3 = i1_3.replacing(with: i1_3 &- vThresh1Vec, where: spk1_0_3)
            let spk1_1_3 = vThresh1Vec .<= (u1Reset0_3 &- (u1Reset0_3 &>> leakShiftVec) &+ i1_3)

            let i1_4 = SIMD16<Int16>(truncatingIfNeeded: (sum4 &+ fourVec32) &>> shift3Vec32) &+ cb1_4
            let spk1_0_4 = vThresh1Vec .<= i1_4
            let u1Reset0_4 = i1_4.replacing(with: i1_4 &- vThresh1Vec, where: spk1_0_4)
            let spk1_1_4 = vThresh1Vec .<= (u1Reset0_4 &- (u1Reset0_4 &>> leakShiftVec) &+ i1_4)

            let i1_5 = SIMD16<Int16>(truncatingIfNeeded: (sum5 &+ fourVec32) &>> shift3Vec32) &+ cb1_5
            let spk1_0_5 = vThresh1Vec .<= i1_5
            let u1Reset0_5 = i1_5.replacing(with: i1_5 &- vThresh1Vec, where: spk1_0_5)
            let spk1_1_5 = vThresh1Vec .<= (u1Reset0_5 &- (u1Reset0_5 &>> leakShiftVec) &+ i1_5)

            let i1_6 = SIMD16<Int16>(truncatingIfNeeded: (sum6 &+ fourVec32) &>> shift3Vec32) &+ cb1_6
            let spk1_0_6 = vThresh1Vec .<= i1_6
            let u1Reset0_6 = i1_6.replacing(with: i1_6 &- vThresh1Vec, where: spk1_0_6)
            let spk1_1_6 = vThresh1Vec .<= (u1Reset0_6 &- (u1Reset0_6 &>> leakShiftVec) &+ i1_6)

            let i1_7 = SIMD16<Int16>(truncatingIfNeeded: (sum7 &+ fourVec32) &>> shift3Vec32) &+ cb1_7
            let spk1_0_7 = vThresh1Vec .<= i1_7
            let u1Reset0_7 = i1_7.replacing(with: i1_7 &- vThresh1Vec, where: spk1_0_7)
            let spk1_1_7 = vThresh1Vec .<= (u1Reset0_7 &- (u1Reset0_7 &>> leakShiftVec) &+ i1_7)

            let anySpk1_0 = spk1_0_0 .| spk1_0_1 .| spk1_0_2 .| spk1_0_3 .| spk1_0_4 .| spk1_0_5 .| spk1_0_6 .| spk1_0_7
            let anySpk1_1 = spk1_1_0 .| spk1_1_1 .| spk1_1_2 .| spk1_1_3 .| spk1_1_4 .| spk1_1_5 .| spk1_1_6 .| spk1_1_7
            let anySpk = anySpk1_0 .| anySpk1_1

            if anySpk == SIMDMask<SIMD16<Int16>.MaskStorage>(repeating: false) {
                UnsafeMutableRawPointer(currRowDst.advanced(by: x)).storeBytes(of: pCurr, as: SIMD16<Int16>.self)
                x &+= 16
                continue
            }

            // Fused Layer 2 (Conv 1x1 8ch->8ch) + Layer 3 (Output Acc) with zero vector spills
            var acc = outBDouble

            @inline(__always)
            func processL2Channel(j: Int, cb2: SIMD16<Int16>) {
                var syn0 = cb2
                var syn1 = cb2
                let baseIdx = j * 8

                syn0 &+= zVec.replacing(with: c2W[baseIdx + 0], where: spk1_0_0)
                syn1 &+= zVec.replacing(with: c2W[baseIdx + 0], where: spk1_1_0)

                syn0 &+= zVec.replacing(with: c2W[baseIdx + 1], where: spk1_0_1)
                syn1 &+= zVec.replacing(with: c2W[baseIdx + 1], where: spk1_1_1)

                syn0 &+= zVec.replacing(with: c2W[baseIdx + 2], where: spk1_0_2)
                syn1 &+= zVec.replacing(with: c2W[baseIdx + 2], where: spk1_1_2)

                syn0 &+= zVec.replacing(with: c2W[baseIdx + 3], where: spk1_0_3)
                syn1 &+= zVec.replacing(with: c2W[baseIdx + 3], where: spk1_1_3)

                syn0 &+= zVec.replacing(with: c2W[baseIdx + 4], where: spk1_0_4)
                syn1 &+= zVec.replacing(with: c2W[baseIdx + 4], where: spk1_1_4)

                syn0 &+= zVec.replacing(with: c2W[baseIdx + 5], where: spk1_0_5)
                syn1 &+= zVec.replacing(with: c2W[baseIdx + 5], where: spk1_1_5)

                syn0 &+= zVec.replacing(with: c2W[baseIdx + 6], where: spk1_0_6)
                syn1 &+= zVec.replacing(with: c2W[baseIdx + 6], where: spk1_1_6)

                syn0 &+= zVec.replacing(with: c2W[baseIdx + 7], where: spk1_0_7)
                syn1 &+= zVec.replacing(with: c2W[baseIdx + 7], where: spk1_1_7)

                let spk2_0 = vThresh2Vec .<= syn0
                let u2Reset0 = syn0.replacing(with: syn0 &- vThresh2Vec, where: spk2_0)
                let spk2_1 = vThresh2Vec .<= (u2Reset0 &- (u2Reset0 &>> leakShiftVec) &+ syn1)

                acc &+= zVec.replacing(with: oW[j], where: spk2_0)
                acc &+= zVec.replacing(with: oW[j], where: spk2_1)
            }

            processL2Channel(j: 0, cb2: cb2_0)
            processL2Channel(j: 1, cb2: cb2_1)
            processL2Channel(j: 2, cb2: cb2_2)
            processL2Channel(j: 3, cb2: cb2_3)
            processL2Channel(j: 4, cb2: cb2_4)
            processL2Channel(j: 5, cb2: cb2_5)
            processL2Channel(j: 6, cb2: cb2_6)
            processL2Channel(j: 7, cb2: cb2_7)

            // Output scaling & clamping
            let delta = (acc &+ eightVec16) &>> shift4Vec16
            let clampedDelta = delta
                .replacing(with: minDelta, where: delta .< minDelta)
                .replacing(with: maxDelta, where: maxDelta .< delta)

            let pRecon = pCurr &+ clampedDelta
            let pFinal = pRecon
                .replacing(with: minPixel, where: pRecon .< minPixel)
                .replacing(with: maxPixel, where: maxPixel .< pRecon)

            UnsafeMutableRawPointer(currRowDst.advanced(by: x)).storeBytes(of: pFinal, as: SIMD16<Int16>.self)

            x &+= 16
        }

        // Handle trailing columns if width is not a multiple of 16
        if wFast16 < width {
            processRowScalarTail(
                src: src,
                dstRow: currRowDst,
                width: width,
                height: height,
                y: y,
                startX: wFast16,
                endX: width,
                blockSize: blockSize,
                skipMask: skipMask,
                numBlocksX: numBlocksX
            )
        }

        y &+= 1
    }
}

// MARK: - Scalar Reference Processing (For Verification & Boundary Cases)

private func processRowScalarTail(
    src: UnsafePointer<Int16>,
    dstRow: UnsafeMutablePointer<Int16>,
    width: Int,
    height: Int,
    y: Int,
    startX: Int,
    endX: Int,
    blockSize: Int,
    skipMask: UnsafePointer<UInt8>? = nil,
    numBlocksX: Int = 1
) {
    let by = y / blockSize
    let byOffset = by * numBlocksX

    let conv1W = SNNWeights.conv1Weights
    let conv1B = SNNWeights.conv1Biases
    let conv2W = SNNWeights.conv2Weights
    let conv2B = SNNWeights.conv2Biases
    let outW = SNNWeights.outWeights
    let outB = SNNWeights.outBias
    let vThresh1 = SNNWeights.vThresh1
    let vThresh2 = SNNWeights.vThresh2
    let leakShift = SNNWeights.leakShift

    for x in startX..<endX {
        if let sm = skipMask {
            let bx = x / blockSize
            if sm[byOffset + bx] != 0 {
                dstRow[x] = src[(y * width) + x]
                continue
            }
        }

        // Extract 3x3 features for 4 channels
        var feat3x3 = [[[Int16]]](
            repeating: [[Int16]](repeating: [Int16](repeating: 0, count: 3), count: 3),
            count: 4
        )

        for ky in -1...1 {
            let iy = max(0, min(height - 1, y + ky))
            let yPrev = max(0, iy - 1)
            let yNext = min(height - 1, iy + 1)
            let yMod = iy % blockSize
            let distY = Int16(min(yMod, blockSize - yMod))

            for kx in -1...1 {
                let ix = max(0, min(width - 1, x + kx))
                let xPrev = max(0, ix - 1)
                let xNext = min(width - 1, ix + 1)

                let centerVal = src[(iy * width) + ix]
                let topVal = src[(yPrev * width) + ix]
                let bottomVal = src[(yNext * width) + ix]
                let leftVal = src[(iy * width) + xPrev]
                let rightVal = src[(iy * width) + xNext]

                let f0 = centerVal
                let f1 = (4 &* centerVal) &- topVal &- bottomVal &- leftVal &- rightVal
                let diffH = rightVal &- leftVal
                let absH: Int16
                if diffH < 0 {
                    absH = 0 &- diffH
                } else {
                    absH = diffH
                }
                let diffV = bottomVal &- topVal
                let absV: Int16
                if diffV < 0 {
                    absV = 0 &- diffV
                } else {
                    absV = diffV
                }
                let f2 = absH &+ absV
                let xMod = ix % blockSize
                let distX = Int16(min(xMod, blockSize - xMod))
                let f3 = (distX &+ distY) &<< 2

                feat3x3[0][ky + 1][kx + 1] = f0
                feat3x3[1][ky + 1][kx + 1] = f1
                feat3x3[2][ky + 1][kx + 1] = f2
                feat3x3[3][ky + 1][kx + 1] = f3
            }
        }

        // Layer 1 Conv 3x3 (4ch -> 8ch)
        var i1 = [Int16](repeating: 0, count: 8)
        for outCh in 0..<8 {
            var sum32: Int32 = 0
            for inC in 0..<4 {
                let wOffset = (outCh * 4 + inC) * 9
                for ky in 0..<3 {
                    for kx in 0..<3 {
                        let w = Int32(conv1W[wOffset + (ky * 3) + kx])
                        let f = Int32(feat3x3[inC][ky][kx])
                        sum32 &+= f &* w
                    }
                }
            }
            let shifted = (sum32 &+ 4) &>> 3
            i1[outCh] = Int16(truncatingIfNeeded: shifted) &+ conv1B[outCh]
        }

        // Layer 1 LIF (T=2)
        var spk1_0 = [Bool](repeating: false, count: 8)
        var u1Reset0 = [Int16](repeating: 0, count: 8)
        for outCh in 0..<8 {
            let u0 = i1[outCh]
            let spk = vThresh1 <= u0
            spk1_0[outCh] = spk
            var uReset = u0
            if spk {
                uReset &-= vThresh1
            }
            u1Reset0[outCh] = uReset
        }

        var spk1_1 = [Bool](repeating: false, count: 8)
        for outCh in 0..<8 {
            let uDecay = u1Reset0[outCh] &- (u1Reset0[outCh] &>> leakShift)
            let u1 = uDecay &+ i1[outCh]
            spk1_1[outCh] = vThresh1 <= u1
        }

        // Layer 2 Conv 1x1 + LIF (T=2)
        var spk2_0 = [Bool](repeating: false, count: 8)
        var u2Reset0 = [Int16](repeating: 0, count: 8)
        for outCh in 0..<8 {
            var syn0 = conv2B[outCh]
            let wOffset = outCh * 8
            for inC in 0..<8 {
                if spk1_0[inC] {
                    syn0 &+= Int16(conv2W[wOffset + inC]) &<< 4
                }
            }
            let u0 = syn0
            let spk = vThresh2 <= u0
            spk2_0[outCh] = spk
            var uReset = u0
            if spk {
                uReset &-= vThresh2
            }
            u2Reset0[outCh] = uReset
        }

        var spk2_1 = [Bool](repeating: false, count: 8)
        for outCh in 0..<8 {
            var syn1 = conv2B[outCh]
            let wOffset = outCh * 8
            for inC in 0..<8 {
                if spk1_1[inC] {
                    syn1 &+= Int16(conv2W[wOffset + inC]) &<< 4
                }
            }
            let uDecay = u2Reset0[outCh] &- (u2Reset0[outCh] &>> leakShift)
            let u1 = uDecay &+ syn1
            spk2_1[outCh] = vThresh2 <= u1
        }

        // Layer 3 Linear Accumulator (8ch -> 1ch, T=2)
        var acc: Int32 = 0
        acc &+= Int32(outB)
        for inC in 0..<8 {
            if spk2_0[inC] {
                acc &+= Int32(outW[inC])
            }
        }
        acc &+= Int32(outB)
        for inC in 0..<8 {
            if spk2_1[inC] {
                acc &+= Int32(outW[inC])
            }
        }

        let delta = Int16(truncatingIfNeeded: (acc &+ 8) &>> 4)
        let clampedDelta = max(-16, min(16, delta))
        let pOrig = src[(y * width) + x]
        let pFinal = max(-128, min(127, pOrig &+ clampedDelta))

        dstRow[x] = pFinal
    }
}

private func applySNNNeuralLoopFilterScalar(
    plane: inout [Int16],
    width: Int,
    height: Int,
    blockSize: Int,
    skipMask: [UInt8]? = nil
) {
    let count = width * height
    var outPlane = [Int16](repeating: 0, count: count)
    let numBlocksX = (width + blockSize - 1) / blockSize
    let totalBlocks = numBlocksX * ((height + blockSize - 1) / blockSize)

    plane.withUnsafeBufferPointer { srcBuf in
        outPlane.withUnsafeMutableBufferPointer { dstBuf in
            let srcPtr = srcBuf.baseAddress!
            let dstPtr = dstBuf.baseAddress!
            let skipPtr: UnsafePointer<UInt8>?
            if let sm = skipMask, sm.count == totalBlocks {
                skipPtr = sm.withUnsafeBufferPointer { $0.baseAddress }
            } else {
                skipPtr = nil
            }
            for y in 0..<height {
                processRowScalarTail(
                    src: srcPtr,
                    dstRow: dstPtr.advanced(by: y * width),
                    width: width,
                    height: height,
                    y: y,
                    startX: 0,
                    endX: width,
                    blockSize: blockSize,
                    skipMask: skipPtr,
                    numBlocksX: numBlocksX
                )
            }
        }
    }

    plane = outPlane
}
