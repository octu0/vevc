// MARK: - σ-Measurement Coefficient Dump (VEVC_DUMP_COEFFS)
//
// Measurement-only instrumentation: when the VEVC_DUMP_COEFFS environment
// variable is set to a file path, the encoder appends one binary record per
// P-frame with
//   (a) CODED: the coded quantized subband planes (post entropy-encode state,
//       i.e. exactly what the decoder reconstructs),
//   (b) PARENT: the quadrant planes of the *reconstructed* parent blocks,
//       which is what the entropy coder's isParentZero actually reads,
//   (c) PYR: the unquantized DWT pyramid of the previous reconstructed frame
//       (decoder-available temporal context, recomputed from pixels).
// SigmaMeasure.swift reads this file to evaluate σ-conditioned entropy models
// offline. Zero overhead when the environment variable is not set.
//
// Record layout (Int32 little-endian, coefficients Int16 host-endian):
//   "FRAM" gop width height
//   18 × qstep   (tables Y2,C2,Y1,C1,Y0,C0 × [qLow, qMid, qHigh])
//   3  × layerBytes (L0, L1, L2)
//   CODED  group: 9 entries (L2 Y,Cb,Cr / L1 Y,Cb,Cr / L0 Y,Cb,Cr)
//   PARENT group: 6 entries (recon L1 quadrants Y,Cb,Cr / recon L0 quadrants Y,Cb,Cr)
//   PYR    group: 9 entries (same order and shape as CODED)
//   entry := nSub, then per subband: w h data[w*h]
//   subband order: 3-sub entries = HL,LH,HH ; 4-sub entries = LL,HL,LH,HH

import Foundation

struct DumpSubPlane {
    let w: Int
    let h: Int
    var data: [Int16]
}

final class CoeffDumper: @unchecked Sendable {
    static let shared: CoeffDumper? = {
        guard let path = ProcessInfo.processInfo.environment["VEVC_DUMP_COEFFS"], path.isEmpty != true else { return nil }
        return CoeffDumper(path: path)
    }()

    private let handle: FileHandle
    private let lock = NSLock()
    private var pending: [String: [DumpSubPlane]] = [:]

    private init?(path: String) {
        FileManager.default.createFile(atPath: path, contents: nil, attributes: nil)
        guard let h = FileHandle(forWritingAtPath: path) else { return nil }
        self.handle = h
        h.write(Data("VSD1".utf8))
    }

    /// Assemble per-subband planes from a block grid. Quadrant layout inside a
    /// block: LL top-left, HL top-right, LH bottom-left, HH bottom-right.
    static func assembleQuadrants(blocks: [BlockView], planeW: Int, planeH: Int, blockSize: Int, includeLL: Bool) -> [DumpSubPlane] {
        let q = blockSize / 2
        let colCount = (planeW + blockSize - 1) / blockSize
        let rowCount = (planeH + blockSize - 1) / blockSize
        let gw = colCount * q
        let gh = rowCount * q
        let offsets: [(x: Int, y: Int)] = includeLL ? [(0, 0), (q, 0), (0, q), (q, q)] : [(q, 0), (0, q), (q, q)]
        var planes = offsets.map { _ in DumpSubPlane(w: gw, h: gh, data: [Int16](repeating: 0, count: gw * gh)) }
        for r in 0..<rowCount {
            for c in 0..<colCount {
                let blk = blocks[r * colCount + c]
                for (pi, off) in offsets.enumerated() {
                    planes[pi].data.withUnsafeMutableBufferPointer { dst in
                        let dBase = dst.baseAddress!
                        for y in 0..<q {
                            let src = blk.base.advanced(by: (off.y + y) * blk.stride + off.x)
                            dBase.advanced(by: (r * q + y) * gw + c * q).update(from: src, count: q)
                        }
                    }
                }
            }
        }
        return planes
    }

    /// One DWT level over a plane: full-block LeGall 5/3 (same code path as the
    /// encoder), returning the AC quadrant grids and the gathered LL plane.
    private static func dwtLevelPlanes(_ plane: [Int16], w: Int, h: Int, blockSize: Int, includeLL: Bool) -> (subs: [DumpSubPlane], ll: [Int16], llW: Int, llH: Int) {
        let q = blockSize / 2
        let colCount = (w + blockSize - 1) / blockSize
        let rowCount = (h + blockSize - 1) / blockSize
        let gw = colCount * q
        let gh = rowCount * q
        let offsets: [(x: Int, y: Int)] = includeLL ? [(0, 0), (q, 0), (0, q), (q, q)] : [(q, 0), (0, q), (q, q)]
        var subs = offsets.map { _ in DumpSubPlane(w: gw, h: gh, data: [Int16](repeating: 0, count: gw * gh)) }
        let llW = (w + 1) / 2
        let llH = (h + 1) / 2
        var ll = [Int16](repeating: 0, count: llW * llH)
        var scratch = [Int16](repeating: 0, count: blockSize * blockSize)
        let reader = Int16Reader(data: plane, width: w, height: h)
        for r in 0..<rowCount {
            for c in 0..<colCount {
                scratch.withUnsafeMutableBufferPointer { sb in
                    let sBase = sb.baseAddress!
                    let view = BlockView(base: sBase, width: blockSize, height: blockSize, stride: blockSize)
                    reader.readBlock(x: c * blockSize, y: r * blockSize, width: blockSize, height: blockSize, into: view)
                    switch blockSize {
                    case 32: dwt2DBlock32(view)
                    case 16: dwt2DBlock16(view)
                    default: dwt2DBlock8(view)
                    }
                    for (pi, off) in offsets.enumerated() {
                        subs[pi].data.withUnsafeMutableBufferPointer { dst in
                            let dBase = dst.baseAddress!
                            for y in 0..<q {
                                let src = sBase.advanced(by: (off.y + y) * blockSize + off.x)
                                dBase.advanced(by: (r * q + y) * gw + c * q).update(from: src, count: q)
                            }
                        }
                    }
                    ll.withUnsafeMutableBufferPointer { dst in
                        let dBase = dst.baseAddress!
                        let dx0 = c * q
                        let dy0 = r * q
                        let copyW = min(q, llW - dx0)
                        if 0 < copyW {
                            for y in 0..<q {
                                let dy = dy0 + y
                                if dy < llH {
                                    let src = sBase.advanced(by: y * blockSize)
                                    dBase.advanced(by: dy * llW + dx0).update(from: src, count: copyW)
                                }
                            }
                        }
                    }
                }
            }
        }
        return (subs, ll, llW, llH)
    }

    /// Full 3-level pyramid mirroring the encoder's layer structure
    /// (32-block level → 16-block level on LL → 8-block level on LL²).
    private static func pyramid(_ plane: [Int16], w: Int, h: Int) -> [[DumpSubPlane]] {
        let l2 = dwtLevelPlanes(plane, w: w, h: h, blockSize: 32, includeLL: false)
        let l1 = dwtLevelPlanes(l2.ll, w: l2.llW, h: l2.llH, blockSize: 16, includeLL: false)
        let l0 = dwtLevelPlanes(l1.ll, w: l1.llW, h: l1.llH, blockSize: 8, includeLL: true)
        return [l2.subs, l1.subs, l0.subs]
    }

    /// Capture a layer/plane's quantized blocks. Must be called after the
    /// entropy encode of that layer (zero-block clearing applied) and before
    /// its reconstruction (which dequantizes the blocks in place).
    func stash(_ key: String, blocks: [BlockView], planeW: Int, planeH: Int, blockSize: Int, includeLL: Bool) {
        let planes = Self.assembleQuadrants(blocks: blocks, planeW: planeW, planeH: planeH, blockSize: blockSize, includeLL: includeLL)
        lock.lock()
        pending[key] = planes
        lock.unlock()
    }

    func finalizePFrame(
        gopPosition: Int, width: Int, height: Int,
        predictedPd: PlaneData420,
        l1yBlocks: [BlockView], l1cbBlocks: [BlockView], l1crBlocks: [BlockView],
        b8yBlocks: [BlockView], b8cbBlocks: [BlockView], b8crBlocks: [BlockView],
        sub2W: Int, sub2H: Int, sub1W: Int, sub1H: Int,
        qtY2: QuantizationTable, qtC2: QuantizationTable,
        qtY1: QuantizationTable, qtC1: QuantizationTable,
        qtY0: QuantizationTable, qtC0: QuantizationTable,
        layer0Bytes: Int, layer1Bytes: Int, layer2Bytes: Int
    ) {
        // Parent planes: quadrant content of the reconstructed parent blocks,
        // matching the state the entropy coder read for isParentZero.
        let l1cbW = (sub2W + 1) / 2
        let l1cbH = (sub2H + 1) / 2
        let l0cbW = (sub1W + 1) / 2
        let l0cbH = (sub1H + 1) / 2
        let parents: [[DumpSubPlane]] = [
            Self.assembleQuadrants(blocks: l1yBlocks, planeW: sub2W, planeH: sub2H, blockSize: 16, includeLL: false),
            Self.assembleQuadrants(blocks: l1cbBlocks, planeW: l1cbW, planeH: l1cbH, blockSize: 16, includeLL: false),
            Self.assembleQuadrants(blocks: l1crBlocks, planeW: l1cbW, planeH: l1cbH, blockSize: 16, includeLL: false),
            Self.assembleQuadrants(blocks: b8yBlocks, planeW: sub1W, planeH: sub1H, blockSize: 8, includeLL: false),
            Self.assembleQuadrants(blocks: b8cbBlocks, planeW: l0cbW, planeH: l0cbH, blockSize: 8, includeLL: false),
            Self.assembleQuadrants(blocks: b8crBlocks, planeW: l0cbW, planeH: l0cbH, blockSize: 8, includeLL: false),
        ]

        let cbW = (width + 1) / 2
        let cbH = (height + 1) / 2
        let pyrY = Self.pyramid(predictedPd.y, w: width, h: height)
        let pyrCb = Self.pyramid(predictedPd.cb, w: cbW, h: cbH)
        let pyrCr = Self.pyramid(predictedPd.cr, w: cbW, h: cbH)
        let pyr: [[DumpSubPlane]] = [
            pyrY[0], pyrCb[0], pyrCr[0],
            pyrY[1], pyrCb[1], pyrCr[1],
            pyrY[2], pyrCb[2], pyrCr[2],
        ]

        lock.lock()
        defer { lock.unlock() }
        let codedKeys = ["L2Y", "L2Cb", "L2Cr", "L1Y", "L1Cb", "L1Cr", "L0Y", "L0Cb", "L0Cr"]
        var coded: [[DumpSubPlane]] = []
        coded.reserveCapacity(codedKeys.count)
        for key in codedKeys {
            guard let planes = pending[key] else { return }
            coded.append(planes)
        }
        pending.removeAll(keepingCapacity: true)

        var buf = [UInt8]()
        buf.reserveCapacity(1 << 23)
        buf.append(contentsOf: Array("FRAM".utf8))
        put32(gopPosition, &buf)
        put32(width, &buf)
        put32(height, &buf)
        for qt in [qtY2, qtC2, qtY1, qtC1, qtY0, qtC0] {
            put32(Int(qt.qLow.step), &buf)
            put32(Int(qt.qMid.step), &buf)
            put32(Int(qt.qHigh.step), &buf)
        }
        put32(layer0Bytes, &buf)
        put32(layer1Bytes, &buf)
        put32(layer2Bytes, &buf)
        for group in [coded, parents, pyr] {
            for entry in group {
                put32(entry.count, &buf)
                for p in entry {
                    put32(p.w, &buf)
                    put32(p.h, &buf)
                    p.data.withUnsafeBufferPointer { src in
                        buf.append(contentsOf: UnsafeRawBufferPointer(src))
                    }
                }
            }
        }
        handle.write(Data(buf))
    }

    @inline(__always)
    private func put32(_ v: Int, _ buf: inout [UInt8]) {
        let u = UInt32(truncatingIfNeeded: v)
        buf.append(UInt8(u & 0xff))
        buf.append(UInt8((u >> 8) & 0xff))
        buf.append(UInt8((u >> 16) & 0xff))
        buf.append(UInt8((u >> 24) & 0xff))
    }
}
