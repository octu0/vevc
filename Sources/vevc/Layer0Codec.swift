//
//  Layer0Codec.swift
//  vevc
//

import Foundation

protocol Layer0Codec: Sendable {
    func encode(
        pd: PlaneData420,
        pool: BlockViewPool,
        sads: [Int]?,
        occlusionScores: [Int]?,
        layer: UInt8,
        qtY: QuantizationTable,
        qtC: QuantizationTable,
        zeroThreshold: Int
    ) async throws -> (
        encodedBytes: [UInt8],
        reconstructed: PlaneData420,
        yBlocks: [BlockView],
        cbBlocks: [BlockView],
        crBlocks: [BlockView],
        releaseFn: @Sendable () -> Void
    )

    func decode(
        r: [UInt8],
        pool: BlockViewPool,
        layer: UInt8,
        dx: Int,
        dy: Int,
        isIFrame: Bool
    ) async throws -> (
        reconstructed: Image16,
        yBlocks: [BlockView],
        cbBlocks: [BlockView],
        crBlocks: [BlockView],
        qtYStep: Int,
        qtCStep: Int
    )
}

struct Layer0CodecFactory {
    static func create(profile: UInt8) -> any Layer0Codec {
        switch profile {
        case 0x01:
            return Layer0DWTCodec()
        case 0x02:
            return Layer0Profile2Codec()
        default:
            return Layer0DWTCodec()
        }
    }
}

final class Layer0Profile2Codec: Layer0Codec {
    let dwtCodec = Layer0DWTCodec()
    
    private static func deriveDCTStep(qstep: Int) -> Int {
        return max(1, qstep / 16)
    }
    
    func encode(
        pd: PlaneData420,
        pool: BlockViewPool,
        sads: [Int]?,
        occlusionScores: [Int]?,
        layer: UInt8,
        qtY: QuantizationTable,
        qtC: QuantizationTable,
        zeroThreshold: Int
    ) async throws -> (
        encodedBytes: [UInt8],
        reconstructed: PlaneData420,
        yBlocks: [BlockView],
        cbBlocks: [BlockView],
        crBlocks: [BlockView],
        releaseFn: @Sendable () -> Void
    ) {
        let isIFrame = (sads == nil)
        if !isIFrame {
            return try await dwtCodec.encode(pd: pd, pool: pool, sads: sads, occlusionScores: occlusionScores, layer: layer, qtY: qtY, qtC: qtC, zeroThreshold: zeroThreshold)
        }
        
        let dx = pd.width
        let dy = pd.height
        let cbDx = ((dx + 1) / 2)
        let cbDy = ((dy + 1) / 2)
        
        let stepY = Self.deriveDCTStep(qstep: Int(qtY.step))
        let stepC = Self.deriveDCTStep(qstep: Int(qtC.step))
        
        let encY = encodeL0PlaneDCT(plane: pd.y, width: dx, height: dy, stride: dx, step: stepY)
        let encCb = encodeL0PlaneDCT(plane: pd.cb, width: cbDx, height: cbDy, stride: cbDx, step: stepC)
        let encCr = encodeL0PlaneDCT(plane: pd.cr, width: cbDx, height: cbDy, stride: cbDx, step: stepC)
        
        let recY = try decodeL0PlaneDCT(bytes: encY.bytes, width: dx, height: dy, step: stepY)
        let recCb = try decodeL0PlaneDCT(bytes: encCb.bytes, width: cbDx, height: cbDy, step: stepC)
        let recCr = try decodeL0PlaneDCT(bytes: encCr.bytes, width: cbDx, height: cbDy, step: stepC)
        
        let reconstructed = PlaneData420(width: dx, height: dy, y: recY, cb: recCb, cr: recCr)
        
        debugLog({
            return "  [Layer \\(layer)/BaseDCT] Y=\\(encY.bytes.count) Cb=\\(encCb.bytes.count) Cr=\\(encCr.bytes.count) bytes"
        }())
        
        let out = VEVCLayerData.serialize(
            qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
            bufY: encY.bytes, bufCb: encCb.bytes, bufCr: encCr.bytes
        )
        
        return (out, reconstructed, [], [], [], { })
    }
    
    func decode(
        r: [UInt8],
        pool: BlockViewPool,
        layer: UInt8,
        dx: Int,
        dy: Int,
        isIFrame: Bool
    ) async throws -> (
        reconstructed: Image16,
        yBlocks: [BlockView],
        cbBlocks: [BlockView],
        crBlocks: [BlockView],
        qtYStep: Int,
        qtCStep: Int
    ) {
        if !isIFrame {
            return try await dwtCodec.decode(r: r, pool: pool, layer: layer, dx: dx, dy: dy, isIFrame: isIFrame)
        }
        
        let (qtY, qtC, bufY, bufCb, bufCr) = try VEVCLayerData.deserialize(from: r, layer: layer, layerLabel: "BaseDCT")
        
        let stepY = Self.deriveDCTStep(qstep: Int(qtY.step))
        let stepC = Self.deriveDCTStep(qstep: Int(qtC.step))
        
        let recY = try decodeL0PlaneDCT(bytes: bufY, width: dx, height: dy, step: stepY)
        let cbDx = (dx + 1) / 2
        let cbDy = (dy + 1) / 2
        let recCb = try decodeL0PlaneDCT(bytes: bufCb, width: cbDx, height: cbDy, step: stepC)
        let recCr = try decodeL0PlaneDCT(bytes: bufCr, width: cbDx, height: cbDy, step: stepC)
        
        var img = Image16(width: dx, height: dy, pool: pool)
        
        // Use withUnsafeMutableBufferPointer to copy efficiently
        img.y.withUnsafeMutableBufferPointer { ptr in
            _ = ptr.initialize(from: recY)
        }
        img.cb.withUnsafeMutableBufferPointer { ptr in
            _ = ptr.initialize(from: recCb)
        }
        img.cr.withUnsafeMutableBufferPointer { ptr in
            _ = ptr.initialize(from: recCr)
        }
        
        return (img, [], [], [], Int(qtY.step), Int(qtC.step))
    }
}

#if DEBUG
final class Layer0DebugTracker: @unchecked Sendable {
    static let shared = Layer0DebugTracker()
    var frameCount = 0
    var lastLLBytes = 0
    var lastHighBytes = 0
    let queue = DispatchQueue(label: "vevc.debug.dump")
}
#endif

@inline(__always)
func evaluateQuantizeBase8(view: BlockView, qt: QuantizationTable) {
    let subs = getSubbands8(view: view)
    let ll = subs.ll
    let hl = subs.hl
    let lh = subs.lh
    let hh = subs.hh
    quantizeSIMD4(ll, q: qt.qLow)
    quantizeSIMDSignedMapping4(hl, q: qt.qMid)
    quantizeSIMDSignedMapping4(lh, q: qt.qMid)
    quantizeSIMDSignedMapping4(hh, q: qt.qHigh)
}

@inline(__always)
func extractSingleTransformBlocksBase8(r: Int16Reader, width: Int, height: Int, pool: BlockViewPool) async -> ([BlockView], @Sendable () -> Void) {
    let rowCount = ((height + 8 - 1) / 8)
    let colCount = ((width + 8 - 1) / 8)
    let totalBlocks = rowCount * colCount
    
    var tmpBlocks = pool.getBlockViewArray(capacity: totalBlocks)
    tmpBlocks.reserveCapacity(totalBlocks)
    for _ in 0..<totalBlocks {
        tmpBlocks.append(pool.get(width: 8, height: 8))
    }
    let blocks = tmpBlocks
    
    let chunkSize = 4
    await withTaskGroup(of: Void.self) { group in
        for sRow in stride(from: 0, to: rowCount, by: chunkSize) {
            let endRow = min(sRow + chunkSize, rowCount)
            group.addTask { [blocks] in
                for i in sRow..<endRow {
                    let h = (i * 8)
                    for j in 0..<colCount {
                        let w = (j * 8)
                        if width <= w || height <= h { continue }
                        let view = blocks[(i * colCount) + j]
                        r.readBlock(x: w, y: h, width: 8, height: 8, into: view)
                        dwt2DBlock8(view)
                    }
                }
            }
        }
    }    
    return (tmpBlocks, { [tmpBlocks] in pool.putBlockViewArray(tmpBlocks) })
}

@inline(__always)
func reconstructPlaneBase8(blocks: [BlockView], width: Int, height: Int, qt: QuantizationTable, pool: BlockViewPool) -> ([Int16], @Sendable () -> Void) {
    let colCount = (width + 7) / 8
    let rowCount = (height + 7) / 8
    var plane = pool.getInt16(count: width * height)
    withUnsafePointers(mut: &plane) { dstBase in
        var idx = 0
        for row in 0..<rowCount {
            let startY = row * 8
            let validEndY = min(height, startY + 8)
            let loopH = validEndY - startY
            let isEdgeY = (loopH < 8)
            
            for col in 0..<colCount {
                let startX = col * 8
                let validEndX = min(width, startX + 8)
                let loopW = validEndX - startX
                let isEdgeX = (loopW < 8)
                
                let blk = blocks[idx]
                idx += 1
                
                let view = blk
                let base = view.base
                let llView = BlockView(base: base, width: 4, height: 4, stride: 8)
                let hlView = BlockView(base: base.advanced(by: 4), width: 4, height: 4, stride: 8)
                let lhView = BlockView(base: base.advanced(by: 32), width: 4, height: 4, stride: 8)
                let hhView = BlockView(base: base.advanced(by: 36), width: 4, height: 4, stride: 8)
                dequantizeSIMD4(llView, q: qt.qLow)
                dequantizeSIMDSignedMapping4(hlView, q: qt.qMid)
                dequantizeSIMDSignedMapping4(lhView, q: qt.qMid)
                dequantizeSIMDSignedMapping4(hhView, q: qt.qHigh)
                inverseDWT2DBlock8(view)
                            
                switch true {
                case isEdgeY != true && isEdgeX != true:
                    let v = blk
                    for h in 0..<8 {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: 8)
                    }
                case 0 < loopH && 0 < loopW:
                    let v = blk
                    for h in 0..<loopH {
                        let srcPtr = v.rowPointer(y: h)
                        let destPtr = dstBase.advanced(by: (startY + h) * width + startX)
                        destPtr.update(from: srcPtr, count: loopW)
                    }
                default:
                    break
                }
            }
        }
    }
    return (plane, { [plane] in pool.putInt16(plane) })
}

final class Layer0DWTCodec: Layer0Codec {
    init() {}

    func encode(
        pd: PlaneData420,
        pool: BlockViewPool,
        sads: [Int]?,
        occlusionScores: [Int]?,
        layer: UInt8,
        qtY: QuantizationTable,
        qtC: QuantizationTable,
        zeroThreshold: Int
    ) async throws -> (
        encodedBytes: [UInt8],
        reconstructed: PlaneData420,
        yBlocks: [BlockView],
        cbBlocks: [BlockView],
        crBlocks: [BlockView],
        releaseFn: @Sendable () -> Void
    ) {
        let dx = pd.width
        let dy = pd.height
        let cbDx = ((dx + 1) / 2)
        let cbDy = ((dy + 1) / 2)
        
        let yColCount8 = (dx + 7) / 8
        let yRowCount8 = (dy + 7) / 8

        #if DEBUG
        let isIFrame = (sads == nil)
        if isIFrame {
            Layer0DebugTracker.shared.queue.sync {
                if Layer0DebugTracker.shared.frameCount < 20 {
                    let f = Layer0DebugTracker.shared.frameCount
                    if layer == 0 {
                        let fnameY = "dump_Y_\(dx)x\(dy)_f\(f).raw"
                        let fnameCb = "dump_Cb_\(cbDx)x\(cbDy)_f\(f).raw"
                        let fnameCr = "dump_Cr_\(cbDx)x\(cbDy)_f\(f).raw"
                        pd.y.withUnsafeBytes { ptr in try? Data(ptr).write(to: URL(fileURLWithPath: fnameY)) }
                        pd.cb.withUnsafeBytes { ptr in try? Data(ptr).write(to: URL(fileURLWithPath: fnameCb)) }
                        pd.cr.withUnsafeBytes { ptr in try? Data(ptr).write(to: URL(fileURLWithPath: fnameCr)) }
                        Layer0DebugTracker.shared.frameCount += 1
                    }
                }
            }
        }
        #endif
        
        async let taskBufY = { () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
            #if DEBUG
            Layer0DebugTracker.shared.queue.sync { Layer0DebugTracker.shared.lastLLBytes = 0; Layer0DebugTracker.shared.lastHighBytes = 0 }
            #endif
            var (blocks, relBlocks) = await extractSingleTransformBlocksBase8(r: pd.rY, width: dx, height: dy, pool: pool)
            let isIFrame = (sads == nil)
            for i in blocks.indices {
                if let sList = sads, i < sList.count {
                    let col = i % yColCount8
                    let row = i / yColCount8
                    let threshold = spatialSADThreshold(baseSAD: scaledSADThreshold(150, step: (Int(qtY.step) + 8) >> 4), blockCol: col, blockRow: row, colCount: yColCount8, rowCount: yRowCount8)
                    if sList[i] < threshold { 
                        let b = blocks[i]
                        clearBlockRegion(base: b.base, width: b.width, height: b.height, stride: b.stride)
                    }
                }
                evaluateQuantizeBase8(view: blocks[i], qt: qtY)
            }
            
            let safeThreshold = min(1, min(zeroThreshold, max(0, Int(qtY.step) / 64)))
            let buf = if isIFrame != true {
                encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold)
            } else {
                encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold)
            }
            
            #if DEBUG
            Layer0DebugTracker.shared.queue.sync {
                let frameType = isIFrame ? "I" : "P"
                print("Layer0 Dump [\(frameType) Y] LL: \(Layer0DebugTracker.shared.lastLLBytes) bytes, High: \(Layer0DebugTracker.shared.lastHighBytes) bytes, Total: \(buf.count) bytes")
            }
            #endif
            
            let quantizedBlocks = blocks
            let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: dx, height: dy, qt: qtY, pool: pool)
            return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
        }()
        
        async let taskBufCb = { () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
            #if DEBUG
            Layer0DebugTracker.shared.queue.sync { Layer0DebugTracker.shared.lastLLBytes = 0; Layer0DebugTracker.shared.lastHighBytes = 0 }
            #endif
            var (blocks, relBlocks) = await extractSingleTransformBlocksBase8(r: pd.rCb, width: cbDx, height: cbDy, pool: pool)
            let isIFrame = (sads == nil)
            for i in blocks.indices {
                evaluateQuantizeBase8(view: blocks[i], qt: qtC)
            }
            
            let safeThreshold = min(8, max(0, (zeroThreshold / 8) - (Int(qtC.step)  / 32)))
            let buf = if isIFrame != true {
                encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold)
            } else {
                encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold)
            }
            
            #if DEBUG
            Layer0DebugTracker.shared.queue.sync {
                let frameType = isIFrame ? "I" : "P"
                print("Layer0 Dump [\(frameType) Cb] LL: \(Layer0DebugTracker.shared.lastLLBytes) bytes, High: \(Layer0DebugTracker.shared.lastHighBytes) bytes, Total: \(buf.count) bytes")
            }
            #endif
            
            let quantizedBlocks = blocks
            let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
            return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
        }()
        
        async let taskBufCr = { () -> ([UInt8], [Int16], @Sendable () -> Void, [BlockView], @Sendable () -> Void) in
            #if DEBUG
            Layer0DebugTracker.shared.queue.sync { Layer0DebugTracker.shared.lastLLBytes = 0; Layer0DebugTracker.shared.lastHighBytes = 0 }
            #endif
            var (blocks, relBlocks) = await extractSingleTransformBlocksBase8(r: pd.rCr, width: cbDx, height: cbDy, pool: pool)
            let isIFrame = (sads == nil)
            for i in blocks.indices {
                evaluateQuantizeBase8(view: blocks[i], qt: qtC)
            }
            
            let safeThreshold = min(8, max(0, (zeroThreshold / 8) - (Int(qtC.step) / 32)))
            let buf = if isIFrame != true {
                encodePlaneBaseSubbands8PFrame(blocks: &blocks, zeroThreshold: safeThreshold)
            } else {
                encodePlaneBaseSubbands8(blocks: &blocks, zeroThreshold: safeThreshold)
            }
            
            #if DEBUG
            Layer0DebugTracker.shared.queue.sync {
                let frameType = isIFrame ? "I" : "P"
                print("Layer0 Dump [\(frameType) Cr] LL: \(Layer0DebugTracker.shared.lastLLBytes) bytes, High: \(Layer0DebugTracker.shared.lastHighBytes) bytes, Total: \(buf.count) bytes")
            }
            #endif
            
            let quantizedBlocks = blocks
            let (reconPlane, rPlane) = reconstructPlaneBase8(blocks: blocks, width: cbDx, height: cbDy, qt: qtC, pool: pool)
            return (buf, reconPlane, rPlane, quantizedBlocks, relBlocks)
        }()

        let (bufY, reconY, r0Y, base8YBlocks, relYBlocks) = await taskBufY
        let (bufCb, reconCb, r0Cb, base8CbBlocks, relCbBlocks) = await taskBufCb
        let (bufCr, reconCr, r0Cr, base8CrBlocks, relCrBlocks) = await taskBufCr
        
        let reconstructed = PlaneData420(width: dx, height: dy, y: reconY, cb: reconCb, cr: reconCr)
        
        debugLog({
            return "  [Layer \(layer)/Base] Y=\(bufY.count) Cb=\(bufCb.count) Cr=\(bufCr.count) bytes"
        }())
        
        let out = VEVCLayerData.serialize(
            qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
            bufY: bufY, bufCb: bufCb, bufCr: bufCr
        )
        
        return (out, reconstructed, base8YBlocks, base8CbBlocks, base8CrBlocks, {
            r0Y()
            r0Cb()
            r0Cr()
            relYBlocks()
            relCbBlocks()
            relCrBlocks()
        })
    }

    func decode(
        r: [UInt8],
        pool: BlockViewPool,
        layer: UInt8,
        dx: Int,
        dy: Int,
        isIFrame: Bool
    ) async throws -> (
        reconstructed: Image16,
        yBlocks: [BlockView],
        cbBlocks: [BlockView],
        crBlocks: [BlockView],
        qtYStep: Int,
        qtCStep: Int
    ) {
        let (qtY, qtC, bufY, bufCb, bufCr) = try VEVCLayerData.deserialize(from: r, layer: layer, layerLabel: "Base8")
        
        var sub = Image16(width: dx, height: dy, pool: pool)
        
        let rowCountY = (dy + 8 - 1) / 8
        let colCountY = (dx + 8 - 1) / 8
        let yBlocks = try decodePlaneBaseSubbands8(data: bufY, pool: pool, blockCount: rowCountY * colCountY, isIFrame: isIFrame)
        
        let cbDx = (dx + 1) / 2
        let cbDy = (dy + 1) / 2
        let rowCountCb = (cbDy + 8 - 1) / 8
        let colCountCb = (cbDx + 8 - 1) / 8
        let cbBlocks = try decodePlaneBaseSubbands8(data: bufCb, pool: pool, blockCount: rowCountCb * colCountCb, isIFrame: isIFrame)
        
        let rowCountCr = (cbDy + 8 - 1) / 8
        let colCountCr = (cbDx + 8 - 1) / 8
        let crBlocks = try decodePlaneBaseSubbands8(data: bufCr, pool: pool, blockCount: rowCountCr * colCountCr, isIFrame: isIFrame)
        
        let resY8 = decodeBase8ProcessY(pool: pool, taskIdx: 0, chunkSize: rowCountY, rowCount: rowCountY, dx: dx, colCount: colCountY, blocks: yBlocks, qt: qtY)
        for j in resY8.indices {
            var blk = resY8[j].0
            let w = resY8[j].1
            let h = resY8[j].2
            sub.updateY(data: &blk, startX: w, startY: h, size: 8)
        }

        let resCb8 = decodeBase8ProcessCb(pool: pool, taskIdx: 0, chunkSize: rowCountCb, rowCount: rowCountCb, dx: cbDx, colCount: colCountCb, blocks: cbBlocks, qt: qtC)
        for j in resCb8.indices {
            var blk = resCb8[j].0
            let w = resCb8[j].1
            let h = resCb8[j].2
            sub.updateCb(data: &blk, startX: w, startY: h, size: 8)
        }
        
        let resCr8 = decodeBase8ProcessCr(pool: pool, taskIdx: 0, chunkSize: rowCountCr, rowCount: rowCountCr, dx: cbDx, colCount: colCountCr, blocks: crBlocks, qt: qtC)
        for j in resCr8.indices {
            var blk = resCr8[j].0
            let w = resCr8[j].1
            let h = resCr8[j].2
            sub.updateCr(data: &blk, startX: w, startY: h, size: 8)
        }
            
        return (sub, yBlocks, cbBlocks, crBlocks, qtYStep: Int(qtY.step), qtCStep: Int(qtC.step))
    }
}
@Sendable @inline(__always)
func decodeBase8ProcessY(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable) -> [(BlockView, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(BlockView, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 8
        for (xIdx, w) in stride(from: 0, to: dx, by: 8).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            let block: BlockView = blocks[blockIndex]
            let half: Int = 8 / 2
            let view = block
            let base = view.base
            let llView = BlockView(base: base, width: half, height: half, stride: 8)
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8)
            let lhView = BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8)
            let hhView = BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
            dequantizeSIMD4(llView, q: qt.qLow)
            dequantizeSIMDSignedMapping4(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping4(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping4(hhView, q: qt.qHigh)
            inverseDWT2DBlock8(view)
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeBase8ProcessCb(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable) -> [(BlockView, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(BlockView, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 8
        for (xIdx, w) in stride(from: 0, to: dx, by: 8).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            let block: BlockView = blocks[blockIndex]
            let half: Int = 8 / 2
            let view = block
            let base = view.base
            let llView = BlockView(base: base, width: half, height: half, stride: 8)
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8)
            let lhView = BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8)
            let hhView = BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
            dequantizeSIMD4(llView, q: qt.qLow)
            dequantizeSIMDSignedMapping4(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping4(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping4(hhView, q: qt.qHigh)
            inverseDWT2DBlock8(view)
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}

@Sendable @inline(__always)
func decodeBase8ProcessCr(pool: BlockViewPool, taskIdx: Int, chunkSize: Int, rowCount: Int, dx: Int, colCount: Int, blocks: [BlockView], qt: QuantizationTable) -> [(BlockView, Int, Int)] {
    let startRow: Int = taskIdx * chunkSize
    let endRow: Int = min(startRow + chunkSize, rowCount)
    guard startRow < endRow else { return [] }
    var rowResults: [(BlockView, Int, Int)] = []
    for i in startRow..<endRow {
        let h: Int = i * 8
        for (xIdx, w) in stride(from: 0, to: dx, by: 8).enumerated() {
            let blockIndex: Int = i * colCount + xIdx
            let block: BlockView = blocks[blockIndex]
            let half: Int = 8 / 2
            let view = block
            let base = view.base
            let llView = BlockView(base: base, width: half, height: half, stride: 8)
            let hlView = BlockView(base: base.advanced(by: half), width: half, height: half, stride: 8)
            let lhView = BlockView(base: base.advanced(by: half * 8), width: half, height: half, stride: 8)
            let hhView = BlockView(base: base.advanced(by: half * 8 + half), width: half, height: half, stride: 8)
            dequantizeSIMD4(llView, q: qt.qLow)
            dequantizeSIMDSignedMapping4(hlView, q: qt.qMid)
            dequantizeSIMDSignedMapping4(lhView, q: qt.qMid)
            dequantizeSIMDSignedMapping4(hhView, q: qt.qHigh)
            inverseDWT2DBlock8(view)
            rowResults.append((block, w, h))
        }
    }
    return rowResults
}