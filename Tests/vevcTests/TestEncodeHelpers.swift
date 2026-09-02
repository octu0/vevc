import Foundation
@testable import vevc

// Layer payload assembly without parent context (parentBlocks: nil), used by
// roundtrip tests that decode with decodePlaneSubbands* directly. Mirrors
// encodeLayer32Payload/encodeLayer16Payload thresholds and geometry.

func encodeLayer32PayloadNoParents(dx: Int, dy: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, yBlocks: inout [BlockView], cbBlocks: inout [BlockView], crBlocks: inout [BlockView]) -> [UInt8] {
    let safeThresholdY = min(3, min(zeroThreshold, max(0, Int(qtY.step) / 64)))
    let safeThresholdC = min(8, min(zeroThreshold, max(0, Int(qtC.step) / 64)))
    let colCountY = (dx + 31) / 32
    let rowCountY = (dy + 31) / 32
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let colCountC = (cbDx + 31) / 32
    let rowCountC = (cbDy + 31) / 32
    let bufY = encodePlaneSubbands32(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: nil, colCount: colCountY, rowCount: rowCountY, history: nil, selectModel: unifiedSelectModel, updateHistory: true)
    let bufCb = encodePlaneSubbands32(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: nil, colCount: colCountC, rowCount: rowCountC, history: nil, selectModel: unifiedSelectModel, updateHistory: true)
    let bufCr = encodePlaneSubbands32(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: nil, colCount: colCountC, rowCount: rowCountC, history: nil, selectModel: unifiedSelectModel, updateHistory: true)
    return VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
        bufY: bufY, bufCb: bufCb, bufCr: bufCr
    )
}

func encodeLayer16PayloadNoParents(dx: Int, dy: Int, qtY: QuantizationTable, qtC: QuantizationTable, zeroThreshold: Int, yBlocks: inout [BlockView], cbBlocks: inout [BlockView], crBlocks: inout [BlockView]) -> [UInt8] {
    let safeThresholdY = min(2, min(zeroThreshold, max(0, Int(qtY.step) / 64)))
    let safeThresholdC = min(8, min(zeroThreshold, max(0, Int(qtC.step) / 64)))
    let colCountY = (dx + 15) / 16
    let rowCountY = (dy + 15) / 16
    let cbDx = (dx + 1) / 2
    let cbDy = (dy + 1) / 2
    let colCountC = (cbDx + 15) / 16
    let rowCountC = (cbDy + 15) / 16
    let bufY = encodePlaneSubbands16(blocks: &yBlocks, zeroThreshold: safeThresholdY, parentBlocks: nil, colCount: colCountY, rowCount: rowCountY, history: nil, selectModel: unifiedSelectModel, updateHistory: true)
    let bufCb = encodePlaneSubbands16(blocks: &cbBlocks, zeroThreshold: safeThresholdC, parentBlocks: nil, colCount: colCountC, rowCount: rowCountC, history: nil, selectModel: unifiedSelectModel, updateHistory: true)
    let bufCr = encodePlaneSubbands16(blocks: &crBlocks, zeroThreshold: safeThresholdC, parentBlocks: nil, colCount: colCountC, rowCount: rowCountC, history: nil, selectModel: unifiedSelectModel, updateHistory: true)
    return VEVCLayerData.serialize(
        qtYStep: UInt16(qtY.step), qtCStep: UInt16(qtC.step),
        bufY: bufY, bufCb: bufCb, bufCr: bufCr
    )
}
