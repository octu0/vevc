import Foundation
@testable import vevc

func lift53(_ buffer: UnsafeMutableBufferPointer<Int16>, count: Int, stride: Int) {
    switch count {
    case 32: lift53Block32(buffer, stride: stride)
    case 16: lift53Block16(buffer, stride: stride)
    case 8: lift53Block8(buffer, stride: stride)
    default: fatalError()
    }
}

func invLift53(_ buffer: UnsafeMutableBufferPointer<Int16>, count: Int, stride: Int) {
    switch count {
    case 32: inverseLift53Block32(buffer, stride: stride)
    case 16: inverseLift53Block16(buffer, stride: stride)
    case 8: inverseLift53Block8(buffer, stride: stride)
    default: fatalError()
    }
}

func dwt2d(_ block: BlockView, size: Int) -> Subbands {
    switch size {
    case 32: return dwt2DBlock32Subbands(block)
    case 16: return dwt2DBlock16Subbands(block)
    case 8: return dwt2DBlock8Subbands(block)
    default: fatalError()
    }
}

func invDwt2d(_ block: BlockView, size: Int) {
    switch size {
    case 32: inverseDWT2DBlock32(block)
    case 16: inverseDWT2DBlock16(block)
    case 8: inverseDWT2DBlock8(block)
    default: fatalError()
    }
}

func dwt2dScalar(_ block: BlockView, size: Int) -> Subbands {
    switch size {
    case 32: return dwt2DBlock32Subbands(block)
    case 16: return dwt2DBlock16Subbands(block)
    case 8: return dwt2DBlock8Subbands(block)
    default: fatalError()
    }
}

func invDwt2dScalar(_ block: BlockView, size: Int) {
    switch size {
    case 32: inverseDWT2DBlock32(block)
    case 16: inverseDWT2DBlock16(block)
    case 8: inverseDWT2DBlock8(block)
    default: fatalError()
    }
}

func blockEncode(encoder: inout EntropyEncoder<DynamicEntropyModel>, block: BlockView, size: Int) {
    switch size {
    case 16: blockEncode16V(encoder: &encoder, block: block)
    case 8: blockEncode8V(encoder: &encoder, block: block)
    default: fatalError()
    }
}

func blockDecode(decoder: inout EntropyDecoder, block: BlockView, size: Int) throws {
    switch size {
    case 16: try blockDecode16V(decoder: &decoder, block: block)
    case 8: try blockDecode8V(decoder: &decoder, block: block)
    default: fatalError()
    }
}
