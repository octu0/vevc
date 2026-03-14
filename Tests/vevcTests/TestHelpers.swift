import Foundation
@testable import vevc

func lift53(_ buffer: UnsafeMutableBufferPointer<Int16>, count: Int, stride: Int) {
    switch count {
    case 32: lift53_32(buffer, stride: stride)
    case 16: lift53_16(buffer, stride: stride)
    case 8: lift53_8(buffer, stride: stride)
    default: fatalError()
    }
}

func invLift53(_ buffer: UnsafeMutableBufferPointer<Int16>, count: Int, stride: Int) {
    switch count {
    case 32: invLift53_32(buffer, stride: stride)
    case 16: invLift53_16(buffer, stride: stride)
    case 8: invLift53_8(buffer, stride: stride)
    default: fatalError()
    }
}

func dwt2d(_ block: inout BlockView, size: Int) -> Subbands {
    switch size {
    case 32: return dwt2d_32_sb(&block)
    case 16: return dwt2d_16_sb(&block)
    case 8: return dwt2d_8_sb(&block)
    default: fatalError()
    }
}

func invDwt2d(_ block: inout BlockView, size: Int) {
    switch size {
    case 32: invDwt2d_32(&block)
    case 16: invDwt2d_16(&block)
    case 8: invDwt2d_8(&block)
    default: fatalError()
    }
}

func dwt2dScalar(_ block: inout BlockView, size: Int) -> Subbands {
    switch size {
    case 32: return dwt2d_32_sb(&block)
    case 16: return dwt2d_16_sb(&block)
    case 8: return dwt2d_8_sb(&block)
    default: fatalError()
    }
}

func invDwt2dScalar(_ block: inout BlockView, size: Int) {
    switch size {
    case 32: invDwt2d_32(&block)
    case 16: invDwt2d_16(&block)
    case 8: invDwt2d_8(&block)
    default: fatalError()
    }
}

func blockEncode(encoder: inout CABACEncoder, block: BlockView, size: Int, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) {
    switch size {
    case 32: blockEncode32(encoder: &encoder, block: block, ctxRun: &ctxRun, ctxMag: &ctxMag)
    case 16: blockEncode16(encoder: &encoder, block: block, ctxRun: &ctxRun, ctxMag: &ctxMag)
    case 8: blockEncode8(encoder: &encoder, block: block, ctxRun: &ctxRun, ctxMag: &ctxMag)
    default: fatalError()
    }
}

func blockDecode(decoder: inout CABACDecoder, block: inout BlockView, size: Int, ctxRun: inout [ContextModel], ctxMag: inout [ContextModel]) throws {
    switch size {
    case 32: try blockDecode32(decoder: &decoder, block: &block, ctxRun: &ctxRun, ctxMag: &ctxMag)
    case 16: try blockDecode16(decoder: &decoder, block: &block, ctxRun: &ctxRun, ctxMag: &ctxMag)
    case 8: try blockDecode8(decoder: &decoder, block: &block, ctxRun: &ctxRun, ctxMag: &ctxMag)
    default: fatalError()
    }
}
