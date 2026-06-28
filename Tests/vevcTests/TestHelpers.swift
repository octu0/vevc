import Foundation
@testable import vevc

func blockEncode(encoder: inout EntropyEncoder, block: BlockView, size: Int) {
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
