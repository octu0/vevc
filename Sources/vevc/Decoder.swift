import Foundation

class CoreDecoder {
    let maxLayer: Int
    let width: Int
    let height: Int
    
    init(maxLayer: Int = 2, width: Int = 0, height: Int = 0) {
        self.maxLayer = maxLayer
        self.width = width
        self.height = height
    }
    
#if (arch(arm64) || arch(x86_64) || arch(wasm32))
    /// Decode a GOP chunk into one or more YCbCrImages.
    /// Mode=0x00 (Temporal): applies inverse temporal DWT to reconstruct original frames.
    /// Mode=0x01 (Direct): outputs frames directly.
    func decodeGOP(chunk: [UInt8]) async throws -> [YCbCrImage] {
        guard chunk.count >= 7 else { throw DecodeError.insufficientData }
        var offset = 0
        
        // Parse GOP header: Mode(1B) + GOPSize(4B) + nLow(2B)
        let mode = chunk[offset]
        offset += 1
        
        let gopSize = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
        let nLow = Int(try readUInt16BEFromBytes(chunk, offset: &offset))
        
        // Parse all frame payloads (sequential, offset-dependent)
        var frameData: [[UInt8]] = []
        for _ in 0..<gopSize {
            let len = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
            guard (offset + len) <= chunk.count else { throw DecodeError.insufficientData }
            frameData.append(Array(chunk[offset..<(offset + len)]))
            offset += len
        }
        
        switch mode {
        case 0x00: // Temporal: decode subbands in parallel, then inverse DWT
            let localMaxLayer = maxLayer
            let localWidth = width
            let localHeight = height
            var decodedPlanes = [PlaneData420?](repeating: nil, count: frameData.count)
            try await withThrowingTaskGroup(of: (Int, PlaneData420).self) { group in
                for (idx, data) in frameData.enumerated() {
                    group.addTask {
                        let img16 = try await decodeSpatialLayers(r: data, maxLayer: localMaxLayer, dx: localWidth, dy: localHeight)
                        return (idx, PlaneData420(img16: img16))
                    }
                }
                for try await (idx, plane) in group {
                    decodedPlanes[idx] = plane
                }
            }
            
            let nHigh = gopSize - nLow
            let lowPlanes = decodedPlanes[0..<nLow].map { $0! }
            let highPlanes = decodedPlanes[nLow..<(nLow + nHigh)].map { $0! }
            
            let subbands = TemporalSubbands(low: Array(lowPlanes), high: Array(highPlanes))
            let reconstructed = try temporalInverseDWT4(subbands: subbands)
            return reconstructed.map { $0.toYCbCr() }
            
        case 0x01: // Direct: decode each frame independently
            var result: [YCbCrImage] = []
            for data in frameData {
                let img16 = try await decodeSpatialLayers(r: data, maxLayer: maxLayer, dx: width, dy: height)
                let pd = PlaneData420(img16: img16)
                result.append(pd.toYCbCr())
            }
            return result
            
        default:
            throw DecodeError.invalidHeader
        }
    }

#else
    func decodeGOP(chunk: [UInt8]) async throws -> [YCbCrImage] {
        throw DecodeError.unsupportedArchitecture
    }
#endif
}

public struct Decoder: Sendable {
    public let maxLayer: Int
    public let maxConcurrency: Int
    public let width: Int
    public let height: Int

    public init(
        maxLayer: Int = 2,
        maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount,
        width: Int = 0,
        height: Int = 0,
    ) {
        self.maxLayer = maxLayer
        self.maxConcurrency = maxConcurrency
        self.width = width
        self.height = height
    }

    #if (arch(arm64) || arch(x86_64) || arch(wasm32))
    /// Decodes a stream of VEGI chunks into a stream of images.
    /// Each chunk is a self-contained VEGI GOP that decodes to one or more frames.
    public func decode(stream: AsyncStream<[UInt8]>) -> AsyncThrowingStream<YCbCrImage, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                var iterator = stream.makeAsyncIterator()
                var currentGOPIndex = 0
                var nextGOPIndexToYield = 0
                var completedGOPs: [Int: [YCbCrImage]] = [:]
                
                do {
                    try await withThrowingTaskGroup(of: (Int, [YCbCrImage]).self) { group in
                        var activeTasks = 0
                        
                        // Fill initial task pool
                        while activeTasks < self.maxConcurrency {
                            guard let chunk = await iterator.next() else { break }
                            let idx = currentGOPIndex
                            currentGOPIndex += 1
                            let localMaxLayer = self.maxLayer
                            let localWidth = self.width
                            let localHeight = self.height
                            group.addTask {
                                let decoder = CoreDecoder(maxLayer: localMaxLayer, width: localWidth, height: localHeight)
                                return (idx, try await decoder.decodeGOP(chunk: chunk))
                            }
                            activeTasks += 1
                        }
                        
                        // Process results and feed new tasks
                        while let result = try await group.next() {
                            activeTasks -= 1
                            let (idx, frames) = result
                            completedGOPs[idx] = frames
                            
                            // Yield frames in order
                            while let consecutiveFrames = completedGOPs[nextGOPIndexToYield] {
                                for frame in consecutiveFrames {
                                    continuation.yield(frame)
                                }
                                completedGOPs.removeValue(forKey: nextGOPIndexToYield)
                                nextGOPIndexToYield += 1
                            }
                            
                            // Schedule next chunk if available
                            if let chunk = await iterator.next() {
                                let newIdx = currentGOPIndex
                                currentGOPIndex += 1
                                let localMaxLayer = self.maxLayer
                                let localWidth = self.width
                                let localHeight = self.height
                                group.addTask {
                                    let decoder = CoreDecoder(maxLayer: localMaxLayer, width: localWidth, height: localHeight)
                                    return (newIdx, try await decoder.decodeGOP(chunk: chunk))
                                }
                                activeTasks += 1
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    /// Decodes an array of chunks. Convenience for tests and compare tool.
    public func decode(chunks: [[UInt8]]) async throws -> [YCbCrImage] {
        let stream = AsyncStream<[UInt8]> { continuation in
            for chunk in chunks {
                continuation.yield(chunk)
            }
            continuation.finish()
        }
        var images: [YCbCrImage] = []
        for try await img in self.decode(stream: stream) {
            images.append(img)
        }
        return images
    }
    #endif
}
