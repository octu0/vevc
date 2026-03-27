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
        guard chunk.count >= 11 else {
            print("decodeGOP insufficientData: chunk.count=\(chunk.count) < 11")
            throw DecodeError.insufficientData
        }
        var offset = 0
        
        // Parse GOP header: DataSize(4B) (already skipped or kept in logic depending on usage, but here we just read GOPSize)
        let _ = try readUInt32BEFromBytes(chunk, offset: &offset) // DataSize (skip)
        
        // No Mode byte anymore!
        let gopSize = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
        let nLow = Int(try readUInt16BEFromBytes(chunk, offset: &offset))
        
        // Parse all frame payloads (sequential, offset-dependent)
        var frameData: [[UInt8]] = []
        for _ in 0..<gopSize {
            let len = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
            guard (offset + len) <= chunk.count else {
                print("decodeGOP insufficientData: offset(\(offset)) + len(\(len)) > chunk.count(\(chunk.count))")
                throw DecodeError.insufficientData
            }
            frameData.append(Array(chunk[offset..<(offset + len)]))
            offset += len
        }
        
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
        
        let validPlanes = decodedPlanes.map { $0! }
        
        if nLow > 0 {
            // Temporal DWT reconstructed
            let nHigh = gopSize - nLow
            let lowPlanes = Array(validPlanes[0..<nLow])
            let highPlanes = Array(validPlanes[nLow..<(nLow + nHigh)])
            
            let subbands = TemporalSubbands(low: lowPlanes, high: highPlanes)
            let reconstructed = try temporalInverseDWT4(subbands: subbands)
            return reconstructed.map { $0.toYCbCr() }
        } else {
            // No Temporal DWT (e.g. single frame spatial direct)
            return validPlanes.map { $0.toYCbCr() }
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
    /// Parse VEVC header chunk to extract width/height from metadata.
    /// Returns (width, height) if the chunk is a valid VEVC header, nil otherwise.
    private static func parseVEVCHeaderChunk(_ chunk: [UInt8]) -> (Int, Int)? {
        guard chunk.count >= 4, chunk[0] == 0x56, chunk[1] == 0x45, chunk[2] == 0x56, chunk[3] == 0x43 else {
            return nil
        }
        // Magic(4B) + MetadataSize(2B) + Profile(1B) + Width(2B) + Height(2B) + ...
        guard chunk.count >= 4 + 2 + 1 + 2 + 2 else { return nil }
        var offset = 4
        let metadataSize = (Int(chunk[offset]) << 8) | Int(chunk[offset + 1])
        offset += 2
        guard metadataSize >= 9 else { return nil }
        let profile = chunk[offset]
        guard profile == 0x01 else { return nil }
        offset += 1
        let w = (Int(chunk[offset]) << 8) | Int(chunk[offset + 1])
        offset += 2
        let h = (Int(chunk[offset]) << 8) | Int(chunk[offset + 1])
        return (w, h)
    }

    /// Decodes a stream of VEVC chunks into a stream of images.
    /// First chunk may be a VEVC header (Magic + Metadata), followed by GOP chunks.
    /// Each GOP chunk is a self-contained unit that decodes to one or more frames.
    public func decode(stream: AsyncStream<[UInt8]>) -> AsyncThrowingStream<YCbCrImage, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                var iterator = stream.makeAsyncIterator()
                var currentGOPIndex = 0
                var nextGOPIndexToYield = 0
                var completedGOPs: [Int: [YCbCrImage]] = [:]
                
                // Use initializer values, may be overridden by VEVC header
                var effectiveWidth = self.width
                var effectiveHeight = self.height
                
                // Consume leading VEVC header chunk(s) before entering decode loop
                var pendingGOPChunk: [UInt8]? = nil
                while let chunk = await iterator.next() {
                    if let (w, h) = Decoder.parseVEVCHeaderChunk(chunk) {
                        effectiveWidth = w
                        effectiveHeight = h
                        continue // skip header chunk
                    }
                    pendingGOPChunk = chunk
                    break
                }
                
                do {
                    try await withThrowingTaskGroup(of: (Int, [YCbCrImage]).self) { group in
                        var activeTasks = 0
                        
                        // Helper to submit a GOP chunk for decoding
                        func submitChunk(_ chunk: [UInt8]) {
                            let idx = currentGOPIndex
                            currentGOPIndex += 1
                            let localMaxLayer = self.maxLayer
                            let localWidth = effectiveWidth
                            let localHeight = effectiveHeight
                            group.addTask {
                                let decoder = CoreDecoder(maxLayer: localMaxLayer, width: localWidth, height: localHeight)
                                return (idx, try await decoder.decodeGOP(chunk: chunk))
                            }
                            activeTasks += 1
                        }
                        
                        // Submit the first pending chunk if present
                        if let first = pendingGOPChunk {
                            submitChunk(first)
                        }
                        
                        // Fill initial task pool
                        while activeTasks < self.maxConcurrency {
                            guard let chunk = await iterator.next() else { break }
                            submitChunk(chunk)
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
                                submitChunk(chunk)
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
