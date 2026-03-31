import Foundation

/// Parse VEVC header chunk to extract width/height from metadata.
/// Returns (width, height) if the chunk is a valid VEVC header, nil otherwise.
@inline(__always)
private func parseVEVCHeaderChunk(_ chunk: [UInt8]) -> (Int, Int, Int)? {
    guard chunk.count >= 4, chunk[0] == 0x56, chunk[1] == 0x45, chunk[2] == 0x56, chunk[3] == 0x43 else {
        return nil
    }
    // Magic(4B) + MetadataSize(2B) + Profile(1B) + Width(2B) + Height(2B) + optional FPS(2B)
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
    offset += 2
    // Skip ColorGamut (1B)
    offset += 1
    guard chunk.count >= offset + 2 else { return (w, h, 30) }
    let fps = (Int(chunk[offset]) << 8) | Int(chunk[offset + 1])
    return (w, h, fps)
}

class CoreDecoder {
    let maxLayer: Int
    let width: Int
    let height: Int
    
    init(maxLayer: Int = 2, width: Int = 0, height: Int = 0) {
        self.maxLayer = maxLayer
        self.width = width
        self.height = height
    }
    
    /// Decode a GOP chunk into one or more YCbCrImages.
    /// Mode=0x00 (Temporal): applies inverse temporal DWT to reconstruct original frames.
    /// Mode=0x01 (Direct): outputs frames directly.
    @inline(__always)
    func decodeGOP(chunk: [UInt8]) async throws -> [YCbCrImage] {
        guard chunk.count >= 11 else {
            throw DecodeError.insufficientDataContext("decodeGOP: chunk.count=\(chunk.count) < 11")
        }
        var offset = 0
        
        // Parse GOP header: DataSize(4B) (already skipped or kept in logic depending on usage, but here we just read GOPSize)
        let _ = try readUInt32BEFromBytes(chunk, offset: &offset) // DataSize (skip)
        
        // No Mode byte anymore!
        let gopSize = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
        
        // Parse all frame payloads (sequential, offset-dependent)
        // Convention: FrameLen == 0 signals a "copy frame" — a frame that is
        // pixel-identical to its predecessor. The encoder detects duplicate
        // input frames (common in 24fps→60fps telecine/pulldown conversions)
        // and emits FrameLen=0 instead of encoding redundant data.
        // The decoder handles this by reusing the previous reconstructed frame.
        var frameData: [[UInt8]] = []
        for _ in 0..<gopSize {
            let len = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
            if len == 0 {
                // Copy frame: empty payload signals reuse of previous frame
                frameData.append([])
                continue
            }
            guard (offset + len) <= chunk.count else {
                throw DecodeError.insufficientDataContext("decodeGOP: offset(\(offset)) + len(\(len)) > chunk.count(\(chunk.count))")
            }
            frameData.append(Array(chunk[offset..<(offset + len)]))
            offset += len
        }
        
        let localMaxLayer = maxLayer
        let localWidth = width
        let localHeight = height
        var decodedPlanes = [PlaneData420?](repeating: nil, count: frameData.count)
        
        var previousReconstructed: PlaneData420? = nil
        for (idx, data) in frameData.enumerated() {
            // Copy frame: reuse previous reconstructed frame verbatim.
            // This is valid because the encoder only emits FrameLen=0 when
            // the input frame was pixel-identical to the previous one.
            if data.isEmpty {
                decodedPlanes[idx] = previousReconstructed
                // Do NOT update previousReconstructed — it stays the same
                continue
            }
            let img16 = try await decodeSpatialLayers(r: data, maxLayer: localMaxLayer, dx: localWidth, dy: localHeight, predictedPd: previousReconstructed)
            let pd = PlaneData420(img16: img16)
            decodedPlanes[idx] = pd
            previousReconstructed = pd
        }
        
        let validPlanes = decodedPlanes.map { $0! }
        
        return validPlanes.map { $0.toYCbCr() }
    }
}

public struct Decoder: Sendable {
    public let maxLayer: Int
    public let maxConcurrency: Int

    public init(
        maxLayer: Int = 2,
        maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount,
    ) {
        self.maxLayer = maxLayer
        self.maxConcurrency = maxConcurrency
    }

    /// Decodes a stream of VEVC chunks into a stream of images.
    /// First chunk may be a VEVC header (Magic + Metadata), followed by GOP chunks.
    /// Each GOP chunk is a self-contained unit that decodes to one or more frames.
    /// Width/height are extracted from the VEVC header in the stream.
    public func decode(stream: AsyncStream<[UInt8]>) -> AsyncThrowingStream<YCbCrImage, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                var iterator = stream.makeAsyncIterator()
                var currentGOPIndex = 0
                var nextGOPIndexToYield = 0
                var completedGOPs: [Int: [YCbCrImage]] = [:]
                
                guard let firstChunk = await iterator.next(), let (effectiveWidth, effectiveHeight, effectiveFps) = parseVEVCHeaderChunk(firstChunk) else {
                    continuation.finish(throwing: DecodeError.insufficientDataContext("missing VEVC header chunk"))
                    return
                }
                
                let decoderMaxLayer = self.maxLayer
                let decoderWidth = effectiveWidth
                let decoderHeight = effectiveHeight
                
                do {
                    try await withThrowingTaskGroup(of: (Int, [YCbCrImage]).self) { group in
                        var activeTasks = 0
                        
                        // Helper to submit a GOP chunk for decoding
                        func submitChunk(_ chunk: [UInt8]) {
                            let idx = currentGOPIndex
                            currentGOPIndex += 1
                            group.addTask {
                                let decoder = CoreDecoder(maxLayer: decoderMaxLayer, width: decoderWidth, height: decoderHeight)
                                return (idx, try await decoder.decodeGOP(chunk: chunk))
                            }
                            activeTasks += 1
                        }
                        
                        // 最初のGOPチャンクはループ内で取得される
                        
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
                                for var frame in consecutiveFrames {
                                    frame.fps = effectiveFps
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
    
    /// Decode VEVC encoded byte array into images.
    /// Parses the VEVC header and GOP chunks from the data, then decodes via streaming pipeline.
    public func decode(data: [UInt8]) async throws -> [YCbCrImage] {
        if data.isEmpty { return [] }
        
        var offset = 0
        var chunks: [[UInt8]] = []
        var headerChunk: [UInt8]? = nil
        
        while offset < data.count {
            guard offset + 4 <= data.count else { break }
            
            let first4Bytes = [data[offset], data[offset+1], data[offset+2], data[offset+3]]
            if first4Bytes == [0x56, 0x45, 0x56, 0x43] {
                // VEVC file header: magic(4B) + metadataSize(2B) + payload
                let headerStart = offset
                offset += 4
                let metadataSize = Int(try readUInt16BEFromBytes(data, offset: &offset))
                offset += metadataSize
                headerChunk = Array(data[headerStart..<offset])
            } else {
                // GOP chunk: first4Bytes is DataSize(4B)
                let chunkStart = offset
                let gopDataSize = Int(try readUInt32BEFromBytes(data, offset: &offset))
                if (offset + gopDataSize) > data.count {
                    throw DecodeError.insufficientData
                }
                let chunkEnd = offset + gopDataSize
                chunks.append(Array(data[chunkStart..<chunkEnd]))
                offset = chunkEnd
            }
        }
        
        let stream = AsyncStream<[UInt8]> { continuation in
            if let header = headerChunk {
                continuation.yield(header)
            }
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
    
    /// Decode VEVC data from a FileHandle (file or stdin).
    /// Reads the VEVC header and GOP chunks, then decodes via streaming pipeline.
    public func decode(fileHandle: FileHandle) -> AsyncThrowingStream<YCbCrImage, Error> {
        let stream = AsyncStream<[UInt8]> { continuation in
            Task {
                do {
                    while true {
                        let first4Data = readFully(fileHandle: fileHandle, count: 4)
                        if first4Data.isEmpty { break } // Normal EOF
                        guard first4Data.count == 4 else {
                            continuation.finish()
                            return
                        }
                        
                        let first4Bytes = [UInt8](first4Data)
                        if first4Bytes == [0x56, 0x45, 0x56, 0x43] {
                            // VEVC file header
                            let metaSizeData = readFully(fileHandle: fileHandle, count: 2)
                            guard metaSizeData.count == 2 else {
                                continuation.finish()
                                return
                            }
                            var msOffset = 0
                            let metadataSize = Int(try readUInt16BEFromBytes([UInt8](metaSizeData), offset: &msOffset))
                            let metaData = readFully(fileHandle: fileHandle, count: metadataSize)
                            guard metaData.count == metadataSize else {
                                continuation.finish()
                                return
                            }
                            
                            // Build complete header chunk for parseVEVCHeaderChunk
                            var headerChunk: [UInt8] = first4Bytes
                            headerChunk.append(contentsOf: metaSizeData)
                            headerChunk.append(contentsOf: metaData)
                            continuation.yield(headerChunk)
                        } else {
                            // GOP chunk: first4Bytes is DataSize(4B)
                            var dsOffset = 0
                            let gopDataSize = Int(try readUInt32BEFromBytes(first4Bytes, offset: &dsOffset))
                            let gopBody = readFully(fileHandle: fileHandle, count: gopDataSize)
                            guard gopBody.count == gopDataSize else {
                                continuation.finish()
                                return
                            }
                            
                            var chunk: [UInt8] = []
                            chunk.reserveCapacity(4 + gopDataSize)
                            chunk.append(contentsOf: first4Bytes)
                            chunk.append(contentsOf: gopBody)
                            continuation.yield(chunk)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
        }
        return self.decode(stream: stream)
    }

    @inline(__always)
    private func readFully(fileHandle: FileHandle, count: Int) -> Data {
        var result = Data()
        var remaining = count
        while remaining > 0 {
            let data = fileHandle.readData(ofLength: remaining)
            if data.isEmpty { break }
            result.append(data)
            remaining -= data.count
        }
        return result
    }
}

