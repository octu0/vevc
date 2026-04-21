import Foundation



class CoreDecoder {
    let maxLayer: Int
    let width: Int
    let height: Int
    let pool: BlockViewPool
    
    init(maxLayer: Int = 2, width: Int = 0, height: Int = 0) {
        self.maxLayer = maxLayer
        self.width = width
        self.height = height
        self.pool = BlockViewPool()
    }
    
    /// Decode a GOP chunk into one or more YCbCrImages.
    /// Mode=0x00 (Temporal): applies inverse temporal DWT to reconstruct original frames.
    /// Mode=0x01 (Direct): outputs frames directly.
    @inline(__always)
    func decodeGOP(chunk: [UInt8]) async throws -> [YCbCrImage] {
        guard 4 <= chunk.count else {
            throw DecodeError.insufficientDataContext("decodeGOP: chunk.count=\(chunk.count) < 4")
        }
        var offset = 0
        let gopHeader = try VEVCGOPHeader.deserialize(from: chunk, offset: &offset)
        
        var frameData: [[UInt8]] = []
        for _ in 0..<gopHeader.frameCount {
            let frameHeaderOffset = offset
            let frameHeader = try VEVCFrameHeader.deserialize(from: chunk, offset: &offset)
            if frameHeader.isCopyFrame {
                frameData.append([]) // Copy frame
            } else {
                let len = frameHeader.payloadSize
                guard (offset + len) <= chunk.count else {
                    throw DecodeError.insufficientDataContext("decodeGOP: offset(\(offset)) + len(\(len)) > chunk.count(\(chunk.count))")
                }
                
                // For decodeSpatialLayers, we pass the memory segment containing the FrameHeader and payload
                let totalLenWithHeader = len + (offset - frameHeaderOffset)
                frameData.append(Array(chunk[frameHeaderOffset..<(frameHeaderOffset + totalLenWithHeader)]))
                offset += len
            }
        }
        
        let localMaxLayer = maxLayer
        let localWidth = width
        let localHeight = height
        var decodedPlanes = [PlaneData420?](repeating: nil, count: frameData.count)
        
        var previousReconstructed: PlaneData420? = nil
        // Bidirectional prediction: keep the reconstructed result of the first frame of GOP (I-frame) as backward reference
        var firstReconstructed: PlaneData420? = nil
        
        for (idx, data) in frameData.enumerated() {
            // Copy frame: reuse previous reconstructed frame verbatim.
            // This is valid because the encoder only emits FrameLen=0 when
            // the input frame was pixel-identical to the previous one.
            if data.isEmpty {
                decodedPlanes[idx] = previousReconstructed
                // Do NOT update previousReconstructed — it stays the same
                continue
            }
            
            // Bidirectional prediction check: [Experiment] Apply bidirectional prediction to all P-frames
            let isPFrame = (previousReconstructed != nil)
            let useBidirectional = isPFrame && firstReconstructed != nil && frameData.count >= 2
            
            let nextPd: PlaneData420? = if useBidirectional { firstReconstructed } else { nil }
            let img16 = try await decodeSpatialLayers(r: data, pool: pool, maxLayer: localMaxLayer, dx: localWidth, dy: localHeight, predictedPd: previousReconstructed, nextPd: nextPd, roundOffset: idx % 2)
            let pd = PlaneData420(img16: img16)
            decodedPlanes[idx] = pd
            previousReconstructed = pd
            
            // Save reconstructed result of first frame (I-frame)
            if firstReconstructed == nil {
                firstReconstructed = pd
            }
        }
        
        let validPlanes = decodedPlanes.compactMap { $0 }
        let result = validPlanes.map { $0.toYCbCr() }
        
        var seenY = Set<UnsafeMutableRawPointer>()
        for p in validPlanes {
            p.y.withUnsafeBufferPointer { yPtr in
                if let yBase = yPtr.baseAddress { // Cast to avoid Array copy
                    let ptr = UnsafeMutableRawPointer(mutating: yBase)
                    if seenY.contains(ptr) != true {
                        seenY.insert(ptr)
                        pool.putInt16(p.y)
                        pool.putInt16(p.cb)
                        pool.putInt16(p.cr)
                    }
                }
            }
        }
        
        return result
    }
}

@inline(__always)
private func submitGOPChunk(
    _ chunk: [UInt8],
    index: Int,
    maxLayer: Int,
    width: Int,
    height: Int,
    group: inout ThrowingTaskGroup<(Int, [YCbCrImage]), Error>
) {
    group.addTask {
        let decoder = CoreDecoder(maxLayer: maxLayer, width: width, height: height)
        return (index, try await decoder.decodeGOP(chunk: chunk))
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
    /// First chunk must be a VEVC header (Magic + Metadata), followed by GOP chunks.
    /// Each GOP chunk is a self-contained unit that decodes to one or more frames.
    /// Width/height are extracted from the VEVC header in the stream.
    public func decode(stream: AsyncStream<[UInt8]>) -> AsyncThrowingStream<YCbCrImage, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                var iterator = stream.makeAsyncIterator()
                var currentGOPIndex = 0
                var nextGOPIndexToYield = 0
                var completedGOPs: [Int: [YCbCrImage]] = [:]
                
                guard let firstChunk = await iterator.next() else {
                    continuation.finish(throwing: DecodeError.insufficientDataContext("missing VEVC header chunk"))
                    return
                }
                
                var headerOffset = 0
                let fileHeader = try VEVCFileHeader.deserialize(from: firstChunk, offset: &headerOffset)
                let effectiveWidth = fileHeader.width
                let effectiveHeight = fileHeader.height
                let effectiveFps = fileHeader.framerate
                
                let decoderMaxLayer = self.maxLayer
                
                do {
                    try await withThrowingTaskGroup(of: (Int, [YCbCrImage]).self) { group in
                        var activeTasks = 0
                        
                        // Fill initial task pool
                        while activeTasks < self.maxConcurrency {
                            guard let chunk = await iterator.next() else { break }
                            submitGOPChunk(chunk, index: currentGOPIndex, maxLayer: decoderMaxLayer, width: effectiveWidth, height: effectiveHeight, group: &group)
                            currentGOPIndex += 1
                            activeTasks += 1
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
                                submitGOPChunk(chunk, index: currentGOPIndex, maxLayer: decoderMaxLayer, width: effectiveWidth, height: effectiveHeight, group: &group)
                                currentGOPIndex += 1
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
                let chunkStart = offset
                let gopHeader = try VEVCGOPHeader.deserialize(from: data, offset: &offset)
                
                for _ in 0..<gopHeader.frameCount {
                    let frameHeader = try VEVCFrameHeader.deserialize(from: data, offset: &offset)
                    offset += frameHeader.payloadSize
                }
                
                let chunkEnd = offset
                chunks.append(Array(data[chunkStart..<chunkEnd]))
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
                            var dsOffset = 0
                            let gopHeader = try VEVCGOPHeader.deserialize(from: first4Bytes, offset: &dsOffset)
                            var chunk: [UInt8] = []
                            chunk.append(contentsOf: first4Bytes)
                            
                            for _ in 0..<gopHeader.frameCount {
                                let flagData = readFully(fileHandle: fileHandle, count: 1)
                                guard flagData.count == 1 else { continuation.finish(); return }
                                chunk.append(contentsOf: flagData)
                                
                                if flagData[0] == 0x01 { continue }
                                
                                let headerBytes = readFully(fileHandle: fileHandle, count: 24) // 6 * 4B
                                guard headerBytes.count == 24 else { continuation.finish(); return }
                                chunk.append(contentsOf: headerBytes)
                                
                                var hsOffset = 0
                                _ = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset)) // MVsCount
                                let mvsSize = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                                let refDirSize = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                                let layer0Size = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                                let layer1Size = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                                let layer2Size = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                                
                                let payloadSize = mvsSize + refDirSize + layer0Size + layer1Size + layer2Size
                                if 0 < payloadSize {
                                    let payloadBody = readFully(fileHandle: fileHandle, count: payloadSize)
                                    guard payloadBody.count == payloadSize else { continuation.finish(); return }
                                    chunk.append(contentsOf: payloadBody)
                                }
                            }
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
        while 0 < remaining {
            let data = fileHandle.readData(ofLength: remaining)
            if data.isEmpty { break }
            result.append(data)
            remaining -= data.count
        }
        return result
    }
}

