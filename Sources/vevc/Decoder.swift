import Foundation



actor StreamingDecoderActor {
    let maxLayer: Int
    let width: Int
    let height: Int
    let pool: BlockViewPool
    
    private var previousReconstructed: PlaneData420?
    private var firstReconstructed: PlaneData420?
    private var roundOffsetIndex = 0
    private var seenY = Set<UnsafeMutableRawPointer>()
    
    init(maxLayer: Int = 2, width: Int = 0, height: Int = 0) {
        self.maxLayer = maxLayer
        self.width = width
        self.height = height
        self.pool = BlockViewPool()
    }
    
    @inline(__always)
    func decodeNextFrame(chunk: [UInt8]) async throws -> YCbCrImage? {
        guard !chunk.isEmpty else { return nil }
        
        var offset = 0
        let frameHeader = try VEVCFrameHeader.deserialize(from: chunk, offset: &offset)
        
        if frameHeader.isCopyFrame {
            guard let prev = previousReconstructed else {
                throw DecodeError.insufficientDataContext("Copy frame without previous frame")
            }
            return prev.toYCbCr()
        }
        
        if frameHeader.isIFrame {
            var oldPlanes = [PlaneData420]()
            if let f = firstReconstructed { oldPlanes.append(f) }
            if let p = previousReconstructed { oldPlanes.append(p) }
            
            for p in oldPlanes {
                p.y.withUnsafeBufferPointer { yPtr in
                    if let yBase = yPtr.baseAddress {
                        let ptr = UnsafeMutableRawPointer(mutating: yBase)
                        if seenY.contains(ptr) {
                            seenY.remove(ptr)
                            pool.putInt16(p.y)
                            pool.putInt16(p.cb)
                            pool.putInt16(p.cr)
                        }
                    }
                }
            }
            seenY.removeAll()
            
            firstReconstructed = nil
            previousReconstructed = nil
            roundOffsetIndex = 0
        }
        
        let isPFrame = (previousReconstructed != nil)
        let useBidirectional = isPFrame && firstReconstructed != nil
        let nextPd: PlaneData420? = useBidirectional ? firstReconstructed : nil
        
        let img16 = try await decodeSpatialLayers(
            r: chunk, pool: pool, maxLayer: maxLayer, dx: width, dy: height,
            predictedPd: previousReconstructed, nextPd: nextPd, roundOffset: roundOffsetIndex % 2
        )
        
        let pd = PlaneData420(img16: img16)
        let yBase = pd.y.withUnsafeBufferPointer { UnsafeMutableRawPointer(mutating: $0.baseAddress!) }
        seenY.insert(yBase)
        
        if let oldPrev = previousReconstructed {
            let oldYBase = oldPrev.y.withUnsafeBufferPointer { UnsafeMutableRawPointer(mutating: $0.baseAddress!) }
            let firstYBase = firstReconstructed?.y.withUnsafeBufferPointer { UnsafeMutableRawPointer(mutating: $0.baseAddress!) }
            if oldYBase != firstYBase {
                if seenY.contains(oldYBase) {
                    seenY.remove(oldYBase)
                    pool.putInt16(oldPrev.y)
                    pool.putInt16(oldPrev.cb)
                    pool.putInt16(oldPrev.cr)
                }
            }
        }
        
        previousReconstructed = pd
        if firstReconstructed == nil {
            firstReconstructed = pd
        }
        
        roundOffsetIndex += 1
        return pd.toYCbCr()
    }
}

public struct Decoder: Sendable {
    public let maxLayer: Int
    public let maxConcurrency: Int

    public init(
        maxLayer: Int = 2,
        maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount
    ) {
        self.maxLayer = maxLayer
        self.maxConcurrency = maxConcurrency
    }

    public func decode(stream: AsyncStream<[UInt8]>) -> AsyncThrowingStream<YCbCrImage, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                var iterator = stream.makeAsyncIterator()
                
                guard let firstChunk = await iterator.next() else {
                    continuation.finish(throwing: DecodeError.insufficientDataContext("missing VEVC header chunk"))
                    return
                }
                
                var headerOffset = 0
                let fileHeader = try VEVCFileHeader.deserialize(from: firstChunk, offset: &headerOffset)
                let effectiveWidth = fileHeader.width
                let effectiveHeight = fileHeader.height
                let effectiveFps = fileHeader.framerate
                
                let decoderActor = StreamingDecoderActor(maxLayer: self.maxLayer, width: effectiveWidth, height: effectiveHeight)
                
                do {
                    while let chunk = await iterator.next() {
                        if let img = try await decoderActor.decodeNextFrame(chunk: chunk) {
                            var mutableImg = img
                            mutableImg.fps = effectiveFps
                            continuation.yield(mutableImg)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    public func decode(data: [UInt8]) async throws -> [YCbCrImage] {
        if data.isEmpty { return [] }
        
        var offset = 0
        var chunks: [[UInt8]] = []
        var headerChunk: [UInt8]? = nil
        
        while offset < data.count {
            if offset + 4 <= data.count && data[offset] == 0x56 && data[offset+1] == 0x45 && data[offset+2] == 0x56 && data[offset+3] == 0x43 {
                let headerStart = offset
                offset += 4
                let metadataSize = Int(try readUInt16BEFromBytes(data, offset: &offset))
                offset += metadataSize
                headerChunk = Array(data[headerStart..<offset])
            } else {
                let chunkStart = offset
                let frameHeader = try VEVCFrameHeader.deserialize(from: data, offset: &offset)
                offset += frameHeader.payloadSize
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
    
    public func decode(fileHandle: FileHandle) -> AsyncThrowingStream<YCbCrImage, Error> {
        let stream = AsyncStream<[UInt8]> { continuation in
            Task {
                do {
                    while true {
                        let firstByteData = readFully(fileHandle: fileHandle, count: 1)
                        if firstByteData.isEmpty { break }
                        let firstByte = firstByteData[0]
                        
                        if firstByte == 0x56 {
                            let next3Data = readFully(fileHandle: fileHandle, count: 3)
                            guard next3Data.count == 3 else { continuation.finish(); return }
                            let next3Bytes = [UInt8](next3Data)
                            if next3Bytes == [0x45, 0x56, 0x43] {
                                let metaSizeData = readFully(fileHandle: fileHandle, count: 2)
                                guard metaSizeData.count == 2 else { continuation.finish(); return }
                                var msOffset = 0
                                let metadataSize = Int(try readUInt16BEFromBytes([UInt8](metaSizeData), offset: &msOffset))
                                let metaData = readFully(fileHandle: fileHandle, count: metadataSize)
                                guard metaData.count == metadataSize else { continuation.finish(); return }
                                
                                var headerChunk: [UInt8] = [0x56, 0x45, 0x56, 0x43]
                                headerChunk.append(contentsOf: metaSizeData)
                                headerChunk.append(contentsOf: metaData)
                                continuation.yield(headerChunk)
                                continue
                            }
                            continuation.finish()
                            return
                        }
                        
                        let flag = firstByte
                        var chunk: [UInt8] = [flag]
                        if flag == 0x01 {
                            continuation.yield(chunk)
                        } else if flag == 0x00 || flag == 0x02 {
                            let headerBytes = readFully(fileHandle: fileHandle, count: 24)
                            guard headerBytes.count == 24 else { continuation.finish(); return }
                            chunk.append(contentsOf: headerBytes)
                            
                            var hsOffset = 0
                            _ = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
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
                            continuation.yield(chunk)
                        } else {
                            continuation.finish()
                            return
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

