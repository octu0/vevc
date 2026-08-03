import Foundation

actor ConcurrencyLimiter {
    private var permits: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(limit: Int) {
        self.permits = limit
    }

    @inline(__always)
    func wait() async {
        if 0 < permits {
            permits -= 1
        } else {
            await withCheckedContinuation { continuation in
                waiters.append(continuation)
            }
        }
    }

    @inline(__always)
    func signal() {
        let isEmpty = waiters.isEmpty
        if isEmpty != true {
            let next = waiters.removeFirst()
            next.resume()
        } else {
            permits += 1
        }
    }
}

public actor StreamingDecoderActor {
    let maxLayer: Int
    let width: Int
    let height: Int
    let pool: BlockViewPool
    let profile: UInt8
    
    private var previousReconstructed: PlaneData420?
    private var firstReconstructed: PlaneData420?
    private var seenY = Set<UnsafeMutableRawPointer>()
    private var roundOffsetIndex = 0
    private var cachedYCbCrImage: YCbCrImage?
    
    public init(maxLayer: Int = 2, width: Int = 0, height: Int = 0, profile: UInt8 = 0x01) {
        self.maxLayer = maxLayer
        self.width = width
        self.height = height
        self.pool = BlockViewPool()
        self.profile = profile
    }

    private func renderToYCbCr(pd: PlaneData420) -> YCbCrImage {
        let w = pd.width
        let h = pd.height
        if var cached = cachedYCbCrImage, cached.width == w && cached.height == h {
            pd.toYCbCr(into: &cached)
            cachedYCbCrImage = cached
            return cached
        } else {
            var newImg = YCbCrImage(width: w, height: h)
            pd.toYCbCr(into: &newImg)
            cachedYCbCrImage = newImg
            return newImg
        }
    }
    
    @inline(__always)
    public func decodeNextFrame(chunk: [UInt8]) async throws -> YCbCrImage? {
        guard chunk.isEmpty != true else { return nil }
        
        var offset = 0
        let frameHeader = try VEVCFrameHeader.deserialize(from: chunk, offset: &offset, profile: profile)
        if frameHeader.isCopyFrame {
            guard let prev = previousReconstructed else {
                throw DecodeError.insufficientDataContext("Copy frame without previous frame")
            }
            roundOffsetIndex += 1
            return renderToYCbCr(pd: prev)
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
        let nextPd: PlaneData420? = if useBidirectional { firstReconstructed } else { nil }
        let img16 = try await decodeSpatialLayers(
            r: chunk, pool: pool, maxLayer: maxLayer, dx: width, dy: height,
            predictedPd: previousReconstructed, nextPd: nextPd, roundOffset: roundOffsetIndex % 2, profile: profile
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
        return renderToYCbCr(pd: pd)
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

    @inline(__always)
    private func createGOPTask(
        gopContinuation: AsyncStream<AsyncThrowingStream<YCbCrImage, Error>>.Continuation,
        limiter: ConcurrencyLimiter,
        maxLayer: Int, width: Int, height: Int, profile: UInt8
    ) -> AsyncStream<[UInt8]>.Continuation {
        let (chunkStream, chunkContinuation) = AsyncStream<[UInt8]>.makeStream()
        let (imgStream, imgContinuation) = AsyncThrowingStream<YCbCrImage, Error>.makeStream()
        gopContinuation.yield(imgStream)
        
        let decoderActor = StreamingDecoderActor(maxLayer: maxLayer, width: width, height: height, profile: profile)
        
        Task {
            await limiter.wait()
            do {
                for await gopChunk in chunkStream {
                    if let img = try await decoderActor.decodeNextFrame(chunk: gopChunk) {
                        imgContinuation.yield(img)
                    }
                }
                imgContinuation.finish()
            } catch {
                imgContinuation.finish(throwing: error)
            }
            await limiter.signal()
        }
        
        return chunkContinuation
    }

    @inline(__always)
    public func decode<S: AsyncSequence & Sendable>(stream: S) -> AsyncThrowingStream<YCbCrImage, Error> where S.Element == [UInt8] {
        return AsyncThrowingStream { continuation in
            Task {
                var iterator = stream.makeAsyncIterator()
                
                do {
                    guard let firstChunk = try await iterator.next() else {
                        continuation.finish(throwing: DecodeError.insufficientDataContext("missing VEVC header chunk"))
                        return
                    }
                    var headerOffset = 0
                    let fileHeader = try VEVCFileHeader.deserialize(from: firstChunk, offset: &headerOffset)
                    let effectiveWidth = fileHeader.width
                    let effectiveHeight = fileHeader.height
                    let effectiveFps = fileHeader.framerate
                    let effectiveProfile = fileHeader.profile
                    let maxConcurrency = self.maxConcurrency
                    let maxLayer = self.maxLayer
                    
                    let (gopStream, gopContinuation) = AsyncStream<AsyncThrowingStream<YCbCrImage, Error>>.makeStream()
                    
                    Task {
                        do {
                            for await gopOutput in gopStream {
                                for try await img in gopOutput {
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
                    
                    let limiter = ConcurrencyLimiter(limit: maxConcurrency)
                    var currentGOPInput: AsyncStream<[UInt8]>.Continuation? = nil
                    
                    let processChunk: ([UInt8]) -> Void = { chunk in
                        var offset = 0
                        let frameHeader = try? VEVCFrameHeader.deserialize(from: chunk, offset: &offset, profile: effectiveProfile)
                        let isIFrame = frameHeader?.isIFrame ?? false
                        
                        if isIFrame {
                            currentGOPInput?.finish()
                            currentGOPInput = createGOPTask(
                                gopContinuation: gopContinuation,
                                limiter: limiter,
                                maxLayer: maxLayer, width: effectiveWidth, height: effectiveHeight, profile: effectiveProfile
                            )
                        }
                        
                        if let currentInput = currentGOPInput {
                            currentInput.yield(chunk)
                        } else {
                            let newInput = createGOPTask(
                                gopContinuation: gopContinuation,
                                limiter: limiter,
                                maxLayer: maxLayer, width: effectiveWidth, height: effectiveHeight, profile: effectiveProfile
                            )
                            currentGOPInput = newInput
                            newInput.yield(chunk)
                        }
                    }

                    if headerOffset < firstChunk.count {
                        let remainder = Array(firstChunk[headerOffset...])
                        processChunk(remainder)
                    }

                    while let chunk = try await iterator.next() {
                        processChunk(chunk)
                    }
                    currentGOPInput?.finish()
                    gopContinuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    @inline(__always)
    public func decode(data: [UInt8]) async throws -> [YCbCrImage] {
        if data.isEmpty { return [] }
        
        var offset = 0
        var chunks: [[UInt8]] = []
        var headerChunk: [UInt8]? = nil
        var currentProfile: UInt8 = 0x01
        while offset < data.count {
            if offset + 4 <= data.count && data[offset] == 0x56 && data[offset+1] == 0x45 && data[offset+2] == 0x56 && data[offset+3] == 0x43 {
                let headerStart = offset
                offset += 4
                let metadataSize = Int(try readUInt16BEFromBytes(data, offset: &offset))
                if metadataSize > 0 {
                    currentProfile = data[offset]
                }
                offset += metadataSize
                headerChunk = Array(data[headerStart..<offset])
            } else {
                let chunkStart = offset
                let frameHeader = try VEVCFrameHeader.deserialize(from: data, offset: &offset, profile: currentProfile)
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

    @inline(__always)
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
    
    @inline(__always)
    public func decode(fileHandle: FileHandle) -> AsyncThrowingStream<YCbCrImage, Error> {
        let stream = AsyncStream<[UInt8]> { continuation in
            Task {
                do {
                    var currentProfile: UInt8 = 0x01
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
                                
                                if metadataSize > 0 {
                                    currentProfile = metaData[0]
                                }
                                
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
                        let copyFrameFlag: UInt8 = 0x01
                        
                        if flag == copyFrameFlag {
                            continuation.yield(chunk)
                        } else {
                            let isPFrame = (flag & 0x0F) == 0x00
                            let headerSize = (currentProfile == 0x02 && isPFrame) ? 24 : 20
                            let headerBytes = readFully(fileHandle: fileHandle, count: headerSize)
                            guard headerBytes.count == headerSize else { continuation.finish(); return }
                            chunk.append(contentsOf: headerBytes)
                            
                            var hsOffset = 0
                            let skipMapSize = (currentProfile == 0x02 && isPFrame) ? Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset)) : 0
                            let mvsSize = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                            let refDirBytes = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                            let layer0Size = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                            let layer1Size = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                            let layer2Size = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                            
                            let payloadSize = skipMapSize + mvsSize + refDirBytes + layer0Size + layer1Size + layer2Size
                            if 0 < payloadSize {
                                let payloadBody = readFully(fileHandle: fileHandle, count: payloadSize)
                                guard payloadBody.count == payloadSize else { continuation.finish(); return }
                                chunk.append(contentsOf: payloadBody)
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
