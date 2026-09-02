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
    let gop: Int
    let temporalLayers: Int
    
    var previousReconstructed: PlaneData420? // internal for drift diagnostics
    private var previousT0Reconstructed: PlaneData420?
    private var firstReconstructed: PlaneData420?
    private var seenY = Set<Int>()
    private var framesSinceKeyframe = 0
    private var temporalFrameIndex = 0
    private var roundOffsetIndex = 0
    let entropyHistories: FrameEntropyHistories? // internal for history-consistency gate tests
    let mvPredictionState: MVPredictionState?
    private var cachedYCbCrImage: YCbCrImage?
    // Quarter-resolution L0 reference chain (One-Pyramid §4). Only needed
    // when decoding above layer0; the maxLayer==0 pipeline is its own chain.
    // Internal so the L0 bit-exactness gate tests can compare chains.
    let l0State = L0RefState()
    // rANSContext scratch (~1.1 MB). Allocated once per decoder instance and
    // reused for every frame. Only the base8 luma plane can carry rANSContext,
    // so the chroma planes decoded concurrently with it never touch it; each
    // GOP-parallel decoder owns its own actor and therefore its own workspace.
    let ransContextWorkspace: rANSContextWorkspace?
    // Concurrent entropy decode of the 9 profile-2 streams. Wins per-frame
    // latency on a single stream; under GOP-parallel throughput decoding it
    // only adds overhead, so the GOP-parallel Decoder turns it off.
    let parallelEntropy: Bool

    public init(maxLayer: Int, width: Int, height: Int, profile: UInt8, gop: Int, temporalLayers: Int, parallelEntropy: Bool) {
        self.maxLayer = maxLayer
        self.width = width
        self.height = height
        self.pool = BlockViewPool()
        self.profile = profile
        self.gop = gop
        self.temporalLayers = temporalLayers
        self.parallelEntropy = parallelEntropy
        let isProfile2 = (profile == 0x02)
        if isProfile2 {
            self.entropyHistories = FrameEntropyHistories()
            self.mvPredictionState = MVPredictionState()
            self.ransContextWorkspace = rANSContextWorkspace()
        } else {
            self.entropyHistories = nil
            self.mvPredictionState = nil
            self.ransContextWorkspace = nil
        }
    }

    public init(width: Int, height: Int) {
        self.init(maxLayer: 2, width: width, height: height, profile: 0x01, gop: 12, temporalLayers: 1, parallelEntropy: true)
    }

    public init(maxLayer: Int, width: Int, height: Int) {
        self.init(maxLayer: maxLayer, width: width, height: height, profile: 0x01, gop: 12, temporalLayers: 1, parallelEntropy: true)
    }

    public init(maxLayer: Int, width: Int, height: Int, profile: UInt8) {
        self.init(maxLayer: maxLayer, width: width, height: height, profile: profile, gop: 12, temporalLayers: 1, parallelEntropy: true)
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
            let isT0 = (temporalFrameIndex % 2 == 0)
            let shouldPromoteLTR: Bool
            if profile == 0x02 && 0 < gop && 0 < framesSinceKeyframe && framesSinceKeyframe % gop == 0 {
                if temporalLayers == 2 {
                    shouldPromoteLTR = isT0
                } else {
                    shouldPromoteLTR = true
                }
            } else {
                shouldPromoteLTR = false
            }

            if shouldPromoteLTR {
                if let old = firstReconstructed, let prevRecon = previousReconstructed {
                    let oldY = storageToken(old.y)
                    let prevY = storageToken(prevRecon.y)
                    let t0Y = previousT0Reconstructed.map { storageToken($0.y) }
                    if oldY != prevY && oldY != t0Y {
                        if seenY.contains(oldY) {
                            seenY.remove(oldY)
                            pool.putInt16(old.y)
                            pool.putInt16(old.cb)
                            pool.putInt16(old.cr)
                        }
                    }
                }
                firstReconstructed = previousReconstructed
                if temporalLayers == 2 {
                    previousT0Reconstructed = previousReconstructed
                }
            } else {
                if temporalLayers == 2 {
                    if isT0 {
                        previousT0Reconstructed = previousReconstructed
                    }
                }
            }
            framesSinceKeyframe += 1
            temporalFrameIndex += 1
            roundOffsetIndex += 1
            return renderToYCbCr(pd: prev)
        }
        
        if frameHeader.isIFrame {
            var oldPlanes = [PlaneData420]()
            if let f = firstReconstructed { oldPlanes.append(f) }
            if let p = previousReconstructed { oldPlanes.append(p) }
            if let t0 = previousT0Reconstructed { oldPlanes.append(t0) }
            
            for p in oldPlanes {
                let token = storageToken(p.y)
                if seenY.contains(token) {
                    seenY.remove(token)
                    pool.putInt16(p.y)
                    pool.putInt16(p.cb)
                    pool.putInt16(p.cr)
                }
            }
            seenY.removeAll()
            
            firstReconstructed = nil
            previousReconstructed = nil
            previousT0Reconstructed = nil
            framesSinceKeyframe = 0
            temporalFrameIndex = 0

            roundOffsetIndex = 0
            entropyHistories?.reset()
            mvPredictionState?.resetForKeyframe()
        }

        let isPFrame = (previousReconstructed != nil)
        let useBidirectional = isPFrame && firstReconstructed != nil
        let nextPd: PlaneData420? = if useBidirectional { firstReconstructed } else { nil }
        
        let predictedPd: PlaneData420?
        if temporalLayers == 2 {
            if let t0 = previousT0Reconstructed {
                predictedPd = t0
            } else {
                predictedPd = previousReconstructed
            }
        } else {
            predictedPd = previousReconstructed
        }
        let isT0 = (temporalFrameIndex % 2 == 0)
        let updateL0: Bool
        if temporalLayers == 2 {
            updateL0 = isT0
        } else {
            updateL0 = true
        }
        let img16: Image16
        if profile == 0x02 {
            switch maxLayer {
            case 0:
                img16 = try await decodeSpatialLayersForProfile2Base8Only(
                    r: chunk, pool: pool, dx: width, dy: height,
                    predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffsetIndex % 2,
                    entropyHistories: entropyHistories, mvState: mvPredictionState, parallelEntropy: parallelEntropy,
                    updateL0Prev: updateL0, ransContextWorkspace: ransContextWorkspace
                )
            case 1:
                img16 = try await decodeSpatialLayersForProfile2WithLayer1(
                    r: chunk, pool: pool, dx: width, dy: height,
                    predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffsetIndex % 2,
                    entropyHistories: entropyHistories, l0State: l0State, mvState: mvPredictionState, parallelEntropy: parallelEntropy,
                    updateL0Prev: updateL0, ransContextWorkspace: ransContextWorkspace
                )
            default:
                img16 = try await decodeSpatialLayersForProfile2Full(
                    r: chunk, pool: pool, dx: width, dy: height,
                    predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffsetIndex % 2,
                    entropyHistories: entropyHistories, l0State: l0State, mvState: mvPredictionState, parallelEntropy: parallelEntropy,
                    updateL0Prev: updateL0, ransContextWorkspace: ransContextWorkspace
                )
            }
        } else {
            switch maxLayer {
            case 0:
                img16 = try await decodeSpatialLayersBase8Only(
                    r: chunk, pool: pool, dx: width, dy: height,
                    predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffsetIndex % 2, entropyHistories: nil
                )
            case 1:
                img16 = try await decodeSpatialLayersWithLayer1(
                    r: chunk, pool: pool, dx: width, dy: height,
                    predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffsetIndex % 2, entropyHistories: nil
                )
            default:
                img16 = try await decodeSpatialLayersFull(
                    r: chunk, pool: pool, dx: width, dy: height,
                    predictedPd: predictedPd, nextPd: nextPd, roundOffset: roundOffsetIndex % 2, entropyHistories: nil
                )
            }
        }
        
        let pd = PlaneData420(img16: img16)
        let yBase = storageToken(pd.y)
        seenY.insert(yBase)
        
        let shouldPromoteLTR: Bool
        if profile == 0x02 && 0 < gop && 0 < framesSinceKeyframe && framesSinceKeyframe % gop == 0 {
            if temporalLayers == 2 {
                shouldPromoteLTR = isT0
            } else {
                shouldPromoteLTR = true
            }
        } else {
            shouldPromoteLTR = false
        }

        if shouldPromoteLTR {
            if let old = firstReconstructed {
                let oldYBase = storageToken(old.y)
                let prevYBase = previousReconstructed.map { storageToken($0.y) }
                let t0YBase = previousT0Reconstructed.map { storageToken($0.y) }
                if oldYBase != prevYBase && oldYBase != t0YBase {
                    if seenY.contains(oldYBase) {
                        seenY.remove(oldYBase)
                        pool.putInt16(old.y)
                        pool.putInt16(old.cb)
                        pool.putInt16(old.cr)
                    }
                }
            }
            firstReconstructed = pd
            if temporalLayers != 2 {
                if let oldPrev = previousReconstructed {
                    let oldPrevYBase = storageToken(oldPrev.y)
                    if seenY.contains(oldPrevYBase) {
                        seenY.remove(oldPrevYBase)
                        pool.putInt16(oldPrev.y)
                        pool.putInt16(oldPrev.cb)
                        pool.putInt16(oldPrev.cr)
                    }
                }
            }
        } else {
            if temporalLayers != 2 {
                if let oldPrev = previousReconstructed {
                    let oldYBase = storageToken(oldPrev.y)
                    let firstYBase = firstReconstructed.map { storageToken($0.y) }
                    if oldYBase != firstYBase {
                        if seenY.contains(oldYBase) {
                            seenY.remove(oldYBase)
                            pool.putInt16(oldPrev.y)
                            pool.putInt16(oldPrev.cb)
                            pool.putInt16(oldPrev.cr)
                        }
                    }
                }
            }
        }
        
        if temporalLayers == 2 {
            if isT0 {
                if let oldT0 = previousT0Reconstructed {
                    let oldT0YBase = storageToken(oldT0.y)
                    let firstYBase = firstReconstructed.map { storageToken($0.y) }
                    if oldT0YBase != firstYBase {
                        if seenY.contains(oldT0YBase) {
                            seenY.remove(oldT0YBase)
                            pool.putInt16(oldT0.y)
                            pool.putInt16(oldT0.cb)
                            pool.putInt16(oldT0.cr)
                        }
                    }
                }
                if let oldPrev = previousReconstructed {
                    let oldPrevYBase = storageToken(oldPrev.y)
                    let firstYBase = firstReconstructed.map { storageToken($0.y) }
                    let oldT0YBase = previousT0Reconstructed.map { storageToken($0.y) }
                    if oldPrevYBase != firstYBase && oldPrevYBase != oldT0YBase && oldPrevYBase != yBase {
                        if seenY.contains(oldPrevYBase) {
                            seenY.remove(oldPrevYBase)
                            pool.putInt16(oldPrev.y)
                            pool.putInt16(oldPrev.cb)
                            pool.putInt16(oldPrev.cr)
                        }
                    }
                }
                previousT0Reconstructed = pd
                previousReconstructed = pd
            } else {
                if let oldPrev = previousReconstructed {
                    let oldPrevYBase = storageToken(oldPrev.y)
                    let firstYBase = firstReconstructed.map { storageToken($0.y) }
                    let t0YBase = previousT0Reconstructed.map { storageToken($0.y) }
                    if oldPrevYBase != firstYBase && oldPrevYBase != t0YBase && oldPrevYBase != yBase {
                        if seenY.contains(oldPrevYBase) {
                            seenY.remove(oldPrevYBase)
                            pool.putInt16(oldPrev.y)
                            pool.putInt16(oldPrev.cb)
                            pool.putInt16(oldPrev.cr)
                        }
                    }
                }
                previousReconstructed = pd
            }
        } else {
            previousReconstructed = pd
        }
        
        if firstReconstructed == nil {
            firstReconstructed = pd
        }
        if previousT0Reconstructed == nil {
            previousT0Reconstructed = pd
        }
        
        framesSinceKeyframe += 1
        temporalFrameIndex += 1
        roundOffsetIndex += 1
        return renderToYCbCr(pd: pd)
    }
}

public struct Decoder: Sendable {
    public let maxLayer: Int
    public let maxConcurrency: Int

    public init(
        maxLayer: Int,
        maxConcurrency: Int
    ) {
        self.maxLayer = maxLayer
        self.maxConcurrency = maxConcurrency
    }

    public init() {
        self.init(maxLayer: 2, maxConcurrency: ProcessInfo.processInfo.activeProcessorCount)
    }

    public init(maxLayer: Int) {
        self.init(maxLayer: maxLayer, maxConcurrency: ProcessInfo.processInfo.activeProcessorCount)
    }

    @inline(__always)
    private func createGOPTask(
        gopContinuation: AsyncStream<AsyncThrowingStream<YCbCrImage, Error>>.Continuation,
        limiter: ConcurrencyLimiter,
        maxLayer: Int, width: Int, height: Int, profile: UInt8, gop: Int, temporalLayers: Int
    ) -> AsyncStream<[UInt8]>.Continuation {
        let (chunkStream, chunkContinuation) = AsyncStream<[UInt8]>.makeStream()
        let (imgStream, imgContinuation) = AsyncThrowingStream<YCbCrImage, Error>.makeStream()
        gopContinuation.yield(imgStream)
        
        // GOP-parallel throughput decoding saturates the cores on its own;
        // per-frame entropy fan-out helps only when this is the sole stream.
        let decoderActor = StreamingDecoderActor(maxLayer: maxLayer, width: width, height: height, profile: profile, gop: gop, temporalLayers: temporalLayers, parallelEntropy: maxConcurrency == 1)
        
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
    public func decodeStream<S: AsyncSequence & Sendable>(stream: S) -> AsyncThrowingStream<YCbCrImage, Error> where S.Element == [UInt8] {
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
                    let effectiveGop = fileHeader.gop
                    let effectiveTemporalLayers = fileHeader.temporalLayers
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
                                maxLayer: maxLayer, width: effectiveWidth, height: effectiveHeight, profile: effectiveProfile, gop: effectiveGop, temporalLayers: effectiveTemporalLayers
                            )
                        }
                        
                        if let currentInput = currentGOPInput {
                            currentInput.yield(chunk)
                        } else {
                            let newInput = createGOPTask(
                                gopContinuation: gopContinuation,
                                limiter: limiter,
                                maxLayer: maxLayer, width: effectiveWidth, height: effectiveHeight, profile: effectiveProfile, gop: effectiveGop, temporalLayers: effectiveTemporalLayers
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
        for try await img in self.decodeStream(stream: stream) {
            images.append(img)
        }
        return images
    }

    @inline(__always)
    public func decodeChunks(chunks: [[UInt8]]) async throws -> [YCbCrImage] {
        let stream = AsyncStream<[UInt8]> { continuation in
            for chunk in chunks {
                continuation.yield(chunk)
            }
            continuation.finish()
        }
        var images: [YCbCrImage] = []
        for try await img in self.decodeStream(stream: stream) {
            images.append(img)
        }
        return images
    }
    
    @inline(__always)
    public func decodeFile(fileHandle: FileHandle) -> AsyncThrowingStream<YCbCrImage, Error> {
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
                            let headerSize: Int
                            if currentProfile == 0x02 && isPFrame {
                                headerSize = 30
                            } else {
                                headerSize = 20
                            }
                            let headerBytes = readFully(fileHandle: fileHandle, count: headerSize)
                            guard headerBytes.count == headerSize else { continuation.finish(); return }
                            chunk.append(contentsOf: headerBytes)
                            
                            var hsOffset = 0
                            let skipMapSize: Int
                            if currentProfile == 0x02 && isPFrame {
                                skipMapSize = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                            } else {
                                skipMapSize = 0
                            }
                            let mvsSize = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                            let refDirBytes = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                            let treeMapBytes: Int
                            if currentProfile == 0x02 && isPFrame {
                                treeMapBytes = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                                // Prediction offsets (lumaOffset, chromaOffset) sit between
                                // treeMapSize and the layer sizes.
                                hsOffset += 2
                            } else {
                                treeMapBytes = 0
                            }
                            let layer0Size = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                            let layer1Size = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                            let layer2Size = Int(try readUInt32BEFromBytes([UInt8](headerBytes), offset: &hsOffset))
                            
                            let payloadSize = skipMapSize + mvsSize + refDirBytes + treeMapBytes + layer0Size + layer1Size + layer2Size
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
        return self.decodeStream(stream: stream)
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
