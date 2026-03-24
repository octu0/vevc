import Foundation

class CoreDecoder {
    let maxLayer: Int
    private var prevReconstructed: PlaneData420? = nil
    
    init(maxLayer: Int = 2) {
        self.maxLayer = maxLayer
    }
    
#if (arch(arm64) || arch(x86_64) || arch(wasm32))
    func decode(chunk: [UInt8]) async throws -> YCbCrImage {
        guard chunk.count >= 4 else { throw DecodeError.insufficientData }
        var offset = 0
        let magic = Array(chunk[offset..<(offset + 4)])
        offset += 4
        
        switch magic {
        case [0x56, 0x45, 0x56, 0x49]:
            let len = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
            guard (offset + len) <= chunk.count else { throw DecodeError.insufficientData }
            let data = Array(chunk[offset..<(offset + len)])
            
            let img16 = try await decodeSpatialLayers(r: data, maxLayer: maxLayer)
            let pd = PlaneData420(img16: img16)
            prevReconstructed = pd
            return pd.toYCbCr()

        case [0x56, 0x45, 0x56, 0x50]:
            let mvsCount = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
            let mvDataLen = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
            guard (offset + mvDataLen) <= chunk.count else { throw DecodeError.insufficientData }

            let mvData = Array(chunk[offset..<(offset + mvDataLen)])
            offset += mvDataLen
            var mvBr = try EntropyDecoder(data: mvData)

            let mbSize = 64
            guard let prevWidth = prevReconstructed?.width, let prevHeight = prevReconstructed?.height else { throw DecodeError.invalidHeader }
            let mbCols = (prevWidth + mbSize - 1) / mbSize
            let mbRows = (prevHeight + mbSize - 1) / mbSize

            var grid = MVGrid(width: prevWidth, height: prevHeight, minSize: 8)
            var ctuNodes = [MotionNode]()
            for mbY in 0..<mbRows {
                let startY = mbY * mbSize
                for mbX in 0..<mbCols {
                    let startX = mbX * mbSize
                    if ctuNodes.count < mvsCount {
                        let node = try decodeMotionQuadtreeNode(w: prevWidth, h: prevHeight, startX: startX, startY: startY, size: mbSize, grid: &grid, br: &mvBr)
                        ctuNodes.append(node)
                    }
                }
            }
            let motionTree = MotionTree(ctuNodes: ctuNodes, width: prevWidth, height: prevHeight)
            
            let len = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
            guard (offset + len) <= chunk.count else { throw DecodeError.insufficientData }
            let data = Array(chunk[offset..<(offset + len)])
            
            let img16 = try await decodeSpatialLayers(r: data, maxLayer: maxLayer)
            let residual = PlaneData420(img16: img16)
            
            if let prev = prevReconstructed {
                let predicted = await applyMotionQuadtree(prev: prev, tree: motionTree)
                let curr = await addPlanes(residual: residual, predicted: predicted)
                prevReconstructed = curr
                return curr.toYCbCr()
            } else {
                return residual.toYCbCr()
            }
            
        case [0x56, 0x45, 0x4F, 0x49]: // VEOI
            let len = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
            guard (offset + len) <= chunk.count else { throw DecodeError.insufficientData }
            let data = Array(chunk[offset..<(offset + len)])
            
            let img16 = try await decodeBase32(r: data, layer: 0)
            let pd = PlaneData420(img16: img16)
            prevReconstructed = pd
            return pd.toYCbCr()

        case [0x56, 0x45, 0x4F, 0x50]: // VEOP
            let mvsCount = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
            let mvDataLen = Int(try readUInt32BEFromBytes(chunk, offset: &offset))

            guard (offset + mvDataLen) <= chunk.count else { throw DecodeError.insufficientData }

            let mvData = Array(chunk[offset..<(offset + mvDataLen)])
            offset += mvDataLen
            var mvBr = try EntropyDecoder(data: mvData)

            let mbSize = 64
            guard let prevWidth = prevReconstructed?.width, let prevHeight = prevReconstructed?.height else { throw DecodeError.invalidHeader }
            let mbCols = (prevWidth + mbSize - 1) / mbSize
            let mbRows = (prevHeight + mbSize - 1) / mbSize

            var grid = MVGrid(width: prevWidth, height: prevHeight, minSize: 8)
            var ctuNodes = [MotionNode]()
            for mbY in 0..<mbRows {
                let startY = mbY * mbSize
                for mbX in 0..<mbCols {
                    let startX = mbX * mbSize
                    if ctuNodes.count < mvsCount {
                        let node = try decodeMotionQuadtreeNode(w: prevWidth, h: prevHeight, startX: startX, startY: startY, size: mbSize, grid: &grid, br: &mvBr)
                        ctuNodes.append(node)
                    }
                }
            }
            let motionTree = MotionTree(ctuNodes: ctuNodes, width: prevWidth, height: prevHeight)
            
            let len = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
            guard (offset + len) <= chunk.count else { throw DecodeError.insufficientData }
            let data = Array(chunk[offset..<(offset + len)])
            
            let img16 = try await decodeBase32(r: data, layer: 0)
            let residual = PlaneData420(img16: img16)
            
            if let prev = prevReconstructed {
                let predicted = await applyMotionQuadtree(prev: prev, tree: motionTree)
                let curr = await addPlanes(residual: residual, predicted: predicted)
                prevReconstructed = curr
                return curr.toYCbCr()
            } else {
                return residual.toYCbCr()
            }
            
        default: 
             throw DecodeError.invalidHeader
        }
    }

#else
    func decode(chunk: [UInt8]) async throws -> YCbCrImage {
        throw DecodeError.unsupportedArchitecture
    }
#endif
}

public struct Decoder: Sendable {
    public let maxLayer: Int
    public let maxConcurrency: Int

    public init(maxLayer: Int = 2, maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount) {
        self.maxLayer = maxLayer
        self.maxConcurrency = maxConcurrency
    }

    #if (arch(arm64) || arch(x86_64) || arch(wasm32))
    /// Decodes a stream of bitstream chunks into a stream of images using GOP-level parallelization.
    public func decode(stream: AsyncStream<[UInt8]>) -> AsyncThrowingStream<YCbCrImage, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                var iterator = stream.makeAsyncIterator()
                var currentGOPIndex = 0
                var nextGOPIndexToYield = 0
                var completedGOPs: [Int: [YCbCrImage]] = [:]
                
                var bufferedChunk: [UInt8]? = nil
                
                func readNextGOP() async throws -> [[UInt8]]? {
                    var gop: [[UInt8]] = []
                    
                    if let first = bufferedChunk {
                        gop.append(first)
                        bufferedChunk = nil
                    } else if let chunk = await iterator.next() {
                        gop.append(chunk)
                    } else {
                        return nil // EOF
                    }
                    
                    while let chunk = await iterator.next() {
                        guard chunk.count >= 4 else { throw DecodeError.insufficientData }
                        let magic = Array(chunk[0..<4])
                        if magic == [0x56, 0x45, 0x56, 0x49] || magic == [0x56, 0x45, 0x4F, 0x49] {
                            bufferedChunk = chunk
                            return gop
                        }
                        gop.append(chunk)
                    }
                    
                    return gop
                }
                
                do {
                    try await withThrowingTaskGroup(of: (Int, [YCbCrImage]).self) { group in
                        var activeTasks = 0
                        
                        while activeTasks < self.maxConcurrency {
                            guard let gop = try await readNextGOP() else { break }
                            let idx = currentGOPIndex
                            currentGOPIndex += 1
                            group.addTask {
                                let decoder = CoreDecoder(maxLayer: self.maxLayer)
                                var decodedFrames: [YCbCrImage] = []
                                for chunk in gop {
                                    let img = try await decoder.decode(chunk: chunk)
                                    decodedFrames.append(img)
                                }
                                return (idx, decodedFrames)
                            }
                            activeTasks += 1
                        }
                        
                        while let result = try await group.next() {
                            activeTasks -= 1
                            let (idx, frames) = result
                            completedGOPs[idx] = frames
                            
                            while let consecutiveFrames = completedGOPs[nextGOPIndexToYield] {
                                for frame in consecutiveFrames {
                                    continuation.yield(frame)
                                }
                                completedGOPs.removeValue(forKey: nextGOPIndexToYield)
                                nextGOPIndexToYield += 1
                            }
                            
                            if let gop = try await readNextGOP() {
                                let newIdx = currentGOPIndex
                                currentGOPIndex += 1
                                group.addTask {
                                    let decoder = CoreDecoder(maxLayer: self.maxLayer)
                                    var newFrames: [YCbCrImage] = []
                                    for chunk in gop {
                                        let img = try await decoder.decode(chunk: chunk)
                                        newFrames.append(img)
                                    }
                                    return (newIdx, newFrames)
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
