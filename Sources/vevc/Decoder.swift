import Foundation

public class Decoder {
    public let maxLayer: Int
    private var prevReconstructed: PlaneData420? = nil
    
    public init(maxLayer: Int = 2) {
        self.maxLayer = maxLayer
    }
    
#if (arch(arm64) || arch(x86_64) || arch(wasm32))
    public func decode(chunk: [UInt8]) async throws -> YCbCrImage {
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
    public func decode(chunk: [UInt8]) async throws -> YCbCrImage {
        throw DecodeError.unsupportedArchitecture
    }
#endif
}
