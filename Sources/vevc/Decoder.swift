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
            var mvs = MotionVectors(count: mvsCount)
            guard (offset + mvDataLen) <= chunk.count else { throw DecodeError.insufficientData }

            let mvData = Array(chunk[offset..<(offset + mvDataLen)])
            offset += mvDataLen
            var mvBr = try EntropyDecoder(data: mvData)

            let mbSize = 64
            guard let prevWidth = prevReconstructed?.width else { throw DecodeError.invalidHeader }
            let mbCols = (prevWidth + mbSize - 1) / mbSize

            for i in 0..<mvsCount {
                let mbX = i % mbCols
                let mbY = i / mbCols
                let pmv = calculatePMV(mvs: mvs, mbX: mbX, mbY: mbY, mbCols: mbCols)

                let isSig = try mvBr.decodeBypass()
                if isSig == 0 {
                    mvs.vectors[i] = SIMD2(Int16(pmv.dx), Int16(pmv.dy))
                } else {
                    let sx = try mvBr.decodeBypass()
                    let mx = try decodeExpGolomb(decoder: &mvBr)

                    let mvdX = sx == 1 ? -1 * Int(mx) : Int(mx)
                    
                    let sy = try mvBr.decodeBypass()
                    let my = try decodeExpGolomb(decoder: &mvBr)

                    let mvdY = sy == 1 ? -1 * Int(my) : Int(my)

                    mvs.vectors[i] = SIMD2(Int16(mvdX + pmv.dx), Int16(mvdY + pmv.dy))
                }
            }
            
            let len = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
            guard (offset + len) <= chunk.count else { throw DecodeError.insufficientData }
            let data = Array(chunk[offset..<(offset + len)])
            
            let img16 = try await decodeSpatialLayers(r: data, maxLayer: maxLayer)
            let residual = PlaneData420(img16: img16)
            
            if let prev = prevReconstructed {
                let predicted = await applyMBME(prev: prev, mvs: mvs)
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
            var mvs = MotionVectors(count: mvsCount)
            guard (offset + mvDataLen) <= chunk.count else { throw DecodeError.insufficientData }

            let mvData = Array(chunk[offset..<(offset + mvDataLen)])
            offset += mvDataLen
            var mvBr = try EntropyDecoder(data: mvData)

            let mbSize = 64
            guard let prevWidth = prevReconstructed?.width else { throw DecodeError.invalidHeader }
            let mbCols = (prevWidth + mbSize - 1) / mbSize

            for i in 0..<mvsCount {
                let mbX = i % mbCols
                let mbY = i / mbCols
                let pmv = calculatePMV(mvs: mvs, mbX: mbX, mbY: mbY, mbCols: mbCols)

                let isSig = try mvBr.decodeBypass()
                if isSig == 0 {
                    mvs.vectors[i] = SIMD2(Int16(pmv.dx), Int16(pmv.dy))
                } else {
                    let sx = try mvBr.decodeBypass()
                    let mx = try decodeExpGolomb(decoder: &mvBr)
                    let mvdX = sx == 1 ? -1 * Int(mx) : Int(mx)

                    let sy = try mvBr.decodeBypass()
                    let my = try decodeExpGolomb(decoder: &mvBr)
                    let mvdY = sy == 1 ? -1 * Int(my) : Int(my)

                    mvs.vectors[i] = SIMD2(Int16(mvdX + pmv.dx), Int16(mvdY + pmv.dy))
                }
            }
            
            let len = Int(try readUInt32BEFromBytes(chunk, offset: &offset))
            guard (offset + len) <= chunk.count else { throw DecodeError.insufficientData }
            let data = Array(chunk[offset..<(offset + len)])
            
            let img16 = try await decodeBase32(r: data, layer: 0)
            let residual = PlaneData420(img16: img16)
            
            if let prev = prevReconstructed {
                let predicted = await applyMBME(prev: prev, mvs: mvs)
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
    
    @inline(__always)
    private func readUInt32BEFromBytes(_ data: [UInt8], offset: inout Int) throws -> UInt32 {
        guard offset + 4 <= data.count else { throw DecodeError.insufficientData }
        let val = (UInt32(data[offset]) << 24) | (UInt32(data[offset+1]) << 16) | (UInt32(data[offset+2]) << 8) | UInt32(data[offset+3])
        offset += 4
        return val
    }
#else
    public func decode(chunk: [UInt8]) async throws -> YCbCrImage {
        throw DecodeError.unsupportedArchitecture
    }
#endif
}
