// MARK: - Cross-Scale Implicit Conditioning & Latent Scale Modulation

import Foundation

/// DWTラテント空間における階層間・暗黙的コンディショニングモジュール
public enum ImplicitConditioning {
    
    /// 下位層の再構築画像（reference）から上位層用コンディショニングマップ（sigmaMap）を生成する
    public static func generateSigmaMap(
        reference: UnsafeBufferPointer<Int16>,
        refWidth: Int,
        refHeight: Int,
        targetWidth: Int,
        targetHeight: Int,
        scale: Float = 0.5,
        baseSigma: Float = 100.0
    ) -> [Float] {
        let totalCount = targetWidth * targetHeight
        if refWidth <= 0 || refHeight <= 0 || targetWidth <= 0 || targetHeight <= 0 {
            return Array(repeating: baseSigma, count: totalCount)
        }
        guard let refBase = reference.baseAddress else {
            return Array(repeating: baseSigma, count: totalCount)
        }
        
        var sigmaMap = [Float](repeating: baseSigma, count: totalCount)
        
        sigmaMap.withUnsafeMutableBufferPointer { sPtr in
            guard let sBase = sPtr.baseAddress else { return }
            
            for ty in 0..<targetHeight {
                let ry = (ty * refHeight) / targetHeight
                let ryNext = min(refHeight - 1, ry + 1)
                let refRowCurr = ry * refWidth
                let refRowNext = ryNext * refWidth
                let targetRow = ty * targetWidth
                
                for tx in 0..<targetWidth {
                    let rx = (tx * refWidth) / targetWidth
                    let rxNext = min(refWidth - 1, rx + 1)
                    
                    let v00 = Float(refBase[refRowCurr + rx])
                    let v01 = Float(refBase[refRowCurr + rxNext])
                    let v10 = Float(refBase[refRowNext + rx])
                    
                    let gradX = abs(v01 - v00)
                    let gradY = abs(v10 - v00)
                    let energy = gradX + gradY
                    
                    // baseSigma(100.0) ensures flat areas get sigma=100.0
                    // scale * energy increases sigma on edges, reducing their coded magnitude.
                    let sigmaVal = max(baseSigma, (scale * energy) + baseSigma)
                    sBase[targetRow + tx] = sigmaVal
                }
            }
        }
        
        return sigmaMap
    }
    
    /// DWTブロック内の高周波サブバンド (HL, LH, HH) に Forward 変調を適用
    /// 誤差を防ぐため、100倍にしてからsigmaで割る
    @inline(__always)
    static func modulateLatentBlock(
        block: BlockView,
        blockSize: Int,
        sigmaMap: UnsafeBufferPointer<Float>,
        sigmaWidth: Int,
        sigmaHeight: Int,
        blockStartX: Int,
        blockStartY: Int,
        threshold: Float = 0.0,
        modScale: Float = 100.0
    ) {
        guard let sBase = sigmaMap.baseAddress else { return }
        let half = blockSize / 2
        
        for y in 0..<blockSize {
            let rowPtr = block.rowPointer(y: y)
            let sigY = min(blockStartY + y, sigmaHeight - 1)
            let sigRow = sigY * sigmaWidth
            
            let isTopHalf = (y < half)
            for x in 0..<blockSize {
                if isTopHalf {
                    if x < half {
                        // LL 成分は変調しない
                        continue
                    }
                }
                
                let sigX = min(blockStartX + x, sigmaWidth - 1)
                let sigma = max(1.0, sBase[sigRow + sigX])
                
                let val = Float(rowPtr[x])
                // N * 100 / sigma
                let zScaled = (val * modScale) / sigma
                
                var finalZ = zScaled
                if 0.0 < threshold {
                    if abs(zScaled) <= threshold {
                        finalZ = 0.0
                    }
                }
                
                let rounded = round(finalZ)
                if rounded < -32768.0 {
                    rowPtr[x] = -32768
                } else {
                    if 32767.0 < rounded {
                        rowPtr[x] = 32767
                    } else {
                        rowPtr[x] = Int16(rounded)
                    }
                }
            }
        }
    }
    
    /// DWTブロック内の高周波サブバンド (HL, LH, HH) に Inverse 逆変調を適用
    @inline(__always)
    static func demodulateLatentBlock(
        block: BlockView,
        blockSize: Int,
        sigmaMap: UnsafeBufferPointer<Float>,
        sigmaWidth: Int,
        sigmaHeight: Int,
        blockStartX: Int,
        blockStartY: Int,
        modScale: Float = 100.0
    ) {
        guard let sBase = sigmaMap.baseAddress else { return }
        let half = blockSize / 2
        
        for y in 0..<blockSize {
            let rowPtr = block.rowPointer(y: y)
            let sigY = min(blockStartY + y, sigmaHeight - 1)
            let sigRow = sigY * sigmaWidth
            
            let isTopHalf = (y < half)
            for x in 0..<blockSize {
                if isTopHalf {
                    if x < half {
                        // LL 成分は逆変調しない
                        continue
                    }
                }
                
                let sigX = min(blockStartX + x, sigmaWidth - 1)
                let sigma = max(1.0, sBase[sigRow + sigX])
                
                let zVal = Float(rowPtr[x])
                // デコード時は (Z * sigma) / 100
                let xRestored = round((zVal * sigma) / modScale)
                
                if xRestored < -32768.0 {
                    rowPtr[x] = -32768
                } else {
                    if 32767.0 < xRestored {
                        rowPtr[x] = 32767
                    } else {
                        rowPtr[x] = Int16(xRestored)
                    }
                }
            }
        }
    }
    
    /// ブロック配列全域に対する Forward 変調
    static func applyLatentModulation(
        blocks: inout [BlockView],
        blockSize: Int,
        planeWidth: Int,
        planeHeight: Int,
        refImagePlane: [Int16],
        refWidth: Int,
        refHeight: Int,
        threshold: Float = 0.0
    ) {
        let colCount = (planeWidth + blockSize - 1) / blockSize
        refImagePlane.withUnsafeBufferPointer { refPtr in
            let sigmaMap = generateSigmaMap(
                reference: refPtr,
                refWidth: refWidth,
                refHeight: refHeight,
                targetWidth: planeWidth,
                targetHeight: planeHeight
            )
            sigmaMap.withUnsafeBufferPointer { sigPtr in
                for i in blocks.indices {
                    let bx = (i % colCount) * blockSize
                    let by = (i / colCount) * blockSize
                    modulateLatentBlock(
                        block: blocks[i],
                        blockSize: blockSize,
                        sigmaMap: sigPtr,
                        sigmaWidth: planeWidth,
                        sigmaHeight: planeHeight,
                        blockStartX: bx,
                        blockStartY: by,
                        threshold: threshold
                    )
                }
            }
        }
    }
    
    /// ブロック配列全域に対する Inverse 逆変調
    static func applyLatentDemodulation(
        blocks: [BlockView],
        blockSize: Int,
        planeWidth: Int,
        planeHeight: Int,
        refImagePlane: [Int16],
        refWidth: Int,
        refHeight: Int
    ) {
        let colCount = (planeWidth + blockSize - 1) / blockSize
        refImagePlane.withUnsafeBufferPointer { refPtr in
            let sigmaMap = generateSigmaMap(
                reference: refPtr,
                refWidth: refWidth,
                refHeight: refHeight,
                targetWidth: planeWidth,
                targetHeight: planeHeight
            )
            sigmaMap.withUnsafeBufferPointer { sigPtr in
                for i in blocks.indices {
                    let bx = (i % colCount) * blockSize
                    let by = (i / colCount) * blockSize
                    demodulateLatentBlock(
                        block: blocks[i],
                        blockSize: blockSize,
                        sigmaMap: sigPtr,
                        sigmaWidth: planeWidth,
                        sigmaHeight: planeHeight,
                        blockStartX: bx,
                        blockStartY: by
                    )
                }
            }
        }
    }
}
