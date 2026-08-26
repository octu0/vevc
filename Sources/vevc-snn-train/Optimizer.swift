import Foundation

public final class AdamOptimizer: @unchecked Sendable {
    public var lr: Float
    public var beta1: Float
    public var beta2: Float
    public var eps: Float
    public var weightDecay: Float
    public var clipNorm: Float

    private var stepCount: Int = 0
    private var mBuffers: [[Float]] = []
    private var vBuffers: [[Float]] = []

    public init(
        lr: Float = 0.005,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        eps: Float = 1e-8,
        weightDecay: Float = 1e-4,
        clipNorm: Float = 1.0
    ) {
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weightDecay = weightDecay
        self.clipNorm = clipNorm
    }

    /// Allocates / initializes moment buffers to match parameter tensors.
    public func registerParameters(paramShapes: [Int]) {
        self.mBuffers = paramShapes.map { [Float](repeating: 0.0, count: $0) }
        self.vBuffers = paramShapes.map { [Float](repeating: 0.0, count: $0) }
        self.stepCount = 0
    }

    /// Performs global gradient clipping across all parameter gradients.
    public func clipGradients(grads: inout [[Float]]) {
        guard 0.0 < clipNorm else { return }

        var sumSq: Float = 0.0
        for g in grads {
            let count = g.count
            g.withUnsafeBufferPointer { ptr in
                var idx = 0
                let simdCount = count & ~15
                while idx < simdCount {
                    let vec = ptr.baseAddress!.advanced(by: idx).withMemoryRebound(to: SIMD16<Float>.self, capacity: 1) { $0.pointee }
                    let sq = vec * vec
                    sumSq += sq.sum()
                    idx &+= 16
                }
                while idx < count {
                    let v = ptr[idx]
                    sumSq += v * v
                    idx &+= 1
                }
            }
        }

        let norm = sqrtf(sumSq)
        if clipNorm < norm {
            let scale = clipNorm / (norm + 1e-8)
            for i in 0..<grads.count {
                let count = grads[i].count
                grads[i].withUnsafeMutableBufferPointer { ptr in
                    var idx = 0
                    let simdCount = count & ~15
                    let scaleVec = SIMD16<Float>(repeating: scale)
                    while idx < simdCount {
                        let p = ptr.baseAddress!.advanced(by: idx).withMemoryRebound(to: SIMD16<Float>.self, capacity: 1) { $0 }
                        p.pointee = p.pointee * scaleVec
                        idx &+= 16
                    }
                    while idx < count {
                        ptr[idx] *= scale
                        idx &+= 1
                    }
                }
            }
        }
    }

    /// Vectorized Adam parameter update using SIMD16.
    public func step(params: inout [[Float]], grads: inout [[Float]]) {
        stepCount &+= 1
        let t = Float(stepCount)
        let beta1Pow = powf(beta1, t)
        let beta2Pow = powf(beta2, t)
        let biasCorrection1 = 1.0 - beta1Pow
        let biasCorrection2 = 1.0 - beta2Pow

        clipGradients(grads: &grads)

        let beta1Val = self.beta1
        let beta2Val = self.beta2
        let oneMinusBeta1 = 1.0 - beta1Val
        let oneMinusBeta2 = 1.0 - beta2Val
        let lrVal = self.lr
        let epsVal = self.eps
        let decayVal = self.weightDecay

        for i in 0..<params.count {
            let count = params[i].count
            params[i].withUnsafeMutableBufferPointer { pPtr in
                grads[i].withUnsafeBufferPointer { gPtr in
                    mBuffers[i].withUnsafeMutableBufferPointer { mPtr in
                        vBuffers[i].withUnsafeMutableBufferPointer { vPtr in
                            var idx = 0
                            let simdCount = count & ~15

                            let b1Vec = SIMD16<Float>(repeating: beta1Val)
                            let b2Vec = SIMD16<Float>(repeating: beta2Val)
                            let oneMb1Vec = SIMD16<Float>(repeating: oneMinusBeta1)
                            let oneMb2Vec = SIMD16<Float>(repeating: oneMinusBeta2)
                            let lrVec = SIMD16<Float>(repeating: lrVal)
                            let epsVec = SIMD16<Float>(repeating: epsVal)
                            let bc1Vec = SIMD16<Float>(repeating: biasCorrection1)
                            let bc2Vec = SIMD16<Float>(repeating: biasCorrection2)
                            let decayVec = SIMD16<Float>(repeating: decayVal)

                            while idx < simdCount {
                                let gVec = gPtr.baseAddress!.advanced(by: idx).withMemoryRebound(to: SIMD16<Float>.self, capacity: 1) { $0.pointee }
                                let pVec = pPtr.baseAddress!.advanced(by: idx).withMemoryRebound(to: SIMD16<Float>.self, capacity: 1) { $0.pointee }
                                let mVec = mPtr.baseAddress!.advanced(by: idx).withMemoryRebound(to: SIMD16<Float>.self, capacity: 1) { $0.pointee }
                                let vVec = vPtr.baseAddress!.advanced(by: idx).withMemoryRebound(to: SIMD16<Float>.self, capacity: 1) { $0.pointee }

                                let newM = (b1Vec * mVec) + (oneMb1Vec * gVec)
                                let newV = (b2Vec * vVec) + (oneMb2Vec * gVec * gVec)

                                mPtr.baseAddress!.advanced(by: idx).withMemoryRebound(to: SIMD16<Float>.self, capacity: 1) { $0.pointee = newM }
                                vPtr.baseAddress!.advanced(by: idx).withMemoryRebound(to: SIMD16<Float>.self, capacity: 1) { $0.pointee = newV }

                                let mHat = newM / bc1Vec
                                let vHat = newV / bc2Vec
                                let denom = vHat.squareRoot() + epsVec
                                let update = (lrVec * (mHat / denom)) + (lrVec * decayVec * pVec)
                                let newP = pVec - update

                                pPtr.baseAddress!.advanced(by: idx).withMemoryRebound(to: SIMD16<Float>.self, capacity: 1) { $0.pointee = newP }

                                idx &+= 16
                            }

                            while idx < count {
                                let g = gPtr[idx]
                                let p = pPtr[idx]
                                let newM = (beta1Val * mPtr[idx]) + (oneMinusBeta1 * g)
                                let newV = (beta2Val * vPtr[idx]) + (oneMinusBeta2 * g * g)

                                mPtr[idx] = newM
                                vPtr[idx] = newV

                                let mHat = newM / biasCorrection1
                                let vHat = newV / biasCorrection2
                                let update = (lrVal * (mHat / (sqrtf(vHat) + epsVal))) + (lrVal * decayVal * p)
                                pPtr[idx] = p - update

                                idx &+= 1
                            }
                        }
                    }
                }
            }
        }
    }
}
