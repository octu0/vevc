import Foundation

public struct IntraPredictor {
    public enum Mode: UInt8 {
        case dc = 0
        case vertical = 1
        case horizontal = 2
        case planar = 3
    }

    /// Intra Prediction: Generates a predictor block using neighboring reconstructed pixels.
    ///
    /// - Parameters:
    ///   - mode: Prediction mode (DC, Vertical, Horizontal, Planar)
    ///   - block: Output buffer for the predicted block (size: width * height)
    ///   - width: Block width
    ///   - height: Block height
    ///   - top: Reconstructed pixels directly above the block (size: width)
    ///   - left: Reconstructed pixels directly to the left of the block (size: height)
    ///   - topLeft: Reconstructed pixel at the top-left diagonal corner
    @inline(__always)
    public static func predict(
        mode: Mode,
        block: inout [Int16],
        width: Int,
        height: Int,
        top: [Int16]?,
        left: [Int16]?,
        topLeft: Int16 = 0
    ) {
        switch mode {
        case .dc:
            predictDC(block: &block, width: width, height: height, top: top, left: left)
        case .vertical:
            predictVertical(block: &block, width: width, height: height, top: top)
        case .horizontal:
            predictHorizontal(block: &block, width: width, height: height, left: left)
        case .planar:
            predictPlanar(block: &block, width: width, height: height, top: top, left: left, topLeft: topLeft)
        }
    }

    @inline(__always)
    private static func predictDC(block: inout [Int16], width: Int, height: Int, top: [Int16]?, left: [Int16]?) {
        var sum: Int32 = 0
        var count: Int32 = 0

        if let t = top {
            for i in 0..<width {
                sum += Int32(t[i])
            }
            count += Int32(width)
        }

        if let l = left {
            for i in 0..<height {
                sum += Int32(l[i])
            }
            count += Int32(height)
        }

        let dcVal: Int16 = count == 0 ? 128 : Int16(sum / count) // Fallback to 128 if no neighbors
        
        // Loop unrolling / bulk update using Swift's array repeating initializer is best,
        // but here we are writing into an existing inout buffer.
        block.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<(width * height) {
                ptr[i] = dcVal
            }
        }
    }

    @inline(__always)
    private static func predictVertical(block: inout [Int16], width: Int, height: Int, top: [Int16]?) {
        guard let t = top, t.count >= width else {
            predictDC(block: &block, width: width, height: height, top: top, left: nil)
            return
        }
        
        block.withUnsafeMutableBufferPointer { ptr in
            // Unroll rows
            var offset = 0
            for _ in 0..<height {
                for x in 0..<width {
                    ptr[offset + x] = t[x]
                }
                offset += width
            }
        }
    }

    @inline(__always)
    private static func predictHorizontal(block: inout [Int16], width: Int, height: Int, left: [Int16]?) {
        guard let l = left, l.count >= height else {
            predictDC(block: &block, width: width, height: height, top: nil, left: left)
            return
        }
        
        block.withUnsafeMutableBufferPointer { ptr in
            var offset = 0
            for y in 0..<height {
                let lVal = l[y]
                for x in 0..<width {
                    ptr[offset + x] = lVal
                }
                offset += width
            }
        }
    }

    @inline(__always)
    private static func predictPlanar(block: inout [Int16], width: Int, height: Int, top: [Int16]?, left: [Int16]?, topLeft: Int16) {
        guard let t = top, t.count >= width, let l = left, l.count >= height else {
            predictDC(block: &block, width: width, height: height, top: top, left: left)
            return
        }
        
        block.withUnsafeMutableBufferPointer { ptr in
            var offset = 0
            for y in 0..<height {
                let lVal = Int32(l[y])
                for x in 0..<width {
                    let tVal = Int32(t[x])
                    // TrueMotion prediction: p = top + left - topLeft
                    let p = tVal + lVal - Int32(topLeft)
                    // Clamp to 10-bit or suitable range? We use Int16 for DWT residuals.
                    // For prediction, just cast back since we will calculate residual.
                    ptr[offset + x] = Int16(max(-32768, min(32767, p)))
                }
                offset += width
            }
        }
    }
}
