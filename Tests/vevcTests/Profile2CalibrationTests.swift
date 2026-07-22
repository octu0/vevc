import XCTest
@testable import vevc

final class Profile2CalibrationTests: XCTestCase {
    func computePSNR(ref: [Int16], test: [Int16]) -> Double {
        var mse: Double = 0
        for i in 0..<ref.count {
            let diff = Double(ref[i]) - Double(test[i])
            mse += diff * diff
        }
        mse /= Double(ref.count)
        if mse == 0 { return 99.0 }
        return 10.0 * log10((255.0 * 255.0) / mse)
    }

    func testCalibrateK() async throws {
        let width = 320
        let height = 240
        let pool = BlockViewPool()
        
        var planeY = [Int16](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                planeY[y * width + x] = Int16(Int.random(in: 0..<256))
            }
        }
        
        let pd = PlaneData420(width: width, height: height, y: planeY, cb: [Int16](repeating: 128, count: (width+1)/2 * (height+1)/2), cr: [Int16](repeating: 128, count: (width+1)/2 * (height+1)/2))
        
        for qstep in [16, 64, 128] {
            let qt = QuantizationTable(baseStep: qstep, isChroma: false, layerIndex: 0)
            let codec1 = Layer0CodecFactory.create(profile: 1)
            let (bytes1, rec1, _, _, _, rel1) = try await codec1.encode(pd: pd, pool: pool, sads: nil, occlusionScores: nil, layer: 0, qtY: qt, qtC: qt, zeroThreshold: 0)
            let psnr1 = computePSNR(ref: planeY, test: rec1.y)
            rel1()
            
            let dx = pd.width
            let dy = pd.height
            let step = max(1, qstep / 16)
            let enc = encodeL0PlaneDCT(plane: pd.y, width: dx, height: dy, stride: dx, step: step)
            let dec = try decodeL0PlaneDCT(bytes: enc.bytes, width: dx, height: dy, step: step)
            let psnr2 = computePSNR(ref: planeY, test: dec)
            
            let diff = abs(psnr2 - psnr1)
            print("qstep \(qstep): Profile 1 PSNR = \(String(format: "%.2f", psnr1)) dB, Profile 2 PSNR = \(String(format: "%.2f", psnr2)) dB, L0 Profile 1 bytes = \(bytes1.count), L0 Profile 2 bytes = \(enc.bytes.count), diff = \(String(format: "%.2f", diff)) dB")
            XCTAssertLessThanOrEqual(diff, 5.0, "Profile 2 DCT PSNR should be within calibrated range of Profile 1 DWT PSNR")
        }
    }
}
