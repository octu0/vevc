import XCTest
@testable import vevc

final class Profile2DichotomyTests: XCTestCase {
    func testDirectCall() async throws {
        let y4mPath = "/Users/octu0/workspace/vevc/ToS_1frame.y4m"
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: y4mPath)) else {
            XCTFail("Could not read y4m file")
            return
        }
        
        // Parse Y4M manually just to get the first frame
        // Header is 59 bytes, FRAME\n is 6 bytes. Frame starts at 65.
        guard let handle = FileHandle(forReadingAtPath: y4mPath) else {
            XCTFail("Could not open file handle")
            return
        }
        let reader = try Y4MReader(fileHandle: handle)
        guard let image = try reader.readFrame() else {
            XCTFail("Could not read frame")
            return
        }
        
        let width = image.width
        let height = image.height
        
        let pool = BlockViewPool()
        let (plane, _) = toPlaneData420(image: image, pool: pool)
        
        let qstep = 64
        let qtY = QuantizationTable(baseStep: qstep, isChroma: false, layerIndex: 0, profile: 0x02)
        let qtC = QuantizationTable(baseStep: qstep, isChroma: true, layerIndex: 0, profile: 0x02)
        
        let payload = try await encodeIntraTiles(pd: plane, pool: pool, qtY: qtY, qtC: qtC)
        let fHeader = VEVCFileHeader(width: width, height: height, framerate: 60, profile: 0x02)
        
        let decoded16 = try await decodeIntraTiles(from: payload, pool: pool, header: fHeader, maxLayer: 2)
        let decodedPd = PlaneData420(img16: decoded16)
        let decodedYCbCr = decodedPd.toYCbCr()
        
        let psnrY = calculatePlanePSNR(original: image.yPlane, decoded: decodedYCbCr.yPlane)
        let psnrCb = calculatePlanePSNR(original: image.cbPlane, decoded: decodedYCbCr.cbPlane)
        let psnrCr = calculatePlanePSNR(original: image.crPlane, decoded: decodedYCbCr.crPlane)
        
        let avg = (4.0 * psnrY + psnrCb + psnrCr) / 6.0
    }
    
    private func calculatePlanePSNR(original: [UInt8], decoded: [UInt8]) -> Double {
        var mse: Double = 0
        for i in 0..<original.count {
            let diff = Double(original[i]) - Double(decoded[i])
            mse += diff * diff
        }
        mse /= Double(original.count)
        if mse == 0 { return 100.0 }
        return 10.0 * log10((255.0 * 255.0) / mse)
    }
}
