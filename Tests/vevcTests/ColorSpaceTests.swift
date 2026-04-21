import Testing
@testable import vevc

struct ColorSpaceTests {
    @Test func testColorSpaceRoundTrip() {
        let width = 64
        let height = 64
        var rawData = [UInt8](repeating: 0, count: width * height * 4)
        
        // Fill with a gradient that might trigger overflow
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                rawData[offset + 0] = UInt8(x * 4) // R
                rawData[offset + 1] = UInt8(y * 4) // G
                rawData[offset + 2] = UInt8((x + y) * 2) // B
                rawData[offset + 3] = 255
            }
        }
        
        let ycbcr = rgbaToYCbCr(data: rawData, width: width, height: height)
        
        // Convert ratio444 to ratio420 as typically done in Encoder
        let pd420 = toPlaneData420(image: ycbcr, pool: BlockViewPool()).0
        let restoredYCbCr = pd420.toYCbCr()
        
        let restoredRGBA = ycbcrToRGBA(img: restoredYCbCr)
        
        // Check if there's any severe corruption (difference > 20 is suspicious since subsampling causes some loss, but shouldn't be huge)
        var maxDiff = 0
        var diffCount = 0
        for i in 0..<rawData.count {
            if i % 4 == 3 { continue } // skip Alpha
            let diff = abs(Int(rawData[i]) - Int(restoredRGBA[i]))
            if maxDiff < diff { maxDiff = diff }
            if 50 < diff {
                diffCount += 1
            }
        }
        
        print("Max Color Diff: \(maxDiff), Pixels with huge diff (>50): \(diffCount)")
        #expect(diffCount == 0, "Color corruption detected!")
    }
}
