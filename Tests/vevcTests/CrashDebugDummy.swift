import XCTest
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
@testable import vevc

final class CrashDebugTests: XCTestCase {
    func testLSCPDebug() {
        print("\n\n--- [DEBUG] TEST START ---")
        fflush(stdout)
        
        var encoder = VevcEncoder()
        let size = 8
        var blockData = [Int16](repeating: 0, count: size * size)
        blockData[0] = 5
        blockData[1] = -3
        blockData[8] = 2
        var block = Block2D(width: size, height: size)
        block.withView { view in
            for y in 0..<size {
                let ptr = view.rowPointer(y: y)
                for x in 0..<size {
                    ptr[x] = blockData[y * size + x]
                }
            }
        }
        
        print("--- [DEBUG] 1. Encode Start ---")
        fflush(stdout)
        block.withView { view in
            blockEncode(encoder: &encoder, block: view, size: size)
        }
        
        print("--- [DEBUG] 2. Encode Flush ---")
        fflush(stdout)
        encoder.flush()
        
        print("--- [DEBUG] 3. Encode GetData ---")
        fflush(stdout)
        let encodedData = encoder.getData()
        print("--- [DEBUG] Encode Data generated: \(encodedData.count) bytes ---")
        fflush(stdout)
        
        print("--- [DEBUG] 4. Decoder Init ---")
        fflush(stdout)
        do {
            var decoder = try VevcDecoder(data: encodedData)
            print("--- [DEBUG] Decoder initialized with coeffs count: \(decoder.coeffs.count) ---")
            fflush(stdout)
            
            var outBlock = Block2D(width: size, height: size)
            try outBlock.withView { view in
                print("--- [DEBUG] 5. Decode Block Start ---")
                fflush(stdout)
                try blockDecode(decoder: &decoder, block: &view, size: size)
                print("--- [DEBUG] 6. Decode Block End ---")
                fflush(stdout)
            }
            
            print("--- [DEBUG] TEST DONE ---")
            fflush(stdout)
        } catch {
            print("--- [DEBUG] Catch: \(error) ---")
            fflush(stdout)
            XCTFail("Decode Failed: \(error)")
        }
    }
}
