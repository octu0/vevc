import Foundation
import vevc

func main() {
    let args = CommandLine.arguments
    guard args.count == 2 else {
        print("usage: vevc-inspect <input.vevc>")
        print("Prints per-frame bitstream statistics (type, qsteps, layer sizes, skip composition) as CSV.")
        exit(1)
    }
    do {
        let data = [UInt8](try Data(contentsOf: URL(fileURLWithPath: args[1])))
        print(try inspectBitstreamCSV(data: data), terminator: "")
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        exit(1)
    }
}

main()
