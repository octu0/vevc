import Foundation
import vevc

func main() {
    let args = CommandLine.arguments
    guard 3 <= args.count, args[1] == "sigma" else {
        print("""
        usage: vevc-training sigma <dumpfile>

        sigma: evaluate sigma-conditioned rANS contexts against the current
               6-context scheme, using a coefficient dump produced by running
               the encoder with VEVC_DUMP_COEFFS=<dumpfile>.
        """)
        exit(1)
    }
    do {
        let report = try runSigmaMeasurement(dumpPath: args[2])
        print(report)
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        exit(1)
    }
}

main()
