import Foundation
import vevc

func main() {
    let args = CommandLine.arguments
    guard 3 <= args.count, args[1] == "sigma" || args[1] == "offsets" else {
        print("""
        usage: vevc-training <sigma|offsets> <dumpfile>

        sigma:   evaluate sigma-conditioned rANS contexts against the current
                 6-context scheme, using a coefficient dump produced by running
                 the encoder with VEVC_DUMP_COEFFS=<dumpfile>.
        offsets: fit per-subband Laplacian scales from quantized histograms and
                 report MSE-optimal dequantization centroid offsets.
        """)
        exit(1)
    }
    do {
        let report = args[1] == "sigma"
            ? try runSigmaMeasurement(dumpPath: args[2])
            : try runDequantOffsetEstimation(dumpPath: args[2])
        print(report)
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        exit(1)
    }
}

main()
