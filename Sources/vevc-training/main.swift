import Foundation
import vevc

func main() {
    let args = CommandLine.arguments
    let usage = """
    usage: vevc-training sigma <dumpfile>
           vevc-training offsets <dumpfile>
           vevc-training train-tables <train.vsd> <test.vsd>

    sigma:        evaluate sigma-conditioned rANS contexts against the current
                  6-context scheme, using a coefficient dump produced by running
                  the encoder with VEVC_DUMP_COEFFS=<dumpfile>.
    offsets:      fit per-subband Laplacian scales from quantized histograms and
                  report MSE-optimal dequantization centroid offsets.
    train-tables: retrain the built-in static rANS tables on one dump, evaluate
                  generalization on another, and emit drop-in Swift constants.
    """
    do {
        switch args.count >= 2 ? args[1] : "" {
        case "sigma" where args.count >= 3:
            print(try runSigmaMeasurement(dumpPath: args[2]))
        case "offsets" where args.count >= 3:
            print(try runDequantOffsetEstimation(dumpPath: args[2]))
        case "train-tables" where args.count >= 4:
            print(try runTableTraining(trainPath: args[2], testPath: args[3]))
        default:
            print(usage)
            exit(1)
        }
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        exit(1)
    }
}

main()
