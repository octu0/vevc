import Foundation
import vevc

func printUsage() {
    let usage = """
    usage: vevc-training dump <input.y4m> <output.vsd> [-profile 1|2] [-b bitrate] [-q qstep]
           vevc-training train-tables <train.vsd[,train2.vsd,...]> <test.vsd>
           vevc-training train-tables-pf <train.vsd[,train2.vsd,...]> <test.vsd>

    dump:            encode an input Y4M video to produce a coefficient dump file (.vsd).
    train-tables:    retrain the built-in static rANS tables on training dump(s), evaluate
                     generalization on a test dump, and emit drop-in Swift constants.
    train-tables-pf: retrain static rANS tables for parent-free context assignment (profile 0x02).
    """
    print(usage)
}

func main() async {
    let args = CommandLine.arguments
    if args.count < 2 {
        printUsage()
        exit(1)
    }

    let command = args[1]
    do {
        switch command {
        case "dump":
            if args.count < 4 {
                printUsage()
                exit(1)
            }
            let inputPath = args[2]
            let outputPath = args[3]
            var profile: UInt8 = 0x01
            var bitrate = 500_000
            var qstep: Int? = nil

            var idx = 4
            while idx < args.count {
                let opt = args[idx]
                switch opt {
                case "-profile", "--profile":
                    if idx + 1 < args.count {
                        if let p = UInt8(args[idx + 1]) {
                            profile = p
                        }
                        idx += 2
                    } else {
                        idx += 1
                    }
                case "-b", "--bitrate":
                    if idx + 1 < args.count {
                        if let b = Int(args[idx + 1]) {
                            bitrate = b
                        }
                        idx += 2
                    } else {
                        idx += 1
                    }
                case "-q", "--qstep":
                    if idx + 1 < args.count {
                        if let q = Int(args[idx + 1]) {
                            qstep = q
                        }
                        idx += 2
                    } else {
                        idx += 1
                    }
                default:
                    idx += 1
                }
            }

            let dumpEncoder = TrainingDumpEncoder()
            try await dumpEncoder.dump(
                inputPath: inputPath,
                outputPath: outputPath,
                profile: profile,
                maxbitrate: bitrate,
                qstep: qstep
            )

        case "train-tables":
            if args.count < 4 {
                printUsage()
                exit(1)
            }
            let trainPath = args[2]
            let testPath = args[3]
            let result = try runTableTraining(trainPath: trainPath, testPath: testPath, parentFree: false)
            print(result)

        case "train-tables-pf":
            if args.count < 4 {
                printUsage()
                exit(1)
            }
            let trainPath = args[2]
            let testPath = args[3]
            let result = try runTableTraining(trainPath: trainPath, testPath: testPath, parentFree: true)
            print(result)

        default:
            printUsage()
            exit(1)
        }
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        exit(1)
    }
}

await main()
