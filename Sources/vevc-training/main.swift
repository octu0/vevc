import Foundation
import vevc

func printUsage() {
    let usage = """
    使用方法:
      vevc-training dump <in.y4m> <out.vsd> [-profile 1|2] [-b bitrate] [-q qstep] [-frames N]
      vevc-training train -model rans|rans-pf -train <a.vsd,b.vsd,...> -test <t.vsd> [-o <out.swift>]
      vevc-training train -model ctx -train <a.vsd,...> -test <t.vsd> -o <out.swift> [-seed 1] [-epochs N]
      vevc-training verify-walker <in.vsd>

    説明:
      dump:            Y4M 入力動画をエンコードし係数ダンプファイル (.vsd) を生成
      train:           エントロピーモデルの再学習 (rans, rans-pf, または ctx)
      verify-walker:   現行重みを用いた .vsd walker の NLL 再構成検算
    """
    print(usage)
}

func runDump(
    inputPath: String,
    outputPath: String,
    profile: UInt8,
    bitrate: Int,
    qstep: Int?,
    maxFrames: Int?
) async throws {
    guard let dumper = CoeffDumpWriter(path: outputPath) else {
        throw TrainingError.invalidInput("ダンプ出力ファイルのオープンに失敗しました: \(outputPath)")
    }
    defer {
        dumper.close()
    }

    guard let fh = FileHandle(forReadingAtPath: inputPath) else {
        throw TrainingError.invalidInput("入力 Y4M ファイルのオープンに失敗しました: \(inputPath)")
    }
    defer {
        try? fh.close()
    }

    let y4m = try Y4MReader(fileHandle: fh)
    var fps = 30
    if y4m.fpsHeader.starts(with: "F") {
        let parts = y4m.fpsHeader.dropFirst().split(separator: ":")
        if parts.count == 2, let num = Int(parts[0]), let den = Int(parts[1]), 0 < den {
            let parsed = num / den
            if parsed != 0 {
                fps = parsed
            }
        }
    }
    let encoder = VEVCEncoder(width: y4m.width, height: y4m.height, profile: profile)
    encoder.qstep = qstep
    encoder.maxbitrate = bitrate
    encoder.framerate = fps
    encoder.dumpWriter = dumper
    var frameIdx = 0
    while let frame = try y4m.readFrame() {
        if let maxF = maxFrames, maxF <= frameIdx {
            break
        }
        let encodedPacket = try await encoder.encode(image: frame)
        if encodedPacket.count < 0 {
            print("異常なパケット")
        }
        frameIdx += 1
    }
}

func loadWeightsBlob(from path: String) throws -> [Int32] {
    guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
        throw EmitterError.mismatch("\(path) を読み込めませんでした")
    }
    guard let startRange = content.range(of: "static let blob: [Int32] = [") else {
        throw EmitterError.mismatch("\(path) 内で blob の開始位置が見つかりませんでした")
    }
    let rest = content[startRange.upperBound...]
    guard let endRange = rest.range(of: "]") else {
        throw EmitterError.mismatch("\(path) 内で blob の終了位置が見つかりませんでした")
    }
    let numbersStr = rest[..<endRange.lowerBound]
    var blob: [Int32] = []
    blob.reserveCapacity(26352)
    for part in numbersStr.split(whereSeparator: { $0 == "," || $0.isWhitespace }) {
        if let val = Int32(part) {
            blob.append(val)
        }
    }
    if blob.count != 26352 {
        throw EmitterError.invalidBlobSize("パースされた要素数が不正です: \(blob.count) (期待値: 26352)")
    }
    return blob
}

func loadShippedWeightsBlob() throws -> [Int32] {
    return try loadWeightsBlob(from: "Sources/vevc/rANSContextWeights.swift")
}

func runVerifyWalker(vsdPath: String) throws {
    print("=== .vsd walker 検算 ===")
    print("入力: \(vsdPath)")
    let shippedBlob = try loadShippedWeightsBlob()
    let weights = RANSContextWeightsContainer(blob: shippedBlob)
    let reader = try VSDReader(path: vsdPath)
    let walker = ContextRANSWalker()

    var totalSamples = 0
    var totalNLLBits: Double = 0.0
    var totalRANSBytes = 0
    var totalLayer0Bytes = 0
    var frameCount = 0

    while let frame = try reader.nextFrame() {
        totalLayer0Bytes += frame.layer0Bytes
        frameCount += 1

        var frameSamples: [RANSContextSample] = []
        walker.walkL0Y(frame: frame) { s in
            frameSamples.append(s)
        }
        totalSamples += frameSamples.count

        let nllBits = walker.evaluateNLLBits(samples: frameSamples, weights: weights)
        totalNLLBits += nllBits

        let ransBytes = walker.simulateRANSBytes(samples: frameSamples, weights: weights)
        totalRANSBytes += ransBytes.count
    }

    let nllBytes = totalNLLBits / 8.0
    let pureRANSBytes = totalRANSBytes - (frameCount * 8)
    print("処理フレーム数: \(frameCount)")
    print("合計サンプル数: \(totalSamples)")
    print("理論符号量 (NLL): \(String(format: "%.1f", nllBytes)) bytes (\(String(format: "%.2f", nllBytes / 1024.0)) KB)")
    print("rANS 実シミュレート符号量: \(totalRANSBytes) bytes (ヘッダ除外: \(pureRANSBytes) bytes)")
    print("実エンコード layer0Bytes: \(totalLayer0Bytes) bytes (\(String(format: "%.2f", Double(totalLayer0Bytes) / 1024.0)) KB)")

    if 0 < pureRANSBytes {
        let diffBytes = abs(nllBytes - Double(pureRANSBytes))
        let diffPercent = (diffBytes / Double(pureRANSBytes)) * 100.0
        print("NLL と 純rANSストリーム差分: \(String(format: "%.2f", diffBytes)) bytes (\(String(format: "%.3f", diffPercent))%)")
        let totalDiff = abs(nllBytes - Double(totalRANSBytes))
        let totalDiffPercent = (totalDiff / Double(totalRANSBytes)) * 100.0
        print("NLL と 総rANS(ヘッダ込)差分: \(String(format: "%.2f", totalDiff)) bytes (\(String(format: "%.3f", totalDiffPercent))%)")
        if diffPercent <= 1.0 {
            print("=> 検算判定: PASS (純粋符号量において誤差 \(String(format: "%.3f", diffPercent))% <= 1.0%)")
        } else {
            print("=> 検算判定: FAIL")
        }
    }
}

func runCompareWeights(
    vsdPath: String,
    oldWeightsPath: String,
    newWeightsPath: String
) throws {
    print("=== 新旧重み 対照計測 (VSD walker) ===")
    print("VSD 入力   : \(vsdPath)")
    print("旧重み     : \(oldWeightsPath)")
    print("新重み     : \(newWeightsPath)")

    let oldBlob = try loadWeightsBlob(from: oldWeightsPath)
    let newBlob = try loadWeightsBlob(from: newWeightsPath)
    let oldWeights = RANSContextWeightsContainer(blob: oldBlob)
    let newWeights = RANSContextWeightsContainer(blob: newBlob)

    let reader = try VSDReader(path: vsdPath)
    let walker = ContextRANSWalker()

    var frameCount = 0
    var totalSamples = 0
    var totalLayer0Bytes = 0

    var oldTotalNLLBits: Double = 0.0
    var oldTotalRANSBytes = 0
    var newTotalNLLBits: Double = 0.0
    var newTotalRANSBytes = 0

    // 連続 Logistic NLL の集計 (仮説1検証用: Float連続限界 vs Q12 buildCDF)
    var newContinuousNLLBits: Double = 0.0

    while let frame = try reader.nextFrame() {
        totalLayer0Bytes += frame.layer0Bytes
        frameCount += 1

        var frameSamples: [RANSContextSample] = []
        walker.walkL0Y(frame: frame) { s in
            frameSamples.append(s)
        }
        totalSamples += frameSamples.count

        // 旧重み評価
        let oldNll = walker.evaluateNLLBits(samples: frameSamples, weights: oldWeights)
        oldTotalNLLBits += oldNll
        let oldRans = walker.simulateRANSBytes(samples: frameSamples, weights: oldWeights)
        oldTotalRANSBytes += oldRans.count

        // 新重み評価
        let newNll = walker.evaluateNLLBits(samples: frameSamples, weights: newWeights)
        newTotalNLLBits += newNll
        let newRans = walker.simulateRANSBytes(samples: frameSamples, weights: newWeights)
        newTotalRANSBytes += newRans.count

        // 新重みでの連続 Logistic NLL (Float 相当)
        for s in frameSamples {
            let (muQ12, invScaleQ12) = newWeights.predict(pos: s.pos, feat: s.feat)
            let mu = Double(muQ12) / 4096.0
            var invScale = Double(invScaleQ12) / 4096.0
            if invScale < 0.0001 {
                invScale = 0.0001
            }
            let val = Double(s.sym) - 64.0
            let z = (val - mu) * invScale
            let absZ = abs(z)
            // logistic log PDF in nats = log(invScale) - |z| - 2*log1p(exp(-|z|))
            let nllNats = -log(invScale) + absZ + 2.0 * log1p(exp(-absZ))
            let nllBits = nllNats / 0.6931471805599453 // ln(2)
            newContinuousNLLBits += nllBits
            if s.isEscape {
                newContinuousNLLBits += 16.0
            }
        }
    }

    let oldPureBytes = oldTotalRANSBytes - (frameCount * 8)
    let newPureBytes = newTotalRANSBytes - (frameCount * 8)
    let oldNllBytes = oldTotalNLLBits / 8.0
    let newNllBytes = newTotalNLLBits / 8.0
    let newContNllBytes = newContinuousNLLBits / 8.0

    print("処理フレーム数       : \(frameCount)")
    print("合計サンプル数       : \(totalSamples)")
    print("L0Y 総バイト         : \(totalLayer0Bytes) bytes (\(String(format: "%.2f", Double(totalLayer0Bytes) / 1024.0)) KB)")
    print("--- 旧重み ---")
    print("  理論 NLL           : \(String(format: "%.1f", oldNllBytes)) bytes")
    print("  純 rANS ストリーム : \(oldPureBytes) bytes")
    let oldErr = 0 < oldPureBytes ? (abs(oldNllBytes - Double(oldPureBytes)) / Double(oldPureBytes)) * 100.0 : 0.0
    print("  Walker 検算誤差    : \(String(format: "%.3f", oldErr))% (判定: \(oldErr <= 1.0 ? "PASS" : "FAIL"))")
    print("--- 新重み ---")
    print("  理論 NLL           : \(String(format: "%.1f", newNllBytes)) bytes")
    print("  純 rANS ストリーム : \(newPureBytes) bytes")
    let newErr = 0 < newPureBytes ? (abs(newNllBytes - Double(newPureBytes)) / Double(newPureBytes)) * 100.0 : 0.0
    print("  Walker 検算誤差    : \(String(format: "%.3f", newErr))% (判定: \(newErr <= 1.0 ? "PASS" : "FAIL"))")
    print("--- 削減率 (旧 -> 新) ---")
    let pureDelta = 0 < oldPureBytes ? ((Double(newPureBytes) - Double(oldPureBytes)) / Double(oldPureBytes)) * 100.0 : 0.0
    let nllDelta = 0 < oldNllBytes ? ((newNllBytes - oldNllBytes) / oldNllBytes) * 100.0 : 0.0
    print("  純 rANS ストリーム削減率 : \(String(format: "%+.2f", pureDelta))%")
    print("  理論 NLL 削減率          : \(String(format: "%+.2f", nllDelta))%")
    print("--- 仮説1検証 (Float連続NLL vs Q12実buildCDF) ---")
    print("  Float 連続 NLL (理論)   : \(String(format: "%.1f", newContNllBytes)) bytes")
    print("  Q12 buildCDF (実測)     : \(String(format: "%.1f", newNllBytes)) bytes")
    let quantLoss = 0 < newContNllBytes ? ((newNllBytes - newContNllBytes) / newContNllBytes) * 100.0 : 0.0
    print("  Q12 量子化・離散化ロス  : \(String(format: "%+.2f", quantLoss))%")
}

func runContextTraining(
    trainPaths: [String],
    testPaths: [String],
    outputPath: String,
    fixedSeed: Int?,
    numEpochs: Int
) throws {
    print("=== rANSContext 再学習開始 ===")
    print("学習データ: \(trainPaths.joined(separator: ", "))")
    print("検証データ: \(testPaths.joined(separator: ", "))")
    print("エポック数: \(numEpochs)")

    let shippedBlob = try loadShippedWeightsBlob()
    let shippedWeights = RANSContextWeightsContainer(blob: shippedBlob)
    let walker = ContextRANSWalker()

    // 1. サンプル収集
    var trainSamplesByPos = [[RANSContextSample]](repeating: [], count: 16)
    var testSamplesByPos = [[RANSContextSample]](repeating: [], count: 16)
    var trainTotalLayer0 = 0
    var testTotalLayer0 = 0

    print("学習データを走査中...")
    for path in trainPaths {
        let reader = try VSDReader(path: path)
        while let frame = try reader.nextFrame() {
            trainTotalLayer0 += frame.layer0Bytes
            walker.walkL0Y(frame: frame) { s in
                trainSamplesByPos[s.pos].append(s)
            }
        }
    }

    print("検証データを走査中...")
    for path in testPaths {
        let reader = try VSDReader(path: path)
        while let frame = try reader.nextFrame() {
            testTotalLayer0 += frame.layer0Bytes
            walker.walkL0Y(frame: frame) { s in
                testSamplesByPos[s.pos].append(s)
            }
        }
    }

    // 現行重みでのベースライン NLL 計測
    var baselineTestBits: Double = 0.0
    for pos in 4..<16 {
        let bits = walker.evaluateNLLBits(samples: testSamplesByPos[pos], weights: shippedWeights)
        baselineTestBits += bits
    }
    let baselineTestBytes = baselineTestBits / 8.0
    print("現行重み (shipped) 検証セット NLL: \(String(format: "%.1f", baselineTestBytes)) bytes")

    // 2. シード選抜 (1..5 または指定シード)
    var seedsToTest: [Int] = [1, 2, 3, 4, 5]
    if let s = fixedSeed {
        seedsToTest = [s]
    }

    var bestSeed = seedsToTest[0]
    var bestModels: QuantizedContextModels? = nil
    var bestValBytes = Double.infinity

    for seed in seedsToTest {
        print("--- シード \(seed) の学習実行中 ---")
        var seedRng = DeterministicRNG(seed: UInt64(seed))

        var q_w2 = [[Int32]](repeating: [], count: 16)
        var q_b1 = [[Int32]](repeating: [], count: 16)
        var q_w1 = [[Int32]](repeating: [], count: 16)
        var q_b2 = [Int32](repeating: 0, count: 16)
        var q_invScale = [Int32](repeating: 0, count: 16)
        var dims = [Int32](repeating: 0, count: 16)

        for pos in 4..<16 {
            let inDim = 57 + pos
            dims[pos] = Int32(inDim)
            let trainer = PositionModelTrainer(pos: pos, inDim: inDim, initialWeights: shippedWeights)
            let samples = trainSamplesByPos[pos]
            let sampleCount = samples.count

            if 0 < sampleCount {
                // サンプルを Float 配列に展開
                var feats = [[Float]](repeating: [], count: sampleCount)
                var vals = [Int16](repeating: 0, count: sampleCount)
                for i in 0..<sampleCount {
                    let s = samples[i]
                    var f = [Float](repeating: 0.0, count: inDim)
                    for d in 0..<inDim {
                        f[d] = Float(s.feat[d]) / 4096.0
                    }
                    feats[i] = f
                    vals[i] = s.val
                }

                // エポックループ
                let batchSize = 256
                for _ in 0..<numEpochs {
                    // Fisher-Yates シャッフル
                    var indices = Array(0..<sampleCount)
                    var i = sampleCount - 1
                    while 0 < i {
                        let j = Int(seedRng.nextUInt64() % UInt64(i + 1))
                        let tmp = indices[i]
                        indices[i] = indices[j]
                        indices[j] = tmp
                        i -= 1
                    }

                    var bStart = 0
                    while bStart < sampleCount {
                        let bEnd = min(bStart + batchSize, sampleCount)
                        let bCount = bEnd - bStart
                        var bFeats = [[Float]](repeating: [], count: bCount)
                        var bVals = [Int16](repeating: 0, count: bCount)
                        for k in 0..<bCount {
                            let idx = indices[bStart + k]
                            bFeats[k] = feats[idx]
                            bVals[k] = vals[idx]
                        }
                        let loss = trainer.trainBatch(feats: bFeats, vals: bVals, learningRate: 0.0005)
                        if loss < 0.0 {
                            print("異常な損失")
                        }
                        bStart += batchSize
                    }
                }
            }

            let q = trainer.quantizeQ12()
            q_w2[pos] = q.w2
            q_b1[pos] = q.b1
            q_w1[pos] = q.w1
            q_b2[pos] = q.b2
            q_invScale[pos] = q.invScale
        }

        let candModels = QuantizedContextModels(
            invScalesQ: q_invScale,
            b2Q: q_b2,
            dims: dims,
            w2Q: q_w2,
            b1Q: q_b1,
            w1FlatQ: q_w1
        )
        let candWeights = RANSContextWeightsContainer(blob: candModels.toBlob())

        // 検証セットで実 buildCDF によるビット数を再計測
        var candValBits: Double = 0.0
        for pos in 4..<16 {
            let bits = walker.evaluateNLLBits(samples: testSamplesByPos[pos], weights: candWeights)
            candValBits += bits
        }
        let candValBytes = candValBits / 8.0
        let delta = ((candValBytes - baselineTestBytes) / baselineTestBytes) * 100.0
        print("シード \(seed): 検証セット NLL = \(String(format: "%.1f", candValBytes)) bytes (差分: \(String(format: "%+.2f", delta))%)")

        if candValBytes < bestValBytes {
            bestValBytes = candValBytes
            bestSeed = seed
            bestModels = candModels
        }
    }

    guard let finalModels = bestModels else {
        throw EmitterError.mismatch("有効な学習モデルが生成されませんでした")
    }

    print("=== 最良シード: \(bestSeed) (検証 NLL: \(String(format: "%.1f", bestValBytes)) bytes) ===")
    let finalDelta = ((bestValBytes - baselineTestBytes) / baselineTestBytes) * 100.0
    print("現行比改善率: \(String(format: "%+.2f", finalDelta))%")

    // 3. 完全な Swift コードの生成と round-trip 検証
    let emitter = ContextRANSWeightsEmitter()
    try emitter.verifyRoundTrip(models: finalModels)
    print("パース round-trip 検証: 成功 (100% bit 一致)")

    let source = emitter.emitSwiftSource(models: finalModels)
    let outUrl = URL(fileURLWithPath: outputPath)
    try source.write(to: outUrl, atomically: true, encoding: .utf8)
    print("完全な Swift ファイルを出力しました: \(outputPath)")
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
            var maxFrames: Int? = nil

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
                case "-frames", "--frames":
                    if idx + 1 < args.count {
                        if let f = Int(args[idx + 1]) {
                            maxFrames = f
                        }
                        idx += 2
                    } else {
                        idx += 1
                    }
                default:
                    idx += 1
                }
            }

            try await runDump(
                inputPath: inputPath,
                outputPath: outputPath,
                profile: profile,
                bitrate: bitrate,
                qstep: qstep,
                maxFrames: maxFrames
            )

        case "verify-walker":
            if args.count < 3 {
                printUsage()
                exit(1)
            }
            let vsdPath = args[2]
            try runVerifyWalker(vsdPath: vsdPath)

        case "compare-weights":
            if args.count < 3 {
                print("使用方法: vevc-training compare-weights <in.vsd> [-old <path>] [-new <path>]")
                exit(1)
            }
            let vsdPath = args[2]
            var oldPath = ".tmp/training_w2/backup/rANSContextWeights.swift"
            var newPath = ".tmp/training_w2/rANSContextWeights.swift"
            var idx = 3
            while idx < args.count {
                switch args[idx] {
                case "-old":
                    if idx + 1 < args.count {
                        oldPath = args[idx + 1]
                        idx += 2
                    } else {
                        idx += 1
                    }
                case "-new":
                    if idx + 1 < args.count {
                        newPath = args[idx + 1]
                        idx += 2
                    } else {
                        idx += 1
                    }
                default:
                    idx += 1
                }
            }
            try runCompareWeights(vsdPath: vsdPath, oldWeightsPath: oldPath, newWeightsPath: newPath)

        case "train":
            var modelType: String? = nil
            var trainPathsStr: String? = nil
            var testPathStr: String? = nil
            var outputPath: String? = nil
            var seed: Int? = nil
            var epochs: Int = 3

            var idx = 2
            while idx < args.count {
                let opt = args[idx]
                switch opt {
                case "-model", "--model":
                    if idx + 1 < args.count {
                        modelType = args[idx + 1]
                        idx += 2
                    } else {
                        idx += 1
                    }
                case "-train", "--train":
                    if idx + 1 < args.count {
                        trainPathsStr = args[idx + 1]
                        idx += 2
                    } else {
                        idx += 1
                    }
                case "-test", "--test":
                    if idx + 1 < args.count {
                        testPathStr = args[idx + 1]
                        idx += 2
                    } else {
                        idx += 1
                    }
                case "-o", "--output":
                    if idx + 1 < args.count {
                        outputPath = args[idx + 1]
                        idx += 2
                    } else {
                        idx += 1
                    }
                case "-seed", "--seed":
                    if idx + 1 < args.count {
                        seed = Int(args[idx + 1])
                        idx += 2
                    } else {
                        idx += 1
                    }
                case "-epochs", "--epochs":
                    if idx + 1 < args.count {
                        if let ep = Int(args[idx + 1]) {
                            epochs = ep
                        }
                        idx += 2
                    } else {
                        idx += 1
                    }
                default:
                    idx += 1
                }
            }

            guard let mType = modelType else {
                printUsage()
                exit(1)
            }

            switch mType {
            case "rans":
                guard let tTrain = trainPathsStr, let tTest = testPathStr else {
                    printUsage()
                    exit(1)
                }
                let result = try runTableTraining(trainPath: tTrain, testPath: tTest, parentFree: false)
                print(result)
                if let out = outputPath {
                    try result.write(to: URL(fileURLWithPath: out), atomically: true, encoding: .utf8)
                }

            case "rans-pf":
                guard let tTrain = trainPathsStr, let tTest = testPathStr else {
                    printUsage()
                    exit(1)
                }
                let result = try runTableTraining(trainPath: tTrain, testPath: tTest, parentFree: true)
                print(result)
                if let out = outputPath {
                    try result.write(to: URL(fileURLWithPath: out), atomically: true, encoding: .utf8)
                }

            case "ctx":
                guard let tTrain = trainPathsStr, let tTest = testPathStr, let out = outputPath else {
                    printUsage()
                    exit(1)
                }
                let trainPaths = tTrain.split(separator: ",").map { String($0) }
                let testPaths = tTest.split(separator: ",").map { String($0) }
                try runContextTraining(
                    trainPaths: trainPaths,
                    testPaths: testPaths,
                    outputPath: out,
                    fixedSeed: seed,
                    numEpochs: epochs
                )

            default:
                print("未知のモデル種別です: \(mType)")
                exit(1)
            }

        default:
            printUsage()
            exit(1)
        }
    } catch {
        FileHandle.standardError.write(Data("エラー: \(error)\n".utf8))
        exit(1)
    }
}

await main()
