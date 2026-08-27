import Foundation

/// DPCM 4x4 ブロックの特徴量ダンプレコード (固定長 164 バイト)
public struct DPCMBlockDumpRecord: Sendable {
    public var frameIndex: UInt32
    public var plane: UInt8          // 0: Y, 1: Cb, 2: Cr
    public var isAllZero: UInt8      // 0: Non-zero, 1: All-zero
    public var blockX: UInt16
    public var blockY: UInt16
    public var qLow: Int16           // qt.qLow.step
    public var lastVal: Int16
    public var padding0: UInt16
    
    public var quantizedValues: (Int16, Int16, Int16, Int16,
                                 Int16, Int16, Int16, Int16,
                                 Int16, Int16, Int16, Int16,
                                 Int16, Int16, Int16, Int16)
    
    public var dpcmErrors: (Int16, Int16, Int16, Int16,
                            Int16, Int16, Int16, Int16,
                            Int16, Int16, Int16, Int16,
                            Int16, Int16, Int16, Int16)
    
    public var topBoundary: (Int16, Int16, Int16, Int16)
    public var leftBoundary: (Int16, Int16, Int16, Int16)
    public var topLeftBoundary: Int16
    public var padding1: Int16
    
    public var mcPred: (Int16, Int16, Int16, Int16,
                        Int16, Int16, Int16, Int16,
                        Int16, Int16, Int16, Int16,
                        Int16, Int16, Int16, Int16)
    
    public var ransBitCostsQ8: (UInt16, UInt16, UInt16, UInt16,
                                UInt16, UInt16, UInt16, UInt16,
                                UInt16, UInt16, UInt16, UInt16,
                                UInt16, UInt16, UInt16, UInt16)
    
    public init(
        frameIndex: UInt32,
        plane: UInt8,
        isAllZero: UInt8,
        blockX: UInt16,
        blockY: UInt16,
        qLow: Int16,
        lastVal: Int16,
        quantizedValues: (Int16, Int16, Int16, Int16,
                          Int16, Int16, Int16, Int16,
                          Int16, Int16, Int16, Int16,
                          Int16, Int16, Int16, Int16),
        dpcmErrors: (Int16, Int16, Int16, Int16,
                     Int16, Int16, Int16, Int16,
                     Int16, Int16, Int16, Int16,
                     Int16, Int16, Int16, Int16),
        topBoundary: (Int16, Int16, Int16, Int16),
        leftBoundary: (Int16, Int16, Int16, Int16),
        topLeftBoundary: Int16,
        mcPred: (Int16, Int16, Int16, Int16,
                 Int16, Int16, Int16, Int16,
                 Int16, Int16, Int16, Int16,
                 Int16, Int16, Int16, Int16),
        ransBitCostsQ8: (UInt16, UInt16, UInt16, UInt16,
                         UInt16, UInt16, UInt16, UInt16,
                         UInt16, UInt16, UInt16, UInt16,
                         UInt16, UInt16, UInt16, UInt16)
    ) {
        self.frameIndex = frameIndex
        self.plane = plane
        self.isAllZero = isAllZero
        self.blockX = blockX
        self.blockY = blockY
        self.qLow = qLow
        self.lastVal = lastVal
        self.padding0 = 0
        self.quantizedValues = quantizedValues
        self.dpcmErrors = dpcmErrors
        self.topBoundary = topBoundary
        self.leftBoundary = leftBoundary
        self.topLeftBoundary = topLeftBoundary
        self.padding1 = 0
        self.mcPred = mcPred
        self.ransBitCostsQ8 = ransBitCostsQ8
    }
}

/// DPCM 特徴量データダンプ管理クラス (シングルトン)
public final class DPCMDumpWriter: @unchecked Sendable {
    public static let shared = DPCMDumpWriter()
    
    public let isEnabled: Bool
    public let outputDir: String?
    private let lock = NSLock()
    private var fileHandle: FileHandle?
    private var buffer = [UInt8]()
    private let bufferCapacity = 65536 // 64KB メモリバッファ
    private var totalBlocks: UInt64 = 0
    private var zeroBlocks: UInt64 = 0
    private var currentFrameIndex: UInt32 = 0
    
    private init() {
        if let val = getenv("VEVC_DPCM_DUMP") {
            let path = String(cString: val)
            if path.isEmpty != true {
                self.isEnabled = true
                self.outputDir = path
                self.setupDirectoryAndFile(path: path)
            } else {
                self.isEnabled = false
                self.outputDir = nil
            }
        } else {
            self.isEnabled = false
            self.outputDir = nil
        }
    }
    
    private func setupDirectoryAndFile(path: String) {
        let fm = FileManager.default
        if fm.fileExists(atPath: path) != true {
            try? fm.createDirectory(atPath: path, withIntermediateDirectories: true)
        }
        let filePath = (path as NSString).appendingPathComponent("dpcm_blocks.bin")
        fm.createFile(atPath: filePath, contents: nil)
        self.fileHandle = FileHandle(forWritingAtPath: filePath)
        
        // ヘッダ 64バイトの初期書き込み (Magic: "VDPD", Version: 1, RecordSize: 164)
        var header = [UInt8](repeating: 0, count: 64)
        header[0] = 0x56 // 'V'
        header[1] = 0x44 // 'D'
        header[2] = 0x50 // 'P'
        header[3] = 0x44 // 'D'
        header[4] = 1; header[5] = 0; header[6] = 0; header[7] = 0 // Version = 1
        header[8] = 164; header[9] = 0; header[10] = 0; header[11] = 0 // RecordSize = 164
        self.fileHandle?.write(Data(header))
    }
    
    public func setFrameIndex(_ frameIndex: UInt32) {
        guard isEnabled else { return }
        lock.lock()
        defer { lock.unlock() }
        self.currentFrameIndex = frameIndex
    }
    
    public func recordBlock(
        plane: UInt8,
        blockX: UInt16,
        blockY: UInt16,
        isAllZero: Bool,
        qLow: Int16,
        lastVal: Int16,
        quantizedValues: (Int16, Int16, Int16, Int16,
                          Int16, Int16, Int16, Int16,
                          Int16, Int16, Int16, Int16,
                          Int16, Int16, Int16, Int16),
        dpcmErrors: (Int16, Int16, Int16, Int16,
                     Int16, Int16, Int16, Int16,
                     Int16, Int16, Int16, Int16,
                     Int16, Int16, Int16, Int16),
        topBoundary: (Int16, Int16, Int16, Int16),
        leftBoundary: (Int16, Int16, Int16, Int16),
        topLeftBoundary: Int16,
        mcPred: (Int16, Int16, Int16, Int16,
                 Int16, Int16, Int16, Int16,
                 Int16, Int16, Int16, Int16,
                 Int16, Int16, Int16, Int16),
        ransBitCostsQ8: (UInt16, UInt16, UInt16, UInt16,
                         UInt16, UInt16, UInt16, UInt16,
                         UInt16, UInt16, UInt16, UInt16,
                         UInt16, UInt16, UInt16, UInt16)
    ) {
        guard isEnabled else { return }
        lock.lock()
        defer { lock.unlock() }
        
        totalBlocks += 1
        if isAllZero {
            zeroBlocks += 1
        }
        
        var isAllZeroVal: UInt8 = 0
        if isAllZero {
            isAllZeroVal = 1
        }
        
        var record = DPCMBlockDumpRecord(
            frameIndex: currentFrameIndex,
            plane: plane,
            isAllZero: isAllZeroVal,
            blockX: blockX,
            blockY: blockY,
            qLow: qLow,
            lastVal: lastVal,
            quantizedValues: quantizedValues,
            dpcmErrors: dpcmErrors,
            topBoundary: topBoundary,
            leftBoundary: leftBoundary,
            topLeftBoundary: topLeftBoundary,
            mcPred: mcPred,
            ransBitCostsQ8: ransBitCostsQ8
        )
        
        withUnsafeBytes(of: &record) { rawPtr in
            buffer.append(contentsOf: rawPtr)
        }
        
        if bufferCapacity <= buffer.count {
            fileHandle?.write(Data(buffer))
            buffer.removeAll(keepingCapacity: true)
        }
    }
    
    public func close() {
        guard isEnabled else { return }
        lock.lock()
        defer { lock.unlock() }
        
        if buffer.isEmpty != true {
            fileHandle?.write(Data(buffer))
            buffer.removeAll()
        }
        
        // ヘッダの totalBlocks (offset 12..19) を更新
        try? fileHandle?.seek(toOffset: 12)
        var total = totalBlocks
        withUnsafeBytes(of: &total) { rawPtr in
            fileHandle?.write(Data(rawPtr))
        }
        try? fileHandle?.close()
        fileHandle = nil
        
        // JSON サマリーファイルの出力
        if let dir = outputDir {
            let summaryPath = (dir as NSString).appendingPathComponent("dpcm_summary.json")
            let nonZero = totalBlocks - zeroBlocks
            let summaryJSON = """
            {
              "totalBlocks": \(totalBlocks),
              "zeroBlocks": \(zeroBlocks),
              "nonZeroBlocks": \(nonZero),
              "recordSize": 164
            }
            """
            try? summaryJSON.write(toFile: summaryPath, atomically: true, encoding: .utf8)
            fputs("[VEVC_DPCM_DUMP] Dump completed: \(totalBlocks) blocks (\(zeroBlocks) zero) written to \(dir)\n", stderr)
        }
    }
}
