public struct FrameRateConverter {
    public let inFps: Int
    public let outFps: Int
    private var acc: Int = 0

    public init(inFps: Int, outFps: Int) {
        precondition(inFps > 0 && outFps > 0, "inFps and outFps must be positive integers")
        self.inFps = inFps
        self.outFps = outFps
    }

    /// 次の入力フレームの出力回数を返す（呼び出し順 = 入力フレーム順）
    public mutating func repeatCount() -> Int {
        acc += outFps
        let count = acc / inFps
        acc %= inFps
        return count
    }
}
