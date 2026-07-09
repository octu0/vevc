import Foundation

public enum DWTFilterType: Equatable {
    case leGall53
    case cdf97
}

public struct IntraTileRect: Equatable {
    public let x: Int
    public let y: Int
    public let size: Int
    public let filter: DWTFilterType
    public let levels: Int
    
    public init(x: Int, y: Int, size: Int, filter: DWTFilterType, levels: Int) {
        self.x = x
        self.y = y
        self.size = size
        self.filter = filter
        self.levels = levels
    }
}

private func splitMargin(_ margin: Int) -> [Int] {
    var rem = margin
    var sizes: [Int] = []
    let candidates = [128, 32, 16, 8]
    for c in candidates {
        while rem >= c {
            sizes.append(c)
            rem -= c
        }
    }
    return sizes
}

public func computeIntraTileMap(width: Int, height: Int) -> [IntraTileRect] {
    let paddedW = (width + 7) & ~7
    let paddedH = (height + 7) & ~7
    let n512x = paddedW / 512
    let remX = paddedW % 512
    let marginX1 = (remX / 2) & ~7
    let marginX2 = remX - marginX1
    let xSizes = splitMargin(marginX1).reversed() + Array(repeating: 512, count: n512x) + splitMargin(marginX2)
    
    let n512y = paddedH / 512
    let remY = paddedH % 512
    let marginY1 = (remY / 2) & ~7
    let marginY2 = remY - marginY1
    let ySizes = splitMargin(marginY1).reversed() + Array(repeating: 512, count: n512y) + splitMargin(marginY2)
    
    var rects: [IntraTileRect] = []
    var cy = 0
    for ys in ySizes {
        var cx = 0
        for xs in xSizes {
            let size = min(xs, ys)
            let nx = xs / size
            let ny = ys / size
            for iy in 0..<ny {
                for ix in 0..<nx {
                    let filter: DWTFilterType = (size == 512 || size == 128) ? .cdf97 : .leGall53
                    let levels = (size == 512) ? 6 : (size == 128 ? 4 : 2)
                    rects.append(IntraTileRect(x: cx + ix * size, y: cy + iy * size, size: size, filter: filter, levels: levels))
                }
            }
            cx += xs
        }
        cy += ys
    }
    return rects
}
