import Foundation
import SwiftUI
import Charts
import AppKit

struct CodecBenchmarkResult {
    let name: String
    let encTimeMs: Double
    let decTimeMs: Double
    let sizeKB: Double
    let stats: QualityStats?
}

struct BitrateSsimPoint: Hashable {
    let codec: String
    let bitrate: Int
    let ssimAvg: Double
    let ssimMin: Double
    let ssimMax: Double
    let sizeKB: Double
}

@available(macOS 13.0, *)
struct SpeedSizeChart: View {
    let results: [CodecBenchmarkResult]
    
    var body: some View {
        let maxTime = results.map { max($0.encTimeMs, $0.decTimeMs) }.max() ?? 1.0
        let maxSize = results.map { $0.sizeKB }.max() ?? 1.0
        let ratio = if 0 < maxSize { (maxTime / maxSize) } else { 1.0 }
        
        VStack(alignment: .leading) {
            Text("Speed & Size Benchmark")
                .font(.title)
                .padding()
            
            Chart(results, id: \.name) { res in
                let isVEVC = res.name.contains("VEVC")
                let normalizedSize = res.sizeKB * ratio
                
                let encodeCategory = if isVEVC { "VEVC Encode" } else { "Encode Time" }
                let decodeCategory = if isVEVC { "VEVC Decode" } else { "Decode Time" }
                let sizeCategory = if isVEVC { "VEVC Size" } else { "Size (KB)" }
                
                BarMark(
                    x: .value("Codec", res.name),
                    y: .value("Time (ms)", res.encTimeMs)
                )
                .foregroundStyle(by: .value("Category", encodeCategory))
                .position(by: .value("Type", "Encode Time"))
                
                BarMark(
                    x: .value("Codec", res.name),
                    y: .value("Time (ms)", res.decTimeMs)
                )
                .foregroundStyle(by: .value("Category", decodeCategory))
                .position(by: .value("Type", "Decode Time"))
                
                LineMark(
                    x: .value("Codec", res.name),
                    y: .value("Time (ms)", normalizedSize)
                )
                .foregroundStyle(by: .value("Category", sizeCategory))
                .symbol(.circle)
                .symbolSize(100)
                .annotation(position: .top) {
                    Text(String(format: "%.1f KB", res.sizeKB))
                        .font(.caption)
                        .padding(2)
                        .background(Color.white.opacity(0.8))
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel {
                        if let time = value.as(Double.self) {
                            Text(String(format: "%.0f ms", time))
                        }
                    }
                }
                AxisMarks(position: .trailing) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel {
                        if let time = value.as(Double.self) {
                            let size = time / ratio
                            Text(String(format: "%.0f KB", size))
                        }
                    }
                }
            }
            .chartForegroundStyleScale([
                "VEVC Encode": Color.orange,
                "VEVC Decode": Color.yellow,
                "VEVC Size": Color.red,
                "Encode Time": Color.blue.opacity(0.4),
                "Decode Time": Color.green.opacity(0.4),
                "Size (KB)": Color.purple.opacity(0.4)
            ])
            .frame(width: 800, height: 500)
            .padding()
        }
        .background(Color.white)
    }
}

@available(macOS 13.0, *)
struct PsnrChart: View {
    let results: [CodecBenchmarkResult]
    
    var body: some View {
        let validResults = results.filter { $0.stats != nil }
        
        VStack(alignment: .leading) {
            Text("PSNR Benchmark (Higher is better)")
                .font(.title)
                .padding()
            
            Chart(validResults, id: \.name) { res in
                let isVEVC = res.name.contains("VEVC")
                let stats = res.stats!
                let color = if isVEVC { Color.orange } else { Color.blue.opacity(0.8) }
                
                RuleMark(
                    x: .value("Codec", res.name),
                    yStart: .value("Min", stats.minPSNR),
                    yEnd: .value("Max", stats.maxPSNR)
                )
                .lineStyle(StrokeStyle(lineWidth: 2))
                .foregroundStyle(color)
                
                BarMark(
                    x: .value("Codec", res.name),
                    yStart: .value("Avg-SD", stats.avgPSNR - stats.stddevPSNR),
                    yEnd: .value("Avg+SD", stats.avgPSNR + stats.stddevPSNR),
                    width: .fixed(20)
                )
                .foregroundStyle(color.opacity(0.5))
                
                PointMark(
                    x: .value("Codec", res.name),
                    y: .value("Median", stats.p50PSNR)
                )
                .symbol(.circle)
                .foregroundStyle(color)
            }
            .chartYScale(domain: .automatic(includesZero: false))
            .frame(width: 800, height: 500)
            .padding()
        }
        .background(Color.white)
    }
}

@available(macOS 13.0, *)
struct SsimChart: View {
    let results: [CodecBenchmarkResult]
    
    var body: some View {
        let validResults = results.filter { $0.stats != nil }
        
        VStack(alignment: .leading) {
            Text("SSIM Benchmark (Closer to 1.0 is better)")
                .font(.title)
                .padding()
            
            Chart(validResults, id: \.name) { res in
                let isVEVC = res.name.contains("VEVC")
                let stats = res.stats!
                let color = if isVEVC { Color.orange } else { Color.green.opacity(0.8) }
                
                RuleMark(
                    x: .value("Codec", res.name),
                    yStart: .value("Min", stats.minSSIM),
                    yEnd: .value("Max", stats.maxSSIM)
                )
                .lineStyle(StrokeStyle(lineWidth: 2))
                .foregroundStyle(color)
                
                BarMark(
                    x: .value("Codec", res.name),
                    yStart: .value("Avg-SD", stats.avgSSIM - stats.stddevSSIM),
                    yEnd: .value("Avg+SD", stats.avgSSIM + stats.stddevSSIM),
                    width: .fixed(20)
                )
                .foregroundStyle(color.opacity(0.5))
                
                PointMark(
                    x: .value("Codec", res.name),
                    y: .value("Median", stats.p50SSIM)
                )
                .symbol(.circle)
                .foregroundStyle(color)
            }
            .chartYScale(domain: .automatic(includesZero: false))
            .frame(width: 800, height: 500)
            .padding()
        }
        .background(Color.white)
    }
}

@available(macOS 13.0, *)
@MainActor
func generateAndSaveCharts(results: [CodecBenchmarkResult], outDir: String = "docs") {
    // Speed & Size Chart
    let speedSizeView = SpeedSizeChart(results: results)
    let speedSizeRenderer = ImageRenderer(content: speedSizeView)
    speedSizeRenderer.scale = 2.0
    if let nsImage = speedSizeRenderer.nsImage,
       let tiffData = nsImage.tiffRepresentation,
       let bitmapImage = NSBitmapImageRep(data: tiffData),
       let pngData = bitmapImage.representation(using: .png, properties: [:]) {
        let path = URL(fileURLWithPath: "\(outDir)/speed_size.png")
        try? pngData.write(to: path)
        print("Saved \(path.path)")
    }
    
    // PSNR Chart
    if results.contains(where: { $0.stats != nil }) {
        let psnrView = PsnrChart(results: results)
        let psnrRenderer = ImageRenderer(content: psnrView)
        psnrRenderer.scale = 2.0
        if let nsImage = psnrRenderer.nsImage,
           let tiffData = nsImage.tiffRepresentation,
           let bitmapImage = NSBitmapImageRep(data: tiffData),
           let pngData = bitmapImage.representation(using: .png, properties: [:]) {
            let path = URL(fileURLWithPath: "\(outDir)/psnr.png")
            try? pngData.write(to: path)
            print("Saved \(path.path)")
        }
    }

    // SSIM Chart
    if results.contains(where: { $0.stats != nil }) {
        let ssimView = SsimChart(results: results)
        let ssimRenderer = ImageRenderer(content: ssimView)
        ssimRenderer.scale = 2.0
        if let nsImage = ssimRenderer.nsImage,
           let tiffData = nsImage.tiffRepresentation,
           let bitmapImage = NSBitmapImageRep(data: tiffData),
           let pngData = bitmapImage.representation(using: .png, properties: [:]) {
            let path = URL(fileURLWithPath: "\(outDir)/ssim.png")
            try? pngData.write(to: path)
            print("Saved \(path.path)")
        }
    }
}

@available(macOS 13.0, *)
struct BitrateSsimChart: View {
    let points: [BitrateSsimPoint]
    
    var body: some View {
        let maxDataSize = points.map { $0.sizeKB }.max() ?? 1000.0
        let maxChartSize = max(ceil(maxDataSize / 1000.0) * 1000.0, 15000.0)
        let ratio = maxChartSize
        
        VStack(alignment: .leading) {
            Text("Size & SSIM vs Bitrate")
                .font(.title)
                .padding()
            
            Chart(points, id: \.self) { pt in
                // Size Bar
                BarMark(
                    x: .value("Bitrate", String(pt.bitrate)),
                    y: .value("Size KB", pt.sizeKB)
                )
                .position(by: .value("Codec", pt.codec))
                .foregroundStyle(by: .value("Codec", pt.codec))
                .opacity(0.3)
                .annotation(position: .overlay, alignment: .bottom) {
                    let yOffset: CGFloat = {
                        if pt.codec.contains("VEVC") { return -35 }
                        if pt.codec.contains("H.264") { return -20 }
                        return -5
                    }()
                    Text(String(format: "%.0f KB", pt.sizeKB))
                        .font(.system(size: 9))
                        .lineLimit(1)
                        .fixedSize()
                        .offset(y: yOffset)
                        .foregroundColor(.secondary)
                }
                
                // SSIM Candlestick
                RuleMark(
                    x: .value("Bitrate", String(pt.bitrate)),
                    yStart: .value("SSIM Min", pt.ssimMin * ratio),
                    yEnd: .value("SSIM Max", pt.ssimMax * ratio)
                )
                .position(by: .value("Codec", pt.codec))
                .foregroundStyle(by: .value("Codec", pt.codec))
                
                // SSIM Avg Point
                PointMark(
                    x: .value("Bitrate", String(pt.bitrate)),
                    y: .value("SSIM Avg", pt.ssimAvg * ratio)
                )
                .position(by: .value("Codec", pt.codec))
                .foregroundStyle(by: .value("Codec", pt.codec))
                .symbol(.circle)
                .annotation(position: .top, alignment: .center) {
                    Text(String(format: "%.4f", pt.ssimAvg))
                        .font(.caption2)
                        .fontWeight(.regular)
                        .foregroundColor(.primary)
                }
            }
            .chartForegroundStyleScale([
                "VEVC (profile1)": Color.orange,
                "VEVC (profile2)": Color.red,
                "HEVC (SW)": Color.blue,
                "H.264 (SW)": Color.green
            ])
            .chartYScale(domain: 0...maxChartSize)
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text("\(Int(v)) KB")
                        }
                    }
                }
                AxisMarks(position: .trailing, values: Array(stride(from: 0.0, through: 1.0, by: 0.1).map { $0 * ratio })) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(String(format: "%.1f", v / ratio))
                        }
                    }
                }
            }
            .frame(width: 800, height: 500)
            .padding()
        }
        .background(Color.white)
    }
}

@available(macOS 13.0, *)
@MainActor
func generateAndSaveBitrateCharts(points: [BitrateSsimPoint], outDir: String = "docs") {
    let chartView = BitrateSsimChart(points: points)
    let renderer = ImageRenderer(content: chartView)
    renderer.scale = 2.0
    if let nsImage = renderer.nsImage,
       let tiffData = nsImage.tiffRepresentation,
       let bitmapImage = NSBitmapImageRep(data: tiffData),
       let pngData = bitmapImage.representation(using: .png, properties: [:]) {
        let path = URL(fileURLWithPath: "\(outDir)/bitrate_ssim.png")
        try? pngData.write(to: path)
        print("Saved \(path.path)")
    }
}
