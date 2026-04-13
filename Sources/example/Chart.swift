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
    let ssim: Double
}

@available(macOS 13.0, *)
struct SpeedSizeChart: View {
    let results: [CodecBenchmarkResult]
    
    var body: some View {
        let maxTime = results.map { max($0.encTimeMs, $0.decTimeMs) }.max() ?? 1.0
        let maxSize = results.map { $0.sizeKB }.max() ?? 1.0
        let ratio = maxSize > 0 ? (maxTime / maxSize) : 1.0
        
        VStack(alignment: .leading) {
            Text("Speed & Size Benchmark")
                .font(.title)
                .padding()
            
            Chart(results, id: \.name) { res in
                // Using BarMarks for Encoding and Decoding Times
                let isVEVC = res.name.contains("VEVC")
                BarMark(
                    x: .value("Codec", res.name),
                    y: .value("Time (ms)", res.encTimeMs)
                )
                .foregroundStyle(by: .value("Category", isVEVC ? "VEVC Encode" : "Encode Time"))
                .position(by: .value("Type", "Encode Time"))
                
                BarMark(
                    x: .value("Codec", res.name),
                    y: .value("Time (ms)", res.decTimeMs)
                )
                .foregroundStyle(by: .value("Category", isVEVC ? "VEVC Decode" : "Decode Time"))
                .position(by: .value("Type", "Decode Time"))
                
                // Normalized Size
                let normalizedSize = res.sizeKB * ratio
                PointMark(
                    x: .value("Codec", res.name),
                    y: .value("Time (ms)", normalizedSize)
                )
                .foregroundStyle(by: .value("Category", isVEVC ? "VEVC Size" : "Size (KB)"))
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
                let color = isVEVC ? Color.orange : Color.blue.opacity(0.8)
                
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
                let color = isVEVC ? Color.orange : Color.green.opacity(0.8)
                
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
        VStack(alignment: .leading) {
            Text("SSIM vs Bitrate (Higher is better)")
                .font(.title)
                .padding()
            
            Chart(points, id: \.self) { pt in
                LineMark(
                    x: .value("Bitrate", pt.bitrate),
                    y: .value("SSIM Avg", pt.ssim)
                )
                .foregroundStyle(by: .value("Codec", pt.codec))
                
                PointMark(
                    x: .value("Bitrate", pt.bitrate),
                    y: .value("SSIM Avg", pt.ssim)
                )
                .foregroundStyle(by: .value("Codec", pt.codec))
                .symbol(.circle)
            }
            .chartForegroundStyleScale([
                "VEVC (Layers)": Color.orange,
                "HEVC (SW)": Color.blue.opacity(0.3),
                "H.264 (SW)": Color.green.opacity(0.3)
            ])
            .chartXScale(domain: 100...1500)
            .chartYScale(domain: .automatic(includesZero: false))
            .chartXAxis {
                AxisMarks(values: Array(stride(from: 100, through: 1500, by: 100))) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel()
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
