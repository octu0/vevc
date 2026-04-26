#!/usr/bin/env swift
import Foundation

let width = 1920
let height = 1080
let frames = 60
let filename = "spec_1080p_60f.y4m"

func generate() {
    let url = URL(fileURLWithPath: filename)
    FileManager.default.createFile(atPath: url.path, contents: nil, attributes: nil)
    guard let fileHandle = try? FileHandle(forWritingTo: url) else {
        print("Failed to open file for writing")
        return
    }
    defer { fileHandle.closeFile() }
    
    let header = "YUV4MPEG2 W\(width) H\(height) F60:1 Ip A0:0 C420\n"
    fileHandle.write(header.data(using: .ascii)!)
    
    let ySize = width * height
    let uvSize = (width / 2) * (height / 2)
    
    var yData = Data(count: ySize)
    var uData = Data(count: uvSize)
    var vData = Data(count: uvSize)
    
    let frameHeader = "FRAME\n".data(using: .ascii)!
    
    for f in 0..<frames {
        fileHandle.write(frameHeader)
        
        yData.withUnsafeMutableBytes { ptr in
            let yBuf = ptr.bindMemory(to: UInt8.self).baseAddress!
            for y in 0..<height {
                let rowOffset = y * width
                for x in 0..<width {
                    // Checkerboard background
                    let checker = ((x / 32) + (y / 32)) % 2 == 0 ? UInt8(64) : UInt8(192)
                    
                    // Moving box (200x200)
                    let boxX = (f * 15) % (width - 200)
                    let boxY = (f * 10) % (height - 200)
                    
                    if x >= boxX && x < boxX + 200 && y >= boxY && y < boxY + 200 {
                        // High frequency noise inside the box
                        let noise = UInt8((x ^ y ^ f) % 255)
                        yBuf[rowOffset + x] = noise
                    } else {
                        yBuf[rowOffset + x] = checker
                    }
                }
            }
        }
        
        uData.withUnsafeMutableBytes { ptr in
            let uBuf = ptr.bindMemory(to: UInt8.self).baseAddress!
            let w2 = width / 2
            let h2 = height / 2
            for y in 0..<h2 {
                let rowOffset = y * w2
                for x in 0..<w2 {
                    let boxX = (f * 15) / 2 % (w2 - 100)
                    let boxY = (f * 10) / 2 % (h2 - 100)
                    
                    if x >= boxX && x < boxX + 100 && y >= boxY && y < boxY + 100 {
                        uBuf[rowOffset + x] = UInt8((x + f) % 255)
                    } else {
                        uBuf[rowOffset + x] = 128
                    }
                }
            }
        }
        
        vData.withUnsafeMutableBytes { ptr in
            let vBuf = ptr.bindMemory(to: UInt8.self).baseAddress!
            let w2 = width / 2
            let h2 = height / 2
            for y in 0..<h2 {
                let rowOffset = y * w2
                for x in 0..<w2 {
                    let boxX = (f * 15) / 2 % (w2 - 100)
                    let boxY = (f * 10) / 2 % (h2 - 100)
                    
                    if x >= boxX && x < boxX + 100 && y >= boxY && y < boxY + 100 {
                        vBuf[rowOffset + x] = UInt8((y + f) % 255)
                    } else {
                        vBuf[rowOffset + x] = 128
                    }
                }
            }
        }
        
        fileHandle.write(yData)
        fileHandle.write(uData)
        fileHandle.write(vData)
        
        if f % 10 == 0 {
            print("Generated frame \(f)")
        }
    }
    print("Done. Generated \(filename)")
}

generate()
