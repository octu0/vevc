import JavaScriptKit
import JavaScriptEventLoop
import vevc
import Foundation

JavaScriptEventLoop.installGlobalExecutor()

// Helper to create JS object from image data
func makeImageObject(width: Int, height: Int, data: [UInt8]) -> JSValue {
    let jsArr = data.withUnsafeBytes { buf in
        JSTypedArray<UInt8>(buffer: buf.bindMemory(to: UInt8.self))
    }
    let resultObj = JSObject()
    resultObj.data = jsArr.jsValue
    resultObj.width = .number(Double(width))
    resultObj.height = .number(Double(height))
    return resultObj.jsValue
}

final class UnsafeEncoder: @unchecked Sendable {
    let raw: vevc.VEVCEncoder
    init(_ raw: vevc.VEVCEncoder) { self.raw = raw }
}

final class UnsafeDecoder: @unchecked Sendable {
    let raw: vevc.Decoder
    init(_ raw: vevc.Decoder) { self.raw = raw }
}

class EncoderSession {
    let id: Int
    let encoder: UnsafeEncoder
    let width: Int
    let height: Int
    let continuation: AsyncStream<YCbCrImage>.Continuation
    var onChunk: JSObject
    
    struct Callbacks: @unchecked Sendable {
        let onChunk: JSObject
    }
    
    init(id: Int, width: Int, height: Int, maxbitrate: Int, framerate: Int, onChunk: JSObject) {
        self.id = id
        self.width = width
        self.height = height
        self.encoder = UnsafeEncoder(vevc.VEVCEncoder(width: width, height: height, maxbitrate: maxbitrate, framerate: framerate))
        self.onChunk = onChunk
        
        let (stream, continuation) = AsyncStream<YCbCrImage>.makeStream()
        self.continuation = continuation
        
        let callbacks = Callbacks(onChunk: onChunk)
        let localEncoder = self.encoder
        
        Task {
            do {
                let outStream = localEncoder.raw.encode(stream: stream)
                for try await chunk in outStream {
                    let result = chunk.withUnsafeBytes { buf in
                        JSTypedArray<UInt8>(buffer: buf.bindMemory(to: UInt8.self))
                    }
                    _ = callbacks.onChunk.callAsFunction(result.jsValue)
                }
            } catch {
                print("Encoder error: \(error)")
            }
        }
    }
}

class DecoderSession {
    let id: Int
    let decoder: UnsafeDecoder
    let continuation: AsyncStream<[UInt8]>.Continuation
    var onFrame: JSObject
    
    struct Callbacks: @unchecked Sendable {
        let onFrame: JSObject
    }
    
    init(id: Int, onFrame: JSObject) {
        self.id = id
        self.decoder = UnsafeDecoder(vevc.Decoder(maxLayer: 2))
        self.onFrame = onFrame
        
        let (stream, continuation) = AsyncStream<[UInt8]>.makeStream()
        self.continuation = continuation
        
        let callbacks = Callbacks(onFrame: onFrame)
        let localDecoder = self.decoder
        
        Task {
            do {
                let outStream = localDecoder.raw.decode(stream: stream)
                for try await img in outStream {
                    let rgba = vevc.ycbcrToRGBA(img: img)
                    _ = callbacks.onFrame.callAsFunction(makeImageObject(width: img.width, height: img.height, data: rgba))
                }
            } catch {
                print("Decoder error: \(error)")
            }
        }
    }
}

nonisolated(unsafe) var encoderSessions = [Int: EncoderSession]()
nonisolated(unsafe) var nextEncoderId = 1

nonisolated(unsafe) var decoderSessions = [Int: DecoderSession]()
nonisolated(unsafe) var nextDecoderId = 1

@JS
func createEncoder(width: Int, height: Int, maxbitrate: Int, framerate: Int, onChunk: JSObject) -> JSValue {
    let id = nextEncoderId
    nextEncoderId += 1
    let session = EncoderSession(id: id, width: width, height: height, maxbitrate: maxbitrate, framerate: framerate, onChunk: onChunk)
    encoderSessions[id] = session
    return .number(Double(id))
}

@JS
func encodeFrame(id: Int, data: JSValue) {
    guard let session = encoderSessions[id] else { return }
    guard let object = data.object, let typedArray = JSTypedArray<UInt8>(from: object.jsValue) else { return }
    
    var localData = [UInt8](repeating: 0, count: typedArray.length)
    localData.withUnsafeMutableBufferPointer { ptr in
        typedArray.copyMemory(to: ptr)
    }
    let width = session.width
    let height = session.height
    Task {
        // Run conversion in task to not block JS heavily
        let ycbcr = vevc.rgbaToYCbCr(data: localData, width: width, height: height)
        session.continuation.yield(ycbcr)
    }
}

@JS
func closeEncoder(id: Int) {
    guard let session = encoderSessions[id] else { return }
    session.continuation.finish()
    encoderSessions.removeValue(forKey: id)
}

@JS
func createDecoder(onFrame: JSObject) -> JSValue {
    let id = nextDecoderId
    nextDecoderId += 1
    let session = DecoderSession(id: id, onFrame: onFrame)
    decoderSessions[id] = session
    return .number(Double(id))
}

@JS
func decodeChunk(id: Int, data: JSValue) {
    guard let session = decoderSessions[id] else { return }
    guard let object = data.object, let typedArray = JSTypedArray<UInt8>(from: object.jsValue) else { return }
    
    var localData = [UInt8](repeating: 0, count: typedArray.length)
    localData.withUnsafeMutableBufferPointer { ptr in
        typedArray.copyMemory(to: ptr)
    }
    session.continuation.yield(localData)
}

@JS
func closeDecoder(id: Int) {
    guard let session = decoderSessions[id] else { return }
    session.continuation.finish()
    decoderSessions.removeValue(forKey: id)
}
