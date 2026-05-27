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

final class ResolverBox: @unchecked Sendable {
    var action: (() -> Void)?
    init(action: @escaping () -> Void) { self.action = action }
}

actor FrameInbox<Element: Sendable>: Sendable {
    private var queue: [Element] = []
    private var waitingProducers: [(Element, CheckedContinuation<Void, Never>)] = []
    private var waitingConsumer: CheckedContinuation<Element?, Never>?
    private let capacity: Int
    private var finished = false

    init(capacity: Int) { self.capacity = capacity }

    func produce(_ frame: Element) async {
        if let c = waitingConsumer {
            waitingConsumer = nil
            c.resume(returning: frame)
            return
        }
        if queue.count < capacity {
            queue.append(frame)
            return
        }
        await withCheckedContinuation { cont in
            waitingProducers.append((frame, cont))
        }
    }

    func consume() async -> Element? {
        if queue.isEmpty != true {
            let f = queue.removeFirst()
            if let (next, cont) = waitingProducers.first {
                waitingProducers.removeFirst()
                queue.append(next)
                cont.resume()
            }
            return f
        }
        if finished { return nil }
        return await withCheckedContinuation { cont in
            waitingConsumer = cont
        }
    }

    func finish() {
        finished = true
        waitingConsumer?.resume(returning: nil)
        waitingConsumer = nil
        for (_, cont) in waitingProducers { cont.resume() }
        waitingProducers.removeAll()
    }
}

struct InboxStream<T: Sendable>: AsyncSequence, AsyncIteratorProtocol, @unchecked Sendable {
    typealias Element = T
    let inbox: FrameInbox<T>
    
    init(_ inbox: FrameInbox<T>) {
        self.inbox = inbox
    }
    
    func makeAsyncIterator() -> InboxStream {
        return self
    }
    
    mutating func next() async -> T? {
        return await inbox.consume()
    }
}

class EncoderSession {
    let id: Int
    let encoder: UnsafeEncoder
    let width: Int
    let height: Int
    let inbox: FrameInbox<YCbCrImage>
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
        
        self.inbox = FrameInbox(capacity: 2)
        let localInbox = self.inbox
        let stream = InboxStream(localInbox)
        
        let callbacks = Callbacks(onChunk: onChunk)
        let localEncoder = self.encoder
        
        Task {
            do {
                print("EncoderTask: starting encode")
                let outStream = await localEncoder.raw.encode(stream: stream)
                for try await chunk in outStream {
                    print("EncoderTask: got chunk of size \(chunk.count)")
                    let result = chunk.withUnsafeBytes { buf in
                        JSTypedArray<UInt8>(buffer: buf.bindMemory(to: UInt8.self))
                    }
                    _ = callbacks.onChunk.callAsFunction(result.jsValue)
                }
                print("EncoderTask: finished outStream")
            } catch {
                print("Encoder error: \(error)")
            }
        }
    }
}

class DecoderSession {
    let id: Int
    let decoder: UnsafeDecoder
    let inbox: FrameInbox<[UInt8]>
    var onFrame: JSObject
    
    struct Callbacks: @unchecked Sendable {
        let onFrame: JSObject
    }
    
    init(id: Int, onFrame: JSObject) {
        self.id = id
        self.decoder = UnsafeDecoder(vevc.Decoder(maxLayer: 2))
        self.onFrame = onFrame
        
        self.inbox = FrameInbox(capacity: 2)
        let localInbox = self.inbox
        let stream = InboxStream(localInbox)
        
        let callbacks = Callbacks(onFrame: onFrame)
        let localDecoder = self.decoder
        
        Task {
            do {
                let outStream = await localDecoder.raw.decode(stream: stream)
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
nonisolated(unsafe) var nextEncoderID = 1

nonisolated(unsafe) var decoderSessions = [Int: DecoderSession]()
nonisolated(unsafe) var nextDecoderID = 1

@JS
func createEncoder(width: Int, height: Int, maxbitrate: Int, framerate: Int, onChunk: JSObject) -> JSValue {
    let id = nextEncoderID
    nextEncoderID += 1
    let session = EncoderSession(id: id, width: width, height: height, maxbitrate: maxbitrate, framerate: framerate, onChunk: onChunk)
    encoderSessions[id] = session
    return .number(Double(id))
}

@JS
func encodeFrame(id: Int, data: JSValue, onDone: JSObject) {
    guard let session = encoderSessions[id] else {
        _ = onDone.callAsFunction()
        return
    }
    
    let width = session.encoder.raw.width
    let height = session.encoder.raw.height
    
    guard let object = data.object, let typedArray = JSTypedArray<UInt8>(from: object.jsValue) else { 
        _ = onDone.callAsFunction()
        return
    }
    
    let localData = [UInt8](unsafeUninitializedCapacity: typedArray.length) { ptr, initializedCount in
        typedArray.copyMemory(to: ptr)
        initializedCount = typedArray.length
    }
    
    let ycbcr = vevc.rgbaToYCbCr(data: localData, width: width, height: height)
    let inbox = session.inbox
    
    let box = ResolverBox { _ = onDone.callAsFunction() }
    Task {
        await inbox.produce(ycbcr)
        box.action?()
    }
}

@JS
func closeEncoder(id: Int) {
    guard let session = encoderSessions[id] else { return }
    let inbox = session.inbox
    Task {
        await inbox.finish()
    }
    encoderSessions.removeValue(forKey: id)
}

@JS
func createDecoder(onFrame: JSObject) -> JSValue {
    let id = nextDecoderID
    nextDecoderID += 1
    let session = DecoderSession(id: id, onFrame: onFrame)
    decoderSessions[id] = session
    return .number(Double(id))
}

@JS
func decodeChunk(id: Int, data: JSValue, onDone: JSObject) {
    guard let session = decoderSessions[id] else {
        _ = onDone.callAsFunction()
        return
    }
    
    guard let object = data.object, let typedArray = JSTypedArray<UInt8>(from: object.jsValue) else { 
        _ = onDone.callAsFunction()
        return
    }
    
    let localData = [UInt8](unsafeUninitializedCapacity: typedArray.length) { ptr, initializedCount in
        typedArray.copyMemory(to: ptr)
        initializedCount = typedArray.length
    }
    
    let inbox = session.inbox
    
    let box = ResolverBox { _ = onDone.callAsFunction() }
    Task {
        await inbox.produce(localData)
        box.action?()
    }
}

@JS
func closeDecoder(id: Int) {
    guard let session = decoderSessions[id] else { return }
    let inbox = session.inbox
    Task {
        await inbox.finish()
    }
    decoderSessions.removeValue(forKey: id)
}
