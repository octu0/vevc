import Foundation

@inlinable
@inline(__always)
func debugLog(_ message: @autoclosure () -> String) {
#if DEBUG
    if ProcessInfo.processInfo.environment["VEVC_DEBUG"] != nil {
        fputs(message() + "\n", stderr)
    }
#endif
}
