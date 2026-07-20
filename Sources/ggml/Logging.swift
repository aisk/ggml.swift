import CGGML
import Foundation

/// Verbosity of a ggml log message. Mirrors `ggml_log_level`.
public enum LogLevel: Equatable, Sendable {
    case none
    case debug
    case info
    case warn
    case error
    /// Continuation of the previous message — ggml emits messages in
    /// fragments; a `cont` fragment belongs to the preceding line.
    case cont

    init(cValue: ggml_log_level) {
        switch cValue {
        case GGML_LOG_LEVEL_DEBUG: self = .debug
        case GGML_LOG_LEVEL_INFO: self = .info
        case GGML_LOG_LEVEL_WARN: self = .warn
        case GGML_LOG_LEVEL_ERROR: self = .error
        case GGML_LOG_LEVEL_CONT: self = .cont
        default: self = .none
        }
    }
}

// ggml_log_set is process-global, so the Swift-side callback storage is a
// process-global too. The C trampoline cannot capture context; it reads the
// current callback from here under a lock (logs can arrive from any thread).
private let logLock = NSLock()
private nonisolated(unsafe) var currentLogCallback: GGML.LogCallback?

extension GGML {
    /// Receives every ggml log fragment. May be called from any thread.
    public typealias LogCallback = @Sendable (LogLevel, String) -> Void

    /// Routes all future ggml log output (backend registration, scheduler
    /// diagnostics, GGUF errors, ...) to `callback` instead of stderr.
    /// Pass `nil` to restore the default stderr output.
    /// Mirrors `ggml_log_set`.
    public static func setLogCallback(_ callback: LogCallback?) {
        logLock.lock()
        currentLogCallback = callback
        logLock.unlock()

        if callback == nil {
            ggml_log_set(nil, nil)
        } else {
            ggml_log_set({ level, text, _ in
                logLock.lock()
                let callback = currentLogCallback
                logLock.unlock()
                guard let callback, let text else { return }
                callback(LogLevel(cValue: level), String(cString: text))
            }, nil)
        }
    }
}
