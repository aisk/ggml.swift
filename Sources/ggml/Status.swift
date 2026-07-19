import CGGML

/// Result of a ggml computation. Mirrors `ggml_status`.
public enum Status: Equatable, Sendable {
    case allocFailed
    case failed
    case success
    case aborted

    init(cValue: ggml_status) {
        switch cValue {
        case GGML_STATUS_ALLOC_FAILED: self = .allocFailed
        case GGML_STATUS_FAILED: self = .failed
        case GGML_STATUS_SUCCESS: self = .success
        case GGML_STATUS_ABORTED: self = .aborted
        default: self = .failed
        }
    }

    var cValue: ggml_status {
        switch self {
        case .allocFailed: return GGML_STATUS_ALLOC_FAILED
        case .failed: return GGML_STATUS_FAILED
        case .success: return GGML_STATUS_SUCCESS
        case .aborted: return GGML_STATUS_ABORTED
        }
    }

    /// Upstream description of the status. Mirrors `ggml_status_to_string`.
    public var message: String {
        String(cString: ggml_status_to_string(cValue))
    }
}

/// Errors thrown by the Swift binding.
public enum GGMLError: Error, Equatable {
    /// `ggml_init` returned NULL.
    case contextInitFailed
    /// A graph computation finished with a non-success status.
    case computeFailed(Status)
    /// `ggml_backend_alloc_ctx_tensors` could not allocate a buffer.
    case bufferAllocationFailed
    /// `gguf_init_from_file` could not read the file.
    case ggufLoadFailed(path: String)
    /// `gguf_write_to_file` could not write the file.
    case ggufWriteFailed(path: String)
}
