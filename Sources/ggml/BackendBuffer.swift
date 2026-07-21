import CGGML

/// A block of memory on a backend device that holds tensor data.
/// Mirrors `ggml_backend_buffer_t` 1:1.
///
/// Buffers are usually created by ``Graph/allocTensors(on:)``,
/// ``GGUF/load(on:)`` or a ``Scheduler``; they must outlive the tensors
/// allocated inside them.
public final class BackendBuffer {
    let rawValue: OpaquePointer

    // The device the buffer lives on must not be freed before the buffer.
    private let backend: Backend

    init(rawValue: OpaquePointer, backend: Backend) {
        self.rawValue = rawValue
        self.backend = backend
    }

    deinit {
        ggml_backend_buffer_free(rawValue)
    }

    /// Name of the buffer's type, e.g. `"CPU"`. Mirrors `ggml_backend_buffer_name`.
    public var name: String {
        String(cString: ggml_backend_buffer_name(rawValue))
    }

    /// Size of the buffer in bytes. Mirrors `ggml_backend_buffer_get_size`.
    public var size: Int {
        ggml_backend_buffer_get_size(rawValue)
    }
}
