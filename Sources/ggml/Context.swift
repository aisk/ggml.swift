import CGGML

/// Owns a `ggml_context`: an arena that holds tensor metadata and compute
/// graphs. Internal — the public API deals in ``Graph``, ``GGUF`` and
/// ``Tensor``, which manage their contexts behind the scenes.
///
/// The context is freed (`ggml_free`) when the last reference to it — a
/// `Graph`, a `GGUF` or any `Tensor` created from it — goes away.
final class Context {
    let rawValue: OpaquePointer

    // Whether this arena belongs to a Graph and may record operations.
    // Storage arenas (GGUF weights, GGUF write staging) must not: they are
    // sized for their tensors only, and ggml aborts on arena overflow.
    let isBuilder: Bool

    // Backend buffers holding data of tensors from this context; retained
    // so the data outlives neither the context nor its tensors.
    var retainedBuffers: [BackendBuffer] = []

    // Contexts owning the storage of tensors that feed operations recorded
    // in this context; retained so cross-context source data (e.g. weights
    // loaded from a GGUF file) outlives the graphs built here.
    var retainedContexts: [Context] = []

    func retain(_ source: Context) {
        if source !== self && !retainedContexts.contains(where: { $0 === source }) {
            retainedContexts.append(source)
        }
    }

    /// Creates a context backed by an internally allocated arena of
    /// `memorySize` bytes. Mirrors `ggml_init`.
    ///
    /// `noAlloc` contexts hold only tensor metadata (data lives in backend
    /// buffers); the data-carrying mode remains solely for tensors staged
    /// in host memory for GGUF writing.
    init(memorySize: Int, noAlloc: Bool, isBuilder: Bool = false) {
        let params = ggml_init_params(mem_size: memorySize, mem_buffer: nil, no_alloc: noAlloc)
        guard let context = ggml_init(params) else {
            preconditionFailure("ggml_init failed to allocate a \(memorySize)-byte arena")
        }
        self.rawValue = context
        self.isBuilder = isBuilder
    }

    /// Takes ownership of an existing `ggml_context` created elsewhere
    /// (e.g. by `gguf_init_from_file`); it is freed on deinit as usual.
    init(adopting rawValue: OpaquePointer) {
        self.rawValue = rawValue
        self.isBuilder = false
    }

    deinit {
        ggml_free(rawValue)
    }

    /// Bytes of arena memory consumed by one tensor's metadata.
    /// Mirrors `ggml_tensor_overhead`.
    static var tensorOverhead: Int {
        ggml_tensor_overhead()
    }

    // MARK: - Tensors

    /// Creates a tensor of up to 4 dimensions. Mirrors `ggml_new_tensor`.
    func tensor(_ type: TensorType, shape: [Int]) -> Tensor {
        precondition((1...4).contains(shape.count), "ggml tensors have 1 to 4 dimensions")
        var ne = shape.map(Int64.init)
        let tensor = ggml_new_tensor(rawValue, type.cValue, Int32(ne.count), &ne)
        return Tensor(rawValue: tensor!, context: self)
    }

    /// Looks up a tensor in this context by name. Mirrors `ggml_get_tensor`.
    func tensor(named name: String) -> Tensor? {
        ggml_get_tensor(rawValue, name).map { Tensor(rawValue: $0, context: self) }
    }

    // MARK: - Backend allocation

    // Whether any non-view tensor still has no data, i.e. whether
    // `allocTensors(on:)` would actually allocate something. Views never
    // dangle on their own: their data resolves with their source.
    private var hasUnallocatedTensors: Bool {
        var tensor = ggml_get_first_tensor(rawValue)
        while let current = tensor {
            if current.pointee.data == nil && current.pointee.view_src == nil {
                return true
            }
            tensor = ggml_get_next_tensor(rawValue, current)
        }
        return false
    }

    /// Allocates all not-yet-allocated tensors of this context in a single
    /// buffer on `backend`, which the context keeps alive. Returns `nil`
    /// when every tensor is already allocated.
    /// Mirrors `ggml_backend_alloc_ctx_tensors`.
    func allocTensors(on backend: Backend) throws -> BackendBuffer? {
        guard hasUnallocatedTensors else { return nil }
        guard let raw = ggml_backend_alloc_ctx_tensors(rawValue, backend.rawValue) else {
            throw GGMLError.bufferAllocationFailed
        }
        let buffer = BackendBuffer(rawValue: raw, backend: backend)
        retainedBuffers.append(buffer)
        return buffer
    }
}
