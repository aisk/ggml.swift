import CGGML

/// Owns a `ggml_context`: an arena that holds tensor metadata, tensor data
/// (unless `noAlloc` is set) and compute graphs. Mirrors `ggml_context` 1:1.
///
/// The context is freed (`ggml_free`) when the last reference to it — either
/// the `Context` itself or any `Tensor`/`Graph` created from it — goes away.
public final class Context {
    let rawValue: OpaquePointer

    /// Creates a context backed by an internally allocated arena of
    /// `memorySize` bytes. Mirrors `ggml_init`.
    ///
    /// - Parameters:
    ///   - memorySize: Total size of the arena. Use ``tensorOverhead`` and
    ///     ``graphOverhead`` plus the tensor data sizes to estimate it.
    ///   - noAlloc: When true, tensors created in this context carry no data
    ///     buffer (their data lives in a backend buffer instead).
    public init(memorySize: Int, noAlloc: Bool = false) throws {
        let params = ggml_init_params(mem_size: memorySize, mem_buffer: nil, no_alloc: noAlloc)
        guard let context = ggml_init(params) else {
            throw GGMLError.contextInitFailed
        }
        self.rawValue = context
    }

    deinit {
        ggml_free(rawValue)
    }

    /// Bytes of arena memory consumed by one tensor's metadata.
    /// Mirrors `ggml_tensor_overhead`.
    public static var tensorOverhead: Int {
        ggml_tensor_overhead()
    }

    /// Bytes of arena memory consumed by an (empty) compute graph.
    /// Mirrors `ggml_graph_overhead`.
    public static var graphOverhead: Int {
        ggml_graph_overhead()
    }

    /// Bytes of the arena currently in use. Mirrors `ggml_used_mem`.
    public var usedMemory: Int {
        ggml_used_mem(rawValue)
    }

    // MARK: - Tensors

    /// Creates a tensor of up to 4 dimensions.
    /// Mirrors `ggml_new_tensor` (and its `_1d` ... `_4d` variants).
    ///
    /// As in ggml, `shape[0]` is the number of elements in the innermost
    /// (contiguous) dimension — for a matrix that is the number of columns.
    public func tensor(_ type: TensorType, _ shape: Int...) -> Tensor {
        tensor(type, shape: shape)
    }

    /// Creates a tensor of up to 4 dimensions.
    /// Mirrors `ggml_new_tensor` (and its `_1d` ... `_4d` variants).
    public func tensor(_ type: TensorType, shape: [Int]) -> Tensor {
        precondition((1...4).contains(shape.count), "ggml tensors have 1 to 4 dimensions")
        var ne = shape.map(Int64.init)
        let tensor = ggml_new_tensor(rawValue, type.cValue, Int32(ne.count), &ne)
        return Tensor(rawValue: tensor!, context: self)
    }

    // MARK: - Operations

    /// Matrix multiplication: `result = a * bᵀ`. Mirrors `ggml_mul_mat`.
    ///
    /// Following ggml's convention, `b` is applied transposed: for row-major
    /// matrices A `(n×k)` and B `(m×k)`, the result is `(m×n)` — i.e.
    /// `result[i][j] = dot(A[j], B[i])`.
    public func mulMat(_ a: Tensor, _ b: Tensor) -> Tensor {
        Tensor(rawValue: ggml_mul_mat(rawValue, a.rawValue, b.rawValue), context: self)
    }

    // MARK: - Graphs

    /// Creates an empty compute graph with the default capacity
    /// (`GGML_DEFAULT_GRAPH_SIZE` nodes). Mirrors `ggml_new_graph`.
    public func graph() -> Graph {
        Graph(context: self)
    }
}
