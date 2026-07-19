import CGGML

/// A view over a `ggml_tensor` that lives inside a ``Context``.
/// Mirrors `ggml_tensor` 1:1.
///
/// Tensors do not own their storage — the context does. Each `Tensor` keeps
/// a strong reference to its context, so a tensor can never outlive the
/// arena its data lives in.
public struct Tensor {
    let rawValue: UnsafeMutablePointer<ggml_tensor>
    let context: Context

    init(rawValue: UnsafeMutablePointer<ggml_tensor>, context: Context) {
        self.rawValue = rawValue
        self.context = context
    }

    /// Element type of the tensor.
    public var type: TensorType {
        TensorType(cValue: rawValue.pointee.type)
    }

    /// Number of dimensions (1 for scalars). Mirrors `ggml_n_dims`.
    public var dimensions: Int {
        Int(ggml_n_dims(rawValue))
    }

    /// Number of elements per dimension, innermost first (`ne` in ggml).
    /// For a matrix this is `[columns, rows]`.
    public var shape: [Int] {
        let ne = rawValue.pointee.ne
        return [ne.0, ne.1, ne.2, ne.3].prefix(dimensions).map(Int.init)
    }

    /// Total number of elements. Mirrors `ggml_nelements`.
    public var elementCount: Int {
        Int(ggml_nelements(rawValue))
    }

    /// Total size of the tensor data in bytes. Mirrors `ggml_nbytes`.
    public var byteCount: Int {
        ggml_nbytes(rawValue)
    }

    /// Tensor name, used for debugging. Mirrors `ggml_get_name`/`ggml_set_name`.
    public var name: String {
        get { String(cString: ggml_get_name(rawValue)) }
        nonmutating set { ggml_set_name(rawValue, newValue) }
    }

    // MARK: - Data access

    /// Copies `values` into the tensor's data buffer.
    /// The tensor must be of type ``TensorType/f32``.
    public func copy(from values: [Float]) {
        precondition(type == .f32, "copy(from: [Float]) requires an f32 tensor, got \(type)")
        precondition(values.count == elementCount,
                     "value count \(values.count) does not match element count \(elementCount)")
        values.withUnsafeBytes { source in
            rawValue.pointee.data.copyMemory(from: source.baseAddress!, byteCount: byteCount)
        }
    }

    /// Reads the tensor's data buffer as an array of `Float`.
    /// The tensor must be of type ``TensorType/f32``.
    public func floats() -> [Float] {
        precondition(type == .f32, "floats() requires an f32 tensor, got \(type)")
        return [Float](unsafeUninitializedCapacity: elementCount) { destination, count in
            destination.baseAddress!.update(
                from: rawValue.pointee.data.assumingMemoryBound(to: Float.self),
                count: elementCount)
            count = elementCount
        }
    }

    /// Raw read-only access to the tensor's data buffer.
    public func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try body(UnsafeRawBufferPointer(start: rawValue.pointee.data, count: byteCount))
    }

    /// Raw mutable access to the tensor's data buffer.
    public func withUnsafeMutableBytes<R>(_ body: (UnsafeMutableRawBufferPointer) throws -> R) rethrows -> R {
        try body(UnsafeMutableRawBufferPointer(start: rawValue.pointee.data, count: byteCount))
    }

    // MARK: - Operations

    /// Matrix multiplication: `result = self * otherᵀ`, recorded in this
    /// tensor's context. Sugar for ``Context/mulMat(_:_:)``.
    public func mulMat(_ other: Tensor) -> Tensor {
        context.mulMat(self, other)
    }
}
