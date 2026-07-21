import CGGML

/// A view over a `ggml_tensor` that lives inside an arena owned by a
/// ``Graph`` or a ``GGUF``. Mirrors `ggml_tensor` 1:1.
///
/// Tensors do not own their storage — the arena and backend buffers do.
/// Each `Tensor` keeps a strong reference to its arena, so a tensor can
/// never outlive the memory its metadata lives in.
public struct Tensor {
    let rawValue: UnsafeMutablePointer<ggml_tensor>
    let context: Context

    // Storage arenas (other than `context`) feeding the expression this
    // tensor is the result of; carried along the chain so that computing
    // the tensor can never read freed cross-arena data. Deduplicated by
    // identity — see retain(_:into:).
    let retained: [Context]

    init(rawValue: UnsafeMutablePointer<ggml_tensor>, context: Context, retained: [Context] = []) {
        self.rawValue = rawValue
        self.context = context
        self.retained = retained
    }

    // Adds `source` to `list` unless it is the recording context itself or
    // already present.
    static func retain(_ source: Context, into list: inout [Context], recording: Context) {
        if source !== recording && !list.contains(where: { $0 === source }) {
            list.append(source)
        }
    }

    /// Returns the same tensor bound to `graph`, so that operations
    /// starting from it are recorded there.
    ///
    /// Needed when the leading operand of an expression does not belong to
    /// the graph being built — typically weights loaded from a ``GGUF``
    /// file: `weights.within(graph).mulMat(x)`. The returned tensor (and
    /// every expression built from it) retains the receiver's own arena,
    /// so the underlying storage stays alive for as long as the results —
    /// and the graphs they are built into — do.
    public func within(_ graph: Graph) -> Tensor {
        var retained = retained
        Tensor.retain(context, into: &retained, recording: graph.context)
        return Tensor(rawValue: rawValue, context: graph.context, retained: retained)
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

    /// Byte stride per dimension (`nb` in ggml), innermost first.
    /// `strides[1]` is the row stride for a matrix.
    public var strides: [Int] {
        let nb = rawValue.pointee.nb
        let all: [Int] = [Int(nb.0), Int(nb.1), Int(nb.2), Int(nb.3)]
        return Array(all.prefix(dimensions))
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

    /// Whether the tensor's data has been allocated yet — graph tensors
    /// are allocated by a ``Scheduler`` or ``Graph/allocTensors(on:)``,
    /// GGUF tensors by ``GGUF/load(on:)``.
    public var isAllocated: Bool {
        rawValue.pointee.data != nil
    }

    // Tensors staged in host memory for GGUF writing carry their data in
    // their own arena and have no backend buffer; everything else lives in
    // a backend buffer once allocated.
    private var isBackendAllocated: Bool {
        rawValue.pointee.buffer != nil
    }

    private func writeRaw(_ source: UnsafeRawBufferPointer) {
        if isBackendAllocated {
            ggml_backend_tensor_set(rawValue, source.baseAddress!, 0, byteCount)
        } else {
            rawValue.pointee.data.copyMemory(from: source.baseAddress!, byteCount: byteCount)
        }
    }

    private func readRaw(into destination: UnsafeMutableRawPointer) {
        if isBackendAllocated {
            ggml_backend_tensor_get(rawValue, destination, 0, byteCount)
        } else {
            destination.copyMemory(from: rawValue.pointee.data, byteCount: byteCount)
        }
    }

    /// Copies `values` into the tensor's data, converting/quantizing to the
    /// tensor's element type on the fly (via `ggml_quantize_chunk` for
    /// non-f32 types). Types requiring an importance matrix are unsupported.
    ///
    /// Backend-allocated tensors go through `ggml_backend_tensor_set`
    /// (handles device memory); GGUF staging tensors are written directly.
    public func copy(from values: [Float]) {
        precondition(values.count == elementCount,
                     "value count \(values.count) does not match element count \(elementCount)")
        if type == .f32 {
            values.withUnsafeBytes(writeRaw)
            return
        }

        precondition(!type.requiresImatrix,
                     "quantizing to \(type) requires an importance matrix")
        let rowWidth = shape[0]
        let rows = elementCount / rowWidth
        var quantized = [UInt8](repeating: 0, count: byteCount)
        let written = quantized.withUnsafeMutableBytes { destination in
            ggml_quantize_chunk(type.cValue, values, destination.baseAddress!,
                                0, Int64(rows), Int64(rowWidth), nil)
        }
        precondition(written == byteCount, "quantization produced \(written) of \(byteCount) bytes")
        quantized.withUnsafeBytes(writeRaw)
    }

    /// Reads the tensor's data as an array of `Float`, dequantizing/
    /// converting from the tensor's element type on the fly (via the type's
    /// `to_float` trait). The tensor must be contiguous.
    ///
    /// Backend-allocated tensors go through `ggml_backend_tensor_get`
    /// (handles device memory); GGUF staging tensors are read directly.
    public func floats() -> [Float] {
        if type == .f32 {
            return [Float](unsafeUninitializedCapacity: elementCount) { destination, count in
                readRaw(into: destination.baseAddress!)
                count = elementCount
            }
        }

        guard let toFloat = ggml_get_type_traits(type.cValue).pointee.to_float else {
            preconditionFailure("\(type) cannot be converted to floats")
        }
        var raw = [UInt8](repeating: 0, count: byteCount)
        raw.withUnsafeMutableBytes { readRaw(into: $0.baseAddress!) }
        return [Float](unsafeUninitializedCapacity: elementCount) { destination, count in
            raw.withUnsafeBytes { source in
                toFloat(source.baseAddress!, destination.baseAddress!, Int64(elementCount))
            }
            count = elementCount
        }
    }

    /// Copies raw bytes into the tensor's data without any conversion —
    /// e.g. quantized blocks read straight from a GGUF file's data section.
    /// The byte count must match the tensor's ``byteCount`` exactly.
    public func copy(from bytes: UnsafeRawBufferPointer) {
        precondition(bytes.count == byteCount,
                     "byte count \(bytes.count) does not match tensor size \(byteCount)")
        writeRaw(bytes)
    }

    /// Copies `values` into the tensor's data.
    /// The tensor must be of type ``TensorType/i32``.
    ///
    /// Disfavored so that untyped integer-literal arrays resolve to the
    /// `[Float]` overload; pass `as [Int32]` explicitly for i32 tensors.
    @_disfavoredOverload
    public func copy(from values: [Int32]) {
        precondition(type == .i32, "copy(from: [Int32]) requires an i32 tensor, got \(type)")
        precondition(values.count == elementCount,
                     "value count \(values.count) does not match element count \(elementCount)")
        values.withUnsafeBytes(writeRaw)
    }

    /// Reads the tensor's data as an array of `Int32`.
    /// The tensor must be of type ``TensorType/i32``.
    public func int32s() -> [Int32] {
        precondition(type == .i32, "int32s() requires an i32 tensor, got \(type)")
        return [Int32](unsafeUninitializedCapacity: elementCount) { destination, count in
            readRaw(into: destination.baseAddress!)
            count = elementCount
        }
    }

    // Backend buffers may live in device memory, where `data` is not a
    // dereferenceable CPU address (`ggml_backend_buffer_is_host` is false).
    private var isHostAccessible: Bool {
        rawValue.pointee.buffer.map { ggml_backend_buffer_is_host($0) } ?? true
    }

    /// Raw read-only access to the tensor's data buffer. Only valid for
    /// host-resident data; for device tensors use ``floats()``.
    public func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        precondition(isHostAccessible,
                     "raw access requires host-resident data; use floats() for device tensors")
        return try body(UnsafeRawBufferPointer(start: rawValue.pointee.data, count: byteCount))
    }

    /// Raw mutable access to the tensor's data buffer. Only valid for
    /// host-resident data; for device tensors use ``copy(from:)``.
    public func withUnsafeMutableBytes<R>(_ body: (UnsafeMutableRawBufferPointer) throws -> R) rethrows -> R {
        precondition(isHostAccessible,
                     "raw access requires host-resident data; use copy(from:) for device tensors")
        return try body(UnsafeMutableRawBufferPointer(start: rawValue.pointee.data, count: byteCount))
    }
}
