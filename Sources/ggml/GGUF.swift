import CGGML
import Foundation

/// A GGUF file: key-value metadata plus named tensors. Mirrors
/// `gguf_context` 1:1, covering both reading and writing.
///
/// Reading: ``init(path:)`` loads the metadata and tensor descriptions;
/// ``load(on:)`` then allocates the tensors on a backend and streams their
/// data from the file. Look them up with ``tensor(named:)``.
/// Writing: create with ``init()``, fill in metadata with `set`, create
/// tensors with ``tensor(_:_:named:)``, then ``write(to:)``.
public final class GGUF {
    let rawValue: OpaquePointer

    // The ggml context holding the tensor descriptions read from the file;
    // nil for a GGUF created empty for writing.
    let context: Context?

    private let path: String?

    /// Creates an empty GGUF for writing. Mirrors `gguf_init_empty`.
    public init() {
        self.rawValue = gguf_init_empty()
        self.context = nil
        self.path = nil
    }

    /// Loads a GGUF file's metadata and tensor descriptions. Tensor data
    /// stays on disk until ``load(on:)`` — or, for custom upload schemes,
    /// ``allocTensors(on:)`` plus ``dataOffset``/``tensorOffset(named:)``
    /// and ``Tensor/copy(from:)-(UnsafeRawBufferPointer)``.
    /// Mirrors `gguf_init_from_file`.
    public init(path: String) throws {
        var dataContext: OpaquePointer?
        let raw = withUnsafeMutablePointer(to: &dataContext) { pointer in
            gguf_init_from_file(path, gguf_init_params(no_alloc: true, ctx: pointer))
        }
        guard let raw else {
            throw GGMLError.ggufLoadFailed(path: path)
        }
        self.rawValue = raw
        self.context = dataContext.map(Context.init(adopting:))
        self.path = path
    }

    deinit {
        gguf_free(rawValue)
    }

    /// Format version of the file. Mirrors `gguf_get_version`.
    public var version: UInt32 {
        gguf_get_version(rawValue)
    }

    // MARK: - Tensors

    /// Number of tensors in the file. Mirrors `gguf_get_n_tensors`.
    public var tensorCount: Int {
        Int(gguf_get_n_tensors(rawValue))
    }

    /// Names of all tensors in the file. Mirrors `gguf_get_tensor_name`.
    public var tensorNames: [String] {
        (0..<tensorCount).map { String(cString: gguf_get_tensor_name(rawValue, Int64($0))) }
    }

    /// Looks up a tensor by name. Mirrors `ggml_get_tensor` on the file's
    /// context. Until ``load(on:)`` the tensor carries only metadata.
    public func tensor(named name: String) -> Tensor? {
        context?.tensor(named: name)
    }

    /// Allocates all tensor descriptions on `backend` without reading any
    /// data — the manual counterpart of ``load(on:)`` for custom upload
    /// schemes. The GGUF keeps the returned buffer alive; `nil` when every
    /// tensor is already allocated. Mirrors `ggml_backend_alloc_ctx_tensors`.
    @discardableResult
    public func allocTensors(on backend: Backend) throws -> BackendBuffer? {
        guard let context else {
            preconditionFailure("allocTensors(on:) requires a GGUF opened from a file")
        }
        return try context.allocTensors(on: backend)
    }

    /// Allocates all tensors on `backend` and reads their data from the
    /// file; afterwards the tensors from ``tensor(named:)`` are ready to
    /// use in graphs.
    public func load(on backend: Backend) throws {
        guard let context, let path else {
            preconditionFailure("load(on:) requires a GGUF opened from a file")
        }
        _ = try context.allocTensors(on: backend)
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path), options: .alwaysMapped) else {
            throw GGMLError.ggufLoadFailed(path: path)
        }
        for name in tensorNames {
            let tensor = context.tensor(named: name)!
            let begin = dataOffset + tensorOffset(named: name)!
            try data[begin..<begin + tensor.byteCount].withUnsafeBytes { try tensor.copy(from: $0) }
        }
    }

    // Tensors registered for writing. `gguf_add_tensor` copies the
    // `ggml_tensor` struct including its raw data pointer, so the sources
    // (and thereby their storage) must stay alive until `write(to:)`.
    private var addedTensors: [Tensor] = []

    /// Creates a named tensor to be written to the file, with its data
    /// staged in host memory — fill it with ``Tensor/copy(from:)`` any time
    /// before ``write(to:)``.
    public func tensor(_ type: TensorType, _ shape: Int..., named name: String) -> Tensor {
        tensor(type, shape: shape, named: name)
    }

    /// Creates a named tensor to be written to the file, with its data
    /// staged in host memory — fill it with ``Tensor/copy(from:)`` any time
    /// before ``write(to:)``.
    public func tensor(_ type: TensorType, shape: [Int], named name: String) -> Tensor {
        precondition((1...4).contains(shape.count), "ggml tensors have 1 to 4 dimensions")
        // A small data-carrying arena for exactly this tensor; 64 bytes of
        // slack absorb the arena's alignment padding.
        let bytes = shape.dropFirst().reduce(type.rowSize(shape[0]), *)
        let staging = Context(memorySize: Context.tensorOverhead + bytes + 64, noAlloc: false)
        let tensor = staging.tensor(type, shape: shape)
        tensor.name = name
        add(tensor)
        return tensor
    }

    /// Registers an existing tensor (metadata and data) to be written —
    /// e.g. one computed on a backend; the GGUF keeps it alive until
    /// written. Mirrors `gguf_add_tensor`.
    public func add(_ tensor: Tensor) {
        gguf_add_tensor(rawValue, tensor.rawValue)
        addedTensors.append(tensor)
    }

    /// Byte offset of the tensor data section within the file.
    /// Mirrors `gguf_get_data_offset`.
    public var dataOffset: Int {
        gguf_get_data_offset(rawValue)
    }

    /// Byte offset of a tensor's data relative to the start of the data
    /// section (``dataOffset``). Mirrors `gguf_get_tensor_offset`.
    public func tensorOffset(named name: String) -> Int? {
        let id = gguf_find_tensor(rawValue, name)
        return id >= 0 ? gguf_get_tensor_offset(rawValue, id) : nil
    }

    // MARK: - Metadata

    /// Number of key-value pairs. Mirrors `gguf_get_n_kv`.
    public var keyCount: Int {
        Int(gguf_get_n_kv(rawValue))
    }

    /// All metadata keys. Mirrors `gguf_get_key`.
    public var keys: [String] {
        (0..<keyCount).map { String(cString: gguf_get_key(rawValue, Int64($0))) }
    }

    private func find(_ key: String) -> Int64? {
        let id = gguf_find_key(rawValue, key)
        return id >= 0 ? id : nil
    }

    /// Reads a string value. Mirrors `gguf_get_val_str`.
    public func string(_ key: String) -> String? {
        guard let id = find(key), gguf_get_kv_type(rawValue, id) == GGUF_TYPE_STRING else {
            return nil
        }
        return String(cString: gguf_get_val_str(rawValue, id))
    }

    /// Reads a value of any integer type. Mirrors `gguf_get_val_u8` ... `_i64`.
    public func int(_ key: String) -> Int? {
        guard let id = find(key) else { return nil }
        switch gguf_get_kv_type(rawValue, id) {
        case GGUF_TYPE_UINT8: return Int(gguf_get_val_u8(rawValue, id))
        case GGUF_TYPE_INT8: return Int(gguf_get_val_i8(rawValue, id))
        case GGUF_TYPE_UINT16: return Int(gguf_get_val_u16(rawValue, id))
        case GGUF_TYPE_INT16: return Int(gguf_get_val_i16(rawValue, id))
        case GGUF_TYPE_UINT32: return Int(gguf_get_val_u32(rawValue, id))
        case GGUF_TYPE_INT32: return Int(gguf_get_val_i32(rawValue, id))
        case GGUF_TYPE_UINT64: return Int(exactly: gguf_get_val_u64(rawValue, id))
        case GGUF_TYPE_INT64: return Int(gguf_get_val_i64(rawValue, id))
        default: return nil
        }
    }

    /// Reads a value of either float type. Mirrors `gguf_get_val_f32`/`_f64`.
    public func double(_ key: String) -> Double? {
        guard let id = find(key) else { return nil }
        switch gguf_get_kv_type(rawValue, id) {
        case GGUF_TYPE_FLOAT32: return Double(gguf_get_val_f32(rawValue, id))
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(rawValue, id)
        default: return nil
        }
    }

    /// Reads a boolean value. Mirrors `gguf_get_val_bool`.
    public func bool(_ key: String) -> Bool? {
        guard let id = find(key), gguf_get_kv_type(rawValue, id) == GGUF_TYPE_BOOL else {
            return nil
        }
        return gguf_get_val_bool(rawValue, id)
    }

    /// Writes a string value. Mirrors `gguf_set_val_str`.
    public func set(_ value: String, forKey key: String) {
        gguf_set_val_str(rawValue, key, value)
    }

    /// Writes an integer value as int64. Mirrors `gguf_set_val_i64`.
    public func set(_ value: Int, forKey key: String) {
        gguf_set_val_i64(rawValue, key, Int64(value))
    }

    /// Writes a double value. Mirrors `gguf_set_val_f64`.
    public func set(_ value: Double, forKey key: String) {
        gguf_set_val_f64(rawValue, key, value)
    }

    /// Writes a boolean value. Mirrors `gguf_set_val_bool`.
    public func set(_ value: Bool, forKey key: String) {
        gguf_set_val_bool(rawValue, key, value)
    }

    // MARK: - Writing

    /// Writes the whole file, or only the metadata section when `onlyMeta`
    /// is true. Mirrors `gguf_write_to_file`.
    public func write(to path: String, onlyMeta: Bool = false) throws {
        guard gguf_write_to_file(rawValue, path, onlyMeta) else {
            throw GGMLError.ggufWriteFailed(path: path)
        }
    }
}
