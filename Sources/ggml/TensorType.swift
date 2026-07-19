import CGGML

/// Element type of a tensor. Mirrors `ggml_type` 1:1.
///
/// Modeled as a struct wrapping the raw C value (instead of a closed enum) so
/// every upstream type stays representable; named constants are added as the
/// binding grows. Quantized types will follow in a later step.
public struct TensorType: RawRepresentable, Equatable, Hashable, Sendable {
    public let rawValue: UInt32

    public init(rawValue: UInt32) {
        self.rawValue = rawValue
    }

    init(cValue: ggml_type) {
        self.rawValue = cValue.rawValue
    }

    var cValue: ggml_type {
        ggml_type(rawValue: rawValue)
    }

    public static let f32 = TensorType(cValue: GGML_TYPE_F32)
    public static let f16 = TensorType(cValue: GGML_TYPE_F16)
    public static let f64 = TensorType(cValue: GGML_TYPE_F64)
    public static let bf16 = TensorType(cValue: GGML_TYPE_BF16)
    public static let i8 = TensorType(cValue: GGML_TYPE_I8)
    public static let i16 = TensorType(cValue: GGML_TYPE_I16)
    public static let i32 = TensorType(cValue: GGML_TYPE_I32)
    public static let i64 = TensorType(cValue: GGML_TYPE_I64)

    /// Size in bytes of all elements in a block. Mirrors `ggml_type_size`.
    /// For non-quantized types this is the size of a single element.
    public var size: Int {
        ggml_type_size(cValue)
    }

    /// Number of elements per block. Mirrors `ggml_blck_size`.
    public var blockSize: Int {
        Int(ggml_blck_size(cValue))
    }

    /// Whether this is a quantized type. Mirrors `ggml_is_quantized`.
    public var isQuantized: Bool {
        ggml_is_quantized(cValue)
    }

    /// Upstream name of the type, e.g. `"f32"`. Mirrors `ggml_type_name`.
    public var name: String {
        String(cString: ggml_type_name(cValue))
    }
}

extension TensorType: CustomStringConvertible {
    public var description: String { name }
}
