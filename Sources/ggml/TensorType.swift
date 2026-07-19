import CGGML

/// Element type of a tensor. Mirrors `ggml_type` 1:1.
///
/// Modeled as a struct wrapping the raw C value (instead of a closed enum)
/// so every upstream type stays representable. Constant names keep ggml's
/// exact spelling (`q4_0`, `q2_K`, ...) since the underscores distinguish
/// variants.
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

    public static let q4_0 = TensorType(cValue: GGML_TYPE_Q4_0)
    public static let q4_1 = TensorType(cValue: GGML_TYPE_Q4_1)
    public static let q5_0 = TensorType(cValue: GGML_TYPE_Q5_0)
    public static let q5_1 = TensorType(cValue: GGML_TYPE_Q5_1)
    public static let q8_0 = TensorType(cValue: GGML_TYPE_Q8_0)
    public static let q8_1 = TensorType(cValue: GGML_TYPE_Q8_1)
    public static let q2_K = TensorType(cValue: GGML_TYPE_Q2_K)
    public static let q3_K = TensorType(cValue: GGML_TYPE_Q3_K)
    public static let q4_K = TensorType(cValue: GGML_TYPE_Q4_K)
    public static let q5_K = TensorType(cValue: GGML_TYPE_Q5_K)
    public static let q6_K = TensorType(cValue: GGML_TYPE_Q6_K)
    public static let q8_K = TensorType(cValue: GGML_TYPE_Q8_K)
    public static let iq2_xxs = TensorType(cValue: GGML_TYPE_IQ2_XXS)
    public static let iq2_xs = TensorType(cValue: GGML_TYPE_IQ2_XS)
    public static let iq3_xxs = TensorType(cValue: GGML_TYPE_IQ3_XXS)
    public static let iq1_s = TensorType(cValue: GGML_TYPE_IQ1_S)
    public static let iq1_m = TensorType(cValue: GGML_TYPE_IQ1_M)
    public static let iq4_nl = TensorType(cValue: GGML_TYPE_IQ4_NL)
    public static let iq3_s = TensorType(cValue: GGML_TYPE_IQ3_S)
    public static let iq2_s = TensorType(cValue: GGML_TYPE_IQ2_S)
    public static let iq4_xs = TensorType(cValue: GGML_TYPE_IQ4_XS)
    public static let tq1_0 = TensorType(cValue: GGML_TYPE_TQ1_0)
    public static let tq2_0 = TensorType(cValue: GGML_TYPE_TQ2_0)
    public static let mxfp4 = TensorType(cValue: GGML_TYPE_MXFP4)
    public static let nvfp4 = TensorType(cValue: GGML_TYPE_NVFP4)
    public static let q1_0 = TensorType(cValue: GGML_TYPE_Q1_0)
    public static let q2_0 = TensorType(cValue: GGML_TYPE_Q2_0)

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

    /// Size in bytes of a row of `ne` elements. Mirrors `ggml_row_size`.
    public func rowSize(_ ne: Int) -> Int {
        ggml_row_size(cValue, Int64(ne))
    }

    /// Whether quantizing to this type requires an importance matrix,
    /// which the plain `copy(from:)` quantization path does not support.
    /// Mirrors `ggml_quantize_requires_imatrix`.
    public var requiresImatrix: Bool {
        ggml_quantize_requires_imatrix(cValue)
    }

    /// Upstream name of the type, e.g. `"f32"`. Mirrors `ggml_type_name`.
    public var name: String {
        String(cString: ggml_type_name(cValue))
    }
}

extension TensorType: CustomStringConvertible {
    public var description: String { name }
}
