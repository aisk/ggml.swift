import CGGML

// Graph-building operations. Each method records an operation in the
// tensor's context and returns the (not yet computed) result tensor.
//
// Naming rule: the `ggml_` prefix is dropped and snake_case becomes
// camelCase — `ggml_mul_mat` → `mulMat`, `ggml_soft_max` → `softMax`.
// Parameter labels follow the upstream parameter names the same way.
extension Tensor {
    // Operations are recorded into the context of their leading operand,
    // which must be a graph arena — storage arenas (GGUF weights, staging
    // tensors) are sized for their tensors only, and recording into them
    // would abort deep inside ggml. Checked before the ggml call is made.
    var recordingContext: OpaquePointer {
        precondition(context.isBuilder,
                     """
                     tensor '\(name)' does not belong to a graph; operations are recorded \
                     into the graph of their leading operand — bind it first with within(_:)
                     """)
        return context.rawValue
    }

    // Additional source operands are passed through so that the result
    // carries their storage arenas in its retained list — cross-arena
    // sources (e.g. GGUF weights) must not be freed while the expression
    // can still be computed. Building the result into a graph then keeps
    // the whole chain alive for the graph's lifetime.
    private func wrap(_ result: UnsafeMutablePointer<ggml_tensor>?, _ sources: Tensor?...) -> Tensor {
        var retained = retained
        for case let source? in sources {
            Tensor.retain(source.context, into: &retained, recording: context)
            for storage in source.retained {
                Tensor.retain(storage, into: &retained, recording: context)
            }
        }
        return Tensor(rawValue: result!, context: context, retained: retained)
    }

    // MARK: - Arithmetic

    /// Element-wise addition (`b` is broadcast). Mirrors `ggml_add`.
    public func add(_ b: Tensor) -> Tensor {
        wrap(ggml_add(recordingContext, rawValue, b.rawValue), b)
    }

    /// Element-wise subtraction (`b` is broadcast). Mirrors `ggml_sub`.
    public func sub(_ b: Tensor) -> Tensor {
        wrap(ggml_sub(recordingContext, rawValue, b.rawValue), b)
    }

    /// Element-wise multiplication (`b` is broadcast). Mirrors `ggml_mul`.
    /// For matrix multiplication use ``mulMat(_:)``.
    public func mul(_ b: Tensor) -> Tensor {
        wrap(ggml_mul(recordingContext, rawValue, b.rawValue), b)
    }

    /// Element-wise division (`b` is broadcast). Mirrors `ggml_div`.
    public func div(_ b: Tensor) -> Tensor {
        wrap(ggml_div(recordingContext, rawValue, b.rawValue), b)
    }

    /// Multiplies every element by `s`. Mirrors `ggml_scale`.
    public func scale(_ s: Float) -> Tensor {
        wrap(ggml_scale(recordingContext, rawValue, s))
    }

    /// Matrix multiplication: `result = self * bᵀ`. Mirrors `ggml_mul_mat`.
    ///
    /// Following ggml's convention, `b` is applied transposed: for row-major
    /// matrices A `(n×k)` and B `(m×k)`, the result is `(m×n)` — i.e.
    /// `result[i][j] = dot(A[j], B[i])`.
    public func mulMat(_ b: Tensor) -> Tensor {
        wrap(ggml_mul_mat(recordingContext, rawValue, b.rawValue), b)
    }

    // MARK: - Activations

    /// Rectified linear unit. Mirrors `ggml_relu`.
    public func relu() -> Tensor {
        wrap(ggml_relu(recordingContext, rawValue))
    }

    /// Gaussian error linear unit. Mirrors `ggml_gelu`.
    public func gelu() -> Tensor {
        wrap(ggml_gelu(recordingContext, rawValue))
    }

    /// Sigmoid linear unit (a.k.a. swish). Mirrors `ggml_silu`.
    public func silu() -> Tensor {
        wrap(ggml_silu(recordingContext, rawValue))
    }

    /// Sigmoid. Mirrors `ggml_sigmoid`.
    public func sigmoid() -> Tensor {
        wrap(ggml_sigmoid(recordingContext, rawValue))
    }

    /// Hyperbolic tangent. Mirrors `ggml_tanh`.
    public func tanh() -> Tensor {
        wrap(ggml_tanh(recordingContext, rawValue))
    }

    /// Softmax over the innermost dimension. Mirrors `ggml_soft_max`.
    public func softMax() -> Tensor {
        wrap(ggml_soft_max(recordingContext, rawValue))
    }

    // MARK: - Normalization

    /// Layer normalization over the innermost dimension (no affine
    /// parameters). Mirrors `ggml_norm`.
    public func norm(eps: Float) -> Tensor {
        wrap(ggml_norm(recordingContext, rawValue, eps))
    }

    /// Root-mean-square normalization over the innermost dimension.
    /// Mirrors `ggml_rms_norm`.
    public func rmsNorm(eps: Float) -> Tensor {
        wrap(ggml_rms_norm(recordingContext, rawValue, eps))
    }

    // MARK: - Shape

    /// Returns a view with a new shape; the tensor must be contiguous and
    /// the element count must not change. Mirrors `ggml_reshape_1d` ... `_4d`.
    public func reshape(_ shape: Int...) -> Tensor {
        let ne = shape.map(Int64.init)
        switch ne.count {
        case 1: return wrap(ggml_reshape_1d(recordingContext, rawValue, ne[0]))
        case 2: return wrap(ggml_reshape_2d(recordingContext, rawValue, ne[0], ne[1]))
        case 3: return wrap(ggml_reshape_3d(recordingContext, rawValue, ne[0], ne[1], ne[2]))
        case 4: return wrap(ggml_reshape_4d(recordingContext, rawValue, ne[0], ne[1], ne[2], ne[3]))
        default: preconditionFailure("ggml tensors have 1 to 4 dimensions")
        }
    }

    /// Returns a view with the same shape as `other`. Mirrors `ggml_reshape`.
    public func reshape(like other: Tensor) -> Tensor {
        wrap(ggml_reshape(recordingContext, rawValue, other.rawValue), other)
    }

    /// Returns a view with the axes reordered; axis `i` of the result maps
    /// to axis `axisN` of the receiver. Mirrors `ggml_permute`.
    public func permute(_ axis0: Int, _ axis1: Int, _ axis2: Int, _ axis3: Int) -> Tensor {
        wrap(ggml_permute(recordingContext, rawValue, Int32(axis0), Int32(axis1), Int32(axis2), Int32(axis3)))
    }

    /// Returns a view with the first two axes swapped — an alias for
    /// `permute(1, 0, 2, 3)`. Mirrors `ggml_transpose`.
    public func transpose() -> Tensor {
        wrap(ggml_transpose(recordingContext, rawValue))
    }

    /// Makes the tensor contiguous in memory, copying if necessary
    /// (short for "contiguous"). Mirrors `ggml_cont`.
    public func cont() -> Tensor {
        wrap(ggml_cont(recordingContext, rawValue))
    }
}

// Rotation mode for ``Tensor/rope(_:nDims:mode:)``.
/// Mirrors the `GGML_ROPE_TYPE_*` constants.
public struct RopeMode: RawRepresentable, Equatable, Sendable {
    public let rawValue: Int32

    public init(rawValue: Int32) {
        self.rawValue = rawValue
    }

    public static let normal = RopeMode(rawValue: GGML_ROPE_TYPE_NORMAL)
    public static let neox = RopeMode(rawValue: GGML_ROPE_TYPE_NEOX)
    public static let mrope = RopeMode(rawValue: GGML_ROPE_TYPE_MROPE)
    public static let vision = RopeMode(rawValue: GGML_ROPE_TYPE_VISION)
    public static let imrope = RopeMode(rawValue: GGML_ROPE_TYPE_IMROPE)
}

extension Tensor {
    // MARK: - Unary math

    /// Element-wise negation. Mirrors `ggml_neg`.
    public func neg() -> Tensor {
        wrap(ggml_neg(recordingContext, rawValue))
    }

    /// Element-wise absolute value. Mirrors `ggml_abs`.
    public func abs() -> Tensor {
        wrap(ggml_abs(recordingContext, rawValue))
    }

    /// Element-wise square. Mirrors `ggml_sqr`.
    public func sqr() -> Tensor {
        wrap(ggml_sqr(recordingContext, rawValue))
    }

    /// Element-wise square root. Mirrors `ggml_sqrt`.
    public func sqrt() -> Tensor {
        wrap(ggml_sqrt(recordingContext, rawValue))
    }

    /// Element-wise natural logarithm. Mirrors `ggml_log`.
    public func log() -> Tensor {
        wrap(ggml_log(recordingContext, rawValue))
    }

    /// Element-wise exponential. Mirrors `ggml_exp`.
    public func exp() -> Tensor {
        wrap(ggml_exp(recordingContext, rawValue))
    }

    /// Element-wise sine. Mirrors `ggml_sin`.
    public func sin() -> Tensor {
        wrap(ggml_sin(recordingContext, rawValue))
    }

    /// Element-wise cosine. Mirrors `ggml_cos`.
    public func cos() -> Tensor {
        wrap(ggml_cos(recordingContext, rawValue))
    }

    /// Clamps every element into `[min, max]`. Mirrors `ggml_clamp`.
    public func clamp(min: Float, max: Float) -> Tensor {
        wrap(ggml_clamp(recordingContext, rawValue, min, max))
    }

    // MARK: - More activations

    /// Leaky rectified linear unit. Mirrors `ggml_leaky_relu`.
    public func leakyRelu(negativeSlope: Float) -> Tensor {
        wrap(ggml_leaky_relu(recordingContext, rawValue, negativeSlope, false))
    }

    /// Fast approximate GELU. Mirrors `ggml_gelu_quick`.
    public func geluQuick() -> Tensor {
        wrap(ggml_gelu_quick(recordingContext, rawValue))
    }

    // MARK: - Reductions

    /// Sum of all elements, as a single-element tensor. Mirrors `ggml_sum`.
    public func sum() -> Tensor {
        wrap(ggml_sum(recordingContext, rawValue))
    }

    /// Sums along rows: shape `[a,b,c,d]` becomes `[1,b,c,d]`.
    /// Mirrors `ggml_sum_rows`.
    public func sumRows() -> Tensor {
        wrap(ggml_sum_rows(recordingContext, rawValue))
    }

    /// Mean along rows: shape `[a,b,c,d]` becomes `[1,b,c,d]`.
    /// Mirrors `ggml_mean`.
    public func mean() -> Tensor {
        wrap(ggml_mean(recordingContext, rawValue))
    }

    /// Index of the maximum element in each row, as an i32 tensor.
    /// Mirrors `ggml_argmax`.
    public func argmax() -> Tensor {
        wrap(ggml_argmax(recordingContext, rawValue))
    }

    // MARK: - Data movement

    /// Copies the tensor into a new contiguous tensor. Mirrors `ggml_dup`.
    public func dup() -> Tensor {
        wrap(ggml_dup(recordingContext, rawValue))
    }

    /// Gathers rows by index; the receiver is the table (`[n_embd, n_rows]`)
    /// and `b` holds i32 row indices. Mirrors `ggml_get_rows`.
    public func getRows(_ b: Tensor) -> Tensor {
        wrap(ggml_get_rows(recordingContext, rawValue, b.rawValue), b)
    }

    /// Repeats the tensor to fit the shape of `other`. Mirrors `ggml_repeat`
    /// (named `repeated` because `repeat` is a Swift keyword).
    public func repeated(like other: Tensor) -> Tensor {
        wrap(ggml_repeat(recordingContext, rawValue, other.rawValue), other)
    }

    /// Concatenates `b` along the given dimension. Mirrors `ggml_concat`.
    public func concat(_ b: Tensor, dim: Int) -> Tensor {
        wrap(ggml_concat(recordingContext, rawValue, b.rawValue, Int32(dim)), b)
    }

    // MARK: - Attention building blocks

    /// Sets elements above the diagonal to `-INF` (causal mask).
    /// Mirrors `ggml_diag_mask_inf`.
    public func diagMaskInf(nPast: Int) -> Tensor {
        wrap(ggml_diag_mask_inf(recordingContext, rawValue, Int32(nPast)))
    }

    /// Rotary position embedding. The receiver is `[head_dim, n_head,
    /// n_tokens]` and `b` holds i32 positions per token. Mirrors `ggml_rope`.
    public func rope(_ b: Tensor, nDims: Int, mode: RopeMode = .normal) -> Tensor {
        wrap(ggml_rope(recordingContext, rawValue, b.rawValue, Int32(nDims), mode.rawValue), b)
    }
}

/// Pooling operation for ``Tensor/pool1d(_:k0:s0:p0:)``. Mirrors `ggml_op_pool`.
public enum PoolOp: Sendable {
    case max
    case avg

    var cValue: ggml_op_pool {
        switch self {
        case .max: return GGML_OP_POOL_MAX
        case .avg: return GGML_OP_POOL_AVG
        }
    }
}

extension Tensor {
    // MARK: - Pooling

    /// 1d pooling along the innermost dimension with kernel size `k0`,
    /// stride `s0` and padding `p0`. Mirrors `ggml_pool_1d`.
    public func pool1d(_ op: PoolOp, k0: Int, s0: Int, p0: Int = 0) -> Tensor {
        wrap(ggml_pool_1d(recordingContext, rawValue, op.cValue, Int32(k0), Int32(s0), Int32(p0)))
    }

    // MARK: - Graph flags

    /// Marks the tensor as a graph input, so allocators keep its buffer
    /// distinct and writable. Mirrors `ggml_set_input`.
    public func setInput() {
        ggml_set_input(rawValue)
    }

    /// Marks the tensor as a graph output, so allocators never reuse its
    /// buffer for other nodes. Mirrors `ggml_set_output`.
    public func setOutput() {
        ggml_set_output(rawValue)
    }
}

/// Sort direction for ``Tensor/argsort(order:)``. Mirrors `ggml_sort_order`.
public enum SortOrder: Sendable {
    case asc
    case desc

    var cValue: ggml_sort_order {
        switch self {
        case .asc: return GGML_SORT_ORDER_ASC
        case .desc: return GGML_SORT_ORDER_DESC
        }
    }
}

extension Tensor {
    // MARK: - Views

    /// A 1d view of `ne0` elements starting at `offset` bytes.
    /// Mirrors `ggml_view_1d`.
    public func view(_ ne0: Int, offset: Int) -> Tensor {
        wrap(ggml_view_1d(recordingContext, rawValue, Int64(ne0), offset))
    }

    /// A 2d view with row stride `nb1` (in bytes). Mirrors `ggml_view_2d`.
    public func view(_ ne0: Int, _ ne1: Int, nb1: Int, offset: Int) -> Tensor {
        wrap(ggml_view_2d(recordingContext, rawValue, Int64(ne0), Int64(ne1), nb1, offset))
    }

    /// A 3d view with row/slice strides in bytes. Mirrors `ggml_view_3d`.
    public func view(_ ne0: Int, _ ne1: Int, _ ne2: Int, nb1: Int, nb2: Int, offset: Int) -> Tensor {
        wrap(ggml_view_3d(recordingContext, rawValue,
                          Int64(ne0), Int64(ne1), Int64(ne2), nb1, nb2, offset))
    }

    /// A 4d view with strides in bytes. Mirrors `ggml_view_4d`.
    public func view(_ ne0: Int, _ ne1: Int, _ ne2: Int, _ ne3: Int,
                     nb1: Int, nb2: Int, nb3: Int, offset: Int) -> Tensor {
        wrap(ggml_view_4d(recordingContext, rawValue,
                          Int64(ne0), Int64(ne1), Int64(ne2), Int64(ne3), nb1, nb2, nb3, offset))
    }

    // MARK: - Copies and casts

    /// Copies the receiver into `b` (which may be a view, e.g. a KV cache
    /// slot) and returns a view of `b`. Mirrors `ggml_cpy`.
    public func cpy(to b: Tensor) -> Tensor {
        wrap(ggml_cpy(recordingContext, rawValue, b.rawValue), b)
    }

    /// Converts the tensor to another element type. Mirrors `ggml_cast`.
    public func cast(to type: TensorType) -> Tensor {
        wrap(ggml_cast(recordingContext, rawValue, type.cValue))
    }

    /// Returns a copy of the receiver with `b` written at `offset` bytes,
    /// treating the data as flat. Mirrors `ggml_set_1d`.
    public func set(_ b: Tensor, offset: Int) -> Tensor {
        wrap(ggml_set_1d(recordingContext, rawValue, b.rawValue, offset), b)
    }

    /// Returns a copy of the receiver with `b` written as a strided block
    /// at `offset` bytes. Mirrors `ggml_set`.
    public func set(_ b: Tensor, nb1: Int, nb2: Int, nb3: Int, offset: Int) -> Tensor {
        wrap(ggml_set(recordingContext, rawValue, b.rawValue, nb1, nb2, nb3, offset), b)
    }

    // MARK: - Attention

    /// Fused scaled masked softmax: `softMax(self * scale + mask)`, with
    /// optional ALiBi bias. Mirrors `ggml_soft_max_ext`.
    public func softMaxExt(mask: Tensor? = nil, scale: Float = 1, maxBias: Float = 0) -> Tensor {
        wrap(ggml_soft_max_ext(recordingContext, rawValue, mask?.rawValue, scale, maxBias), mask)
    }

    /// Fused attention: `softMax(self · kᵀ * scale + mask) · v`, where the
    /// receiver is the query. Mirrors `ggml_flash_attn_ext`; see the ggml
    /// header for the expected q/k/v layouts.
    public func flashAttnExt(
        k: Tensor,
        v: Tensor,
        mask: Tensor? = nil,
        scale: Float,
        maxBias: Float = 0,
        logitSoftcap: Float = 0
    ) -> Tensor {
        wrap(ggml_flash_attn_ext(recordingContext, rawValue, k.rawValue, v.rawValue,
                                 mask?.rawValue, scale, maxBias, logitSoftcap),
             k, v, mask)
    }

    // MARK: - Sorting

    /// Indices that would sort each row, as an i32 tensor.
    /// Mirrors `ggml_argsort`.
    public func argsort(order: SortOrder) -> Tensor {
        wrap(ggml_argsort(recordingContext, rawValue, order.cValue))
    }

    /// Indices of the `k` largest elements per row (in no particular
    /// order), as an i32 tensor. Mirrors `ggml_top_k`.
    public func topK(_ k: Int) -> Tensor {
        wrap(ggml_top_k(recordingContext, rawValue, Int32(k)))
    }
}

// Operator sugar for the element-wise arithmetic operations. Only
// unambiguous mappings get an operator; matrix multiplication does not.
extension Tensor {
    public static func + (a: Tensor, b: Tensor) -> Tensor { a.add(b) }
    public static func - (a: Tensor, b: Tensor) -> Tensor { a.sub(b) }
    public static func * (a: Tensor, b: Tensor) -> Tensor { a.mul(b) }
    public static func / (a: Tensor, b: Tensor) -> Tensor { a.div(b) }
    public static func * (a: Tensor, s: Float) -> Tensor { a.scale(s) }
    public static func * (s: Float, a: Tensor) -> Tensor { a.scale(s) }
}
