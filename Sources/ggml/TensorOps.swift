import CGGML

// Graph-building operations. Each method records an operation in the
// tensor's context and returns the (not yet computed) result tensor.
//
// Naming rule: the `ggml_` prefix is dropped and snake_case becomes
// camelCase — `ggml_mul_mat` → `mulMat`, `ggml_soft_max` → `softMax`.
// Parameter labels follow the upstream parameter names the same way.
extension Tensor {
    private func wrap(_ result: UnsafeMutablePointer<ggml_tensor>?) -> Tensor {
        Tensor(rawValue: result!, context: context)
    }

    // MARK: - Arithmetic

    /// Element-wise addition (`b` is broadcast). Mirrors `ggml_add`.
    public func add(_ b: Tensor) -> Tensor {
        wrap(ggml_add(context.rawValue, rawValue, b.rawValue))
    }

    /// Element-wise subtraction (`b` is broadcast). Mirrors `ggml_sub`.
    public func sub(_ b: Tensor) -> Tensor {
        wrap(ggml_sub(context.rawValue, rawValue, b.rawValue))
    }

    /// Element-wise multiplication (`b` is broadcast). Mirrors `ggml_mul`.
    /// For matrix multiplication use ``mulMat(_:)``.
    public func mul(_ b: Tensor) -> Tensor {
        wrap(ggml_mul(context.rawValue, rawValue, b.rawValue))
    }

    /// Element-wise division (`b` is broadcast). Mirrors `ggml_div`.
    public func div(_ b: Tensor) -> Tensor {
        wrap(ggml_div(context.rawValue, rawValue, b.rawValue))
    }

    /// Multiplies every element by `s`. Mirrors `ggml_scale`.
    public func scale(_ s: Float) -> Tensor {
        wrap(ggml_scale(context.rawValue, rawValue, s))
    }

    /// Matrix multiplication: `result = self * bᵀ`. Mirrors `ggml_mul_mat`.
    ///
    /// Following ggml's convention, `b` is applied transposed: for row-major
    /// matrices A `(n×k)` and B `(m×k)`, the result is `(m×n)` — i.e.
    /// `result[i][j] = dot(A[j], B[i])`.
    public func mulMat(_ b: Tensor) -> Tensor {
        wrap(ggml_mul_mat(context.rawValue, rawValue, b.rawValue))
    }

    // MARK: - Activations

    /// Rectified linear unit. Mirrors `ggml_relu`.
    public func relu() -> Tensor {
        wrap(ggml_relu(context.rawValue, rawValue))
    }

    /// Gaussian error linear unit. Mirrors `ggml_gelu`.
    public func gelu() -> Tensor {
        wrap(ggml_gelu(context.rawValue, rawValue))
    }

    /// Sigmoid linear unit (a.k.a. swish). Mirrors `ggml_silu`.
    public func silu() -> Tensor {
        wrap(ggml_silu(context.rawValue, rawValue))
    }

    /// Sigmoid. Mirrors `ggml_sigmoid`.
    public func sigmoid() -> Tensor {
        wrap(ggml_sigmoid(context.rawValue, rawValue))
    }

    /// Hyperbolic tangent. Mirrors `ggml_tanh`.
    public func tanh() -> Tensor {
        wrap(ggml_tanh(context.rawValue, rawValue))
    }

    /// Softmax over the innermost dimension. Mirrors `ggml_soft_max`.
    public func softMax() -> Tensor {
        wrap(ggml_soft_max(context.rawValue, rawValue))
    }

    // MARK: - Normalization

    /// Layer normalization over the innermost dimension (no affine
    /// parameters). Mirrors `ggml_norm`.
    public func norm(eps: Float) -> Tensor {
        wrap(ggml_norm(context.rawValue, rawValue, eps))
    }

    /// Root-mean-square normalization over the innermost dimension.
    /// Mirrors `ggml_rms_norm`.
    public func rmsNorm(eps: Float) -> Tensor {
        wrap(ggml_rms_norm(context.rawValue, rawValue, eps))
    }

    // MARK: - Shape

    /// Returns a view with a new shape; the tensor must be contiguous and
    /// the element count must not change. Mirrors `ggml_reshape_1d` ... `_4d`.
    public func reshape(_ shape: Int...) -> Tensor {
        let ne = shape.map(Int64.init)
        switch ne.count {
        case 1: return wrap(ggml_reshape_1d(context.rawValue, rawValue, ne[0]))
        case 2: return wrap(ggml_reshape_2d(context.rawValue, rawValue, ne[0], ne[1]))
        case 3: return wrap(ggml_reshape_3d(context.rawValue, rawValue, ne[0], ne[1], ne[2]))
        case 4: return wrap(ggml_reshape_4d(context.rawValue, rawValue, ne[0], ne[1], ne[2], ne[3]))
        default: preconditionFailure("ggml tensors have 1 to 4 dimensions")
        }
    }

    /// Returns a view with the same shape as `other`. Mirrors `ggml_reshape`.
    public func reshape(like other: Tensor) -> Tensor {
        wrap(ggml_reshape(context.rawValue, rawValue, other.rawValue))
    }

    /// Returns a view with the axes reordered; axis `i` of the result maps
    /// to axis `axisN` of the receiver. Mirrors `ggml_permute`.
    public func permute(_ axis0: Int, _ axis1: Int, _ axis2: Int, _ axis3: Int) -> Tensor {
        wrap(ggml_permute(context.rawValue, rawValue, Int32(axis0), Int32(axis1), Int32(axis2), Int32(axis3)))
    }

    /// Returns a view with the first two axes swapped — an alias for
    /// `permute(1, 0, 2, 3)`. Mirrors `ggml_transpose`.
    public func transpose() -> Tensor {
        wrap(ggml_transpose(context.rawValue, rawValue))
    }

    /// Makes the tensor contiguous in memory, copying if necessary
    /// (short for "contiguous"). Mirrors `ggml_cont`.
    public func cont() -> Tensor {
        wrap(ggml_cont(context.rawValue, rawValue))
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
        wrap(ggml_neg(context.rawValue, rawValue))
    }

    /// Element-wise absolute value. Mirrors `ggml_abs`.
    public func abs() -> Tensor {
        wrap(ggml_abs(context.rawValue, rawValue))
    }

    /// Element-wise square. Mirrors `ggml_sqr`.
    public func sqr() -> Tensor {
        wrap(ggml_sqr(context.rawValue, rawValue))
    }

    /// Element-wise square root. Mirrors `ggml_sqrt`.
    public func sqrt() -> Tensor {
        wrap(ggml_sqrt(context.rawValue, rawValue))
    }

    /// Element-wise natural logarithm. Mirrors `ggml_log`.
    public func log() -> Tensor {
        wrap(ggml_log(context.rawValue, rawValue))
    }

    /// Element-wise exponential. Mirrors `ggml_exp`.
    public func exp() -> Tensor {
        wrap(ggml_exp(context.rawValue, rawValue))
    }

    /// Element-wise sine. Mirrors `ggml_sin`.
    public func sin() -> Tensor {
        wrap(ggml_sin(context.rawValue, rawValue))
    }

    /// Element-wise cosine. Mirrors `ggml_cos`.
    public func cos() -> Tensor {
        wrap(ggml_cos(context.rawValue, rawValue))
    }

    /// Clamps every element into `[min, max]`. Mirrors `ggml_clamp`.
    public func clamp(min: Float, max: Float) -> Tensor {
        wrap(ggml_clamp(context.rawValue, rawValue, min, max))
    }

    // MARK: - More activations

    /// Leaky rectified linear unit. Mirrors `ggml_leaky_relu`.
    public func leakyRelu(negativeSlope: Float) -> Tensor {
        wrap(ggml_leaky_relu(context.rawValue, rawValue, negativeSlope, false))
    }

    /// Fast approximate GELU. Mirrors `ggml_gelu_quick`.
    public func geluQuick() -> Tensor {
        wrap(ggml_gelu_quick(context.rawValue, rawValue))
    }

    // MARK: - Reductions

    /// Sum of all elements, as a single-element tensor. Mirrors `ggml_sum`.
    public func sum() -> Tensor {
        wrap(ggml_sum(context.rawValue, rawValue))
    }

    /// Sums along rows: shape `[a,b,c,d]` becomes `[1,b,c,d]`.
    /// Mirrors `ggml_sum_rows`.
    public func sumRows() -> Tensor {
        wrap(ggml_sum_rows(context.rawValue, rawValue))
    }

    /// Mean along rows: shape `[a,b,c,d]` becomes `[1,b,c,d]`.
    /// Mirrors `ggml_mean`.
    public func mean() -> Tensor {
        wrap(ggml_mean(context.rawValue, rawValue))
    }

    /// Index of the maximum element in each row, as an i32 tensor.
    /// Mirrors `ggml_argmax`.
    public func argmax() -> Tensor {
        wrap(ggml_argmax(context.rawValue, rawValue))
    }

    // MARK: - Data movement

    /// Copies the tensor into a new contiguous tensor. Mirrors `ggml_dup`.
    public func dup() -> Tensor {
        wrap(ggml_dup(context.rawValue, rawValue))
    }

    /// Gathers rows by index; the receiver is the table (`[n_embd, n_rows]`)
    /// and `b` holds i32 row indices. Mirrors `ggml_get_rows`.
    public func getRows(_ b: Tensor) -> Tensor {
        wrap(ggml_get_rows(context.rawValue, rawValue, b.rawValue))
    }

    /// Repeats the tensor to fit the shape of `other`. Mirrors `ggml_repeat`
    /// (named `repeated` because `repeat` is a Swift keyword).
    public func repeated(like other: Tensor) -> Tensor {
        wrap(ggml_repeat(context.rawValue, rawValue, other.rawValue))
    }

    /// Concatenates `b` along the given dimension. Mirrors `ggml_concat`.
    public func concat(_ b: Tensor, dim: Int) -> Tensor {
        wrap(ggml_concat(context.rawValue, rawValue, b.rawValue, Int32(dim)))
    }

    // MARK: - Attention building blocks

    /// Sets elements above the diagonal to `-INF` (causal mask).
    /// Mirrors `ggml_diag_mask_inf`.
    public func diagMaskInf(nPast: Int) -> Tensor {
        wrap(ggml_diag_mask_inf(context.rawValue, rawValue, Int32(nPast)))
    }

    /// Rotary position embedding. The receiver is `[head_dim, n_head,
    /// n_tokens]` and `b` holds i32 positions per token. Mirrors `ggml_rope`.
    public func rope(_ b: Tensor, nDims: Int, mode: RopeMode = .normal) -> Tensor {
        wrap(ggml_rope(context.rawValue, rawValue, b.rawValue, Int32(nDims), mode.rawValue))
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
