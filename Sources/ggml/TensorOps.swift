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
