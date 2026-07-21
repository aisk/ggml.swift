import CGGML

/// A compute graph together with the arena its tensors are recorded in.
/// Wraps `ggml_cgraph` and owns the `ggml_context` behind it.
///
/// Workflow: create a graph, create input tensors with ``tensor(_:_:)``,
/// chain operations (see `TensorOps`), pass the result to
/// ``buildForwardExpand(_:)``, then either let a ``Scheduler`` allocate and
/// compute the graph, or call ``allocTensors(on:)`` followed by
/// ``Backend/compute(_:)`` for a single backend. Tensor data always lives
/// in backend buffers; upload it with ``Tensor/copy(from:)`` after
/// allocation.
public final class Graph {
    /// Default tensor capacity of a graph (`GGML_DEFAULT_GRAPH_SIZE`).
    public static let defaultCapacity = Int(GGML_DEFAULT_GRAPH_SIZE)

    let rawValue: OpaquePointer
    let context: Context

    /// Creates an empty graph with room for `capacity` tensors — inputs
    /// and operation results combined. Mirrors `ggml_new_graph_custom`,
    /// with the metadata arena sized automatically.
    public init(capacity: Int = Graph.defaultCapacity) {
        precondition(capacity > 0, "graph capacity must be positive")
        let memorySize = Context.tensorOverhead * capacity
            + ggml_graph_overhead_custom(capacity, false)
        self.context = Context(memorySize: memorySize, noAlloc: true)
        self.rawValue = ggml_new_graph_custom(context.rawValue, capacity, false)
    }

    // MARK: - Tensors

    /// Creates a tensor of up to 4 dimensions.
    /// Mirrors `ggml_new_tensor` (and its `_1d` ... `_4d` variants).
    ///
    /// As in ggml, `shape[0]` is the number of elements in the innermost
    /// (contiguous) dimension — for a matrix that is the number of columns.
    public func tensor(_ type: TensorType, _ shape: Int...) -> Tensor {
        context.tensor(type, shape: shape)
    }

    /// Creates a tensor of up to 4 dimensions.
    /// Mirrors `ggml_new_tensor` (and its `_1d` ... `_4d` variants).
    public func tensor(_ type: TensorType, shape: [Int]) -> Tensor {
        context.tensor(type, shape: shape)
    }

    // MARK: - Building

    /// Number of operation nodes in the graph. Mirrors `ggml_graph_n_nodes`.
    public var nodeCount: Int {
        Int(ggml_graph_n_nodes(rawValue))
    }

    /// Returns the i-th node. Negative indices count from the end, so
    /// `node(-1)` is the last node. Mirrors `ggml_graph_node`.
    public func node(_ index: Int) -> Tensor {
        Tensor(rawValue: ggml_graph_node(rawValue, Int32(index)), context: context)
    }

    /// The last node of the graph — for a single-output graph this is the
    /// result tensor. `nil` for an empty graph.
    public var output: Tensor? {
        nodeCount > 0 ? node(-1) : nil
    }

    /// Expands the graph with the operations needed to compute `tensor`.
    /// Mirrors `ggml_build_forward_expand`.
    public func buildForwardExpand(_ tensor: Tensor) {
        ggml_build_forward_expand(rawValue, tensor.rawValue)
    }

    // MARK: - Allocation

    /// Allocates every not-yet-allocated tensor recorded in the graph's
    /// arena in one buffer on `backend` — inputs, weights and intermediate
    /// results alike. For multi-backend execution use a ``Scheduler``
    /// instead, which allocates the graph itself.
    ///
    /// The graph keeps the returned buffer alive; the result can be ignored
    /// unless the buffer itself is needed. Returns `nil` when every tensor
    /// is already allocated. Mirrors `ggml_backend_alloc_ctx_tensors`.
    @discardableResult
    public func allocTensors(on backend: Backend) throws -> BackendBuffer? {
        try context.allocTensors(on: backend)
    }
}
