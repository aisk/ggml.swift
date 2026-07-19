import CGGML

/// A compute graph allocated inside a ``Context``. Mirrors `ggml_cgraph` 1:1.
///
/// Like tensors, graphs are owned by their context; the class only keeps the
/// context alive and exposes the graph API.
public final class Graph {
    /// Default node capacity of a graph (`GGML_DEFAULT_GRAPH_SIZE`).
    public static let defaultSize = Int(GGML_DEFAULT_GRAPH_SIZE)

    let rawValue: OpaquePointer
    let context: Context

    init(context: Context) {
        self.rawValue = ggml_new_graph(context.rawValue)
        self.context = context
    }

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

    /// Computes the graph on the CPU, allocating scratch memory from the
    /// graph's own context. Mirrors `ggml_graph_compute_with_ctx`.
    ///
    /// This is the legacy compute path (tensor data lives in the context);
    /// backend-based computation will be added in a later step.
    public func compute(threads: Int = 1) throws {
        let status = Status(cValue: ggml_graph_compute_with_ctx(context.rawValue, rawValue, Int32(threads)))
        guard status == .success else {
            throw GGMLError.computeFailed(status)
        }
    }
}
