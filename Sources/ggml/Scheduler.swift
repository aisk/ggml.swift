import CGGML

/// Distributes the nodes of a compute graph across multiple backends,
/// allocating their tensors in backend buffers and copying data between
/// backends as needed. Mirrors `ggml_backend_sched_t` 1:1.
///
/// This is the modern compute path: build the graph in a `noAlloc` context,
/// let the scheduler allocate it, upload input data, then ``compute(_:)``.
public final class Scheduler {
    let rawValue: ggml_backend_sched_t

    // Backends must stay alive for the scheduler's whole lifetime; keeping
    // them referenced here also guarantees they are freed after the
    // scheduler itself in deinit.
    private let backends: [Backend]

    /// Creates a scheduler over the given backends, in order of priority
    /// (the last one is expected to be a CPU backend able to run anything).
    /// Mirrors `ggml_backend_sched_new`.
    public init(
        backends: [Backend],
        graphSize: Int = Graph.defaultSize,
        parallel: Bool = false,
        opOffload: Bool = true
    ) {
        precondition(!backends.isEmpty, "a scheduler needs at least one backend")
        var raw: [ggml_backend_t?] = backends.map { $0.rawValue }
        self.rawValue = ggml_backend_sched_new(
            &raw, nil, Int32(backends.count), graphSize, parallel, opOffload)
        self.backends = backends
    }

    deinit {
        ggml_backend_sched_free(rawValue)
    }

    /// Clears the allocation of the previous graph.
    /// Mirrors `ggml_backend_sched_reset`.
    public func reset() {
        ggml_backend_sched_reset(rawValue)
    }

    /// Splits the graph across the backends and allocates its tensors in
    /// backend buffers. Mirrors `ggml_backend_sched_alloc_graph`.
    @discardableResult
    public func allocGraph(_ graph: Graph) -> Bool {
        ggml_backend_sched_alloc_graph(rawValue, graph.rawValue)
    }

    /// Executes the graph, allocating it first if needed.
    /// Mirrors `ggml_backend_sched_graph_compute`.
    public func compute(_ graph: Graph) throws {
        let status = Status(cValue: ggml_backend_sched_graph_compute(rawValue, graph.rawValue))
        guard status == .success else {
            throw GGMLError.computeFailed(status)
        }
    }
}
