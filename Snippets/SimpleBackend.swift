// Multiply two matrices on the backend + scheduler path — a port of
// ggml's examples/simple/simple-backend.cpp. The user code stays identical
// when a GPU backend (Metal, CUDA, ...) is available.
import ggml

Backend.loadAll()
guard let best = Backend.best(), let cpu = Backend(type: .cpu) else {
    fatalError("no backend available")
}
let scheduler = Scheduler(backends: [best, cpu])

// The graph owns an arena for tensor metadata; tensor data lives in
// backend buffers allocated by the scheduler.
let graph = Graph()

// Row-major matrices; shape starts with the innermost dimension,
// so a 4x2 matrix is tensor(.f32, columns, rows).
let a = graph.tensor(.f32, 2, 4)
let b = graph.tensor(.f32, 2, 3)

// result = a * bᵀ — operations are recorded, not yet computed.
let result = a.mulMat(b)
graph.buildForwardExpand(result)

// The scheduler splits the graph across backends and allocates buffers.
scheduler.allocGraph(graph)

// Data uploads go through the backend (handles device memory).
a.copy(from: [
    2, 8,
    5, 1,
    4, 2,
    8, 6,
])
b.copy(from: [
    10, 5,
    9, 9,
    5, 4,
])

try scheduler.compute(graph)

print("computed on:", best.name)
print("mul mat \(result.shape) (transposed result):")
print(result.floats())
// [60.0, 55.0, 50.0, 110.0, 90.0, 54.0, 54.0, 126.0, 42.0, 29.0, 28.0, 64.0]
