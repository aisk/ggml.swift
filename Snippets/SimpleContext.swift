// Multiply two matrices on the CPU with the legacy context workflow —
// a port of ggml's examples/simple/simple-ctx.cpp.
import ggml

// The context arena holds tensor data, metadata and the compute graph.
let context = try Context(memorySize: 16 * 1024 * 1024)

// Row-major matrices; shape starts with the innermost dimension,
// so a 4x2 matrix is tensor(.f32, columns, rows).
let a = context.tensor(.f32, 2, 4)
a.copy(from: [
    2, 8,
    5, 1,
    4, 2,
    8, 6,
])
let b = context.tensor(.f32, 2, 3)
b.copy(from: [
    10, 5,
    9, 9,
    5, 4,
])

// result = a * bᵀ — operations are recorded, not yet computed.
let result = a.mulMat(b)

let graph = context.graph()
graph.buildForwardExpand(result)
try graph.compute()

print("mul mat \(result.shape) (transposed result):")
print(result.floats())
// [60.0, 55.0, 50.0, 110.0, 90.0, 54.0, 54.0, 126.0, 42.0, 29.0, 28.0, 64.0]
