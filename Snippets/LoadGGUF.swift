// Save a tiny fully connected model as a GGUF file, load it back and
// run inference from the loaded weights.
import Foundation
import ggml

let path = FileManager.default.temporaryDirectory
    .appendingPathComponent("tiny-fc.gguf").path

// snippet.hide
defer { try? FileManager.default.removeItem(atPath: path) }
// snippet.show

// --- Write: a 4 -> 3 -> 2 network with named tensors and metadata.
do {
    let weights = try Context(memorySize: 1024 * 1024)

    let fc1Weight = weights.tensor(.f32, 4, 3)
    fc1Weight.name = "fc1.weight"
    fc1Weight.copy(from: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1])
    let fc1Bias = weights.tensor(.f32, 3)
    fc1Bias.name = "fc1.bias"
    fc1Bias.copy(from: [0, -3, 0])
    let fc2Weight = weights.tensor(.f32, 3, 2)
    fc2Weight.name = "fc2.weight"
    fc2Weight.copy(from: [1, 1, 1, -1, 0, 1])
    let fc2Bias = weights.tensor(.f32, 2)
    fc2Bias.name = "fc2.bias"
    fc2Bias.copy(from: [0.5, -0.5])

    let gguf = GGUF()
    gguf.set("tiny-fc", forKey: "general.architecture")
    gguf.add(fc1Weight)
    gguf.add(fc1Bias)
    gguf.add(fc2Weight)
    gguf.add(fc2Bias)
    try gguf.write(to: path)
}

// --- Read: tensors are materialized into gguf.context.
let gguf = try GGUF(path: path)
print("architecture:", gguf.string("general.architecture") ?? "?")
print("tensors:", gguf.tensorNames)

let fc1Weight = gguf.tensor(named: "fc1.weight")!
let fc1Bias = gguf.tensor(named: "fc1.bias")!
let fc2Weight = gguf.tensor(named: "fc2.weight")!
let fc2Bias = gguf.tensor(named: "fc2.bias")!

// --- Infer: build the graph in a separate compute context. Loaded
// weights join an expression via within(_:).
let compute = try Context(memorySize: 1024 * 1024)
let x = compute.tensor(.f32, 4)
x.copy(from: [1, 2, 3, 4])

let hidden = fc1Weight.within(compute).mulMat(x).add(fc1Bias).relu()
let logits = fc2Weight.within(compute).mulMat(hidden).add(fc2Bias)

let graph = compute.graph()
graph.buildForwardExpand(logits)
try graph.compute()

print("logits:", logits.floats()) // [8.5, 5.5]
