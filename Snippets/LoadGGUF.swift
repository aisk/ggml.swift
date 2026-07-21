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
    let gguf = GGUF()
    gguf.set("tiny-fc", forKey: "general.architecture")

    gguf.tensor(.f32, 4, 3, named: "fc1.weight")
        .copy(from: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1])
    gguf.tensor(.f32, 3, named: "fc1.bias")
        .copy(from: [0, -3, 0])
    gguf.tensor(.f32, 3, 2, named: "fc2.weight")
        .copy(from: [1, 1, 1, -1, 0, 1])
    gguf.tensor(.f32, 2, named: "fc2.bias")
        .copy(from: [0.5, -0.5])

    try gguf.write(to: path)
}

// --- Read: metadata first, then the tensor data onto a backend.
guard let cpu = Backend(type: .cpu) else {
    fatalError("no CPU backend available")
}
let gguf = try GGUF(path: path)
print("architecture:", gguf.string("general.architecture") ?? "?")
print("tensors:", gguf.tensorNames)
try gguf.load(on: cpu)

let fc1Weight = gguf.tensor(named: "fc1.weight")!
let fc1Bias = gguf.tensor(named: "fc1.bias")!
let fc2Weight = gguf.tensor(named: "fc2.weight")!
let fc2Bias = gguf.tensor(named: "fc2.bias")!

// --- Infer: loaded weights join a graph's expressions via within(_:).
let graph = Graph()
let x = graph.tensor(.f32, 4)

let hidden = fc1Weight.within(graph).mulMat(x).add(fc1Bias).relu()
let logits = fc2Weight.within(graph).mulMat(hidden).add(fc2Bias)
graph.buildForwardExpand(logits)

try graph.allocTensors(on: cpu)
x.copy(from: [1, 2, 3, 4])
try cpu.compute(graph)

print("logits:", logits.floats()) // [8.5, 5.5]
