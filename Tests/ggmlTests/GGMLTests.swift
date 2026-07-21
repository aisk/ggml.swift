import XCTest
@testable import ggml

final class GGMLTests: XCTestCase {
    func testVendoredGGMLReleaseIsAvailable() {
        XCTAssertEqual(GGML.version, "0.16.0")
        XCTAssertEqual(GGML.commit, "524f974bb21a1013408f76d71c15732482c0c3fe")
    }

    func testLogCallback() {
        final class Sink: @unchecked Sendable {
            private let lock = NSLock()
            private var entries: [(LogLevel, String)] = []

            func append(_ level: LogLevel, _ message: String) {
                lock.lock()
                entries.append((level, message))
                lock.unlock()
            }

            var errors: [String] {
                lock.lock()
                defer { lock.unlock() }
                return entries.filter { $0.0 == .error }.map(\.1)
            }
        }

        let sink = Sink()
        GGML.setLogCallback { level, message in sink.append(level, message) }
        defer { GGML.setLogCallback(nil) }

        // A failing GGUF load logs through GGML_LOG_ERROR.
        XCTAssertThrowsError(try GGUF(path: "/nonexistent/model.gguf"))
        XCTAssertFalse(sink.errors.isEmpty)
        XCTAssertTrue(sink.errors.joined().contains("failed"))
    }
}

final class GraphTests: XCTestCase {
    func testTensorMetadata() {
        let graph = Graph()

        let tensor = graph.tensor(.f32, 3, 2)
        tensor.name = "weights"

        XCTAssertEqual(tensor.type, .f32)
        XCTAssertEqual(tensor.type.name, "f32")
        XCTAssertEqual(tensor.dimensions, 2)
        XCTAssertEqual(tensor.shape, [3, 2])
        XCTAssertEqual(tensor.elementCount, 6)
        XCTAssertEqual(tensor.byteCount, 6 * MemoryLayout<Float>.size)
        XCTAssertEqual(tensor.name, "weights")
        XCTAssertFalse(tensor.isAllocated)

        XCTAssertEqual(graph.nodeCount, 0)
        XCTAssertNil(graph.output)
    }
}

/// Port of ggml's `examples/simple/simple-backend.cpp`.
final class SimpleBackendTests: XCTestCase {
    func testMatMulViaScheduler() throws {
        Backend.loadAll()

        let best = try XCTUnwrap(Backend.best())
        let cpu = try XCTUnwrap(Backend(type: .cpu))
        let scheduler = Scheduler(backends: [best, cpu])

        // The graph's arena only holds tensor metadata; tensor data is
        // allocated in backend buffers by the scheduler.
        let graph = Graph()

        let a = graph.tensor(.f32, 2, 4)
        let b = graph.tensor(.f32, 2, 3)
        let result = a.mulMat(b)
        graph.buildForwardExpand(result)

        scheduler.reset()
        XCTAssertTrue(scheduler.allocGraph(graph))
        XCTAssertTrue(a.isAllocated)

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

        XCTAssertEqual(graph.output?.rawValue, result.rawValue)
        XCTAssertEqual(result.type, .f32)
        XCTAssertEqual(result.shape, [4, 3])
        XCTAssertEqual(result.floats(), [
            60, 55, 50, 110,
            90, 54, 54, 126,
            42, 29, 28, 64,
        ])
    }

    func testCPUBackendProperties() throws {
        let cpu = try XCTUnwrap(Backend(type: .cpu))
        XCTAssertEqual(cpu.name, "CPU")

        let byName = Backend(name: "CPU")
        XCTAssertNotNil(byName)
    }
}

/// GGUF write/read round trip, then inference from the loaded weights —
/// a miniature of ggml's `examples/mnist` eval path.
final class GGUFTests: XCTestCase {
    private var path: String!
    private var cpu: Backend!

    override func setUpWithError() throws {
        cpu = try XCTUnwrap(Backend(type: .cpu))
        path = FileManager.default.temporaryDirectory
            .appendingPathComponent("ggml-swift-test-\(UUID().uuidString).gguf").path
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: path)
        cpu = nil
    }

    private func writeModel() throws {
        let gguf = GGUF()
        gguf.set("mnist-fc", forKey: "general.architecture")
        gguf.set(3, forKey: "test.epochs")
        gguf.set(0.05, forKey: "test.learning_rate")
        gguf.set(true, forKey: "test.trained")

        // A 4 -> 3 -> 2 fully connected network with hand-picked weights.
        gguf.tensor(.f32, 4, 3, named: "fc1.weight").copy(from: [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 1,
        ])
        gguf.tensor(.f32, 3, named: "fc1.bias").copy(from: [0, -3, 0])
        gguf.tensor(.f32, 3, 2, named: "fc2.weight").copy(from: [
            1, 1, 1,
            -1, 0, 1,
        ])
        gguf.tensor(.f32, 2, named: "fc2.bias").copy(from: [0.5, -0.5])
        try gguf.write(to: path)
    }

    func testRoundTripMetadataAndTensors() throws {
        try writeModel()

        let gguf = try GGUF(path: path)

        XCTAssertEqual(gguf.string("general.architecture"), "mnist-fc")
        XCTAssertEqual(gguf.int("test.epochs"), 3)
        XCTAssertEqual(gguf.double("test.learning_rate"), 0.05)
        XCTAssertEqual(gguf.bool("test.trained"), true)
        XCTAssertNil(gguf.string("missing.key"))
        XCTAssertEqual(Set(gguf.keys).isSuperset(of: [
            "general.architecture", "test.epochs", "test.learning_rate", "test.trained",
        ]), true)

        XCTAssertEqual(gguf.tensorCount, 4)
        XCTAssertEqual(Set(gguf.tensorNames),
                       ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"])

        // Before load(on:) the tensors carry only metadata.
        let fc1Weight = try XCTUnwrap(gguf.tensor(named: "fc1.weight"))
        XCTAssertEqual(fc1Weight.shape, [4, 3])
        XCTAssertFalse(fc1Weight.isAllocated)

        try gguf.load(on: cpu)
        XCTAssertTrue(fc1Weight.isAllocated)
        XCTAssertEqual(fc1Weight.floats(), [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 1,
        ])
        XCTAssertNil(gguf.tensor(named: "fc3.weight"))
    }

    func testInferenceFromLoadedModel() throws {
        try writeModel()

        let gguf = try GGUF(path: path)
        try gguf.load(on: cpu)
        let fc1Weight = try XCTUnwrap(gguf.tensor(named: "fc1.weight"))
        let fc1Bias = try XCTUnwrap(gguf.tensor(named: "fc1.bias"))
        let fc2Weight = try XCTUnwrap(gguf.tensor(named: "fc2.weight"))
        let fc2Bias = try XCTUnwrap(gguf.tensor(named: "fc2.bias"))

        // Build the eval graph; the loaded weights join via within(_:).
        let graph = Graph()
        let x = graph.tensor(.f32, 4)

        let hidden = fc1Weight.within(graph).mulMat(x).add(fc1Bias).relu()
        let logits = fc2Weight.within(graph).mulMat(hidden).add(fc2Bias)
        graph.buildForwardExpand(logits)

        try graph.allocTensors(on: cpu)
        x.copy(from: [1, 2, 3, 4])
        try cpu.compute(graph)

        // fc1: [1, 2, 7] + [0, -3, 0] -> relu -> [1, 0, 7]
        // fc2: [8, 6] + [0.5, -0.5] -> [8.5, 5.5]
        XCTAssertEqual(logits.floats(), [8.5, 5.5])
    }

    func testGraphRetainsWeightContexts() throws {
        try writeModel()

        let graph = Graph()
        let x = graph.tensor(.f32, 4)

        // Drop the GGUF (and the arena owning the weights) before
        // computing; within(_:) and the recorded operations must keep the
        // weight storage alive.
        let logits: Tensor
        do {
            let gguf = try GGUF(path: path)
            try gguf.load(on: cpu)
            let fc1Weight = try XCTUnwrap(gguf.tensor(named: "fc1.weight"))
            let fc1Bias = try XCTUnwrap(gguf.tensor(named: "fc1.bias"))
            let fc2Weight = try XCTUnwrap(gguf.tensor(named: "fc2.weight"))
            let fc2Bias = try XCTUnwrap(gguf.tensor(named: "fc2.bias"))
            let hidden = fc1Weight.within(graph).mulMat(x).add(fc1Bias).relu()
            logits = fc2Weight.within(graph).mulMat(hidden).add(fc2Bias)
        }

        graph.buildForwardExpand(logits)
        try graph.allocTensors(on: cpu)
        x.copy(from: [1, 2, 3, 4])
        try cpu.compute(graph)

        XCTAssertEqual(logits.floats(), [8.5, 5.5])
    }

    func testAddedTensorsSurviveSourceRelease() throws {
        // A backend-allocated tensor registered via add(_:) must stay
        // alive (and readable by the writer) after its graph is released.
        let gguf = GGUF()
        do {
            let graph = Graph()
            let weight = graph.tensor(.f32, 4)
            weight.name = "w"
            try graph.allocTensors(on: cpu)
            weight.copy(from: [1, 2, 3, 4])
            gguf.add(weight)
        }
        try gguf.write(to: path)

        let loaded = try GGUF(path: path)
        try loaded.load(on: cpu)
        XCTAssertEqual(try XCTUnwrap(loaded.tensor(named: "w")).floats(), [1, 2, 3, 4])
    }

    func testLoadFailureThrows() {
        XCTAssertThrowsError(try GGUF(path: "/nonexistent/model.gguf")) { error in
            XCTAssertEqual(error as? GGMLError,
                           .ggufLoadFailed(path: "/nonexistent/model.gguf"))
        }
    }
}

/// Explicit backend-buffer allocation and direct backend compute — the
/// single-backend alternative to a ``Scheduler``.
final class BackendBufferTests: XCTestCase {
    func testAllocTensorsAndCompute() throws {
        let cpu = try XCTUnwrap(Backend(type: .cpu))
        cpu.cpuSetNThreads(2)

        let graph = Graph()
        let a = graph.tensor(.f32, 2, 4)
        let b = graph.tensor(.f32, 2, 3)
        let result = a.mulMat(b)
        graph.buildForwardExpand(result)

        // Allocates every tensor in the graph — inputs and intermediate
        // results alike — in one buffer on the backend.
        let buffer = try XCTUnwrap(graph.allocTensors(on: cpu))
        XCTAssertGreaterThan(buffer.size, 0)
        XCTAssertFalse(buffer.name.isEmpty)
        XCTAssertTrue(a.isAllocated)

        // A second call has nothing left to allocate.
        XCTAssertNil(try graph.allocTensors(on: cpu))

        a.copy(from: [2, 8, 5, 1, 4, 2, 8, 6])
        b.copy(from: [10, 5, 9, 9, 5, 4])
        try cpu.compute(graph)

        XCTAssertEqual(result.floats(), [
            60, 55, 50, 110,
            90, 54, 54, 126,
            42, 29, 28, 64,
        ])
    }

    func testDeviceRegistry() throws {
        let devices = Device.all
        XCTAssertFalse(devices.isEmpty)

        let cpu = try XCTUnwrap(Device(type: .cpu))
        XCTAssertEqual(cpu.name, "CPU")
        XCTAssertEqual(cpu.type, .cpu)
        XCTAssertFalse(cpu.description.isEmpty)
        XCTAssertGreaterThan(cpu.memory.total, 0)

        let backend = try XCTUnwrap(cpu.makeBackend())
        XCTAssertEqual(backend.name, "CPU")
    }
}

final class TensorOpsTests: XCTestCase {
    private var cpu: Backend!
    private var graph: Graph!

    override func setUpWithError() throws {
        cpu = try XCTUnwrap(Backend(type: .cpu))
        graph = Graph()
    }

    override func tearDown() {
        graph = nil
        cpu = nil
    }

    /// Creates a leaf tensor allocated on the CPU backend, filled with
    /// `values` (a flat array of `shape.count` elements when given).
    private func tensor(_ values: [Float], type: TensorType = .f32, shape: [Int]? = nil) -> Tensor {
        let tensor = graph.tensor(type, shape: shape ?? [values.count])
        try! graph.allocTensors(on: cpu)
        tensor.copy(from: values)
        return tensor
    }

    private func tensor(int32 values: [Int32], shape: [Int]? = nil) -> Tensor {
        let tensor = graph.tensor(.i32, shape: shape ?? [values.count])
        try! graph.allocTensors(on: cpu)
        tensor.copy(from: values)
        return tensor
    }

    /// Computes the graph up to `tensor` on the CPU backend.
    private func run(_ tensor: Tensor) throws {
        graph.buildForwardExpand(tensor)
        try graph.allocTensors(on: cpu)
        try cpu.compute(graph)
    }

    /// Computes the graph for `tensor` and returns its values.
    private func evaluate(_ tensor: Tensor) throws -> [Float] {
        try run(tensor)
        return tensor.floats()
    }

    func testArithmetic() throws {
        let a = tensor([1, 2, 3, 4])
        let b = tensor([10, 20, 30, 40])

        XCTAssertEqual(try evaluate(a.add(b)), [11, 22, 33, 44])
        XCTAssertEqual(try evaluate(b.sub(a)), [9, 18, 27, 36])
        XCTAssertEqual(try evaluate(a.mul(b)), [10, 40, 90, 160])
        XCTAssertEqual(try evaluate(b.div(a)), [10, 10, 10, 10])
        XCTAssertEqual(try evaluate(a.scale(2)), [2, 4, 6, 8])
    }

    func testOperatorSugar() throws {
        let a = tensor([1, 2, 3, 4])
        let b = tensor([10, 20, 30, 40])

        XCTAssertEqual(try evaluate(a + b), [11, 22, 33, 44])
        XCTAssertEqual(try evaluate(b - a), [9, 18, 27, 36])
        XCTAssertEqual(try evaluate(a * b), [10, 40, 90, 160])
        XCTAssertEqual(try evaluate(b / a), [10, 10, 10, 10])
        XCTAssertEqual(try evaluate(a * 3), [3, 6, 9, 12])
        XCTAssertEqual(try evaluate(0.5 * b), [5, 10, 15, 20])
    }

    func testActivations() throws {
        XCTAssertEqual(try evaluate(tensor([-1, 0, 2, -3]).relu()), [0, 0, 2, 0])
        XCTAssertEqual(try evaluate(tensor([0]).sigmoid()), [0.5])
        XCTAssertEqual(try evaluate(tensor([0]).tanh()), [0])
        XCTAssertEqual(try evaluate(tensor([0]).gelu()), [0])
        XCTAssertEqual(try evaluate(tensor([0]).silu()), [0])
        XCTAssertEqual(try evaluate(tensor([0, 0, 0, 0]).softMax()), [0.25, 0.25, 0.25, 0.25])
    }

    func testNormalization() throws {
        // rms([3, 4]) = sqrt((9 + 16) / 2) = sqrt(12.5)
        let rms = Float(12.5).squareRoot()
        let normalized = try evaluate(tensor([3, 4]).rmsNorm(eps: 0))
        XCTAssertEqual(normalized[0], 3 / rms, accuracy: 1e-5)
        XCTAssertEqual(normalized[1], 4 / rms, accuracy: 1e-5)

        // norm() standardizes to zero mean and unit variance.
        let standardized = try evaluate(tensor([1, 3]).norm(eps: 0))
        XCTAssertEqual(standardized[0], -1, accuracy: 1e-5)
        XCTAssertEqual(standardized[1], 1, accuracy: 1e-5)
    }

    func testShapeOps() throws {
        // Row-major 2x3 matrix (3 columns, 2 rows): [1 2 3; 4 5 6]
        let m = tensor([1, 2, 3, 4, 5, 6], shape: [3, 2])

        XCTAssertEqual(m.transpose().shape, [2, 3])
        XCTAssertEqual(m.permute(1, 0, 2, 3).shape, [2, 3])
        XCTAssertEqual(m.reshape(6).shape, [6])
        XCTAssertEqual(m.reshape(2, 3).shape, [2, 3])
        XCTAssertEqual(m.reshape(like: tensor([0, 0, 0, 0, 0, 0])).shape, [6])

        // cont() materializes the transposed view into contiguous memory.
        let transposed = try evaluate(m.transpose().cont())
        XCTAssertEqual(transposed, [1, 4, 2, 5, 3, 6])
    }

    func testUnaryMath() throws {
        XCTAssertEqual(try evaluate(tensor([-1, 4]).neg()), [1, -4])
        XCTAssertEqual(try evaluate(tensor([-1, 4]).abs()), [1, 4])
        XCTAssertEqual(try evaluate(tensor([3, -2]).sqr()), [9, 4])
        XCTAssertEqual(try evaluate(tensor([9, 16]).sqrt()), [3, 4])
        XCTAssertEqual(try evaluate(tensor([1]).log()), [0])
        XCTAssertEqual(try evaluate(tensor([0]).exp()), [1])
        XCTAssertEqual(try evaluate(tensor([0]).sin()), [0])
        XCTAssertEqual(try evaluate(tensor([0]).cos()), [1])
        XCTAssertEqual(try evaluate(tensor([-2, 0.5, 3]).clamp(min: -1, max: 1)), [-1, 0.5, 1])
        XCTAssertEqual(try evaluate(tensor([1, 2]).dup()), [1, 2])
    }

    func testMoreActivations() throws {
        let leaky = try evaluate(tensor([-2, 3]).leakyRelu(negativeSlope: 0.1))
        XCTAssertEqual(leaky[0], -0.2, accuracy: 1e-6)
        XCTAssertEqual(leaky[1], 3)
        XCTAssertEqual(try evaluate(tensor([0]).geluQuick()), [0])
    }

    func testReductions() throws {
        XCTAssertEqual(try evaluate(tensor([1, 2, 3, 4]).sum()), [10])

        // [1 2 3; 4 5 6] as a 3-column, 2-row matrix.
        let m = tensor([1, 2, 3, 4, 5, 6], shape: [3, 2])

        let rowSums = m.sumRows()
        XCTAssertEqual(try evaluate(rowSums), [6, 15])
        XCTAssertEqual(rowSums.shape, [1, 2])
        XCTAssertEqual(try evaluate(m.mean()), [2, 5])

        let peaks = tensor([1, 5, 3, 9, 2, 4], shape: [3, 2])
        let indices = peaks.argmax()
        try run(indices)
        XCTAssertEqual(indices.type, .i32)
        XCTAssertEqual(indices.int32s(), [1, 0])
    }

    func testDataMovement() throws {
        // Embedding-style row gather.
        let table = tensor([1, 2, 3, 4, 5, 6], shape: [3, 2])
        let ids = tensor(int32: [1, 0, 1])
        let gathered = table.getRows(ids)
        XCTAssertEqual(try evaluate(gathered), [4, 5, 6, 1, 2, 3, 4, 5, 6])
        XCTAssertEqual(gathered.shape, [3, 3])

        let pattern = tensor([1, 2])
        let target = graph.tensor(.f32, 2, 2)
        XCTAssertEqual(try evaluate(pattern.repeated(like: target)), [1, 2, 1, 2])

        XCTAssertEqual(try evaluate(tensor([1, 2]).concat(tensor([3]), dim: 0)), [1, 2, 3])
    }

    func testCausalMask() throws {
        let scores = tensor([Float](repeating: 1, count: 9), shape: [3, 3])

        let masked = try evaluate(scores.diagMaskInf(nPast: 0))
        XCTAssertEqual(masked[0], 1)
        XCTAssertEqual(masked[1], -.infinity)
        XCTAssertEqual(masked[2], -.infinity)
        XCTAssertEqual(masked[3], 1)
        XCTAssertEqual(masked[4], 1)
        XCTAssertEqual(masked[5], -.infinity)
        XCTAssertEqual(Array(masked[6...]), [1, 1, 1])
    }

    func testRope() throws {
        // [head_dim = 4, n_head = 1, n_tokens = 2]
        let values: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        let q = tensor(values, shape: [4, 1, 2])

        // At position 0 the rotation is the identity.
        let zeros = tensor(int32: [0, 0])
        XCTAssertEqual(try evaluate(q.rope(zeros, nDims: 4)), values)

        // A non-zero position rotates the second token.
        let positions = tensor(int32: [0, 1])
        let rotated = try evaluate(q.rope(positions, nDims: 4))
        XCTAssertEqual(Array(rotated[0..<4]), [1, 2, 3, 4])
        XCTAssertNotEqual(Array(rotated[4...]), [5, 6, 7, 8])
    }

    func testViews() throws {
        let t = tensor([1, 2, 3, 4, 5, 6])
        let floatSize = MemoryLayout<Float>.size

        let slice = t.view(2, offset: 2 * floatSize)
        XCTAssertEqual(try evaluate(slice.cont()), [3, 4])

        // Top-left 2x2 block of [1 2 3; 4 5 6].
        let m = tensor([1, 2, 3, 4, 5, 6], shape: [3, 2])
        let block = m.view(2, 2, nb1: m.strides[1], offset: 0)
        XCTAssertEqual(try evaluate(block.cont()), [1, 2, 4, 5])
    }

    func testCopyAndSet() throws {
        // KV-cache pattern: write a token's vector into a slot of the cache.
        let cache = tensor([0, 0, 0, 0, 0, 0])
        let entry = tensor([7, 8])
        let slot = cache.view(2, offset: 2 * MemoryLayout<Float>.size)

        try run(entry.cpy(to: slot))
        XCTAssertEqual(cache.floats(), [0, 0, 7, 8, 0, 0])

        XCTAssertEqual(
            try evaluate(tensor([0, 0, 0, 0]).set(tensor([7, 8]), offset: MemoryLayout<Float>.size)),
            [0, 7, 8, 0])
    }

    func testCast() throws {
        let roundTripped = tensor([1.5, -2, 0.25]).cast(to: .f16).cast(to: .f32)
        XCTAssertEqual(try evaluate(roundTripped), [1.5, -2, 0.25])
    }

    func testSoftMaxExt() throws {
        XCTAssertEqual(try evaluate(tensor([0, 0]).softMaxExt()), [0.5, 0.5])

        // The mask sends the second logit to -inf.
        let mask = tensor([0, -.infinity])
        let masked = try evaluate(tensor([0, 0]).softMaxExt(mask: mask, scale: 1))
        XCTAssertEqual(masked, [1, 0])
    }

    func testSorting() throws {
        let indices = tensor([3, 1, 2]).argsort(order: .asc)
        try run(indices)
        XCTAssertEqual(indices.int32s(), [1, 2, 0])

        let top = tensor([1, 9, 5, 7]).topK(2)
        try run(top)
        XCTAssertEqual(Set(top.int32s()), [1, 3])
    }

    func testFlashAttention() throws {
        // One head, head_dim 4, 2 queries over 3 kv entries. Compare the
        // fused op against attention composed from primitive ops.
        let qValues: [Float] = [0.1, 0.2, 0.3, 0.4, -0.2, 0.1, 0.5, -0.1]
        let kValues: [Float] = [0.3, 0.1, -0.2, 0.4, 0.05, -0.3, 0.2, 0.1, -0.1, 0.25, 0.3, -0.2]
        let vValues: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        let scale = Float(0.5)

        let q = tensor(qValues, shape: [4, 2, 1])
        let k = tensor(kValues, shape: [4, 3, 1])
        let v = tensor(vValues, shape: [4, 3, 1])

        let fused = try evaluate(q.flashAttnExt(k: k, v: v, scale: scale))

        let reference = try evaluate(
            v.transpose().cont().mulMat(k.mulMat(q).softMaxExt(scale: scale)))

        XCTAssertEqual(fused.count, reference.count)
        for (f, r) in zip(fused, reference) {
            XCTAssertEqual(f, r, accuracy: 1e-4)
        }
    }

    func testQuantizedTypeProperties() {
        XCTAssertEqual(TensorType.q8_0.name, "q8_0")
        XCTAssertTrue(TensorType.q8_0.isQuantized)
        XCTAssertEqual(TensorType.q8_0.blockSize, 32)
        XCTAssertFalse(TensorType.q8_0.requiresImatrix)
        XCTAssertFalse(TensorType.f16.isQuantized)
        XCTAssertEqual(TensorType.q4_K.name, "q4_K")
        XCTAssertEqual(TensorType.f32.rowSize(10), 40)
    }

    func testF16RoundTrip() throws {
        let half = tensor([1.5, -2, 0.25, 100], type: .f16)
        XCTAssertEqual(half.floats(), [1.5, -2, 0.25, 100])
    }

    func testQuantizedRoundTrip() throws {
        let values = (0..<32).map { Float($0) / 16 - 1 }
        let quantized = tensor(values, type: .q8_0)

        for (dequantized, original) in zip(quantized.floats(), values) {
            XCTAssertEqual(dequantized, original, accuracy: 0.01)
        }
    }

    func testQuantizedMatMul() throws {
        // The llama.cpp pattern: quantized weights, f32 activations.
        let weights = (0..<64).map { Float($0 % 7) / 4 - 0.7 }
        let x = (0..<32).map { Float($0 % 5) / 3 - 0.6 }

        let wq = tensor(weights, type: .q8_0, shape: [32, 2])
        let wf = tensor(weights, shape: [32, 2])
        let input = tensor(x)

        let quantizedResult = try evaluate(wq.mulMat(input))
        let referenceResult = try evaluate(wf.mulMat(input))

        for (q, r) in zip(quantizedResult, referenceResult) {
            XCTAssertEqual(q, r, accuracy: 0.1)
        }
    }

    func testChainedGraph() throws {
        // (a · wᵀ + bias).relu() — one dense layer.
        let w = tensor([1, 0, 0, 1, -1, -1], shape: [2, 3])
        let x = tensor([3, 5])
        let bias = tensor([1, 1, 1])

        let out = try evaluate((w.mulMat(x) + bias).relu())
        XCTAssertEqual(out, [4, 6, 0])
    }
}
