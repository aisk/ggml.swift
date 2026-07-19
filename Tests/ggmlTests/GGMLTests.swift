import XCTest
@testable import ggml

final class GGMLTests: XCTestCase {
    func testVendoredGGMLReleaseIsAvailable() {
        XCTAssertEqual(GGML.version, "0.16.0")
        XCTAssertEqual(GGML.commit, "524f974bb21a1013408f76d71c15732482c0c3fe")
    }
}

/// Port of ggml's `examples/simple/simple-ctx.cpp`.
final class SimpleContextTests: XCTestCase {
    func testMatMul() throws {
        let rowsA = 4, colsA = 2
        let matrixA: [Float] = [
            2, 8,
            5, 1,
            4, 2,
            8, 6,
        ]

        let rowsB = 3, colsB = 2
        // Transpose([10, 9, 5,
        //            5, 9, 4])
        let matrixB: [Float] = [
            10, 5,
            9, 9,
            5, 4,
        ]

        var contextSize = 0
        contextSize += rowsA * colsA * TensorType.f32.size
        contextSize += rowsB * colsB * TensorType.f32.size
        contextSize += 2 * Context.tensorOverhead
        contextSize += Context.graphOverhead
        contextSize += 1024 // some overhead

        let context = try Context(memorySize: contextSize)

        let a = context.tensor(.f32, colsA, rowsA)
        let b = context.tensor(.f32, colsB, rowsB)
        a.copy(from: matrixA)
        b.copy(from: matrixB)

        let graph = context.graph()
        let result = a.mulMat(b)
        graph.buildForwardExpand(result)
        try graph.compute(threads: 1)

        XCTAssertEqual(graph.output?.rawValue, result.rawValue)
        XCTAssertEqual(result.type, .f32)
        XCTAssertEqual(result.shape, [rowsA, rowsB])
        XCTAssertEqual(result.floats(), [
            60, 55, 50, 110,
            90, 54, 54, 126,
            42, 29, 28, 64,
        ])
    }

    func testTensorMetadata() throws {
        let context = try Context(memorySize: 16 * 1024)

        let tensor = context.tensor(.f32, 3, 2)
        tensor.name = "weights"

        XCTAssertEqual(tensor.type, .f32)
        XCTAssertEqual(tensor.type.name, "f32")
        XCTAssertEqual(tensor.dimensions, 2)
        XCTAssertEqual(tensor.shape, [3, 2])
        XCTAssertEqual(tensor.elementCount, 6)
        XCTAssertEqual(tensor.byteCount, 6 * MemoryLayout<Float>.size)
        XCTAssertEqual(tensor.name, "weights")
        XCTAssertGreaterThan(context.usedMemory, 0)
    }
}

/// Port of ggml's `examples/simple/simple-backend.cpp`.
final class SimpleBackendTests: XCTestCase {
    func testMatMulViaScheduler() throws {
        Backend.loadAll()

        let best = try XCTUnwrap(Backend.best())
        let cpu = try XCTUnwrap(Backend(type: .cpu))
        let scheduler = Scheduler(backends: [best, cpu])

        // The context only holds tensor metadata and the graph; tensor data
        // is allocated in backend buffers by the scheduler.
        let contextSize = Context.tensorOverhead * Graph.defaultSize + Context.graphOverhead
        let context = try Context(memorySize: contextSize, noAlloc: true)

        let a = context.tensor(.f32, 2, 4)
        let b = context.tensor(.f32, 2, 3)
        let result = a.mulMat(b)

        let graph = context.graph()
        graph.buildForwardExpand(result)

        scheduler.reset()
        XCTAssertTrue(scheduler.allocGraph(graph))
        XCTAssertTrue(a.isBackendAllocated)

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

    override func setUp() {
        path = FileManager.default.temporaryDirectory
            .appendingPathComponent("ggml-swift-test-\(UUID().uuidString).gguf").path
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: path)
    }

    private func writeModel() throws {
        let context = try Context(memorySize: 1024 * 1024)

        // A 4 -> 3 -> 2 fully connected network with hand-picked weights.
        let fc1Weight = context.tensor(.f32, 4, 3)
        fc1Weight.name = "fc1.weight"
        fc1Weight.copy(from: [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 1,
        ])
        let fc1Bias = context.tensor(.f32, 3)
        fc1Bias.name = "fc1.bias"
        fc1Bias.copy(from: [0, -3, 0])

        let fc2Weight = context.tensor(.f32, 3, 2)
        fc2Weight.name = "fc2.weight"
        fc2Weight.copy(from: [
            1, 1, 1,
            -1, 0, 1,
        ])
        let fc2Bias = context.tensor(.f32, 2)
        fc2Bias.name = "fc2.bias"
        fc2Bias.copy(from: [0.5, -0.5])

        let gguf = GGUF()
        gguf.set("mnist-fc", forKey: "general.architecture")
        gguf.set(3, forKey: "test.epochs")
        gguf.set(0.05, forKey: "test.learning_rate")
        gguf.set(true, forKey: "test.trained")
        gguf.add(fc1Weight)
        gguf.add(fc1Bias)
        gguf.add(fc2Weight)
        gguf.add(fc2Bias)
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

        let fc1Weight = try XCTUnwrap(gguf.tensor(named: "fc1.weight"))
        XCTAssertEqual(fc1Weight.shape, [4, 3])
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
        let fc1Weight = try XCTUnwrap(gguf.tensor(named: "fc1.weight"))
        let fc1Bias = try XCTUnwrap(gguf.tensor(named: "fc1.bias"))
        let fc2Weight = try XCTUnwrap(gguf.tensor(named: "fc2.weight"))
        let fc2Bias = try XCTUnwrap(gguf.tensor(named: "fc2.bias"))

        // Build and run the eval graph in a separate compute context; the
        // loaded weights participate via within(_:).
        let compute = try Context(memorySize: 1024 * 1024)
        let x = compute.tensor(.f32, 4)
        x.copy(from: [1, 2, 3, 4])

        let hidden = fc1Weight.within(compute).mulMat(x).add(fc1Bias).relu()
        let logits = fc2Weight.within(compute).mulMat(hidden).add(fc2Bias)

        let graph = compute.graph()
        graph.buildForwardExpand(logits)
        try graph.compute()

        // fc1: [1, 2, 7] + [0, -3, 0] -> relu -> [1, 0, 7]
        // fc2: [8, 6] + [0.5, -0.5] -> [8.5, 5.5]
        XCTAssertEqual(logits.floats(), [8.5, 5.5])
    }

    func testLoadFailureThrows() {
        XCTAssertThrowsError(try GGUF(path: "/nonexistent/model.gguf")) { error in
            XCTAssertEqual(error as? GGMLError,
                           .ggufLoadFailed(path: "/nonexistent/model.gguf"))
        }
    }
}

final class TensorOpsTests: XCTestCase {
    private var context: Context!

    override func setUpWithError() throws {
        context = try Context(memorySize: 16 * 1024 * 1024)
    }

    override func tearDown() {
        context = nil
    }

    /// Computes the graph for `tensor` on the CPU and returns its values.
    private func evaluate(_ tensor: Tensor) throws -> [Float] {
        let graph = context.graph()
        graph.buildForwardExpand(tensor)
        try graph.compute()
        return tensor.floats()
    }

    private func tensor(_ values: [Float]) -> Tensor {
        let t = context.tensor(.f32, values.count)
        t.copy(from: values)
        return t
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
        let m = context.tensor(.f32, 3, 2)
        m.copy(from: [1, 2, 3, 4, 5, 6])

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
        let m = context.tensor(.f32, 3, 2)
        m.copy(from: [1, 2, 3, 4, 5, 6])

        let rowSums = m.sumRows()
        XCTAssertEqual(try evaluate(rowSums), [6, 15])
        XCTAssertEqual(rowSums.shape, [1, 2])
        XCTAssertEqual(try evaluate(m.mean()), [2, 5])

        let peaks = context.tensor(.f32, 3, 2)
        peaks.copy(from: [1, 5, 3, 9, 2, 4])
        let indices = peaks.argmax()
        let graph = context.graph()
        graph.buildForwardExpand(indices)
        try graph.compute()
        XCTAssertEqual(indices.type, .i32)
        XCTAssertEqual(indices.int32s(), [1, 0])
    }

    func testDataMovement() throws {
        // Embedding-style row gather.
        let table = context.tensor(.f32, 3, 2)
        table.copy(from: [1, 2, 3, 4, 5, 6])
        let ids = context.tensor(.i32, 3)
        ids.copy(from: [1, 0, 1] as [Int32])
        let gathered = table.getRows(ids)
        XCTAssertEqual(try evaluate(gathered), [4, 5, 6, 1, 2, 3, 4, 5, 6])
        XCTAssertEqual(gathered.shape, [3, 3])

        let pattern = tensor([1, 2])
        let target = context.tensor(.f32, 2, 2)
        XCTAssertEqual(try evaluate(pattern.repeated(like: target)), [1, 2, 1, 2])

        XCTAssertEqual(try evaluate(tensor([1, 2]).concat(tensor([3]), dim: 0)), [1, 2, 3])
    }

    func testCausalMask() throws {
        let scores = context.tensor(.f32, 3, 3)
        scores.copy(from: [Float](repeating: 1, count: 9))

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
        let q = context.tensor(.f32, 4, 1, 2)
        let values: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
        q.copy(from: values)

        // At position 0 the rotation is the identity.
        let zeros = context.tensor(.i32, 2)
        zeros.copy(from: [0, 0] as [Int32])
        XCTAssertEqual(try evaluate(q.rope(zeros, nDims: 4)), values)

        // A non-zero position rotates the second token.
        let positions = context.tensor(.i32, 2)
        positions.copy(from: [0, 1] as [Int32])
        let rotated = try evaluate(q.rope(positions, nDims: 4))
        XCTAssertEqual(Array(rotated[0..<4]), [1, 2, 3, 4])
        XCTAssertNotEqual(Array(rotated[4...]), [5, 6, 7, 8])
    }

    func testChainedGraph() throws {
        // (a · wᵀ + bias).relu() — one dense layer.
        let w = context.tensor(.f32, 2, 3)
        w.copy(from: [1, 0, 0, 1, -1, -1])
        let x = context.tensor(.f32, 2)
        x.copy(from: [3, 5])
        let bias = tensor([1, 1, 1])

        let out = try evaluate((w.mulMat(x) + bias).relu())
        XCTAssertEqual(out, [4, 6, 0])
    }
}
