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
