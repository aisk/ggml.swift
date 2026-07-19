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
