// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "ggml.swift",
    products: [
        .library(name: "ggml", targets: ["ggml"]),
    ],
    targets: [
        // Vendored C/C++ sources are part of this package checkout, so a
        // downstream SwiftPM client does not need to initialize a submodule.
        .target(
            name: "CGGML",
            path: "Sources/CGGML",
            sources: [
                "src/ggml.c", "src/ggml.cpp", "src/ggml-alloc.c",
                "src/ggml-backend.cpp", "src/ggml-backend-meta.cpp",
                "src/ggml-backend-reg.cpp", "src/ggml-backend-dl.cpp",
                "src/ggml-opt.cpp",
                "src/ggml-threading.cpp", "src/ggml-quants.c", "src/gguf.cpp",
                "src/ggml-cpu/ggml-cpu.c", "src/ggml-cpu/ggml-cpu.cpp",
                "src/ggml-cpu/repack.cpp", "src/ggml-cpu/hbm.cpp",
                "src/ggml-cpu/quants.c", "src/ggml-cpu/traits.cpp",
                "src/ggml-cpu/amx/amx.cpp", "src/ggml-cpu/amx/mmq.cpp",
                "src/ggml-cpu/binary-ops.cpp", "src/ggml-cpu/unary-ops.cpp",
                "src/ggml-cpu/vec.cpp", "src/ggml-cpu/ops.cpp",
            ],
            publicHeadersPath: "include",
            cSettings: [
                .define("_GNU_SOURCE"),
                .define("GGML_USE_CPU"), .define("GGML_CPU_GENERIC"),
                .define("GGML_VERSION", to: "\"0.16.0\""),
                .define("GGML_COMMIT", to: "\"524f974bb21a1013408f76d71c15732482c0c3fe\""),
                .headerSearchPath("include"),
                .headerSearchPath("src"), .headerSearchPath("src/ggml-cpu"),
            ],
            cxxSettings: [
                .define("GGML_USE_CPU"), .define("GGML_CPU_GENERIC"),
                .define("GGML_VERSION", to: "\"0.16.0\""),
                .define("GGML_COMMIT", to: "\"524f974bb21a1013408f76d71c15732482c0c3fe\""),
                .headerSearchPath("include"),
                .headerSearchPath("src"), .headerSearchPath("src/ggml-cpu"),
            ],
            linkerSettings: [
                .linkedLibrary("dl", .when(platforms: [.linux])),
                .linkedLibrary("pthread", .when(platforms: [.linux])),
            ]
        ),
        .target(name: "ggml", dependencies: ["CGGML"]),
        .testTarget(name: "ggmlTests", dependencies: ["ggml"]),
    ],
    cLanguageStandard: .c11,
    cxxLanguageStandard: .cxx17
)
