# ggml.swift

A Swift object-oriented wrapper around ggml that vendors the upstream C/C++ sources directly.

## Vendored ggml

- Release: [`v0.16.0`](https://github.com/ggml-org/ggml/releases/tag/v0.16.0)
- Source commit: [`524f974bb21a1013408f76d71c15732482c0c3fe`](https://github.com/ggml-org/ggml/commit/524f974bb21a1013408f76d71c15732482c0c3fe)
- Tag object: `eb7f30b8a58f1fad4103bbbf06da4c411517d2ba`
- License: MIT; see [`Sources/CGGML/LICENSE`](Sources/CGGML/LICENSE).

`Sources/CGGML` contains the vendored ggml source tree. The source is committed directly instead of using a git submodule, so downstream Swift Package Manager clients receive it without an additional submodule initialization step.
