import CGGML

/// A compute device (CPU, GPU, accelerator) that can allocate tensor buffers
/// and execute graphs. Mirrors `ggml_backend_t` 1:1.
public final class Backend {
    let rawValue: ggml_backend_t

    init(rawValue: ggml_backend_t) {
        self.rawValue = rawValue
    }

    deinit {
        ggml_backend_free(rawValue)
    }

    /// Device categories. Mirrors `ggml_backend_dev_type`.
    public enum DeviceType: Sendable {
        /// CPU device using system memory.
        case cpu
        /// GPU device using dedicated memory.
        case gpu
        /// Integrated GPU device using host memory.
        case igpu
        /// Accelerator devices intended to be used together with the CPU
        /// backend (e.g. BLAS or AMX).
        case accel
        /// Meta devices wrapping other devices.
        case meta

        var cValue: ggml_backend_dev_type {
            switch self {
            case .cpu: return GGML_BACKEND_DEVICE_TYPE_CPU
            case .gpu: return GGML_BACKEND_DEVICE_TYPE_GPU
            case .igpu: return GGML_BACKEND_DEVICE_TYPE_IGPU
            case .accel: return GGML_BACKEND_DEVICE_TYPE_ACCEL
            case .meta: return GGML_BACKEND_DEVICE_TYPE_META
            }
        }
    }

    /// Loads all backends that are available as dynamic libraries.
    /// Mirrors `ggml_backend_load_all`.
    public static func loadAll() {
        ggml_backend_load_all()
    }

    /// Initializes the best available backend (GPU first, CPU as fallback).
    /// Mirrors `ggml_backend_init_best`.
    public static func best() -> Backend? {
        ggml_backend_init_best().map(Backend.init(rawValue:))
    }

    /// Initializes the first available device of the given type.
    /// Mirrors `ggml_backend_init_by_type`.
    public convenience init?(type: DeviceType, params: String? = nil) {
        guard let backend = ggml_backend_init_by_type(type.cValue, params) else {
            return nil
        }
        self.init(rawValue: backend)
    }

    /// Initializes a device by its registry name (e.g. `"CPU"`).
    /// Mirrors `ggml_backend_init_by_name`.
    public convenience init?(name: String, params: String? = nil) {
        guard let backend = ggml_backend_init_by_name(name, params) else {
            return nil
        }
        self.init(rawValue: backend)
    }

    /// Name of the backend, e.g. `"CPU"`. Mirrors `ggml_backend_name`.
    public var name: String {
        String(cString: ggml_backend_name(rawValue))
    }

    /// Executes a graph directly on this backend. All graph tensors must
    /// already be allocated in buffers this backend can use (e.g. via
    /// ``Graph/allocTensors(on:)``). For multi-backend execution or
    /// automatic allocation use a ``Scheduler`` instead.
    /// Mirrors `ggml_backend_graph_compute`.
    public func compute(_ graph: Graph) throws {
        let status = Status(cValue: ggml_backend_graph_compute(rawValue, graph.rawValue))
        guard status == .success else {
            throw GGMLError.computeFailed(status)
        }
    }

    /// Sets the number of threads a CPU backend computes with. Must only
    /// be called on a CPU backend. Mirrors `ggml_backend_cpu_set_n_threads`.
    public func cpuSetNThreads(_ nThreads: Int) {
        ggml_backend_cpu_set_n_threads(rawValue, Int32(nThreads))
    }
}

extension Backend: CustomStringConvertible {
    public var description: String { name }
}
