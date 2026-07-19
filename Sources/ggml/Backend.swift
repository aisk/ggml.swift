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
}

extension Backend: CustomStringConvertible {
    public var description: String { name }
}
