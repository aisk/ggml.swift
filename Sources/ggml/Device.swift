import CGGML

/// An entry in ggml's device registry describing a physical or logical
/// compute device. Mirrors `ggml_backend_dev_t` 1:1.
///
/// Devices are owned by the registry and never freed; this type is a
/// lightweight handle. Use ``all`` to enumerate what is available (after
/// ``Backend/loadAll()`` for dynamically loaded backends) and
/// ``makeBackend(params:)`` to create a backend on a specific device.
public struct Device: Equatable {
    let rawValue: OpaquePointer

    /// All registered devices. Mirrors `ggml_backend_dev_count`/`_get`.
    public static var all: [Device] {
        (0..<ggml_backend_dev_count()).map { Device(rawValue: ggml_backend_dev_get($0)) }
    }

    /// The first registered device of the given type.
    /// Mirrors `ggml_backend_dev_by_type`.
    public init?(type: Backend.DeviceType) {
        guard let raw = ggml_backend_dev_by_type(type.cValue) else {
            return nil
        }
        self.rawValue = raw
    }

    init(rawValue: OpaquePointer) {
        self.rawValue = rawValue
    }

    /// Short name, e.g. `"CPU"`. Mirrors `ggml_backend_dev_name`.
    public var name: String {
        String(cString: ggml_backend_dev_name(rawValue))
    }

    /// Human-readable description, e.g. the CPU model.
    /// Mirrors `ggml_backend_dev_description`.
    public var description: String {
        String(cString: ggml_backend_dev_description(rawValue))
    }

    /// Device category. Mirrors `ggml_backend_dev_type`.
    public var type: Backend.DeviceType {
        Backend.DeviceType(cValue: ggml_backend_dev_type(rawValue))
    }

    /// Free and total device memory in bytes. Mirrors `ggml_backend_dev_memory`.
    public var memory: (free: Int, total: Int) {
        var free = 0, total = 0
        ggml_backend_dev_memory(rawValue, &free, &total)
        return (free, total)
    }

    /// Creates a backend instance on this device. Mirrors `ggml_backend_dev_init`.
    public func makeBackend(params: String? = nil) -> Backend? {
        ggml_backend_dev_init(rawValue, params).map(Backend.init(rawValue:))
    }
}

extension Backend.DeviceType {
    init(cValue: ggml_backend_dev_type) {
        switch cValue {
        case GGML_BACKEND_DEVICE_TYPE_CPU: self = .cpu
        case GGML_BACKEND_DEVICE_TYPE_GPU: self = .gpu
        case GGML_BACKEND_DEVICE_TYPE_IGPU: self = .igpu
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: self = .accel
        default: self = .meta
        }
    }
}
