import CGGML

/// Entry point for the Swift object-oriented wrapper around vendored ggml.
/// The raw C module remains an implementation detail.
public enum GGML {
    /// The exact upstream ggml release compiled into this package.
    public static let version = String(cString: ggml_version())

    /// The upstream source revision used for the vendored release.
    public static let commit = String(cString: ggml_commit())
}
