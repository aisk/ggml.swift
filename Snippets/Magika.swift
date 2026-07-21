// File type detection with Google's Magika model — a port of ggml's
// examples/magika/main.cpp on the backend + scheduler path.
//
// Obtain the model as described in ggml's examples/magika/README.md
// (convert the pinned model.h5 with convert.py), then:
//
//     swift run Magika magika.gguf Package.swift README.md /bin/ls
import Foundation
import ggml

let labels = [
    "ai", "apk", "appleplist", "asm", "asp",
    "batch", "bmp", "bzip", "c", "cab",
    "cat", "chm", "coff", "crx", "cs",
    "css", "csv", "deb", "dex", "dmg",
    "doc", "docx", "elf", "emf", "eml",
    "epub", "flac", "gif", "go", "gzip",
    "hlp", "html", "ico", "ini", "internetshortcut",
    "iso", "jar", "java", "javabytecode", "javascript",
    "jpeg", "json", "latex", "lisp", "lnk",
    "m3u", "macho", "makefile", "markdown", "mht",
    "mp3", "mp4", "mscompress", "msi", "mum",
    "odex", "odp", "ods", "odt", "ogg",
    "outlook", "pcap", "pdf", "pebin", "pem",
    "perl", "php", "png", "postscript", "powershell",
    "ppt", "pptx", "python", "pythonbytecode", "rar",
    "rdf", "rpm", "rst", "rtf", "ruby",
    "rust", "scala", "sevenzip", "shell", "smali",
    "sql", "squashfs", "svg", "swf", "symlinktext",
    "tar", "tga", "tiff", "torrent", "ttf",
    "txt", "unknown", "vba", "wav", "webm",
    "webp", "winregistry", "wmf", "xar", "xls",
    "xlsb", "xlsx", "xml", "xpi", "xz",
    "yaml", "zip", "zlibstream",
]

// Model hyperparameters (fixed for the standard_v1 model).
let begSize = 512, midSize = 512, endSize = 512
let inputSize = begSize + midSize + endSize // 1536 bytes per file
let paddingToken = 256                      // one-hot depth is 257
let normEps: Float = 0.001

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    print("usage: \(arguments.first ?? "Magika") <model.gguf> <file1> [<file2> ...]")
    exit(1)
}
let modelPath = arguments[1]
let files = Array(arguments[2...])

// --- Load: metadata first, then the weights into a backend buffer.
guard let backend = Backend(type: .cpu) else {
    fatalError("no CPU backend available")
}

let gguf: GGUF
do {
    gguf = try GGUF(path: modelPath)
    try gguf.load(on: backend)
} catch {
    print("failed to load model from \(modelPath)")
    exit(1)
}

let weight = { (name: String) -> Tensor in
    guard let tensor = gguf.tensor(named: name) else {
        fatalError("tensor '\(name)' not found in \(modelPath)")
    }
    return tensor
}

// --- Input files: read each one up front, skipping anything that is not
// a readable regular file (directories, missing paths, ...).
var inputs: [(path: String, data: Data)] = []
for file in files {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: file), options: .alwaysMapped) else {
        print("skipping \(file): not a readable file")
        continue
    }
    inputs.append((file, data))
}
guard !inputs.isEmpty else {
    exit(1)
}

// --- Graph: the model sees 1536 one-hot encoded bytes per file.
let graph = Graph()

let input = graph.tensor(.f32, 257, inputSize, inputs.count) // one-hot
input.setInput()

var cur = weight("dense/kernel:0").within(graph)
    .mulMat(input)
    .add(weight("dense/bias:0"))                 // [128, 1536, nFiles]
    .gelu()
    .reshape(512, 384, inputs.count)
    .transpose().cont()                          // [384, 512, nFiles]
cur = cur.norm(eps: normEps)
    .mul(weight("layer_normalization/gamma:0"))
    .add(weight("layer_normalization/beta:0"))
    .transpose().cont()                          // [512, 384, nFiles]
cur = weight("dense_1/kernel:0").within(graph)
    .mulMat(cur)
    .add(weight("dense_1/bias:0"))               // [256, 384, nFiles]
    .gelu()
cur = weight("dense_2/kernel:0").within(graph)
    .mulMat(cur)
    .add(weight("dense_2/bias:0"))               // [256, 384, nFiles]
    .gelu()
cur = cur.transpose().cont()                     // [384, 256, nFiles]
    .pool1d(.max, k0: 384, s0: 384)              // global max pooling
    .reshape(256, inputs.count)
cur = cur.norm(eps: normEps)
    .mul(weight("layer_normalization_1/gamma:0"))
    .add(weight("layer_normalization_1/beta:0"))
let probs = weight("target_label/kernel:0").within(graph)
    .mulMat(cur)
    .add(weight("target_label/bias:0"))
    .softMax()                                   // [nLabels, nFiles]
probs.setOutput()

graph.buildForwardExpand(probs)

let scheduler = Scheduler(backends: [backend])
scheduler.allocGraph(graph)

// --- Input: 512 bytes from the beginning, middle and end of each file,
// padded with the padding token, then one-hot encoded.
var oneHot = [Float](repeating: 0, count: input.elementCount)
for (i, (_, data)) in inputs.enumerated() {
    var tokens = [Int](repeating: paddingToken, count: inputSize)

    let beg = data.prefix(begSize)
    for (j, byte) in beg.enumerated() { // pad at the end
        tokens[j] = Int(byte)
    }

    let midOffset = max(0, (data.count - midSize) / 2)
    let mid = data.dropFirst(midOffset).prefix(midSize)
    for (j, byte) in mid.enumerated() { // pad at both ends
        tokens[begSize + midSize / 2 - mid.count / 2 + j] = Int(byte)
    }

    let end = data.suffix(endSize)
    for (j, byte) in end.enumerated() { // pad at the beginning
        tokens[inputSize - end.count + j] = Int(byte)
    }

    for (j, token) in tokens.enumerated() {
        oneHot[257 * (inputSize * i + j) + token] = 1
    }
}
try input.copy(from: oneHot)

try scheduler.compute(graph)

// --- Output: top 5 labels per file.
let allProbs = probs.floats()
for (i, (file, _)) in inputs.enumerated() {
    let fileProbs = allProbs[labels.count * i..<labels.count * (i + 1)]
    let top = zip(labels, fileProbs).sorted { $0.1 > $1.1 }.prefix(5)
    let summary = top.map { String(format: "%@ (%.2f%%)", $0.0, $0.1 * 100) }
    print("\(file): \(summary.joined(separator: " "))")
}
