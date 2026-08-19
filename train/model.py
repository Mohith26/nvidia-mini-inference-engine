"""Float32 numpy reference model (the Python oracle) + binary export format.

Binary formats (all little-endian):
  model.bin : u32 magic (0x54464F52 "TFOR") | u32 num_layers |
              per layer: u32 in | u32 out | f32 W[in*out] row-major | f32 b[out]
  matrix    : u32 N | u32 D | f32 data[N*D]   (images, logits)
  labels    : u32 N | u8 data[N]              (labels, predictions)
"""
import struct

import numpy as np

MAGIC = 0x54464F52  # "TFOR"


def forward(x, weights, biases):
    """Float32 reference forward pass; ReLU between layers, raw logits out."""
    h = np.ascontiguousarray(x, dtype=np.float32)
    n = len(weights)
    for i, (w, b) in enumerate(zip(weights, biases)):
        h = h @ w.astype(np.float32) + b.astype(np.float32)
        if i < n - 1:
            h = np.maximum(h, np.float32(0.0))
    return h


def predict(x, weights, biases):
    return np.argmax(forward(x, weights, biases), axis=1).astype(np.uint8)


def export_model(path, weights, biases):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", MAGIC, len(weights)))
        for w, b in zip(weights, biases):
            w32 = np.ascontiguousarray(w, dtype=np.float32)
            b32 = np.ascontiguousarray(b, dtype=np.float32)
            f.write(struct.pack("<II", w32.shape[0], w32.shape[1]))
            f.write(w32.tobytes())
            f.write(b32.tobytes())


def load_model(path):
    with open(path, "rb") as f:
        raw = f.read()
    magic, n_layers = struct.unpack_from("<II", raw, 0)
    if magic != MAGIC:
        raise ValueError(f"bad model magic: {magic:#x}")
    off = 8
    weights, biases = [], []
    for _ in range(n_layers):
        d_in, d_out = struct.unpack_from("<II", raw, off)
        off += 8
        w = np.frombuffer(raw, dtype="<f4", count=d_in * d_out, offset=off)
        off += 4 * d_in * d_out
        b = np.frombuffer(raw, dtype="<f4", count=d_out, offset=off)
        off += 4 * d_out
        weights.append(w.reshape(d_in, d_out).copy())
        biases.append(b.copy())
    return weights, biases


def export_matrix(path, arr):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", a.shape[0], a.shape[1]))
        f.write(a.tobytes())


def load_matrix(path):
    with open(path, "rb") as f:
        raw = f.read()
    n, d = struct.unpack_from("<II", raw, 0)
    return np.frombuffer(raw, dtype="<f4", count=n * d, offset=8).reshape(n, d).copy()


def export_labels(path, y):
    a = np.ascontiguousarray(y, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", a.shape[0]))
        f.write(a.tobytes())


def load_labels(path):
    with open(path, "rb") as f:
        raw = f.read()
    (n,) = struct.unpack_from("<I", raw, 0)
    return np.frombuffer(raw, dtype=np.uint8, count=n, offset=4).copy()
