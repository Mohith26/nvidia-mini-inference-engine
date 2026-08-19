"""MNIST download, checksum verification, and IDX parsing utilities."""
import gzip
import hashlib
import os
import struct
import urllib.request

import numpy as np

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

# SHA-256 checksums of the canonical gzipped IDX files.
CHECKSUMS = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def download_mnist(data_dir):
    """Download all four MNIST files, trying each mirror; verify checksums.

    Raises RuntimeError if every mirror fails for any file.
    """
    os.makedirs(data_dir, exist_ok=True)
    for fname, want in CHECKSUMS.items():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path) and sha256_file(path) == want:
            continue
        ok = False
        for mirror in MIRRORS:
            try:
                urllib.request.urlretrieve(mirror + fname, path)
                if sha256_file(path) == want:
                    ok = True
                    break
                print(f"checksum mismatch for {fname} from {mirror}")
            except Exception as exc:  # noqa: BLE001 - report and try next mirror
                print(f"download failed for {fname} from {mirror}: {exc}")
        if not ok:
            raise RuntimeError(f"could not download {fname} from any mirror")


def parse_idx_images(raw):
    """Parse an (uncompressed) IDX3 image buffer -> uint8 array [N, rows*cols]."""
    magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"bad IDX3 magic: {magic}")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows * cols)


def parse_idx_labels(raw):
    """Parse an (uncompressed) IDX1 label buffer -> uint8 array [N]."""
    magic, n = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"bad IDX1 magic: {magic}")
    return np.frombuffer(raw, dtype=np.uint8, offset=8)[:n]


def load_mnist(data_dir):
    """Return (X_train, y_train, X_test, y_test); X float32 in [0,1]."""
    def gz(name):
        with gzip.open(os.path.join(data_dir, name), "rb") as f:
            return f.read()

    x_train = parse_idx_images(gz("train-images-idx3-ubyte.gz"))
    y_train = parse_idx_labels(gz("train-labels-idx1-ubyte.gz"))
    x_test = parse_idx_images(gz("t10k-images-idx3-ubyte.gz"))
    y_test = parse_idx_labels(gz("t10k-labels-idx1-ubyte.gz"))
    return (
        (x_train.astype(np.float32) / 255.0),
        y_train.astype(np.uint8),
        (x_test.astype(np.float32) / 255.0),
        y_test.astype(np.uint8),
    )
