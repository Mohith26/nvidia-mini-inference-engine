"""pytest tests for the trainer/exporter (pure functions; no MNIST download)."""
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
import mnist  # noqa: E402
import model  # noqa: E402
import synthetic  # noqa: E402


def _idx3(images):
    n, rows, cols = images.shape
    return struct.pack(">IIII", 2051, n, rows, cols) + images.astype(np.uint8).tobytes()


def _idx1(labels):
    return struct.pack(">II", 2049, len(labels)) + labels.astype(np.uint8).tobytes()


def test_idx_image_parse():
    imgs = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    out = mnist.parse_idx_images(_idx3(imgs))
    assert out.shape == (2, 9)
    assert np.array_equal(out, imgs.reshape(2, 9))


def test_idx_image_bad_magic_raises():
    raw = struct.pack(">IIII", 1234, 1, 2, 2) + bytes(4)
    with pytest.raises(ValueError):
        mnist.parse_idx_images(raw)


def test_idx_label_parse():
    labels = np.array([3, 1, 4, 1, 5], dtype=np.uint8)
    out = mnist.parse_idx_labels(_idx1(labels))
    assert np.array_equal(out, labels)


def test_sha256_known_vector():
    assert (
        mnist.sha256_bytes(b"abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_model_export_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    weights = [rng.normal(size=(5, 4)).astype(np.float32), rng.normal(size=(4, 3)).astype(np.float32)]
    biases = [rng.normal(size=4).astype(np.float32), rng.normal(size=3).astype(np.float32)]
    path = str(tmp_path / "m.bin")
    model.export_model(path, weights, biases)
    w2, b2 = model.load_model(path)
    for a, b in zip(weights, w2):
        assert np.array_equal(a, b)
    for a, b in zip(biases, b2):
        assert np.array_equal(a, b)


def test_matrix_export_roundtrip(tmp_path):
    arr = np.random.default_rng(1).normal(size=(7, 5)).astype(np.float32)
    path = str(tmp_path / "x.bin")
    model.export_matrix(path, arr)
    assert np.array_equal(model.load_matrix(path), arr)


def test_labels_export_roundtrip(tmp_path):
    y = np.array([0, 9, 5, 2], dtype=np.uint8)
    path = str(tmp_path / "y.bin")
    model.export_labels(path, y)
    assert np.array_equal(model.load_labels(path), y)


def test_forward_hand_computed():
    # x=[1,2], W1=[[1,0],[0,1]], b1=[0,-5] -> pre=[1,-3] -> relu=[1,0]
    # W2=[[2],[7]], b2=[1] -> logits=[3]
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    weights = [np.eye(2, dtype=np.float32), np.array([[2.0], [7.0]], dtype=np.float32)]
    biases = [np.array([0.0, -5.0], dtype=np.float32), np.array([1.0], dtype=np.float32)]
    out = model.forward(x, weights, biases)
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(3.0)


def test_forward_relu_only_between_layers():
    # Final layer must NOT be ReLU-clipped: negative logits must survive.
    x = np.array([[1.0]], dtype=np.float32)
    weights = [np.array([[1.0]], dtype=np.float32), np.array([[-4.0]], dtype=np.float32)]
    biases = [np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)]
    out = model.forward(x, weights, biases)
    assert out[0, 0] == pytest.approx(-4.0)


def test_predict_argmax():
    x = np.array([[1.0, 0.0]], dtype=np.float32)
    weights = [np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float32)]
    biases = [np.zeros(3, dtype=np.float32)]
    assert model.predict(x, weights, biases)[0] == 2


def test_synthetic_seeded_reproducible():
    a = synthetic.generate(seed=7, n_train=50, n_test=20)
    b = synthetic.generate(seed=7, n_train=50, n_test=20)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_synthetic_shapes_and_ranges():
    x_train, y_train, x_test, y_test = synthetic.generate(seed=1, n_train=30, n_test=10)
    assert x_train.shape == (30, 784) and x_test.shape == (10, 784)
    assert x_train.min() >= 0.0 and x_train.max() <= 1.0
    assert y_train.min() >= 0 and y_train.max() <= 9 and y_test.shape == (10,)
