"""Train an MLP on MNIST and export weights + eval data for the C++ engine.

Outputs under data/ (all gitignored, regeneratable by re-running this script):
  model.bin          trained weights (784-256-128-10, float32)
  test_images.bin    10k test images, float32 [0,1]
  test_labels.bin    10k test labels
  calib_images.bin   1000 calibration images (first 1000 training images)
  oracle_logits.bin  Python oracle float32 logits on the test set
  oracle_preds.bin   Python oracle top-1 predictions on the test set
  golden_images.bin  8 test images for the C++ end-to-end golden test
  golden_logits.bin  oracle logits for the golden images
  meta.json          dataset source (real-mnist vs SYNTHETIC) + train stats
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mnist  # noqa: E402
import model  # noqa: E402
import synthetic  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SEED = 42
HIDDEN = (256, 128)
N_CALIB = 1000


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    source = "real-mnist"
    try:
        mnist.download_mnist(DATA_DIR)
        x_train, y_train, x_test, y_test = mnist.load_mnist(DATA_DIR)
    except Exception as exc:  # noqa: BLE001 - documented fallback
        print(f"WARNING: MNIST download failed ({exc}); using SYNTHETIC fallback")
        source = "SYNTHETIC"
        x_train, y_train, x_test, y_test = synthetic.generate(seed=SEED)

    print(f"dataset: {source}  train={x_train.shape}  test={x_test.shape}")
    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN,
        activation="relu",
        solver="adam",
        max_iter=40,
        random_state=SEED,
        verbose=False,
    )
    t0 = time.time()
    clf.fit(x_train, y_train)
    train_secs = time.time() - t0
    print(f"trained in {train_secs:.1f}s, iters={clf.n_iter_}, loss={clf.loss_:.4f}")

    model.export_model(os.path.join(DATA_DIR, "model.bin"), clf.coefs_, clf.intercepts_)
    # Reload so the oracle uses the exact float32 weights the C++ engine sees.
    weights, biases = model.load_model(os.path.join(DATA_DIR, "model.bin"))

    logits = model.forward(x_test, weights, biases)
    preds = np.argmax(logits, axis=1).astype(np.uint8)
    oracle_acc = float(np.mean(preds == y_test))
    sk_acc = float(clf.score(x_test, y_test))
    print(f"oracle (float32 numpy) test accuracy: {oracle_acc:.4f}  sklearn: {sk_acc:.4f}")

    model.export_matrix(os.path.join(DATA_DIR, "test_images.bin"), x_test)
    model.export_labels(os.path.join(DATA_DIR, "test_labels.bin"), y_test)
    model.export_matrix(os.path.join(DATA_DIR, "calib_images.bin"), x_train[:N_CALIB])
    model.export_matrix(os.path.join(DATA_DIR, "oracle_logits.bin"), logits)
    model.export_labels(os.path.join(DATA_DIR, "oracle_preds.bin"), preds)
    model.export_matrix(os.path.join(DATA_DIR, "golden_images.bin"), x_test[:8])
    model.export_matrix(os.path.join(DATA_DIR, "golden_logits.bin"), logits[:8])

    meta = {
        "source": source,
        "seed": SEED,
        "hidden_layers": list(HIDDEN),
        "train_examples": int(x_train.shape[0]),
        "test_examples": int(x_test.shape[0]),
        "calib_examples": N_CALIB,
        "train_seconds": round(train_secs, 1),
        "sklearn_iters": int(clf.n_iter_),
        "oracle_test_accuracy": oracle_acc,
        "sklearn_test_accuracy": sk_acc,
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("export complete ->", DATA_DIR)


if __name__ == "__main__":
    main()
