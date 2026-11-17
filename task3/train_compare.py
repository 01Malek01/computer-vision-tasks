"""Task 3: custom CNN vs. pretrained EfficientNetB0 on MNIST.

Both models are trained/evaluated on exactly the same balanced subset of MNIST,
so the comparison is fair. Results (metrics, curves, confusion matrices) are
written to task3/results/.

Usage:
    python task3/train_compare.py
    python task3/train_compare.py --train-per-class 500 --cnn-epochs 3 --eff-epochs 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow import keras
from tensorflow.keras import applications, layers

SEED = 42
CLASS_NAMES = [str(i) for i in range(10)]
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-per-class", type=int, default=1000,
                   help="training images per digit class (0 = use all 6000)")
    p.add_argument("--test-per-class", type=int, default=200,
                   help="test images per digit class")
    p.add_argument("--cnn-epochs", type=int, default=6)
    p.add_argument("--eff-epochs", type=int, default=4,
                   help="epochs for the EfficientNet classification head")
    p.add_argument("--eff-finetune-epochs", type=int, default=0,
                   help="optional fine-tuning epochs for the last EfficientNet layers")
    p.add_argument("--eff-image-size", type=int, default=32,
                   help="MNIST is upscaled to this size for EfficientNetB0")
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(seed)


def balanced_subset(x: np.ndarray, y: np.ndarray, per_class: int | None,
                    seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Take at most `per_class` images of every digit (per_class=None -> all)."""
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in range(10):
        indices = np.flatnonzero(y == cls)
        rng.shuffle(indices)
        chosen.append(indices if not per_class else indices[:per_class])
    idx = rng.permutation(np.concatenate(chosen))
    return x[idx], y[idx]


def cnn_dataset(x: np.ndarray, y: np.ndarray, batch_size: int,
                train: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if train:
        ds = ds.shuffle(len(x), reshuffle_each_iteration=True)
    # Keep raw uint8 pixels here; the model's Rescaling layer normalizes them.
    ds = ds.map(lambda img, label: (tf.cast(img, tf.float32), label),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def efficientnet_dataset(x: np.ndarray, y: np.ndarray, batch_size: int,
                         image_size: int, train: bool) -> tf.data.Dataset:
    """Upscale grayscale MNIST and replicate it to RGB for ImageNet weights."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if train:
        ds = ds.shuffle(len(x), reshuffle_each_iteration=True)

    def prepare(img, label):
        img = tf.expand_dims(img, axis=-1)  # (28, 28) -> (28, 28, 1)
        img = tf.image.resize(img, (image_size, image_size))
        img = tf.repeat(img, repeats=3, axis=-1)
        img = applications.efficientnet.preprocess_input(tf.cast(img, tf.float32))
        return img, label

    ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_custom_cnn(num_classes: int = 10) -> keras.Model:
    """A small CNN trained from scratch (random weights)."""
    inputs = keras.Input(shape=(28, 28, 1), name="image")
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")


def build_efficientnet(num_classes: int = 10, image_size: int = 32) -> keras.Model:
    """EfficientNetB0 pretrained on ImageNet, with a new classification head."""
    base = applications.EfficientNetB0(
        weights="imagenet", include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False  # feature extraction first
    inputs = base.input
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class")(x)
    model = keras.Model(inputs, outputs, name="efficientnet_b0")
    # Keras 3 flattens nested Functional models, so keep the backbone layer
    # names to be able to unfreeze them for optional fine-tuning.
    model.backbone_layer_names = [layer.name for layer in base.layers]
    return model


def evaluate(model: keras.Model, dataset: tf.data.Dataset, y_true: np.ndarray,
             batch_size: int) -> dict:
    """Return loss/accuracy plus F1, per-class accuracy and inference time."""
    loss, accuracy = model.evaluate(dataset, verbose=0)

    start = time.perf_counter()
    probs = model.predict(dataset, verbose=0)
    inference_time = time.perf_counter() - start
    y_pred = probs.argmax(axis=1)

    per_class = {
        CLASS_NAMES[c]: float((y_pred[y_true == c] == c).mean())
        for c in range(10)
    }
    return {
        "loss": round(float(loss), 4),
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "per_class_accuracy": per_class,
        "inference_ms_per_image": round(inference_time / len(y_true) * 1000, 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "predictions": y_pred,
    }


def plot_training_curves(histories: dict[str, keras.callbacks.History]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, history in histories.items():
        # Merge phases (head training + optional fine-tuning) into one curve.
        acc = [v for h in history for v in h.history["accuracy"]]
        val_acc = [v for h in history for v in h.history["val_accuracy"]]
        loss = [v for h in history for v in h.history["loss"]]
        val_loss = [v for h in history for v in h.history["val_loss"]]
        epochs = np.arange(1, len(acc) + 1)
        axes[0].plot(epochs, acc, marker="o", label=f"{name} train")
        axes[0].plot(epochs, val_acc, marker="s", ls="--", label=f"{name} val")
        axes[1].plot(epochs, loss, marker="o", label=f"{name} train")
        axes[1].plot(epochs, val_loss, marker="s", ls="--", label=f"{name} val")
    axes[0].set_title("Accuracy")
    axes[1].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[1].set_xlabel("epoch")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = RESULTS_DIR / "training_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrices(results: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, metrics) in zip(axes, results.items()):
        cm = np.array(metrics["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name} (acc={metrics['accuracy']:.3f})")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(range(10), CLASS_NAMES)
        ax.set_yticks(range(10), CLASS_NAMES)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    fig.tight_layout()
    path = RESULTS_DIR / "confusion_matrices.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_sample_predictions(x_test: np.ndarray, y_test: np.ndarray,
                            predictions: dict) -> Path:
    samples = np.arange(12)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax, i in zip(axes.flat, samples):
        ax.imshow(x_test[i].squeeze(), cmap="gray")
        labels = [f"{n}: {pred[i]}" for n, pred in predictions.items()]
        ax.set_title(f"true={y_test[i]} | " + " ".join(labels), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    path = RESULTS_DIR / "sample_predictions.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    set_seeds()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading MNIST ...", flush=True)
    (x_train_all, y_train_all), (x_test_all, y_test_all) = keras.datasets.mnist.load_data()

    x_train, y_train = balanced_subset(x_train_all, y_train_all, args.train_per_class)
    x_test, y_test = balanced_subset(x_test_all, y_test_all, args.test_per_class)
    print(f"Train: {len(x_train)} images, test: {len(x_test)} images", flush=True)

    # Fixed validation split (last 10% of the already shuffled training subset).
    n_val = max(1, int(0.1 * len(x_train)))
    x_fit, y_fit = x_train[:-n_val], y_train[:-n_val]
    x_val, y_val = x_train[-n_val:], y_train[-n_val:]

    histories: dict[str, list[keras.callbacks.History]] = {}
    results: dict[str, dict] = {}

    # ---------------------------- custom CNN ----------------------------
    print("\n=== Training custom CNN (from scratch) ===", flush=True)
    cnn = build_custom_cnn()
    cnn.compile(optimizer=keras.optimizers.Adam(1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    cnn_ds = cnn_dataset(x_fit, y_fit, args.batch_size, train=True)
    cnn_val_ds = cnn_dataset(x_val, y_val, args.batch_size, train=False)
    cnn_test_ds = cnn_dataset(x_test, y_test, args.batch_size, train=False)
    t0 = time.perf_counter()
    h = cnn.fit(cnn_ds, validation_data=cnn_val_ds, epochs=args.cnn_epochs, verbose=2)
    histories["Custom CNN"] = [h]
    results["Custom CNN"] = evaluate(cnn, cnn_test_ds, y_test, args.batch_size)
    results["Custom CNN"]["train_seconds"] = round(time.perf_counter() - t0, 1)
    results["Custom CNN"]["total_params"] = int(cnn.count_params())
    results["Custom CNN"]["trainable_params"] = int(
        sum(np.prod(w.shape) for w in cnn.trainable_weights))
    models_dir = HERE / "models"
    models_dir.mkdir(exist_ok=True)
    cnn.save(models_dir / "custom_cnn.keras")

    # ------------------------- EfficientNetB0 ---------------------------
    print("\n=== Training EfficientNetB0 (ImageNet pretrained) ===", flush=True)
    eff = build_efficientnet(image_size=args.eff_image_size)
    eff.compile(optimizer=keras.optimizers.Adam(1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    eff_ds = efficientnet_dataset(x_fit, y_fit, args.batch_size,
                                  args.eff_image_size, train=True)
    eff_val_ds = efficientnet_dataset(x_val, y_val, args.batch_size,
                                      args.eff_image_size, train=False)
    eff_test_ds = efficientnet_dataset(x_test, y_test, args.batch_size,
                                       args.eff_image_size, train=False)
    t0 = time.perf_counter()
    h1 = eff.fit(eff_ds, validation_data=eff_val_ds, epochs=args.eff_epochs, verbose=2)
    eff_histories = [h1]

    if args.eff_finetune_epochs > 0:
        print("Fine-tuning the last EfficientNet layers ...", flush=True)
        names = eff.backbone_layer_names
        for name in names[:-20]:
            eff.get_layer(name).trainable = False
        for name in names[-20:]:
            eff.get_layer(name).trainable = True
        eff.compile(optimizer=keras.optimizers.Adam(1e-5),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
        h2 = eff.fit(eff_ds, validation_data=eff_val_ds,
                     epochs=args.eff_finetune_epochs, verbose=2)
        eff_histories.append(h2)

    histories["EfficientNetB0"] = eff_histories
    results["EfficientNetB0"] = evaluate(eff, eff_test_ds, y_test, args.batch_size)
    results["EfficientNetB0"]["train_seconds"] = round(time.perf_counter() - t0, 1)
    results["EfficientNetB0"]["total_params"] = int(eff.count_params())
    results["EfficientNetB0"]["trainable_params"] = int(
        sum(np.prod(w.shape) for w in eff.trainable_weights))

    # ---------------------------- report --------------------------------
    predictions = {name: m["predictions"] for name, m in results.items()}
    for metrics in results.values():
        metrics.pop("predictions", None)

    summary = {
        "dataset": "MNIST (balanced per-class subset)",
        "train_images": int(len(x_train)),
        "test_images": int(len(x_test)),
        "effnet_image_size": args.eff_image_size,
        "cnn_epochs": args.cnn_epochs,
        "efficientnet_head_epochs": args.eff_epochs,
        "efficientnet_finetune_epochs": args.eff_finetune_epochs,
        "seed": SEED,
        "models": results,
    }
    (RESULTS_DIR / "comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    csv_lines = ["model,accuracy,loss,f1_macro,f1_weighted,total_params,"
                 "train_seconds,inference_ms_per_image"]
    for name, m in results.items():
        csv_lines.append(f"{name},{m['accuracy']},{m['loss']},{m['f1_macro']},"
                         f"{m['f1_weighted']},{m['total_params']},"
                         f"{m['train_seconds']},{m['inference_ms_per_image']}")
    (RESULTS_DIR / "comparison.csv").write_text("\n".join(csv_lines) + "\n",
                                                encoding="utf-8")

    class_lines = ["model,digit,accuracy"] + [
        f"{name},{digit},{acc:.4f}"
        for name, m in results.items()
        for digit, acc in m["per_class_accuracy"].items()
    ]
    (RESULTS_DIR / "per_class_accuracy.csv").write_text(
        "\n".join(class_lines) + "\n", encoding="utf-8")

    plot_training_curves(histories)
    plot_confusion_matrices(results)
    plot_sample_predictions(x_test, y_test, predictions)

    print("\n=== Comparison ===")
    for name, m in results.items():
        print(f"{name:16s} acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
              f"params={m['total_params']:,} train={m['train_seconds']}s "
              f"inference={m['inference_ms_per_image']}ms/img")
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
