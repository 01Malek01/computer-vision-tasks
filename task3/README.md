# Task 3 — Custom CNN vs. Pretrained EfficientNetB0

Image classification on **MNIST** with a fair, head-to-head comparison between:

1. **Custom CNN** — built from scratch, trained from random weights.
2. **EfficientNetB0** — downloaded with **ImageNet pretrained weights** (Keras cache:
   `~/.keras/models/efficientnetb0_notop.h5`) and adapted to MNIST with a new
   classification head (transfer learning).

Both models see **exactly the same balanced subset** of MNIST (same seed, same
digits, same train/validation/test split), so the comparison is meaningful.

## Dataset

- MNIST: 70,000 grayscale 28×28 handwritten-digit images (10 classes).
- Default subset used here: 1,000 training + 200 test images per class
  (10,000 train / 2,000 test). Change with `--train-per-class` / `--test-per-class`.
- Custom CNN input: `(28, 28, 1)`, scaled to `[0, 1]`.
- EfficientNet input: grayscale image upscaled to 32×32 and replicated to
  3 channels (ImageNet convention), fed through EfficientNet's preprocessing.

## Models

| Model | Architecture | Training |
|---|---|---|
| Custom CNN | 2×(Conv→Conv→Pool→Dropout) + Dense(128) + Dropout + Softmax(10) | Adam 1e-3, from scratch |
| EfficientNetB0 | ImageNet backbone (frozen) + GAP + Dense(128) + Dropout + Softmax(10) | Adam 1e-3 for the head; optional fine-tuning of last 20 layers at 1e-5 |

EfficientNetB0 was downloaded using
`EfficientNetB0(weights="imagenet", include_top=False, input_shape=(32, 32, 3))`.
The first download is cached by Keras; subsequent runs are offline.

## Results

See [`results/comparison.csv`](results/comparison.csv),
[`results/comparison.json`](results/comparison.json) and
[`results/per_class_accuracy.csv`](results/per_class_accuracy.csv).

| Model | Accuracy | Macro F1 | Params | Trainable params | Train (s) | Inference (ms/img) |
|---|---|---|---|---|---|---|
| Custom CNN | **0.9855** | **0.9855** | 467,818 | 467,818 | 43.0 | **0.32** |
| EfficientNetB0 (ImageNet) | 0.8170 | 0.8133 | 4,214,829 | 165,258 | 55.6 | 3.404 |

**Conclusion.** On this dataset the small CNN trained from scratch clearly wins:
+16.85 percentage points of accuracy, 9× fewer parameters and ~10× faster
inference. ImageNet features are not a good match for 28×28 grayscale digits
upscaled to 32×32: ImageNet filters respond to natural textures/colors that MNIST
does not have. Transfer learning is not universally better — it helps most when
the target domain resembles the pretraining domain (or when labeled data is very
scarce). EfficientNetB0's best classes are 0/1; it mostly confuses digits 2, 5
and 9 (see the confusion matrix).

Note: unfreezing the last EfficientNet layers (`--eff-finetune-epochs 3`) made
results worse (~0.43 accuracy) in this CPU setup because the pretrained BatchNorm
statistics were overwritten on a small, low-resolution dataset; the frozen
backbone is the configuration reported above.

Visualizations:

- [`results/training_curves.png`](results/training_curves.png) — accuracy/loss curves.
- [`results/confusion_matrices.png`](results/confusion_matrices.png) — per-model confusion matrices.
- [`results/sample_predictions.png`](results/sample_predictions.png) — predictions on 12 test digits.

## How to run

```bash
pip install -r task3/requirements.txt
python task3/train_compare.py
```

Useful options:

```bash
# tiny smoke test
python task3/train_compare.py --train-per-class 100 --test-per-class 20 \
    --cnn-epochs 2 --eff-epochs 2

# experimental: fine-tune the last EfficientNet layers as a second phase
# (degrades accuracy on this dataset; see note above)
python task3/train_compare.py --eff-finetune-epochs 3
```

CPU-only training is supported; runtime depends on the number of cores.

## Project structure

```
task3/
├── train_compare.py        # end-to-end training + evaluation + plots
├── requirements.txt
├── README.md
├── models/                 # generated (git-ignored): custom_cnn.keras
└── results/                # generated & committed: metrics + plots
```
