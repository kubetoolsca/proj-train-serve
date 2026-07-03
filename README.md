# proj-train-serve

`proj-train-serve` is an implementation project for the full machine learning lifecycle.

The goal is to understand how model training, experiment tracking, workflow automation, model export, model serving, and API consumption connect together in a production-style ML system.

---

## Current Dataset Plan

This project will start with image classification.

Datasets:

```text
Stage 1: Fashion-MNIST
Stage 2: CIFAR-10
```

Fashion-MNIST is used first because it is simple, grayscale, and useful for learning CNN training and inference basics.

CIFAR-10 is used second because it introduces RGB images and more realistic image preprocessing.

---

## High-Level ML Lifecycle

```text
Developer pushes code
      ↓
GitHub Actions CI runs
      ↓
Training image is built
      ↓
Argo Events receives GitHub/webhook event
      ↓
Argo Workflows starts ML pipeline
      ↓
Data validation
      ↓
Training with PyTorch Lightning
      ↓
Validation and test metrics
      ↓
TensorBoard experiment tracking
      ↓
Hyperparameter tuning
      ↓
Best checkpoint selection
      ↓
Export checkpoint to ONNX
      ↓
Validate ONNX output
      ↓
Package model into Triton model repository
      ↓
Serve with NVIDIA Triton
      ↓
FastAPI handles preprocessing/postprocessing
      ↓
Prediction JSON response
```

---

## Development Standard

This project uses:

```text
Python 3.11
pyproject.toml
uv
ruff (instead of Flake8 / isort / black we use ruff)
pytest
GitHub Actions
self-hosted runners
```

---

## Quick Start

Install uv if not already installed.

Then run:

```bash
uv sync --dev
```

Run tests:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check .
```

Run format check:

```bash
uv run ruff format --check .
```

---

## Local Training — Fashion-MNIST

### What is Fashion-MNIST?

Fashion-MNIST is a dataset of Zalando article images. It contains 60,000 training and 10,000 test grayscale images across 10 clothing categories.

### Image and Output Shapes

```text
Single image shape:  [1, 28, 28]   (channels, height, width)
Batch shape:         [batch_size, 1, 28, 28]
Model output:        [batch_size, 10]   (raw logits, one per class)
```

The 10 output values are **raw logits** (unnormalized scores). Softmax is **not** applied inside the model. To get probabilities, apply `torch.softmax(logits, dim=1)` externally.

### The 10 Classes

```text
0: T-shirt/top    5: Sandal
1: Trouser        6: Shirt
2: Pullover       7: Sneaker
3: Dress          8: Bag
4: Coat           9: Ankle boot
```

### CNN Architecture

```text
Input: [batch_size, 1, 28, 28]
        │
        ▼
┌───────────────────────┐
│ Conv2d(1→32, 3×3, p1) │
│ ReLU                  │
│ MaxPool2d(2)          │
│ → [batch_size, 32, 14, 14]
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ Conv2d(32→64, 3×3, p1)│
│ ReLU                  │
│ MaxPool2d(2)          │
│ → [batch_size, 64, 7, 7]
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ Flatten               │
│ → [batch_size, 3136]  │
│ Linear(3136→128)      │
│ ReLU                  │
│ Linear(128→10)        │
│ → [batch_size, 10]    │
└───────────────────────┘
        │
        ▼
Output: raw logits [batch_size, 10]
```

### Training Command

```bash
uv run python -m src.image_classifier.train \
    --data-dir data \
    --output-dir outputs \
    --batch-size 64 \
    --learning-rate 0.001 \
    --epochs 5 \
    --num-workers 2 \
    --seed 42
```

### Output Artifacts

Each training run creates:

```text
outputs/
└── runs/
    └── <run-id>/
        ├── config.json      # hyperparameters and device
        ├── metrics.json     # final_train_loss, final_test_loss, final_test_accuracy
        └── model.pt         # state_dict only
```

### Running Tests

```bash
uv sync --dev --group ml
uv run pytest -vv
```

Tests use synthetic data and do not download the full Fashion-MNIST dataset.

---

## Lightning Training — Fashion-MNIST

The project also includes a **PyTorch Lightning** training path under `src/image_classifier/pytorch/`. Lightning wraps the same `SimpleCNN` model with cleaner training structure, automatic checkpointing, TensorBoard logging, and `torchmetrics` integration.

See [src/image_classifier/pytorch/README.md](src/image_classifier/pytorch/README.md) for full details on architecture, DataModule, and LightningModule design.

### Lightning Training Command

```bash
uv run python -m src.image_classifier.pytorch.train \
    --data-dir data \
    --output-dir outputs \
    --batch-size 64 \
    --learning-rate 0.001 \
    --epochs 5 \
    --num-workers 0 \
    --seed 42 \
    --run-name fashion-mnist-lightning \
    --accelerator cpu \
    --devices 1
```

### Lightning Output Artifacts

Each run creates a unique directory with a readable ID (e.g. `2026-07-02-120530-fashion-mnist-lightning`):

```text
outputs/
└── runs/
    └── <run-id>/
        ├── config.json          # training hyperparameters
        ├── metrics.json         # best val + final test metrics
        ├── run_metadata.json    # git commit, branch, python/torch versions
        ├── checkpoints/
        │   ├── best.ckpt        # best model (lowest val_loss)
        │   └── last.ckpt        # latest checkpoint
        └── tensorboard/         # TensorBoard event files
```

See [src/image_classifier/pytorch/README.md](src/image_classifier/pytorch/README.md) for full experiment tracking documentation.

### TensorBoard

```bash
uv run tensorboard --logdir outputs/runs
```

---

## Branching Model

```text
develop -> integration branch
main    -> stable release branch
```

All normal work should branch from `develop`.

Example:

```bash
git checkout develop
git pull origin develop
git checkout -b feat/12-fastapi-inference-api
```

---

## Directory Structure

```text
proj-train-serve/
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
├── .editorconfig
├── .pre-commit-config.yaml
├── .yamllint.yaml
├── .github/
│   ├── workflows/
│   │   └── ci.yaml
├── docs/
│   ├── architecture.md
│   ├── project-plan.md
│   ├── dependency-management.md
│   ├── branching-and-release.md
│   ├── ci-cd.md
│   ├── self-hosted-runners.md
├── src/
│   └── image_classifier/
│       └── __init__.py
├── serving/
├── workflows/
├── events/
├── k8s/
└── tests/
```

---

## Contributing

Please read:

```text
CONTRIBUTING.md
```

before creating branches or pull requests.
```
