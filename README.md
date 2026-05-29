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




