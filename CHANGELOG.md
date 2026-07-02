# Changelog

## [Unreleased]

### Added

- Initial repository foundation.
- Contribution guidelines.
- Branching and release rules.
- Initial CI workflow plan.
- Dataset plan for Fashion-MNIST and CIFAR-10.
- Fashion-MNIST data loading with `create_dataloaders`.
- Fashion-MNIST class labels (`FASHION_MNIST_LABELS`).
- `SimpleCNN` model returning raw logits `[batch_size, 10]`.
- Plain PyTorch training loop with CLI (`train.py`).
- Reusable `evaluate()` function returning loss and accuracy.
- `EvalMetrics` dataclass for evaluation results.
- Training output artifacts: `config.json`, `metrics.json`, `model.pt`.
- Lightweight unit tests: `test_data`, `test_model`, `test_training`.
- CI updated to install `ml` dependency group for tests.
- README documentation for Fashion-MNIST training.
- `FashionMNISTDataModule` for Lightning data loading with train/val/test splits.
- `FashionMNISTClassifier` LightningModule wrapping `SimpleCNN` with `torchmetrics`.
- Lightning training CLI with `ModelCheckpoint` and `TensorBoardLogger`.
- Lightning tests including `fast_dev_run` integration test.
- PyTorch Lightning README under `src/image_classifier/pytorch/`.

### Changed

### Fixed

### Removed
