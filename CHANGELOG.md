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
- Readable run-id format: `YYYY-MM-DD-HHMMSS-<run-name>`.
- `config.json` with full training configuration (dataset, model, accelerator, devices).
- `run_metadata.json` with git commit, branch, dirty status, and version info.
- `best.ckpt` and `last.ckpt` checkpoint saving via two `ModelCheckpoint` callbacks.
- TensorBoard logging of `train_loss`, `val_loss`, `val_acc`, `test_loss`, `test_acc`, `learning_rate`.
- Structured `metrics.json` with `best_val_loss`, `best_val_accuracy`, `final_test_loss`, `final_test_accuracy`.
- CLI flags `--run-name`, `--accelerator`, `--devices` for Lightning training.
- Lightweight experiment tracking tests (`test_experiment_tracking.py`).

### Changed

### Fixed

### Removed
