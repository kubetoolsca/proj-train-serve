# Changelog

## [Unreleased]

### Added

- Readable run-ID format (`YYYY-MM-DD-HHMMSS-<run-name>`).
- `run_metadata.json` with git commit, branch, dirty status, and environment versions.
- `best.ckpt` and `last.ckpt` checkpoint naming (no epoch/metric suffixes).
- `--run-name`, `--accelerator`, `--devices` CLI arguments for Lightning training.
- `learning_rate` logging to TensorBoard.
- Spec-compliant `metrics.json` keys (`best_val_loss`, `best_val_accuracy`, `final_test_loss`, `final_test_accuracy`).
- Expanded `config.json` with `dataset`, `model`, `accelerator`, `devices` fields.
- Experiment tracking tests (`test_experiment_tracking.py`).

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
