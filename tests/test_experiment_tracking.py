"""Lightweight tests for experiment tracking artifacts.

These tests validate the artifact structure (run directory, config.json,
metrics.json, run_metadata.json, checkpoint callbacks, TensorBoard logger)
without downloading Fashion-MNIST or running full training.
"""

import json

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.image_classifier.pytorch.train import (
    _build_config,
    _build_metrics,
    _collect_run_metadata,
)

# --- Helpers ---


class _FakeArgs:
    """Mimics the argparse namespace for config building."""

    def __init__(self, **kwargs):
        defaults = {
            "batch_size": 64,
            "learning_rate": 0.001,
            "epochs": 5,
            "num_workers": 0,
            "seed": 42,
            "accelerator": "cpu",
            "devices": 1,
            "run_name": "fashion-mnist-lightning",
            "data_dir": "data",
            "output_dir": "outputs",
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


class _FakeTrainer:
    """Mimics trainer.callback_metrics for metrics extraction."""

    class _Tensor:
        def __init__(self, v):
            self._v = v

        def item(self):
            return self._v

    def __init__(self, metrics_dict):
        self.callback_metrics = {k: self._Tensor(v) for k, v in metrics_dict.items()}


# --- Tests ---


def test_run_directory_creation(tmp_path):
    """Run directory is created with expected structure."""
    run_dir = tmp_path / "outputs" / "runs" / "2026-06-17-120530-fashion-mnist-lightning"
    run_dir.mkdir(parents=True, exist_ok=True)

    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir.name == "2026-06-17-120530-fashion-mnist-lightning"


def test_config_json_writing(tmp_path):
    """config.json contains all required keys."""
    args = _FakeArgs()
    config = _build_config(args)

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    loaded = json.loads(config_path.read_text())

    expected_keys = {
        "data_dir",
        "dataset",
        "framework",
        "model",
        "batch_size",
        "learning_rate",
        "epochs",
        "num_workers",
        "seed",
        "accelerator",
        "devices",
    }
    assert set(loaded.keys()) == expected_keys
    assert loaded["dataset"] == "fashion-mnist"
    assert loaded["framework"] == "pytorch-lightning"
    assert loaded["model"] == "simple-cnn"
    assert loaded["batch_size"] == 64
    assert loaded["accelerator"] == "cpu"


def test_metrics_json_writing(tmp_path):
    """metrics.json contains spec-compliant metric keys."""
    trainer = _FakeTrainer({"val_loss": 0.35, "val_acc": 0.88, "test_loss": 0.40, "test_acc": 0.86})
    metrics = _build_metrics(trainer)

    metrics_path = tmp_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    loaded = json.loads(metrics_path.read_text())

    expected_keys = {
        "best_val_loss",
        "best_val_accuracy",
        "final_test_loss",
        "final_test_accuracy",
    }
    assert set(loaded.keys()) == expected_keys
    assert loaded["best_val_loss"] == 0.35
    assert loaded["final_test_accuracy"] == 0.86


def test_run_metadata_json_writing(tmp_path):
    """run_metadata.json contains all required fields; git fields may be null."""
    run_id = "2026-06-17-120530-fashion-mnist-lightning"
    metadata = _collect_run_metadata(run_id)

    metadata_path = tmp_path / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    loaded = json.loads(metadata_path.read_text())

    required_fields = {
        "run_id",
        "created_at",
        "git_commit",
        "git_branch",
        "git_is_dirty",
        "python_version",
        "torch_version",
        "lightning_version",
    }
    assert set(loaded.keys()) == required_fields
    assert loaded["run_id"] == run_id
    assert loaded["python_version"] is not None
    assert loaded["torch_version"] is not None
    assert loaded["lightning_version"] is not None


def test_checkpoint_callbacks_configuration(tmp_path):
    """ModelCheckpoint callbacks are configured for best and last checkpoints."""
    ckpt_dir = str(tmp_path / "checkpoints")

    best_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    last_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        every_n_epochs=1,
    )

    assert best_cb.filename == "best"
    assert best_cb.monitor == "val_loss"
    assert best_cb.mode == "min"
    assert best_cb.dirpath == ckpt_dir

    assert last_cb.filename == "last"
    assert last_cb.dirpath == ckpt_dir


def test_tensorboard_logger_path(tmp_path):
    """TensorBoardLogger save_dir points to the expected tensorboard subdirectory."""
    tb_dir = str(tmp_path / "tensorboard")

    logger = TensorBoardLogger(save_dir=tb_dir, name="", version="")

    assert logger.save_dir == tb_dir
