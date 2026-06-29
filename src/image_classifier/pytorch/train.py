"""Lightning training CLI for Fashion-MNIST with experiment tracking."""

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from .data_module import FashionMNISTDataModule
from .lightning_module import FashionMNISTClassifier


def _collect_git_metadata() -> dict:
    """Collect git commit, branch, and dirty status.

    Returns a dict with git_commit, git_branch, and git_is_dirty.
    All values default to None if git is unavailable.
    """
    result = {"git_commit": None, "git_branch": None, "git_is_dirty": None}

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if commit.returncode == 0:
            result["git_commit"] = commit.stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if branch.returncode == 0:
            result["git_branch"] = branch.stdout.strip()

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if dirty.returncode == 0:
            result["git_is_dirty"] = len(dirty.stdout.strip()) > 0

    except (FileNotFoundError, subprocess.TimeoutExpired):
        # git binary not found or timed out
        print("Warning: Could not collect git metadata (git not available or timed out).")

    return result


def _collect_run_metadata(run_id: str) -> dict:
    """Build the run_metadata.json payload."""
    git_info = _collect_git_metadata()

    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_info["git_commit"],
        "git_branch": git_info["git_branch"],
        "git_is_dirty": git_info["git_is_dirty"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "lightning_version": L.__version__,
    }


def _build_config(args) -> dict:
    """Build the config.json payload from CLI args."""
    return {
        "data_dir": args.data_dir,
        "dataset": "fashion-mnist",
        "framework": "pytorch-lightning",
        "model": "simple-cnn",
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "accelerator": args.accelerator,
        "devices": args.devices,
    }


def _build_metrics(trainer) -> dict:
    """Extract spec-compliant metrics from trainer callback metrics."""
    cb = trainer.callback_metrics
    metrics = {}

    # Map callback metric keys to spec keys
    key_map = {
        "val_loss": "best_val_loss",
        "val_acc": "best_val_accuracy",
        "test_loss": "final_test_loss",
        "test_acc": "final_test_accuracy",
    }

    for src_key, dst_key in key_map.items():
        value = cb.get(src_key)
        if value is not None:
            metrics[dst_key] = value.item() if hasattr(value, "item") else value

    return metrics


def train(args):
    """Run a Lightning training session with full experiment tracking."""
    L.seed_everything(args.seed, workers=True)

    # Run directory with readable run-id
    run_id = datetime.now().strftime("%Y-%m-%d-%H%M%S") + f"-{args.run_name}"
    run_dir = Path(args.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = _build_config(args)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save run metadata
    metadata = _collect_run_metadata(run_id)
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Data module
    dm = FashionMNISTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = FashionMNISTClassifier(learning_rate=args.learning_rate)

    # Callbacks — best and last checkpoints
    best_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    last_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="last",
        every_n_epochs=1,
    )

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=str(run_dir / "tensorboard"),
        name="",
        version="",
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        callbacks=[best_cb, last_cb],
        logger=tb_logger,
        enable_progress_bar=True,
        accelerator=args.accelerator,
        devices=args.devices,
    )

    # Fit + test
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    # Save final metrics
    metrics = _build_metrics(trainer)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nRun artifacts saved to: {run_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train Fashion-MNIST classifier with PyTorch Lightning",
    )

    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="fashion-mnist-lightning")
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", type=int, default=1)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
