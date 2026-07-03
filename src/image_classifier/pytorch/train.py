"""Lightning training CLI for Fashion-MNIST."""

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from .data_module import FashionMNISTDataModule
from .lightning_module import FashionMNISTClassifier


# Helpers
def _collect_git_metadata():
    """Return git commit, branch, and dirty status or None for each on failure."""
    git_info = {"git_commit": None, "git_branch": None, "git_is_dirty": None}

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_info["git_commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_info["git_branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_info["git_is_dirty"] = len(result.stdout.strip()) > 0

    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "WARNING: Could not collect git metadata. "
            "Git fields will be null in run_metadata.json.",
            file=sys.stderr,
        )

    return git_info


def _collect_run_metadata(run_id):
    """Build the run_metadata dict with environment and git information."""
    import torch

    git_info = _collect_git_metadata()

    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        **git_info,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "lightning_version": L.__version__,
    }


def _build_config(args):
    """Build the config dict from parsed CLI arguments."""
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


# Training
def train(args):
    """Run a Lightning training session."""

    L.seed_everything(args.seed, workers=True)

    # run directory
    run_id = datetime.now().strftime("%Y-%m-%d-%H%M%S") + f"-{args.run_name}"
    run_dir = Path(args.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # save config
    config = _build_config(args)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # save run metadata
    run_metadata = _collect_run_metadata(run_id)
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    # data module
    dm = FashionMNISTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # model
    model = FashionMNISTClassifier(learning_rate=args.learning_rate)

    # callbacks
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

    # logger
    tb_logger = TensorBoardLogger(
        save_dir=str(run_dir / "tensorboard"),
        name="",
        version="",
    )

    # trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        callbacks=[best_cb, last_cb],
        logger=tb_logger,
        enable_progress_bar=True,
        accelerator=args.accelerator,
        devices=args.devices,
    )

    # fit + test
    trainer.fit(model, datamodule=dm)
    test_results = trainer.test(model, datamodule=dm, ckpt_path="best")

    # save final metrics
    best_val_loss = best_cb.best_model_score
    metrics = {
        "best_val_loss": best_val_loss.item() if best_val_loss is not None else None,
        "best_val_accuracy": trainer.callback_metrics.get("val_acc"),
        "final_test_loss": test_results[0].get("test_loss") if test_results else None,
        "final_test_accuracy": test_results[0].get("test_acc") if test_results else None,
    }

    # convert tensors to plain floats
    for key, value in metrics.items():
        if hasattr(value, "item"):
            metrics[key] = value.item()

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nRun artifacts saved to: {run_dir}")


if __name__ == "__main__":
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
