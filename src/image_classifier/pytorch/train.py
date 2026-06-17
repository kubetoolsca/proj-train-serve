"""Lightning training CLI for Fashion-MNIST."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from .data_module import FashionMNISTDataModule
from .lightning_module import FashionMNISTClassifier


def train(args):
    """Run a Lightning training session."""

    L.seed_everything(args.seed, workers=True)

    # run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # save config
    config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "seed": args.seed,
        "framework": "lightning",
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # data module
    dm = FashionMNISTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # model
    model = FashionMNISTClassifier(learning_rate=args.learning_rate)

    # callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
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
        callbacks=[checkpoint_cb],
        logger=tb_logger,
        enable_progress_bar=True,
    )

    # fit + test
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    # save final metrics
    metrics = {}
    for key, value in trainer.callback_metrics.items():
        metrics[key] = value.item() if hasattr(value, "item") else value

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nRun saved to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Fashion-MNIST classifier with PyTorch Lightning",
    )

    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
