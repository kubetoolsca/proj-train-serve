import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from .data import create_dataloaders
from .evaluate import evaluate
from .model import SimpleCNN


def train(args):

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    train_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path(args.output_dir) / "runs" / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": device,
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    final_train_loss = 0.0

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        final_train_loss = running_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{args.epochs} — train_loss: {final_train_loss:.4f}")

    eval_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    metrics = {
        "final_train_loss": final_train_loss,
        "final_test_loss": eval_metrics.loss,
        "final_test_accuracy": eval_metrics.accuracy,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(
        model.state_dict(),
        run_dir / "model.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(args)
