import torch

from .metrics import EvalMetrics


def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            total_loss += loss.item()

            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

    return EvalMetrics(
        loss=total_loss / len(dataloader),
        accuracy=correct / total,
    )
