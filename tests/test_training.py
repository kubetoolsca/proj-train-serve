"""Tests for training loop and evaluation using synthetic data."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.image_classifier.evaluate import evaluate
from src.image_classifier.metrics import EvalMetrics
from src.image_classifier.model import SimpleCNN


def _make_synthetic_loader(n=32, batch_size=8):
    """Create a synthetic dataloader mimicking Fashion-MNIST."""
    images = torch.randn(n, 1, 28, 28)
    labels = torch.randint(0, 10, (n,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size)


def test_single_training_step():
    """One forward-backward-step runs without error and loss is finite."""
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loader = _make_synthetic_loader()
    images, labels = next(iter(loader))

    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(torch.tensor(loss.item()))


def test_parameters_update():
    """Model parameters change after one training step."""
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    params_before = [p.clone() for p in model.parameters()]

    loader = _make_synthetic_loader()
    images, labels = next(iter(loader))

    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    any_changed = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, model.parameters(), strict=False)
    )
    assert any_changed, "No parameters changed after training step"


def test_evaluate_returns_metrics():
    """evaluate() returns an EvalMetrics with loss and accuracy."""
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    loader = _make_synthetic_loader()

    result = evaluate(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device="cpu",
    )

    assert isinstance(result, EvalMetrics)
    assert isinstance(result.loss, float)
    assert isinstance(result.accuracy, float)
    assert 0.0 <= result.accuracy <= 1.0


def test_evaluate_loss_is_finite():
    """Evaluation loss is a finite number."""
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    loader = _make_synthetic_loader()

    result = evaluate(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device="cpu",
    )

    assert torch.isfinite(torch.tensor(result.loss))
