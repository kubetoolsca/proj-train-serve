"""Tests for SimpleCNN model architecture and output."""

import torch

from src.image_classifier.model import SimpleCNN


def test_model_output_shape():
    """Model output has shape [batch_size, 10]."""
    model = SimpleCNN()
    batch = torch.randn(4, 1, 28, 28)
    output = model(batch)

    assert output.shape == (4, 10)


def test_model_output_is_raw_logits():
    """Model returns raw logits, not softmax probabilities.

    Raw logits can be negative and do not sum to 1.
    """
    model = SimpleCNN()
    batch = torch.randn(4, 1, 28, 28)
    output = model(batch)

    has_negative = (output < 0).any().item()
    row_sums = output.sum(dim=1)
    sums_to_one = torch.allclose(row_sums, torch.ones(4), atol=0.01)

    assert has_negative or not sums_to_one, (
        "Output looks like softmax probabilities, not raw logits"
    )


def test_model_single_image():
    """Model handles a single image (batch_size=1)."""
    model = SimpleCNN()
    single = torch.randn(1, 1, 28, 28)
    output = model(single)

    assert output.shape == (1, 10)
