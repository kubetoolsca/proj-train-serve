"""Tests for data loading shapes using synthetic data."""

import torch
from torch.utils.data import DataLoader, TensorDataset


def test_dataloader_image_shape():
    """Synthetic dataloader returns images with shape [batch_size, 1, 28, 28]."""
    batch_size = 8
    images = torch.randn(32, 1, 28, 28)
    labels = torch.randint(0, 10, (32,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size)

    batch_images, batch_labels = next(iter(loader))

    assert batch_images.shape == (batch_size, 1, 28, 28)
    assert batch_labels.shape == (batch_size,)


def test_dataloader_label_range():
    """Labels are integers in the range [0, 9]."""
    labels = torch.randint(0, 10, (32,))
    assert labels.min() >= 0
    assert labels.max() <= 9


def test_dataloader_dtype():
    """Images are float tensors, labels are integer tensors."""
    images = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))

    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
