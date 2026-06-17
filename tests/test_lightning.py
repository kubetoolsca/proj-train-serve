"""Lightweight tests for the PyTorch Lightning integration."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.image_classifier.pytorch.data_module import FashionMNISTDataModule
from src.image_classifier.pytorch.lightning_module import FashionMNISTClassifier


# Helpers
def _make_synthetic_loader(n=32, batch_size=8):
    """Return a DataLoader with random Fashion-MNIST–shaped data."""
    images = torch.randn(n, 1, 28, 28)
    labels = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def _make_synthetic_datamodule():
    """Return a DataModule whose loaders use synthetic tensors (no download)."""
    dm = FashionMNISTDataModule(data_dir="data", batch_size=8, num_workers=0)
    # Bypass prepare_data / setup and inject synthetic datasets directly
    n_train, n_val, n_test = 48, 16, 16
    dm.train_ds = TensorDataset(torch.randn(n_train, 1, 28, 28), torch.randint(0, 10, (n_train,)))
    dm.val_ds = TensorDataset(torch.randn(n_val, 1, 28, 28), torch.randint(0, 10, (n_val,)))
    dm.test_ds = TensorDataset(torch.randn(n_test, 1, 28, 28), torch.randint(0, 10, (n_test,)))
    return dm


# LightningModule tests
def test_lightning_module_forward():
    """Forward pass returns [batch_size, 10] logits."""
    model = FashionMNISTClassifier(learning_rate=1e-3)
    batch = torch.randn(4, 1, 28, 28)
    logits = model(batch)
    assert logits.shape == (4, 10)


def test_lightning_module_training_step():
    """training_step returns a finite scalar loss."""
    model = FashionMNISTClassifier(learning_rate=1e-3)
    loader = _make_synthetic_loader()
    batch = next(iter(loader))

    loss = model.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


def test_lightning_module_configure_optimizers():
    """configure_optimizers returns an Adam optimizer."""
    model = FashionMNISTClassifier(learning_rate=1e-3)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)


def test_lightning_module_reuses_simple_cnn():
    """LightningModule wraps the existing SimpleCNN."""
    from src.image_classifier.model import SimpleCNN

    model = FashionMNISTClassifier()
    assert isinstance(model.model, SimpleCNN)


# DataModule tests
def test_datamodule_hyperparameters():
    """DataModule stores hyperparameters correctly."""
    dm = FashionMNISTDataModule(data_dir="data", batch_size=32, num_workers=0)
    assert dm.hparams.batch_size == 32
    assert dm.hparams.num_workers == 0


# Trainer integration (fast_dev_run)
def test_lightning_fast_dev_run():
    """Trainer.fit + test with fast_dev_run=True completes without error."""
    import lightning as L

    dm = _make_synthetic_datamodule()
    model = FashionMNISTClassifier(learning_rate=1e-3)

    trainer = L.Trainer(
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
