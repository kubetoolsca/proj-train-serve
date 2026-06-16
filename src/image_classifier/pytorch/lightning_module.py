"""Lightning Module for Fashion-MNIST classification."""

import lightning as L
import torch.nn as nn
import torchmetrics

from ..model import SimpleCNN


class FashionMNISTClassifier(L.LightningModule):
    """Wraps the existing SimpleCNN with Lightning training / eval logic.

    Logs loss and accuracy for train, validation, and test stages.
    Uses torchmetrics.Accuracy for consistent metric computation.
    """

    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = SimpleCNN()
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    # Forward
    def forward(self, x):
        return self.model(x)

    # Training
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        self.train_acc(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # Validation
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        self.val_acc(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    # Test
    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        self.test_acc(preds, labels)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)


    # Optimizer
    def configure_optimizers(self):
        return __import__("torch").optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
