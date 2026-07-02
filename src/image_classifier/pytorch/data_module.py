"""Lightning DataModule for Fashion-MNIST."""

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class FashionMNISTDataModule(L.LightningDataModule):
    """Handles Fashion-MNIST downloading, splitting, and DataLoader creation.

    Splits the 60 000 training images into 50 000 train + 10 000 validation.
    The 10 000 official test images are kept as-is.
    """

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 64,
        num_workers: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.transform = transforms.ToTensor()

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    # Lightning hooks
    def prepare_data(self):
        """Download dataset (called on rank-0 only)."""
        datasets.FashionMNIST(root=self.data_dir, train=True, download=True)
        datasets.FashionMNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Create train / val / test splits."""
        if stage == "fit" or stage is None:
            full_train = datasets.FashionMNIST(
                root=self.data_dir,
                train=True,
                download=False,
                transform=self.transform,
            )
            generator = torch.Generator().manual_seed(self.seed)
            self.train_ds, self.val_ds = random_split(
                full_train, [50_000, 10_000], generator=generator
            )

        if stage == "test" or stage is None:
            self.test_ds = datasets.FashionMNIST(
                root=self.data_dir,
                train=False,
                download=False,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
