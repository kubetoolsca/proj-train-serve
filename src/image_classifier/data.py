from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    train_ds = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_ds = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
