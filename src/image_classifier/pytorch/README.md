# PyTorch Lightning — Fashion-MNIST

This sub-package adds **PyTorch Lightning** on top of the existing plain PyTorch training code.

---

## Why Lightning After Plain PyTorch?

The plain PyTorch training loop in `src/image_classifier/train.py` works but requires manual boilerplate for:

- Device management (`model.to(device)`, `images.to(device)`)
- Training / evaluation mode toggling
- Gradient zeroing and stepping
- Metric computation
- Checkpointing
- Logging

Lightning eliminates this boilerplate while keeping full control over the model and training logic. It also provides built-in support for:

- **Automatic checkpointing** — save and resume from the best model
- **TensorBoard logging** — track loss and accuracy curves
- **Multi-GPU / distributed** — scale training with one flag
- **Reproducibility** — `seed_everything` handles all random states

---

## How LightningModule Maps to Model / Loss / Optimizer

```text
Plain PyTorch                         LightningModule
─────────────────────────────         ──────────────────────────────
model = SimpleCNN()                   self.model = SimpleCNN()
criterion = nn.CrossEntropyLoss()     self.criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())  configure_optimizers() → Adam
forward pass + loss.backward()        training_step(batch, batch_idx)
model.eval() + torch.no_grad()        validation_step / test_step
manual accuracy computation           torchmetrics.Accuracy
```

The `FashionMNISTClassifier` wraps the **existing `SimpleCNN`** — no model code is duplicated.

---

## How DataModule Handles Data Loading

`FashionMNISTDataModule` encapsulates all data logic:

| Method              | What it does                                     |
|---------------------|--------------------------------------------------|
| `prepare_data()`    | Downloads Fashion-MNIST (runs on rank-0 only)    |
| `setup(stage)`      | Splits 60k train → 50k train + 10k val           |
| `train_dataloader()`| Returns shuffled training DataLoader             |
| `val_dataloader()`  | Returns validation DataLoader                    |
| `test_dataloader()` | Returns test DataLoader (10k official test set)  |

---

## How to Run Lightning Training

```bash
uv run python -m src.image_classifier.pytorch.train \
    --data-dir data \
    --output-dir outputs \
    --batch-size 64 \
    --learning-rate 0.001 \
    --epochs 5 \
    --num-workers 2 \
    --seed 42
```

---

## Output Structure

Each run creates:

```text
outputs/
└── runs/
    └── <run-id>/
        ├── config.json          # hyperparameters
        ├── metrics.json         # final train/val/test metrics
        ├── checkpoints/         # Lightning .ckpt files
        │   └── best-epoch=XX-val_loss=X.XXXX.ckpt
        └── tensorboard/        # TensorBoard event files
```

### Checkpoints

The `ModelCheckpoint` callback saves the best model (lowest `val_loss`) under `checkpoints/`.

### TensorBoard

Launch TensorBoard to view training curves:

```bash
tensorboard --logdir outputs/runs/<run-id>/tensorboard
```
