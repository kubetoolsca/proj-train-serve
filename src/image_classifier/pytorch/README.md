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
    --num-workers 0 \
    --seed 42 \
    --run-name fashion-mnist-lightning \
    --accelerator cpu \
    --devices 1
```

### CLI Options

| Flag               | Default                      | Description                          |
|--------------------|------------------------------|--------------------------------------|
| `--data-dir`       | `data`                       | Dataset download directory           |
| `--output-dir`     | `outputs`                    | Root directory for run artifacts     |
| `--batch-size`     | `64`                         | Training batch size                  |
| `--learning-rate`  | `0.001`                      | Adam learning rate                   |
| `--epochs`         | `5`                          | Number of training epochs            |
| `--num-workers`    | `0`                          | DataLoader worker processes          |
| `--seed`           | `42`                         | Random seed for reproducibility      |
| `--run-name`       | `fashion-mnist-lightning`    | Suffix for the run directory name    |
| `--accelerator`    | `cpu`                        | Lightning accelerator (cpu/gpu)    |
| `--devices`        | `1`                          | Number of devices to use             |

---

## Experiment Tracking

Every training run creates a unique directory with a structured artifact layout:

```text
outputs/
└── runs/
    └── <run-id>/
        ├── config.json          # full training configuration
        ├── metrics.json         # best_val_loss, best_val_accuracy, final_test_loss, final_test_accuracy
        ├── run_metadata.json    # git commit, branch, python/torch/lightning versions
        ├── checkpoints/
        │   ├── best.ckpt        # best model (lowest val_loss)
        │   └── last.ckpt        # latest checkpoint at end of training
        └── tensorboard/         # TensorBoard event files
```

The `<run-id>` format is: `YYYY-MM-DD-HHMMSS-<run-name>`

Example: `2026-06-17-120530-fashion-mnist-lightning`

### Logged Metrics

The following metrics are logged to TensorBoard:

```text
train_loss
val_loss
val_acc (val_accuracy)
test_loss
test_acc (test_accuracy)
learning_rate
```

### Checkpoints

Two `ModelCheckpoint` callbacks save:

- `best.ckpt` — best model checkpoint (lowest `val_loss`)
- `last.ckpt` — latest checkpoint at the end of training

### Run Metadata

`run_metadata.json` captures environment details for reproducibility:

```json
{
  "run_id": "2026-06-17-120530-fashion-mnist-lightning",
  "created_at": "2026-06-17T12:05:30+00:00",
  "git_commit": "abc1234...",
  "git_branch": "feat/experiment-tracking",
  "git_is_dirty": false,
  "python_version": "3.11.9",
  "torch_version": "2.3.0",
  "lightning_version": "2.3.0"
}
```

If git is not available (no binary, not a repo, CI detached HEAD), git fields default to `null` and training continues normally.

### TensorBoard

Launch TensorBoard to view training curves across all runs:

```bash
uv run tensorboard --logdir outputs/runs
```

Or for a specific run:

```bash
uv run tensorboard --logdir outputs/runs/<run-id>/tensorboard
```
