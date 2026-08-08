"""Train U3Net on the human segmentation Parquet dataset."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# Some newer AMD GPUs do not yet have a packaged MIOpen Find-Db entry.  FAST
# avoids the solver benchmarking path (EvaluateInvokers) on a database miss and
# falls back to immediate mode instead.  Set these before importing torch so
# they are visible when MIOpen is initialized.  Existing user settings win.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import HumanSegDataset, get_train_transform, get_valid_transform
from loss import DeepSupervisionLoss
from model import U3Net, modern_u2net_tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Modern U3Net")
    parser.add_argument("--data", type=str, default="humanseg.parquet")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny", action="store_true", help="use the smaller model")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_data = HumanSegDataset(
        args.data,
        transform=get_train_transform(args.image_size, args.image_size),
    )
    valid_data = HumanSegDataset(
        args.data,
        transform=get_valid_transform(args.image_size, args.image_size),
    )
    sample_count = len(train_data)
    if sample_count < 2:
        raise ValueError("training requires at least two samples")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(sample_count, generator=generator).tolist()
    valid_count = max(1, int(round(sample_count * args.val_ratio)))
    valid_count = min(valid_count, sample_count - 1)
    valid_indices = indices[:valid_count]
    train_indices = indices[valid_count:]

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        Subset(train_data, train_indices),
        shuffle=True,
        drop_last=len(train_indices) >= args.batch_size,
        **loader_options,
    )
    valid_loader = DataLoader(
        Subset(valid_data, valid_indices),
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    return train_loader, valid_loader


def segmentation_scores(
        probability: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-7,
) -> tuple[float, float]:
    prediction = probability >= 0.5
    target = target >= 0.5
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction & target).sum(dim=dims).float()
    pred_sum = prediction.sum(dim=dims).float()
    target_sum = target.sum(dim=dims).float()
    union = (prediction | target).sum(dim=dims).float()
    dice = ((2 * intersection + eps) / (pred_sum + target_sum + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()
    return dice.item(), iou.item()


def train_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        device: torch.device,
        amp_enabled: bool,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size

    return total_loss / seen


@torch.inference_mode()
def validate(
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        amp_enabled: bool,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    seen = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, masks)

        fused = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        dice, iou = segmentation_scores(fused, masks)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice * batch_size
        total_iou += iou * batch_size
        seen += batch_size

    return total_loss / seen, total_dice / seen, total_iou / seen


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # On ROCm this API is backed by MIOpen.  Benchmarking repeats the solver
        # search that FAST mode above is intended to avoid.
        torch.backends.cudnn.benchmark = False
    amp_enabled = device.type == "cuda" and not args.no_amp
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader = create_loaders(args)
    model = (
        modern_u2net_tiny(in_ch=3, out_ch=1)
        if args.tiny
        else U3Net(in_ch=3, out_ch=1)
    ).to(device)
    criterion = DeepSupervisionLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best_dice = -1.0

    print(
        f"device={device}, train={len(train_loader.dataset)}, "
        f"valid={len(valid_loader.dataset)}, amp={amp_enabled}, "
        f"miopen_find_mode={os.environ['MIOPEN_FIND_MODE']}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp_enabled
        )
        valid_loss, dice, iou = validate(
            model, valid_loader, criterion, device, amp_enabled
        )
        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_dice": max(best_dice, dice),
            "args": vars(args),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if dice > best_dice:
            best_dice = dice
            torch.save(checkpoint, output_dir / "best.pt")

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | lr {lr:.2e} | "
            f"train_loss {train_loss:.4f} | val_loss {valid_loss:.4f} | "
            f"dice {dice:.4f} | iou {iou:.4f}"
        )


if __name__ == "__main__":
    main()
