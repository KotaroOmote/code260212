#!/usr/bin/env python3
"""
Train and evaluate a multi-label classifier from CSV splits.

Expected CSV columns:
- frame_source_path
- label_<class_name> for each target class
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_NAMES: List[str] = [
    "アナグマ",
    "アライグマ",
    "ハクビシン",
    "タヌキ",
    "ネコ",
    "ノウサギ",
    "テン",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-label wildlife classifier.")
    parser.add_argument(
        "--train-csv",
        type=str,
        default="./artifacts/metadata/train_known.csv",
        help="Path to train split CSV.",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="./artifacts/metadata/val_known.csv",
        help="Path to validation split CSV.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="./artifacts/metadata/test_known.csv",
        help="Path to test split CSV.",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="./artifacts/models/multilabel_resnet18_best.pt",
        help="Output path for best model checkpoint.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="./artifacts/models/multilabel_test_metrics.json",
        help="Output path for final test metrics (JSON).",
    )
    parser.add_argument(
        "--history-out",
        type=str,
        default="./artifacts/models/multilabel_train_history.csv",
        help="Output path for train/val history CSV.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for positive labels.")
    parser.add_argument("--img-size", type=int, default=224, help="Image size for model input.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "efficientnet_v2_s",
        ],
        help="Backbone architecture.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not use ImageNet pretrained weights.",
    )
    parser.add_argument("--max-train", type=int, default=0, help="Limit train rows for quick tests. 0 means all.")
    parser.add_argument("--max-val", type=int, default=0, help="Limit val rows for quick tests. 0 means all.")
    parser.add_argument("--max-test", type=int, default=0, help="Limit test rows for quick tests. 0 means all.")
    args, _unknown = parser.parse_known_args()

    if args.epochs <= 0:
        parser.error("--epochs must be > 0.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")
    if args.lr <= 0:
        parser.error("--lr must be > 0.")
    if not (0.0 < args.threshold < 1.0):
        parser.error("--threshold must be in (0, 1).")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def label_columns() -> List[str]:
    return [f"label_{c}" for c in CLASS_NAMES]


def read_split(csv_path: Path, max_rows: int = 0) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"split csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = ["frame_source_path"] + label_columns()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns in {csv_path}: {missing}")

    df = df.copy()
    df["frame_source_path"] = df["frame_source_path"].astype(str)
    exists = df["frame_source_path"].apply(lambda p: Path(p).exists())
    dropped = int((~exists).sum())
    if dropped > 0:
        print(f"[WARN] drop missing files from {csv_path}: {dropped}")
    df = df[exists].reset_index(drop=True)

    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df


class MultiLabelCsvDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: T.Compose):
        self.paths = df["frame_source_path"].tolist()
        self.labels = df[label_columns()].values.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, target


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif arch == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    elif arch == "resnet101":
        weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet101(weights=weights)
    elif arch == "resnet152":
        weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet152(weights=weights)
    elif arch == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    else:  # pragma: no cover
        raise ValueError(f"Unsupported architecture: {arch}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(mode=training)

    total_loss = 0.0
    y_true_batches: List[np.ndarray] = []
    y_pred_batches: List[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        preds = (torch.sigmoid(logits).detach().cpu().numpy() >= threshold).astype(int)
        y_pred_batches.append(preds)
        y_true_batches.append(targets.detach().cpu().numpy().astype(int))

    if not y_true_batches:
        y_true = np.zeros((0, len(CLASS_NAMES)), dtype=int)
        y_pred = np.zeros((0, len(CLASS_NAMES)), dtype=int)
    else:
        y_true = np.vstack(y_true_batches)
        y_pred = np.vstack(y_pred_batches)

    avg_loss = total_loss / max(1, len(loader.dataset))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return avg_loss, micro_f1, macro_f1, y_true, y_pred


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        out[class_name] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support_pos": int(yt.sum()),
        }
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")

    train_df = read_split(Path(args.train_csv).expanduser().resolve(), max_rows=args.max_train)
    val_df = read_split(Path(args.val_csv).expanduser().resolve(), max_rows=args.max_val)
    test_df = read_split(Path(args.test_csv).expanduser().resolve(), max_rows=args.max_test)
    print(f"train: {len(train_df)} val: {len(val_df)} test: {len(test_df)}")

    train_tf = T.Compose(
        [
            T.Resize((256, 256)),
            T.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop(args.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = MultiLabelCsvDataset(train_df, train_tf)
    val_ds = MultiLabelCsvDataset(val_df, eval_tf)
    test_ds = MultiLabelCsvDataset(test_df, eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        arch=args.arch,
        num_classes=len(CLASS_NAMES),
        pretrained=(not args.no_pretrained),
    )
    model = model.to(device)

    train_labels = train_df[label_columns()].values.astype(np.float32)
    pos = train_labels.sum(axis=0)
    neg = len(train_labels) - pos
    pos_weight = np.clip(neg / np.clip(pos, 1.0, None), 1.0, 20.0)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    model_out = Path(args.model_out).expanduser().resolve()
    history_out = Path(args.history_out).expanduser().resolve()
    metrics_out = Path(args.metrics_out).expanduser().resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)
    history_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    best_val_micro = -1.0
    history_rows: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_micro, train_macro, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            threshold=args.threshold,
            optimizer=optimizer,
        )
        val_loss, val_micro, val_macro, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device=device,
            threshold=args.threshold,
            optimizer=None,
        )
        scheduler.step(val_loss)

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"train loss={train_loss:.4f} microF1={train_micro:.4f} macroF1={train_macro:.4f} | "
            f"val loss={val_loss:.4f} microF1={val_micro:.4f} macroF1={val_macro:.4f}"
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_micro_f1": train_micro,
                "train_macro_f1": train_macro,
                "val_loss": val_loss,
                "val_micro_f1": val_micro,
                "val_macro_f1": val_macro,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_micro > best_val_micro:
            best_val_micro = val_micro
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "threshold": args.threshold,
                    "img_size": args.img_size,
                    "seed": args.seed,
                    "arch": args.arch,
                },
                model_out,
            )
            print(f"  -> saved best: {model_out}")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_out, index=False)
    print(f"saved history: {history_out}")

    checkpoint = torch.load(model_out, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_micro, test_macro, y_true, y_pred = run_epoch(
        model,
        test_loader,
        criterion,
        device=device,
        threshold=args.threshold,
        optimizer=None,
    )
    per_class = class_metrics(y_true, y_pred)

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "class_names": CLASS_NAMES,
        "threshold": float(args.threshold),
        "arch": args.arch,
        "best_val_micro_f1": float(best_val_micro),
        "test": {
            "loss": float(test_loss),
            "micro_f1": float(test_micro),
            "macro_f1": float(test_macro),
        },
        "per_class": per_class,
    }
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(
        f"[TEST] loss={test_loss:.4f} microF1={test_micro:.4f} macroF1={test_macro:.4f}\n"
        f"saved model: {model_out}\n"
        f"saved metrics: {metrics_out}"
    )


if __name__ == "__main__":
    main()
