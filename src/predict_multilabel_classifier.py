#!/usr/bin/env python3
"""
Run multi-label inference on images with a trained classifier checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import models
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_CLASS_NAMES: List[str] = [
    "アナグマ",
    "アライグマ",
    "ハクビシン",
    "タヌキ",
    "ネコ",
    "ノウサギ",
    "テン",
]
UNKNOWN_LABEL = "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-label inference on one image or a directory.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Image file path or directory containing images.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./artifacts/predictions/predictions.csv",
        help="Where to save predictions CSV.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "efficientnet_v2_s"],
        help="Override architecture. By default, use checkpoint arch.",
    )
    parser.add_argument("--img-size", type=int, default=0, help="Override input size. 0 uses checkpoint value.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Global threshold in [0,1]. By default, use checkpoint threshold or 0.5.",
    )
    parser.add_argument(
        "--thresholds-json",
        type=str,
        default=None,
        help="JSON string or JSON file path: {\"アナグマ\":0.6, ...}",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help="Optional metrics JSON. If it has `thresholds`, they are applied.",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan directories recursively.")
    parser.add_argument("--max-images", type=int, default=0, help="Max images to run. 0 means all.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
    args, _unknown = parser.parse_known_args()

    if args.img_size < 0:
        parser.error("--img-size must be >= 0")
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold must be in [0, 1]")
    if args.max_images < 0:
        parser.error("--max-images must be >= 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet101":
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet152":
        model = models.resnet152(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {arch}")


def parse_threshold_overrides(value: str, class_names: List[str]) -> Dict[str, float]:
    raw = value.strip()
    p = Path(raw)
    if p.exists() and p.is_file():
        payload = json.loads(p.read_text(encoding="utf-8"))
    else:
        payload = json.loads(raw)

    if not isinstance(payload, dict):
        raise ValueError("--thresholds-json must be a dict")

    out: Dict[str, float] = {}
    for c in class_names:
        if c in payload:
            v = float(payload[c])
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"threshold for {c} must be in [0,1]")
            out[c] = v
    return out


def load_thresholds(
    *,
    class_names: List[str],
    checkpoint: Dict,
    threshold: float | None,
    thresholds_json: str | None,
    metrics_json: str | None,
) -> Dict[str, float]:
    base_threshold = float(checkpoint.get("threshold", 0.5))
    if threshold is not None:
        base_threshold = float(threshold)

    thresholds = {c: base_threshold for c in class_names}

    if metrics_json:
        metrics = json.loads(Path(metrics_json).expanduser().read_text(encoding="utf-8"))
        if isinstance(metrics, dict):
            if "thresholds" in metrics and isinstance(metrics["thresholds"], dict):
                for c in class_names:
                    if c in metrics["thresholds"]:
                        thresholds[c] = float(metrics["thresholds"][c])
            elif "threshold" in metrics:
                t = float(metrics["threshold"])
                thresholds = {c: t for c in class_names}

    if thresholds_json:
        overrides = parse_threshold_overrides(thresholds_json, class_names)
        thresholds.update(overrides)

    for c, v in thresholds.items():
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"threshold for {c} must be in [0,1]")
    return thresholds


def collect_images(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise SystemExit(f"input path does not exist: {input_path}")

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    images = [p for p in iterator if p.is_file() and p.suffix.lower() in exts]
    return sorted(images)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    ckpt_path = Path(args.model_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"model not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise SystemExit("checkpoint missing key: model_state_dict")

    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    if not isinstance(class_names, list) or not class_names:
        class_names = DEFAULT_CLASS_NAMES
    class_names = [str(x) for x in class_names]

    arch = args.arch or str(checkpoint.get("arch", "resnet18"))
    img_size = int(args.img_size or checkpoint.get("img_size", 224))
    thresholds = load_thresholds(
        class_names=class_names,
        checkpoint=checkpoint,
        threshold=args.threshold,
        thresholds_json=args.thresholds_json,
        metrics_json=args.metrics_json,
    )

    model = build_model(arch=arch, num_classes=len(class_names))
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError:
        # Fallback to relaxed loading for checkpoints with non-critical key mismatches.
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device).eval()

    tf = T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_path = Path(args.input_path).expanduser().resolve()
    image_paths = collect_images(input_path, recursive=args.recursive)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        raise SystemExit("no images found")

    out_csv = Path(args.output_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    label_counts = {c: 0 for c in class_names}
    unknown_count = 0

    with torch.inference_mode():
        for i in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[i : i + args.batch_size]
            batch_tensors = []
            valid_paths = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_tensors.append(tf(img))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"[WARN] skip unreadable image: {p} ({e})")

            if not batch_tensors:
                continue

            x = torch.stack(batch_tensors, dim=0).to(device)
            probs = torch.sigmoid(model(x)).detach().cpu().numpy()

            for path, prob in zip(valid_paths, probs):
                labels = [c for c, p in zip(class_names, prob) if float(p) >= thresholds[c]]
                if not labels:
                    labels = [UNKNOWN_LABEL]
                    unknown_count += 1
                else:
                    for c in labels:
                        label_counts[c] += 1

                row: Dict[str, object] = {
                    "image_path": str(path),
                    "predicted_labels_json": json.dumps(labels, ensure_ascii=False),
                    "predicted_labels_text": "|".join(labels),
                    "is_unknown": int(labels == [UNKNOWN_LABEL]),
                }
                for c, p in zip(class_names, prob):
                    row[f"prob_{c}"] = float(p)
                rows.append(row)

    fieldnames = ["image_path", "predicted_labels_json", "predicted_labels_text", "is_unknown"] + [
        f"prob_{c}" for c in class_names
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"device: {device}")
    print(f"arch: {arch}")
    print(f"img_size: {img_size}")
    print(f"images: {len(rows)}")
    print(f"unknown: {unknown_count}")
    print(f"label_counts: {label_counts}")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
