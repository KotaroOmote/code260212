#!/usr/bin/env python3
"""
Build a reproducible multi-label dataset CSV from OpenAI annotation logs.

Inputs:
- image annotations CSV
- video annotations CSV

Outputs:
- final_multilabel_dataset.csv
- final_multilabel_dataset_known_only.csv
- unknown_samples.csv
- train_known.csv / val_known.csv / test_known.csv
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_NAMES: List[str] = [
    "アナグマ",
    "アライグマ",
    "ハクビシン",
    "タヌキ",
    "ネコ",
    "ノウサギ",
    "テン",
]
UNKNOWN_LABEL = "unknown"

BASE_COLS = [
    "video_path",
    "video_id",
    "frame_index",
    "timestamp_sec",
    "frame_source_path",
    "assigned_labels_json",
    "confidence",
    "reason",
    "model",
    "response_id",
    "response_text",
    "copied_paths_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final multi-label dataset CSV files.")
    parser.add_argument(
        "--image-csv",
        type=str,
        default="/content/drive/MyDrive/RG/metadata/openai_image_annotations.csv",
        help="Path to image annotation CSV.",
    )
    parser.add_argument(
        "--video-csv",
        type=str,
        default="/content/drive/MyDrive/RG/metadata/openai_video_annotations.csv",
        help="Path to video annotation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/content/drive/MyDrive/RG/metadata",
        help="Output directory for final CSV files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio.")
    parser.add_argument(
        "--dedupe-key",
        type=str,
        default="frame_source_path",
        choices=["frame_source_path", "video_path_frame_index"],
        help="Deduplication key.",
    )
    args, _unknown = parser.parse_known_args()

    if args.val_ratio <= 0 or args.test_ratio <= 0:
        parser.error("--val-ratio and --test-ratio must be > 0.")
    if args.val_ratio + args.test_ratio >= 1.0:
        parser.error("val_ratio + test_ratio must be < 1.0.")
    return args


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "source_path" in df.columns and "video_path" not in df.columns:
        rename_map["source_path"] = "video_path"
    if "source_id" in df.columns and "video_id" not in df.columns:
        rename_map["source_id"] = "video_id"
    if "item_index" in df.columns and "frame_index" not in df.columns:
        rename_map["item_index"] = "frame_index"
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in BASE_COLS:
        if col not in df.columns:
            df[col] = None

    return df[BASE_COLS].copy()


def parse_labels(value) -> List[str]:
    if pd.isna(value):
        return [UNKNOWN_LABEL]

    raw = None
    if isinstance(value, list):
        raw = value
    else:
        s = str(value).strip()
        if not s:
            return [UNKNOWN_LABEL]
        try:
            raw = json.loads(s)
        except Exception:
            try:
                raw = ast.literal_eval(s)
            except Exception:
                raw = [s]

    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return [UNKNOWN_LABEL]

    labels: List[str] = []
    for x in raw:
        item = str(x).strip()
        if not item:
            continue
        if item in CLASS_NAMES or item == UNKNOWN_LABEL:
            if item not in labels:
                labels.append(item)

    # If at least one valid target class exists, remove unknown.
    if any(label in CLASS_NAMES for label in labels):
        labels = [label for label in labels if label != UNKNOWN_LABEL]

    return labels or [UNKNOWN_LABEL]


def sample_id_for_row(row: pd.Series) -> str:
    source = f"{row.get('video_path')}::{row.get('frame_index')}::{row.get('frame_source_path')}"
    return hashlib.md5(source.encode("utf-8")).hexdigest()[:16]


def make_splits(
    known_df: pd.DataFrame,
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    video_df = known_df[known_df["source_type"] == "video"].copy()
    image_df = known_df[known_df["source_type"] == "image"].copy()

    def split_image_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df.copy(), df.copy(), df.copy()

        holdout = val_ratio + test_ratio
        stratify = df["labels_text"] if df["labels_text"].nunique() > 1 else None
        try:
            train, rest = train_test_split(df, test_size=holdout, random_state=seed, stratify=stratify)
            rest_val_ratio = val_ratio / holdout
            rest_stratify = rest["labels_text"] if rest["labels_text"].nunique() > 1 else None
            val, test = train_test_split(
                rest,
                test_size=(1.0 - rest_val_ratio),
                random_state=seed,
                stratify=rest_stratify,
            )
        except ValueError:
            train, rest = train_test_split(df, test_size=holdout, random_state=seed)
            rest_val_ratio = val_ratio / holdout
            val, test = train_test_split(rest, test_size=(1.0 - rest_val_ratio), random_state=seed)
        return train, val, test

    def split_video_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df.copy(), df.copy(), df.copy()

        groups = sorted(df["video_id"].astype(str).dropna().unique().tolist())
        if len(groups) < 3:
            # Not enough groups for strict group split. Fall back to row split.
            return split_image_rows(df)

        holdout = val_ratio + test_ratio
        g_train, g_rest = train_test_split(groups, test_size=holdout, random_state=seed)
        rest_val_ratio = val_ratio / holdout
        g_val, g_test = train_test_split(g_rest, test_size=(1.0 - rest_val_ratio), random_state=seed)

        train = df[df["video_id"].astype(str).isin(set(g_train))].copy()
        val = df[df["video_id"].astype(str).isin(set(g_val))].copy()
        test = df[df["video_id"].astype(str).isin(set(g_test))].copy()
        return train, val, test

    v_train, v_val, v_test = split_video_groups(video_df)
    i_train, i_val, i_test = split_image_rows(image_df)

    train_df = pd.concat([v_train, i_train], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(
        drop=True
    )
    val_df = pd.concat([v_val, i_val], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat([v_test, i_test], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(
        drop=True
    )
    return train_df, val_df, test_df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_csv = Path(args.image_csv).expanduser().resolve()
    video_csv = Path(args.video_csv).expanduser().resolve()
    if not image_csv.exists():
        raise SystemExit(f"image csv not found: {image_csv}")
    if not video_csv.exists():
        raise SystemExit(f"video csv not found: {video_csv}")

    image_df = normalize_columns(pd.read_csv(image_csv))
    image_df["source_type"] = "image"
    video_df = normalize_columns(pd.read_csv(video_csv))
    video_df["source_type"] = "video"

    all_df = pd.concat([image_df, video_df], ignore_index=True)
    all_df["confidence"] = pd.to_numeric(all_df["confidence"], errors="coerce").fillna(0.0)
    all_df["frame_source_path"] = all_df["frame_source_path"].astype(str)
    all_df["video_path"] = all_df["video_path"].astype(str)
    all_df["frame_index"] = pd.to_numeric(all_df["frame_index"], errors="coerce").fillna(-1).astype(int)
    all_df["labels"] = all_df["assigned_labels_json"].apply(parse_labels)

    if args.dedupe_key == "frame_source_path":
        dedupe_cols = ["frame_source_path"]
    else:
        dedupe_cols = ["video_path", "frame_index"]
    all_df = (
        all_df.sort_values("confidence", ascending=False)
        .drop_duplicates(subset=dedupe_cols, keep="first")
        .reset_index(drop=True)
    )

    all_df["labels_json"] = all_df["labels"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    all_df["labels_text"] = all_df["labels"].apply(lambda x: "|".join(x))
    all_df["num_labels"] = all_df["labels"].apply(len)
    all_df["is_unknown"] = all_df["labels"].apply(lambda x: int(x == [UNKNOWN_LABEL]))
    for class_name in CLASS_NAMES:
        all_df[f"label_{class_name}"] = all_df["labels"].apply(lambda x, c=class_name: int(c in x))
    all_df["label_unknown"] = all_df["labels"].apply(lambda x: int(UNKNOWN_LABEL in x))
    all_df["sample_id"] = all_df.apply(sample_id_for_row, axis=1)

    final_cols = [
        "sample_id",
        "source_type",
        "video_path",
        "video_id",
        "frame_index",
        "timestamp_sec",
        "frame_source_path",
        "confidence",
        "reason",
        "model",
        "labels_json",
        "labels_text",
        "num_labels",
        "is_unknown",
    ] + [f"label_{c}" for c in CLASS_NAMES] + ["label_unknown"]

    final_df = all_df[final_cols].copy()
    known_df = final_df[final_df["label_unknown"] == 0].copy()
    unknown_df = final_df[final_df["label_unknown"] == 1].copy()

    train_df, val_df, test_df = make_splits(
        known_df,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    final_path = out_dir / "final_multilabel_dataset.csv"
    known_path = out_dir / "final_multilabel_dataset_known_only.csv"
    unknown_path = out_dir / "unknown_samples.csv"
    train_path = out_dir / "train_known.csv"
    val_path = out_dir / "val_known.csv"
    test_path = out_dir / "test_known.csv"

    final_df.to_csv(final_path, index=False, encoding="utf-8-sig")
    known_df.to_csv(known_path, index=False, encoding="utf-8-sig")
    unknown_df.to_csv(unknown_path, index=False, encoding="utf-8-sig")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    def overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
        a_set = set(a[a["source_type"] == "video"]["video_id"].astype(str))
        b_set = set(b[b["source_type"] == "video"]["video_id"].astype(str))
        return len(a_set & b_set)

    print(f"saved: {final_path} rows={len(final_df)}")
    print(f"saved: {known_path} rows={len(known_df)}")
    print(f"saved: {unknown_path} rows={len(unknown_df)}")
    print(f"saved: {train_path} rows={len(train_df)}")
    print(f"saved: {val_path} rows={len(val_df)}")
    print(f"saved: {test_path} rows={len(test_df)}")
    print(f"source counts: {dict(final_df['source_type'].value_counts())}")
    label_totals: Dict[str, int] = {c: int(final_df[f'label_{c}'].sum()) for c in CLASS_NAMES}
    label_totals[UNKNOWN_LABEL] = int(final_df["label_unknown"].sum())
    print(f"label counts: {label_totals}")
    print(
        "video leakage "
        f"train∩val={overlap(train_df, val_df)}, "
        f"train∩test={overlap(train_df, test_df)}, "
        f"val∩test={overlap(val_df, test_df)}"
    )


if __name__ == "__main__":
    main()
