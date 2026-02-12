#!/usr/bin/env python3
"""
Dataset builder for 7-class wildlife classification.

What this script does:
1) Collect images from iNaturalist API (optional)
2) Extract frames from a local video (optional)
3) Save metadata/sources.csv
4) Create metadata/train.csv, metadata/val.csv, metadata/test.csv

Example (Colab):
python notebooks/01_make_dataset_colab.py \
  --project-root /content/drive/MyDrive/code260212 \
  --max-images-per-class 120 \
  --max-pages 8

Example (local):
python notebooks/01_make_dataset_colab.py \
  --project-root "/Users/k.omote/Documents/New project/code260212" \
  --max-images-per-class 100
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

INAT_ENDPOINT = "https://api.inaturalist.org/v1/observations"
PHOTO_LICENSE = "cc0,cc-by,cc-by-sa,cc-by-nc"

CLASS_TAXA: Dict[str, str] = {
    "アナグマ": "Meles anakuma",
    "アライグマ": "Procyon lotor",
    "ハクビシン": "Paguma larvata",
    "タヌキ": "Nyctereutes viverrinus",
    "ネコ": "Felis catus",
    "ノウサギ": "Lepus brachyurus",
    "テン": "Martes melampus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset metadata for 7-class wildlife detection.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path.")
    parser.add_argument("--max-images-per-class", type=int, default=300, help="Max iNaturalist images/class.")
    parser.add_argument("--max-pages", type=int, default=15, help="Max iNaturalist pages/class.")
    parser.add_argument("--per-page", type=int, default=200, help="iNaturalist per_page (max 200).")
    parser.add_argument("--place-id", type=str, default=None, help="Optional iNaturalist place_id.")
    parser.add_argument("--skip-inat", action="store_true", help="Skip iNaturalist collection.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--video-path", type=str, default=None, help="Optional local video file path.")
    parser.add_argument("--video-class", type=str, default=None, help="Class name for extracted video frames.")
    parser.add_argument("--every-n-frames", type=int, default=10, help="Extract one frame every N frames.")
    parser.add_argument("--max-video-frames", type=int, default=200, help="Max frames extracted from video.")
    # In notebook runtimes (Colab/Jupyter), extra kernel args like "-f <kernel.json>"
    # may be injected. Ignore unknown args to keep CLI and notebook execution compatible.
    args, _unknown = parser.parse_known_args()
    return args


def resolve_project_root(project_root_arg: Optional[str]) -> Path:
    if project_root_arg:
        return Path(project_root_arg).expanduser().resolve()

    in_colab = "google.colab" in sys.modules
    if in_colab:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        return Path("/content/drive/MyDrive/code260212")

    return Path.cwd().resolve()


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))


def inat_large_url(photo: dict) -> str:
    url = photo.get("url", "")
    if "/square." in url:
        return url.replace("/square.", "/large.")
    return url.replace("square", "large")


def download_image(url: str, out_path: Path, timeout: int = 30) -> bool:
    try:
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            f.write(res.content)
        return True
    except Exception:
        return False


def collect_inat_images(
    raw_dir: Path,
    project_root: Path,
    class_name: str,
    taxon_name: str,
    max_images: int,
    max_pages: int,
    per_page: int,
    place_id: Optional[str],
) -> List[dict]:
    rows: List[dict] = []
    out_dir = raw_dir / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    downloaded = 0

    for page in range(1, max_pages + 1):
        params = {
            "taxon_name": taxon_name,
            "quality_grade": "research",
            "photos": "true",
            "photo_license": PHOTO_LICENSE,
            "per_page": min(per_page, 200),
            "page": page,
        }
        if place_id:
            params["place_id"] = place_id

        resp = session.get(INAT_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break

        for obs in results:
            obs_id = obs.get("id")
            obs_url = obs.get("uri")
            observed_on = obs.get("observed_on")
            for photo in obs.get("photos", []):
                if downloaded >= max_images:
                    break
                photo_id = photo.get("id")
                license_code = photo.get("license_code")
                image_url = inat_large_url(photo)
                file_name = safe_name(f"inat_{obs_id}_{photo_id}.jpg")
                local_path = out_dir / file_name

                if not (local_path.exists() or download_image(image_url, local_path)):
                    continue

                downloaded += 1
                rows.append(
                    {
                        "class_name": class_name,
                        "taxon_name": taxon_name,
                        "source_dataset": "iNaturalist",
                        "observation_id": obs_id,
                        "photo_id": photo_id,
                        "observed_on": observed_on,
                        "observation_url": obs_url,
                        "source_url": image_url,
                        "license_code": license_code,
                        "file_path": str(local_path.relative_to(project_root)),
                    }
                )

            if downloaded >= max_images:
                break

        if downloaded >= max_images:
            break

        time.sleep(0.3)

    return rows


def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    every_n_frames: int,
    max_frames: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    saved_paths: List[Path] = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n_frames == 0:
            out_name = f"{video_path.stem}_f{frame_idx:06d}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved_paths.append(out_path)
            if len(saved_paths) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved_paths


def make_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    x = df.copy().dropna(subset=["class_name", "file_path"]).reset_index(drop=True)
    if x.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        train_val, test = train_test_split(
            x, test_size=test_size, random_state=seed, stratify=x["class_name"]
        )
        val_ratio_in_trainval = val_size / (1.0 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_in_trainval,
            random_state=seed,
            stratify=train_val["class_name"],
        )
    except ValueError:
        # Fallback for tiny class counts where stratified split fails.
        x = x.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(x)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        test = x.iloc[:n_test]
        val = x.iloc[n_test : n_test + n_val]
        train = x.iloc[n_test + n_val :]

    return train, val, test


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    raw_dir = project_root / "data" / "raw"
    meta_dir = project_root / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"PROJECT_ROOT: {project_root}")
    print(f"RAW_DIR: {raw_dir}")
    print(f"CLASSES: {', '.join(CLASS_TAXA.keys())}")

    all_rows: List[dict] = []

    if not args.skip_inat:
        for class_name, taxon_name in CLASS_TAXA.items():
            rows = collect_inat_images(
                raw_dir=raw_dir,
                project_root=project_root,
                class_name=class_name,
                taxon_name=taxon_name,
                max_images=args.max_images_per_class,
                max_pages=args.max_pages,
                per_page=args.per_page,
                place_id=args.place_id,
            )
            all_rows.extend(rows)
            print(f"{class_name}: {len(rows)} images")

    if args.video_path:
        if not args.video_class:
            raise ValueError("--video-path was set, but --video-class is missing.")
        if args.video_class not in CLASS_TAXA:
            raise ValueError(f"--video-class must be one of: {', '.join(CLASS_TAXA.keys())}")

        video_path = Path(args.video_path).expanduser().resolve()
        out_dir = raw_dir / args.video_class
        saved_paths = extract_frames_from_video(
            video_path=video_path,
            out_dir=out_dir,
            every_n_frames=args.every_n_frames,
            max_frames=args.max_video_frames,
        )
        for file_path in saved_paths:
            all_rows.append(
                {
                    "class_name": args.video_class,
                    "taxon_name": CLASS_TAXA[args.video_class],
                    "source_dataset": "user_video",
                    "observation_id": None,
                    "photo_id": None,
                    "observed_on": None,
                    "observation_url": None,
                    "source_url": str(video_path),
                    "license_code": "user_data",
                    "file_path": str(file_path.relative_to(project_root)),
                }
            )
        print(f"video frames extracted: {len(saved_paths)}")

    new_df = pd.DataFrame(all_rows)
    sources_csv = meta_dir / "sources.csv"

    if sources_csv.exists() and sources_csv.stat().st_size > 0:
        old_df = pd.read_csv(sources_csv)
    else:
        old_df = pd.DataFrame()

    merged = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    if not merged.empty:
        merged = merged.drop_duplicates(subset=["file_path"], keep="last")
    merged.to_csv(sources_csv, index=False)
    print(f"saved: {sources_csv}")

    train_df, val_df, test_df = make_split(
        merged, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )
    train_path = meta_dir / "train.csv"
    val_path = meta_dir / "val.csv"
    test_path = meta_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"train/val/test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"saved: {train_path}")
    print(f"saved: {val_path}")
    print(f"saved: {test_path}")


if __name__ == "__main__":
    main()
