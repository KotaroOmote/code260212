#!/usr/bin/env python3
"""
Auto-annotate wildlife videos and images with OpenAI vision models.

Features:
- Extract frames from videos at fixed intervals.
- Classify each frame/image into 7 target classes (multi-label allowed) or "unknown".
- Copy one frame/image into multiple class folders when multiple animals are present.
- Save audit metadata CSV for reproducibility and resume.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2

TARGET_CLASSES: List[str] = [
    "アナグマ",
    "アライグマ",
    "ハクビシン",
    "タヌキ",
    "ネコ",
    "ノウサギ",
    "テン",
]
UNKNOWN_LABEL = "unknown"
ALL_ALLOWED: Set[str] = set(TARGET_CLASSES + [UNKNOWN_LABEL])

ALIAS_MAP: Dict[str, str] = {
    "badger": "アナグマ",
    "japanese badger": "アナグマ",
    "raccoon": "アライグマ",
    "masked palm civet": "ハクビシン",
    "civet": "ハクビシン",
    "raccoon dog": "タヌキ",
    "tanuki": "タヌキ",
    "cat": "ネコ",
    "domestic cat": "ネコ",
    "hare": "ノウサギ",
    "japanese hare": "ノウサギ",
    "marten": "テン",
    "japanese marten": "テン",
    "none": UNKNOWN_LABEL,
    "other": UNKNOWN_LABEL,
    "others": UNKNOWN_LABEL,
    "unknown": UNKNOWN_LABEL,
    "uncertain": UNKNOWN_LABEL,
}

CSV_HEADERS = [
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

SYSTEM_PROMPT = (
    "You are a strict wildlife image annotator for camera trap and field video frames. "
    "Return JSON only."
)

USER_PROMPT_TEMPLATE = """\
次の画像を、以下の7クラスだけで判定してください。
対象クラス: アナグマ, アライグマ, ハクビシン, タヌキ, ネコ, ノウサギ, テン

ルール:
1) 複数種が同時に写っていれば labels に複数入れる。
2) 対象外の動物・人工物・風景のみ・判別不能(暗い/ブレ/遠すぎ)は labels=["unknown"]。
3) 推測しすぎない。明確に見えるものだけを採用。
4) 出力は必ずJSONのみ。説明文をJSON外に書かない。

JSON形式:
{
  "labels": ["タヌキ", "ネコ"],
  "confidence": 0.0,
  "reason": "短い理由"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate video frames and images with OpenAI.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data/input_videos",
        help="Directory containing input videos (recursive scan).",
    )
    parser.add_argument(
        "--video-list-file",
        type=str,
        default=None,
        help="Optional text file with one video path per line. If set, this overrides input-dir scan.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./data/raw",
        help="Directory containing input images (recursive scan).",
    )
    parser.add_argument(
        "--image-list-file",
        type=str,
        default=None,
        help="Optional text file with one image path per line. If set, this overrides image-dir scan.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./artifacts/ai_annotated_frames",
        help="Root output folder. Class folders and _frames are created here.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="./artifacts/metadata/openai_video_annotations.csv",
        help="CSV path for annotation logs.",
    )
    parser.add_argument("--model", type=str, default="gpt-5.2", help="OpenAI model name.")
    parser.add_argument(
        "--sample-every-n-frames",
        type=int,
        default=45,
        help="Sample one frame every N frames.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=180,
        help="Maximum sampled frames per video. <=0 means no cap.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Maximum videos to process. 0 means all.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum images to process. 0 means all.",
    )
    parser.add_argument(
        "--image-max-side",
        type=int,
        default=1024,
        help="Resize frame before API call if width/height exceeds this value.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for API input/storage.")
    parser.add_argument(
        "--detail",
        type=str,
        default="low",
        choices=["low", "high", "auto"],
        help="Vision detail level.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help="Optional reasoning effort hint for supported models.",
    )
    parser.add_argument("--retry-count", type=int, default=3, help="Retries per API call.")
    parser.add_argument("--api-sleep-sec", type=float, default=0.0, help="Sleep between API calls.")
    parser.add_argument(
        "--continue-on-api-error",
        action="store_true",
        help="Continue processing and assign unknown when API keeps failing.",
    )
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)

    args, _unknown = parser.parse_known_args()
    if args.sample_every_n_frames <= 0:
        parser.error("--sample-every-n-frames must be > 0.")
    if args.image_max_side <= 0:
        parser.error("--image-max-side must be > 0.")
    if not (1 <= args.jpeg_quality <= 100):
        parser.error("--jpeg-quality must be in [1, 100].")
    if args.retry_count <= 0:
        parser.error("--retry-count must be > 0.")
    return args


def load_api_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY")
    # Ignore placeholder-like values that are commonly pasted in examples.
    if key and key.strip() and key.strip().upper() not in {
        "YOUR_API_KEY",
        "YOUR_OPENAI_API_KEY",
        "OPENAI_API_KEY",
    }:
        return key

    # Optional fallback for Colab Secrets.
    if "google.colab" in sys.modules:
        try:
            from google.colab import userdata  # type: ignore

            key = userdata.get("OPENAI_API_KEY")
            if key:
                os.environ["OPENAI_API_KEY"] = key
                return key
        except Exception:
            return None
    return None


def load_paths_from_list(list_file: Path, kind: str) -> List[Path]:
    paths: List[Path] = []
    with list_file.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            p = Path(text).expanduser().resolve()
            if p.exists():
                paths.append(p)
            else:
                print(f"[WARN] Missing {kind} path in list file: {p}")
    return sorted(set(paths))


def load_video_paths(input_dir: Path, video_list_file: Optional[Path]) -> List[Path]:
    if video_list_file:
        return load_paths_from_list(video_list_file, "video")

    suffixes = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".mts"}
    videos = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
    return sorted(videos)


def load_image_paths(image_dir: Path, image_list_file: Optional[Path]) -> List[Path]:
    if image_list_file:
        return load_paths_from_list(image_list_file, "image")

    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    images = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
    return sorted(images)


def iterate_sampled_frames(
    video_path: Path, sample_every_n_frames: int, max_frames_per_video: int
) -> Iterable[Tuple[int, float, "cv2.typing.MatLike"]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    frame_index = 0
    sampled = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % sample_every_n_frames == 0:
                timestamp_sec = frame_index / fps
                yield frame_index, timestamp_sec, frame
                sampled += 1
                if max_frames_per_video > 0 and sampled >= max_frames_per_video:
                    break

            frame_index += 1
    finally:
        cap.release()


def encode_frame_to_jpeg(frame, image_max_side: int, jpeg_quality: int) -> bytes:
    h, w = frame.shape[:2]
    max_hw = max(h, w)
    if max_hw > image_max_side:
        scale = image_max_side / float(max_hw)
        frame = cv2.resize(
            frame,
            dsize=(max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to encode frame into JPEG.")
    return bytes(buf)


def encode_image_file_to_jpeg(image_path: Path, image_max_side: int, jpeg_quality: int) -> bytes:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return encode_frame_to_jpeg(img, image_max_side=image_max_side, jpeg_quality=jpeg_quality)


def path_id(path: Path) -> str:
    return hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]


def extract_json_object(text: str) -> Optional[dict]:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_label_item(item: str) -> Optional[str]:
    label = str(item).strip()
    if not label:
        return None
    if label in ALL_ALLOWED:
        return label

    key = label.lower().strip()
    key = re.sub(r"\s+", " ", key)
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]

    return None


def normalize_labels(raw_labels) -> List[str]:
    if isinstance(raw_labels, str):
        pieces = [p.strip() for p in re.split(r"[,|/]", raw_labels) if p.strip()]
    elif isinstance(raw_labels, list):
        pieces = [str(x).strip() for x in raw_labels if str(x).strip()]
    else:
        pieces = []

    labels: List[str] = []
    for item in pieces:
        normalized = normalize_label_item(item)
        if normalized and normalized not in labels:
            labels.append(normalized)

    has_target = any(label in TARGET_CLASSES for label in labels)
    if has_target:
        labels = [label for label in labels if label != UNKNOWN_LABEL]

    if not labels:
        labels = [UNKNOWN_LABEL]
    return labels


def extract_response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(response, "output", None)
    if not output:
        return ""

    parts: List[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if not content:
            continue
        for c in content:
            c_text = getattr(c, "text", None)
            if isinstance(c_text, str) and c_text.strip():
                parts.append(c_text.strip())
    return "\n".join(parts)


def classify_frame_with_openai(
    client: Any,
    *,
    model: str,
    jpeg_bytes: bytes,
    detail: str,
    reasoning_effort: Optional[str],
    retry_count: int,
) -> Tuple[List[str], float, str, str, str]:
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_PROMPT_TEMPLATE},
                    {"type": "input_image", "image_url": data_url, "detail": detail},
                ],
            },
        ],
        "max_output_tokens": 220,
        "text": {"format": {"type": "json_object"}},
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    payload_variants: List[dict] = [payload]
    if "text" in payload:
        v = dict(payload)
        v.pop("text", None)
        payload_variants.append(v)
    if "reasoning" in payload:
        v = dict(payload)
        v.pop("reasoning", None)
        payload_variants.append(v)
    if "text" in payload and "reasoning" in payload:
        v = dict(payload)
        v.pop("text", None)
        v.pop("reasoning", None)
        payload_variants.append(v)

    last_exc: Optional[Exception] = None
    for attempt in range(1, retry_count + 1):
        try:
            response = None
            for variant in payload_variants:
                try:
                    response = client.responses.create(**variant)
                    break
                except TypeError as exc:
                    last_exc = exc
                    continue
            if response is None:
                raise RuntimeError(f"No compatible payload variant succeeded: {last_exc}")

            response_text = extract_response_text(response)
            response_id = str(getattr(response, "id", ""))
            obj = extract_json_object(response_text) or {}
            labels = normalize_labels(obj.get("labels"))

            confidence_raw = obj.get("confidence", 0.0)
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            reason = str(obj.get("reason", "")).strip()
            return labels, confidence, reason, response_id, response_text
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            if attempt == retry_count:
                break
            sleep_sec = min(8.0, 2.0 ** (attempt - 1))
            print(f"[WARN] API error (attempt {attempt}/{retry_count}): {exc}; retry in {sleep_sec:.1f}s")
            time.sleep(sleep_sec)

    raise RuntimeError(f"OpenAI API call failed after {retry_count} attempts: {last_exc}")


def load_processed_keys(csv_path: Path) -> Set[Tuple[str, int]]:
    keys: Set[Tuple[str, int]] = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return keys

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                keys.add((row["video_path"], int(row["frame_index"])))
            except Exception:
                continue
    return keys


def append_row(csv_path: Path, row: Dict[str, str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_item_and_copy_to_labels(
    *,
    output_root: Path,
    source_path: Path,
    item_index: int,
    item_tag: str,
    jpeg_bytes: bytes,
    labels: Sequence[str],
) -> Tuple[Path, List[Path]]:
    source_hash = path_id(source_path)
    stem_clean = re.sub(r"[^0-9A-Za-z._-]+", "_", source_path.stem)
    if item_index >= 0:
        suffix = f"{item_tag}{item_index:06d}"
    else:
        suffix = item_tag
    file_name = f"{source_hash}_{stem_clean}_{suffix}.jpg"

    frame_src_dir = output_root / "_frames" / source_hash
    frame_src_dir.mkdir(parents=True, exist_ok=True)
    frame_src_path = frame_src_dir / file_name
    if not frame_src_path.exists():
        frame_src_path.write_bytes(jpeg_bytes)

    copied: List[Path] = []
    for label in labels:
        label_dir = output_root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        dst = label_dir / file_name
        if not dst.exists():
            shutil.copy2(frame_src_path, dst)
        copied.append(dst)

    return frame_src_path, copied


def main() -> None:
    args = parse_args()
    api_key = load_api_key()
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Set it in environment or Colab Secrets. Do not hardcode keys in notebooks."
        )

    input_dir = Path(args.input_dir).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    video_list_file = Path(args.video_list_file).expanduser().resolve() if args.video_list_file else None
    image_list_file = Path(args.image_list_file).expanduser().resolve() if args.image_list_file else None
    output_root = Path(args.output_root).expanduser().resolve()
    metadata_csv = Path(args.metadata_csv).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    videos = load_video_paths(input_dir=input_dir, video_list_file=video_list_file)
    images = load_image_paths(image_dir=image_dir, image_list_file=image_list_file)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    if args.max_images > 0:
        images = images[: args.max_images]

    if not videos and not images:
        raise SystemExit(
            f"No inputs found. video_input={input_dir} image_input={image_dir} "
            f"video_list_file={video_list_file} image_list_file={image_list_file}"
        )

    processed_keys = load_processed_keys(metadata_csv) if args.skip_existing else set()
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Please install openai package: pip install openai") from exc

    client = OpenAI(api_key=api_key)
    label_counts: Dict[str, int] = {label: 0 for label in TARGET_CLASSES + [UNKNOWN_LABEL]}
    total_items = 0
    skipped_items = 0

    print(f"videos: {len(videos)}")
    print(f"images: {len(images)}")
    print(f"output_root: {output_root}")
    print(f"metadata_csv: {metadata_csv}")
    print(f"model: {args.model}")

    def process_one_sample(
        *,
        source_path: Path,
        item_index: int,
        timestamp_sec: float,
        item_tag: str,
        jpeg_bytes: bytes,
    ) -> None:
        nonlocal total_items, skipped_items

        key = (str(source_path), int(item_index))
        if args.skip_existing and key in processed_keys:
            skipped_items += 1
            return

        try:
            labels, confidence, reason, response_id, response_text = classify_frame_with_openai(
                client,
                model=args.model,
                jpeg_bytes=jpeg_bytes,
                detail=args.detail,
                reasoning_effort=args.reasoning_effort,
                retry_count=args.retry_count,
            )
        except Exception as exc:  # pragma: no cover
            if not args.continue_on_api_error:
                raise RuntimeError(
                    "OpenAI API failed. Use --continue-on-api-error to keep running and map failures to unknown."
                ) from exc
            labels = [UNKNOWN_LABEL]
            confidence = 0.0
            reason = f"api_error:{exc}"
            response_id = ""
            response_text = ""

        frame_src_path, copied_paths = save_item_and_copy_to_labels(
            output_root=output_root,
            source_path=source_path,
            item_index=item_index,
            item_tag=item_tag,
            jpeg_bytes=jpeg_bytes,
            labels=labels,
        )

        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        row = {
            "video_path": str(source_path),
            "video_id": path_id(source_path),
            "frame_index": str(item_index),
            "timestamp_sec": f"{timestamp_sec:.3f}",
            "frame_source_path": str(frame_src_path),
            "assigned_labels_json": json.dumps(labels, ensure_ascii=False),
            "confidence": f"{confidence:.4f}",
            "reason": reason,
            "model": args.model,
            "response_id": response_id,
            "response_text": response_text,
            "copied_paths_json": json.dumps([str(p) for p in copied_paths], ensure_ascii=False),
        }
        append_row(metadata_csv, row)
        processed_keys.add(key)
        total_items += 1

        if total_items % 20 == 0:
            print(f"  processed items: {total_items} (skipped: {skipped_items})")

        if args.api_sleep_sec > 0:
            time.sleep(args.api_sleep_sec)

    for video_no, video_path in enumerate(videos, start=1):
        print(f"\n[{video_no}/{len(videos)}] {video_path}")
        try:
            frame_iter = iterate_sampled_frames(
                video_path=video_path,
                sample_every_n_frames=args.sample_every_n_frames,
                max_frames_per_video=args.max_frames_per_video,
            )
            for frame_index, timestamp_sec, frame in frame_iter:
                jpeg_bytes = encode_frame_to_jpeg(
                    frame=frame,
                    image_max_side=args.image_max_side,
                    jpeg_quality=args.jpeg_quality,
                )
                process_one_sample(
                    source_path=video_path,
                    item_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    item_tag="f",
                    jpeg_bytes=jpeg_bytes,
                )

        except Exception as exc:
            print(f"[WARN] failed to process video {video_path}: {exc}")
            continue

    if images:
        print("\n[Images] start")
    for image_no, image_path in enumerate(images, start=1):
        if image_no % 50 == 1 or image_no == len(images):
            print(f"[img {image_no}/{len(images)}] {image_path}")
        try:
            jpeg_bytes = encode_image_file_to_jpeg(
                image_path=image_path,
                image_max_side=args.image_max_side,
                jpeg_quality=args.jpeg_quality,
            )
            process_one_sample(
                source_path=image_path,
                item_index=-1,
                timestamp_sec=-1.0,
                item_tag="img",
                jpeg_bytes=jpeg_bytes,
            )
        except Exception as exc:
            print(f"[WARN] failed to process image {image_path}: {exc}")
            continue

    print("\nDone")
    print(f"processed items: {total_items}")
    print(f"skipped items: {skipped_items}")
    print("label counts:")
    for label in TARGET_CLASSES + [UNKNOWN_LABEL]:
        print(f"  {label}: {label_counts.get(label, 0)}")


if __name__ == "__main__":
    main()
