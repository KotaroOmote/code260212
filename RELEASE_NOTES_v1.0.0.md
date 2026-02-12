# code260212 v1.0.0

7種動物（アナグマ・アライグマ・ハクビシン・タヌキ・ネコ・ノウサギ・テン）の  
マルチラベル分類パイプライン初回安定版リリースです。

- 対応コミット: `de46305`
- リポジトリ: https://github.com/KotaroOmote/code260212
- データセット: https://huggingface.co/datasets/KotaroOmote/rg-7wildlife-multilabel-v1

## Included

- OpenAI API による動画/画像アノテーション
  - `src/annotate_videos_with_openai.py`
- 画像CSV + 動画CSV の統合とリーク防止split
  - `src/build_multilabel_dataset.py`
- 学習/評価（ResNet + EfficientNetV2-S）
  - `src/train_multilabel_classifier.py`
- 推論CLI（画像一括推論CSV出力）
  - `src/predict_multilabel_classifier.py`
- 再現手順README / MITライセンス
  - `README.md`
  - `LICENSE`

## Reproducibility

```bash
python -m pip install -r requirements.txt
```

1. アノテーション:

```bash
python src/annotate_videos_with_openai.py \
  --video-list-file metadata/video_list.txt \
  --image-dir data/raw \
  --output-root "./artifacts/ai_annotated_frames" \
  --metadata-csv "./artifacts/metadata/openai_video_annotations.csv" \
  --model "gpt-5.2"
```

2. 統合CSVとsplit作成:

```bash
python src/build_multilabel_dataset.py \
  --image-csv "./artifacts/metadata/openai_image_annotations.csv" \
  --video-csv "./artifacts/metadata/openai_video_annotations.csv" \
  --output-dir "./artifacts/metadata" \
  --seed 42
```

3. 学習/評価:

```bash
python src/train_multilabel_classifier.py \
  --train-csv "./artifacts/metadata/train_known.csv" \
  --val-csv "./artifacts/metadata/val_known.csv" \
  --test-csv "./artifacts/metadata/test_known.csv" \
  --model-out "./artifacts/models/multilabel_resnet101_best.pt" \
  --metrics-out "./artifacts/models/multilabel_resnet101_metrics.json" \
  --history-out "./artifacts/models/multilabel_resnet101_history.csv" \
  --arch "resnet101"
```

4. 推論:

```bash
python src/predict_multilabel_classifier.py \
  --model-path "./artifacts/models/multilabel_resnet101_best.pt" \
  --input-path "./data/infer_images" \
  --recursive \
  --output-csv "./artifacts/predictions/resnet101_predictions.csv"
```

## Key Results (fixed split)

- ResNet101: test micro F1 `0.6356`
- EfficientNetV2-S + tuned thresholds: test micro F1 `0.6852`
- Ensemble (EffNetV2-S + ResNet101 + ResNet50) tuned: test micro F1 `0.6856`

## Notes

- unknown クラスを明示的に分離
- 動画は `video_id` 単位で split して train/val/test 間リークを防止
- APIキー/トークンは含めない運用を徹底
