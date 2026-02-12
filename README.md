# code260212

7種動物（アナグマ・アライグマ・ハクビシン・タヌキ・ネコ・ノウサギ・テン）を対象に、  
**データ作成 → マルチラベル分類学習 → 評価** を再現可能に実行するリポジトリです。

## 対象クラス

- アナグマ
- アライグマ
- ハクビシン
- タヌキ
- ネコ
- ノウサギ
- テン

## 方針

- 1画像/1フレームに複数種がいる場合は `multi-label`
- 対象外/判別不能は `unknown`
- 動画データは `video_id` 単位で分割し、train/val/test 間リークを防止

## リポジトリ構成

- `src/annotate_videos_with_openai.py`  
  動画+画像を OpenAI API でアノテーションし、クラス別フォルダに振り分け
- `src/build_multilabel_dataset.py`  
  画像CSV + 動画CSV を統合し、最終CSVと split CSV を作成
- `src/train_multilabel_classifier.py`  
  PyTorch（ResNet18）で multi-label 学習・評価を実行
- `metadata/rg_video_list.txt`  
  指定動画リスト

## セットアップ

```bash
pip install -r requirements.txt
```

Colab の場合:

```bash
!pip -q install -r "/content/drive/MyDrive/code260212/requirements.txt"
```

`OPENAI_API_KEY` は環境変数または Colab Secrets に登録してください。  
キー文字列をノートブックやコードに直書きしないでください。

## 1. アノテーション（動画+画像）

### 動画リストを指定して実行

```bash
python src/annotate_videos_with_openai.py \
  --video-list-file metadata/rg_video_list.txt \
  --image-dir data/raw \
  --output-root "/content/drive/MyDrive/RG/ai_annotated_frames" \
  --metadata-csv "/content/drive/MyDrive/RG/metadata/openai_video_annotations.csv" \
  --model "gpt-5.2" \
  --sample-every-n-frames 45 \
  --max-frames-per-video 180
```

### 画像のみ実行（必要に応じて）

```bash
python src/annotate_videos_with_openai.py \
  --input-dir "/content/drive/MyDrive/RG/no_videos" \
  --image-dir "/content/drive/MyDrive/code260212/data/raw" \
  --output-root "/content/drive/MyDrive/RG/ai_annotated_frames" \
  --metadata-csv "/content/drive/MyDrive/RG/metadata/openai_image_annotations.csv" \
  --model "gpt-5.2"
```

## 2. 統合CSV作成と分割

```bash
python src/build_multilabel_dataset.py \
  --image-csv "/content/drive/MyDrive/RG/metadata/openai_image_annotations.csv" \
  --video-csv "/content/drive/MyDrive/RG/metadata/openai_video_annotations.csv" \
  --output-dir "/content/drive/MyDrive/RG/metadata" \
  --seed 42
```

出力:

- `final_multilabel_dataset.csv`
- `final_multilabel_dataset_known_only.csv`
- `unknown_samples.csv`
- `train_known.csv`
- `val_known.csv`
- `test_known.csv`

## 3. 学習と評価（PyTorch, multi-label）

```bash
python src/train_multilabel_classifier.py \
  --train-csv "/content/drive/MyDrive/RG/metadata/train_known.csv" \
  --val-csv "/content/drive/MyDrive/RG/metadata/val_known.csv" \
  --test-csv "/content/drive/MyDrive/RG/metadata/test_known.csv" \
  --model-out "/content/drive/MyDrive/RG/models/multilabel_resnet18_best.pt" \
  --metrics-out "/content/drive/MyDrive/RG/models/multilabel_test_metrics.json" \
  --history-out "/content/drive/MyDrive/RG/models/multilabel_train_history.csv" \
  --arch "resnet18" \
  --epochs 12 \
  --batch-size 32 \
  --threshold 0.5
```

評価結果は JSON として保存され、再利用できます。

### バックボーン比較（ResNet）

`--arch` で切り替え可能:

- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `resnet152`

例:

```bash
python src/train_multilabel_classifier.py \
  --train-csv "/content/drive/MyDrive/RG/metadata/train_known.csv" \
  --val-csv "/content/drive/MyDrive/RG/metadata/val_known.csv" \
  --test-csv "/content/drive/MyDrive/RG/metadata/test_known.csv" \
  --model-out "/content/drive/MyDrive/RG/models/multilabel_resnet50_best.pt" \
  --metrics-out "/content/drive/MyDrive/RG/models/multilabel_resnet50_metrics.json" \
  --history-out "/content/drive/MyDrive/RG/models/multilabel_resnet50_history.csv" \
  --arch "resnet50" \
  --epochs 12 \
  --batch-size 32 \
  --threshold 0.5
```

## 直近の実行結果（2026-02-12）

- split: train 1227 / val 263 / test 263
- video leakage: train∩val=0, train∩test=0, val∩test=0
- test: micro F1=0.5975, macro F1=0.5584

## Git運用メモ

- GitHubに上げる: コード、設定、CSVメタデータ（必要な範囲）
- GitHubに上げない: 生動画、生画像、APIキー、巨大中間ファイル
