# 🤟 ASL Bridge — Bidirectional ASL ↔ Audio Translation

**Fully local. No cloud APIs. No billing. Runs on-device.**

ASL Bridge is a real-time, bidirectional American Sign Language translation system that operates entirely on your local machine using computer vision and deep learning.

---

## 🎯 Features

### Mode A: Signs → Voice
- 📷 Real-time webcam hand tracking via MediaPipe Holistic
- 🧠 GRU + Attention classifier for ASL sign recognition
- 🔊 Offline text-to-speech output via pyttsx3
- ⚡ Target latency: <50ms per frame

### Mode B: Voice → Signs
- 🎤 Speech-to-text via SpeechRecognition (offline capable)
- 🤟 Animated ASL fingerspelling display
- 📝 Real-time transcript

---

## 📁 Project Structure

```
asl-bridge/
├── config.yaml              # Master configuration (all hyperparameters)
├── requirements.txt          # Pinned Python dependencies
├── README.md                 # This file
│
├── data/
│   ├── raw/                  # Downloaded datasets
│   │   ├── asl_alphabet/     # ASL Alphabet (Kaggle) — 87K images
│   │   ├── asl_mnist/        # ASL MNIST (HuggingFace) — 28x28
│   │   ├── wlasl/            # WLASL-2000 word-level videos
│   │   ├── how2sign/         # How2Sign continuous ASL
│   │   └── ms_asl/           # MS-ASL 25K videos
│   ├── processed/            # Extracted keypoint .npy files
│   └── augmented/            # Augmented training samples
│
├── models/
│   ├── classifier.py         # GRU + Attention classifier (PyTorch)
│   ├── train.py              # Training loop with early stopping
│   ├── evaluate.py           # Confusion matrix, per-class accuracy
│   └── checkpoints/          # Saved .pth model files
│
├── pipeline/
│   ├── extractor.py          # MediaPipe landmark extraction
│   ├── preprocessor.py       # Normalize, pad, augment keypoints
│   ├── inference.py          # Real-time inference engine
│   ├── tts.py                # pyttsx3 text-to-speech wrapper
│   └── stt.py                # SpeechRecognition wrapper
│
├── server/
│   └── app.py                # FastAPI REST + SSE server
│
├── frontend/
│   └── index.html            # Web UI (camera + signs + transcript)
│
├── scripts/
│   ├── download_data.py      # Dataset downloader & verifier
│   ├── extract_keypoints.py  # Batch MediaPipe extraction
│   └── build_vocab.py        # Build gloss vocabulary
│
└── tests/
    └── test_pipeline.py      # Unit tests
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd asl-bridge
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Check Datasets
```bash
python scripts/download_data.py
```

### 3. Copy Your Local Data
If you have the ASL Alphabet dataset on your Desktop:
```bash
python scripts/download_data.py --copy-local
```

### 4. Build Vocabulary
```bash
python scripts/build_vocab.py --dataset asl_alphabet
```

### 5. Extract Keypoints
```bash
python scripts/extract_keypoints.py --dataset asl_alphabet
```

### 6. Train the Model
```bash
python models/train.py --dataset asl_alphabet
```

### 7. Evaluate
```bash
python models/evaluate.py --checkpoint models/checkpoints/best_model.pth --dataset asl_alphabet --plot
```

### 8. Start the Server
```bash
python server/app.py
```
Then open http://127.0.0.1:8000/ui in your browser.

---

## 📜 Script Reference

### `scripts/download_data.py`
**Purpose:** Check dataset availability and auto-download where possible.
- **Inputs:** `--dataset` (optional), `--copy-local`, `--download`
- **Outputs:** Status report for each dataset

### `scripts/extract_keypoints.py`
**Purpose:** Batch extract MediaPipe hand landmarks from images/videos.
- **Inputs:** `--dataset`, `--split`, `--input` (single file)
- **Outputs:** `.npy` files in `data/processed/`

### `scripts/build_vocab.py`
**Purpose:** Build label ↔ ID vocabulary mappings.
- **Inputs:** `--dataset` or `--all`
- **Outputs:** `vocab.json` in `data/processed/{dataset}/`

### `pipeline/extractor.py`
**Purpose:** MediaPipe Holistic landmark extraction engine.
- **Inputs:** `--webcam`, `--video`, `--image`
- **Outputs:** Keypoint arrays of shape (21 landmarks × 3 coords)

### `pipeline/inference.py`
**Purpose:** Real-time inference with sliding window + stabilizer.
- **Inputs:** `--webcam`, `--checkpoint`
- **Outputs:** Stabilized sign predictions

### `pipeline/tts.py`
**Purpose:** Offline text-to-speech.
- **Inputs:** `--text`, `--interactive`
- **Outputs:** Audio through speakers

### `pipeline/stt.py`
**Purpose:** Speech-to-text recognition.
- **Inputs:** `--interactive`, `--audio` (file)
- **Outputs:** Transcribed text

---

## 🧠 Model Architecture

```
Input: (batch, 30, 63) — 30 frames × 21 hand landmarks × 3 coords

Layer 1: Linear(63, 128) + LayerNorm + ReLU
Layer 2: GRU(128, 256, layers=2, dropout=0.3)
Layer 3: Attention Pooling over GRU outputs
Layer 4: Linear(256, 128) + ReLU + Dropout(0.4)
Output:  Linear(128, num_classes) + Softmax
```

**Training:** Adam lr=1e-3, CosineAnnealingLR, CrossEntropyLoss, batch=32, epochs=50

---

## 📊 Datasets

| Priority | Dataset | Classes | Type | Source |
|----------|---------|---------|------|--------|
| 1 | ASL Alphabet | 29 | Static images | Kaggle |
| 2 | ASL MNIST | 24 | 28×28 grayscale | HuggingFace |
| 3 | WLASL-2000 | 2000 | Video clips | Kaggle |
| 4 | How2Sign | — | Continuous video | how2sign.github.io |
| 5 | MS-ASL | 1000 | Video clips | Microsoft Research |

---

## ⚙️ Configuration

All settings are in `config.yaml`. Key sections:
- `datasets` — paths and metadata for each dataset
- `model` — architecture hyperparameters
- `training` — optimizer, scheduler, early stopping
- `augmentation` — temporal jitter, flip, noise
- `inference` — sliding window, stabilizer, confidence threshold
- `server` — host, port, CORS origins

---

## 🧪 Testing

```bash
pytest tests/test_pipeline.py -v
```

---

## 📄 License

This project is for educational and research purposes.
