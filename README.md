# AI Defect Inspector — NCR System

A Raspberry Pi-powered system that uses a camera and AI to inspect manufactured parts for surface defects. When a defect is detected, a Non-Conformance Report (NCR) form auto-populates — saving time and reducing human error.

---

## How It Works

```
Camera captures image
        ↓
AI model analyzes for defects (98.6% accuracy)
        ↓
  Defect found? ──No──→ PASS (green)
        ↓ Yes
NCR form auto-fills with:
  - Defect type (scratch, dent, crack, stain, chip)
  - Confidence score
  - Inspection ID + timestamp
        ↓
Operator reviews and submits NCR
```

---

## Features

- Real-time defect detection using MobileNetV2 (TensorFlow Lite)
- 6 defect categories: scratch, crack, dent, stain, chip, good
- Web-based UI — open on any device on the same network
- NCR form auto-fills from AI detection results
- Upload mode for testing without a camera
- NCR history with severity tracking
- 82ms inference time on Raspberry Pi 4

---

## Project Structure

```
├── app.py                  # Flask web server
├── detect.py               # TFLite inference engine + camera
├── train_model.py          # MobileNetV2 training → TFLite
├── capture_training.py     # Pi camera training data collection
├── prepare_dataset.py      # Dataset preparation script
├── templates/
│   └── index.html          # Web UI
├── requirements.txt
├── PROJECT_GUIDE.md        # Full beginner setup guide
└── TRAINING_LOG.md         # Step-by-step learning notes
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/crackcode09/ai-defect-identify.git
cd ai-defect-identify
pip install -r requirements.txt
```

### 2. Prepare dataset

Download the [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) from Kaggle, extract to `archive/`, then:

```bash
python prepare_dataset.py
```

### 3. Train the model

```bash
python train_model.py
```

Training takes ~10 min on a laptop GPU, ~60 min on Raspberry Pi 4.

### 4. Run the app

```bash
# With camera (Raspberry Pi)
python app.py

# Without camera (upload mode, for testing on PC)
python app.py --no-camera
```

Open `http://localhost:5000` in your browser.

---

## Hardware (Raspberry Pi deployment)

| Item | Purpose |
|------|---------|
| Raspberry Pi 4 (4GB+) | Main computer |
| Pi Camera Module v2 / HQ | Image capture |
| MicroSD Card (32GB+) | Storage |
| USB-C Power Supply (5V/3A) | Power |

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for full hardware setup and Pi deployment steps.

---

## Tech Stack

- **AI Model:** MobileNetV2 (transfer learning) → TensorFlow Lite
- **Backend:** Python + Flask
- **Frontend:** Vanilla HTML/CSS/JS
- **Camera:** picamera2 (Pi) / OpenCV (webcam fallback)
- **Training dataset:** NEU Surface Defect Database

---

## Model Performance

| Metric | Value |
|--------|-------|
| Validation accuracy | 98.6% |
| Inference time (Pi 4) | ~83ms |
| Training images | 1,440 |
| Test images | 360 |
| Categories | 6 |
