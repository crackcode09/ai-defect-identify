# Defect Inspector — Training Log & Learning Notes

What we did, why we did it, in plain English.

---

## What This Project Is

A Raspberry Pi with a camera inspects manufactured parts for defects.
- Camera takes photo → AI analyzes → Defect found? → NCR form auto-fills → Human submits report.

---

## Step 1 — Dataset Collection

**What is a dataset?**
AI learns by example. You show it thousands of labeled photos ("this is a scratch", "this is good") and it learns the patterns. That collection of photos = dataset.

**What we used:** NEU Surface Defect Database (free, from Kaggle)
- Real steel surface photos used in industrial research
- 6 defect types, ~300 images each

**Category mapping (NEU → our system):**

| NEU-DET name    | Our name | What it looks like         |
|-----------------|----------|----------------------------|
| scratches       | scratch  | Linear surface marks       |
| crazing         | stain    | Network of fine cracks     |
| inclusion       | chip     | Embedded foreign material  |
| patches         | dent     | Irregular surface patches  |
| pitted_surface  | crack    | Small pits/holes           |
| rolled-in_scale | (skipped)| No matching category       |

**The "good" class problem:**
The NEU dataset has no defect-free images. Without "good" examples, the AI never learns what a passing part looks like. We generated 240 synthetic "good" images — plain gray metal-like surfaces with subtle noise and gradients.

**Final dataset:**

| Split    | Per category | Categories | Total  |
|----------|-------------|------------|--------|
| Training | 240 images  | 6          | 1,440  |
| Test     | 60 images   | 6          | 360    |

Script: `prepare_dataset.py`

---

## Step 2 — Training the Model (in progress)

**What is training?**
The AI engine (TensorFlow) looks at each training image thousands of times, adjusting internal numbers (called "weights") until it can reliably tell categories apart.

**What model are we using?**
MobileNetV2 — a lightweight neural network designed for small devices like phones and Raspberry Pi. We use "transfer learning": the model already knows how to see edges, textures, and shapes from being trained on millions of internet photos. We just teach it the final step — "here's what our specific defects look like."

**Why MobileNetV2 and not something bigger?**
Bigger models = more accurate but too slow for a Pi. MobileNetV2 runs in ~100ms per image on Pi 4. Good enough for real-time inspection.

**Output of training:**
- `models/defect_model.tflite` — the trained AI, compressed for Pi
- `models/labels.txt` — category names in order
- `models/model_config.json` — image size and class info

Script: `train_model.py`

---

## Environment Notes

- Development machine: Windows 11
- Python version needed: **3.12** (3.15 too new — no numpy/TensorFlow wheels)
- Install command: `py -3.12 -m pip install tensorflow numpy Pillow flask`

---

## Files In This Project

| File                  | Purpose                                      |
|-----------------------|----------------------------------------------|
| `detect.py`           | Loads TFLite model, runs inference on images |
| `app.py`              | Flask web server — 7 API routes              |
| `train_model.py`      | MobileNetV2 transfer learning → TFLite       |
| `capture_training.py` | Pi camera training data collection           |
| `prepare_dataset.py`  | Copies NEU dataset + generates "good" images |
| `templates/index.html`| Full web UI — inspect, NCR form, history     |
| `requirements.txt`    | Python dependencies                          |

---

## Roadmap

- [x] Write backend code (detect, app, train, capture)
- [x] Write frontend (index.html)
- [x] Download and prepare dataset
- [ ] Train model on Windows
- [ ] Test web app in upload mode (no camera)
- [ ] Deploy to Raspberry Pi
- [ ] Test with real camera
