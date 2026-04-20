# Raspberry Pi Defect Detection & NCR System

## Complete Beginner's Guide

---

## What This Project Does

This system uses a Raspberry Pi with a camera to automatically inspect manufactured parts for surface defects. When a defect is detected, the system automatically opens a Non-Conformance Report (NCR) form with the defect type pre-selected, saving time and reducing human error.

### System Flow

```
Camera captures image
        ↓
AI Model analyzes for defects
        ↓
  Defect found? ──No──→ "PASS" (green light)
        ↓ Yes
NCR Form auto-populates with:
  - Defect type (scratch, dent, crack, etc.)
  - Confidence level
  - Captured image
  - Timestamp
        ↓
Operator reviews & submits NCR
```

---

## PHASE 1: Hardware Setup

### What You Need

| Item | Purpose | Approx. Cost |
|------|---------|-------------|
| Raspberry Pi 4 (4GB+ RAM) | Main computer | $55-75 |
| Raspberry Pi Camera Module v2 (or HQ Camera) | Image capture | $25-50 |
| MicroSD Card (32GB+) | Storage | $10 |
| USB-C Power Supply (5V/3A) | Power | $10 |
| Monitor, keyboard, mouse | Initial setup | (use existing) |
| LED light (optional) | Consistent lighting | $10-20 |
| Camera mount/stand (optional) | Stable positioning | $15-30 |

### Step 1: Install Raspberry Pi OS

1. Download **Raspberry Pi Imager** from https://www.raspberrypi.com/software/
2. Insert your MicroSD card into your computer
3. Open Raspberry Pi Imager:
   - Choose OS → **Raspberry Pi OS (64-bit)** (recommended)
   - Choose Storage → select your MicroSD card
   - Click the gear icon to set Wi-Fi, username/password, SSH
   - Click **Write**
4. Insert the MicroSD into your Raspberry Pi and power on

### Step 2: Connect the Camera

1. Power OFF the Raspberry Pi
2. Locate the CSI camera port (between the HDMI and audio jack)
3. Lift the plastic clip gently
4. Insert the ribbon cable with the blue side facing the USB ports
5. Press the clip back down
6. Power ON the Raspberry Pi

### Step 3: Enable the Camera

Open a terminal and run:

```bash
sudo raspi-config
```

Navigate to: **Interface Options → Camera → Enable → Finish → Reboot**

### Step 4: Test the Camera

```bash
# Take a test photo
libcamera-still -o test.jpg

# View it
xdg-open test.jpg
```

If you see your test photo, the camera is working!

---

## PHASE 2: Software Environment Setup

### Step 1: Update Your Pi

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2: Install Python Dependencies

```bash
# Install pip if not already installed
sudo apt install python3-pip python3-venv -y

# Create a project directory
mkdir ~/defect-inspector
cd ~/defect-inspector

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install tensorflow-lite-runtime  # or tflite-runtime
pip install opencv-python-headless
pip install numpy
pip install Pillow
pip install flask
pip install picamera2   # Raspberry Pi camera library
```

> **Note:** If `tflite-runtime` fails, try:
> ```bash
> pip install tflite-runtime --index-url https://google-coral.github.io/py-repo/
> ```
> Or install full TensorFlow (slower but works):
> ```bash
> pip install tensorflow
> ```

### Step 3: Create Project Structure

```bash
mkdir -p ~/defect-inspector/{models,dataset,dataset/train,dataset/test,static,templates,captured,ncr_reports}
mkdir -p ~/defect-inspector/dataset/train/{good,scratch,dent,crack,stain,chip}
mkdir -p ~/defect-inspector/dataset/test/{good,scratch,dent,crack,stain,chip}
```

Your folder structure should look like:

```
~/defect-inspector/
├── models/              ← Trained AI model goes here
├── dataset/
│   ├── train/           ← Training images
│   │   ├── good/        ← Images of good parts
│   │   ├── scratch/     ← Images with scratches
│   │   ├── dent/        ← Images with dents
│   │   ├── crack/       ← Images with cracks
│   │   ├── stain/       ← Images with stains
│   │   └── chip/        ← Images with chips
│   └── test/            ← Test images (same subfolders)
├── static/              ← Web app static files
├── templates/           ← Web app HTML templates
├── captured/            ← Auto-captured inspection images
├── ncr_reports/         ← Saved NCR reports
├── train_model.py       ← Model training script
├── detect.py            ← Main detection script
└── app.py               ← NCR web application
```

---

## PHASE 3: Collecting Your Defect Images

### How Many Images Do You Need?

For a beginner project that works reasonably well:

| Category | Minimum | Recommended |
|----------|---------|-------------|
| Per defect type | 50 images | 200+ images |
| Total training | 300 images | 1,200+ images |

### Tips for Collecting Good Training Data

1. **Consistent lighting** — Use the same lighting setup every time
2. **Same distance** — Keep the camera at a fixed distance from parts
3. **Same background** — Use a plain, consistent background
4. **Variety in defects** — Capture different sizes, positions, and severities
5. **Label accurately** — Put each image in the correct folder

### Capturing Training Images with the Pi Camera

Use the included `capture_training.py` script:

```bash
python capture_training.py --category scratch --count 50
```

This will take 50 photos, showing a preview between each one so you can swap parts.

---

## PHASE 4: Training Your AI Model

Run the training script on a **computer with more power** (laptop/desktop), or directly on the Pi (will be slower):

```bash
python train_model.py
```

This will:
1. Load all your labeled images
2. Train a lightweight MobileNetV2 model
3. Convert it to TensorFlow Lite format (optimized for Pi)
4. Save the model to `models/defect_model.tflite`

Training takes approximately:
- On a laptop with GPU: 5-15 minutes
- On Raspberry Pi 4: 30-90 minutes

---

## PHASE 5: Running the System

### Start the NCR Web App

```bash
python app.py
```

This starts:
- The camera detection system
- A web-based NCR form at `http://<your-pi-ip>:5000`

### Daily Usage

1. Open a browser on any device on the same network
2. Go to `http://<your-pi-ip>:5000`
3. Place a part in front of the camera
4. Click **"Inspect"** or wait for auto-inspection
5. If a defect is found, the NCR form auto-fills
6. Review, add notes, and submit

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not detected | Check ribbon cable connection, run `sudo raspi-config` to enable camera |
| Import errors | Make sure virtual environment is activated: `source venv/bin/activate` |
| Model accuracy is low | Add more training images, ensure consistent lighting |
| Web app not accessible | Check firewall: `sudo ufw allow 5000` |
| Out of memory | Use Raspberry Pi 4 with 4GB+ RAM, close other programs |

---

## Next Steps (Advanced)

- **Auto-trigger**: Add a proximity sensor to auto-capture when a part is placed
- **Email alerts**: Send NCR reports via email automatically
- **Database**: Store NCR records in SQLite or PostgreSQL
- **Dashboard**: Add charts showing defect trends over time
- **Multi-camera**: Run multiple inspection stations from one Pi
- **Edge TPU**: Add a Google Coral USB Accelerator for 10x faster inference
