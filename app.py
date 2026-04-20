#!/usr/bin/env python3
"""
app.py — Flask web application for defect inspection and NCR form.

This is the main application that ties everything together:
- Camera capture and live preview
- AI-powered defect detection
- Auto-populated Non-Conformance Report (NCR) form
- Report saving and history

Usage:
    python app.py
    python app.py --port 8080 --debug

Then open http://<raspberry-pi-ip>:5000 in a browser.
"""

import argparse
import json
import os
import time
import uuid
from datetime import datetime

from flask import (Flask, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)

from detect import CameraCapture, DefectDetector

# --- Flask App ---
app = Flask(__name__)
app.secret_key = "defect-inspector-secret-key"  # Change in production!

# Global objects (initialized in main)
detector = None
camera = None

# Directories
CAPTURED_DIR = "captured"
NCR_DIR = "ncr_reports"
STATIC_DIR = "static"

os.makedirs(CAPTURED_DIR, exist_ok=True)
os.makedirs(NCR_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# ===================== ROUTES =====================

@app.route("/")
def index():
    """Main inspection dashboard."""
    return render_template("index.html")


@app.route("/inspect", methods=["POST"])
def inspect():
    """
    Capture image, run detection, and return results.
    Called when user clicks "Inspect" button.
    """
    global camera, detector

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_id = str(uuid.uuid4())[:8]
    filename = f"inspection_{timestamp}_{image_id}.jpg"
    filepath = os.path.join(CAPTURED_DIR, filename)

    # Capture image
    if camera is not None:
        frame = camera.capture(save_path=filepath)
        if frame is None:
            return jsonify({"error": "Failed to capture image"}), 500
    else:
        return jsonify({"error": "No camera available"}), 500

    # Run detection
    result = detector.predict(filepath)

    # Add metadata
    result["image_filename"] = filename
    result["image_path"] = f"/captured/{filename}"
    result["timestamp"] = datetime.now().isoformat()
    result["inspection_id"] = f"INS-{timestamp}-{image_id}"

    return jsonify(result)


@app.route("/inspect/upload", methods=["POST"])
def inspect_upload():
    """
    Accept an uploaded image and run detection on it.
    Useful for testing without a camera.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_id = str(uuid.uuid4())[:8]
    filename = f"upload_{timestamp}_{image_id}.jpg"
    filepath = os.path.join(CAPTURED_DIR, filename)
    file.save(filepath)

    # Run detection
    result = detector.predict(filepath)
    result["image_filename"] = filename
    result["image_path"] = f"/captured/{filename}"
    result["timestamp"] = datetime.now().isoformat()
    result["inspection_id"] = f"INS-{timestamp}-{image_id}"

    return jsonify(result)


@app.route("/ncr/submit", methods=["POST"])
def submit_ncr():
    """Save an NCR report."""
    data = request.get_json()

    # Generate NCR number
    ncr_count = len(os.listdir(NCR_DIR)) + 1
    ncr_number = f"NCR-{datetime.now().strftime('%Y%m%d')}-{ncr_count:04d}"

    # Build NCR report
    ncr_report = {
        "ncr_number": ncr_number,
        "created_at": datetime.now().isoformat(),
        "status": "Open",
        # Inspection data (auto-filled)
        "inspection_id": data.get("inspection_id", ""),
        "defect_type": data.get("defect_type", ""),
        "confidence": data.get("confidence", 0),
        "image_path": data.get("image_path", ""),
        # Form data (user-filled)
        "part_number": data.get("part_number", ""),
        "part_name": data.get("part_name", ""),
        "batch_lot": data.get("batch_lot", ""),
        "quantity_inspected": data.get("quantity_inspected", 0),
        "quantity_rejected": data.get("quantity_rejected", 0),
        "severity": data.get("severity", "Minor"),
        "defect_location": data.get("defect_location", ""),
        "description": data.get("description", ""),
        "immediate_action": data.get("immediate_action", ""),
        "inspector_name": data.get("inspector_name", ""),
        "supervisor_name": data.get("supervisor_name", ""),
    }

    # Save to file
    ncr_path = os.path.join(NCR_DIR, f"{ncr_number}.json")
    with open(ncr_path, 'w') as f:
        json.dump(ncr_report, f, indent=2)

    return jsonify({
        "success": True,
        "ncr_number": ncr_number,
        "message": f"NCR {ncr_number} created successfully"
    })


@app.route("/ncr/history")
def ncr_history():
    """List all NCR reports."""
    reports = []
    for filename in sorted(os.listdir(NCR_DIR), reverse=True):
        if filename.endswith(".json"):
            with open(os.path.join(NCR_DIR, filename), 'r') as f:
                reports.append(json.load(f))
    return jsonify(reports)


@app.route("/ncr/<ncr_number>")
def ncr_detail(ncr_number):
    """View a specific NCR report."""
    ncr_path = os.path.join(NCR_DIR, f"{ncr_number}.json")
    if os.path.exists(ncr_path):
        with open(ncr_path, 'r') as f:
            report = json.load(f)
        return jsonify(report)
    return jsonify({"error": "NCR not found"}), 404


@app.route("/captured/<filename>")
def serve_captured(filename):
    """Serve captured inspection images."""
    return send_from_directory(CAPTURED_DIR, filename)


@app.route("/status")
def system_status():
    """System status check."""
    return jsonify({
        "model_loaded": detector.model_loaded if detector else False,
        "camera_available": camera is not None and camera.camera is not None,
        "defect_classes": detector.class_names if detector else [],
        "ncr_count": len([f for f in os.listdir(NCR_DIR) if f.endswith('.json')]),
    })


# ===================== MAIN =====================

def main():
    global detector, camera

    parser = argparse.ArgumentParser(description="Defect Inspector Web App")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0 = all interfaces)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-camera", action="store_true",
                        help="Run without camera (upload-only mode)")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    print("=" * 60)
    print("  DEFECT INSPECTOR — NCR SYSTEM")
    print("=" * 60)

    # Initialize detector
    detector = DefectDetector(model_dir=args.model_dir)

    # Initialize camera
    if not args.no_camera:
        camera = CameraCapture()
    else:
        camera = None
        print("Running in upload-only mode (no camera)")

    print()
    print(f"Starting web server on http://{args.host}:{args.port}")
    print(f"Open this URL in your browser to use the system.")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
