#!/usr/bin/env python3
"""
capture_training.py — Capture training images for defect detection.

Usage:
    python capture_training.py --category scratch --count 50
    python capture_training.py --category good --count 100

This script captures images one at a time, letting you swap parts between shots.
Press ENTER to capture, 'q' to quit early.
"""

import argparse
import os
import time
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Capture training images for defect detection")
    parser.add_argument("--category", required=True,
                        choices=["good", "scratch", "dent", "crack", "stain", "chip"],
                        help="Defect category for this batch of images")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of images to capture (default: 50)")
    parser.add_argument("--output-dir", default="dataset/train",
                        help="Base output directory (default: dataset/train)")
    parser.add_argument("--resolution", default="640x640",
                        help="Image resolution WxH (default: 640x640)")
    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split("x"))

    # Create output directory
    save_dir = os.path.join(args.output_dir, args.category)
    os.makedirs(save_dir, exist_ok=True)

    # Count existing images to avoid overwriting
    existing = len([f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png'))])

    print("=" * 60)
    print(f"  TRAINING IMAGE CAPTURE")
    print(f"  Category : {args.category}")
    print(f"  Target   : {args.count} images")
    print(f"  Save to  : {save_dir}")
    print(f"  Existing : {existing} images already in folder")
    print("=" * 60)
    print()

    # Try to import picamera2 (Raspberry Pi camera)
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        config = camera.create_still_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        time.sleep(2)  # Warm-up time for camera

        use_picamera = True
        print("Camera ready! (Raspberry Pi Camera detected)")
    except ImportError:
        use_picamera = False
        print("NOTE: picamera2 not found. Using OpenCV webcam instead.")
        import cv2
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not camera.isOpened():
            print("ERROR: Could not open camera!")
            return

    print()
    print("Instructions:")
    print("  1. Place the part in front of the camera")
    print("  2. Press ENTER to capture")
    print("  3. Swap the part and repeat")
    print("  4. Type 'q' and press ENTER to quit early")
    print()

    captured = 0
    for i in range(args.count):
        user_input = input(f"  [{captured + 1}/{args.count}] Press ENTER to capture (or 'q' to quit): ")
        if user_input.strip().lower() == 'q':
            break

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.category}_{existing + captured + 1:04d}_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)

        # Capture image
        if use_picamera:
            import numpy as np
            from PIL import Image
            array = camera.capture_array()
            img = Image.fromarray(array)
            img.save(filepath, quality=95)
        else:
            ret, frame = camera.read()
            if ret:
                cv2.imwrite(filepath, frame)
            else:
                print("    WARNING: Failed to capture frame, skipping...")
                continue

        captured += 1
        print(f"    Saved: {filename}")

    # Cleanup
    if use_picamera:
        camera.stop()
    else:
        camera.release()

    print()
    print(f"Done! Captured {captured} images in '{save_dir}'")
    print(f"Total images in folder: {existing + captured}")

    if existing + captured < 50:
        print(f"\nTIP: You have {existing + captured} images. Aim for at least 50 per category")
        print("     for basic accuracy, or 200+ for good results.")


if __name__ == "__main__":
    main()
