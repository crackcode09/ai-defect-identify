#!/usr/bin/env python3
"""
prepare_dataset.py — Copy NEU-DET images into our dataset folder structure
and generate synthetic "good" (defect-free) images.

Run once:
    python prepare_dataset.py
"""

import os
import shutil
import random
import numpy as np
from PIL import Image, ImageFilter

# --- Mapping: NEU-DET folder name → our category name ---
CATEGORY_MAP = {
    "scratches":      "scratch",
    "crazing":        "stain",
    "inclusion":      "chip",
    "patches":        "dent",
    "pitted_surface": "crack",
    # rolled-in_scale skipped — no matching category
}

ARCHIVE_TRAIN = "archive/NEU-DET/train/images"
ARCHIVE_VAL   = "archive/NEU-DET/validation/images"
TRAIN_OUT     = "dataset/train"
TEST_OUT      = "dataset/test"

GOOD_TRAIN_COUNT = 240
GOOD_TEST_COUNT  = 60


def copy_images(src_dir, dst_dir):
    """Copy all images from src_dir into dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    for f in files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    return len(files)


def generate_good_images(out_dir, count, image_size=200):
    """
    Generate synthetic 'good' images — plain steel-like surfaces with
    subtle noise and gradients. No defects. Teaches the AI what PASS looks like.
    """
    os.makedirs(out_dir, exist_ok=True)
    generated = 0

    for i in range(count):
        # Random base gray tone (steel-like: mid-gray range)
        base = random.randint(100, 180)

        # Create base gray image
        arr = np.full((image_size, image_size), base, dtype=np.uint8)

        # Add subtle random noise (simulates surface texture)
        noise = np.random.randint(-15, 15, (image_size, image_size), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Random gradient (simulates uneven lighting on metal)
        gradient_strength = random.randint(5, 20)
        direction = random.choice(['h', 'v', 'diag'])
        if direction == 'h':
            grad = np.linspace(-gradient_strength, gradient_strength, image_size, dtype=np.int16)
            grad = np.tile(grad, (image_size, 1))
        elif direction == 'v':
            grad = np.linspace(-gradient_strength, gradient_strength, image_size, dtype=np.int16)
            grad = np.tile(grad.reshape(-1, 1), (1, image_size))
        else:
            gx = np.linspace(-gradient_strength, gradient_strength, image_size, dtype=np.int16)
            gy = np.linspace(-gradient_strength, gradient_strength, image_size, dtype=np.int16)
            grad = np.add.outer(gy, gx)

        arr = np.clip(arr.astype(np.int16) + grad, 0, 255).astype(np.uint8)

        # Convert to PIL and apply very slight blur (smooths out harsh noise)
        img = Image.fromarray(arr, mode='L').convert('RGB')
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        filename = f"good_{i+1:04d}.jpg"
        img.save(os.path.join(out_dir, filename), quality=90)
        generated += 1

    return generated


def main():
    print("=" * 55)
    print("  DATASET PREPARATION")
    print("=" * 55)

    # --- Copy defect images ---
    print("\nCopying defect images from archive...\n")

    total_train = 0
    total_test = 0

    for neu_name, our_name in CATEGORY_MAP.items():
        # Train
        src = os.path.join(ARCHIVE_TRAIN, neu_name)
        dst = os.path.join(TRAIN_OUT, our_name)
        if os.path.exists(src):
            n = copy_images(src, dst)
            print(f"  train/{our_name:10s} ← {neu_name:20s} ({n} images)")
            total_train += n
        else:
            print(f"  SKIP: {src} not found")

        # Test / validation
        src = os.path.join(ARCHIVE_VAL, neu_name)
        dst = os.path.join(TEST_OUT, our_name)
        if os.path.exists(src):
            n = copy_images(src, dst)
            print(f"  test/{our_name:11s} ← {neu_name:20s} ({n} images)")
            total_test += n

    # --- Generate "good" images ---
    print(f"\nGenerating {GOOD_TRAIN_COUNT} 'good' images for training...")
    n = generate_good_images(os.path.join(TRAIN_OUT, "good"), GOOD_TRAIN_COUNT)
    print(f"  train/good         → {n} images generated")

    print(f"\nGenerating {GOOD_TEST_COUNT} 'good' images for testing...")
    n = generate_good_images(os.path.join(TEST_OUT, "good"), GOOD_TEST_COUNT)
    print(f"  test/good          → {n} images generated")

    # --- Summary ---
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"\n  Training set:")
    for cat in os.listdir(TRAIN_OUT):
        path = os.path.join(TRAIN_OUT, cat)
        if os.path.isdir(path):
            count = len(os.listdir(path))
            print(f"    {cat:12s}: {count} images")

    print(f"\n  Test set:")
    for cat in os.listdir(TEST_OUT):
        path = os.path.join(TEST_OUT, cat)
        if os.path.isdir(path):
            count = len(os.listdir(path))
            print(f"    {cat:12s}: {count} images")

    print("\nDone! Next step: python train_model.py")


if __name__ == "__main__":
    main()
