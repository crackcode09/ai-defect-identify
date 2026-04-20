#!/usr/bin/env python3
"""
train_model.py — Train a defect detection model using MobileNetV2.

This script:
1. Loads labeled images from dataset/train/ and dataset/test/
2. Fine-tunes a MobileNetV2 model (pre-trained on ImageNet)
3. Converts the model to TensorFlow Lite format for Raspberry Pi
4. Saves the model and a label file

Usage:
    python train_model.py
    python train_model.py --epochs 20 --batch-size 16

The trained model will be saved to:
    models/defect_model.tflite
    models/labels.txt
"""

import argparse
import json
import os
import sys

def check_dependencies():
    """Check that required packages are installed."""
    missing = []
    try:
        import tensorflow
    except ImportError:
        missing.append("tensorflow")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        print("\nInstall them and try again.")
        sys.exit(1)


def count_images(directory):
    """Count images in each subdirectory."""
    counts = {}
    if not os.path.exists(directory):
        return counts
    for category in sorted(os.listdir(directory)):
        cat_path = os.path.join(directory, category)
        if os.path.isdir(cat_path):
            n = len([f for f in os.listdir(cat_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            counts[category] = n
    return counts


def main():
    parser = argparse.ArgumentParser(description="Train defect detection model")
    parser.add_argument("--train-dir", default="dataset/train",
                        help="Training data directory (default: dataset/train)")
    parser.add_argument("--test-dir", default="dataset/test",
                        help="Test data directory (default: dataset/test)")
    parser.add_argument("--model-dir", default="models",
                        help="Output model directory (default: models)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16, use 8 on Raspberry Pi)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size (default: 224)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    check_dependencies()

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import numpy as np

    print("=" * 60)
    print("  DEFECT DETECTION MODEL TRAINER")
    print("=" * 60)
    print(f"  TensorFlow version: {tf.__version__}")
    print(f"  Training directory : {args.train_dir}")
    print(f"  Test directory     : {args.test_dir}")
    print(f"  Image size         : {args.image_size}x{args.image_size}")
    print(f"  Epochs             : {args.epochs}")
    print(f"  Batch size         : {args.batch_size}")
    print()

    # --- Check Dataset ---
    print("Checking dataset...")
    train_counts = count_images(args.train_dir)
    test_counts = count_images(args.test_dir)

    if not train_counts:
        print(f"\nERROR: No image folders found in '{args.train_dir}'")
        print("Please add images to category folders like:")
        print(f"  {args.train_dir}/good/")
        print(f"  {args.train_dir}/scratch/")
        print(f"  {args.train_dir}/dent/")
        print("\nUse capture_training.py to collect images.")
        sys.exit(1)

    print("\nTraining data:")
    total_train = 0
    for cat, count in train_counts.items():
        status = "OK" if count >= 20 else "LOW"
        print(f"  {cat:12s}: {count:4d} images  [{status}]")
        total_train += count

    if test_counts:
        print("\nTest data:")
        total_test = 0
        for cat, count in test_counts.items():
            print(f"  {cat:12s}: {count:4d} images")
            total_test += count
    else:
        print("\nNote: No test data found. Will split training data 80/20.")

    if total_train < 30:
        print(f"\nWARNING: Only {total_train} total training images.")
        print("For decent results, aim for 50+ images per category.")
        proceed = input("Continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(0)

    # --- Data Generators ---
    print("\nPreparing data...")

    # Data augmentation for training (helps model generalize with small datasets)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,               # Normalize pixel values to 0-1
        rotation_range=20,                # Random rotation up to 20 degrees
        width_shift_range=0.1,            # Random horizontal shift
        height_shift_range=0.1,           # Random vertical shift
        shear_range=0.1,                  # Random shearing
        zoom_range=0.1,                   # Random zoom
        horizontal_flip=True,             # Random horizontal flip
        brightness_range=[0.8, 1.2],      # Random brightness
        fill_mode='nearest',
        validation_split=0.2 if not test_counts else 0.0  # Split if no test data
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='training' if not test_counts else None,
        shuffle=True
    )

    # Load validation data
    if test_counts:
        val_generator = test_datagen.flow_from_directory(
            args.test_dir,
            target_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            class_mode='categorical',
            shuffle=False
        )
    else:
        val_generator = train_datagen.flow_from_directory(
            args.train_dir,
            target_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

    num_classes = train_generator.num_classes
    class_names = list(train_generator.class_indices.keys())

    print(f"\nDetected {num_classes} defect categories: {class_names}")
    print(f"Training samples  : {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")

    # --- Build Model ---
    print("\nBuilding model (MobileNetV2 + custom head)...")

    # Load MobileNetV2 pre-trained on ImageNet (without top classification layer)
    base_model = MobileNetV2(
        input_shape=(args.image_size, args.image_size, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model layers (use pre-trained features)
    base_model.trainable = False

    # Add custom classification layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)       # Convert feature maps to single vector
    x = Dense(128, activation='relu')(x)   # Hidden layer
    x = Dropout(0.3)(x)                    # Prevent overfitting
    x = Dense(64, activation='relu')(x)    # Another hidden layer
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # Output layer

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Model parameters: {model.count_params():,}")
    print(f"  Trainable      : {sum(p.numpy().size for p in model.trainable_weights):,}")
    print(f"  Non-trainable  : {sum(p.numpy().size for p in model.non_trainable_weights):,}")

    # --- Train ---
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 40)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    # --- Evaluate ---
    print("\n" + "=" * 40)
    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    print(f"Final Validation Accuracy: {val_acc * 100:.1f}%")
    print(f"Final Validation Loss    : {val_loss:.4f}")

    if val_acc < 0.6:
        print("\nWARNING: Accuracy is below 60%. Consider:")
        print("  - Adding more training images (200+ per category)")
        print("  - Improving lighting consistency")
        print("  - Increasing epochs (--epochs 30)")
    elif val_acc < 0.8:
        print("\nAccuracy is decent but could improve with more data.")
    else:
        print("\nGood accuracy! Model is ready for use.")

    # --- Fine-tune (optional second pass) ---
    print("\nFine-tuning: Unfreezing top layers of base model...")
    base_model.trainable = True
    # Only fine-tune the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate * 0.1),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    print(f"\nAfter fine-tuning — Accuracy: {val_acc * 100:.1f}%")

    # --- Save Model ---
    os.makedirs(args.model_dir, exist_ok=True)

    # Save as TensorFlow Lite model (optimized for Raspberry Pi)
    print("\nConverting to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantize for speed
    tflite_model = converter.convert()

    tflite_path = os.path.join(args.model_dir, "defect_model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    model_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"Model saved: {tflite_path} ({model_size_mb:.1f} MB)")

    # Save labels
    labels_path = os.path.join(args.model_dir, "labels.txt")
    with open(labels_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Labels saved: {labels_path}")

    # Save training config
    config = {
        "image_size": args.image_size,
        "class_names": class_names,
        "num_classes": num_classes,
        "final_accuracy": float(val_acc),
        "epochs_trained": len(history.history['loss']) + len(history_fine.history['loss']),
        "training_samples": train_generator.samples,
    }
    config_path = os.path.join(args.model_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Model    : {tflite_path}")
    print(f"  Accuracy : {val_acc * 100:.1f}%")
    print(f"  Classes  : {', '.join(class_names)}")
    print("=" * 60)
    print("\nNext step: Run 'python app.py' to start the detection system!")


if __name__ == "__main__":
    main()
