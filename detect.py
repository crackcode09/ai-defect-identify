#!/usr/bin/env python3
"""
detect.py — Defect detection engine using TensorFlow Lite.

This module loads a trained TFLite model and classifies images
for manufacturing defects. Used by the Flask web app (app.py).
"""

import json
import os
import time

import numpy as np
from PIL import Image


class DefectDetector:
    """Loads a TFLite model and classifies images for defects."""

    def __init__(self, model_dir="models"):
        """
        Initialize the detector with a trained model.

        Args:
            model_dir: Directory containing defect_model.tflite,
                       labels.txt, and model_config.json
        """
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "defect_model.tflite")
        self.labels_path = os.path.join(model_dir, "labels.txt")
        self.config_path = os.path.join(model_dir, "model_config.json")

        # Load model configuration
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            self.image_size = self.config.get("image_size", 224)
            self.class_names = self.config.get("class_names", [])
        else:
            self.image_size = 224
            self.class_names = []

        # Load labels (fallback)
        if not self.class_names and os.path.exists(self.labels_path):
            with open(self.labels_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]

        # Load TFLite model
        if os.path.exists(self.model_path):
            self._load_model()
            self.model_loaded = True
            print(f"Model loaded: {self.model_path}")
            print(f"Classes: {self.class_names}")
        else:
            self.model_loaded = False
            print(f"WARNING: No model found at {self.model_path}")
            print("Run train_model.py first to create a model.")

    def _load_model(self):
        """Load the TFLite model and set up the interpreter."""
        try:
            # Try tflite_runtime first (lighter, preferred on Pi)
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            # Fall back to full TensorFlow
            from tensorflow.lite.python.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Expected input shape
        self.input_shape = self.input_details[0]['shape']  # e.g., [1, 224, 224, 3]
        print(f"Model input shape: {self.input_shape}")

    def preprocess_image(self, image_path):
        """
        Load and preprocess an image for the model.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed numpy array ready for inference
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size))
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to 0-1
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    def preprocess_frame(self, frame):
        """
        Preprocess a camera frame (numpy array) for the model.

        Args:
            frame: OpenCV/picamera2 image array (BGR or RGB)

        Returns:
            Preprocessed numpy array ready for inference
        """
        img = Image.fromarray(frame)
        img = img.convert('RGB')
        img = img.resize((self.image_size, self.image_size))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_input):
        """
        Run defect detection on an image.

        Args:
            image_input: Either a file path (str) or numpy array (frame)

        Returns:
            dict with keys:
                - 'defect_type': Name of the detected class
                - 'confidence': Confidence score (0.0 to 1.0)
                - 'is_defective': True if defect detected (not 'good')
                - 'all_scores': Dict of all class scores
                - 'inference_time_ms': Time taken for inference
        """
        if not self.model_loaded:
            return {
                'defect_type': 'unknown',
                'confidence': 0.0,
                'is_defective': False,
                'all_scores': {},
                'inference_time_ms': 0,
                'error': 'No model loaded. Run train_model.py first.'
            }

        # Preprocess
        if isinstance(image_input, str):
            input_data = self.preprocess_image(image_input)
        else:
            input_data = self.preprocess_frame(image_input)

        # Run inference
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Process results
        scores = output_data[0]
        top_index = np.argmax(scores)
        confidence = float(scores[top_index])

        if top_index < len(self.class_names):
            defect_type = self.class_names[top_index]
        else:
            defect_type = f"class_{top_index}"

        # Build all scores dict
        all_scores = {}
        for i, score in enumerate(scores):
            name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            all_scores[name] = round(float(score) * 100, 1)  # Percentage

        return {
            'defect_type': defect_type,
            'confidence': round(confidence, 4),
            'confidence_pct': round(confidence * 100, 1),
            'is_defective': defect_type.lower() != 'good',
            'all_scores': all_scores,
            'inference_time_ms': round(inference_time, 1)
        }


class CameraCapture:
    """Handle camera capture on Raspberry Pi or fallback to webcam."""

    def __init__(self, resolution=(640, 640)):
        self.resolution = resolution
        self.camera = None
        self.use_picamera = False

        try:
            from picamera2 import Picamera2
            self.camera = Picamera2()
            config = self.camera.create_still_configuration(
                main={"size": resolution, "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            self.use_picamera = True
            time.sleep(2)
            print("Raspberry Pi Camera initialized")
        except (ImportError, RuntimeError):
            print("Pi Camera not available, trying webcam...")
            try:
                import cv2
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                if self.camera.isOpened():
                    print("Webcam initialized")
                else:
                    print("WARNING: No camera available!")
                    self.camera = None
            except ImportError:
                print("WARNING: Neither picamera2 nor OpenCV available!")
                self.camera = None

    def capture(self, save_path=None):
        """
        Capture a single frame.

        Args:
            save_path: Optional path to save the captured image

        Returns:
            numpy array of the captured image, or None on failure
        """
        if self.camera is None:
            return None

        if self.use_picamera:
            frame = self.camera.capture_array()
        else:
            import cv2
            ret, frame = self.camera.read()
            if not ret:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if save_path and frame is not None:
            img = Image.fromarray(frame)
            img.save(save_path, quality=95)

        return frame

    def release(self):
        """Release camera resources."""
        if self.camera is not None:
            if self.use_picamera:
                self.camera.stop()
            else:
                self.camera.release()


# --- Quick test ---
if __name__ == "__main__":
    import sys

    detector = DefectDetector()

    if len(sys.argv) > 1:
        # Test with a specific image
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            result = detector.predict(image_path)
            print(f"\nResults for: {image_path}")
            print(f"  Defect type : {result['defect_type']}")
            print(f"  Confidence  : {result['confidence_pct']}%")
            print(f"  Is defective: {result['is_defective']}")
            print(f"  Inference   : {result['inference_time_ms']} ms")
            print(f"  All scores  : {result['all_scores']}")
        else:
            print(f"File not found: {image_path}")
    else:
        print("\nUsage: python detect.py <image_path>")
        print("Or import DefectDetector in your own code.")
