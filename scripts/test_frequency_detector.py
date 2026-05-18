"""
Test Frequency Detector
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image

from config import *
from models.frequency_detector import FrequencyDetector

def test_image(detector, image_path):
    """Test single image"""
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img)
    
    features = detector.extract_batch_features([img_array])
    prediction = detector.predict(features)[0]
    probabilities = detector.predict_proba(features)[0]
    
    # Convert prediction to int
    prediction = int(prediction)
    
    return prediction, probabilities

def main():
    
    print("TESTING FREQUENCY DETECTOR")
    
    model_path = CHECKPOINTS_DIR / "frequency_detector.pkl"
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: python scripts/train_frequency_detector.py")
        return
    
    print(f"\nLoading model: {model_path}")
    detector = FrequencyDetector()
    detector.load(model_path)
    
    val_path = PROCESSED_DATA_DIR / "val_medium"
    real_dir = val_path / "real"
    fake_dir = val_path / "synthetic"
    
    # Test real images
    print("\n" + "-" * 70)
    print("Real Images")
    print("-" * 70)
    
    real_images = list(real_dir.glob('*'))[:5]
    for img_path in real_images:
        pred, probs = test_image(detector, img_path)
        label = "Real" if pred == 0 else "Fake"
        confidence = float(probs[pred]) * 100
        correct = "CORRECT" if pred == 0 else "WRONG"
        print(f"{correct:8} {img_path.name:30} {label:10} {confidence:6.1f}%")
    
    # Test fake images
    print("\n" + "-" * 70)
    print("Fake Images")
    print("-" * 70)
    
    fake_images = list(fake_dir.glob('*'))[:5]
    for img_path in fake_images:
        pred, probs = test_image(detector, img_path)
        label = "Real" if pred == 0 else "Fake"
        confidence = float(probs[pred]) * 100
        correct = "CORRECT" if pred == 1 else "WRONG"
        print(f"{correct:8} {img_path.name:30} {label:10} {confidence:6.1f}%")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()