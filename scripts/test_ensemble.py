"""
Test ensemble detector on individual images
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from config import *
from models.ensemble_detector import EnsembleDetector

def test_image(ensemble, image_path, transform):
    """Test single image with ensemble"""
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Prepare for spatial model
    img_tensor = transform(img).unsqueeze(0).to(ensemble.device)
    
    # Prepare for frequency model
    img_array = np.array(img_resized)
    
    # Predict
    prediction, confidence, spatial_probs, frequency_probs = ensemble.predict(
        img_tensor, img_array
    )
    
    return prediction, confidence, spatial_probs, frequency_probs

def main():
    
    print("TESTING ENSEMBLE DETECTOR")

    # Load ensemble
    print("\nLoading ensemble")
    ensemble = EnsembleDetector(spatial_weight=0.6, frequency_weight=0.4)
    
    spatial_checkpoint = MODEL_CHECKPOINT
    frequency_checkpoint = CHECKPOINTS_DIR / "frequency_detector.pkl"
    
    if not spatial_checkpoint.exists() or not frequency_checkpoint.exists():
        print("Models not found")
        return
    
    ensemble.load_models(spatial_checkpoint, frequency_checkpoint, MODEL_NAME)
    
    # Check for custom weights
    config_path = CHECKPOINTS_DIR / "ensemble_config.txt"
    if config_path.exists():
        with open(config_path, 'r') as f:
            for line in f:
                if line.startswith("Spatial weight:"):
                    ensemble.spatial_weight = float(line.split(":")[1].strip())
                elif line.startswith("Frequency weight:"):
                    ensemble.frequency_weight = float(line.split(":")[1].strip())
        print(f"Using optimized weights: {ensemble.spatial_weight:.1f} / {ensemble.frequency_weight:.1f}")
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Test images
    val_path = PROCESSED_DATA_DIR / "val_medium"
    
    # Test real images
    print("\n" + "-" * 70)
    print("Real Images")
    print("-" * 70)
    print(f"{'Result':8} {'Filename':30} {'Ensemble':10} {'Conf':6} {'Spatial':8} {'Frequency':8}")
    print("-" * 70)
    
    real_images = list((val_path / "real").glob('*'))[:5]
    for img_path in real_images:
        pred, conf, spatial_probs, freq_probs = test_image(ensemble, img_path, transform)
        
        label = "Real" if pred == 0 else "Fake"
        correct = "CORRECT" if pred == 0 else "WRONG"
        
        spatial_conf = float(spatial_probs[0]) * 100
        freq_conf = float(freq_probs[0]) * 100
        
        print(f"{correct:8} {img_path.name:30} {label:10} {conf*100:5.1f}% {spatial_conf:7.1f}% {freq_conf:7.1f}%")
    
    # Test fake images
    print("\n" + "-" * 70)
    print("Fake Images")
    print("-" * 70)
    print(f"{'Result':8} {'Filename':30} {'Ensemble':10} {'Conf':6} {'Spatial':8} {'Frequency':8}")
    print("-" * 70)
    
    fake_images = list((val_path / "synthetic").glob('*'))[:5]
    for img_path in fake_images:
        pred, conf, spatial_probs, freq_probs = test_image(ensemble, img_path, transform)
        
        label = "Real" if pred == 0 else "Fake"
        correct = "CORRECT" if pred == 1 else "WRONG"
        
        spatial_conf = float(spatial_probs[1]) * 100
        freq_conf = float(freq_probs[1]) * 100
        
        print(f"{correct:8} {img_path.name:30} {label:10} {conf*100:5.1f}% {spatial_conf:7.1f}% {freq_conf:7.1f}%")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()