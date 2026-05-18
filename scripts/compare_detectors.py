"""
Compare Spatial and Frequency Detectors
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import *
from models.spatial_detector import SpatialDetector
from models.frequency_detector import FrequencyDetector

def load_spatial_model():
    """Load spatial detector"""
    model = SpatialDetector(MODEL_NAME, pretrained=False)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model

def load_frequency_model():
    """Load frequency detector"""
    detector = FrequencyDetector()
    detector.load(CHECKPOINTS_DIR / "frequency_detector.pkl")
    return detector

def evaluate_spatial(model, image_paths):
    """Evaluate spatial detector"""
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    predictions = []
    
    print("Testing spatial detector")
    for i, img_path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"  {i}/{len(image_paths)}")
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
        
        predictions.append(pred)
    
    return np.array(predictions)

def evaluate_frequency(detector, image_paths):
    """Evaluate frequency detector"""
    
    images = []
    print("Loading images")
    
    for i, img_path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"  {i}/{len(image_paths)}")
        
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        images.append(np.array(img))
    
    print("Extracting features")
    features = detector.extract_batch_features(images)
    
    print("Making predictions")
    predictions = detector.predict(features)
    
    return predictions

def main():
    
    print("=" * 70)
    print("COMPARING DETECTORS")
    print("=" * 70)
    
    print("\nLoading models")
    spatial_model = load_spatial_model()
    frequency_model = load_frequency_model()
    
    val_path = PROCESSED_DATA_DIR / "val_medium"
    real_images = list((val_path / "real").glob('*'))
    fake_images = list((val_path / "synthetic").glob('*'))
    
    all_images = real_images + fake_images
    all_labels = np.array([0] * len(real_images) + [1] * len(fake_images))
    
    print(f"\nTest set: {len(all_images)} images")
    print(f"  Real: {len(real_images)}")
    print(f"  Fake: {len(fake_images)}")
    
    # Evaluate spatial
    print("\n" + "=" * 70)
    print("SPATIAL DETECTOR")
    print("=" * 70)
    spatial_preds = evaluate_spatial(spatial_model, all_images)
    
    spatial_acc = accuracy_score(all_labels, spatial_preds)
    spatial_prec = precision_score(all_labels, spatial_preds)
    spatial_rec = recall_score(all_labels, spatial_preds)
    spatial_f1 = f1_score(all_labels, spatial_preds)
    
    print("\nResults:")
    print(f"  Accuracy:  {spatial_acc*100:.2f}%")
    print(f"  Precision: {spatial_prec*100:.2f}%")
    print(f"  Recall:    {spatial_rec*100:.2f}%")
    print(f"  F1-Score:  {spatial_f1*100:.2f}%")
    
    # Evaluate frequency
    print("\n" + "=" * 70)
    print("FREQUENCY DETECTOR")
    print("=" * 70)
    frequency_preds = evaluate_frequency(frequency_model, all_images)
    
    freq_acc = accuracy_score(all_labels, frequency_preds)
    freq_prec = precision_score(all_labels, frequency_preds)
    freq_rec = recall_score(all_labels, frequency_preds)
    freq_f1 = f1_score(all_labels, frequency_preds)
    
    print("\nResults:")
    print(f"  Accuracy:  {freq_acc*100:.2f}%")
    print(f"  Precision: {freq_prec*100:.2f}%")
    print(f"  Recall:    {freq_rec*100:.2f}%")
    print(f"  F1-Score:  {freq_f1*100:.2f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<15} {'Spatial':<12} {'Frequency':<12} {'Difference'}")
    print("-" * 70)
    print(f"{'Accuracy':<15} {spatial_acc*100:6.2f}%      {freq_acc*100:6.2f}%      {(spatial_acc-freq_acc)*100:+6.2f}%")
    print(f"{'Precision':<15} {spatial_prec*100:6.2f}%      {freq_prec*100:6.2f}%      {(spatial_prec-freq_prec)*100:+6.2f}%")
    print(f"{'Recall':<15} {spatial_rec*100:6.2f}%      {freq_rec*100:6.2f}%      {(spatial_rec-freq_rec)*100:+6.2f}%")
    print(f"{'F1-Score':<15} {spatial_f1*100:6.2f}%      {freq_f1*100:6.2f}%      {(spatial_f1-freq_f1)*100:+6.2f}%")
    
    agreement = np.sum(spatial_preds == frequency_preds) / len(all_labels)
    disagreement = np.sum(spatial_preds != frequency_preds)
    
    print(f"\nAgreement: {agreement*100:.2f}%")
    print(f"Disagreement: {disagreement} cases")

if __name__ == "__main__":
    main()