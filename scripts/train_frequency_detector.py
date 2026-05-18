"""
Train Frequency Detector
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import *
from models.frequency_detector import FrequencyDetector

def load_images(directory, max_samples=None):
    """Load images from directory"""
    
    directory = Path(directory)
    image_files = list(directory.glob('*'))
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    images = []
    print(f"Loading {len(image_files)} images from {directory.name}")
    
    for i, img_path in enumerate(image_files):
        if i % 500 == 0:
            print(f"  Loaded {i}/{len(image_files)}")
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            images.append(np.array(img))
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
            continue
    
    return images

def extract_features(real_dir, fake_dir, detector):
    """Extract features from real and fake images"""
    
    print("\nExtracting features")
    
    real_images = load_images(real_dir)
    fake_images = load_images(fake_dir)
    
    print("Extracting features from real images")
    real_features = detector.extract_batch_features(real_images)
    
    print("Extracting features from fake images")
    fake_features = detector.extract_batch_features(fake_images)
    
    features = np.vstack([real_features, fake_features])
    labels = np.hstack([
        np.zeros(len(real_features)),
        np.ones(len(fake_features))
    ])
    
    return features, labels

def evaluate(detector, features, labels):
    """Evaluate model"""
    
    predictions = detector.predict(features)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    return accuracy, precision, recall, f1

def main():
    
    print("FREQUENCY DETECTOR TRAINING")
    
    subset = DEFAULT_SUBSET
    train_path, val_path = get_subset_paths(subset)
    
    train_real = train_path / "real"
    train_fake = train_path / "synthetic"
    val_real = val_path / "real"
    val_fake = val_path / "synthetic"
    
    if not train_real.exists():
        print("Dataset not found")
        print("Run: python scripts/create_subsets.py")
        return
    
    print(f"\nUsing dataset: {subset}")
    print(f"Training path: {train_path}")
    print(f"Validation path: {val_path}")
    
    # Create detector
    detector = FrequencyDetector(num_bands=8, n_estimators=100)
    
    # Extract training features

    print("TRAINING DATA")
    train_features, train_labels = extract_features(train_real, train_fake, detector)
    print(f"Feature shape: {train_features.shape}")
    
    # Extract validation features
    print("VALIDATION DATA")
    val_features, val_labels = extract_features(val_real, val_fake, detector)
    print(f"Feature shape: {val_features.shape}")
    
    # Train
    print("TRAINING")
    detector.train(train_features, train_labels)
    
    # Evaluate
    print("\nEvaluating on training set")
    train_acc, train_prec, train_rec, train_f1 = evaluate(detector, train_features, train_labels)
    
    print("\nTraining Results:")
    print(f"  Accuracy:  {train_acc*100:.2f}%")
    print(f"  Precision: {train_prec*100:.2f}%")
    print(f"  Recall:    {train_rec*100:.2f}%")
    print(f"  F1-Score:  {train_f1*100:.2f}%")
    
    print("\nEvaluating on validation set")
    val_acc, val_prec, val_rec, val_f1 = evaluate(detector, val_features, val_labels)
    
    print("\nValidation Results:")
    print(f"  Accuracy:  {val_acc*100:.2f}%")
    print(f"  Precision: {val_prec*100:.2f}%")
    print(f"  Recall:    {val_rec*100:.2f}%")
    print(f"  F1-Score:  {val_f1*100:.2f}%")
    
    # Save
    model_path = CHECKPOINTS_DIR / "frequency_detector.pkl"
    detector.save(model_path)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Model saved: {model_path}")
    print(f"Validation accuracy: {val_acc*100:.2f}%")

if __name__ == "__main__":
    main()