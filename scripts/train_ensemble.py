"""
Evaluate and optimize ensemble detector
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
import matplotlib.pyplot as plt

from config import *
from models.ensemble_detector import EnsembleDetector

def load_test_data(real_dir, fake_dir):
    """Load test images"""
    
    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)
    
    real_paths = list(real_dir.glob('*'))
    fake_paths = list(fake_dir.glob('*'))
    
    all_paths = real_paths + fake_paths
    labels = np.array([0] * len(real_paths) + [1] * len(fake_paths))
    
    return all_paths, labels

def prepare_images(image_paths, transform):
    """Prepare images for both models"""
    
    tensors = []
    arrays = []
    
    print(f"Loading {len(image_paths)} images")
    
    for i, img_path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"  {i}/{len(image_paths)}")
        
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # For spatial model
        tensor = transform(img)
        tensors.append(tensor)
        
        # For frequency model
        array = np.array(img_resized)
        arrays.append(array)
    
    return torch.stack(tensors), arrays

def evaluate_ensemble(ensemble, image_tensors, image_arrays, labels, batch_size=32):
    """Evaluate ensemble on dataset"""
    
    all_predictions = []
    
    num_batches = (len(labels) + batch_size - 1) // batch_size
    
    print("Making predictions")
    for i in range(num_batches):
        if i % 10 == 0:
            print(f"  Batch {i}/{num_batches}")
        
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(labels))
        
        batch_tensors = image_tensors[start_idx:end_idx].to(ensemble.device)
        batch_arrays = image_arrays[start_idx:end_idx]
        
        predictions, _ = ensemble.predict_batch(batch_tensors, batch_arrays)
        all_predictions.extend(predictions)
    
    predictions = np.array(all_predictions)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    return accuracy, precision, recall, f1, predictions

def find_best_weights(ensemble, image_tensors, image_arrays, labels):
    """Try different weight combinations"""
    
    print("\nSearching for best weights")
    print("-" * 70)
    
    best_acc = 0
    best_weights = (0.6, 0.4)
    
    weight_combinations = [
        (1.0, 0.0),  # Spatial only
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),  # Default
        (0.5, 0.5),
        (0.4, 0.6),
        (0.3, 0.7),
        (0.2, 0.8),
        (0.1, 0.9),
        (0.0, 1.0),  # Frequency only
    ]
    
    for spatial_w, freq_w in weight_combinations:
        ensemble.spatial_weight = spatial_w
        ensemble.frequency_weight = freq_w
        
        acc, _, _, _, _ = evaluate_ensemble(ensemble, image_tensors, image_arrays, labels)
        
        print(f"Spatial: {spatial_w:.1f}, Frequency: {freq_w:.1f} -> Accuracy: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_weights = (spatial_w, freq_w)
    
    print(f"\nBest weights: Spatial {best_weights[0]:.1f}, Frequency {best_weights[1]:.1f}")
    print(f"Best accuracy: {best_acc*100:.2f}%")
    
    return best_weights

def main():

    print("ENSEMBLE DETECTOR EVALUATION")
    
    # Load ensemble
    print("\nLoading ensemble detector")
    ensemble = EnsembleDetector(spatial_weight=0.6, frequency_weight=0.4)
    
    spatial_checkpoint = MODEL_CHECKPOINT
    frequency_checkpoint = CHECKPOINTS_DIR / "frequency_detector.pkl"
    
    if not spatial_checkpoint.exists():
        print(f"Spatial model not found: {spatial_checkpoint}")
        return
    
    if not frequency_checkpoint.exists():
        print(f"Frequency model not found: {frequency_checkpoint}")
        return
    
    ensemble.load_models(spatial_checkpoint, frequency_checkpoint, MODEL_NAME)
    print("Models loaded")
    
    # Load validation data
    val_path = PROCESSED_DATA_DIR / "val_medium"
    image_paths, labels = load_test_data(
        val_path / "real",
        val_path / "synthetic"
    )
    
    print(f"\nValidation set: {len(labels)} images")
    print(f"  Real: {sum(labels == 0)}")
    print(f"  Fake: {sum(labels == 1)}")
    
    # Prepare images
    print("\nPreparing images")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    image_tensors, image_arrays = prepare_images(image_paths, transform)
    print(f"Tensor shape: {image_tensors.shape}")
    
    # Find best weights
    best_weights = find_best_weights(ensemble, image_tensors, image_arrays, labels)
    
    # Set best weights
    ensemble.spatial_weight = best_weights[0]
    ensemble.frequency_weight = best_weights[1]
    
    # Final evaluation
    print("FINAL EVALUATION WITH BEST WEIGHTS")
    
    acc, prec, rec, f1, predictions = evaluate_ensemble(
        ensemble, image_tensors, image_arrays, labels
    )
    
    print("\nEnsemble Results:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    
    # Save configuration
    config_path = CHECKPOINTS_DIR / "ensemble_config.txt"
    with open(config_path, 'w') as f:
        f.write(f"Spatial weight: {best_weights[0]}\n")
        f.write(f"Frequency weight: {best_weights[1]}\n")
        f.write(f"Accuracy: {acc*100:.2f}%\n")
        f.write(f"Precision: {prec*100:.2f}%\n")
        f.write(f"Recall: {rec*100:.2f}%\n")
        f.write(f"F1-Score: {f1*100:.2f}%\n")
    
    print(f"\nConfiguration saved: {config_path}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()