"""
Training script for AI Image Detector

training pipeline
Load and prepare data
Create model
Train with validation
Save best model
Plot results
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from config import *
from models.spatial_detector import SpatialDetector
from scripts.data_utils import ImageDataset


def get_transforms(training=True):
    """
    Get image transformations for training or validation

    """
    if training and USE_AUGMENTATION:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(AUGMENTATION_CONFIG['random_rotation']),
            transforms.RandomHorizontalFlip(
                p=AUGMENTATION_CONFIG['horizontal_flip_prob']
            ),
            transforms.ColorJitter(
                brightness=AUGMENTATION_CONFIG['color_jitter']['brightness'],
                contrast=AUGMENTATION_CONFIG['color_jitter']['contrast'],
                saturation=AUGMENTATION_CONFIG['color_jitter']['saturation'],
                hue=AUGMENTATION_CONFIG['color_jitter']['hue']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

# TRAINING FUNCTIONS

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch

    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Validate model on validation set
    
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=8)
    ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-o', label='Training Accuracy', linewidth=2, markersize=8)
    ax2.plot(epochs, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training plot saved: {save_path}")
    plt.show()


# MAIN TRAINING FUNCTION

def main():
    
    print("\n" + "=" * 70)
    print("AI IMAGE DETECTOR - TRAINING")
    print("=" * 70)
    
    # Print configuration
    print_config()
    
    # Get dataset paths
    subset = DEFAULT_SUBSET  
    train_path, val_path = get_subset_paths(subset)
    
    # Verify subset exists
    train_real = train_path / "real"
    train_fake = train_path / "synthetic"
    val_real = val_path / "real"
    val_fake = val_path / "synthetic"
    
    if not all([train_real.exists(), train_fake.exists(), val_real.exists(), val_fake.exists()]):
        print("\nDataset subset not found!")
        print(f"Expected: {train_path}")
        print("\nRun: python scripts/create_subsets.py")
        return
    
    print(f"\nLoading {subset} subset...")
    train_transform = get_transforms(training=True)
    val_transform = get_transforms(training=False)
    
    # Create datasets
    train_dataset = ImageDataset(train_real, train_fake, transform=train_transform)
    val_dataset = ImageDataset(val_real, val_fake, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nData loaded:")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    

    print(f"\nCreating {MODEL_NAME} model...")
    model = SpatialDetector(MODEL_NAME, pretrained=PRETRAINED, freeze_layers=FREEZE_LAYERS)
    model = model.to(DEVICE)
    
    total, trainable = model.count_parameters()
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("=" * 70)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    start_time = datetime.now()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 70)
        

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_acc)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, MODEL_CHECKPOINT)
            print(f"   Best model saved! (Val Acc: {val_acc:.2f}%)")
    

    elapsed_time = datetime.now() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f" Total time: {elapsed_time}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_CHECKPOINT}")
    
    plot_path = RESULTS_DIR / f"{MODEL_NAME}_training_history.png"
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    print("\nTraining history plotted and saved")
    print("=" * 70)


if __name__ == "__main__":
    main()