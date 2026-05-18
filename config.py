"""
Central Configuration for AI Image Detector Project

This file stores all hyperparameters, paths, and settings.
Changes here propagate throughout the entire project.

References:
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
  Practical Methodology - discusses importance of organized configs
"""

from pathlib import Path
import torch

print("Loading configuration...")

# PROJECT PATHS

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# CIFAKE dataset structure
TRAIN_REAL = RAW_DATA_DIR / "train" / "REAL"
TRAIN_FAKE = RAW_DATA_DIR / "train" / "FAKE"
TEST_REAL = RAW_DATA_DIR / "test" / "REAL"
TEST_FAKE = RAW_DATA_DIR / "test" / "FAKE"

# Ensure critical directories exist
for directory in [CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# TRAINING HYPERPARAMETERS

# Image preprocessing
IMAGE_SIZE = 224  # Standard for ResNet (224x224 pixels)
                  # Reference: He et al. (2016) "Deep Residual Learning"

# Training parameters
BATCH_SIZE = 32   # Images processed simultaneously
                  # 32 works well for 12GB GPU
                  # Larger = faster training but more memory

LEARNING_RATE = 0.001  # Step size for gradient descent
                       # 0.001 is standard starting point for Adam
                       # Reference: Kingma & Ba (2014) "Adam Optimizer"

NUM_EPOCHS = 10   # Complete passes through training data
                  # Start with 10, monitor for convergence

NUM_WORKERS = 4   # Parallel data loading threads
                  # 4 is good for modern CPUs
                  # Set to 0 if you encounter errors

# Early stopping
PATIENCE = 3      # Stop if no improvement for N epochs
MIN_DELTA = 0.001 # Minimum improvement to count as progress

# MODEL CONFIGURATION

MODEL_NAME = "resnet18"  # Options: resnet18, resnet50, mobilenet_v2
                         # resnet18: Fast, 11M params, good baseline
                         # resnet50: Slower, 25M params, higher accuracy

PRETRAINED = True  # Use ImageNet pre-trained weights
                   # True = Transfer learning (RECOMMENDED)
                   # False = Train from scratch (much slower)

FREEZE_LAYERS = 10  # Number of early layers to freeze
                    # Frozen layers keep ImageNet features
                    # Only later layers learn fake detection

# Model save path
MODEL_CHECKPOINT = CHECKPOINTS_DIR / f"{MODEL_NAME}_detector_best.pth"

# DATA AUGMENTATION
# Data augmentation helps prevent overfitting
# Reference: Shorten & Khoshgoftaar (2019) "Survey on Image Data Augmentation"

USE_AUGMENTATION = True

# Augmentation parameters
AUGMENTATION_CONFIG = {
    'random_rotation': 15,        # Rotate ±15 degrees
    'horizontal_flip_prob': 0.5,  # 50% chance to flip
    'color_jitter': {
        'brightness': 0.2,        # ±20% brightness
        'contrast': 0.2,          # ±20% contrast
        'saturation': 0.1,        # ±10% saturation
        'hue': 0.05              # ±5% hue
    }
}

# NORMALIZATION (ImageNet statistics)

# These are standard values for ImageNet pre-trained models
# Reference: He et al. (2016) - ResNet paper

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB channel means
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB channel std devs

# DEVICE CONFIGURATION

# Automatically use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU optimization settings
if torch.cuda.is_available():
    # Enable TensorFloat32 for faster computation on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN auto-tuner for optimal convolution algorithms
    torch.backends.cudnn.benchmark = True
    
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: No GPU detected, training will be slow!")

# LOGGING CONFIGURATION

LOG_INTERVAL = 10     # Print stats every N batches
SAVE_BEST_ONLY = True # Only save when validation improves

# Log file path
LOG_FILE = LOGS_DIR / f"{MODEL_NAME}_training.log"

# EVALUATION METRICS
# Reference: 

METRICS = ['accuracy', 'precision', 'recall', 'f1_score']

# Class names
CLASS_NAMES = ['Real', 'AI-Generated']

# DATASET CONFIGURATION


# Dataset split ratios 
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

# Subset sizes for quick testing
SUBSET_SIZES = {
    'tiny': {'train': 500, 'val': 100},      # Quick sanity check
    'small': {'train': 1000, 'val': 200},    # Debugging
    'medium': {'train': 2000, 'val': 500},   # Development
    'large': {'train': 5000, 'val': 1000},   # Pre-final
    'full': {'train': 60000, 'val': 20000}   # Full dataset
}

# Default subset to use
DEFAULT_SUBSET = 'medium'

# UTILITY FUNCTIONS

def print_config():
    """Print current configuration summary"""
    print("\n" + "=" * 70)
    print("AI IMAGE DETECTOR - CONFIGURATION")
    print("=" * 70)
    
    print("\n Paths:")
    print(f"   Project root: {PROJECT_ROOT}")
    print(f"   Data: {DATA_DIR}")
    print(f"   Checkpoints: {CHECKPOINTS_DIR}")
    print(f"   Results: {RESULTS_DIR}")
    
    print("\n Model:")
    print(f"   Architecture: {MODEL_NAME}")
    print(f"   Pre-trained: {PRETRAINED}")
    print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    print("\n Training:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Augmentation: {USE_AUGMENTATION}")
    
    print("\n Device:")
    print(f"   Using: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\n" + "=" * 70 + "\n")

def get_subset_paths(subset_name='medium'):
    """Get paths for a specific dataset subset"""
    train_path = PROCESSED_DATA_DIR / f"train_{subset_name}"
    val_path = PROCESSED_DATA_DIR / f"val_{subset_name}"
    return train_path, val_path

# INITIALIZE

if __name__ == "__main__":
    print_config()