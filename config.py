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


for directory in [CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# TRAINING HYPERPARAMETERS

# Image preprocessing
IMAGE_SIZE = 224  


BATCH_SIZE = 32   

LEARNING_RATE = 0.001  

NUM_EPOCHS = 10   

NUM_WORKERS = 4   

PATIENCE = 3      
MIN_DELTA = 0.001 

# MODEL CONFIGURATION

MODEL_NAME = "resnet18"  

PRETRAINED = True  

FREEZE_LAYERS = 10  


MODEL_CHECKPOINT = CHECKPOINTS_DIR / f"{MODEL_NAME}_detector_best.pth"

# DATA AUGMENTATION

USE_AUGMENTATION = True

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

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB channel means
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB channel std devs

# DEVICE 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    torch.backends.cudnn.benchmark = True
    
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: No GPU detected, training will be slow!")


LOG_INTERVAL = 10     
SAVE_BEST_ONLY = True 

LOG_FILE = LOGS_DIR / f"{MODEL_NAME}_training.log"


METRICS = ['accuracy', 'precision', 'recall', 'f1_score']

CLASS_NAMES = ['Real', 'AI-Generated']



TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

SUBSET_SIZES = {
    'tiny': {'train': 500, 'val': 100},      # Quick sanity check
    'small': {'train': 1000, 'val': 200},    # Debugging
    'medium': {'train': 2000, 'val': 500},   # Development
    'large': {'train': 5000, 'val': 1000},   # Pre-final
    'full': {'train': 60000, 'val': 20000}   # Full dataset
}

DEFAULT_SUBSET = 'medium'


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


if __name__ == "__main__":
    print_config()