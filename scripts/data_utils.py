"""
Custom PyTorch Dataset class for loading images
Functions to create training subsets
Image counting and validation utilities

References:
- PyTorch Data Loading Tutorial: pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import shutil
from tqdm import tqdm

class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for real/fake image classification
    
    This class handles:
        Loading images from directories
        Labeling (0=real, 1=fake)
        Applying transformations
        Error handling for corrupted images
    
        real_dir (str/Path): Directory containing real images
        fake_dir (str/Path): Directory containing AI-generated images
        transform (callable, optional): Transform to apply to images
    """
    
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load real images (label = 0)
        real_dir = Path(real_dir)
        if real_dir.exists():
            for img_path in real_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(str(img_path))
                    self.labels.append(0)
        
        # Load fake images (label = 1)
        fake_dir = Path(fake_dir)
        if fake_dir.exists():
            for img_path in fake_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(str(img_path))
                    self.labels.append(1)
        
        print(f"Loaded {len(self.images)} images:")
        print(f"  Real: {sum(1 for l in self.labels if l == 0):,}")
        print(f"  Fake: {sum(1 for l in self.labels if l == 1):,}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Load and return one image-label pair
        
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if loading fails
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224))), label
            return Image.new('RGB', (224, 224)), label


def count_images(directory):
    """
    Count number of images in a directory
    
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    count = sum(1 for f in directory.glob('*') if f.suffix.lower() in extensions)
    return count


def create_subset(source_real, source_fake, dest_dir, n_samples, seed=42):
    """
    Create a balanced subset of images for faster experimentation
    
    """
    random.seed(seed)
    
    dest_real = Path(dest_dir) / "real"
    dest_fake = Path(dest_dir) / "synthetic"
    
    # Create directories
    dest_real.mkdir(parents=True, exist_ok=True)
    dest_fake.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    source_real = Path(source_real)
    source_fake = Path(source_fake)
    
    real_files = [f for f in source_real.glob('*') 
                  if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    fake_files = [f for f in source_fake.glob('*') 
                  if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    print(f"Available: {len(real_files):,} real, {len(fake_files):,} fake")
    
    # Random sampling (ensures no duplicates)
    real_sample = random.sample(real_files, min(n_samples, len(real_files)))
    fake_sample = random.sample(fake_files, min(n_samples, len(fake_files)))
    
    # Copy files with progress bar
    for i, src in enumerate(tqdm(real_sample, desc="Copying real images")):
        dest = dest_real / f"real_{i:05d}{src.suffix}"
        shutil.copy2(src, dest)
    
    for i, src in enumerate(tqdm(fake_sample, desc="Copying fake images")):
        dest = dest_fake / f"fake_{i:05d}{src.suffix}"
        shutil.copy2(src, dest)
    
    # Verify
    copied_real = len(list(dest_real.glob('*')))
    copied_fake = len(list(dest_fake.glob('*')))
    
    print(f"Created {dest_dir.name}:")
    print(f"   Real: {copied_real:,}")
    print(f"   Fake: {copied_fake:,}")
    print(f"   Total: {copied_real + copied_fake:,}")
    
    return copied_real, copied_fake


# TEST CODE

if __name__ == "__main__":
    """Test the data utilities"""
    from config import TRAIN_REAL, TRAIN_FAKE
    
    print("TESTING DATA UTILITIES")
    
    # Test counting
    print("\n📊 Counting images in dataset...")
    if TRAIN_REAL.exists():
        real_count = count_images(TRAIN_REAL)
        fake_count = count_images(TRAIN_FAKE)
        
        print(f"Training Real: {real_count:,}")
        print(f"Training Fake: {fake_count:,}")
    else:
        print("Dataset not found!")
        print("Run: python scripts/download_dataset.py")