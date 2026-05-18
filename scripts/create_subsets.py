"""
Create training/validation subsets of different sizes

This creates multiple subset sizes
- tiny (1k total): Quick testing (~30 sec/epoch)
- small (2k total): Debugging (~1 min/epoch)
- medium (4k total): Development (~2-3 min/epoch) ← We'll use this
- large (10k total): Pre-final training (~5-8 min/epoch)

Usage:
    python scripts/create_subsets.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import TRAIN_REAL, TRAIN_FAKE, TEST_REAL, TEST_FAKE, PROCESSED_DATA_DIR
from scripts.data_utils import create_subset

def main():
    """Create all subset sizes"""
    
    print("CREATING DATASET SUBSETS")
    
    
    # Verify source data exists
    if not TRAIN_REAL.exists():
        print("\n Source dataset not found!")
        print("Run: python scripts/download_dataset.py")
        return
    
    # Define subset configurations
    subsets = [
        ("tiny", 500, 100),
        ("small", 1000, 200),
        ("medium", 2000, 500),    
        ("large", 5000, 1000),
    ]
    
    for name, train_n, val_n in subsets:
        print(f"\n{'='*70}")
        print(f"Creating '{name}' subset")
        print(f"{'='*70}")
        
        # Create training subset
        print(f"\nTraining set ({train_n} per class):")
        create_subset(
            TRAIN_REAL,
            TRAIN_FAKE,
            PROCESSED_DATA_DIR / f"train_{name}",
            n_samples=train_n
        )
        
        # Create validation subset
        print(f"\n Validation set ({val_n} per class):")
        create_subset(
            TEST_REAL,
            TEST_FAKE,
            PROCESSED_DATA_DIR / f"val_{name}",
            n_samples=val_n
        )
    
    print("\n" + "=" * 70)
    print("ALL SUBSETS CREATED!")
    print("=" * 70)
    
    # Print summary
    print("\nSubset Summary:")
    print(f"{'Subset':<10} {'Train Size':<12} {'Val Size':<12}")
    for name, train_n, val_n in subsets:
        print(f"{name:<10} {train_n*2:>6,}        {val_n*2:>6,}")
    
    print("\nRecommended for development: 'medium' subset (4,000 images)")

if __name__ == "__main__":
    main()