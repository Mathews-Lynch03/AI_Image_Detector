"""
Download CIFAKE dataset from Kaggle

Dataset Information:
- Name: CIFAKE - Real and AI-Generated Synthetic Images  
- Size: ~6GB
- Composition:
  * 60,000 real images (from CIFAR-10)
  * 60,000 AI-generated images (Stable Diffusion, DALL-E, Midjourney)

Reference:
Bird, J. J., & Lotfi, A. (2023). CIFAKE: Real and AI-Generated Synthetic Images.
arXiv preprint arXiv:2303.14126.

Usage:
    python scripts/download_dataset.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import RAW_DATA_DIR

def check_kaggle_setup():
    """
    Verify Kaggle API is configured correctly
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("Kaggle API not configured!")
        return False
    
    print(f"Kaggle API configured: {kaggle_json}")
    return True

def download_cifake():
    """
    Download and extract CIFAKE dataset
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("CIFAKE DATASET DOWNLOAD")
    
    # Check Kaggle setup
    if not check_kaggle_setup():
        return False
    
    # Try importing kaggle
    try:
        import kaggle
        print("Kaggle module imported")
    except ImportError:
        print("Kaggle module not found!")
        print("Install with: pip install kaggle")
        return False
    
    # Create raw data directory
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownload location: {RAW_DATA_DIR}")
    
    # Check if already downloaded
    expected_dirs = [
        RAW_DATA_DIR / "train" / "REAL",
        RAW_DATA_DIR / "train" / "FAKE",
        RAW_DATA_DIR / "test" / "REAL",
        RAW_DATA_DIR / "test" / "FAKE"
    ]
    
    if all(d.exists() for d in expected_dirs):
        print("\nDataset already downloaded!")
        
        # Count images
        train_real = len(list(expected_dirs[0].glob('*')))
        train_fake = len(list(expected_dirs[1].glob('*')))
        test_real = len(list(expected_dirs[2].glob('*')))
        test_fake = len(list(expected_dirs[3].glob('*')))
        
        print(f"\nDataset Statistics:")
        print(f"   Training Real: {train_real:,} images")
        print(f"   Training Fake: {train_fake:,} images")
        print(f"   Test Real:     {test_real:,} images")
        print(f"   Test Fake:     {test_fake:,} images")
        print(f"   Total:         {train_real + train_fake + test_real + test_fake:,} images")
        
        return True
    
    # Download dataset
    print("\nDownloading CIFAKE dataset...")
    print(" This will take 10-20 minutes")
    print(" Size: ~6GB")
    
    try:
        kaggle.api.dataset_download_files(
            'birdy654/cifake-real-and-ai-generated-synthetic-images',
            path=str(RAW_DATA_DIR),
            unzip=True,
            quiet=False
        )
        
        print("\n Download complete!")
        
        # Verify structure
        print("\n Verifying dataset structure...")
        all_exist = True
        for dir_path in expected_dirs:
            exists = dir_path.exists()
            status = "Found" if exists else "Not Found"
            rel_path = dir_path.relative_to(RAW_DATA_DIR)
            print(f"   {status} {rel_path}")
            all_exist = all_exist and exists
        
        if all_exist:
            # Count images
            counts = [len(list(d.glob('*'))) for d in expected_dirs]
            
            print(f"\nDataset Statistics:")
            print(f"   Training Real: {counts[0]:,} images")
            print(f"   Training Fake: {counts[1]:,} images")
            print(f"   Test Real:     {counts[2]:,} images")
            print(f"   Test Fake:     {counts[3]:,} images")
            print(f"   Total:         {sum(counts):,} images")
            
            print("DATASET READY!")
            return True
        else:
            print("\nDataset structure incomplete")
            return False
            
    except Exception as e:
        print(f"\nDownload error: {e}")
        return False

def main():
    """Main entry point"""
    success = download_cifake()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()