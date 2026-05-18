@echo off
REM Git Commit Script for AI Image Detector Project
REM Run this from your project root directory

echo Setting up Git repository...
echo.

REM Initialize git if not already done
git init

echo Creating organized commits...
echo.

REM Commit 1: Initial setup
git add .gitignore README.md
git commit -m "Initial commit: Project structure and documentation"

REM Commit 2: Configuration
git add config.py
git commit -m "Add project configuration file"

REM Commit 3: Data utilities
git add scripts/__init__.py scripts/data_utils.py
git commit -m "Implement dataset utilities and data loading"

REM Commit 4: Spatial detector
git add models/__init__.py models/spatial_detector.py
git commit -m "Implement spatial detector using ResNet18"

REM Commit 5: Frequency detector
git add models/frequency_detector.py
git commit -m "Implement frequency-domain detector using FFT"

REM Commit 6: Ensemble detector
git add models/ensemble_detector.py
git commit -m "Implement ensemble detector with weighted voting"

REM Commit 7: Dataset download
git add scripts/download_dataset.py
git commit -m "Add dataset download script for CIFAKE"

REM Commit 8: Subset creation
git add scripts/create_subsets.py
git commit -m "Add dataset subset creation for faster training"

REM Commit 9: Spatial training
git add scripts/train_detector.py
git commit -m "Implement spatial detector training pipeline"

REM Commit 10: Frequency training
git add scripts/train_frequency_detector.py
git commit -m "Implement frequency detector training pipeline"

REM Commit 11: Ensemble training
git add scripts/train_ensemble.py
git commit -m "Implement ensemble weight optimization"

REM Commit 12: Testing scripts
git add scripts/test_model.py scripts/test_frequency_detector.py scripts/test_ensemble.py
git commit -m "Add testing scripts for all three detectors"

REM Commit 13: Comparison scripts
git add scripts/compare_detectors.py scripts/compare_all_methods.py
git commit -m "Add performance comparison scripts"

REM Commit 14: API backend
git add api/__init__.py api/main.py
git commit -m "Implement FastAPI backend with REST endpoints"

REM Commit 15: API testing
git add api/test_api.py api/example_client.py api/start_server.bat
git commit -m "Add API testing utilities and example client"

REM Commit 16: API documentation
git add api/README.md api/INSTALLATION.md api/DEPLOYMENT.md api/requirements.txt
git commit -m "Add comprehensive API documentation"

REM Commit 17: Ensemble config
git add checkpoints/ensemble_config.txt
git commit -m "Add ensemble configuration with optimal weights"

REM Commit 18: Remaining files
git add .
git commit -m "Add remaining project files and documentation"

echo.
echo ========================================
echo All commits created successfully!
echo ========================================
echo.
echo Next steps to upload to GitHub:
echo.
echo 1. Go to https://github.com and create a new repository
echo    - Name it: AI-Image-Detector
echo    - Do NOT initialize with README
echo.
echo 2. Copy your repository URL, then run these commands:
echo.
echo    git remote add origin https://github.com/YOUR_USERNAME/AI-Image-Detector.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo Replace YOUR_USERNAME with your actual GitHub username
echo.
pause
