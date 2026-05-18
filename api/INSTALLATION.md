# Complete Installation Guide for FastAPI Backend

## Step 1: Install Required Packages

Open Anaconda Prompt and activate your environment:

```bash
conda activate ai-detector
```

Install FastAPI and dependencies:

```bash
pip install fastapi uvicorn[standard] python-multipart
```

## Step 2: Copy API Files

Copy the `api` folder to your project directory:

```
C:\Users\smath\OneDrive\Documents\DKIT_Year_4\Project\AI-Image-Detector\api\
```

Your project structure should look like:

```
AI-Image-Detector/
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── test_api.py
│   ├── example_client.py
│   └── README.md
├── models/
│   ├── spatial_detector.py
│   ├── frequency_detector.py
│   └── ensemble_detector.py
├── checkpoints/
│   ├── resnet18_detector_best.pth
│   ├── frequency_detector.pkl
│   └── ensemble_config.txt
├── config.py
└── ...
```

## Step 3: Verify Checkpoint Files

Make sure these files exist:

```
checkpoints/resnet18_detector_best.pth
checkpoints/frequency_detector.pkl
checkpoints/ensemble_config.txt
```

If any are missing, run the training scripts first:

```bash
python scripts/train_detector.py
python scripts/train_frequency_detector.py
python scripts/train_ensemble.py
```

## Step 4: Test the Server

Navigate to your project directory:

```bash
cd C:\Users\smath\OneDrive\Documents\DKIT_Year_4\Project\AI-Image-Detector
```

Start the server:

```bash
uvicorn api.main:app --reload
```

You should see output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Loading models...
INFO:     Spatial model loaded from checkpoints/resnet18_detector_best.pth
INFO:     Frequency model loaded from checkpoints/frequency_detector.pkl
INFO:     Ensemble weights: 0.6 / 0.4
INFO:     All models loaded successfully
```

## Step 5: Verify the API Works

Open a new terminal and run:

```bash
curl http://localhost:8000/health
```

Or visit http://localhost:8000/docs in your browser to see the interactive API documentation.

## Step 6: Test Detection

Run the test script:

```bash
python api/test_api.py
```

Or test with an actual image:

```bash
python api/test_api.py data/raw/test/REAL/0001.png
```

## Alternative: Use the Batch File (Windows)

Double-click `start_server.bat` to start the server automatically.

## Common Issues and Solutions

### Issue: "Module not found: fastapi"

Solution:
```bash
conda activate ai-detector
pip install fastapi uvicorn python-multipart
```

### Issue: "Checkpoint file not found"

Solution: Train the models first:
```bash
python scripts/train_detector.py
python scripts/train_frequency_detector.py
python scripts/train_ensemble.py
```

### Issue: "Port 8000 already in use"

Solution: Use a different port:
```bash
uvicorn api.main:app --reload --port 8001
```

### Issue: CUDA out of memory

Solution: The API automatically handles this by processing one image at a time.

### Issue: Models load slowly

This is normal on first startup (5-10 seconds). Models stay in memory after loading.

## Next Steps

After the API is running:

1. Test all endpoints using the interactive docs at http://localhost:8000/docs
2. Try the example client: `python api/example_client.py path/to/image.jpg`
3. Integrate with a frontend application
4. Deploy to production following DEPLOYMENT.md

## API Endpoints Summary

Once running, you can access:

- **Health Check**: GET http://localhost:8000/
- **Detection (Ensemble)**: POST http://localhost:8000/detect
- **Spatial Only**: POST http://localhost:8000/detect/spatial
- **Frequency Only**: POST http://localhost:8000/detect/frequency
- **Model Info**: GET http://localhost:8000/models/info
- **Interactive Docs**: http://localhost:8000/docs

## Example Usage from Python

```python
from api.example_client import AIDetectorClient

client = AIDetectorClient()
result = client.detect("path/to/image.jpg")
print(f"{result['prediction']}: {result['confidence']:.1f}%")
```

## Example Usage from Command Line

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@path/to/image.jpg"
```

## Support

If you encounter issues:

1. Check that conda environment is activated
2. Verify all checkpoint files exist
3. Check server logs for error messages
4. Try running with verbose logging: `uvicorn api.main:app --reload --log-level debug`
