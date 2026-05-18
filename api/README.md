# AI Image Detector API

FastAPI backend for detecting AI-generated images using ensemble detection.

## Installation

Make sure you have the required packages installed:

```bash
conda activate ai-detector
pip install fastapi uvicorn python-multipart
```

## Running the Server

### Development Mode (with auto-reload)

```bash
cd C:\Users\smath\OneDrive\Documents\DKIT_Year_4\Project\AI-Image-Detector
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
cd C:\Users\smath\OneDrive\Documents\DKIT_Year_4\Project\AI-Image-Detector
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### Health Check

**GET** `/`

Returns basic health status.

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "status": "running",
  "models_loaded": true,
  "device": "cuda"
}
```

### Detailed Health Check

**GET** `/health`

Returns detailed health information.

```bash
curl http://localhost:8000/health
```

### Ensemble Detection (Main Endpoint)

**POST** `/detect`

Upload an image and get predictions from all three methods.

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@path/to/image.jpg"
```

Response:
```json
{
  "prediction": "AI-Generated",
  "confidence": 85.7,
  "spatial_prediction": "AI-Generated",
  "spatial_confidence": 81.4,
  "frequency_prediction": "AI-Generated",
  "frequency_confidence": 76.3,
  "ensemble_weights": {
    "spatial": 0.6,
    "frequency": 0.4
  }
}
```

### Spatial-Only Detection

**POST** `/detect/spatial`

Use only the spatial detector.

```bash
curl -X POST http://localhost:8000/detect/spatial \
  -F "file=@path/to/image.jpg"
```

Response:
```json
{
  "method": "spatial",
  "prediction": "AI-Generated",
  "confidence": 81.4,
  "probabilities": {
    "real": 18.6,
    "fake": 81.4
  }
}
```

### Frequency-Only Detection

**POST** `/detect/frequency`

Use only the frequency detector.

```bash
curl -X POST http://localhost:8000/detect/frequency \
  -F "file=@path/to/image.jpg"
```

Response:
```json
{
  "method": "frequency",
  "prediction": "Real",
  "confidence": 76.3,
  "probabilities": {
    "real": 76.3,
    "fake": 23.7
  }
}
```

### Model Information

**GET** `/models/info`

Get information about loaded models.

```bash
curl http://localhost:8000/models/info
```

Response:
```json
{
  "loaded": true,
  "spatial": {
    "architecture": "resnet18",
    "total_parameters": 11689538,
    "trainable_parameters": 4456962,
    "frozen_parameters": 7232576
  },
  "frequency": {
    "method": "FFT + Random Forest",
    "num_bands": 8,
    "num_features": 36,
    "n_estimators": 100
  },
  "ensemble": {
    "spatial_weight": 0.6,
    "frequency_weight": 0.4
  }
}
```

## Interactive API Documentation

Once the server is running, you can access interactive documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing the API

### Using Python

```python
import requests

url = "http://localhost:8000/detect"
files = {"file": open("test_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Using JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/detect', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Error Handling

The API returns appropriate HTTP status codes:

- **200**: Success
- **400**: Bad request (invalid file type)
- **500**: Server error (processing failed)
- **503**: Service unavailable (models not loaded)

## CORS Configuration

The API is configured to accept requests from:
- http://localhost:3000 (React development server)
- http://localhost:5173 (Vite development server)

To add more origins, edit the `allow_origins` list in `main.py`.

## Performance

- Model loading: ~5-10 seconds on startup
- Inference time: ~1-2 seconds per image
- Concurrent requests: Supported via async handlers

## Requirements

The API requires the following files to be present:

```
AI-Image-Detector/
├── checkpoints/
│   ├── resnet18_detector_best.pth
│   ├── frequency_detector.pkl
│   └── ensemble_config.txt
├── models/
│   ├── spatial_detector.py
│   ├── frequency_detector.py
│   └── ensemble_detector.py
├── config.py
└── api/
    ├── __init__.py
    └── main.py
```

## Troubleshooting

### Models not loading

Make sure all checkpoint files exist:
- `checkpoints/resnet18_detector_best.pth`
- `checkpoints/frequency_detector.pkl`
- `checkpoints/ensemble_config.txt`

### CUDA out of memory

If running on GPU with limited memory, the API processes images one at a time to avoid memory issues.

### Port already in use

If port 8000 is busy, use a different port:
```bash
uvicorn api.main:app --reload --port 8001
```
