# Deployment Guide

## Quick Start

### 1. Install Dependencies

```bash
conda activate ai-detector
pip install fastapi uvicorn python-multipart
```

### 2. Start the Server

Navigate to your project directory:

```bash
cd C:\Users\smath\OneDrive\Documents\DKIT_Year_4\Project\AI-Image-Detector
```

Run the server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- Main API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## Development Setup

### Running with Auto-Reload

For development, use the `--reload` flag:

```bash
uvicorn api.main:app --reload
```

This automatically restarts the server when you make code changes.

### Custom Port

If port 8000 is in use:

```bash
uvicorn api.main:app --reload --port 8001
```

### Verbose Logging

For detailed logs:

```bash
uvicorn api.main:app --reload --log-level debug
```

## Production Deployment

### Multiple Workers

For production, run with multiple workers:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Number of workers = (2 × CPU cores) + 1

### Running as Background Service (Windows)

Create a batch file `start_api.bat`:

```batch
@echo off
cd C:\Users\smath\OneDrive\Documents\DKIT_Year_4\Project\AI-Image-Detector
call conda activate ai-detector
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Running as Background Service (Linux/Mac)

Create a systemd service file:

```ini
[Unit]
Description=AI Image Detector API
After=network.target

[Service]
User=your-username
WorkingDirectory=/path/to/AI-Image-Detector
Environment="PATH=/path/to/conda/envs/ai-detector/bin"
ExecStart=/path/to/conda/envs/ai-detector/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

[Install]
WantedBy=multi-user.target
```

## Testing the Deployment

### Basic Health Check

```bash
curl http://localhost:8000/
```

Should return:
```json
{
  "status": "running",
  "models_loaded": true,
  "device": "cuda"
}
```

### Test Image Detection

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@test_image.jpg"
```

### Using the Test Script

```bash
python api/test_api.py path/to/test/image.jpg
```

## Performance Optimization

### GPU Memory

If you encounter CUDA out of memory errors:

1. Reduce batch size in config.py
2. Process images sequentially (already implemented)
3. Use CPU for inference:
   ```python
   # In config.py
   DEVICE = torch.device("cpu")
   ```

### Response Time

Typical response times:
- First request: ~2-3 seconds (model warming up)
- Subsequent requests: ~1-2 seconds

To improve:
- Models are kept in memory (already implemented)
- Use GPU for faster inference (already configured)
- Enable multiple workers for concurrent requests

## Troubleshooting

### Models Not Loading

Check that all checkpoint files exist:

```bash
ls checkpoints/
# Should show:
# resnet18_detector_best.pth
# frequency_detector.pkl
# ensemble_config.txt
```

### Import Errors

Make sure you're in the correct conda environment:

```bash
conda activate ai-detector
python -c "import torch; import fastapi; print('OK')"
```

### Port Already in Use

Windows:
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

Linux/Mac:
```bash
lsof -i :8000
kill -9 <PID>
```

### CORS Errors

If the frontend can't access the API, check CORS settings in `api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    # Add your frontend URL here
)
```

## Security Considerations

### For Production

1. Remove wildcard CORS: Don't use `"*"` in `allow_origins`
2. Add authentication if needed
3. Rate limiting for public APIs
4. HTTPS/SSL certificates
5. Input validation (already implemented)

### File Upload Limits

Current limit: No explicit limit set

To add a limit, modify the endpoint:

```python
@app.post("/detect")
async def detect_image(file: UploadFile = File(..., max_length=10_000_000)):  # 10MB
    ...
```

## Monitoring

### Logs

Uvicorn logs show:
- Request/response times
- Error traces
- Server startup/shutdown

### Health Monitoring

Set up a monitoring service to check:
- GET http://localhost:8000/health
- Should return status 200
- Models should be loaded

## Integration with Frontend

The API is designed to work with a React frontend:

```javascript
// Frontend code example
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/detect', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.prediction, result.confidence);
```

## Cloud Deployment

### Docker (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.11

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t ai-detector-api .
docker run -p 8000:8000 ai-detector-api
```

### Cloud Platforms

The API can be deployed to:
- AWS EC2 / Lambda
- Google Cloud Run
- Azure App Service
- Heroku
- DigitalOcean

Note: GPU support may require specific instance types.
