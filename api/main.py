"""
FastAPI Backend for AI Image Detector

This API provides endpoints for uploading images and receiving
classification results from the ensemble detector.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import torch
import numpy as np
from PIL import Image
import io

from config import *
from models.spatial_detector import SpatialDetector
from models.frequency_detector import FrequencyDetector
from models.ensemble_detector import EnsembleDetector

app = FastAPI(
    title="AI Image Detector API",
    description="Detect AI-generated images using ensemble of spatial and frequency analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionResult(BaseModel):
    prediction: str
    confidence: float
    spatial_prediction: str
    spatial_confidence: float
    frequency_prediction: str
    frequency_confidence: float
    ensemble_weights: dict

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str

models = {
    "spatial": None,
    "frequency": None,
    "ensemble": None,
    "loaded": False
}

def load_models():
    """Load all trained models into memory"""
    
    print("Loading models...")
    
    try:
        spatial_model = SpatialDetector(MODEL_NAME, pretrained=False)
        spatial_checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
        spatial_model.load_state_dict(spatial_checkpoint['model_state_dict'])
        spatial_model = spatial_model.to(DEVICE)
        spatial_model.eval()
        models["spatial"] = spatial_model
        print(f"Spatial model loaded from {MODEL_CHECKPOINT}")
        
        frequency_model = FrequencyDetector()
        frequency_checkpoint = CHECKPOINTS_DIR / "frequency_detector.pkl"
        frequency_model.load(frequency_checkpoint)
        models["frequency"] = frequency_model
        print(f"Frequency model loaded from {frequency_checkpoint}")
        
        ensemble_model = EnsembleDetector()
        ensemble_model.load_models(
            MODEL_CHECKPOINT,
            frequency_checkpoint,
            MODEL_NAME
        )
        
        config_path = CHECKPOINTS_DIR / "ensemble_config.txt"
        if config_path.exists():
            with open(config_path, 'r') as f:
                for line in f:
                    if line.startswith("Spatial weight:"):
                        ensemble_model.spatial_weight = float(line.split(":")[1].strip())
                    elif line.startswith("Frequency weight:"):
                        ensemble_model.frequency_weight = float(line.split(":")[1].strip())
            print(f"Ensemble weights: {ensemble_model.spatial_weight:.1f} / {ensemble_model.frequency_weight:.1f}")
        
        models["ensemble"] = ensemble_model
        models["loaded"] = True
        
        print("All models loaded successfully")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

def preprocess_image_for_spatial(image: Image.Image):
    """Preprocess image for spatial detector"""
    
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return transform(image).unsqueeze(0)

def preprocess_image_for_frequency(image: Image.Image):
    """Preprocess image for frequency detector"""
    
    image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(image_resized)

@app.on_event("startup")
async def startup_event():
    """Load models when server starts"""
    load_models()

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    
    return HealthResponse(
        status="running",
        models_loaded=models["loaded"],
        device=str(DEVICE)
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    
    return HealthResponse(
        status="healthy" if models["loaded"] else "models not loaded",
        models_loaded=models["loaded"],
        device=str(DEVICE)
    )

@app.post("/detect", response_model=DetectionResult)
async def detect_image(file: UploadFile = File(...)):
    """
    Detect if an uploaded image is AI-generated
    
    Args:
        file: Image file upload
    
    Returns:
        DetectionResult with predictions from all detectors
    """
    
    if not models["loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        spatial_tensor = preprocess_image_for_spatial(image).to(DEVICE)
        frequency_array = preprocess_image_for_frequency(image)
        
        prediction, confidence, spatial_probs, frequency_probs = models["ensemble"].predict(
            spatial_tensor,
            frequency_array
        )
        
        prediction_label = "AI-Generated" if prediction == 1 else "Real"
        spatial_pred = "AI-Generated" if np.argmax(spatial_probs) == 1 else "Real"
        frequency_pred = "AI-Generated" if np.argmax(frequency_probs) == 1 else "Real"
        
        spatial_conf = float(spatial_probs[np.argmax(spatial_probs)]) * 100
        frequency_conf = float(frequency_probs[np.argmax(frequency_probs)]) * 100
        
        return DetectionResult(
            prediction=prediction_label,
            confidence=confidence * 100,
            spatial_prediction=spatial_pred,
            spatial_confidence=spatial_conf,
            frequency_prediction=frequency_pred,
            frequency_confidence=frequency_conf,
            ensemble_weights={
                "spatial": models["ensemble"].spatial_weight,
                "frequency": models["ensemble"].frequency_weight
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/detect/spatial")
async def detect_spatial_only(file: UploadFile = File(...)):
    """Detect using only spatial detector"""
    
    if not models["loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        spatial_tensor = preprocess_image_for_spatial(image).to(DEVICE)
        
        with torch.no_grad():
            outputs = models["spatial"](spatial_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(outputs, dim=1).item()
        
        prediction_label = "AI-Generated" if prediction == 1 else "Real"
        confidence = float(probabilities[prediction]) * 100
        
        return {
            "method": "spatial",
            "prediction": prediction_label,
            "confidence": confidence,
            "probabilities": {
                "real": float(probabilities[0]) * 100,
                "fake": float(probabilities[1]) * 100
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/frequency")
async def detect_frequency_only(file: UploadFile = File(...)):
    """Detect using only frequency detector"""
    
    if not models["loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        frequency_array = preprocess_image_for_frequency(image)
        
        features = models["frequency"].extract_batch_features([frequency_array])
        prediction = models["frequency"].predict(features)[0]
        probabilities = models["frequency"].predict_proba(features)[0]
        
        prediction = int(prediction)
        prediction_label = "AI-Generated" if prediction == 1 else "Real"
        confidence = float(probabilities[prediction]) * 100
        
        return {
            "method": "frequency",
            "prediction": prediction_label,
            "confidence": confidence,
            "probabilities": {
                "real": float(probabilities[0]) * 100,
                "fake": float(probabilities[1]) * 100
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def models_info():
    """Get information about loaded models"""
    
    if not models["loaded"]:
        return {"loaded": False, "message": "Models not loaded"}
    
    total_params, trainable_params = models["spatial"].count_parameters()
    
    return {
        "loaded": True,
        "spatial": {
            "architecture": MODEL_NAME,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params
        },
        "frequency": {
            "method": "FFT + Random Forest",
            "num_bands": 8,
            "num_features": 36,
            "n_estimators": 100
        },
        "ensemble": {
            "spatial_weight": models["ensemble"].spatial_weight,
            "frequency_weight": models["ensemble"].frequency_weight
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
