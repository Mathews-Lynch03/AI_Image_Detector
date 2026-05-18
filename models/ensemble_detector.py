"""
Ensemble Detector combining Spatial and Frequency methods
"""

import torch
import numpy as np
from pathlib import Path

from .spatial_detector import SpatialDetector
from .frequency_detector import FrequencyDetector

class EnsembleDetector:
    """Combines spatial and frequency detectors using weighted voting"""
    
    def __init__(self, spatial_weight=0.6, frequency_weight=0.4):
        """
            spatial_weight: Weight for spatial detector (default 0.6)
            frequency_weight: Weight for frequency detector (default 0.4)
        """
        self.spatial_weight = spatial_weight
        self.frequency_weight = frequency_weight
        
        self.spatial_model = None
        self.frequency_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_models(self, spatial_checkpoint, frequency_checkpoint, model_name='resnet18'):
        """Load both models"""
        
        # Load spatial model
        self.spatial_model = SpatialDetector(model_name, pretrained=False)
        checkpoint = torch.load(spatial_checkpoint, map_location=self.device)
        self.spatial_model.load_state_dict(checkpoint['model_state_dict'])
        self.spatial_model = self.spatial_model.to(self.device)
        self.spatial_model.eval()
        
        # Load frequency model
        self.frequency_model = FrequencyDetector()
        self.frequency_model.load(frequency_checkpoint)
    
    def predict_spatial(self, image_tensor):
        """Get spatial detector prediction"""
        
        with torch.no_grad():
            outputs = self.spatial_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def predict_frequency(self, image_array):
        """Get frequency detector prediction"""
        
        features = self.frequency_model.extract_batch_features([image_array])
        probabilities = self.frequency_model.predict_proba(features)
        
        return probabilities
    
    def predict(self, image_tensor, image_array):
        """
        Predict using ensemble
        
            image_tensor: Preprocessed tensor for spatial model
            image_array: Numpy array for frequency model

            tuple: (prediction, confidence, spatial_prob, frequency_prob)
        """
        
        # Get predictions from both models
        spatial_probs = self.predict_spatial(image_tensor)[0]
        frequency_probs = self.predict_frequency(image_array)[0]
        
        # Weighted ensemble
        ensemble_probs = (
            self.spatial_weight * spatial_probs +
            self.frequency_weight * frequency_probs
        )
        
        # Final prediction
        prediction = int(np.argmax(ensemble_probs))
        confidence = float(ensemble_probs[prediction])
        
        return prediction, confidence, spatial_probs, frequency_probs
    
    def predict_batch(self, image_tensors, image_arrays):
        """
        Predict on batch of images
        
            image_tensors: Batch tensor for spatial model
            image_arrays: List of numpy arrays for frequency model
        
            numpy array: Predictions for batch
        """
        
        # Spatial predictions
        with torch.no_grad():
            outputs = self.spatial_model(image_tensors)
            spatial_probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # Frequency predictions
        features = self.frequency_model.extract_batch_features(image_arrays)
        frequency_probs = self.frequency_model.predict_proba(features)
        
        # Ensemble
        ensemble_probs = (
            self.spatial_weight * spatial_probs +
            self.frequency_weight * frequency_probs
        )
        
        predictions = np.argmax(ensemble_probs, axis=1)
        
        return predictions, ensemble_probs