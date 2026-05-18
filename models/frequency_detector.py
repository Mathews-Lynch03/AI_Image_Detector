"""
Frequency-Domain Detector for AI-Generated Images
Uses FFT to analyze frequency patterns in images
"""

import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path

class FrequencyFeatureExtractor:
    """Extract frequency features using FFT"""
    
    def __init__(self, num_bands=8):
        self.num_bands = num_bands
    
    def extract_features(self, image):
        """Extract frequency features from image"""
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shifted = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shifted)
        magnitude_log = np.log1p(magnitude)
        
        # Extract band features
        features = self._extract_band_features(magnitude_log)
        return features
    
    def _extract_band_features(self, magnitude):
        """Extract statistics from frequency bands"""
        
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Create distance map
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Define bands
        max_dist = min(center_x, center_y)
        band_edges = np.linspace(0, max_dist, self.num_bands + 1)
        
        features = []
        
        # Extract stats from each band
        for i in range(self.num_bands):
            inner = band_edges[i]
            outer = band_edges[i + 1]
            mask = (distance >= inner) & (distance < outer)
            band_values = magnitude[mask]
            
            if len(band_values) > 0:
                features.extend([
                    np.mean(band_values),
                    np.std(band_values),
                    np.max(band_values),
                    np.percentile(band_values, 75)
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        # Global stats
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude),
            np.sum(magnitude > np.percentile(magnitude, 90))
        ])
        
        return np.array(features)


class FrequencyDetector:
    """Frequency-based detector using Random Forest"""
    
    def __init__(self, num_bands=8, n_estimators=100):
        self.extractor = FrequencyFeatureExtractor(num_bands=num_bands)
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
    
    def extract_batch_features(self, images):
        """Extract features from multiple images"""
        
        features = []
        for img in images:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            feat = self.extractor.extract_features(img)
            features.append(feat)
        
        return np.array(features)
    
    def train(self, train_features, train_labels):
        """Train classifier"""
        self.classifier.fit(train_features, train_labels)
        self.is_trained = True
    
    def predict(self, features):
        """Predict labels"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.classifier.predict(features)
    
    def predict_proba(self, features):
        """Predict probabilities"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.classifier.predict_proba(features)
    
    def save(self, path):
        """Save model"""
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'num_bands': self.extractor.num_bands,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, path):
        """Load model"""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.classifier = data['classifier']
        self.extractor = FrequencyFeatureExtractor(num_bands=data['num_bands'])
        self.is_trained = data['is_trained']