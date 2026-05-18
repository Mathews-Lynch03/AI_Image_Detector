"""
Example client for AI Image Detector API

This script demonstrates how to use the API from 
"""

import requests
from pathlib import Path
import sys

class AIDetectorClient:
    """Client for interacting with the AI Image Detector API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def check_health(self):
        """Check if the API is running and models are loaded"""
        
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def detect(self, image_path):
        """
        Detect if an image is AI-generated using ensemble method
    
        """
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/detect", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Detection failed: {response.status_code} - {response.text}")
    
    def detect_spatial(self, image_path):
        """Detect using only spatial detector"""
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/detect/spatial", files=files)
        
        return response.json() if response.status_code == 200 else None
    
    def detect_frequency(self, image_path):
        """Detect using only frequency detector"""
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/detect/frequency", files=files)
        
        return response.json() if response.status_code == 200 else None
    
    def get_models_info(self):
        """Get information about loaded models"""
        
        response = requests.get(f"{self.base_url}/models/info")
        return response.json()

def main():
    
    # Create client
    client = AIDetectorClient()
    
    # Check health
    print("Checking API health...")
    health = client.check_health()
    print(f"API Status: {health['status']}")
    print(f"Models Loaded: {health['models_loaded']}")
    print(f"Device: {health['device']}")
    print()
    
    # Get models info
    print("Getting models info...")
    info = client.get_models_info()
    if info['loaded']:
        print(f"Spatial: {info['spatial']['architecture']}")
        print(f"  Parameters: {info['spatial']['total_parameters']:,}")
        print(f"Frequency: {info['frequency']['method']}")
        print(f"Ensemble: {info['ensemble']['spatial_weight']:.1f} / {info['ensemble']['frequency_weight']:.1f}")
    print()
    
    # Detect image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        if not Path(image_path).exists():
            print(f"Error: Image not found: {image_path}")
            return
        
        print(f"Analyzing image: {image_path}")
        print("-" * 50)
        
        # Ensemble detection
        result = client.detect(image_path)
        print(f"\nEnsemble Result:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  Spatial: {result['spatial_prediction']} ({result['spatial_confidence']:.1f}%)")
        print(f"  Frequency: {result['frequency_prediction']} ({result['frequency_confidence']:.1f}%)")
        
        # Individual detectors
        spatial_result = client.detect_spatial(image_path)
        print(f"\nSpatial Only:")
        print(f"  Prediction: {spatial_result['prediction']}")
        print(f"  Confidence: {spatial_result['confidence']:.1f}%")
        
        frequency_result = client.detect_frequency(image_path)
        print(f"\nFrequency Only:")
        print(f"  Prediction: {frequency_result['prediction']}")
        print(f"  Confidence: {frequency_result['confidence']:.1f}%")
    else:
        print("To analyze an image, run:")
        print("python example_client.py path/to/image.jpg")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to API")
        print("Make sure the API server is running:")
        print("uvicorn api.main:app --reload")
    except Exception as e:
        print(f"\nError: {e}")
