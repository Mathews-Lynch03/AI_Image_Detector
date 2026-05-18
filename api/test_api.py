"""
Simple script to test the API endpoints
"""

import requests
import sys
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    
    print("Testing health check...")
    response = requests.get(f"{API_BASE_URL}/")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Models loaded: {data['models_loaded']}")
        print(f"Device: {data['device']}")
        return True
    else:
        print(f"Health check failed: {response.status_code}")
        return False

def test_models_info():
    """Test the models info endpoint"""
    
    print("\nTesting models info...")
    response = requests.get(f"{API_BASE_URL}/models/info")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Models loaded: {data['loaded']}")
        
        if data['loaded']:
            print(f"Spatial model: {data['spatial']['architecture']}")
            print(f"  Total params: {data['spatial']['total_parameters']:,}")
            print(f"  Trainable: {data['spatial']['trainable_parameters']:,}")
            print(f"Frequency model: {data['frequency']['method']}")
            print(f"Ensemble weights: {data['ensemble']['spatial_weight']:.1f} / {data['ensemble']['frequency_weight']:.1f}")
        
        return True
    else:
        print(f"Models info failed: {response.status_code}")
        return False

def test_detection(image_path):
    """Test image detection"""
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return False
    
    print(f"\nTesting detection on {image_path}...")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_BASE_URL}/detect", files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Prediction: {data['prediction']}")
        print(f"Confidence: {data['confidence']:.1f}%")
        print(f"  Spatial: {data['spatial_prediction']} ({data['spatial_confidence']:.1f}%)")
        print(f"  Frequency: {data['frequency_prediction']} ({data['frequency_confidence']:.1f}%)")
        return True
    else:
        print(f"Detection failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def test_spatial_only(image_path):
    """Test spatial-only detection"""
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return False
    
    print(f"\nTesting spatial-only detection...")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_BASE_URL}/detect/spatial", files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Prediction: {data['prediction']}")
        print(f"Confidence: {data['confidence']:.1f}%")
        return True
    else:
        print(f"Spatial detection failed: {response.status_code}")
        return False

def test_frequency_only(image_path):
    """Test frequency-only detection"""
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return False
    
    print(f"\nTesting frequency-only detection...")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_BASE_URL}/detect/frequency", files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Prediction: {data['prediction']}")
        print(f"Confidence: {data['confidence']:.1f}%")
        return True
    else:
        print(f"Frequency detection failed: {response.status_code}")
        return False

if __name__ == "__main__":
    print("AI Image Detector API Test")
    print("=" * 50)
    
    try:
        # Test basic endpoints
        if not test_health_check():
            print("\nAPI server may not be running.")
            print("Start it with: uvicorn api.main:app --reload")
            sys.exit(1)
        
        test_models_info()
        
        # Test detection if image path provided
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            test_detection(image_path)
            test_spatial_only(image_path)
            test_frequency_only(image_path)
        else:
            print("\nTo test detection, run:")
            print("python test_api.py path/to/image.jpg")
        
        print("\nAll tests completed successfully")
        
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to API server")
        print("Make sure the server is running:")
        print("uvicorn api.main:app --reload")
        sys.exit(1)
