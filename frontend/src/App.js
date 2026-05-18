import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    } else {
      setError('Please select a valid image file');
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Detection failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Failed to analyze image. Make sure the API server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return '#10b981';
    if (confidence >= 60) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="App">
      <header className="header">
        <h1>AI Image Detector</h1>
        <p>Upload an image to check if it's AI-generated</p>
      </header>

      <main className="main">
        <div className="upload-section">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            id="file-input"
            className="file-input"
          />
          <label htmlFor="file-input" className="file-label">
            {selectedFile ? 'Change Image' : 'Select Image'}
          </label>

          {preview && (
            <div className="preview-container">
              <img src={preview} alt="Preview" className="preview-image" />
            </div>
          )}

          {error && <div className="error">{error}</div>}

          <div className="button-group">
            {selectedFile && !result && (
              <button 
                onClick={handleAnalyze} 
                disabled={loading}
                className="button button-primary"
              >
                {loading ? 'Analyzing...' : 'Analyze Image'}
              </button>
            )}

            {(result || selectedFile) && (
              <button 
                onClick={handleReset}
                className="button button-secondary"
              >
                Reset
              </button>
            )}
          </div>
        </div>

        {result && (
          <div className="results-section">
            <div className="result-card main-result">
              <h2>Result</h2>
              <div 
                className="prediction"
                style={{ color: result.prediction === 'Real' ? '#10b981' : '#ef4444' }}
              >
                {result.prediction}
              </div>
              <div className="confidence">
                <div className="confidence-label">Confidence</div>
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill"
                    style={{ 
                      width: `${result.confidence}%`,
                      backgroundColor: getConfidenceColor(result.confidence)
                    }}
                  >
                    <span className="confidence-text">{result.confidence.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="details-grid">
              <div className="result-card">
                <h3>Spatial Analysis</h3>
                <div className="method-result">
                  <div className="method-prediction">{result.spatial_prediction}</div>
                  <div className="method-confidence">{result.spatial_confidence.toFixed(1)}%</div>
                </div>
                <div className="method-description">CNN-based pixel analysis</div>
              </div>

              <div className="result-card">
                <h3>Frequency Analysis</h3>
                <div className="method-result">
                  <div className="method-prediction">{result.frequency_prediction}</div>
                  <div className="method-confidence">{result.frequency_confidence.toFixed(1)}%</div>
                </div>
                <div className="method-description">FFT-based frequency analysis</div>
              </div>
            </div>

            <div className="ensemble-info">
              <p>
                Ensemble weights: Spatial {(result.ensemble_weights.spatial * 100).toFixed(0)}% / 
                Frequency {(result.ensemble_weights.frequency * 100).toFixed(0)}%
              </p>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>BSc Cloud Computing Final Year Project - Seb Mathews-Lynch</p>
      </footer>
    </div>
  );
}

export default App;
