# AI Image Detector - Frontend

Simple React web app for detecting AI-generated images.

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Start the API Backend

Make sure the FastAPI backend is running on port 8000:

```bash
cd C:\Users\smath\OneDrive\Documents\DKIT_Year_4\Project\AI-Image-Detector
uvicorn api.main:app --reload
```

### 3. Start the Frontend

```bash
npm start
```

The app will open at http://localhost:3000

## Features

- Upload images via file picker
- See image preview before analysis
- Get instant detection results
- View ensemble, spatial, and frequency predictions
- See confidence scores with visual indicators
- Clean, responsive design

## How It Works

1. Click "Select Image" and choose an image file
2. Click "Analyze Image" to detect if it's AI-generated
3. View results showing:
   - Overall prediction (Real or AI-Generated)
   - Confidence score with color-coded bar
   - Individual spatial and frequency analysis results
   - Ensemble weights used in detection

## API Connection

The app connects to the FastAPI backend at:
- http://localhost:8000/detect

Make sure the API is running before using the frontend.

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── App.js          # Main component with upload and detection
│   ├── App.css         # Styling
│   ├── index.js        # Entry point
│   └── index.css       # Base styles
├── package.json
└── README.md
```

## Building for Production

```bash
npm run build
```

Creates optimized production build in `build/` folder.

## Technologies Used

- React 18
- CSS3 (no external UI libraries)
- Fetch API for backend communication

## Requirements

- Node.js 14+ and npm
- API backend running on port 8000
- Modern web browser
