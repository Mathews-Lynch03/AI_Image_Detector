@echo off
echo Starting AI Image Detector API...
echo.

cd /d C:\Users\smath\OneDrive\Documents\DKIT_Year_4\Project\AI-Image-Detector

echo Activating conda environment...
call conda activate ai-detector

echo.
echo Starting server on http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
