@echo off
echo Starting AI Image Detector
echo.

echo Starting Backend API...
cd /d C:\Users\smath\OneDrive\Documents\DKIT_Year_4\AI-Image-Detector
start cmd /k "conda activate ai-detector && uvicorn api.main:app --reload"

echo Waiting for API to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend...
cd frontend
start cmd /k "npm start"

echo.
echo Both servers started!
echo.
echo Backend API: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit this window...
pause > nul
