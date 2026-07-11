@echo off
echo.
echo  ████████╗██╗  ██╗██████╗ ██╗███████╗████████╗
echo     ██╔══╝██║  ██║██╔══██╗██║██╔════╝╚══██╔══╝
echo     ██║   ███████║██████╔╝██║█████╗     ██║
echo     ██║   ██╔══██║██╔══██╗██║██╔══╝     ██║
echo     ██║   ██║  ██║██║  ██║██║██║        ██║
echo     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝
echo.
echo  Token Heuristic Routing with Intelligent Fallback Trees
echo  AMD Developer Hackathon ACT II -- Track 1
echo.

REM Check for FW_API_KEY
if "%FW_API_KEY%"=="" (
    echo  WARNING: FW_API_KEY not set. Tier 2 - Fireworks - will be unavailable.
    echo  Set it with:  set FW_API_KEY=your-key-here
    echo.
)

REM Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo  Virtual environment activated.
) else (
    echo  No venv found -- using system Python.
)

echo  Starting THRIFT server at http://localhost:8000
echo  Dashboard: http://localhost:8000
echo  API docs:  http://localhost:8000/docs
echo  Press Ctrl+C to stop.
echo.

uvicorn app:app --host 0.0.0.0 --port 8000 --reload