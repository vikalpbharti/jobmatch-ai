@echo off
REM Double-click to start JobMatch AI, then open http://127.0.0.1:8000
cd /d "%~dp0"
start "" http://127.0.0.1:8000
.venv\Scripts\python.exe -m uvicorn api:app --port 8000
