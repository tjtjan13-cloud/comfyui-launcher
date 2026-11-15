@echo off
REM ComfyUI Launcher - double-click to start ComfyUI on Windows.
REM First run: creates a virtual environment, installs dependencies,
REM clones ComfyUI, then starts the server and opens the browser.
title ComfyUI Launcher
cd /d "%~dp0"
python launcher.py %*
if errorlevel 1 (
    echo.
    echo The launcher stopped with an error. See the messages above.
    pause
)
