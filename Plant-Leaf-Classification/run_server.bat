@echo off
setlocal
set "PY=%USERPROFILE%\leaf-venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Virtual environment not found at "%PY%".
    echo Create it with: py -3.12 -m venv "%%USERPROFILE%%\leaf-venv"
    pause
    exit /b 1
)

cd /d "%~dp0"
"%PY%" app.py
