@echo off
echo Installing Tesseract OCR...

:: Check if chocolatey is installed
choco --version >nul 2>&1
if errorlevel 1 (
    echo Chocolatey not found. Please install manually:
    echo 1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
    echo 2. Install and add to PATH
    echo 3. Restart terminal
) else (
    echo Installing Tesseract via Chocolatey...
    choco install tesseract -y
)

:: Check installation
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo Installation failed. Please install manually.
) else (
    echo Tesseract installed successfully!
    tesseract --version
)