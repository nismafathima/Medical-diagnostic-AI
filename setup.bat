@echo off
REM MedAI Setup Script - Windows Edition
REM Usage: setup.bat
REM Run this from the medai project directory

echo.
echo ============================================
echo 4 MedAI Assistant - CPU Optimization Setup
echo ============================================
echo.

REM Check Python version
echo * Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo * Creating virtual environment...
if exist "venv" (
    echo   Virtual environment already exists. Skipping...
) else (
    python -m venv venv
    echo   Virtual environment created
)

REM Activate virtual environment
echo.
echo * Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo   Virtual environment activated

REM Upgrade pip
echo.
echo * Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)
echo   pip upgraded

REM Install requirements
echo.
echo * Installing dependencies (CPU-optimized)...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Check your internet connection and try again
    pause
    exit /b 1
)
echo   Dependencies installed

REM Create cache directory
echo.
echo * Creating cache directories...
if not exist ".streamlit" mkdir .streamlit
if not exist "%USERPROFILE%\.cache\huggingface" mkdir "%USERPROFILE%\.cache\huggingface"
echo   Cache directories created

REM Create Streamlit config
echo.
echo * Creating Streamlit config...
if not exist ".streamlit\config.toml" (
    (
        echo [client]
        echo showErrorDetails = true
        echo logger.level = "info"
        echo.
        echo [server]
        echo maxUploadSize = 200
        echo maxMessageSize = 200
        echo runOnSave = false
        echo.
        echo [browser]
        echo gatherUsageStats = false
        echo.
        echo [logger]
        echo level = "info"
    ) > ".streamlit\config.toml"
    echo   Streamlit config created
) else (
    echo   Streamlit config already exists. Skipping...
)

REM Create environment file
echo.
echo * Creating environment file...
if not exist ".env" (
    (
        echo # MedAI Environment Configuration
        echo # CPU Optimization Settings
        echo.
        echo # Set number of PyTorch threads
        echo SET TORCH_NUM_THREADS=4
        echo.
        echo # Optional: OpenAI API Key
        echo REM SET OPENAI_API_KEY=sk-...
    ) > ".env"
    echo   Environment file created
) else (
    echo   Environment file already exists. Skipping...
)

echo.
echo ============================================
echo ** Setup Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Run the application:
echo    python app.py
echo.
echo 2. Open browser to:
echo    http://localhost:7860
echo.
echo Note: First run will download models (^~1.5GB, 5-15 minutes^)
echo.
echo For more information, see:
echo   - README_OPTIMIZED.md (Getting started^)
echo   - OPTIMIZATION_GUIDE.md (Performance details^)
echo.
pause
