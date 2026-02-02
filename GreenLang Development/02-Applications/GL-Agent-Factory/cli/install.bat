@echo off
REM GreenLang Agent Factory CLI - Windows Installation Script

echo.
echo ============================================================
echo  GreenLang Agent Factory CLI - Installation
echo ============================================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or higher
    pause
    exit /b 1
)

echo Checking Python version...
python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.11 or higher is required
    python --version
    pause
    exit /b 1
)

echo Python version OK
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    echo Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo.

REM Install CLI
echo Installing GreenLang Agent Factory CLI...
pip install -e .
if errorlevel 1 (
    echo ERROR: Installation failed
    pause
    exit /b 1
)
echo.

REM Verify installation
echo Verifying installation...
gl --version
if errorlevel 1 (
    echo ERROR: CLI installation verification failed
    pause
    exit /b 1
)
echo.

echo ============================================================
echo  Installation Complete!
echo ============================================================
echo.
echo The 'gl' command is now available.
echo.
echo To activate the environment in future sessions:
echo   venv\Scripts\activate
echo.
echo Quick start:
echo   gl --help
echo   gl init
echo   gl agent list
echo.
echo Documentation:
echo   README.md - Complete guide
echo   QUICKSTART.md - 5-minute tutorial
echo   INSTALL.md - Installation details
echo.

pause
