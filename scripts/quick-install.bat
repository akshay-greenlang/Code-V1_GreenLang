@echo off
REM Quick installation script for greenlang-cli without re-downloading (Windows)

echo GreenLang CLI Quick Installer
echo =============================
echo.

REM Check if wheels directory exists
if exist "wheels" (
    echo [FAST MODE] Installing from local wheels directory...
    pip install --no-index --find-links wheels greenlang-cli
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo Installation successful!
        gl --version
        exit /b 0
    ) else (
        echo Local installation failed, falling back to online mode...
    )
)

REM Check if requirements-lock.txt exists
if exist "requirements-lock.txt" (
    echo Installing from locked requirements...
    pip install -r requirements-lock.txt
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo Installation successful!
        gl --version
        exit /b 0
    )
)

REM Fallback to regular installation
echo Installing from PyPI...
pip install greenlang-cli
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Installation successful!
    gl --version
)

pause