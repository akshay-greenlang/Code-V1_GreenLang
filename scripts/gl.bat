@echo off
REM Windows batch file to run gl command
REM This ensures gl works on Windows after pip install
REM Enhanced version with smart Python detection

REM Try to run gl directly first (if already in PATH)
where gl.exe >nul 2>&1
if %errorlevel% equ 0 (
    gl.exe %*
    exit /b %errorlevel%
)

REM Fallback to Python module
python -m greenlang.cli %*
if %errorlevel% neq 0 (
    echo ERROR: Could not run GreenLang CLI
    echo Try: python -m greenlang.cli %*
    echo Or run: gl doctor --setup-path
)