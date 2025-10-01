@echo off
REM GreenLang Windows Wrapper
REM Auto-detects Python installation and runs gl command
REM Works with system, user, conda, and virtual environment installations

setlocal enabledelayedexpansion

REM Try to run gl directly first (if already in PATH)
where gl.exe >nul 2>&1
if %errorlevel% equ 0 (
    gl.exe %*
    exit /b %errorlevel%
)

REM If gl.exe not found, try various Python installation paths
set "PYTHON_CMD="
set "GL_PATH="

REM Check if python is available and try running gl module directly
python --version >nul 2>&1
if %errorlevel% equ 0 (
    python -c "import greenlang.cli.main; greenlang.cli.main.main()" %* 2>nul
    if %errorlevel% equ 0 (
        exit /b 0
    )
)

REM Try to find gl.exe in common Python installation paths
for %%P in (
    "%USERPROFILE%\AppData\Roaming\Python\Python313\Scripts"
    "%USERPROFILE%\AppData\Roaming\Python\Python312\Scripts"
    "%USERPROFILE%\AppData\Roaming\Python\Python311\Scripts"
    "%USERPROFILE%\AppData\Roaming\Python\Python310\Scripts"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python313\Scripts"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python312\Scripts"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python311\Scripts"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python310\Scripts"
    "C:\Python313\Scripts"
    "C:\Python312\Scripts"
    "C:\Python311\Scripts"
    "C:\Python310\Scripts"
    "C:\ProgramData\Anaconda3\Scripts"
    "C:\ProgramData\Miniconda3\Scripts"
    "%USERPROFILE%\Anaconda3\Scripts"
    "%USERPROFILE%\Miniconda3\Scripts"
    "%LOCALAPPDATA%\Programs\Python\Launcher"
) do (
    if exist "%%P\gl.exe" (
        "%%P\gl.exe" %*
        exit /b %errorlevel%
    )
)

REM Try to find Python and run the module directly
for %%P in (
    "%USERPROFILE%\AppData\Roaming\Python\Python313"
    "%USERPROFILE%\AppData\Roaming\Python\Python312"
    "%USERPROFILE%\AppData\Roaming\Python\Python311"
    "%USERPROFILE%\AppData\Roaming\Python\Python310"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python313"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python312"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python311"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python310"
    "C:\Python313"
    "C:\Python312"
    "C:\Python311"
    "C:\Python310"
    "C:\ProgramData\Anaconda3"
    "C:\ProgramData\Miniconda3"
    "%USERPROFILE%\Anaconda3"
    "%USERPROFILE%\Miniconda3"
) do (
    if exist "%%P\python.exe" (
        "%%P\python.exe" -c "import greenlang.cli.main; greenlang.cli.main.main()" %* 2>nul
        if %errorlevel% equ 0 (
            exit /b 0
        )
    )
)

REM If nothing worked, provide helpful error message
echo ERROR: Could not find GreenLang installation.
echo.
echo Troubleshooting:
echo 1. Make sure GreenLang is installed: pip install greenlang-cli
echo 2. Add Python Scripts directory to your PATH
echo 3. Run: gl doctor --setup-path
echo 4. Or use: python -m greenlang.cli %*
echo.
echo For help, visit: https://greenlang.io/docs/installation
exit /b 1