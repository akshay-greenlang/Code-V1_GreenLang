@echo off
REM Quick installation script for greenlang-cli with enhanced Windows support

echo GreenLang CLI Quick Installer (Enhanced)
echo =========================================
echo.

REM Check if wheels directory exists
if exist "wheels" (
    echo [FAST MODE] Installing from local wheels directory...
    pip install --no-index --find-links wheels greenlang-cli
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo Installation successful!
        goto :test_installation
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
        goto :test_installation
    )
)

REM Fallback to regular installation
echo Installing from PyPI...
pip install greenlang-cli
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Installation successful!
    goto :test_installation
) else (
    echo.
    echo Installation failed!
    goto :error_help
)

:test_installation
echo.
echo Testing gl command...
gl --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] gl command is working!
    gl --version
) else (
    echo [WARNING] gl command not found in PATH
    echo.
    echo Trying to fix PATH issues...
    python -c "from greenlang.utils.windows_path import setup_windows_path; success, msg = setup_windows_path(); print(f'Setup result: {msg}')" 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo PATH setup completed. Please restart your command prompt.
    ) else (
        echo.
        echo Manual setup required:
        echo 1. Run: gl doctor --setup-path
        echo 2. Or use: python -m greenlang.cli
        echo 3. Or restart your command prompt
    )
)

echo.
echo Run 'gl doctor' to verify your installation
goto :end

:error_help
echo.
echo Installation failed. Troubleshooting:
echo 1. Make sure Python 3.10+ is installed
echo 2. Check your internet connection
echo 3. Try: python -m pip install --user greenlang-cli
echo 4. For help, visit: https://greenlang.io/docs/installation

:end
pause