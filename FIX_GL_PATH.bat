@echo off
REM This batch file will add the Python Scripts directory to your PATH for this session
REM and provide instructions for permanent setup

echo ============================================
echo GreenLang CLI (gl) Installation Fix
echo ============================================
echo.

REM Check if gl.exe exists
if exist "C:\Users\rshar\AppData\Roaming\Python\Python313\Scripts\gl.exe" (
    echo [FOUND] gl.exe is installed at:
    echo         C:\Users\rshar\AppData\Roaming\Python\Python313\Scripts\gl.exe
    echo.
) else (
    echo [ERROR] gl.exe not found. Please run:
    echo         pip install greenlang-cli==0.3.0
    echo.
    exit /b 1
)

echo Testing gl command...
"C:\Users\rshar\AppData\Roaming\Python\Python313\Scripts\gl.exe" --version
echo.

echo ============================================
echo TEMPORARY FIX (for this session only):
echo ============================================
echo Adding Python Scripts to PATH for this session...
set PATH=C:\Users\rshar\AppData\Roaming\Python\Python313\Scripts;%PATH%
echo Done! You can now use 'gl' command in THIS window.
echo.

echo ============================================
echo PERMANENT FIX (recommended):
echo ============================================
echo To make 'gl' work permanently in all command prompts:
echo.
echo Option 1: Run this PowerShell command as Administrator:
echo    [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Users\rshar\AppData\Roaming\Python\Python313\Scripts", [EnvironmentVariableTarget]::User)
echo.
echo Option 2: Manual steps:
echo    1. Press Win + X, select "System"
echo    2. Click "Advanced system settings"
echo    3. Click "Environment Variables"
echo    4. Under "User variables", select "Path" and click "Edit"
echo    5. Click "New" and add: C:\Users\rshar\AppData\Roaming\Python\Python313\Scripts
echo    6. Click "OK" on all windows
echo    7. Close and reopen any command prompts
echo.
echo ============================================
echo IMMEDIATE WORKAROUND:
echo ============================================
echo You can always run gl using the full path:
echo    "C:\Users\rshar\AppData\Roaming\Python\Python313\Scripts\gl.exe" --version
echo.
echo Or create an alias by adding this line to a batch file in your PATH:
echo    @"C:\Users\rshar\AppData\Roaming\Python\Python313\Scripts\gl.exe" %%*
echo.

REM Test if gl works now
echo Testing if 'gl' command works now...
gl --version 2>nul
if %errorlevel% equ 0 (
    echo.
    echo SUCCESS! The 'gl' command is now working in this session.
    echo.
) else (
    echo.
    echo NOTE: 'gl' command will work after you follow the permanent fix steps above.
    echo For now, you can use the full path shown above.
)

echo.
echo Press any key to exit...
pause >nul