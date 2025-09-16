@echo off
echo ============================================
echo GitHub Branch Protection Setup
echo ============================================
echo.

REM Check if GITHUB_TOKEN is set
if "%GITHUB_TOKEN%"=="" (
    echo ERROR: GITHUB_TOKEN environment variable is not set!
    echo.
    echo Please follow these steps:
    echo.
    echo 1. Create a Personal Access Token at:
    echo    https://github.com/settings/tokens/new
    echo    - Name: "GreenLang Branch Protection"
    echo    - Scope: [x] repo - Full control of private repositories
    echo    - Click "Generate token"
    echo    - COPY THE TOKEN NOW! You won't see it again
    echo.
    echo 2. Set the token in this terminal:
    echo    set GITHUB_TOKEN=ghp_YourTokenHere
    echo.
    echo 3. Edit scripts\setup_branch_protection.py:
    echo    - Change REPO_OWNER to your GitHub username
    echo.
    echo 4. Run this script again:
    echo    scripts\setup_branch_protection.bat
    echo.
    pause
    exit /b 1
)

echo Token found. Running protection setup...
echo.

python scripts\setup_branch_protection.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: Branch protection configured!
) else (
    echo.
    echo FAILED: See error messages above
)

echo.
pause