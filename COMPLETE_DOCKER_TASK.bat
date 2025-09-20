@echo off
echo ==========================================
echo COMPLETE DOCKER MULTI-ARCH BUILD TASK
echo ==========================================
echo.
echo This script will help you complete the Docker task.
echo.

echo OPTION 1: Install GitHub CLI and Trigger
echo ------------------------------------------
echo.
echo Step 1: Download GitHub CLI
echo Opening download page...
start https://github.com/cli/cli/releases/download/v2.63.2/gh_2.63.2_windows_amd64.msi
echo.
echo Please install GitHub CLI, then press any key to continue...
pause >nul

echo.
echo Step 2: Authenticate with GitHub
gh auth login

echo.
echo Step 3: Trigger the workflow
gh workflow run release-docker.yml -f version=0.2.0

echo.
echo Step 4: Monitor the workflow
echo Opening GitHub Actions page...
start https://github.com/akshay-greenlang/Code-V1_GreenLang/actions/workflows/release-docker.yml

echo.
echo ==========================================
echo MANUAL ALTERNATIVE (if CLI doesn't work)
echo ==========================================
echo.
echo 1. Go to: https://github.com/akshay-greenlang/Code-V1_GreenLang/actions/workflows/release-docker.yml
echo 2. Click "Run workflow" button (top right)
echo 3. Enter version: 0.2.0
echo 4. Click green "Run workflow" button
echo.
echo ==========================================
echo AFTER WORKFLOW COMPLETES (10-15 minutes)
echo ==========================================
echo.
echo Run verification:
echo scripts\verify-docker-dod.bat akshay-greenlang 0.2.0
echo.
pause