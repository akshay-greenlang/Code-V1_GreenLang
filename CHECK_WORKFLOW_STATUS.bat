@echo off
echo ==========================================
echo DOCKER WORKFLOW STATUS CHECK
echo ==========================================
echo.

echo Checking if the manual trigger worked...
echo.

echo Please verify the following:
echo.
echo 1. Go to: https://github.com/akshay-greenlang/Code-V1_GreenLang/actions
echo.
echo 2. Look for a NEW workflow run with:
echo    - Name: "Release Docker Images"
echo    - Trigger: "workflow_dispatch" (not "push")
echo    - Status: Running (yellow circle) or Queued
echo.
echo 3. If you see it running, wait 10-15 minutes for completion
echo.
echo 4. If you DON'T see a new run, try again:
echo    a. Go to: https://github.com/akshay-greenlang/Code-V1_GreenLang/actions/workflows/release-docker.yml
echo    b. Click "Run workflow" button
echo    c. Make sure to select:
echo       - Branch: master (not main or v0.2.0)
echo       - Version: 0.2.0
echo    d. Click green "Run workflow" button
echo.
echo ==========================================
echo TROUBLESHOOTING IF WORKFLOW WON'T START
echo ==========================================
echo.
echo Possible issues:
echo 1. Permissions - Make sure you're logged in as repo owner
echo 2. Branch - Must use "master" branch, not "main"
echo 3. Workflow file - Check if .github/workflows/release-docker.yml exists
echo.
echo To check manually with curl:
echo curl -H "Accept: application/vnd.github+json" ^
  https://api.github.com/repos/akshay-greenlang/Code-V1_GreenLang/actions/workflows
echo.
echo ==========================================
echo AFTER WORKFLOW COMPLETES SUCCESSFULLY
echo ==========================================
echo.
echo Run this to verify DoD:
echo scripts\verify-docker-dod.bat akshay-greenlang 0.2.0
echo.
echo Or check manually:
echo docker pull ghcr.io/akshay-greenlang/greenlang-runner:0.2.0
echo docker run --rm ghcr.io/akshay-greenlang/greenlang-runner:0.2.0 --version
echo.
pause