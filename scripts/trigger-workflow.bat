@echo off
echo ========================================
echo Docker Release Workflow Trigger Script
echo ========================================
echo.

REM Check if gh is installed
where gh >nul 2>&1
if errorlevel 1 goto NoGH

echo GitHub CLI found. Checking authentication...
gh auth status >nul 2>&1
if errorlevel 1 goto NeedAuth

:TriggerWorkflow
echo.
echo Triggering Docker Release Workflow for version 0.2.0...
gh workflow run release-docker.yml -f version=0.2.0

if errorlevel 1 goto Failed

echo.
echo SUCCESS! Workflow triggered.
echo.
echo You can monitor the progress at:
echo https://github.com/akshay-greenlang/Code-V1_GreenLang/actions
echo.
echo After the workflow completes (usually 10-15 minutes), run:
echo scripts\verify-docker-dod.bat akshay-greenlang 0.2.0
goto End

:NoGH
echo GitHub CLI (gh) is not installed!
echo.
echo Please install it first:
echo 1. Download from: https://github.com/cli/cli/releases/latest
echo 2. Download the Windows installer (.msi file)
echo 3. Run the installer
echo 4. Restart this command prompt after installation
echo.
echo Alternative: Use curl method with trigger-workflow-curl.bat
goto End

:NeedAuth
echo.
echo You need to authenticate with GitHub first.
echo Running authentication setup...
echo.
gh auth login
if errorlevel 1 goto Failed
goto TriggerWorkflow

:Failed
echo.
echo Failed to trigger workflow. Please check your authentication.

:End
pause