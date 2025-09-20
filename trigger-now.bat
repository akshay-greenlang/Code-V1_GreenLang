@echo off
echo Checking GitHub CLI authentication...
gh auth status

if errorlevel 1 (
    echo Need to authenticate first...
    gh auth login
)

echo.
echo Triggering Docker Release Workflow v0.2.0...
gh workflow run release-docker.yml -f version=0.2.0

if errorlevel 1 (
    echo Failed to trigger workflow. Trying again...
    gh workflow run .github/workflows/release-docker.yml -f version=0.2.0
)

echo.
echo Checking workflow status...
timeout /t 5 >nul
gh run list --workflow=release-docker.yml --limit=1

echo.
echo Workflow triggered! Monitor at:
echo https://github.com/akshay-greenlang/Code-V1_GreenLang/actions
echo.
pause