@echo off
echo =========================================
echo Docker Release Workflow Trigger via CURL
echo =========================================
echo.
echo This script uses curl to trigger the workflow.
echo You need a GitHub Personal Access Token.
echo.

set /p TOKEN="Enter your GitHub Personal Access Token: "
if "%TOKEN%"=="" (
    echo No token provided. Exiting.
    pause
    exit /b 1
)

echo.
echo Triggering workflow for version 0.2.0...
echo.

curl -X POST ^
  -H "Accept: application/vnd.github+json" ^
  -H "Authorization: Bearer %TOKEN%" ^
  -H "X-GitHub-Api-Version: 2022-11-28" ^
  https://api.github.com/repos/akshay-greenlang/Code-V1_GreenLang/actions/workflows/release-docker.yml/dispatches ^
  -d "{\"ref\":\"master\",\"inputs\":{\"version\":\"0.2.0\"}}"

echo.
echo.
echo Request sent! Check the workflow status at:
echo https://github.com/akshay-greenlang/Code-V1_GreenLang/actions
echo.
echo After completion, run:
echo scripts\verify-docker-dod.bat akshay-greenlang 0.2.0
echo.
pause