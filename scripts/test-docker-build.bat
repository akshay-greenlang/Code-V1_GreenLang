@echo off
setlocal enabledelayedexpansion

REM Docker Multi-Arch Build Test Script for Windows
echo ===============================================
echo GreenLang Docker Multi-Arch Build Test
echo ===============================================

REM Configuration
set GL_VERSION=0.2.0
set REGISTRY=ghcr.io/akshay-greenlang
set PLATFORMS=linux/amd64

echo Version: %GL_VERSION%
echo Registry: %REGISTRY%
echo Platforms: %PLATFORMS%
echo.

REM Check prerequisites
echo Checking prerequisites...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not in PATH
    exit /b 1
)

docker buildx version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker buildx is not available
    exit /b 1
)

echo Prerequisites check passed
echo.

REM Setup buildx builder
echo Setting up Docker buildx builder...
set BUILDER_NAME=greenlang-builder

docker buildx inspect %BUILDER_NAME% >nul 2>&1
if %errorlevel% neq 0 (
    echo Creating new builder: %BUILDER_NAME%
    docker buildx create --name %BUILDER_NAME% --use --platform=%PLATFORMS%
    docker buildx inspect --bootstrap
) else (
    echo Using existing builder: %BUILDER_NAME%
    docker buildx use %BUILDER_NAME%
)

echo Builder '%BUILDER_NAME%' is ready
echo.

REM Get build metadata
for /f "tokens=*" %%i in ('powershell -command "Get-Date -Format yyyy-MM-ddTHH:mm:ssZ"') do set BUILD_DATE=%%i
for /f "tokens=*" %%i in ('git rev-parse --short HEAD 2^>nul') do set VCS_REF=%%i
if "%VCS_REF%"=="" set VCS_REF=unknown

echo Build metadata:
echo   Build Date: %BUILD_DATE%
echo   VCS Ref: %VCS_REF%
echo.

REM Test runner image build
echo Building runner image (local test)...
docker buildx build ^
    --platform=%PLATFORMS% ^
    --file=Dockerfile.runner ^
    --build-arg GL_VERSION=%GL_VERSION% ^
    --build-arg BUILD_DATE=%BUILD_DATE% ^
    --build-arg VCS_REF=%VCS_REF% ^
    --tag %REGISTRY%/greenlang-runner:test ^
    --load ^
    .

if %errorlevel% neq 0 (
    echo ERROR: Runner image build failed
    exit /b 1
)

echo Runner image built successfully
echo.

REM Test full image build
echo Building full image (local test)...
docker buildx build ^
    --platform=%PLATFORMS% ^
    --file=Dockerfile.full ^
    --build-arg GL_VERSION=%GL_VERSION% ^
    --build-arg BUILD_DATE=%BUILD_DATE% ^
    --build-arg VCS_REF=%VCS_REF% ^
    --tag %REGISTRY%/greenlang-full:test ^
    --load ^
    .

if %errorlevel% neq 0 (
    echo ERROR: Full image build failed
    exit /b 1
)

echo Full image built successfully
echo.

REM Test runner image
echo Testing runner image...

docker run --rm %REGISTRY%/greenlang-runner:test --version
if %errorlevel% neq 0 (
    echo ERROR: Runner image test failed: gl --version
    exit /b 1
)

docker run --rm %REGISTRY%/greenlang-runner:test --help >nul
if %errorlevel% neq 0 (
    echo ERROR: Runner image test failed: gl --help
    exit /b 1
)

REM Verify user ID
for /f "tokens=*" %%i in ('docker run --rm %REGISTRY%/greenlang-runner:test id -u') do set USER_ID=%%i
if not "%USER_ID%"=="10001" (
    echo ERROR: Runner image has wrong user ID: %USER_ID% ^(expected 10001^)
    exit /b 1
)

echo Runner image tests passed
echo.

REM Test full image
echo Testing full image...

docker run --rm %REGISTRY%/greenlang-full:test gl --version
if %errorlevel% neq 0 (
    echo ERROR: Full image test failed: gl --version
    exit /b 1
)

docker run --rm %REGISTRY%/greenlang-full:test python -c "import pytest, mypy, black; print('Dev tools OK')"
if %errorlevel% neq 0 (
    echo ERROR: Full image test failed: dev tools import
    exit /b 1
)

REM Verify user ID
for /f "tokens=*" %%i in ('docker run --rm %REGISTRY%/greenlang-full:test id -u') do set USER_ID=%%i
if not "%USER_ID%"=="10001" (
    echo ERROR: Full image has wrong user ID: %USER_ID% ^(expected 10001^)
    exit /b 1
)

echo Full image tests passed
echo.

REM Image size report
echo Image size report:
for /f "tokens=*" %%i in ('docker image inspect %REGISTRY%/greenlang-runner:test --format="{{.Size}}"') do set /a RUNNER_SIZE=%%i/1024/1024
for /f "tokens=*" %%i in ('docker image inspect %REGISTRY%/greenlang-full:test --format="{{.Size}}"') do set /a FULL_SIZE=%%i/1024/1024

echo   Runner image: %RUNNER_SIZE% MB
echo   Full image: %FULL_SIZE% MB
echo.

REM Test docker-compose
echo Testing docker-compose setup...

docker-compose build
if %errorlevel% neq 0 (
    echo ERROR: docker-compose build failed
    exit /b 1
)

echo docker-compose build successful
echo.

REM Cleanup test images
echo Cleaning up test images...
docker rmi %REGISTRY%/greenlang-runner:test >nul 2>&1
docker rmi %REGISTRY%/greenlang-full:test >nul 2>&1

echo Cleanup complete
echo.

echo ===============================================
echo All Docker build tests passed successfully!
echo ===============================================
echo.
echo Next steps:
echo   1. Tag and push images: git tag v%GL_VERSION% ^&^& git push --tags
echo   2. Trigger CI workflow: gh workflow run release-docker.yml
echo   3. Verify signatures: cosign verify %REGISTRY%/greenlang-runner:%GL_VERSION%

endlocal