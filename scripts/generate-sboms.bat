@echo off
REM Generate comprehensive SBOMs for GreenLang packages and Docker images
REM This script produces CycloneDX (primary) and SPDX (secondary) format SBOMs

setlocal enabledelayedexpansion

REM Configuration
set VERSION=%1
if "%VERSION%"=="" set VERSION=0.2.0
set ARTIFACTS_DIR=artifacts\sbom
set SYFT_VERSION=v1.0.0
set COSIGN_VERSION=v2.2.4

REM Colors (Windows 10+)
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set NC=[0m

REM Helper functions
goto :main

:log_info
echo %BLUE%[INFO]%NC% %~1
exit /b

:log_success
echo %GREEN%[SUCCESS]%NC% %~1
exit /b

:log_error
echo %RED%[ERROR]%NC% %~1
exit /b

:log_warn
echo %YELLOW%[WARNING]%NC% %~1
exit /b

:install_syft
call :log_info "Checking for Syft installation..."
where syft >nul 2>&1
if %errorlevel%==0 (
    call :log_info "Syft is already installed"
    syft version
) else (
    call :log_info "Installing Syft %SYFT_VERSION%..."

    REM Download Syft for Windows
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/anchore/syft/releases/download/%SYFT_VERSION%/syft_Windows_x86_64.zip' -OutFile 'syft.zip'}"

    REM Extract Syft
    powershell -Command "Expand-Archive -Path syft.zip -DestinationPath syft_temp -Force"

    REM Move to a directory in PATH or current directory
    move syft_temp\syft.exe . >nul
    del syft.zip
    rmdir /s /q syft_temp

    call :log_success "Syft installed successfully"
)
exit /b

:install_cosign
call :log_info "Checking for Cosign installation..."
where cosign >nul 2>&1
if %errorlevel%==0 (
    call :log_info "Cosign is already installed"
    cosign version
) else (
    call :log_info "Installing Cosign %COSIGN_VERSION%..."

    REM Download Cosign for Windows
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/sigstore/cosign/releases/download/%COSIGN_VERSION%/cosign-windows-amd64.exe' -OutFile 'cosign.exe'}"

    call :log_success "Cosign installed successfully"
)
exit /b

:generate_wheel_sbom
set wheel_file=%~1
if not exist "%wheel_file%" (
    call :log_error "Wheel file not found: %wheel_file%"
    exit /b 1
)

for %%F in ("%wheel_file%") do set wheel_name=%%~nxF
for /f "tokens=1,2 delims=-" %%a in ("%wheel_name%") do (
    set pkg_name=%%a
    set pkg_version=%%b
)

call :log_info "Generating SBOM for wheel: %wheel_name%"

REM CycloneDX format (PRIMARY)
syft "%wheel_file%" ^
    -o cyclonedx-json="%ARTIFACTS_DIR%\sbom-%pkg_name%-%pkg_version%-wheel.cdx.json" ^
    --name "%pkg_name%-wheel" ^
    --source-name "%wheel_name%" ^
    --source-version "%pkg_version%"

REM SPDX format (SECONDARY)
syft "%wheel_file%" ^
    -o spdx-json="%ARTIFACTS_DIR%\sbom-%pkg_name%-%pkg_version%-wheel.spdx.json" ^
    --name "%pkg_name%-wheel" ^
    --source-name "%wheel_name%" ^
    --source-version "%pkg_version%"

call :log_success "Generated wheel SBOMs (CycloneDX + SPDX)"
exit /b

:generate_sdist_sbom
set sdist_file=%~1
if not exist "%sdist_file%" (
    call :log_error "Sdist file not found: %sdist_file%"
    exit /b 1
)

for %%F in ("%sdist_file%") do set sdist_name=%%~nxF
for /f "tokens=1,2 delims=-" %%a in ("%sdist_name%") do (
    set pkg_name=%%a
    set pkg_version=%%b
)
REM Remove .tar.gz extension from version
set pkg_version=%pkg_version:.tar=%

call :log_info "Generating SBOM for sdist: %sdist_name%"

REM CycloneDX format (REQUIRED)
syft "%sdist_file%" ^
    -o cyclonedx-json="%ARTIFACTS_DIR%\sbom-%pkg_name%-%pkg_version%-sdist.cdx.json" ^
    --name "%pkg_name%-sdist" ^
    --source-name "%sdist_name%" ^
    --source-version "%pkg_version%"

REM SPDX format (OPTIONAL)
syft "%sdist_file%" ^
    -o spdx-json="%ARTIFACTS_DIR%\sbom-%pkg_name%-%pkg_version%-sdist.spdx.json" ^
    --name "%pkg_name%-sdist" ^
    --source-name "%sdist_name%" ^
    --source-version "%pkg_version%"

call :log_success "Generated sdist SBOMs (CycloneDX + SPDX)"
exit /b

:generate_docker_sbom
set image_name=%~1
set image_tag=%~2
set image_type=%~3

call :log_info "Generating SBOM for Docker image: %image_name%:%image_tag%"

REM Sanitize image name for filename
set safe_name=%image_name:/=-%
set safe_name=%safe_name::=-%

REM CycloneDX format (PRIMARY)
syft "docker:%image_name%:%image_tag%" ^
    -o cyclonedx-json="%ARTIFACTS_DIR%\sbom-image-%safe_name%-%image_tag%.cdx.json" ^
    --name "greenlang-%image_type%" ^
    --source-name "%image_name%" ^
    --source-version "%image_tag%"

REM SPDX format (SECONDARY)
syft "docker:%image_name%:%image_tag%" ^
    -o spdx-json="%ARTIFACTS_DIR%\sbom-image-%safe_name%-%image_tag%.spdx.json" ^
    --name "greenlang-%image_type%" ^
    --source-name "%image_name%" ^
    --source-version "%image_tag%"

call :log_success "Generated Docker image SBOMs (CycloneDX + SPDX)"
exit /b

:display_summary
call :log_info "SBOM Generation Summary"
echo ========================
echo.

if exist "%ARTIFACTS_DIR%" (
    echo Generated SBOMs:
    for %%F in ("%ARTIFACTS_DIR%\*.json") do (
        for %%A in ("%%F") do set size=%%~zA
        set /a size_kb=!size!/1024
        echo   - %%~nxF (!size_kb! KB^)
    )
    echo.

    REM Count total SBOMs
    set count=0
    for %%F in ("%ARTIFACTS_DIR%\*.json") do set /a count+=1
    call :log_success "Total SBOMs generated: !count!"
) else (
    call :log_warn "No SBOMs generated"
)
exit /b

:main
echo ======================================
echo GreenLang SBOM Generation Tool v%VERSION%
echo ======================================
echo.

REM Parse arguments
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help
if "%2"=="--help" goto :show_help
if "%2"=="-h" goto :show_help

REM Setup
if "%2"=="--clean" (
    call :log_info "Cleaning artifacts directory..."
    if exist "%ARTIFACTS_DIR%" rmdir /s /q "%ARTIFACTS_DIR%"
)
if "%3"=="--clean" (
    call :log_info "Cleaning artifacts directory..."
    if exist "%ARTIFACTS_DIR%" rmdir /s /q "%ARTIFACTS_DIR%"
)

if not exist "%ARTIFACTS_DIR%" mkdir "%ARTIFACTS_DIR%"

REM Install required tools
call :install_syft
if "%2"=="--docker" call :install_cosign
if "%2"=="" call :install_cosign

REM Generate Python package SBOMs
if "%2"=="--python" goto :python_sboms
if "%2"=="" goto :python_sboms
goto :docker_sboms

:python_sboms
call :log_info "Building Python packages..."

where python >nul 2>&1
if %errorlevel%==0 (
    python -m pip install --quiet --upgrade pip build
    python -m build

    REM Generate SBOMs for wheel
    for %%F in (dist\*.whl) do (
        if exist "%%F" (
            call :generate_wheel_sbom "%%F"
            goto :wheel_done
        )
    )
    :wheel_done

    REM Generate SBOMs for sdist
    for %%F in (dist\*.tar.gz) do (
        if exist "%%F" (
            call :generate_sdist_sbom "%%F"
            goto :sdist_done
        )
    )
    :sdist_done
) else (
    call :log_error "Python not found. Skipping Python package SBOMs."
)

if "%2"=="--python" goto :finish

:docker_sboms
if "%2"=="--docker" goto :process_docker
if "%2"=="" goto :process_docker
goto :finish

:process_docker
call :log_info "Processing Docker images..."

REM Build and generate SBOM for runner image
if exist "Dockerfile.runner" (
    docker images | findstr /C:"greenlang-runner" | findstr /C:"%VERSION%" >nul 2>&1
    if %errorlevel% neq 0 (
        call :log_info "Building runner image..."
        docker build -t "greenlang-runner:%VERSION%" ^
            --build-arg GL_VERSION="%VERSION%" ^
            -f Dockerfile.runner .
    )
    call :generate_docker_sbom "greenlang-runner" "%VERSION%" "runner"
)

REM Build and generate SBOM for full image
if exist "Dockerfile.full" (
    docker images | findstr /C:"greenlang-full" | findstr /C:"%VERSION%" >nul 2>&1
    if %errorlevel% neq 0 (
        call :log_info "Building full image..."
        docker build -t "greenlang-full:%VERSION%" ^
            --build-arg GL_VERSION="%VERSION%" ^
            -f Dockerfile.full .
    )
    call :generate_docker_sbom "greenlang-full" "%VERSION%" "full"
)

:finish
echo.
call :display_summary
echo.
echo ======================================
call :log_success "SBOM generation complete!"
echo.
echo Next steps:
echo   1. Review generated SBOMs in %ARTIFACTS_DIR%\
echo   2. Sign SBOMs with: cosign sign-blob --output-signature ^<sbom^>.sig ^<sbom^>.json
echo   3. Push images and create attestations in CI/CD
echo ======================================
goto :end

:show_help
echo Usage: %0 [version] [options]
echo.
echo Options:
echo   --python     Generate SBOMs for Python packages only
echo   --docker     Generate SBOMs for Docker images only
echo   --clean      Clean artifacts directory before generation
echo.
echo Examples:
echo   %0                    Generate all SBOMs for version 0.2.0
echo   %0 0.3.0              Generate all SBOMs for version 0.3.0
echo   %0 0.2.0 --python     Generate Python SBOMs only
echo   %0 0.2.0 --docker     Generate Docker SBOMs only

:end
endlocal