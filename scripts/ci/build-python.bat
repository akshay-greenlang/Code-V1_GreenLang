@echo off
REM GreenLang CI Build Script (Windows)
REM Builds Python packages (wheels and source distribution) for CI

setlocal EnableDelayedExpansion

REM Script configuration
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."
set "BUILD_DIR=%PROJECT_ROOT%\dist"
set "LOG_FILE=%PROJECT_ROOT%\build.log"

REM Initialize log file
echo. > "%LOG_FILE%"

REM Logging functions
set "LOG_PREFIX=[BUILD]"

:log
echo %LOG_PREFIX% %~1
echo %LOG_PREFIX% %~1 >> "%LOG_FILE%"
goto :eof

:log_success
echo [SUCCESS] %~1
echo [SUCCESS] %~1 >> "%LOG_FILE%"
goto :eof

:log_warning
echo [WARNING] %~1
echo [WARNING] %~1 >> "%LOG_FILE%"
goto :eof

:log_error
echo [ERROR] %~1
echo [ERROR] %~1 >> "%LOG_FILE%"
goto :eof

:handle_error
call :log_error "Build failed with exit code %ERRORLEVEL%"
call :log_error "Check %LOG_FILE% for details"
exit /b %ERRORLEVEL%

REM Main build function
:main
call :log "Starting GreenLang Python package build"
call :log "Project root: %PROJECT_ROOT%"
call :log "Build directory: %BUILD_DIR%"

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
call :log "Python version: %PYTHON_VERSION%"

REM Get pip version
for /f "tokens=2" %%i in ('pip --version 2^>^&1') do set "PIP_VERSION=%%i"
call :log "Pip version: %PIP_VERSION%"

cd /d "%PROJECT_ROOT%"
if errorlevel 1 (
    call :log_error "Failed to change to project root directory"
    goto handle_error
)

REM Clean previous builds
call :log "Cleaning previous build artifacts..."
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
if exist "build" rmdir /s /q "build"
if exist "*.egg-info" rmdir /s /q "*.egg-info"
mkdir "%BUILD_DIR%"

REM Verify project structure
call :log "Verifying project structure..."
if not exist "pyproject.toml" (
    call :log_error "pyproject.toml not found in project root"
    exit /b 1
)

if not exist "greenlang" (
    call :log_error "greenlang package directory not found"
    exit /b 1
)

REM Extract version from pyproject.toml
for /f "tokens=2 delims=^" %%i in ('findstr "^version = " pyproject.toml') do (
    set "VERSION_LINE=%%i"
)
for /f "tokens=2 delims=^"" %%i in ("%VERSION_LINE%") do set "VERSION=%%i"
call :log "Building version: %VERSION%"

REM Install build dependencies
call :log "Installing build dependencies..."
python -m pip install --upgrade pip >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log_error "Failed to upgrade pip"
    goto handle_error
)

pip install build wheel setuptools >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log_error "Failed to install build dependencies"
    goto handle_error
)

REM Validate package configuration
call :log "Validating package configuration..."
python -c "import toml; import sys; config = toml.load('pyproject.toml'); project = config.get('project', {}); required = ['name', 'version', 'description']; missing = [f for f in required if f not in project]; sys.exit(1 if missing else 0) or print('Package configuration is valid')" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log_error "Invalid package configuration"
    goto handle_error
)

REM Build source distribution
call :log "Building source distribution..."
python -m build --sdist --outdir "%BUILD_DIR%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log_error "Failed to build source distribution"
    goto handle_error
)

REM Build wheel
call :log "Building wheel..."
python -m build --wheel --outdir "%BUILD_DIR%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log_error "Failed to build wheel"
    goto handle_error
)

REM Verify build artifacts
call :log "Verifying build artifacts..."
set WHEEL_COUNT=0
set SDIST_COUNT=0

for %%f in ("%BUILD_DIR%\*.whl") do set /a WHEEL_COUNT+=1
for %%f in ("%BUILD_DIR%\*.tar.gz") do set /a SDIST_COUNT+=1

if %WHEEL_COUNT% EQU 0 (
    call :log_error "No wheel files found in build output"
    exit /b 1
)

if %SDIST_COUNT% EQU 0 (
    call :log_error "No source distribution files found in build output"
    exit /b 1
)

call :log_success "Built %WHEEL_COUNT% wheel(s) and %SDIST_COUNT% source distribution(s)"

REM List build artifacts
call :log "Build artifacts:"
for %%f in ("%BUILD_DIR%\*") do (
    call :log "  %%~nxf"
)

REM Basic package validation
call :log "Performing basic package validation..."

REM Check wheel contents
for %%f in ("%BUILD_DIR%\*.whl") do (
    call :log "Validating wheel: %%~nxf"
    python -m zipfile -l "%%f" | findstr /c:"greenlang/" > nul
    if errorlevel 1 (
        call :log_error "Wheel does not contain greenlang package"
        exit /b 1
    )
)

REM Test wheel installation in temporary environment
call :log "Testing wheel installation..."
set "TEMP_VENV=%TEMP%\gl-test-venv-%RANDOM%"
python -m venv "%TEMP_VENV%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log_error "Failed to create test virtual environment"
    goto handle_error
)

call "%TEMP_VENV%\Scripts\activate.bat"
pip install --upgrade pip >> "%LOG_FILE%" 2>&1

REM Install the wheel
for %%f in ("%BUILD_DIR%\*.whl") do (
    pip install "%%f" >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        call :log_error "Failed to install wheel %%~nxf"
        call "%TEMP_VENV%\Scripts\deactivate.bat"
        rmdir /s /q "%TEMP_VENV%"
        goto handle_error
    )
)

REM Basic import test
python -c "import greenlang; print(f'Successfully imported GreenLang {greenlang.__version__}')" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log_error "Failed to import greenlang package"
    call "%TEMP_VENV%\Scripts\deactivate.bat"
    rmdir /s /q "%TEMP_VENV%"
    goto handle_error
)

REM Test CLI availability
gl --version >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log_error "CLI not available"
    call "%TEMP_VENV%\Scripts\deactivate.bat"
    rmdir /s /q "%TEMP_VENV%"
    goto handle_error
)

call "%TEMP_VENV%\Scripts\deactivate.bat"
rmdir /s /q "%TEMP_VENV%"

call :log_success "Wheel installation test passed"

REM Generate build metadata
call :log "Generating build metadata..."
set "TIMESTAMP="
for /f "tokens=1,2,3,4 delims=/ " %%a in ('date /t') do (
    set "TIMESTAMP=%%d-%%b-%%c"
)
for /f "tokens=1,2 delims=: " %%a in ('time /t') do (
    set "TIMESTAMP=!TIMESTAMP!T%%a:%%b:00Z"
)

REM Get Git SHA if available
set "GIT_SHA=unknown"
if defined GITHUB_SHA (
    set "GIT_SHA=%GITHUB_SHA%"
) else (
    for /f %%i in ('git rev-parse HEAD 2^>nul') do set "GIT_SHA=%%i"
)

echo { > "%BUILD_DIR%\build-metadata.json"
echo     "build_timestamp": "%TIMESTAMP%", >> "%BUILD_DIR%\build-metadata.json"
echo     "version": "%VERSION%", >> "%BUILD_DIR%\build-metadata.json"
echo     "python_version": "%PYTHON_VERSION%", >> "%BUILD_DIR%\build-metadata.json"
echo     "platform": "Windows", >> "%BUILD_DIR%\build-metadata.json"
echo     "build_script": "%~nx0", >> "%BUILD_DIR%\build-metadata.json"
echo     "git_sha": "%GIT_SHA%", >> "%BUILD_DIR%\build-metadata.json"
echo     "artifacts": [ >> "%BUILD_DIR%\build-metadata.json"

set FIRST=1
for %%f in ("%BUILD_DIR%\*.whl" "%BUILD_DIR%\*.tar.gz") do (
    if !FIRST! EQU 1 (
        echo         "%%~nxf" >> "%BUILD_DIR%\build-metadata.json"
        set FIRST=0
    ) else (
        echo         ,"%%~nxf" >> "%BUILD_DIR%\build-metadata.json"
    )
)

echo     ] >> "%BUILD_DIR%\build-metadata.json"
echo } >> "%BUILD_DIR%\build-metadata.json"

call :log_success "Build completed successfully!"
call :log "Build summary:"
call :log "  Version: %VERSION%"
call :log "  Artifacts: %WHEEL_COUNT% wheel(s), %SDIST_COUNT% source distribution(s)"
call :log "  Output directory: %BUILD_DIR%"
call :log "  Log file: %LOG_FILE%"

exit /b 0

REM Script entry point
:eof
if "%~1"=="main" call :main
if "%~1"=="" call :main