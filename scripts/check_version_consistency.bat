@echo off
setlocal EnableDelayedExpansion

echo Checking version consistency for GreenLang...
echo =============================================

:: Get the root directory
set "ROOT_DIR=%~dp0.."
cd /d "%ROOT_DIR%"

:: 1. Read the VERSION file
if not exist "VERSION" (
    echo ERROR: VERSION file not found in root directory
    exit /b 1
)

set /p ROOT_VERSION=<VERSION
echo Root VERSION file: %ROOT_VERSION%

:: 2. Check pyproject.toml for hardcoded version
echo Checking pyproject.toml...
findstr /R "^[[:space:]]*version[[:space:]]*=" pyproject.toml >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ERROR: pyproject.toml must not set a hardcoded version
    echo        It should use dynamic = ["version"] instead
    exit /b 1
)
echo   OK - No hardcoded version found

:: 3. Check setup.py for hardcoded version
echo Checking setup.py...
if exist "setup.py" (
    findstr /R "version.*=.*['\"][0-9]" setup.py >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo ERROR: setup.py hardcodes a version. It should read from VERSION file
        exit /b 1
    )
    echo   OK - No hardcoded version found
) else (
    echo   Not found - OK
)

:: 4. Check VERSION.md mentions current version
echo Checking VERSION.md...
if exist "VERSION.md" (
    findstr "%ROOT_VERSION%" VERSION.md >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo   OK - Current version found
    ) else (
        echo   WARNING: VERSION.md doesn't mention current version %ROOT_VERSION%
    )
) else (
    echo   Not found
)

:: 5. Check Python package version
echo Checking Python package version...
for /f "delims=" %%i in ('python -c "import greenlang; print(greenlang.__version__)" 2^>nul') do set PYTHON_VERSION=%%i

if "%PYTHON_VERSION%"=="" (
    echo   Package not installed - run 'pip install -e .' to test
) else if "%PYTHON_VERSION%"=="%ROOT_VERSION%" (
    echo   OK - Version matches
) else (
    echo   ERROR: Python package reports version %PYTHON_VERSION% but VERSION file has %ROOT_VERSION%
    exit /b 1
)

:: Summary
echo.
echo =============================================
echo Version consistency check complete!
echo Version: %ROOT_VERSION%
echo.
echo To bump version:
echo   1. Edit VERSION file with new version
echo   2. Update VERSION.md with release notes
echo   3. Commit: git commit -m "chore(release): bump to X.Y.Z"
echo   4. Tag: git tag vX.Y.Z ^&^& git push --tags
echo   5. CI will build and publish with correct version

endlocal