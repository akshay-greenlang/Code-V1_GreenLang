@echo off
setlocal enabledelayedexpansion

REM GitHub Release Script for GreenLang v0.2.0b2 (Windows)
REM This script creates a pre-release on GitHub with all artifacts

REM Configuration
set "VERSION=v0.2.0b2"
set "RELEASE_TITLE=v0.2.0b2 – Infra Seed (Beta 2)"
set "PRERELEASE=true"

REM Get script directory and project root
set "SCRIPT_DIR=%~dp0"
for %%i in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fi"
set "DIST_DIR=%PROJECT_ROOT%\dist"
set "SBOM_DIR=%PROJECT_ROOT%\sbom"
set "RELEASE_NOTES=%PROJECT_ROOT%\RELEASE_NOTES_v0.2.0b2.md"

echo 🚀 Creating GitHub pre-release for GreenLang %VERSION%
echo 📁 Project root: %PROJECT_ROOT%

REM Verify required files exist
echo 🔍 Verifying artifacts...

REM Check distribution files
set "WHEEL_FILE=%DIST_DIR%\greenlang-0.2.0b2-py3-none-any.whl"
set "TARBALL_FILE=%DIST_DIR%\greenlang-0.2.0b2.tar.gz"

if not exist "%WHEEL_FILE%" (
    echo ❌ ERROR: Wheel file not found: %WHEEL_FILE%
    echo    Run 'python -m build' to generate distribution files
    exit /b 1
)

if not exist "%TARBALL_FILE%" (
    echo ❌ ERROR: Tarball not found: %TARBALL_FILE%
    echo    Run 'python -m build' to generate distribution files
    exit /b 1
)

echo ✅ Distribution files found

REM Check SBOM files
set "SBOM_FULL=%SBOM_DIR%\greenlang-full-0.2.0.spdx.json"
set "SBOM_DIST=%SBOM_DIR%\greenlang-dist-0.2.0.spdx.json"
set "SBOM_RUNNER=%SBOM_DIR%\greenlang-runner-0.2.0.spdx.json"
set "SBOM_ARGS="

if exist "%SBOM_FULL%" (
    echo ✅ SBOM found: greenlang-full-0.2.0.spdx.json
    set "SBOM_ARGS=!SBOM_ARGS! "%SBOM_FULL%""
) else (
    echo ⚠️  WARNING: SBOM file not found: %SBOM_FULL%
)

if exist "%SBOM_DIST%" (
    echo ✅ SBOM found: greenlang-dist-0.2.0.spdx.json
    set "SBOM_ARGS=!SBOM_ARGS! "%SBOM_DIST%""
) else (
    echo ⚠️  WARNING: SBOM file not found: %SBOM_DIST%
)

if exist "%SBOM_RUNNER%" (
    echo ✅ SBOM found: greenlang-runner-0.2.0.spdx.json
    set "SBOM_ARGS=!SBOM_ARGS! "%SBOM_RUNNER%""
) else (
    echo ⚠️  WARNING: SBOM file not found: %SBOM_RUNNER%
)

REM Check release notes
if not exist "%RELEASE_NOTES%" (
    echo ❌ ERROR: Release notes not found: %RELEASE_NOTES%
    exit /b 1
)

echo ✅ Release notes found

REM Verify gh CLI is installed and authenticated
where gh >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: GitHub CLI (gh) is not installed
    echo    Install from: https://cli.github.com/
    exit /b 1
)

gh auth status >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: GitHub CLI is not authenticated
    echo    Run 'gh auth login' to authenticate
    exit /b 1
)

echo ✅ GitHub CLI is ready

REM Check if we're in a git repository
if not exist "%PROJECT_ROOT%\.git" (
    echo ❌ ERROR: Not in a git repository
    exit /b 1
)

REM Check if tag already exists
git tag -l | findstr /x "%VERSION%" >nul
if %errorlevel% equ 0 (
    echo ⚠️  WARNING: Tag %VERSION% already exists
    echo    Use 'git tag -d %VERSION%' to delete it if needed
) else (
    echo ✅ Tag %VERSION% is available
)

echo.
echo 📋 Release Summary:
echo    Version: %VERSION%
echo    Title: %RELEASE_TITLE%
echo    Pre-release: %PRERELEASE%
echo    Wheel: greenlang-0.2.0b2-py3-none-any.whl
echo    Tarball: greenlang-0.2.0b2.tar.gz
if exist "%SBOM_FULL%" echo    SBOM: greenlang-full-0.2.0.spdx.json
if exist "%SBOM_DIST%" echo    SBOM: greenlang-dist-0.2.0.spdx.json
if exist "%SBOM_RUNNER%" echo    SBOM: greenlang-runner-0.2.0.spdx.json
echo.

REM Parse command line arguments
if "%1"=="--execute" goto :execute_release
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help

REM Default: show dry run
goto :show_dry_run

:execute_release
echo ⚡ EXECUTING RELEASE...

echo 🏷️  Creating git tag...
git tag "%VERSION%" -m "Release %VERSION%"
if %errorlevel% neq 0 (
    echo ❌ ERROR: Failed to create git tag
    exit /b 1
)

echo 📤 Pushing tag to origin...
git push origin "%VERSION%"
if %errorlevel% neq 0 (
    echo ❌ ERROR: Failed to push tag
    exit /b 1
)

echo 🎉 Creating GitHub release...

REM Build the gh release create command
set "RELEASE_CMD=gh release create "%VERSION%" --notes-file "%RELEASE_NOTES%" --title "%RELEASE_TITLE%" --prerelease "%WHEEL_FILE%" "%TARBALL_FILE%"!SBOM_ARGS!"

echo 📝 Executing: !RELEASE_CMD!
!RELEASE_CMD!
if %errorlevel% neq 0 (
    echo ❌ ERROR: Failed to create GitHub release
    exit /b 1
)

echo.
echo 🎉 GitHub release created successfully!

REM Get repository info for URL
for /f "tokens=*" %%i in ('gh repo view --json owner,name -q ".owner.login + \"/\" + .name"') do set "REPO_PATH=%%i"
echo 🔗 View at: https://github.com/!REPO_PATH!/releases/tag/%VERSION%

goto :end

:show_dry_run
echo 🧪 DRY RUN - Commands that would be executed:
echo.
echo 1. Create git tag:
echo    git tag %VERSION% -m "Release %VERSION%"
echo.
echo 2. Push tag:
echo    git push origin %VERSION%
echo.
echo 3. Create GitHub release:
echo    gh release create "%VERSION%" --notes-file "%RELEASE_NOTES%" --title "%RELEASE_TITLE%" --prerelease "%WHEEL_FILE%" "%TARBALL_FILE%"!SBOM_ARGS!
echo.
echo 🎯 To execute the release, run:
echo    %0 --execute
goto :end

:show_help
echo Usage: %0 [--execute^|--help]
echo.
echo Options:
echo   --execute    Execute the release (default is dry run)
echo   --help       Show this help message
echo.
echo By default, this script runs in dry-run mode to show what would be executed.
goto :end

:end
endlocal