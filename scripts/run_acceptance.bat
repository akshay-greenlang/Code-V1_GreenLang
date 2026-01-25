@echo off
REM GreenLang Acceptance Test Runner for Windows
REM Run this script to validate all acceptance criteria

echo ==================================================
echo     GreenLang Acceptance Test Suite
echo ==================================================
echo.

REM Check Python
echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

echo.
echo Checking required tools...

REM Check gl command
where gl >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] gl is installed
) else (
    echo [WARNING] gl is not installed - some tests may be skipped
)

REM Check cosign
where cosign >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] cosign is installed
) else (
    echo [WARNING] cosign is not installed - some tests may be skipped
)

REM Check oras
where oras >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] oras is installed
) else (
    echo [WARNING] oras is not installed - some tests may be skipped
)

REM Check opa
where opa >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] opa is installed
) else (
    echo [WARNING] opa is not installed - some tests may be skipped
)

echo.
echo ==================================================
echo Running Acceptance Tests
echo ==================================================
echo.

REM Check for quick mode
if "%1"=="--quick" (
    echo Running quick tests only...
    python acceptance_test.py --test scaffolding
    python acceptance_test.py --test determinism
) else (
    echo Running full test suite...
    python acceptance_test.py --verbose --export-results acceptance-results.json
)

if %errorlevel% equ 0 (
    echo.
    echo ==================================================
    echo ALL ACCEPTANCE TESTS PASSED!
    echo ==================================================
    echo.
    echo Next steps:
    echo 1. Review test results in acceptance-results.json
    echo 2. Check performance metrics
    echo 3. Create PR with results
) else (
    echo.
    echo ==================================================
    echo SOME TESTS FAILED
    echo ==================================================
    echo.
    echo Please review the failures above and fix before merging.
    exit /b 1
)

REM Performance summary
echo.
echo Performance Summary:
echo -------------------

if exist acceptance-results.json (
    python -c "import json; data = json.load(open('acceptance-results.json')); timings = data.get('timings', {}); print(f'Tests run: {len(timings)}') if timings else None"
)

echo.
echo ==================================================
echo Test run complete!
echo ==================================================

pause