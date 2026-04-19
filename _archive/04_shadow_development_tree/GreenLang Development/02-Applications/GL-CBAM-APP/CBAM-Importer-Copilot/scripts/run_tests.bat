@echo off
REM CBAM Importer Copilot - Test Runner Script (Windows)
REM
REM Runs comprehensive test suite with multiple configurations
REM
REM Usage:
REM   run_tests.bat              - Run all tests
REM   run_tests.bat unit         - Run unit tests only
REM   run_tests.bat integration  - Run integration tests only
REM   run_tests.bat fast         - Run fast tests only (skip slow)
REM   run_tests.bat compliance   - Run compliance tests only
REM   run_tests.bat performance  - Run performance tests only
REM   run_tests.bat coverage     - Run with coverage report
REM
REM Version: 1.0.0

echo ========================================
echo CBAM Importer Copilot - Test Suite
echo ========================================
echo.

REM Check if pytest is installed
python -c "import pytest" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pytest not installed
    echo Installing pytest...
    pip install pytest pytest-cov
)

REM Set PYTHONPATH
set PYTHONPATH=%CD%

REM Default: run all tests
set TEST_MODE=%1
if "%TEST_MODE%"=="" set TEST_MODE=all

echo Test Mode: %TEST_MODE%
echo.

REM Run tests based on mode
if "%TEST_MODE%"=="all" (
    echo Running ALL tests...
    pytest tests/ -v --tb=short
    goto :end
)

if "%TEST_MODE%"=="unit" (
    echo Running UNIT tests only...
    pytest tests/ -v --tb=short -m unit
    goto :end
)

if "%TEST_MODE%"=="integration" (
    echo Running INTEGRATION tests only...
    pytest tests/ -v --tb=short -m integration
    goto :end
)

if "%TEST_MODE%"=="fast" (
    echo Running FAST tests only (skipping slow tests)...
    pytest tests/ -v --tb=short -m "not slow"
    goto :end
)

if "%TEST_MODE%"=="compliance" (
    echo Running COMPLIANCE tests only...
    pytest tests/ -v --tb=short -m compliance
    goto :end
)

if "%TEST_MODE%"=="performance" (
    echo Running PERFORMANCE tests only...
    pytest tests/ -v --tb=short -m performance
    goto :end
)

if "%TEST_MODE%"=="coverage" (
    echo Running tests with COVERAGE report...
    pytest tests/ -v --tb=short --cov=. --cov-report=html --cov-report=term
    echo.
    echo Coverage report generated: htmlcov/index.html
    goto :end
)

if "%TEST_MODE%"=="security" (
    echo Running SECURITY tests only...
    pytest tests/ -v --tb=short -m security
    goto :end
)

REM Invalid mode
echo ERROR: Unknown test mode: %TEST_MODE%
echo.
echo Valid modes: all, unit, integration, fast, compliance, performance, coverage, security
exit /b 1

:end
echo.
echo ========================================
echo Test run complete!
echo ========================================
