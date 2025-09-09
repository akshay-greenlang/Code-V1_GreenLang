@echo off
REM GreenLang Comprehensive QA Test Suite for Windows
REM Automated testing script for production readiness

setlocal enabledelayedexpansion

REM Initialize counters
set TOTAL_TESTS=0
set PASSED_TESTS=0
set FAILED_TESTS=0
set WARNINGS=0

REM Get start time
set START_TIME=%time%

echo ========================================
echo GreenLang v0.0.1 - QA Test Suite
echo ========================================
echo Date: %date%
echo Platform: Windows
python --version
echo.

REM ============================================
REM 1. ENVIRONMENT CHECKS
REM ============================================
echo.
echo ======== 1. Environment Verification ========
echo.

REM Check Python version
python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"
if %errorlevel% equ 0 (
    echo [PASS] Python version is supported
    set /a PASSED_TESTS+=1
) else (
    echo [FAIL] Python version is not supported
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1

REM Check required tools
for %%T in (pip pytest mypy ruff black) do (
    where %%T >nul 2>&1
    if !errorlevel! equ 0 (
        echo [PASS] %%T is installed
        set /a PASSED_TESTS+=1
    ) else (
        echo [FAIL] %%T is not installed
        set /a FAILED_TESTS+=1
    )
    set /a TOTAL_TESTS+=1
)

REM ============================================
REM 2. DEPENDENCY CHECKS
REM ============================================
echo.
echo ======== 2. Dependency Installation ========
echo.

REM Install dependencies
echo [INFO] Installing dependencies...
pip install -q -r requirements.txt
if %errorlevel% equ 0 (
    echo [PASS] Dependencies installed successfully
    set /a PASSED_TESTS+=1
) else (
    echo [FAIL] Failed to install dependencies
    set /a FAILED_TESTS+=1
    goto :END
)
set /a TOTAL_TESTS+=1

REM Install package in development mode
pip install -q -e .
if %errorlevel% equ 0 (
    echo [PASS] GreenLang installed in development mode
    set /a PASSED_TESTS+=1
) else (
    echo [FAIL] Failed to install GreenLang
    set /a FAILED_TESTS+=1
    goto :END
)
set /a TOTAL_TESTS+=1

REM ============================================
REM 3. SECURITY SCANNING
REM ============================================
echo.
echo ======== 3. Security Scanning ========
echo.

REM Install security tools
pip install -q pip-audit safety bandit

REM Run pip-audit
echo [INFO] Running pip-audit for vulnerability scanning...
pip-audit --desc > security_audit.txt 2>&1
if %errorlevel% equ 0 (
    echo [PASS] No dependency vulnerabilities found
    set /a PASSED_TESTS+=1
) else (
    echo [WARN] Dependency vulnerabilities detected - see security_audit.txt
    set /a WARNINGS+=1
)
set /a TOTAL_TESTS+=1

REM Run bandit
echo [INFO] Running bandit security linter...
bandit -r greenlang/ -ll -f txt > bandit_report.txt 2>&1
findstr /C:"No issues identified" bandit_report.txt >nul
if %errorlevel% equ 0 (
    echo [PASS] No security issues in code
    set /a PASSED_TESTS+=1
) else (
    echo [WARN] Security issues found - see bandit_report.txt
    set /a WARNINGS+=1
)
set /a TOTAL_TESTS+=1

REM ============================================
REM 4. CODE QUALITY CHECKS
REM ============================================
echo.
echo ======== 4. Code Quality Analysis ========
echo.

REM Type checking with mypy
echo [INFO] Running mypy type checker...
mypy greenlang/ --strict --ignore-missing-imports > mypy_report.txt 2>&1
if %errorlevel% equ 0 (
    echo [PASS] Type checking passed - strict mode
    set /a PASSED_TESTS+=1
) else (
    echo [WARN] Type checking issues found - see mypy_report.txt
    set /a WARNINGS+=1
)
set /a TOTAL_TESTS+=1

REM Linting with ruff
echo [INFO] Running ruff linter...
ruff check greenlang/ tests/ --statistics > ruff_report.txt 2>&1
if %errorlevel% equ 0 (
    echo [PASS] Code linting passed
    set /a PASSED_TESTS+=1
) else (
    echo [WARN] Linting issues found - see ruff_report.txt
    set /a WARNINGS+=1
)
set /a TOTAL_TESTS+=1

REM Code formatting check
echo [INFO] Checking code formatting with black...
black --check greenlang/ tests/ > black_report.txt 2>&1
if %errorlevel% equ 0 (
    echo [PASS] Code formatting is correct
    set /a PASSED_TESTS+=1
) else (
    echo [WARN] Code formatting issues - run: black greenlang/ tests/
    set /a WARNINGS+=1
)
set /a TOTAL_TESTS+=1

REM ============================================
REM 5. UNIT TESTS
REM ============================================
echo.
echo ======== 5. Unit Tests ========
echo.

echo [INFO] Running unit tests...
pytest tests/unit/ -v --tb=short --timeout=30 -q > unit_test_results.txt 2>&1
if %errorlevel% equ 0 (
    echo [PASS] All unit tests passed
    set /a PASSED_TESTS+=1
) else (
    echo [FAIL] Unit tests failed - see unit_test_results.txt
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1

REM ============================================
REM 6. INTEGRATION TESTS
REM ============================================
echo.
echo ======== 6. Integration Tests ========
echo.

if exist tests\integration (
    echo [INFO] Running integration tests...
    pytest tests/integration/ -v --tb=short --timeout=60 -q > integration_test_results.txt 2>&1
    if !errorlevel! equ 0 (
        echo [PASS] All integration tests passed
        set /a PASSED_TESTS+=1
    ) else (
        echo [FAIL] Integration tests failed - see integration_test_results.txt
        set /a FAILED_TESTS+=1
    )
) else (
    echo [INFO] No integration tests found
)
set /a TOTAL_TESTS+=1

REM ============================================
REM 7. CLI TESTING
REM ============================================
echo.
echo ======== 7. CLI Command Testing ========
echo.

REM Test basic CLI commands
echo [INFO] Testing: greenlang --version
python -m greenlang.cli.main --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [PASS] Command: greenlang --version
    set /a PASSED_TESTS+=1
) else (
    echo [FAIL] Command failed: greenlang --version
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1

echo [INFO] Testing: greenlang --help
python -m greenlang.cli.main --help >nul 2>&1
if %errorlevel% equ 0 (
    echo [PASS] Command: greenlang --help
    set /a PASSED_TESTS+=1
) else (
    echo [FAIL] Command failed: greenlang --help
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1

echo [INFO] Testing: gl agents
python -m greenlang.cli.main agents >nul 2>&1
if %errorlevel% equ 0 (
    echo [PASS] Command: gl agents
    set /a PASSED_TESTS+=1
) else (
    echo [FAIL] Command failed: gl agents
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1

REM ============================================
REM 8. TEST COVERAGE
REM ============================================
echo.
echo ======== 8. Test Coverage Analysis ========
echo.

echo [INFO] Calculating test coverage...
pytest --cov=greenlang --cov-report=term --cov-report=html --cov-fail-under=85 -q > coverage_report.txt 2>&1
if %errorlevel% equ 0 (
    echo [PASS] Test coverage >= 85%%
    echo Coverage report generated in htmlcov/
    set /a PASSED_TESTS+=1
) else (
    echo [FAIL] Test coverage ^< 85%% - see coverage_report.txt
    set /a FAILED_TESTS+=1
)
set /a TOTAL_TESTS+=1

REM ============================================
REM 9. DOCUMENTATION VALIDATION
REM ============================================
echo.
echo ======== 9. Documentation Validation ========
echo.

REM Check if key documentation files exist
for %%F in (README.md GREENLANG_DOCUMENTATION.md requirements.txt) do (
    if exist %%F (
        echo [PASS] Documentation: %%F exists
        set /a PASSED_TESTS+=1
    ) else (
        echo [FAIL] Missing documentation: %%F
        set /a FAILED_TESTS+=1
    )
    set /a TOTAL_TESTS+=1
)

REM ============================================
REM 10. JSON SCHEMA VALIDATION
REM ============================================
echo.
echo ======== 10. Data Schema Validation ========
echo.

REM Check if schema files exist
if exist schemas (
    for %%S in (schemas\*.json) do (
        echo [INFO] Validating schema: %%~nxS
        python -m json.tool "%%S" >nul 2>&1
        if !errorlevel! equ 0 (
            echo [PASS] Valid JSON schema: %%~nxS
            set /a PASSED_TESTS+=1
        ) else (
            echo [FAIL] Invalid JSON schema: %%~nxS
            set /a FAILED_TESTS+=1
        )
        set /a TOTAL_TESTS+=1
    )
) else (
    echo [WARN] No schema directory found
    set /a WARNINGS+=1
)

REM ============================================
REM FINAL REPORT
REM ============================================
:END
echo.
echo ========================================
echo QA Test Suite Summary
echo ========================================
echo.

REM Get end time
set END_TIME=%time%

echo Test Execution Completed: %END_TIME%
echo Total Tests Run: %TOTAL_TESTS%
echo Passed: %PASSED_TESTS%
echo Warnings: %WARNINGS%
echo Failed: %FAILED_TESTS%
echo.

REM Calculate pass rate
if %TOTAL_TESTS% gtr 0 (
    set /a PASS_RATE=PASSED_TESTS*100/TOTAL_TESTS
    echo Pass Rate: !PASS_RATE!%%
)

REM Generate detailed report
set REPORT_FILE=qa_report_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.txt
(
    echo GreenLang QA Report
    echo ===================
    echo Date: %date%
    echo Time: %time%
    echo.
    echo Results Summary:
    echo - Total Tests: %TOTAL_TESTS%
    echo - Passed: %PASSED_TESTS%
    echo - Failed: %FAILED_TESTS%
    echo - Warnings: %WARNINGS%
    echo - Pass Rate: !PASS_RATE!%%
    echo.
    echo Generated Reports:
    dir /b *.txt 2>nul || echo No report files generated
) > "%REPORT_FILE%"

echo.
echo Detailed report saved to: %REPORT_FILE%

REM Exit code based on failures
if %FAILED_TESTS% equ 0 (
    echo.
    echo [SUCCESS] QA SUITE PASSED - Ready for production
    exit /b 0
) else (
    echo.
    echo [ERROR] QA SUITE FAILED - %FAILED_TESTS% tests failed
    echo Please review the detailed reports and fix issues before release.
    exit /b 1
)