@echo off
REM Chaos Engineering Test Suite Runner for Windows
REM
REM Usage: run_chaos_tests.bat [options]
REM
REM Options:
REM   all            Run all chaos tests (default)
REM   failover       Run failover tests only
REM   database       Run database resilience tests only
REM   latency        Run latency/timeout tests only
REM   resource       Run resource pressure tests only
REM   integration    Run integration tests only
REM   quick          Run quick tests only
REM   verbose        Show detailed output
REM   report         Generate JSON and HTML reports
REM   help           Show this help message

setlocal enabledelayedexpansion

REM Get script directory
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..\..\
set REPORT_DIR=%PROJECT_DIR%chaos-reports

REM Create report directory
if not exist "%REPORT_DIR%" mkdir "%REPORT_DIR%"

REM Default options
set RUN_MODE=all
set VERBOSE=false
set GENERATE_REPORT=false

REM Parse arguments
:parse_args
if "%1"=="" goto args_done
if /i "%1"=="all" (
    set RUN_MODE=all
    shift
    goto parse_args
)
if /i "%1"=="failover" (
    set RUN_MODE=failover
    shift
    goto parse_args
)
if /i "%1"=="database" (
    set RUN_MODE=database
    shift
    goto parse_args
)
if /i "%1"=="latency" (
    set RUN_MODE=latency
    shift
    goto parse_args
)
if /i "%1"=="resource" (
    set RUN_MODE=resource
    shift
    goto parse_args
)
if /i "%1"=="integration" (
    set RUN_MODE=integration
    shift
    goto parse_args
)
if /i "%1"=="quick" (
    set RUN_MODE=quick
    shift
    goto parse_args
)
if /i "%1"=="verbose" (
    set VERBOSE=true
    shift
    goto parse_args
)
if /i "%1"=="report" (
    set GENERATE_REPORT=true
    shift
    goto parse_args
)
if /i "%1"=="help" (
    echo Chaos Engineering Test Suite Runner for Windows
    echo.
    echo Usage: run_chaos_tests.bat [options]
    echo.
    echo Options:
    echo   all            Run all chaos tests (default)
    echo   failover       Run failover tests only
    echo   database       Run database resilience tests only
    echo   latency        Run latency/timeout tests only
    echo   resource       Run resource pressure tests only
    echo   integration    Run integration tests only
    echo   quick          Run quick tests only
    echo   verbose        Show detailed output
    echo   report         Generate JSON and HTML reports
    echo   help           Show this help message
    exit /b 0
)
shift
goto parse_args

:args_done

REM Build pytest command
set PYTEST_CMD=pytest "%SCRIPT_DIR%"
set PYTEST_ARGS=-v -m chaos

if /i "%RUN_MODE%"=="failover" (
    set "PYTEST_ARGS=!PYTEST_ARGS! and chaos_failover"
    echo Running Failover Tests...
) else if /i "%RUN_MODE%"=="database" (
    set "PYTEST_ARGS=!PYTEST_ARGS! and chaos_database"
    echo Running Database Resilience Tests...
) else if /i "%RUN_MODE%"=="latency" (
    set "PYTEST_ARGS=!PYTEST_ARGS! and chaos_latency"
    echo Running Latency/Timeout Tests...
) else if /i "%RUN_MODE%"=="resource" (
    set "PYTEST_ARGS=!PYTEST_ARGS! and chaos_resource"
    echo Running Resource Pressure Tests...
) else if /i "%RUN_MODE%"=="integration" (
    set "PYTEST_CMD=pytest "%SCRIPT_DIR%test_process_heat_agent_chaos.py""
    echo Running Integration Tests...
) else if /i "%RUN_MODE%"=="quick" (
    set "PYTEST_ARGS=!PYTEST_ARGS! and not chaos_slow"
    echo Running Quick Tests (excluding slow tests^)...
) else (
    echo Running All Chaos Tests...
)

REM Add verbose logging if requested
if "%VERBOSE%"=="true" (
    set "PYTEST_ARGS=!PYTEST_ARGS! -s --log-cli-level=DEBUG"
)

REM Add report generation if requested
if "%GENERATE_REPORT%"=="true" (
    set "PYTEST_ARGS=!PYTEST_ARGS! --junitxml=%REPORT_DIR%\chaos-results.xml --html=%REPORT_DIR%\chaos-report.html"
    echo Reports will be saved to: %REPORT_DIR%
)

REM Add timeout and other options
set "PYTEST_ARGS=!PYTEST_ARGS! --timeout=300 --tb=short"

REM Run tests
echo Command: %PYTEST_CMD% %PYTEST_ARGS%
echo ---
echo.

%PYTEST_CMD% %PYTEST_ARGS%

if %errorlevel% equ 0 (
    echo.
    echo ===== CHAOS TESTS PASSED =====
    if "%GENERATE_REPORT%"=="true" (
        echo Reports available:
        echo   - JSON: %REPORT_DIR%\chaos-results.xml
        echo   - HTML: %REPORT_DIR%\chaos-report.html
    )
    exit /b 0
) else (
    echo.
    echo ===== CHAOS TESTS FAILED =====
    exit /b 1
)
