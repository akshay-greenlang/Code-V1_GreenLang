@echo off
REM GreenLang CI Smoke Test Script (Windows)
REM Runs basic functionality tests to verify the package works correctly

setlocal EnableDelayedExpansion

REM Script configuration
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."
set "TEST_DIR=%PROJECT_ROOT%\smoke-test-results"
set "LOG_FILE=%TEST_DIR%\smoke-test.log"

REM Test counters
set TESTS_PASSED=0
set TESTS_FAILED=0
set TESTS_TOTAL=0

REM Setup test directory
if not exist "%TEST_DIR%" mkdir "%TEST_DIR%"
echo. > "%LOG_FILE%"

REM Logging functions
:log
echo [SMOKE] %~1
echo [SMOKE] %~1 >> "%LOG_FILE%"
goto :eof

:log_success
echo [SUCCESS] %~1
echo [SUCCESS] %~1 >> "%LOG_FILE%"
set /a TESTS_PASSED+=1
goto :eof

:log_warning
echo [WARNING] %~1
echo [WARNING] %~1 >> "%LOG_FILE%"
goto :eof

:log_error
echo [ERROR] %~1
echo [ERROR] %~1 >> "%LOG_FILE%"
set /a TESTS_FAILED+=1
goto :eof

:log_test
echo [TEST] %~1
echo [TEST] %~1 >> "%LOG_FILE%"
set /a TESTS_TOTAL+=1
goto :eof

REM Test execution wrapper
:run_test
set "TEST_NAME=%~1"
set "TEST_COMMAND=%~2"
set "EXPECT_SUCCESS=%~3"
if "%EXPECT_SUCCESS%"=="" set "EXPECT_SUCCESS=true"

call :log_test "Running: %TEST_NAME%"

if "%EXPECT_SUCCESS%"=="true" (
    call %TEST_COMMAND% >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        call :log_error "%TEST_NAME% - FAILED"
    ) else (
        call :log_success "%TEST_NAME% - PASSED"
    )
) else (
    call %TEST_COMMAND% >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        call :log_success "%TEST_NAME% - PASSED (expected failure)"
    ) else (
        call :log_error "%TEST_NAME% - FAILED (expected to fail)"
    )
)
goto :eof

REM Test functions
:test_basic_import
python -c "import greenlang; print(f'GreenLang version: {greenlang.__version__}')" 2>> "%LOG_FILE%"
goto :eof

:test_cli_availability
gl --version 2>> "%LOG_FILE%"
goto :eof

:test_cli_help
gl --help | findstr /c:"GreenLang" > nul 2>> "%LOG_FILE%"
goto :eof

:test_basic_calculation
echo {"building_type": "office", "area": 1000, "energy_efficiency": "standard"} | gl calculate building-emissions --input-format json --output-format json > "%TEMP%\calc_result.json" 2>> "%LOG_FILE%"
if errorlevel 1 goto :eof

python -c "import sys; import json; data=json.load(open('%TEMP%\\calc_result.json')); assert any(k in data for k in ['emissions', 'energy_consumption', 'carbon_footprint']); print('Basic calculation test passed')" 2>> "%LOG_FILE%"
goto :eof

:test_pipeline_validation
if exist "%PROJECT_ROOT%\test_simple.yaml" (
    gl pipeline validate "%PROJECT_ROOT%\test_simple.yaml" 2>> "%LOG_FILE%"
) else (
    echo test_simple.yaml not found, skipping pipeline validation test >> "%LOG_FILE%"
)
goto :eof

:test_pack_validation
if exist "%PROJECT_ROOT%\packs" (
    REM Find first pack directory
    for /f %%i in ('dir /s /b "%PROJECT_ROOT%\packs\pack.yaml" 2^>nul') do (
        set "PACK_DIR=%%~dpi"
        goto found_pack
    )
    echo No pack.yaml found in packs directory, skipping pack validation test >> "%LOG_FILE%"
    goto :eof
    :found_pack
    gl pack validate "!PACK_DIR!" 2>> "%LOG_FILE%"
) else (
    echo packs directory not found, skipping pack validation test >> "%LOG_FILE%"
)
goto :eof

:test_sdk_import
python -c "from greenlang.sdk import GreenLangClient; print('SDK client import successful'); client = GreenLangClient(); print('SDK client initialization successful')" 2>> "%LOG_FILE%"
goto :eof

:test_core_modules
python -c "modules = ['greenlang.core', 'greenlang.cli', 'greenlang.agents', 'greenlang.utils']; [print(f'✓ {m}') if __import__(m) else None for m in modules]" 2>> "%LOG_FILE%"
goto :eof

:test_config_validation
python -c "import os; from pathlib import Path; root = Path('%PROJECT_ROOT%'); files = ['pyproject.toml', 'VERSION']; [print(f'✓ {f} exists') if (root/f).exists() and (root/f).stat().st_size > 0 else print(f'✗ {f} missing/empty') for f in files if f != 'VERSION' or print(f'ℹ {f} optional')]" 2>> "%LOG_FILE%"
goto :eof

:test_environment_compatibility
python -c "import sys, platform; print(f'Python: {sys.version}'); print(f'Platform: {platform.platform()}'); min_ver=(3,10); curr=sys.version_info[:2]; print(f'✓ Version OK') if curr>=min_ver else sys.exit(1)" 2>> "%LOG_FILE%"
goto :eof

:test_dependency_availability
python -c "import pkg_resources; deps=['typer','pydantic','pyyaml','rich','jsonschema']; [print(f'✓ {d}') if pkg_resources.get_distribution(d) else None for d in deps]" 2>> "%LOG_FILE%"
goto :eof

REM Main test execution
:main
call :log "Starting GreenLang smoke tests"
call :log "Project root: %PROJECT_ROOT%"
call :log "Test directory: %TEST_DIR%"

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
call :log "Python version: %PYTHON_VERSION%"

cd /d "%PROJECT_ROOT%"
if errorlevel 1 (
    call :log_error "Failed to change to project root directory"
    exit /b 1
)

REM Run smoke tests
call :log "=== Environment Tests ==="
call :run_test "Environment compatibility" ":test_environment_compatibility"
call :run_test "Configuration validation" ":test_config_validation"
call :run_test "Dependency availability" ":test_dependency_availability"

call :log "=== Import Tests ==="
call :run_test "Basic package import" ":test_basic_import"
call :run_test "Core modules import" ":test_core_modules"
call :run_test "SDK import" ":test_sdk_import"

call :log "=== CLI Tests ==="
call :run_test "CLI availability" ":test_cli_availability"
call :run_test "CLI help command" ":test_cli_help"

call :log "=== Functional Tests ==="
call :run_test "Basic calculation" ":test_basic_calculation"
call :run_test "Pipeline validation" ":test_pipeline_validation"
call :run_test "Pack validation" ":test_pack_validation"

REM Generate test report
call :log "=== Test Summary ==="
call :log "Tests run: %TESTS_TOTAL%"
call :log "Tests passed: %TESTS_PASSED%"
call :log "Tests failed: %TESTS_FAILED%"

REM Get timestamp
set "TIMESTAMP="
for /f "tokens=1,2,3,4 delims=/ " %%a in ('date /t') do (
    set "TIMESTAMP=%%d-%%b-%%c"
)
for /f "tokens=1,2 delims=: " %%a in ('time /t') do (
    set "TIMESTAMP=!TIMESTAMP!T%%a:%%b:00Z"
)

REM Get GreenLang version
set "GL_VERSION=unknown"
for /f %%i in ('python -c "import greenlang; print(greenlang.__version__)" 2^>nul') do set "GL_VERSION=%%i"

REM Create test results file
echo GreenLang Smoke Test Results > "%TEST_DIR%\smoke-test-results.txt"
echo ============================ >> "%TEST_DIR%\smoke-test-results.txt"
echo. >> "%TEST_DIR%\smoke-test-results.txt"
echo Timestamp: %TIMESTAMP% >> "%TEST_DIR%\smoke-test-results.txt"
echo Platform: Windows >> "%TEST_DIR%\smoke-test-results.txt"
echo Python: %PYTHON_VERSION% >> "%TEST_DIR%\smoke-test-results.txt"
echo GreenLang: %GL_VERSION% >> "%TEST_DIR%\smoke-test-results.txt"
echo. >> "%TEST_DIR%\smoke-test-results.txt"
echo Test Summary: >> "%TEST_DIR%\smoke-test-results.txt"
echo - Total tests: %TESTS_TOTAL% >> "%TEST_DIR%\smoke-test-results.txt"
echo - Passed: %TESTS_PASSED% >> "%TEST_DIR%\smoke-test-results.txt"
echo - Failed: %TESTS_FAILED% >> "%TEST_DIR%\smoke-test-results.txt"
echo. >> "%TEST_DIR%\smoke-test-results.txt"

if %TESTS_FAILED% EQU 0 (
    echo Status: PASSED >> "%TEST_DIR%\smoke-test-results.txt"
    call :log_success "All smoke tests passed!"
    echo SMOKE_TEST_STATUS=passed > "%TEST_DIR%\test-status.env"
    exit /b 0
) else (
    echo Status: FAILED >> "%TEST_DIR%\smoke-test-results.txt"
    call :log_error "%TESTS_FAILED% test(s) failed"
    echo SMOKE_TEST_STATUS=failed > "%TEST_DIR%\test-status.env"
    exit /b 1
)

echo. >> "%TEST_DIR%\smoke-test-results.txt"
echo Detailed log: %LOG_FILE% >> "%TEST_DIR%\smoke-test-results.txt"

REM Error handler
:handle_error
call :log_error "Smoke tests failed with exit code %ERRORLEVEL%"
echo SMOKE_TEST_STATUS=error > "%TEST_DIR%\test-status.env"
exit /b %ERRORLEVEL%

REM Script entry point
if "%~1"=="main" call :main
if "%~1"=="" call :main