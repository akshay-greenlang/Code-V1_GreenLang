@echo off
REM GreenLang Quickstart Examples - Windows Runner
REM This script runs all quickstart examples in sequence

setlocal enabledelayedexpansion

echo 🌍 GreenLang Quickstart Examples Runner
echo ======================================
echo.

REM Check if GreenLang is installed
echo 🔍 Checking GreenLang installation...
gl version >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ GreenLang CLI not found in PATH
    echo 💡 Install with: pip install greenlang-cli==0.3.0
    pause
    exit /b 1
)

REM Get version info
for /f "tokens=*" %%a in ('gl version 2^>nul') do set GL_VERSION=%%a
echo ✅ GreenLang CLI found: !GL_VERSION!

REM Check Python environment
echo.
echo 🐍 Checking Python environment...
python -c "import greenlang; print('✅ GreenLang Python package available')" 2>nul
if !errorlevel! neq 0 (
    echo ❌ GreenLang Python package not found
    echo 💡 Install with: pip install greenlang-cli[analytics]==0.3.0
    pause
    exit /b 1
)

echo.
echo 🚀 Running Examples...
echo =====================

REM Create results directory
if not exist "results" mkdir results

REM Example 1: Hello World
echo.
echo 📍 Example 1: Hello World Calculation
echo --------------------------------------
python hello-world.py
if !errorlevel! neq 0 (
    echo ❌ Hello World example failed
    echo Check your GreenLang installation and try again
    pause
    exit /b 1
)

echo.
echo ✅ Hello World example completed successfully!

REM Example 2: Data Processing
echo.
echo 📍 Example 2: Portfolio Data Processing
echo ---------------------------------------
python process-data.py
if !errorlevel! neq 0 (
    echo ❌ Data processing example failed
    echo Check the error messages above
    pause
    exit /b 1
)

echo.
echo ✅ Data processing example completed successfully!

REM Example 3: CLI Usage
echo.
echo 📍 Example 3: CLI Usage Demonstration
echo -------------------------------------

echo Testing basic CLI calculation...
gl calc --fuel-type electricity --consumption 1000 --unit kWh --location "San Francisco" --output results/cli_test_simple.json
if !errorlevel! neq 0 (
    echo ❌ Simple CLI test failed
    pause
    exit /b 1
)

echo ✅ Simple CLI calculation completed

echo Testing building-specific calculation...
gl calc --building-type office --area 2500 --fuels "electricity:50000:kWh,natural_gas:1000:therms" --location "San Francisco" --output results/cli_test_building.json
if !errorlevel! neq 0 (
    echo ❌ Building CLI test failed
    pause
    exit /b 1
)

echo ✅ Building-specific CLI calculation completed

REM Example 4: JSON Input Processing
echo.
echo 📍 Example 4: JSON File Processing
echo ----------------------------------

echo Processing single building...
gl calc --input sample-building.json --output results/single_building_result.json
if !errorlevel! neq 0 (
    echo ❌ Single building processing failed
    pause
    exit /b 1
)

echo ✅ Single building processing completed

REM Display results summary
echo.
echo 📊 RESULTS SUMMARY
echo ==================

echo Generated files:
dir /b results\*.json results\*.csv results\*.txt 2>nul | findstr /v "^$" | for /f %%f in ('more') do echo   📄 %%f

echo.
echo 🎉 All examples completed successfully!
echo.
echo 📚 What's next?
echo   • Check the 'results' directory for generated reports
echo   • Modify the sample data files with your own building data
echo   • Explore more examples in ../tutorials/
echo   • Read the documentation: https://greenlang.io/docs
echo   • Join our community: https://discord.gg/greenlang
echo.
echo 🌱 Start making an impact with GreenLang!

pause