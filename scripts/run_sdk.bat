@echo off
echo ==========================================
echo        GreenLang SDK Runner
echo ==========================================
echo.
echo Available SDK Demos:
echo.
echo 1. Emissions Calculator
echo 2. Building Analysis
echo 3. Country Comparison
echo 4. Full SDK Demo
echo 5. Custom Python Command
echo 6. Interactive Python Shell
echo.
set /p choice="Enter your choice (1-6): "

if %choice%==1 (
    echo.
    echo Running Emissions Calculator...
    echo ==========================================
    python sdk_emissions.py
)

if %choice%==2 (
    echo.
    echo Select country for building analysis:
    echo US, IN, EU, CN, JP, BR, KR
    set /p country="Enter country code: "
    echo.
    echo Running Building Analysis for %country%...
    echo ==========================================
    python sdk_building.py %country%
)

if %choice%==3 (
    echo.
    echo Running Country Comparison...
    echo ==========================================
    python sdk_compare.py
)

if %choice%==4 (
    echo.
    echo Running Full SDK Demo...
    echo ==========================================
    python sdk_full_demo.py
)

if %choice%==5 (
    echo.
    echo Enter Python command to run:
    echo Example: from greenlang.sdk import GreenLangClient; print(GreenLangClient().list_agents())
    set /p cmd=">>> "
    echo.
    python -c "%cmd%"
)

if %choice%==6 (
    echo.
    echo Starting Interactive Python Shell...
    echo Type 'exit()' to quit
    echo ==========================================
    python
)

echo.
pause