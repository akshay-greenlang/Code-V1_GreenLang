@echo off
echo ==========================================
echo    GreenLang v0.0.2 - Command Demo
echo ==========================================
echo.

echo Press any key to see each command in action...
pause >nul

echo.
echo 1. Showing version:
echo    Command: greenlang --version
echo    ---------------------
greenlang --version
echo.
pause

echo.
echo 2. Listing all agents:
echo    Command: greenlang agents
echo    ---------------------
greenlang agents
echo.
pause

echo.
echo 3. Showing agent details:
echo    Command: greenlang agent grid_factor
echo    ---------------------
greenlang agent grid_factor
echo.
pause

echo.
echo 4. Viewing benchmarks for India:
echo    Command: greenlang benchmark --type commercial_office --country IN
echo    ---------------------
greenlang benchmark --type commercial_office --country IN
echo.
pause

echo.
echo 5. Initializing a sample workflow:
echo    Command: greenlang init --output demo_workflow.yaml
echo    ---------------------
greenlang init --output demo_workflow.yaml
echo.
pause

echo.
echo ==========================================
echo    Demo Complete!
echo ==========================================
echo.
echo Remember: All commands must start with 'greenlang'
echo.
echo Key Commands:
echo   greenlang calc               - Simple calculator
echo   greenlang calc --building    - Building calculator
echo   greenlang analyze [file]     - Analyze building
echo   greenlang benchmark          - View benchmarks
echo   greenlang recommend          - Get recommendations
echo   greenlang ask "question"     - Ask AI assistant
echo.
echo For full documentation, see GREENLANG_DOCUMENTATION.md
echo For command reference, see COMMANDS_REFERENCE.md
echo.