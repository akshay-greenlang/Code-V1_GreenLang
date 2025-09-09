@echo off
echo ==========================================
echo    GreenLang v0.0.2 - Command Demo
echo ==========================================
echo.

echo Press any key to see each command in action...
pause >nul

echo.
echo 1. Showing version:
echo    Command: gl --version
echo    ---------------------
gl --version
echo.
pause

echo.
echo 2. Listing all agents:
echo    Command: gl agents
echo    ---------------------
gl agents
echo.
pause

echo.
echo 3. Showing agent details:
echo    Command: gl agent grid_factor
echo    ---------------------
gl agent grid_factor
echo.
pause

echo.
echo 4. Viewing benchmarks for India:
echo    Command: gl benchmark --type commercial_office --country IN
echo    ---------------------
gl benchmark --type commercial_office --country IN
echo.
pause

echo.
echo 5. Initializing a sample workflow:
echo    Command: gl init --output demo_workflow.yaml
echo    ---------------------
gl init --output demo_workflow.yaml
echo.
pause

echo.
echo ==========================================
echo    Demo Complete!
echo ==========================================
echo.
echo Remember: All commands must start with 'gl'
echo.
echo Key Commands:
echo   gl calc               - Simple calculator
echo   gl calc --building    - Building calculator
echo   gl analyze [file]     - Analyze building
echo   gl benchmark          - View benchmarks
echo   gl recommend          - Get recommendations
echo   gl ask "question"     - Ask AI assistant
echo.
echo For full documentation, see GREENLANG_DOCUMENTATION.md
echo For command reference, see COMMANDS_REFERENCE.md
echo.