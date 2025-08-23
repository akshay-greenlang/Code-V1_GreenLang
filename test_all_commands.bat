@echo off
echo ========================================
echo Testing All GreenLang Commands
echo ========================================
echo.

echo 1. Testing version command:
echo --------------------------
call greenlang --version
echo.

echo 2. Testing help command:
echo ------------------------
call greenlang --help
echo.

echo 3. Testing agents list:
echo -----------------------
call greenlang agents
echo.

echo 4. Testing agent details:
echo -------------------------
call greenlang agent fuel
echo.

echo 5. Testing benchmark:
echo ---------------------
call greenlang benchmark --type commercial_office --country US
echo.

echo 6. Testing init:
echo ----------------
call greenlang init --output test_workflow.yaml
echo.

echo ========================================
echo All command tests complete!
echo ========================================
echo.
echo CORRECT USAGE:
echo --------------
echo greenlang --version                     (Show version)
echo greenlang --help                        (Show help)
echo greenlang calc                          (Simple calculator)
echo greenlang calc --building               (Building calculator)
echo greenlang agents                        (List all agents)
echo greenlang agent [agent_id]              (Show agent details)
echo greenlang analyze [file]                (Analyze building file)
echo greenlang benchmark --type [type] --country [country]
echo greenlang recommend                     (Get recommendations)
echo greenlang ask "question"                (Ask AI assistant)
echo greenlang run [workflow_file]           (Run workflow)
echo greenlang init                          (Initialize project)
echo greenlang dev                          (Developer interface)