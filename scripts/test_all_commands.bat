@echo off
echo ========================================
echo Testing All GreenLang Commands
echo ========================================
echo.

echo 1. Testing version command:
echo --------------------------
call gl --version
echo.

echo 2. Testing help command:
echo ------------------------
call gl --help
echo.

echo 3. Testing agents list:
echo -----------------------
call gl agents
echo.

echo 4. Testing agent details:
echo -------------------------
call gl agent fuel
echo.

echo 5. Testing benchmark:
echo ---------------------
call gl benchmark --type commercial_office --country US
echo.

echo 6. Testing init:
echo ----------------
call gl init --output test_workflow.yaml
echo.

echo ========================================
echo All command tests complete!
echo ========================================
echo.
echo CORRECT USAGE:
echo --------------
echo gl --version                     (Show version)
echo gl --help                        (Show help)
echo gl calc                          (Simple calculator)
echo gl calc --building               (Building calculator)
echo gl agents                        (List all agents)
echo gl agent [agent_id]              (Show agent details)
echo gl analyze [file]                (Analyze building file)
echo gl benchmark --type [type] --country [country]
echo gl recommend                     (Get recommendations)
echo gl ask "question"                (Ask AI assistant)
echo gl run [workflow_file]           (Run workflow)
echo gl init                          (Initialize project)
echo gl dev                          (Developer interface)