@echo off
REM CBAM Importer Copilot - Security Scanning Script
REM
REM Runs comprehensive security scans:
REM - Bandit: Python code security analysis
REM - Safety: Dependency vulnerability scanning
REM - Secrets detection
REM
REM Version: 1.0.0

echo ========================================
echo CBAM Importer Copilot - Security Scan
echo ========================================
echo.

REM Check if security tools are installed
python -c "import bandit" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing security tools...
    pip install bandit safety
)

echo.
echo [1/3] Running Bandit (Code Security Analysis)...
echo ========================================
bandit -r . -f txt -o security_report_bandit.txt -x ./tests,./venv,./.venv
if %ERRORLEVEL% EQU 0 (
    echo ✓ No high-severity issues found
) else (
    echo ⚠ Security issues detected - check security_report_bandit.txt
)

echo.
echo [2/3] Running Safety (Dependency Vulnerability Scan)...
echo ========================================
safety check --output text > security_report_safety.txt 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ No known vulnerabilities in dependencies
) else (
    echo ⚠ Vulnerable dependencies found - check security_report_safety.txt
)

echo.
echo [3/3] Checking for Hardcoded Secrets...
echo ========================================
findstr /S /I /C:"password" /C:"api_key" /C:"secret" /C:"token" *.py > nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ⚠ Potential secrets found - manual review recommended
    findstr /S /I /N /C:"password" /C:"api_key" /C:"secret" /C:"token" *.py
) else (
    echo ✓ No obvious secrets detected
)

echo.
echo ========================================
echo Security Scan Complete!
echo ========================================
echo.
echo Reports generated:
echo   - security_report_bandit.txt
echo   - security_report_safety.txt
echo.
echo Review these files for detailed findings.
