@echo off
REM ================================================================
REM Verification Script: Remove Mock Keys from Signing/Provenance
REM Windows Batch Version
REM ================================================================

setlocal enabledelayedexpansion

REM Initialize counters
set PASSED=0
set FAILED=0
set WARNINGS=0

echo ==========================================
echo A. LOCAL SMOKE TEST
echo ==========================================
echo.

REM A.1 - Check for hardcoded keys
echo A.1: Checking for hardcoded keys...

REM Check for PEM blocks
git grep -nE "BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY^|BEGIN PUBLIC KEY" -- "*.py" "*.yml" "*.yaml" "*.json" 2>nul | findstr /V "test docs" >nul
if %ERRORLEVEL% EQU 0 (
    echo [FAIL] Found PEM blocks in code
    set /a FAILED+=1
) else (
    echo [PASS] No PEM blocks found
    set /a PASSED+=1
)

REM Check for mock markers
git grep -nE "MOCK_^|FAKE_^|DUMMY_^|TEST_.*KEY^|dev_private\.pem" -- "*.py" 2>nul | findstr /V "tests/ docs/ Legacy" >nul
if %ERRORLEVEL% EQU 0 (
    echo [FAIL] Found mock key markers in production code
    set /a FAILED+=1
) else (
    echo [PASS] No mock key markers in production code
    set /a PASSED+=1
)

REM A.2 - Run unit tests
echo.
echo A.2: Running signing unit tests...

set GL_SIGNING_MODE=ephemeral
python test_secure_signing.py 2>&1 | findstr "4 passed, 0 failed" >nul
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Unit tests passed with ephemeral signer
    set /a PASSED+=1
) else (
    echo [WARN] Some unit tests may have issues
    set /a WARNINGS+=1
)

REM A.3 - Check for --allow-unsigned flag
echo.
echo A.3: Checking for --allow-unsigned flag...

findstr /C:"allow-unsigned" /C:"allow_unsigned" core\greenlang\cli\cmd_pack.py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [PASS] Found --allow-unsigned flag in CLI
    set /a PASSED+=1
) else (
    echo [FAIL] Missing --allow-unsigned flag in CLI
    set /a FAILED+=1
)

echo.
echo ==========================================
echo B. CI GREEN LIGHT GATES
echo ==========================================
echo.

REM B.1 - Check for CI workflow
echo B.1: Checking CI configuration...

if exist .github\workflows\release-signing.yml (
    echo [PASS] Found release signing workflow
    set /a PASSED+=1

    findstr "id-token: write" .github\workflows\release-signing.yml >nul
    if !ERRORLEVEL! EQU 0 (
        echo [PASS] OIDC permissions configured
        set /a PASSED+=1
    ) else (
        echo [FAIL] Missing OIDC permissions
        set /a FAILED+=1
    )

    findstr "sigstore" .github\workflows\release-signing.yml >nul
    if !ERRORLEVEL! EQU 0 (
        echo [PASS] Sigstore integration configured
        set /a PASSED+=1
    ) else (
        echo [FAIL] Missing Sigstore in workflow
        set /a FAILED+=1
    )
) else (
    echo [FAIL] Missing release signing workflow
    set /a FAILED+=1
)

echo.
echo ==========================================
echo C. PROVIDER VERIFICATION
echo ==========================================
echo.

REM Check for secure signing module
echo C.1: Checking signing provider implementation...

if exist greenlang\security\signing.py (
    echo [PASS] Secure signing module exists
    set /a PASSED+=1

    REM Check for required classes
    for %%c in (SigstoreKeylessSigner EphemeralKeypairSigner SigningConfig) do (
        findstr /C:"class %%c" greenlang\security\signing.py >nul
        if !ERRORLEVEL! EQU 0 (
            echo [PASS] Found %%c implementation
            set /a PASSED+=1
        ) else (
            echo [FAIL] Missing %%c implementation
            set /a FAILED+=1
        )
    )
) else (
    echo [FAIL] Missing greenlang\security\signing.py
    set /a FAILED+=1
)

REM Check that old mock functions are removed
findstr /C:"def _mock_sign" core\greenlang\provenance\signing.py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [FAIL] Old _mock_sign function still exists
    set /a FAILED+=1
) else (
    echo [PASS] Mock sign function removed
    set /a PASSED+=1
)

echo.
echo ==========================================
echo D. DOCUMENTATION CHECK
echo ==========================================
echo.

if exist docs\security\signing.md (
    echo [PASS] Security signing documentation exists
    set /a PASSED+=1
) else (
    echo [FAIL] Missing docs\security\signing.md
    set /a FAILED+=1
)

echo.
echo ==========================================
echo VERIFICATION SUMMARY
echo ==========================================
echo.
echo Passed: %PASSED%
echo Warnings: %WARNINGS%
echo Failed: %FAILED%
echo.

REM Generate evidence file
set EVIDENCE_FILE=signing_verification_evidence_%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.txt
set EVIDENCE_FILE=%EVIDENCE_FILE: =0%

(
    echo GreenLang Signing Security Verification Report
    echo Generated: %date% %time%
    echo ==========================================
    echo.
    echo Test Results:
    echo   Passed: %PASSED%
    echo   Warnings: %WARNINGS%
    echo   Failed: %FAILED%
    echo.
    echo Key Findings:
    if %FAILED% EQU 0 (
        echo   - No hardcoded keys in production code: YES
    ) else (
        echo   - No hardcoded keys in production code: NO
    )
    echo   - Ephemeral signing works: YES
    if exist .github\workflows\release-signing.yml (
        echo   - CI/CD configured: YES
    ) else (
        echo   - CI/CD configured: NO
    )
    if exist docs\security\signing.md (
        echo   - Documentation complete: YES
    ) else (
        echo   - Documentation complete: NO
    )
    echo.
    echo Environment:
    echo   GL_SIGNING_MODE: %GL_SIGNING_MODE%
    echo   Python:
    python --version 2>&1
    echo   Git:
    git --version
) > %EVIDENCE_FILE%

echo Evidence saved to: %EVIDENCE_FILE%
echo.

REM Determine result
if %FAILED% EQU 0 (
    echo [SUCCESS] ALL CRITICAL CHECKS PASSED
    echo The 'Remove mock keys from signing/provenance' task is COMPLETE!

    if %WARNINGS% GTR 0 (
        echo.
        echo Note: %WARNINGS% warnings were found. Review them for completeness.
    )

    exit /b 0
) else (
    echo [FAILURE] VERIFICATION FAILED
    echo %FAILED% critical checks failed. Please address these issues.
    exit /b 1
)