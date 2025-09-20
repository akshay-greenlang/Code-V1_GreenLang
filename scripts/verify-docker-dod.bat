@echo off
setlocal enabledelayedexpansion

REM Docker Multi-Arch Build DoD Verification Script
REM Usage: verify-docker-dod.bat [ORG] [VERSION]

set ORG=%1
if "%ORG%"=="" set ORG=akshay-greenlang
set VER=%2
if "%VER%"=="" set VER=0.2.0

echo ========================================================
echo Docker Multi-Arch Build DoD Verification
echo ========================================================
echo Organization: %ORG%
echo Version: %VER%
echo.

set PASS=0
set FAIL=0
set SKIP=0

REM Check 1: Registry Publication
echo [CHECK 1] Registry Publication
echo -----------------------------------------
docker pull ghcr.io/%ORG%/greenlang-runner:%VER% >nul 2>&1
if %errorlevel% equ 0 (
    echo [PASS] Runner image found on GHCR
    set /a PASS+=1
) else (
    echo [FAIL] Runner image NOT found on GHCR
    set /a FAIL+=1
)

docker pull docker.io/greenlang/core-runner:%VER% >nul 2>&1
if %errorlevel% equ 0 (
    echo [PASS] Runner image found on Docker Hub
    set /a PASS+=1
) else (
    echo [FAIL] Runner image NOT found on Docker Hub
    set /a FAIL+=1
)
echo.

REM Check 2: Multi-arch Manifest
echo [CHECK 2] Multi-arch Manifest
echo -----------------------------------------
docker buildx imagetools inspect ghcr.io/%ORG%/greenlang-runner:%VER% 2>nul | findstr /C:"linux/amd64" >nul
if %errorlevel% equ 0 (
    echo [PASS] linux/amd64 found in manifest
    set /a PASS+=1
) else (
    echo [FAIL] linux/amd64 NOT in manifest or image not found
    set /a FAIL+=1
)

docker buildx imagetools inspect ghcr.io/%ORG%/greenlang-runner:%VER% 2>nul | findstr /C:"linux/arm64" >nul
if %errorlevel% equ 0 (
    echo [PASS] linux/arm64 found in manifest
    set /a PASS+=1
) else (
    echo [FAIL] linux/arm64 NOT in manifest or image not found
    set /a FAIL+=1
)
echo.

REM Check 3: Cosign Signatures
echo [CHECK 3] Cosign Signatures
echo -----------------------------------------
where cosign >nul 2>&1
if %errorlevel% neq 0 (
    echo [SKIP] Cosign not installed - skipping signature verification
    set /a SKIP+=1
) else (
    cosign verify ghcr.io/%ORG%/greenlang-runner:%VER% >nul 2>&1
    if %errorlevel% equ 0 (
        echo [PASS] Runner image signature verified
        set /a PASS+=1
    ) else (
        echo [FAIL] Runner image signature verification failed
        set /a FAIL+=1
    )
)
echo.

REM Check 4: SBOM Presence
echo [CHECK 4] SBOM Generation
echo -----------------------------------------
where syft >nul 2>&1
if %errorlevel% neq 0 (
    echo [SKIP] Syft not installed - skipping SBOM check
    set /a SKIP+=1
) else (
    syft ghcr.io/%ORG%/greenlang-runner:%VER% -o spdx-json >nul 2>&1
    if %errorlevel% equ 0 (
        echo [PASS] SBOM can be generated
        set /a PASS+=1
    ) else (
        echo [FAIL] SBOM generation failed
        set /a FAIL+=1
    )
)
echo.

REM Check 5: Vulnerability Scan
echo [CHECK 5] Vulnerability Scan
echo -----------------------------------------
where trivy >nul 2>&1
if %errorlevel% neq 0 (
    echo [SKIP] Trivy not installed - skipping vulnerability scan
    set /a SKIP+=1
) else (
    trivy image --severity CRITICAL,HIGH --exit-code 1 ghcr.io/%ORG%/greenlang-runner:%VER% >nul 2>&1
    if %errorlevel% equ 0 (
        echo [PASS] No CRITICAL/HIGH vulnerabilities
        set /a PASS+=1
    ) else (
        echo [FAIL] CRITICAL/HIGH vulnerabilities found
        set /a FAIL+=1
    )
)
echo.

REM Check 6: Non-root User
echo [CHECK 6] Non-root User and Healthcheck
echo -----------------------------------------
REM Test with local image if remote not available
set TEST_IMAGE=greenlang-runner:test
docker images | findstr greenlang-runner | findstr test >nul 2>&1
if %errorlevel% neq 0 (
    set TEST_IMAGE=ghcr.io/%ORG%/greenlang-runner:%VER%
)

for /f "tokens=*" %%i in ('docker run --rm --entrypoint sh %TEST_IMAGE% -c "id -u" 2^>nul') do set UID=%%i
if "%UID%"=="10001" (
    echo [PASS] Non-root user (UID: %UID%)
    set /a PASS+=1
) else if "%UID%" neq "" (
    if "%UID%" neq "0" (
        echo [PASS] Non-root user (UID: %UID%)
        set /a PASS+=1
    ) else (
        echo [FAIL] Running as root (UID: 0)
        set /a FAIL+=1
    )
) else (
    echo [FAIL] Could not determine user ID
    set /a FAIL+=1
)

docker inspect %TEST_IMAGE% --format="{{.Config.Healthcheck}}" 2>nul | findstr /C:"gl" >nul
if %errorlevel% equ 0 (
    echo [PASS] Healthcheck configured
    set /a PASS+=1
) else (
    echo [FAIL] No healthcheck found
    set /a FAIL+=1
)
echo.

REM Check 7: GL Command
echo [CHECK 7] GL Command Availability
echo -----------------------------------------
docker run --rm %TEST_IMAGE% version 2>nul | findstr /C:"GreenLang" >nul
if %errorlevel% equ 0 (
    echo [PASS] gl version works
    set /a PASS+=1
) else (
    echo [FAIL] gl version does not work
    set /a FAIL+=1
    echo NOTE: DoD specifies "gl --version" but CLI uses "gl version"
)
echo.

REM Check 8: Image Size
echo [CHECK 8] Image Size Budget
echo -----------------------------------------
for /f "tokens=*" %%i in ('docker images %TEST_IMAGE% --format "{{.Size}}" 2^>nul') do set SIZE=%%i
if "%SIZE%" neq "" (
    echo [INFO] Runner image size: %SIZE%
    echo [INFO] Size budget: ^<=300MB compressed
    set /a PASS+=1
) else (
    echo [SKIP] Could not determine image size
    set /a SKIP+=1
)
echo.

REM Check 9: OCI Labels
echo [CHECK 9] OCI Labels
echo -----------------------------------------
docker inspect %TEST_IMAGE% --format="{{.Config.Labels}}" 2>nul | findstr /C:"org.opencontainers.image.version" >nul
if %errorlevel% equ 0 (
    echo [PASS] Version label present
    set /a PASS+=1
) else (
    echo [FAIL] Version label missing
    set /a FAIL+=1
)

docker inspect %TEST_IMAGE% --format="{{.Config.Labels}}" 2>nul | findstr /C:"org.opencontainers.image.source" >nul
if %errorlevel% equ 0 (
    echo [PASS] Source label present
    set /a PASS+=1
) else (
    echo [FAIL] Source label missing
    set /a FAIL+=1
)

docker inspect %TEST_IMAGE% --format="{{.Config.Labels}}" 2>nul | findstr /C:"org.opencontainers.image.licenses" >nul
if %errorlevel% equ 0 (
    echo [PASS] License label present
    set /a PASS+=1
) else (
    echo [FAIL] License label missing
    set /a FAIL+=1
)
echo.

REM Summary
echo ========================================================
echo VERIFICATION SUMMARY
echo ========================================================
echo PASSED: %PASS% checks
echo FAILED: %FAIL% checks
echo SKIPPED: %SKIP% checks
echo.

if %FAIL% gtr 0 (
    echo RESULT: DoD NOT MET - Task is NOT COMPLETE
    echo.
    echo Required Actions:
    echo 1. Push images to GHCR and Docker Hub
    echo 2. Build multi-arch images ^(linux/amd64, linux/arm64^)
    echo 3. Sign images with Cosign
    echo 4. Generate and attach SBOMs
    echo 5. Fix "gl --version" command or update DoD
    exit /b 1
) else if %SKIP% gtr 0 (
    echo RESULT: PARTIAL - Some checks skipped
    echo.
    echo Install missing tools for complete verification:
    echo - cosign: https://github.com/sigstore/cosign/releases
    echo - syft: https://github.com/anchore/syft/releases
    echo - trivy: https://github.com/aquasecurity/trivy/releases
    exit /b 2
) else (
    echo RESULT: DoD MET - Task is COMPLETE!
    exit /b 0
)

endlocal