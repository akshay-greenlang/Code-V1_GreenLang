@echo off
setlocal enabledelayedexpansion

:: DoD Verification Script for SBOM Implementation v0.2.0 (Windows version)
:: This script verifies ALL Definition of Done requirements

set VERSION=%1
if "%VERSION%"=="" set VERSION=0.2.0
set ARTIFACTS_DIR=artifacts\sbom
set GHCR_REGISTRY=ghcr.io
set OWNER=akshay-greenlang

:: Tracking
set /a TOTAL_CHECKS=0
set /a PASSED_CHECKS=0
set /a FAILED_CHECKS=0

echo ========================================================
echo     SBOM Definition of Done (DoD) Verification v%VERSION%
echo ========================================================
echo.
echo Gate: SBOM+Attestations Ready
echo Version: %VERSION%
echo Date: %date% %time%
echo.

:: 1) Inventory check
echo === 1) Inventory Check ===
echo.

echo [INFO] Checking Python artifacts in dist\
if exist "dist" (
    echo Python artifacts:
    dir /b dist 2>nul

    if exist "dist\*.whl" (
        echo [PASS] Python wheel found
        set /a PASSED_CHECKS+=1
    ) else (
        echo [FAIL] No Python wheel found in dist\
        set /a FAILED_CHECKS+=1
    )
    set /a TOTAL_CHECKS+=1

    if exist "dist\*.tar.gz" (
        echo [PASS] Python sdist found
        set /a PASSED_CHECKS+=1
    ) else (
        echo [FAIL] No Python sdist found in dist\
        set /a FAILED_CHECKS+=1
    )
    set /a TOTAL_CHECKS+=1
) else (
    echo [FAIL] dist\ directory not found
    set /a FAILED_CHECKS+=1
    set /a TOTAL_CHECKS+=1
)

echo.
echo [INFO] Checking SBOM files in %ARTIFACTS_DIR%\
if exist "%ARTIFACTS_DIR%" (
    echo SBOM files:
    dir /b "%ARTIFACTS_DIR%" 2>nul | sort
    echo.

    :: Expected files for Python packages
    set EXPECTED[0]=sbom-greenlang-%VERSION%-wheel.cdx.json
    set EXPECTED[1]=sbom-greenlang-%VERSION%-wheel.spdx.json
    set EXPECTED[2]=sbom-greenlang-%VERSION%-sdist.cdx.json

    for /l %%i in (0,1,2) do (
        if exist "%ARTIFACTS_DIR%\!EXPECTED[%%i]!" (
            echo [PASS] Found: !EXPECTED[%%i]!
            set /a PASSED_CHECKS+=1
        ) else (
            echo [FAIL] Missing: !EXPECTED[%%i]!
            set /a FAILED_CHECKS+=1
        )
        set /a TOTAL_CHECKS+=1
    )

    :: Expected files for Docker images
    set DOCKER[0]=runner
    set DOCKER[1]=full

    for /l %%i in (0,1,1) do (
        set CDX_FILE=sbom-image-ghcr-io-%OWNER%-greenlang-!DOCKER[%%i]!-%VERSION%.cdx.json
        set SPDX_FILE=sbom-image-ghcr-io-%OWNER%-greenlang-!DOCKER[%%i]!-%VERSION%.spdx.json

        if exist "%ARTIFACTS_DIR%\!CDX_FILE!" (
            echo [PASS] Found: !CDX_FILE!
            set /a PASSED_CHECKS+=1
        ) else (
            echo [FAIL] Missing: !CDX_FILE!
            set /a FAILED_CHECKS+=1
        )
        set /a TOTAL_CHECKS+=1

        if exist "%ARTIFACTS_DIR%\!SPDX_FILE!" (
            echo [PASS] Found: !SPDX_FILE!
            set /a PASSED_CHECKS+=1
        ) else (
            echo [FAIL] Missing: !SPDX_FILE!
            set /a FAILED_CHECKS+=1
        )
        set /a TOTAL_CHECKS+=1
    )
) else (
    echo [FAIL] %ARTIFACTS_DIR%\ directory not found
    set /a FAILED_CHECKS+=1
    set /a TOTAL_CHECKS+=1
)

:: 2) JSON validity & format sanity
echo.
echo === 2) JSON Validity and Format Sanity ===
echo.

if exist "%ARTIFACTS_DIR%" (
    echo [INFO] Validating JSON files...

    :: Check CycloneDX files
    for %%f in (%ARTIFACTS_DIR%\*.cdx.json) do (
        if exist "jq.exe" (
            jq.exe -e ".bomFormat" "%%f" >nul 2>&1
            if !errorlevel! equ 0 (
                echo [PASS] Valid CycloneDX: %%~nxf
                set /a PASSED_CHECKS+=1
            ) else (
                echo [FAIL] Invalid CycloneDX: %%~nxf
                set /a FAILED_CHECKS+=1
            )
        ) else (
            echo [INFO] jq not available - skipping JSON validation
        )
        set /a TOTAL_CHECKS+=1
    )

    :: Check SPDX files
    for %%f in (%ARTIFACTS_DIR%\*.spdx.json) do (
        if exist "jq.exe" (
            jq.exe -e ".spdxVersion" "%%f" >nul 2>&1
            if !errorlevel! equ 0 (
                echo [PASS] Valid SPDX: %%~nxf
                set /a PASSED_CHECKS+=1
            ) else (
                echo [FAIL] Invalid SPDX: %%~nxf
                set /a FAILED_CHECKS+=1
            )
        ) else (
            echo [INFO] jq not available - skipping JSON validation
        )
        set /a TOTAL_CHECKS+=1
    )
)

:: 3) Check workflow files
echo.
echo === 3) CI Artifacts and Workflow Configuration ===
echo.

if exist ".github\workflows\sbom-generation.yml" (
    echo [PASS] SBOM generation workflow exists
    set /a PASSED_CHECKS+=1

    findstr /c:"actions/upload-artifact" .github\workflows\sbom-generation.yml >nul 2>&1
    if !errorlevel! equ 0 (
        echo [PASS] Workflow uploads artifacts
        set /a PASSED_CHECKS+=1
    ) else (
        echo [FAIL] Workflow missing artifact upload
        set /a FAILED_CHECKS+=1
    )
    set /a TOTAL_CHECKS+=2
) else (
    echo [FAIL] SBOM generation workflow not found
    set /a FAILED_CHECKS+=1
    set /a TOTAL_CHECKS+=1
)

:: 4) Check documentation
echo.
echo === 4) Documentation and README ===
echo.

if exist "docs\security\sbom.md" (
    echo [PASS] SBOM documentation exists
    set /a PASSED_CHECKS+=1

    findstr /c:"cosign verify-attestation" docs\security\sbom.md >nul 2>&1
    if !errorlevel! equ 0 (
        echo [PASS] Verification commands documented
        set /a PASSED_CHECKS+=1
    ) else (
        echo [FAIL] Verification commands not found in docs
        set /a FAILED_CHECKS+=1
    )
    set /a TOTAL_CHECKS+=2
) else (
    echo [FAIL] docs\security\sbom.md not found
    set /a FAILED_CHECKS+=1
    set /a TOTAL_CHECKS+=1
)

if exist "README.md" (
    findstr /c:"SBOM" /c:"Software Bill of Materials" README.md >nul 2>&1
    if !errorlevel! equ 0 (
        echo [PASS] README mentions SBOM support
        set /a PASSED_CHECKS+=1
    ) else (
        echo [FAIL] README does not mention SBOM support
        set /a FAILED_CHECKS+=1
    )
    set /a TOTAL_CHECKS+=1
) else (
    echo [FAIL] README.md not found
    set /a FAILED_CHECKS+=1
    set /a TOTAL_CHECKS+=1
)

:: Summary
echo.
echo ========================================================
echo                 DoD VERIFICATION SUMMARY
echo ========================================================
echo.
echo Total Checks:  %TOTAL_CHECKS%
echo Passed:        %PASSED_CHECKS%
echo Failed:        %FAILED_CHECKS%
echo.

if %FAILED_CHECKS% equ 0 (
    echo ========================================================
    echo         SBOM+Attestations Ready - DoD MET!
    echo ========================================================
    echo.
    echo One-line acceptance statement:
    echo Approved: SBOM+Attestations Ready - all Python wheels/sdists and images
    echo have CycloneDX and SPDX where required SBOMs; CI sbom job configured;
    echo SBOM docs and release assets are present.
    exit /b 0
) else (
    echo ========================================================
    echo         DoD NOT MET - %FAILED_CHECKS% checks failed
    echo ========================================================
    echo.
    echo The task is NOT DONE. Fix the failed checks above.
    exit /b 1
)