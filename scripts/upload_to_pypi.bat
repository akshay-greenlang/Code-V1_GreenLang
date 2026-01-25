@echo off
echo ========================================
echo Uploading GreenLang v0.2.0 to PyPI
echo ========================================
echo.

REM Step 1: Install/upgrade twine
echo Step 1: Ensuring twine is installed...
pip install --upgrade twine
echo.

REM Step 2: Check the distribution files
echo Step 2: Checking distribution files...
twine check dist\greenlang-0.2.0*
echo.

REM Step 3: Display files to be uploaded
echo Step 3: Files to be uploaded:
dir dist\greenlang-0.2.0*
echo.

REM Step 4: Confirm before upload
echo ========================================
echo READY TO UPLOAD TO PRODUCTION PyPI
echo ========================================
echo.
echo This will upload:
echo - greenlang-0.2.0-py3-none-any.whl
echo - greenlang-0.2.0.tar.gz
echo.
echo You will need to enter:
echo   Username: __token__
echo   Password: pypi-[YOUR-API-TOKEN]
echo.
echo To get an API token:
echo 1. Go to https://pypi.org/manage/account/token/
echo 2. Create a new API token
echo 3. Copy the token (starts with pypi-)
echo.
set /p confirm="Type 'yes' to continue with upload: "
if /i not "%confirm%"=="yes" (
    echo Upload cancelled.
    exit /b 1
)

REM Step 5: Upload to PyPI
echo.
echo Step 5: Uploading to PyPI...
echo When prompted:
echo   Username: __token__
echo   Password: [paste your PyPI API token]
echo.
twine upload dist\greenlang-0.2.0*

REM Step 6: Check result
if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo SUCCESS! GreenLang v0.2.0 is now on PyPI!
    echo ========================================
    echo.
    echo Package URL: https://pypi.org/project/greenlang/0.2.0/
    echo.
    echo Users can now install with:
    echo   pip install greenlang
    echo   pip install greenlang[analytics]
    echo.
    echo Next steps:
    echo 1. Test: pip install greenlang==0.2.0
    echo 2. Create GitHub release for v0.2.0
    echo 3. Announce general availability
    echo.
) else (
    echo.
    echo ========================================
    echo Upload failed. Please check the error above.
    echo ========================================
    echo.
    echo Common issues:
    echo - Wrong API token
    echo - Network connection
    echo - Package name already taken
    echo - Version already exists
    echo.
)

pause