@echo off
echo ========================================
echo Testing GreenLang v0.2.0b2 from TestPyPI
echo ========================================
echo.

REM Clean up old test environment
if exist .venv-testpypi-test (
    echo Removing old test environment...
    rmdir /s /q .venv-testpypi-test
)

REM Create clean test environment
echo Creating clean virtual environment...
python -m venv .venv-testpypi-test
call .venv-testpypi-test\Scripts\activate

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install from TestPyPI
echo.
echo Installing greenlang==0.2.0b2 from TestPyPI...
pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple greenlang==0.2.0b2

REM Test installation
echo.
echo ========================================
echo Testing Installation...
echo ========================================
echo.

echo 1. Testing gl command version:
gl --version

echo.
echo 2. Testing gl help:
gl --help

echo.
echo 3. Testing Python import:
python -c "import greenlang; print('Python import OK, version:', greenlang.__version__)"

echo.
echo 4. Testing gl doctor command:
gl doctor

echo.
echo ========================================
echo Test Results:
echo ========================================
echo If all commands above worked without errors, the beta is ready!
echo.
echo Next steps:
echo 1. Share with beta testers
echo 2. Work on test coverage (currently 9.43%%, target 40%%)
echo 3. Prepare for final v0.2.0 release
echo.
pause