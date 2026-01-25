@echo off
echo Testing GreenLang v0.2.0b1 from TestPyPI
echo =========================================

REM Create clean test environment
echo Creating clean virtual environment...
python -m venv .venv-testpypi-test
call .venv-testpypi-test\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install from TestPyPI
echo Installing greenlang==0.2.0b1 from TestPyPI...
pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple greenlang==0.2.0b1

REM Test installation
echo.
echo Testing installation...
echo -----------------------
gl --version
echo.
gl --help
echo.
python -c "import greenlang; print('Python import OK, version:', greenlang.__version__)"

echo.
echo Test complete! If all commands worked, the beta is ready.
pause