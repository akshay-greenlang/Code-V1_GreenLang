@echo off
REM GreenLang Agent Factory CLI - Windows Test Script

echo.
echo ============================================================
echo  GreenLang Agent Factory CLI - Running Tests
echo ============================================================
echo.

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found
    echo Please run install.bat first
    pause
    exit /b 1
)

REM Install development dependencies
echo Installing development dependencies...
pip install -e ".[dev]"
echo.

REM Run tests
echo Running tests...
pytest tests/ -v --cov=cli --cov-report=term-missing --cov-report=html
echo.

REM Run linting
echo Running linters...
echo.
echo --- Black (formatting check) ---
black --check cli/
echo.

echo --- Ruff (linting) ---
ruff check cli/
echo.

echo --- MyPy (type checking) ---
mypy cli/
echo.

echo ============================================================
echo  Tests Complete!
echo ============================================================
echo.
echo Coverage report: htmlcov\index.html
echo.

pause
