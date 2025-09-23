@echo off
echo ====================================================
echo       SIMPLE PyPI UPLOAD for GreenLang v0.2.0
echo ====================================================
echo.

REM Make sure we're in the right directory
cd /d "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"

echo Current directory:
cd
echo.

echo Files to upload:
dir dist\greenlang-0.2.0* /b
echo.

echo ====================================================
echo READY TO UPLOAD TO PyPI
echo ====================================================
echo.
echo When prompted:
echo   Username: __token__
echo   Password: [Paste your PyPI token - it won't show]
echo.
pause

REM Use python -m to run twine
python -m twine upload dist\greenlang-0.2.0-py3-none-any.whl dist\greenlang-0.2.0.tar.gz

if %errorlevel% equ 0 (
    echo.
    echo ====================================================
    echo SUCCESS! GreenLang is now on PyPI!
    echo ====================================================
    echo.
    echo View at: https://pypi.org/project/greenlang/
    echo.
    echo Test installation:
    echo   pip install greenlang
    echo.
) else (
    echo.
    echo ====================================================
    echo If upload failed, try this manual command:
    echo ====================================================
    echo.
    echo cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"
    echo python -m twine upload dist\greenlang-0.2.0-py3-none-any.whl dist\greenlang-0.2.0.tar.gz
    echo.
)

pause