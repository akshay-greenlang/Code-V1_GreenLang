@echo off
echo Publishing to GitHub Packages instead of PyPI
echo.

REM Create GitHub release with artifacts
gh release create v0.2.0 ^
  --title "v0.2.0 - Production Release" ^
  --notes "Install from GitHub: pip install https://github.com/akshay-greenlang/Code-V1_GreenLang/releases/download/v0.2.0/greenlang_cli-0.2.0-py3-none-any.whl" ^
  dist\greenlang_cli-0.2.0-py3-none-any.whl ^
  dist\greenlang_cli-0.2.0.tar.gz

echo.
echo Success! Users can install with:
echo pip install https://github.com/akshay-greenlang/Code-V1_GreenLang/releases/download/v0.2.0/greenlang_cli-0.2.0-py3-none-any.whl
pause