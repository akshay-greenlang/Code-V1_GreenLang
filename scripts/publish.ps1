# PowerShell script for publishing GreenLang to PyPI (Windows)
# Usage: .\scripts\publish.ps1 [testpypi|pypi]

param(
    [Parameter(Position=0)]
    [ValidateSet("testpypi", "pypi")]
    [string]$Target = "testpypi"
)

$ErrorActionPreference = "Stop"

# Configuration
$ProjectName = "greenlang-cli"
$Version = "0.3.0"

Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "   GreenLang PyPI Publishing Script (Windows)" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green

# Check Python version
Write-Host "[INFO] Checking Python version..." -ForegroundColor Green
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

if ([float]$pythonVersion -lt 3.11) {
    Write-Host "[ERROR] Python 3.11 or higher is required (found $pythonVersion)" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Python $pythonVersion detected ✓" -ForegroundColor Green

# Check if we're in the project root
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "[ERROR] pyproject.toml not found. Please run this script from the project root." -ForegroundColor Red
    exit 1
}

# Set target URLs
if ($Target -eq "testpypi") {
    $RepoUrl = "https://test.pypi.org/legacy/"
    $ViewUrl = "https://test.pypi.org/project/$ProjectName/"
    $InstallCmd = "pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ $ProjectName"
    Write-Host "[INFO] Publishing to TEST PyPI" -ForegroundColor Green
} else {
    $RepoUrl = "https://upload.pypi.org/legacy/"
    $ViewUrl = "https://pypi.org/project/$ProjectName/"
    $InstallCmd = "pip install $ProjectName"
    Write-Host "[WARN] Publishing to PRODUCTION PyPI" -ForegroundColor Yellow
}

# Install build tools
Write-Host "[INFO] Installing build tools..." -ForegroundColor Green
python -m pip install --upgrade pip build twine wheel setuptools check-wheel-contents

# Clean previous builds
Write-Host "[INFO] Cleaning previous builds..." -ForegroundColor Green
Remove-Item -Path "build", "dist", "*.egg-info" -Recurse -Force -ErrorAction SilentlyContinue

# Extract version from pyproject.toml
Write-Host "[INFO] Extracting version from pyproject.toml..." -ForegroundColor Green
$ActualVersion = python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
Write-Host "[INFO] Building version: $ActualVersion" -ForegroundColor Green

# Build the package
Write-Host "[INFO] Building distribution packages..." -ForegroundColor Green
python -m build

# List built files
Write-Host "[INFO] Built packages:" -ForegroundColor Green
Get-ChildItem dist

# Check the wheel contents
Write-Host "[INFO] Checking wheel contents..." -ForegroundColor Green
$wheelFile = Get-ChildItem dist/*.whl | Select-Object -First 1
check-wheel-contents $wheelFile.FullName

# Run twine check
Write-Host "[INFO] Running twine check..." -ForegroundColor Green
python -m twine check dist/*

# Generate SHA256 checksums
Write-Host "[INFO] Generating SHA256 checksums..." -ForegroundColor Green
Push-Location dist
Get-ChildItem | ForEach-Object {
    $hash = Get-FileHash $_.Name -Algorithm SHA256
    "$($hash.Hash.ToLower())  $($_.Name)" | Out-File -Append SHA256SUMS
}
Get-Content SHA256SUMS
Pop-Location

# Check for credentials
Write-Host "[INFO] Checking for PyPI credentials..." -ForegroundColor Green

if ($Target -eq "testpypi") {
    if (-not $env:TESTPYPI_API_TOKEN) {
        Write-Host "[WARN] TESTPYPI_API_TOKEN environment variable not set" -ForegroundColor Yellow
        Write-Host "[INFO] You can set it with: `$env:TESTPYPI_API_TOKEN='pypi-...'" -ForegroundColor Green
    }
} else {
    if (-not $env:PYPI_API_TOKEN) {
        Write-Host "[WARN] PYPI_API_TOKEN environment variable not set" -ForegroundColor Yellow
        Write-Host "[INFO] You can set it with: `$env:PYPI_API_TOKEN='pypi-...'" -ForegroundColor Green
    }
}

# Confirm before upload
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host "Ready to upload to $Target" -ForegroundColor Yellow
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Yellow
Write-Host "Package: $ProjectName"
Write-Host "Version: $ActualVersion"
Write-Host "Target: $RepoUrl"
Write-Host ""

$confirm = Read-Host "Continue with upload? (y/N)"
if ($confirm -ne 'y') {
    Write-Host "[INFO] Upload cancelled" -ForegroundColor Green
    exit 0
}

# Upload to PyPI/TestPyPI
Write-Host "[INFO] Uploading to $Target..." -ForegroundColor Green

try {
    if ($Target -eq "testpypi" -and $env:TESTPYPI_API_TOKEN) {
        python -m twine upload --repository-url $RepoUrl `
            --username __token__ `
            --password $env:TESTPYPI_API_TOKEN `
            --verbose `
            dist/*
    } elseif ($Target -eq "pypi" -and $env:PYPI_API_TOKEN) {
        python -m twine upload --repository-url $RepoUrl `
            --username __token__ `
            --password $env:PYPI_API_TOKEN `
            --verbose `
            dist/*
    } else {
        # Fall back to interactive authentication
        python -m twine upload --repository-url $RepoUrl `
            --verbose `
            dist/*
    }

    Write-Host "[INFO] Upload successful! ✓" -ForegroundColor Green
    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "   Package published successfully!" -ForegroundColor Green
    Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "View at: $ViewUrl"
    Write-Host ""
    Write-Host "Install with:"
    Write-Host "  $InstallCmd"
    Write-Host ""

    if ($Target -eq "testpypi") {
        Write-Host "After testing, publish to production PyPI with:"
        Write-Host "  .\scripts\publish.ps1 pypi"
    }
} catch {
    Write-Host "[ERROR] Upload failed: $_" -ForegroundColor Red
    exit 1
}

# Test installation (optional)
Write-Host ""
$testInstall = Read-Host "Test installation from $Target? (y/N)"

if ($testInstall -eq 'y') {
    Write-Host "[INFO] Waiting 60 seconds for package to be available..." -ForegroundColor Green
    Start-Sleep -Seconds 60

    Write-Host "[INFO] Creating test virtual environment..." -ForegroundColor Green
    python -m venv test_env
    & test_env\Scripts\Activate.ps1

    Write-Host "[INFO] Installing from $Target..." -ForegroundColor Green
    Invoke-Expression $InstallCmd

    Write-Host "[INFO] Testing import..." -ForegroundColor Green
    python -c "import greenlang; print(f'Successfully imported greenlang {greenlang.__version__}')"

    Write-Host "[INFO] Testing CLI..." -ForegroundColor Green
    gl --version
    greenlang --version

    deactivate
    Remove-Item -Path test_env -Recurse -Force

    Write-Host "[INFO] Installation test successful! ✓" -ForegroundColor Green
}

Write-Host "`nPublishing complete!" -ForegroundColor Green