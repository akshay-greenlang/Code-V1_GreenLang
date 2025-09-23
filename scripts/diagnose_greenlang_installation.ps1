# Simple diagnostic script for GreenLang installation
param(
    [string]$TestDir = "C:\tmp\gl-diagnose",
    [string]$Version = "0.2.0b2"
)

Write-Host "=== GreenLang v$Version Installation Diagnostics ===" -ForegroundColor Green
Write-Host "Test directory: $TestDir" -ForegroundColor Yellow

# Clean up any existing test directory
if (Test-Path $TestDir) {
    Remove-Item -Path $TestDir -Recurse -Force
}

# Create test directory
New-Item -Path $TestDir -ItemType Directory -Force | Out-Null
Set-Location $TestDir

Write-Host "`n1. Creating virtual environment..." -ForegroundColor Cyan
python -m venv gl-venv

Write-Host "`n2. Activating virtual environment..." -ForegroundColor Cyan
$activateScript = Join-Path $TestDir "gl-venv\Scripts\Activate.ps1"
& $activateScript

Write-Host "`n3. Installing GreenLang from TestPyPI..." -ForegroundColor Cyan
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ greenlang==$Version

Write-Host "`n4. Checking Python import..." -ForegroundColor Cyan
python -c "import greenlang; print(f'GreenLang version: {greenlang.__version__}'); print(f'Module location: {greenlang.__file__}')"

Write-Host "`n5. Testing python -m greenlang..." -ForegroundColor Cyan
python -m greenlang --version
python -m greenlang --help | Select-String -Pattern "usage"

Write-Host "`n6. Checking entry points..." -ForegroundColor Cyan
python -c "
try:
    import importlib.metadata as metadata
    eps = metadata.entry_points()
    console_scripts = eps.get('console_scripts', [])
    gl_eps = [ep for ep in console_scripts if ep.name == 'gl']
    print(f'Console scripts entry points found: {len(gl_eps)}')
    for ep in gl_eps:
        print(f'  {ep.name} -> {ep.value}')

    # Also check all entry points for greenlang
    all_gl_eps = [ep for group in eps.values() for ep in group if 'greenlang' in str(ep.value)]
    print(f'All greenlang-related entry points: {len(all_gl_eps)}')
    for ep in all_gl_eps:
        print(f'  {ep.name} -> {ep.value}')
except Exception as e:
    print(f'Entry point check failed: {e}')
"

Write-Host "`n7. Checking if gl command exists in PATH..." -ForegroundColor Cyan
try {
    $glPath = Get-Command gl -ErrorAction SilentlyContinue
    if ($glPath) {
        Write-Host "GL command found at: $($glPath.Source)" -ForegroundColor Green
        gl --version
    } else {
        Write-Host "GL command NOT found in PATH" -ForegroundColor Red
        Write-Host "Checking Scripts directory..." -ForegroundColor Yellow
        Get-ChildItem "$TestDir\gl-venv\Scripts\" | Where-Object { $_.Name -like "*gl*" }
    }
} catch {
    Write-Host "Error checking gl command: $_" -ForegroundColor Red
}

Write-Host "`n8. Checking installed packages..." -ForegroundColor Cyan
python -m pip list | Select-String -Pattern "greenlang"

Write-Host "`n9. Final verification..." -ForegroundColor Cyan
Write-Host "Installation Summary:" -ForegroundColor Green
Write-Host "- Package installed: $(if (python -c 'import greenlang' 2>&1) { 'NO' } else { 'YES' })" -ForegroundColor $(if (python -c 'import greenlang' 2>&1) { 'Red' } else { 'Green' })
Write-Host "- Python module works: $(if (python -m greenlang --version 2>&1) { 'NO' } else { 'YES' })" -ForegroundColor $(if (python -m greenlang --version 2>&1) { 'Red' } else { 'Green' })
$hasGL = try { Get-Command gl -ErrorAction SilentlyContinue } catch { $null }
Write-Host "- GL command available: $(if ($hasGL) { 'YES' } else { 'NO' })" -ForegroundColor $(if ($hasGL) { 'Green' } else { 'Red' })

Write-Host "`nDiagnostics complete!" -ForegroundColor Green

# Cleanup
Set-Location "C:\"
Write-Host "`nCleaning up test directory..." -ForegroundColor Yellow
Remove-Item -Path $TestDir -Recurse -Force