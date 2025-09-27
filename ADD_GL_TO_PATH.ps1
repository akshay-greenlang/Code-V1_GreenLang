# PowerShell script to add Python Scripts directory to User PATH permanently
# Run this script as: powershell -ExecutionPolicy Bypass -File ADD_GL_TO_PATH.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "GreenLang CLI (gl) - PATH Setup Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$pythonScriptsPath = "C:\Users\rshar\AppData\Roaming\Python\Python313\Scripts"
$glExePath = Join-Path $pythonScriptsPath "gl.exe"

# Check if gl.exe exists
if (Test-Path $glExePath) {
    Write-Host "[FOUND] gl.exe located at: $glExePath" -ForegroundColor Green
} else {
    Write-Host "[ERROR] gl.exe not found at: $glExePath" -ForegroundColor Red
    Write-Host "Please install GreenLang first: pip install greenlang-cli==0.3.0" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Test gl.exe
Write-Host ""
Write-Host "Testing gl.exe..." -ForegroundColor Yellow
& $glExePath --version
Write-Host ""

# Get current user PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::User)

# Check if already in PATH
if ($currentPath -like "*$pythonScriptsPath*") {
    Write-Host "[INFO] Python Scripts directory is already in PATH" -ForegroundColor Yellow
    Write-Host "If 'gl' command still doesn't work, please:" -ForegroundColor Yellow
    Write-Host "  1. Close and reopen your command prompt" -ForegroundColor White
    Write-Host "  2. Or restart your computer" -ForegroundColor White
} else {
    Write-Host "Adding Python Scripts directory to User PATH..." -ForegroundColor Yellow

    # Add to PATH
    $newPath = $currentPath
    if ($currentPath -and $currentPath[-1] -ne ";") {
        $newPath += ";"
    }
    $newPath += $pythonScriptsPath

    try {
        [Environment]::SetEnvironmentVariable("Path", $newPath, [EnvironmentVariableTarget]::User)
        Write-Host "[SUCCESS] Added to PATH successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "IMPORTANT: You must close and reopen your command prompt" -ForegroundColor Yellow
        Write-Host "           for the changes to take effect!" -ForegroundColor Yellow
    } catch {
        Write-Host "[ERROR] Failed to update PATH: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please add manually through System Settings:" -ForegroundColor Yellow
        Write-Host "  1. Press Win + X, select 'System'" -ForegroundColor White
        Write-Host "  2. Click 'Advanced system settings'" -ForegroundColor White
        Write-Host "  3. Click 'Environment Variables'" -ForegroundColor White
        Write-Host "  4. Under 'User variables', select 'Path' and click 'Edit'" -ForegroundColor White
        Write-Host "  5. Click 'New' and add: $pythonScriptsPath" -ForegroundColor White
        Write-Host "  6. Click 'OK' on all windows" -ForegroundColor White
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now use: gl --version" -ForegroundColor Green
Write-Host "(Remember to open a NEW command prompt first!)" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit"