# Simple GreenLang v0.2.0b2 Cross-Platform Test
param(
    [string]$TestDir = "C:\tmp\gl-simple-test",
    [string]$Version = "0.2.0b2"
)

Write-Host "=== GreenLang v$Version Simple Cross-Platform Test ==="
Write-Host "Test Date: $(Get-Date)"
Write-Host "Test Directory: $TestDir"
Write-Host ""

$Results = @{
    InstallSuccess = $false
    GLVersionWorks = $false
    PackCommandsWork = $false
    Issues = @()
}

# Cleanup and setup
if (Test-Path $TestDir) { Remove-Item -Path $TestDir -Recurse -Force }
New-Item -Path $TestDir -ItemType Directory -Force | Out-Null
Set-Location $TestDir

Write-Host "1. Creating virtual environment..."
python -m venv gl-venv
& "$TestDir\gl-venv\Scripts\Activate.ps1"

Write-Host "2. Installing GreenLang from TestPyPI..."
try {
    python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ greenlang==$Version | Out-Host
    if ($LASTEXITCODE -eq 0) {
        $Results.InstallSuccess = $true
        Write-Host "[PASS] Installation succeeded" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Installation failed with exit code $LASTEXITCODE" -ForegroundColor Red
        $Results.Issues += "Installation failed"
    }
} catch {
    Write-Host "[FAIL] Installation exception: $_" -ForegroundColor Red
    $Results.Issues += "Installation exception: $_"
}

Write-Host "3. Testing gl --version..."
try {
    $versionOutput = gl --version 2>&1
    if ($LASTEXITCODE -eq 0 -and $versionOutput -like "*$Version*") {
        $Results.GLVersionWorks = $true
        Write-Host "[PASS] gl --version works and shows $Version" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] gl --version failed or wrong version: $versionOutput" -ForegroundColor Red
        $Results.Issues += "GL version check failed: $versionOutput"
    }
} catch {
    Write-Host "[FAIL] gl --version exception: $_" -ForegroundColor Red
    $Results.Issues += "GL version exception: $_"
}

Write-Host "4. Testing gl pack commands..."
try {
    $packOutput = gl pack --help 2>&1
    if ($LASTEXITCODE -eq 0 -and ($packOutput -like "*usage*" -or $packOutput -like "*pack*")) {
        $Results.PackCommandsWork = $true
        Write-Host "[PASS] gl pack commands work" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] gl pack commands failed: $packOutput" -ForegroundColor Red
        $Results.Issues += "Pack commands failed: $packOutput"
    }
} catch {
    Write-Host "[FAIL] gl pack exception: $_" -ForegroundColor Red
    $Results.Issues += "Pack commands exception: $_"
}

Write-Host ""
Write-Host "=== TEST RESULTS ==="

$passCount = 0
if ($Results.InstallSuccess) { $passCount++ }
if ($Results.GLVersionWorks) { $passCount++ }
if ($Results.PackCommandsWork) { $passCount++ }

Write-Host "Pass Criteria:"
Write-Host "- Install succeeds: $($Results.InstallSuccess)" -ForegroundColor $(if($Results.InstallSuccess) {"Green"} else {"Red"})
Write-Host "- gl --version shows 0.2.0b2: $($Results.GLVersionWorks)" -ForegroundColor $(if($Results.GLVersionWorks) {"Green"} else {"Red"})
Write-Host "- gl pack commands work: $($Results.PackCommandsWork)" -ForegroundColor $(if($Results.PackCommandsWork) {"Green"} else {"Red"})

$allPass = $Results.InstallSuccess -and $Results.GLVersionWorks -and $Results.PackCommandsWork

Write-Host ""
Write-Host "Passed: $passCount / 3 tests"
Write-Host "OVERALL RESULT: $(if($allPass) {'PASS'} else {'FAIL'})" -ForegroundColor $(if($allPass) {"Green"} else {"Red"})

if ($Results.Issues.Count -gt 0) {
    Write-Host ""
    Write-Host "Issues found:"
    foreach ($issue in $Results.Issues) {
        Write-Host "- $issue" -ForegroundColor Red
    }
}

# Create report
$report = @{
    TestDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Version = $Version
    Results = $Results
    PassedTests = $passCount
    TotalTests = 3
    AllTestsPassed = $allPass
} | ConvertTo-Json -Depth 10

Set-Content -Path "greenlang_simple_test_report.json" -Value $report
Write-Host ""
Write-Host "Report saved to: greenlang_simple_test_report.json"

# Cleanup
Set-Location "C:\"
Remove-Item -Path $TestDir -Recurse -Force
Write-Host "Test completed and cleanup done."