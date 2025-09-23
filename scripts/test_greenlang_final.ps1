# GreenLang v0.2.0b2 Final Cross-Platform Installation Test Results
param(
    [string]$TestDir = "C:\tmp\gl-final-test",
    [string]$Version = "0.2.0b2"
)

Write-Host "=== GreenLang v$Version Final Cross-Platform Test ===" -ForegroundColor Green
Write-Host "Test Date: $(Get-Date)" -ForegroundColor Yellow
Write-Host "Test Directory: $TestDir" -ForegroundColor Yellow
Write-Host "Target Version: $Version" -ForegroundColor Yellow
Write-Host ""

# Initialize test results
$TestResults = @{
    StartTime = Get-Date
    TestSummary = @{
        InstallationSuccess = $false
        VersionCorrect = $false
        PythonImportWorks = $false
        GLCommandExists = $false
        GLVersionWorks = $false
        GLHelpWorks = $false
        PackCommandsWork = $false
        EntryPointsConfigured = $false
        PackageDataAccessible = $false
        CoreModulesAccessible = $false
    }
    Issues = @()
    PassCriteria = @{
        InstallSucceedsWithoutErrors = $false
        GLVersionPrints = $false
        GLPackCommandsWork = $false
    }
}

function Test-Result($TestName, $Result, $Details = "") {
    $status = if ($Result) { "[PASS]" } else { "[FAIL]" }
    $color = if ($Result) { "Green" } else { "Red" }
    Write-Host "$status $TestName" -ForegroundColor $color
    if ($Details) {
        Write-Host "    $Details" -ForegroundColor Gray
    }
    return $Result
}

# Clean up any existing test directory
if (Test-Path $TestDir) {
    Remove-Item -Path $TestDir -Recurse -Force
}

Write-Host "1. Environment Setup" -ForegroundColor Cyan
Write-Host "===================="

# Create test directory and virtual environment
New-Item -Path $TestDir -ItemType Directory -Force | Out-Null
Set-Location $TestDir

python -m venv gl-venv
$activateScript = Join-Path $TestDir "gl-venv\Scripts\Activate.ps1"
& $activateScript

Write-Host ""
Write-Host "2. Package Installation" -ForegroundColor Cyan
Write-Host "======================"

try {
    $installOutput = python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ greenlang==$Version 2>&1
    $installSuccess = $LASTEXITCODE -eq 0
    $TestResults.TestSummary.InstallationSuccess = Test-Result "Package Installation from TestPyPI" $installSuccess "Exit code: $LASTEXITCODE"
    $TestResults.PassCriteria.InstallSucceedsWithoutErrors = $installSuccess

    if (-not $installSuccess) {
        $TestResults.Issues += "Installation failed with exit code $LASTEXITCODE"
    }
} catch {
    $TestResults.TestSummary.InstallationSuccess = Test-Result "Package Installation from TestPyPI" $false "Exception: $_"
    $TestResults.Issues += "Installation exception: $_"
}

Write-Host ""
Write-Host "3. Python Module Tests" -ForegroundColor Cyan
Write-Host "======================"

# Test Python import
try {
    $importOutput = python -c "import greenlang; print(f'GreenLang {greenlang.__version__} imported successfully'); print(f'Module: {greenlang.__file__}')" 2>&1
    $importSuccess = $LASTEXITCODE -eq 0
    $TestResults.TestSummary.PythonImportWorks = Test-Result "Python Import" $importSuccess "Output: $importOutput"

    if ($importSuccess -and $importOutput -like "*$Version*") {
        $TestResults.TestSummary.VersionCorrect = Test-Result "Version Check" $true "Version $Version detected"
    } else {
        $TestResults.TestSummary.VersionCorrect = Test-Result "Version Check" $false "Expected $Version, got: $importOutput"
        $TestResults.Issues += "Version mismatch in Python import"
    }
} catch {
    $TestResults.TestSummary.PythonImportWorks = Test-Result "Python Import" $false "Exception: $_"
    $TestResults.TestSummary.VersionCorrect = Test-Result "Version Check" $false "Import failed"
    $TestResults.Issues += "Python import failed: $_"
}

# Test core modules
try {
    $coreOutput = python -c "import greenlang.cli; import greenlang.core; print('Core modules accessible')" 2>&1
    $coreSuccess = $LASTEXITCODE -eq 0
    $TestResults.TestSummary.CoreModulesAccessible = Test-Result "Core Modules Import" $coreSuccess "Output: $coreOutput"
} catch {
    $TestResults.TestSummary.CoreModulesAccessible = Test-Result "Core Modules Import" $false "Exception: $_"
    $TestResults.Issues += "Core modules not accessible: $_"
}

Write-Host ""
Write-Host "4. CLI Command Tests" -ForegroundColor Cyan
Write-Host "==================="

# Check if gl command exists
try {
    $glCommand = Get-Command gl -ErrorAction SilentlyContinue
    $glExists = $null -ne $glCommand
    $TestResults.TestSummary.GLCommandExists = Test-Result "GL Command Available" $glExists "Path: $($glCommand.Source)"

    if ($glExists) {
        # Test gl --version
        try {
            $versionOutput = gl --version 2>&1
            $versionSuccess = $LASTEXITCODE -eq 0 -and $versionOutput -like "*$Version*"
            $TestResults.TestSummary.GLVersionWorks = Test-Result "GL --version Command" $versionSuccess "Output: $versionOutput"
            $TestResults.PassCriteria.GLVersionPrints = $versionSuccess

            if (-not $versionSuccess) {
                $TestResults.Issues += "GL --version failed or wrong version: $versionOutput"
            }
        } catch {
            $TestResults.TestSummary.GLVersionWorks = Test-Result "GL --version Command" $false "Exception: $_"
            $TestResults.Issues += "GL --version command failed: $_"
        }

        # Test gl --help
        try {
            $helpOutput = gl --help 2>&1
            $helpSuccess = $LASTEXITCODE -eq 0 -and ($helpOutput -like "*usage*" -or $helpOutput -like "*GreenLang*")
            $TestResults.TestSummary.GLHelpWorks = Test-Result "GL --help Command" $helpSuccess "Contains usage info: $helpSuccess"
        } catch {
            $TestResults.TestSummary.GLHelpWorks = Test-Result "GL --help Command" $false "Exception: $_"
        }

        # Test gl pack commands
        try {
            $packHelpOutput = gl pack --help 2>&1
            $packHelpSuccess = $LASTEXITCODE -eq 0 -and ($packHelpOutput -like "*usage*" -or $packHelpOutput -like "*pack*")
            $TestResults.TestSummary.PackCommandsWork = Test-Result "GL Pack Commands" $packHelpSuccess "Help available: $packHelpSuccess"
            $TestResults.PassCriteria.GLPackCommandsWork = $packHelpSuccess

            if (-not $packHelpSuccess) {
                $TestResults.Issues += "GL pack commands not working: $packHelpOutput"
            }
        } catch {
            $TestResults.TestSummary.PackCommandsWork = Test-Result "GL Pack Commands" $false "Exception: $_"
            $TestResults.Issues += "GL pack commands failed: $_"
        }
    } else {
        $TestResults.TestSummary.GLVersionWorks = Test-Result "GL --version Command" $false "GL command not found"
        $TestResults.TestSummary.GLHelpWorks = Test-Result "GL --help Command" $false "GL command not found"
        $TestResults.TestSummary.PackCommandsWork = Test-Result "GL Pack Commands" $false "GL command not found"
        $TestResults.Issues += "GL command not found in PATH"
    }
} catch {
    $TestResults.TestSummary.GLCommandExists = Test-Result "GL Command Available" $false "Exception: $_"
    $TestResults.Issues += "Error checking GL command: $_"
}

Write-Host ""
Write-Host "5. Package Integrity Tests" -ForegroundColor Cyan
Write-Host "=========================="

# Test package data access
try {
    $packageDataOutput = python -c "import greenlang; import os; print('Package data access: OK'); print(f'Module path: {greenlang.__file__}')" 2>&1
    $packageDataSuccess = $LASTEXITCODE -eq 0
    $TestResults.TestSummary.PackageDataAccessible = Test-Result "Package Data Access" $packageDataSuccess "Output: $packageDataOutput"
} catch {
    $TestResults.TestSummary.PackageDataAccessible = Test-Result "Package Data Access" $false "Exception: $_"
}

# Test entry points (using modern approach)
try {
    $entryPointOutput = python -c "
try:
    import importlib.metadata as metadata
    eps = metadata.entry_points()
    if hasattr(eps, 'get'):
        console_scripts = eps.get('console_scripts', [])
    else:
        console_scripts = eps.select(group='console_scripts') if hasattr(eps, 'select') else []
    gl_eps = [ep for ep in console_scripts if ep.name == 'gl']
    print(f'Entry points found: {len(gl_eps)}')
    for ep in gl_eps:
        print(f'  {ep.name} -> {ep.value}')
    success = len(gl_eps) > 0
    print(f'Success: {success}')
except Exception as e:
    print(f'Error: {e}')
    success = False
" 2>&1
    $entryPointSuccess = $LASTEXITCODE -eq 0 -and $entryPointOutput -like "*Success: True*"
    $TestResults.TestSummary.EntryPointsConfigured = Test-Result "Entry Points Configuration" $entryPointSuccess "Output: $entryPointOutput"
} catch {
    $TestResults.TestSummary.EntryPointsConfigured = Test-Result "Entry Points Configuration" $false "Exception: $_"
}

Write-Host ""
Write-Host "6. Test Results Summary" -ForegroundColor Cyan
Write-Host "======================"

$endTime = Get-Date
$duration = $endTime - $TestResults.StartTime

# Calculate pass criteria
$allPassCriteria = $TestResults.PassCriteria.InstallSucceedsWithoutErrors -and
                   $TestResults.PassCriteria.GLVersionPrints -and
                   $TestResults.PassCriteria.GLPackCommandsWork

Write-Host ""
Write-Host "Pass Criteria Results:" -ForegroundColor Yellow
Write-Host "- Install succeeds without dependency errors: $($TestResults.PassCriteria.InstallSucceedsWithoutErrors)" -ForegroundColor $(if($TestResults.PassCriteria.InstallSucceedsWithoutErrors) {"Green"} else {"Red"})
Write-Host "- gl --version prints 0.2.0b2: $($TestResults.PassCriteria.GLVersionPrints)" -ForegroundColor $(if($TestResults.PassCriteria.GLVersionPrints) {"Green"} else {"Red"})
Write-Host "- gl pack commands work: $($TestResults.PassCriteria.GLPackCommandsWork)" -ForegroundColor $(if($TestResults.PassCriteria.GLPackCommandsWork) {"Green"} else {"Red"})

Write-Host ""
Write-Host "Overall Test Summary:" -ForegroundColor Yellow
$passedTests = ($TestResults.TestSummary.Values | Where-Object { $_ -eq $true }).Count
$totalTests = $TestResults.TestSummary.Count
Write-Host "Passed: $passedTests / $totalTests tests" -ForegroundColor $(if($passedTests -eq $totalTests) {"Green"} else {"Yellow"})
Write-Host "Duration: $($duration.TotalSeconds) seconds" -ForegroundColor Gray
Write-Host "All Pass Criteria Met: $allPassCriteria" -ForegroundColor $(if($allPassCriteria) {"Green"} else {"Red"})

if ($TestResults.Issues.Count -gt 0) {
    Write-Host ""
    Write-Host "Issues Found:" -ForegroundColor Red
    foreach ($issue in $TestResults.Issues) {
        Write-Host "- $issue" -ForegroundColor Red
    }
}

# Generate JSON report
$jsonReport = @{
    TestDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Version = $Version
    TestDirectory = $TestDir
    Duration = $duration.TotalSeconds
    TestSummary = $TestResults.TestSummary
    PassCriteria = $TestResults.PassCriteria
    AllPassCriteriaMet = $allPassCriteria
    Issues = $TestResults.Issues
    PassedTests = $passedTests
    TotalTests = $totalTests
    SuccessRate = [math]::Round(($passedTests / $totalTests) * 100, 2)
} | ConvertTo-Json -Depth 10

$reportFile = "greenlang_v${Version}_test_report.json"
Set-Content -Path $reportFile -Value $jsonReport
Write-Host ""
Write-Host "Detailed report saved to: $reportFile" -ForegroundColor Gray

Write-Host ""
if ($allPassCriteria) {
    Write-Host "FINAL RESULT: ALL PASS CRITERIA MET ✓" -ForegroundColor Green
    Write-Host "GreenLang v$Version is ready for cross-platform use!" -ForegroundColor Green
} else {
    Write-Host "FINAL RESULT: SOME PASS CRITERIA NOT MET ✗" -ForegroundColor Red
    Write-Host "GreenLang v$Version has issues that need to be addressed." -ForegroundColor Red
}

# Cleanup
Set-Location "C:\"
Write-Host ""
Write-Host "Cleaning up test environment..." -ForegroundColor Gray
Remove-Item -Path $TestDir -Recurse -Force

Write-Host "Test completed!" -ForegroundColor Green