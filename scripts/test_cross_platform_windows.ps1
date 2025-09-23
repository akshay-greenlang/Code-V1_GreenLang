# GreenLang v0.2.0b2 Cross-Platform Installation and Smoke Tests for Windows
# This script tests installation from TestPyPI and validates core functionality

param(
    [string]$TestDir = "C:\tmp\gl-testpypi",
    [string]$Version = "0.2.0b2",
    [string]$LogFile = "$TestDir\greenlang_test_results.log"
)

# Initialize test results
$TestResults = @{
    StartTime = Get-Date
    TotalTests = 0
    PassedTests = 0
    FailedTests = 0
    Results = @()
}

function Write-TestLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $logMessage
    Add-Content -Path $LogFile -Value $logMessage
}

function Test-Command {
    param(
        [string]$TestName,
        [string]$Command,
        [string]$ExpectedOutput = "",
        [string]$ExpectedPattern = "",
        [bool]$ShouldSucceed = $true
    )

    $TestResults.TotalTests++
    Write-TestLog "Running test: $TestName"
    Write-TestLog "Command: $Command"

    try {
        $output = Invoke-Expression $Command 2>&1
        $exitCode = $LASTEXITCODE

        Write-TestLog "Exit Code: $exitCode"
        Write-TestLog "Output: $output"

        $testPassed = $false

        if ($ShouldSucceed) {
            if ($exitCode -eq 0) {
                if ($ExpectedOutput -and $output -notlike "*$ExpectedOutput*") {
                    Write-TestLog "FAIL: Expected output '$ExpectedOutput' not found" "ERROR"
                } elseif ($ExpectedPattern -and $output -notmatch $ExpectedPattern) {
                    Write-TestLog "FAIL: Output doesn't match pattern '$ExpectedPattern'" "ERROR"
                } else {
                    Write-TestLog "PASS: Command succeeded as expected" "SUCCESS"
                    $testPassed = $true
                }
            } else {
                Write-TestLog "FAIL: Command failed with exit code $exitCode" "ERROR"
            }
        } else {
            if ($exitCode -ne 0) {
                Write-TestLog "PASS: Command failed as expected" "SUCCESS"
                $testPassed = $true
            } else {
                Write-TestLog "FAIL: Command succeeded but should have failed" "ERROR"
            }
        }

        if ($testPassed) {
            $TestResults.PassedTests++
        } else {
            $TestResults.FailedTests++
        }

        $TestResults.Results += @{
            Name = $TestName
            Command = $Command
            ExitCode = $exitCode
            Output = $output
            Passed = $testPassed
        }

    } catch {
        Write-TestLog "FAIL: Exception occurred: $_" "ERROR"
        $TestResults.FailedTests++
        $TestResults.Results += @{
            Name = $TestName
            Command = $Command
            ExitCode = -1
            Output = $_.Exception.Message
            Passed = $false
        }
    }

    Write-TestLog "----------------------------------------"
}

function Cleanup-TestEnvironment {
    Write-TestLog "Cleaning up test environment..."
    if (Test-Path $TestDir) {
        try {
            Remove-Item -Path $TestDir -Recurse -Force
            Write-TestLog "Removed test directory: $TestDir"
        } catch {
            Write-TestLog "Warning: Could not remove test directory: $_" "WARN"
        }
    }
}

function Generate-TestReport {
    $endTime = Get-Date
    $duration = $endTime - $TestResults.StartTime

    Write-TestLog "========================================="
    Write-TestLog "GREENLANG v$Version TEST REPORT"
    Write-TestLog "========================================="
    Write-TestLog "Start Time: $($TestResults.StartTime)"
    Write-TestLog "End Time: $endTime"
    Write-TestLog "Duration: $($duration.TotalSeconds) seconds"
    Write-TestLog "Total Tests: $($TestResults.TotalTests)"
    Write-TestLog "Passed: $($TestResults.PassedTests)"
    Write-TestLog "Failed: $($TestResults.FailedTests)"
    Write-TestLog "Success Rate: $(if ($TestResults.TotalTests -gt 0) { [math]::Round(($TestResults.PassedTests / $TestResults.TotalTests) * 100, 2) } else { 0 })%"
    Write-TestLog "========================================="

    if ($TestResults.FailedTests -gt 0) {
        Write-TestLog "FAILED TESTS:" "ERROR"
        foreach ($result in $TestResults.Results) {
            if (-not $result.Passed) {
                Write-TestLog "- $($result.Name): $($result.Command)" "ERROR"
            }
        }
    }

    # Generate JSON report
    $jsonReport = $TestResults | ConvertTo-Json -Depth 10
    $jsonFile = "greenlang_test_results.json"
    Set-Content -Path $jsonFile -Value $jsonReport
    Write-TestLog "Detailed results saved to: $jsonFile"

    return $TestResults.FailedTests -eq 0
}

# Main test execution
Write-TestLog "Starting GreenLang v$Version cross-platform installation and smoke tests"
Write-TestLog "Test directory: $TestDir"
Write-TestLog "Python version check..."

# Check Python availability
Test-Command -TestName "Python Availability Check" -Command "python --version" -ShouldSucceed $true

# Clean up any existing test directory
Cleanup-TestEnvironment

# Create test directory
Write-TestLog "Creating test directory: $TestDir"
try {
    New-Item -Path $TestDir -ItemType Directory -Force | Out-Null
    Set-Location $TestDir
    Write-TestLog "Successfully created and changed to test directory"
} catch {
    Write-TestLog "FATAL: Cannot create test directory: $_" "ERROR"
    exit 1
}

# Create virtual environment
Test-Command -TestName "Virtual Environment Creation" -Command "python -m venv gl-venv" -ShouldSucceed $true

# Activate virtual environment
$activateScript = Join-Path $TestDir "gl-venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-TestLog "Activating virtual environment..."
    & $activateScript
} else {
    Write-TestLog "FATAL: Virtual environment activation script not found" "ERROR"
    exit 1
}

# Upgrade pip
Test-Command -TestName "Pip Upgrade" -Command "python -m pip install --upgrade pip" -ShouldSucceed $true

# Install from TestPyPI
Write-TestLog "Installing GreenLang v$Version from TestPyPI..."
$installCmd = "python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ greenlang==$Version"
Test-Command -TestName "GreenLang Installation from TestPyPI" -Command $installCmd -ShouldSucceed $true

# Test gl command availability first
Write-TestLog "Checking gl command availability..."
try {
    $glPath = Get-Command gl -ErrorAction SilentlyContinue
    if ($glPath) {
        Write-TestLog "GL command found at: $($glPath.Source)"
        # Test gl --version
        Test-Command -TestName "GL Version Check" -Command "gl --version" -ExpectedOutput $Version -ShouldSucceed $true
        # Test gl --help
        Test-Command -TestName "GL Help Command" -Command "gl --help" -ExpectedPattern "usage:" -ShouldSucceed $true
    } else {
        Write-TestLog "GL command not found in PATH, trying python -m greenlang..."
        Test-Command -TestName "GL Version Check (via python -m)" -Command "python -m greenlang --version" -ExpectedOutput $Version -ShouldSucceed $true
        Test-Command -TestName "GL Help Command (via python -m)" -Command "python -m greenlang --help" -ExpectedPattern "usage:" -ShouldSucceed $true
    }
} catch {
    Write-TestLog "Error checking gl command: $_" "ERROR"
    # Fallback to python -m approach
    Test-Command -TestName "GL Version Check (fallback)" -Command "python -m greenlang --version" -ExpectedOutput $Version -ShouldSucceed $true
    Test-Command -TestName "GL Help Command (fallback)" -Command "python -m greenlang --help" -ExpectedPattern "usage:" -ShouldSucceed $true
}

# Test Python import
$importTest = "python -c `"import greenlang; print('GreenLang import successful')`""
Test-Command -TestName "Python Import Check" -Command $importTest -ExpectedOutput "GreenLang import successful" -ShouldSucceed $true

# Test gl pack commands (if available)
Write-TestLog "Testing GL pack commands..."

# Use the same approach as above for pack commands
try {
    $glPath = Get-Command gl -ErrorAction SilentlyContinue
    if ($glPath) {
        # Check if gl pack command exists
        Test-Command -TestName "GL Pack Help" -Command "gl pack --help" -ShouldSucceed $true
        # Test gl pack create (if command exists)
        $packCreateCmd = "gl pack create smoke-pack"
        Test-Command -TestName "GL Pack Create" -Command $packCreateCmd -ShouldSucceed $true
        # Test gl pack list (if command exists)
        Test-Command -TestName "GL Pack List" -Command "gl pack list" -ShouldSucceed $true
    } else {
        Write-TestLog "Using python -m greenlang for pack commands..."
        Test-Command -TestName "GL Pack Help (via python -m)" -Command "python -m greenlang pack --help" -ShouldSucceed $true
        Test-Command -TestName "GL Pack Create (via python -m)" -Command "python -m greenlang pack create smoke-pack" -ShouldSucceed $true
        Test-Command -TestName "GL Pack List (via python -m)" -Command "python -m greenlang pack list" -ShouldSucceed $true
    }
} catch {
    Write-TestLog "Error with pack commands, using fallback: $_" "ERROR"
    Test-Command -TestName "GL Pack Help (fallback)" -Command "python -m greenlang pack --help" -ShouldSucceed $true
}

# Test package data access
$packageDataTest = "python -c `"import greenlang; import os; print('Package data access test'); print(f'GreenLang module path: {greenlang.__file__}')`""
Test-Command -TestName "Package Data Access" -Command $packageDataTest -ShouldSucceed $true

# Test entry points configuration
$entryPointTest = "python -c `"try: import importlib.metadata as metadata; except ImportError: import importlib_metadata as metadata; eps = metadata.entry_points(); gl_eps = [ep for ep in eps.get('console_scripts', []) if ep.name == 'gl']; print(f'Entry points found: {len(gl_eps)}'); [print(f'  {ep.name} -> {ep.value}') for ep in gl_eps]; except Exception as e: print(f'Entry point check failed: {e}')`""
Test-Command -TestName "Entry Points Check" -Command $entryPointTest -ShouldSucceed $true

# Test CLI functionality with basic commands
try {
    $glPath = Get-Command gl -ErrorAction SilentlyContinue
    if ($glPath) {
        Test-Command -TestName "GL Status Command" -Command "gl status" -ShouldSucceed $true
    } else {
        Test-Command -TestName "GL Status Command (via python -m)" -Command "python -m greenlang status" -ShouldSucceed $true
    }
} catch {
    Write-TestLog "Error with status command, using fallback: $_" "ERROR"
    Test-Command -TestName "GL Status Command (fallback)" -Command "python -m greenlang status" -ShouldSucceed $true
}

# Validate installation completeness
Write-TestLog "Validating installation completeness..."
$validateCmd = "python -c `"import greenlang.cli; import greenlang.core; print('Core modules accessible')`""
Test-Command -TestName "Core Modules Validation" -Command $validateCmd -ExpectedOutput "Core modules accessible" -ShouldSucceed $true

# Generate final report
Write-TestLog "Generating test report..."
$success = Generate-TestReport

# Cleanup
Write-TestLog "Cleaning up test environment..."
Set-Location "C:\"
Cleanup-TestEnvironment

if ($success) {
    Write-TestLog "ALL TESTS PASSED! GreenLang v$Version installation and smoke tests completed successfully." "SUCCESS"
    exit 0
} else {
    Write-TestLog "SOME TESTS FAILED! Please review the test results above." "ERROR"
    exit 1
}