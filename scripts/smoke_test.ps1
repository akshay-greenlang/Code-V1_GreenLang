# ==============================================================================
# GreenLang Release Smoke Test Script (PowerShell)
# ==============================================================================
#
# Run after PyPI publish to verify the release works correctly on Windows.
#
# Usage:
#   .\scripts\smoke_test.ps1                     # Test latest version
#   .\scripts\smoke_test.ps1 -Version 0.3.0     # Test specific version
#   .\scripts\smoke_test.ps1 -FromTestPyPI      # Test from TestPyPI
#   .\scripts\smoke_test.ps1 -Local             # Test local installation
#
# ==============================================================================

[CmdletBinding()]
param(
    [Parameter(Position=0)]
    [string]$Version = "",

    [switch]$FromTestPyPI,

    [switch]$Local,

    [switch]$Strict,

    [switch]$KeepEnv,

    [int]$Timeout = 300
)

# Configuration
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvDir = Join-Path $env:TEMP "gl-smoke-test-$PID"
$LogFile = Join-Path $env:TEMP "gl-smoke-test-$PID.log"

# Test counters
$TestsPassed = 0
$TestsFailed = 0

# ==============================================================================
# Helper Functions
# ==============================================================================

function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$Timestamp $Message" | Tee-Object -FilePath $LogFile -Append | Write-Host
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor Blue -NoNewline
    Write-Host $Message
    Add-Content -Path $LogFile -Value "[INFO] $Message"
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host $Message
    Add-Content -Path $LogFile -Value "[OK] $Message"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
    Add-Content -Path $LogFile -Value "[WARN] $Message"
}

function Write-Error {
    param([string]$Message)
    Write-Host "[FAIL] " -ForegroundColor Red -NoNewline
    Write-Host $Message
    Add-Content -Path $LogFile -Value "[FAIL] $Message"
}

function Run-Test {
    param(
        [string]$TestName,
        [scriptblock]$TestScript
    )

    Write-Info "Testing: $TestName"

    try {
        $result = & $TestScript 2>&1
        if ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE) {
            Write-Success $TestName
            $script:TestsPassed++
            return $true
        } else {
            Write-Error "$TestName (exit code: $LASTEXITCODE)"
            $script:TestsFailed++
            if ($Strict) {
                throw "Test failed: $TestName"
            }
            return $false
        }
    } catch {
        Write-Error "$TestName - $_"
        $script:TestsFailed++
        if ($Strict) {
            throw
        }
        return $false
    }
}

function Cleanup {
    if (-not $KeepEnv -and (Test-Path $VenvDir)) {
        Write-Info "Cleaning up virtual environment..."
        Remove-Item -Recurse -Force $VenvDir -ErrorAction SilentlyContinue
    } else {
        Write-Info "Keeping virtual environment at: $VenvDir"
    }
}

# ==============================================================================
# Main Script
# ==============================================================================

try {
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "          GreenLang Release Smoke Test (Windows)"
    Write-Host "========================================================================"
    Write-Host ""

    # Get version from pyproject.toml if not specified
    if ([string]::IsNullOrEmpty($Version) -and -not $Local) {
        $pyprojectPath = Join-Path $ProjectRoot "pyproject.toml"
        if (Test-Path $pyprojectPath) {
            $content = Get-Content $pyprojectPath -Raw
            if ($content -match 'version\s*=\s*"([^"]+)"') {
                $Version = $Matches[1]
            }
        }
    }

    # Display configuration
    Write-Info "Configuration:"
    Write-Info "  Version:        $(if ($Version) { $Version } else { 'latest' })"
    Write-Info "  Source:         $(if ($FromTestPyPI) { 'TestPyPI' } elseif ($Local) { 'Local' } else { 'PyPI' })"
    Write-Info "  Strict Mode:    $Strict"
    Write-Info "  Timeout:        ${Timeout}s"
    Write-Info "  Log File:       $LogFile"
    Write-Host ""

    # Step 1: Create virtual environment
    Write-Host "========================================================================"
    Write-Host "Step 1: Creating Virtual Environment"
    Write-Host "========================================================================"

    Write-Info "Creating virtual environment at $VenvDir..."
    python -m venv $VenvDir
    if (-not $?) { throw "Failed to create virtual environment" }

    # Activate virtual environment
    $activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    . $activateScript

    Write-Success "Virtual environment created and activated"

    # Upgrade pip
    Write-Info "Upgrading pip..."
    python -m pip install --upgrade pip 2>&1 | Out-File -Append $LogFile
    Write-Success "pip upgraded"

    # Step 2: Install GreenLang
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "Step 2: Installing GreenLang"
    Write-Host "========================================================================"

    if ($Local) {
        Write-Info "Installing from local source (editable)..."
        pip install -e "$ProjectRoot[test]" 2>&1 | Out-File -Append $LogFile
        Write-Success "Local installation complete"
    } elseif ($FromTestPyPI) {
        Write-Info "Installing from TestPyPI..."
        if ($Version) {
            pip install --index-url https://test.pypi.org/simple/ `
                --extra-index-url https://pypi.org/simple/ `
                "greenlang-cli==$Version" 2>&1 | Out-File -Append $LogFile
        } else {
            pip install --index-url https://test.pypi.org/simple/ `
                --extra-index-url https://pypi.org/simple/ `
                greenlang-cli 2>&1 | Out-File -Append $LogFile
        }
        Write-Success "TestPyPI installation complete"
    } else {
        Write-Info "Installing from PyPI..."
        if ($Version) {
            pip install "greenlang-cli==$Version" 2>&1 | Out-File -Append $LogFile
        } else {
            pip install greenlang-cli 2>&1 | Out-File -Append $LogFile
        }
        Write-Success "PyPI installation complete"
    }

    # Install test dependencies
    Write-Info "Installing test dependencies..."
    pip install pytest pytest-timeout 2>&1 | Out-File -Append $LogFile
    Write-Success "Test dependencies installed"

    # Step 3: Verify Installation
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "Step 3: Verifying Installation"
    Write-Host "========================================================================"

    # Check CLI is accessible
    Write-Info "Checking CLI availability..."
    $glPath = Get-Command gl -ErrorAction SilentlyContinue
    if ($glPath) {
        Write-Success "gl command is available at: $($glPath.Source)"
    } else {
        Write-Error "gl command not found in PATH"
        throw "CLI not installed correctly"
    }

    # Check version
    Write-Info "Checking version..."
    $versionOutput = gl --version 2>&1
    $installedVersion = if ($versionOutput -match '(\d+\.\d+\.\d+)') { $Matches[1] } else { $null }

    if ($installedVersion) {
        Write-Success "Installed version: $installedVersion"
        if ($Version -and $installedVersion -ne $Version) {
            Write-Warning "Version mismatch: expected $Version, got $installedVersion"
            if ($Strict) { throw "Version mismatch" }
        }
    } else {
        Write-Warning "Could not determine installed version from output: $versionOutput"
    }

    # Step 4: Run Basic CLI Tests
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "Step 4: Running Basic CLI Tests"
    Write-Host "========================================================================"

    Run-Test "gl --version" { gl --version }
    Run-Test "gl --help" { gl --help }
    Run-Test "gl doctor" { gl doctor }
    Run-Test "gl version" { gl version }
    Run-Test "gl pack --help" { gl pack --help }
    Run-Test "gl pack list" { gl pack list }

    # Step 5: Run Python Import Tests
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "Step 5: Running Python Import Tests"
    Write-Host "========================================================================"

    Run-Test "Import greenlang" { python -c "import greenlang; print(greenlang.__version__)" }
    Run-Test "Import BaseAgent" { python -c "from greenlang.agents.base import BaseAgent; print(BaseAgent)" }
    Run-Test "Import PackLoader" { python -c "from greenlang.ecosystem.packs.loader import PackLoader; print(PackLoader)" }
    Run-Test "Import CLI" { python -c "from greenlang.cli.main import app; print(app)" }

    # Step 6: Run Pytest Smoke Tests (if available)
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "Step 6: Running Pytest Smoke Tests"
    Write-Host "========================================================================"

    $smokeTestFile = Join-Path $ProjectRoot "tests\smoke\test_release_smoke.py"
    if (Test-Path $smokeTestFile) {
        Write-Info "Running pytest smoke tests..."

        $env:GL_EXPECTED_VERSION = if ($Version) { $Version } else { $installedVersion }
        $env:GL_SMOKE_STRICT = if ($Strict) { "1" } else { "0" }

        $pytestArgs = @("-v", "--tb=short", "--timeout=$Timeout", $smokeTestFile)
        if ($Strict) { $pytestArgs += "-x" }

        $pytestResult = python -m pytest @pytestArgs 2>&1
        $pytestResult | Out-File -Append $LogFile

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Pytest smoke tests passed"
            $TestsPassed++
        } else {
            Write-Error "Pytest smoke tests failed"
            $TestsFailed++
            Write-Warning "Last lines of output:"
            $pytestResult | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }
        }
    } else {
        Write-Warning "Smoke test file not found: $smokeTestFile"
    }

    # Step 7: Summary
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "                        Test Summary"
    Write-Host "========================================================================"
    Write-Host ""
    Write-Info "Tests Passed: $TestsPassed"
    Write-Info "Tests Failed: $TestsFailed"
    Write-Info "Log File:     $LogFile"
    Write-Host ""

    if ($TestsFailed -gt 0) {
        Write-Error "SMOKE TESTS FAILED"
        Write-Warning "Check the log file for details: $LogFile"
        exit 1
    } else {
        Write-Success "ALL SMOKE TESTS PASSED"
        Write-Host ""
        $displayVersion = if ($installedVersion) { $installedVersion } else { $Version }
        Write-Success "GreenLang $displayVersion is ready for use!"
    }

    Write-Host "========================================================================"
    exit 0

} catch {
    Write-Error "Smoke test failed with error: $_"
    Write-Warning "Check the log file for details: $LogFile"
    exit 1
} finally {
    Cleanup
}
