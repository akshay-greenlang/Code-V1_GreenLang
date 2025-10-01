# GreenLang Windows Installation Test Script
# Tests various installation scenarios and PATH configurations

param(
    [switch]$Verbose,
    [switch]$DryRun
)

# Set error action preference
$ErrorActionPreference = "Continue"

# Colors for output
$ColorInfo = "Cyan"
$ColorSuccess = "Green"
$ColorWarning = "Yellow"
$ColorError = "Red"
$ColorTest = "Magenta"

function Write-TestInfo($Message) {
    Write-Host "🔍 $Message" -ForegroundColor $ColorTest
}

function Write-Info($Message) {
    Write-Host "ℹ️  $Message" -ForegroundColor $ColorInfo
}

function Write-Success($Message) {
    Write-Host "✅ $Message" -ForegroundColor $ColorSuccess
}

function Write-Warning($Message) {
    Write-Host "⚠️  $Message" -ForegroundColor $ColorWarning
}

function Write-Error($Message) {
    Write-Host "❌ $Message" -ForegroundColor $ColorError
}

function Test-PythonInstallation($PythonPath) {
    try {
        $version = & $PythonPath --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $version -match "Python (\d+)\.(\d+)") {
            return @{
                Valid = $true
                Path = $PythonPath
                Version = $version.Trim()
                Major = [int]$matches[1]
                Minor = [int]$matches[2]
            }
        }
    } catch {
        # Python not found
    }
    return @{ Valid = $false }
}

function Test-GreenLangImport($PythonPath) {
    try {
        & $PythonPath -c "import greenlang.cli.main; print('Import successful')" 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Test-GlCommand {
    try {
        $result = gl --version 2>&1
        return @{
            Available = $LASTEXITCODE -eq 0
            Output = $result
        }
    } catch {
        return @{
            Available = $false
            Output = "Command not found"
        }
    }
}

function Test-PathConfiguration {
    Write-TestInfo "Testing PATH configuration scenarios"

    # Get current PATH
    $currentPath = $env:PATH
    $pathEntries = $currentPath -split ";" | Where-Object { $_ }

    Write-Info "Current PATH has $($pathEntries.Count) entries"

    # Look for Python Scripts directories in PATH
    $pythonScriptsInPath = $pathEntries | Where-Object { $_ -like "*\Scripts" -or $_ -like "*Python*" }

    if ($pythonScriptsInPath.Count -gt 0) {
        Write-Success "Found Python-related directories in PATH:"
        foreach ($dir in $pythonScriptsInPath) {
            Write-Host "  - $dir" -ForegroundColor Green

            # Check if gl.exe exists in this directory
            $glExe = Join-Path $dir "gl.exe"
            if (Test-Path $glExe) {
                Write-Success "    ✓ gl.exe found"
            }
        }
    } else {
        Write-Warning "No Python Scripts directories found in PATH"
    }

    return @{
        TotalEntries = $pathEntries.Count
        PythonScriptsCount = $pythonScriptsInPath.Count
        PythonScriptsPaths = $pythonScriptsInPath
    }
}

function Test-PythonInstallations {
    Write-TestInfo "Scanning for Python installations"

    $pythonPaths = @(
        # Standard Python.org installations
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",

        # System-wide installations
        "C:\Python313\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python310\python.exe",

        # Anaconda/Miniconda
        "$env:USERPROFILE\Anaconda3\python.exe",
        "$env:USERPROFILE\Miniconda3\python.exe",
        "C:\ProgramData\Anaconda3\python.exe",
        "C:\ProgramData\Miniconda3\python.exe",

        # Windows Store Python
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe",

        # PATH-based python
        "python"
    )

    $validInstallations = @()
    $installationTypes = @{}

    foreach ($pythonPath in $pythonPaths) {
        $result = Test-PythonInstallation $pythonPath
        if ($result.Valid) {
            $validInstallations += $result

            # Categorize installation type
            $type = switch -Wildcard ($pythonPath) {
                "*LOCALAPPDATA*Python*" { "User Install (Python.org)" }
                "C:\Python*" { "System Install (Python.org)" }
                "*Anaconda*" { "Anaconda" }
                "*Miniconda*" { "Miniconda" }
                "*WindowsApps*" { "Windows Store" }
                "python" { "PATH-based" }
                default { "Unknown" }
            }

            $result.Type = $type

            if (-not $installationTypes.ContainsKey($type)) {
                $installationTypes[$type] = @()
            }
            $installationTypes[$type] += $result

            Write-Success "Found: $($result.Version) [$type] at $($result.Path)"

            # Test GreenLang import
            if (Test-GreenLangImport $result.Path) {
                Write-Success "  ✓ GreenLang is installed"
            } else {
                Write-Warning "  ⚠ GreenLang not installed in this Python"
            }

            # Check Scripts directory
            $scriptsDir = (Split-Path $result.Path) + "\Scripts"
            if (Test-Path $scriptsDir) {
                $glExe = Join-Path $scriptsDir "gl.exe"
                if (Test-Path $glExe) {
                    Write-Success "  ✓ gl.exe found in Scripts directory"
                } else {
                    Write-Warning "  ⚠ gl.exe not found in Scripts directory"
                }
            } else {
                Write-Warning "  ⚠ Scripts directory not found"
            }
        }
    }

    Write-Info "Summary: Found $($validInstallations.Count) Python installations"
    foreach ($type in $installationTypes.Keys) {
        Write-Info "  - $type: $($installationTypes[$type].Count) installation(s)"
    }

    return @{
        Installations = $validInstallations
        Types = $installationTypes
        Count = $validInstallations.Count
    }
}

function Test-WindowsUtilities {
    Write-TestInfo "Testing Windows PATH utilities"

    try {
        # Try to import the utilities
        $testScript = @"
import sys
try:
    from greenlang.utils.windows_path import (
        diagnose_path_issues,
        setup_windows_path,
        find_gl_executable,
        get_python_scripts_directories
    )

    print("✓ Windows utilities imported successfully")

    # Run diagnostics
    diagnosis = diagnose_path_issues()
    print(f"Python executable: {diagnosis['python_executable']}")
    print(f"Scripts directories: {len(diagnosis['scripts_directories'])}")
    print(f"gl.exe found: {diagnosis['gl_executable_found']}")
    if diagnosis['gl_executable_found']:
        print(f"gl.exe path: {diagnosis['gl_executable_path']}")
        print(f"In PATH: {diagnosis['in_path']}")

    print(f"Recommendations: {len(diagnosis['recommendations'])}")
    for rec in diagnosis['recommendations']:
        print(f"  - {rec}")

except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error running diagnostics: {e}")
    sys.exit(1)
"@

        $tempFile = [System.IO.Path]::GetTempFileName() + ".py"
        $testScript | Out-File -FilePath $tempFile -Encoding UTF8

        $result = python $tempFile 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Windows utilities test passed"
            if ($Verbose) {
                $result | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
            }
        } else {
            Write-Error "Windows utilities test failed:"
            $result | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        }

        Remove-Item $tempFile -ErrorAction SilentlyContinue

    } catch {
        Write-Error "Failed to test Windows utilities: $_"
    }
}

function Test-InstallationScenarios {
    Write-TestInfo "Testing installation scenarios"

    # Test 1: Direct gl command
    Write-Info "Test 1: Direct gl command execution"
    $glTest = Test-GlCommand
    if ($glTest.Available) {
        Write-Success "gl command works: $($glTest.Output)"
    } else {
        Write-Warning "gl command not available: $($glTest.Output)"
    }

    # Test 2: Python module execution
    Write-Info "Test 2: Python module execution (python -m greenlang.cli)"
    try {
        $moduleResult = python -m greenlang.cli --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Module execution works: $moduleResult"
        } else {
            Write-Error "Module execution failed: $moduleResult"
        }
    } catch {
        Write-Error "Module execution test failed: $_"
    }

    # Test 3: Batch wrapper (if exists)
    Write-Info "Test 3: Batch wrapper execution"
    $batchWrapper = ".\scripts\gl-wrapper.bat"
    if (Test-Path $batchWrapper) {
        try {
            $wrapperResult = & $batchWrapper --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Batch wrapper works: $wrapperResult"
            } else {
                Write-Warning "Batch wrapper failed: $wrapperResult"
            }
        } catch {
            Write-Warning "Batch wrapper test failed: $_"
        }
    } else {
        Write-Info "Batch wrapper not found at $batchWrapper"
    }
}

function Test-DoctorCommand {
    Write-TestInfo "Testing gl doctor command functionality"

    try {
        # Test basic doctor command
        Write-Info "Running: gl doctor"
        $doctorResult = gl doctor 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "gl doctor command works"
            if ($Verbose) {
                $doctorResult | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
            }
        } else {
            Write-Warning "gl doctor command failed"
            $doctorResult | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
        }

        # Test verbose doctor command
        Write-Info "Running: gl doctor --verbose"
        $doctorVerbose = gl doctor --verbose 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "gl doctor --verbose works"
        } else {
            Write-Warning "gl doctor --verbose failed"
        }

    } catch {
        Write-Error "Doctor command test failed: $_"
    }
}

# Main test execution
function Main {
    Write-Host ""
    Write-Host "GreenLang Windows Installation Test" -ForegroundColor White -BackgroundColor DarkBlue
    Write-Host "===================================" -ForegroundColor White -BackgroundColor DarkBlue
    Write-Host ""

    $testResults = @{}

    # Test 1: Python installations
    $testResults.PythonInstallations = Test-PythonInstallations

    # Test 2: PATH configuration
    $testResults.PathConfiguration = Test-PathConfiguration

    # Test 3: Windows utilities
    Test-WindowsUtilities

    # Test 4: Installation scenarios
    Test-InstallationScenarios

    # Test 5: Doctor command
    Test-DoctorCommand

    # Summary
    Write-Host ""
    Write-Host "Test Summary" -ForegroundColor White -BackgroundColor DarkGreen
    Write-Host "============" -ForegroundColor White -BackgroundColor DarkGreen
    Write-Host ""

    Write-Info "Python installations found: $($testResults.PythonInstallations.Count)"
    Write-Info "PATH entries: $($testResults.PathConfiguration.TotalEntries)"
    Write-Info "Python Scripts in PATH: $($testResults.PathConfiguration.PythonScriptsCount)"

    # Test gl command availability
    $glTest = Test-GlCommand
    if ($glTest.Available) {
        Write-Success "Final result: gl command is WORKING ✅"
    } else {
        Write-Warning "Final result: gl command NOT WORKING ⚠️"
        Write-Info "Try these solutions:"
        Write-Info "1. Run: gl doctor --setup-path"
        Write-Info "2. Restart your command prompt"
        Write-Info "3. Use: python -m greenlang.cli"
    }

    Write-Host ""
    Write-Info "Test completed. For more details, run with -Verbose flag."
}

# Run main function
if (-not $DryRun) {
    Main
} else {
    Write-Info "Dry run mode - would execute installation tests"
}