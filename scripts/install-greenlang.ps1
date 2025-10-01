# GreenLang Universal Windows Installer
# One-line install: irm https://greenlang.io/install.ps1 | iex

param(
    [switch]$Force,
    [switch]$User,
    [switch]$System,
    [switch]$SkipPathSetup,
    [string]$PythonPath,
    [switch]$Verbose
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
$ColorInfo = "Cyan"
$ColorSuccess = "Green"
$ColorWarning = "Yellow"
$ColorError = "Red"

function Write-Info($Message) {
    Write-Host $Message -ForegroundColor $ColorInfo
}

function Write-Success($Message) {
    Write-Host "✓ $Message" -ForegroundColor $ColorSuccess
}

function Write-Warning($Message) {
    Write-Host "⚠ $Message" -ForegroundColor $ColorWarning
}

function Write-Error($Message) {
    Write-Host "✗ $Message" -ForegroundColor $ColorError
}

function Test-PythonInstallation($PythonExe) {
    try {
        $version = & $PythonExe --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $version -match "Python (\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            if ($major -ge 3 -and $minor -ge 10) {
                return @{
                    Valid = $true
                    Path = $PythonExe
                    Version = $version.Trim()
                    Major = $major
                    Minor = $minor
                }
            }
        }
    } catch {
        # Python not found or invalid
    }
    return @{ Valid = $false }
}

function Find-PythonInstallations {
    Write-Info "Searching for Python installations..."

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

        # PATH-based python
        "python"
    )

    $validInstallations = @()

    foreach ($pythonPath in $pythonPaths) {
        $result = Test-PythonInstallation $pythonPath
        if ($result.Valid) {
            $validInstallations += $result
            if ($Verbose) {
                Write-Info "Found: $($result.Version) at $($result.Path)"
            }
        }
    }

    return $validInstallations
}

function Get-ScriptsDirectory($PythonPath) {
    $pythonDir = Split-Path $PythonPath
    return Join-Path $pythonDir "Scripts"
}

function Test-InPath($Directory) {
    $pathDirs = $env:PATH -split ";" | Where-Object { $_ }
    return $pathDirs -contains $Directory
}

function Add-ToUserPath($Directory) {
    try {
        # Get current user PATH
        $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")

        # Check if already in PATH
        if ($currentPath -split ";" -contains $Directory) {
            Write-Info "Directory already in user PATH: $Directory"
            return $true
        }

        # Add to PATH
        $newPath = if ($currentPath) { "$Directory;$currentPath" } else { $Directory }
        [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")

        # Update current session
        $env:PATH = "$Directory;$env:PATH"

        Write-Success "Added to user PATH: $Directory"
        return $true
    } catch {
        Write-Error "Failed to update PATH: $_"
        return $false
    }
}

function Install-GreenLang($PythonInfo, $UseUser) {
    $pythonExe = $PythonInfo.Path

    Write-Info "Installing GreenLang CLI with $($PythonInfo.Version)..."

    try {
        # Determine install arguments
        $installArgs = @("install")
        if ($UseUser) {
            $installArgs += "--user"
        }
        $installArgs += "greenlang-cli"

        # Run pip install
        & $pythonExe -m pip @installArgs

        if ($LASTEXITCODE -eq 0) {
            Write-Success "GreenLang CLI installed successfully"
            return $true
        } else {
            Write-Error "Installation failed with exit code $LASTEXITCODE"
            return $false
        }
    } catch {
        Write-Error "Installation failed: $_"
        return $false
    }
}

function Test-GreenLangInstallation($PythonInfo) {
    try {
        # Test module import
        & $PythonInfo.Path -c "import greenlang.cli.main; print('GreenLang CLI import successful')" 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Setup-WindowsPath($PythonInfo) {
    if ($SkipPathSetup) {
        Write-Info "Skipping PATH setup as requested"
        return $true
    }

    $scriptsDir = Get-ScriptsDirectory $PythonInfo.Path

    if (Test-Path (Join-Path $scriptsDir "gl.exe")) {
        if (Test-InPath $scriptsDir) {
            Write-Success "gl.exe is already in PATH"
            return $true
        } else {
            Write-Info "Adding Scripts directory to PATH: $scriptsDir"
            return Add-ToUserPath $scriptsDir
        }
    } else {
        Write-Warning "gl.exe not found in $scriptsDir"

        # Try to run the diagnosis and setup
        try {
            & $PythonInfo.Path -c "from greenlang.utils.windows_path import setup_windows_path; success, msg = setup_windows_path(); print(f'Setup: {success} - {msg}')"
            if ($LASTEXITCODE -eq 0) {
                Write-Success "PATH setup completed via Python utilities"
                return $true
            }
        } catch {
            # Continue with manual setup
        }

        Write-Warning "Manual PATH setup may be required"
        return $false
    }
}

function Create-BatchWrapper($PythonInfo) {
    $scriptsDir = Get-ScriptsDirectory $PythonInfo.Path
    $batchFile = Join-Path $scriptsDir "gl.bat"

    if (-not (Test-Path $batchFile)) {
        try {
            $batchContent = @"
@echo off
REM GreenLang CLI Wrapper - Auto-generated
REM Try direct execution first
where gl.exe >nul 2>&1
if %errorlevel% equ 0 (
    gl.exe %*
    exit /b %errorlevel%
)

REM Fallback to Python module
"$($PythonInfo.Path)" -m greenlang.cli %*
"@
            $batchContent | Out-File -FilePath $batchFile -Encoding ASCII
            Write-Success "Created batch wrapper: $batchFile"
            return $true
        } catch {
            Write-Warning "Could not create batch wrapper: $_"
            return $false
        }
    } else {
        Write-Info "Batch wrapper already exists: $batchFile"
        return $true
    }
}

function Test-GreenLangCommand {
    Write-Info "Testing gl command..."

    try {
        $result = gl --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "gl command works: $result"
            return $true
        }
    } catch {
        # Command not found
    }

    # Try with full path
    $scriptsPath = ""
    $pythonInstalls = Find-PythonInstallations
    foreach ($python in $pythonInstalls) {
        $scriptsDir = Get-ScriptsDirectory $python.Path
        $glExe = Join-Path $scriptsDir "gl.exe"
        if (Test-Path $glExe) {
            $scriptsPath = $scriptsDir
            break
        }
    }

    if ($scriptsPath) {
        Write-Warning "gl.exe found but not in PATH: $scriptsPath"
        Write-Info "Try adding $scriptsPath to your PATH, or restart your terminal"
        return $false
    } else {
        Write-Error "gl command not accessible"
        return $false
    }
}

# Main installation logic
function Main {
    Write-Host ""
    Write-Host "GreenLang CLI Universal Windows Installer" -ForegroundColor White -BackgroundColor DarkBlue
    Write-Host "=========================================" -ForegroundColor White -BackgroundColor DarkBlue
    Write-Host ""

    # Check if running as admin (for system install)
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

    if ($System -and -not $isAdmin) {
        Write-Error "System installation requires administrator privileges"
        Write-Info "Run PowerShell as Administrator or use -User flag for user installation"
        exit 1
    }

    # Find Python installations
    $pythonInstallations = if ($PythonPath) {
        $result = Test-PythonInstallation $PythonPath
        if ($result.Valid) { @($result) } else { @() }
    } else {
        Find-PythonInstallations
    }

    if ($pythonInstallations.Count -eq 0) {
        Write-Error "No suitable Python installation found (requires Python 3.10+)"
        Write-Info "Install Python from https://python.org or https://anaconda.com"
        exit 1
    }

    # Select Python installation
    $selectedPython = $pythonInstallations[0]
    if ($pythonInstallations.Count -gt 1 -and -not $PythonPath) {
        Write-Info "Multiple Python installations found:"
        for ($i = 0; $i -lt $pythonInstallations.Count; $i++) {
            Write-Host "  $($i + 1). $($pythonInstallations[$i].Version) at $($pythonInstallations[$i].Path)"
        }
        Write-Info "Using: $($selectedPython.Version) at $($selectedPython.Path)"
    } else {
        Write-Info "Using Python: $($selectedPython.Version) at $($selectedPython.Path)"
    }

    # Check if already installed
    if (-not $Force -and (Test-GreenLangInstallation $selectedPython)) {
        Write-Success "GreenLang CLI is already installed"
    } else {
        # Install GreenLang
        $useUser = $User -or (-not $System -and -not $isAdmin)
        $installSuccess = Install-GreenLang $selectedPython $useUser

        if (-not $installSuccess) {
            Write-Error "Installation failed"
            exit 1
        }
    }

    # Set up PATH
    $pathSuccess = Setup-WindowsPath $selectedPython

    # Create batch wrapper
    $wrapperSuccess = Create-BatchWrapper $selectedPython

    # Test installation
    Write-Host ""
    Write-Info "Testing installation..."
    $testSuccess = Test-GreenLangCommand

    # Final status
    Write-Host ""
    if ($testSuccess) {
        Write-Success "GreenLang CLI installation completed successfully!"
        Write-Info "You can now use the 'gl' command"
    } else {
        Write-Warning "Installation completed but 'gl' command may not be available"
        Write-Info "Try one of these alternatives:"
        Write-Info "1. Restart your command prompt/PowerShell"
        Write-Info "2. Use: python -m greenlang.cli"
        Write-Info "3. Run: gl doctor --setup-path"
    }

    Write-Host ""
    Write-Info "Get started with: gl doctor"
    Write-Info "Documentation: https://greenlang.io/docs"
}

# Run main function
Main