# =============================================================================
# run_factors_tests.ps1 — Windows launcher for the GreenLang Factors suite
# =============================================================================
# Closes the CTO-flagged "no reproducible test environment" gap on Windows
# (the primary dev platform per CLAUDE.md).  Mirrors run_factors_tests.sh
# so behavior is identical across OSes.
#
# Strategy:
#   1. If Docker Desktop is running, bring up the dedicated factors-test
#      compose stack and surface its exit code.
#   2. Otherwise fall back to a venv-based local install of the
#      `factors-test` extras and run pytest directly.
#
# Usage:
#   pwsh -File scripts/run_factors_tests.ps1
#   pwsh -File scripts/run_factors_tests.ps1 tests/factors/billing -k credits
#
# Env:
#   $env:GL_FACTORS_FORCE_LOCAL = "1"   # skip Docker even if available
#   $env:GL_FACTORS_VENV        = ".venv"
# =============================================================================
[CmdletBinding()]
param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]] $PytestArgs
)

$ErrorActionPreference = "Stop"

$RepoRoot    = (Resolve-Path "$PSScriptRoot\..").Path
$ComposeFile = Join-Path $RepoRoot "deployment\docker\docker-compose.factors-test.yml"
$VenvPath    = if ($env:GL_FACTORS_VENV) { $env:GL_FACTORS_VENV } else { Join-Path $RepoRoot ".venv" }

function Write-Log($Message) {
  Write-Host "[factors-test] $Message" -ForegroundColor Cyan
}

function Test-Cmd($Name) {
  try   { $null = Get-Command $Name -ErrorAction Stop; return $true }
  catch { return $false }
}

function Invoke-DockerRun {
  Write-Log "Docker detected -- launching reproducible test stack."
  Write-Log "Compose file: $ComposeFile"

  $compose = $null
  & docker compose version *>$null
  if ($LASTEXITCODE -eq 0) {
    $compose = @("docker", "compose")
  } elseif (Test-Cmd "docker-compose") {
    $compose = @("docker-compose")
  } else {
    Write-Log "WARN: docker present but compose missing -- falling back to local."
    return 99
  }

  if ($PytestArgs.Count -gt 0) {
    Write-Log "Running custom pytest invocation: $($PytestArgs -join ' ')"
    & $compose[0] $compose[1..($compose.Length - 1)] -f $ComposeFile up -d postgres redis
    & $compose[0] $compose[1..($compose.Length - 1)] -f $ComposeFile run --rm factors-test pytest @PytestArgs
    $rc = $LASTEXITCODE
    & $compose[0] $compose[1..($compose.Length - 1)] -f $ComposeFile down -v | Out-Null
    return $rc
  }

  & $compose[0] $compose[1..($compose.Length - 1)] -f $ComposeFile up --build --abort-on-container-exit --exit-code-from factors-test
  $rc = $LASTEXITCODE
  & $compose[0] $compose[1..($compose.Length - 1)] -f $ComposeFile down -v | Out-Null
  return $rc
}

function Invoke-LocalRun {
  Write-Log "Running locally (no Docker)."

  if (-not (Test-Path $VenvPath)) {
    Write-Log "Creating venv at $VenvPath"
    & python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv" }
  }

  # Activate venv (Windows path).
  $activate = Join-Path $VenvPath "Scripts\Activate.ps1"
  if (-not (Test-Path $activate)) {
    # Linux-style venv (e.g. WSL) fallback.
    $activate = Join-Path $VenvPath "bin\Activate.ps1"
  }
  . $activate

  & python -c "import pytest" 2>$null
  if ($LASTEXITCODE -ne 0) {
    Write-Log "Installing factors-test extras into $VenvPath"
    & python -m pip install --upgrade pip
    & python -m pip install -e "$RepoRoot[factors-test]"
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
  }

  Push-Location $RepoRoot
  try {
    if ($PytestArgs.Count -gt 0) {
      & pytest @PytestArgs
    } else {
      & pytest tests/factors -v --maxfail=10 `
          --cov=greenlang.factors --cov-report=term-missing
    }
    return $LASTEXITCODE
  } finally {
    Pop-Location
  }
}

# ----------------------------- main --------------------------------------
if ($env:GL_FACTORS_FORCE_LOCAL -eq "1") {
  exit (Invoke-LocalRun)
}

if (Test-Cmd "docker") {
  $rc = Invoke-DockerRun
  if ($rc -eq 99) {
    exit (Invoke-LocalRun)
  }
  exit $rc
}

exit (Invoke-LocalRun)
