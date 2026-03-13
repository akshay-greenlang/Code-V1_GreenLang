# =============================================================================
# Apply EUDR Migrations V118-V128 (PowerShell Version)
# =============================================================================
# This script applies database migrations V118 through V128 for EUDR agents
# 030-040 (Documentation & Reporting category)
#
# EUDR Agents covered:
#   V118 - AGENT-EUDR-030: Documentation Generator
#   V119 - AGENT-EUDR-031: Stakeholder Engagement Tool
#   V120 - AGENT-EUDR-032: Grievance Mechanism Manager
#   V121 - AGENT-EUDR-033: Continuous Monitoring Agent
#   V122 - AGENT-EUDR-034: Annual Review Scheduler
#   V123 - AGENT-EUDR-035: Improvement Plan Creator
#   V124 - AGENT-EUDR-036: EU Information System Interface
#   V125 - AGENT-EUDR-037: Due Diligence Statement Creator
#   V126 - AGENT-EUDR-038: Reference Number Generator
#   V127 - AGENT-EUDR-039: Customs Declaration Support
#   V128 - AGENT-EUDR-040: Authority Communication Manager
#
# Prerequisites:
#   - Docker Desktop installed and running
#   - PostgreSQL database accessible
#   - Flyway Docker image available
#
# Usage:
#   .\apply_v118_v128.ps1 [-Environment dev|staging|prod]
#
# Examples:
#   .\apply_v118_v128.ps1                    # Apply to development database
#   .\apply_v118_v128.ps1 -Environment dev   # Apply to development database
#   .\apply_v118_v128.ps1 -Environment prod  # Apply to production database
#
# =============================================================================

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('dev', 'staging', 'prod')]
    [string]$Environment = 'dev'
)

$ErrorActionPreference = "Stop"

# =============================================================================
# Configuration
# =============================================================================

# Script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MigrationsDir = Join-Path $ScriptDir "sql"
$ConfigFile = Join-Path $ScriptDir "flyway.conf"

# Database connection settings (override with environment variables)
$DbHost = if ($env:FLYWAY_DB_HOST) { $env:FLYWAY_DB_HOST } else { "localhost" }
$DbPort = if ($env:FLYWAY_DB_PORT) { $env:FLYWAY_DB_PORT } else { "5432" }
$DbName = if ($env:FLYWAY_DB_NAME) { $env:FLYWAY_DB_NAME } else { "greenlang_platform" }
$DbUser = if ($env:FLYWAY_DB_USER) { $env:FLYWAY_DB_USER } else { "greenlang_admin" }
$DbPassword = if ($env:FLYWAY_DB_PASSWORD) { $env:FLYWAY_DB_PASSWORD } else { "greenlang_secure_2024" }

# JDBC URL
$JdbcUrl = "jdbc:postgresql://${DbHost}:${DbPort}/${DbName}"

# Flyway Docker image
$FlywayImage = if ($env:FLYWAY_IMAGE) { $env:FLYWAY_IMAGE } else { "flyway/flyway:9.22.3" }

# =============================================================================
# Helper Functions
# =============================================================================

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-Docker {
    Write-Info "Checking Docker status..."
    try {
        $dockerVersion = docker version --format '{{.Server.Version}}' 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Docker is not running. Please start Docker Desktop and try again."
            exit 1
        }
        Write-Success "Docker is running (version: $dockerVersion)"
    }
    catch {
        Write-Error "Docker is not installed or not running. Please install Docker Desktop and try again."
        exit 1
    }
}

function Test-Database {
    Write-Info "Checking database connectivity..."

    $result = docker run --rm `
        postgres:15-alpine `
        pg_isready -h $DbHost -p $DbPort -U $DbUser 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Database is accessible at ${DbHost}:${DbPort}"
    }
    else {
        Write-Error "Cannot connect to database at ${DbHost}:${DbPort}"
        Write-Info "Make sure PostgreSQL is running and accessible"
        Write-Info "Error: $result"
        exit 1
    }
}

function Invoke-Flyway {
    param([string]$Command)

    Write-Info "Running Flyway $Command..."

    # Convert Windows paths to Unix-style for Docker
    $UnixMigrationsDir = $MigrationsDir -replace '\\', '/' -replace '^C:', '/c'
    $UnixConfigFile = $ConfigFile -replace '\\', '/' -replace '^C:', '/c'

    docker run --rm `
        --network host `
        -v "${MigrationsDir}:/flyway/sql:ro" `
        -v "${ConfigFile}:/flyway/conf/flyway.conf:ro" `
        $FlywayImage `
        -url="$JdbcUrl" `
        -user="$DbUser" `
        -password="$DbPassword" `
        -locations="filesystem:/flyway/sql" `
        -configFiles="/flyway/conf/flyway.conf" `
        $Command

    return $LASTEXITCODE
}

# =============================================================================
# Main Execution
# =============================================================================

Write-Host ""
Write-Host "=============================================================================" -ForegroundColor Cyan
Write-Host "GreenLang Database Migrations V118-V128" -ForegroundColor Cyan
Write-Host "=============================================================================" -ForegroundColor Cyan
Write-Host "Environment: $Environment" -ForegroundColor Cyan
Write-Host "Database: ${DbHost}:${DbPort}/${DbName}" -ForegroundColor Cyan
Write-Host "=============================================================================" -ForegroundColor Cyan
Write-Host ""

# Pre-flight checks
Write-Info "Running pre-flight checks..."
Test-Docker
Test-Database

# Show current migration status
Write-Host ""
Write-Info "Current migration status:"
Invoke-Flyway "info"

# Show pending migrations
Write-Host ""
Write-Info "Checking for pending migrations..."
$pendingResult = Invoke-Flyway "info -pending"

# Validate migrations
Write-Host ""
Write-Info "Validating migrations..."
$validateResult = Invoke-Flyway "validate"

if ($validateResult -ne 0) {
    Write-Error "Migration validation failed"
    exit 1
}
Write-Success "Migration validation passed"

# Prompt for confirmation
Write-Host ""
Write-Warning "Ready to apply migrations V118-V128"
$confirmation = Read-Host "Do you want to proceed? (yes/no)"

if ($confirmation -ne "yes") {
    Write-Warning "Migration cancelled by user"
    exit 0
}

# Apply migrations
Write-Host ""
Write-Info "Applying migrations V118-V128..."
$migrateResult = Invoke-Flyway "migrate"

if ($migrateResult -ne 0) {
    Write-Error "Migration failed"
    exit 1
}
Write-Success "Migrations applied successfully"

# Show final migration status
Write-Host ""
Write-Info "Final migration status:"
Invoke-Flyway "info"

Write-Host ""
Write-Host "=============================================================================" -ForegroundColor Green
Write-Success "Migration complete! Database is now at version V128"
Write-Host "=============================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "EUDR Agents 030-040 (Documentation & Reporting) are now ready:" -ForegroundColor Green
Write-Host "  - V118: Documentation Generator" -ForegroundColor Green
Write-Host "  - V119: Stakeholder Engagement Tool" -ForegroundColor Green
Write-Host "  - V120: Grievance Mechanism Manager" -ForegroundColor Green
Write-Host "  - V121: Continuous Monitoring Agent" -ForegroundColor Green
Write-Host "  - V122: Annual Review Scheduler" -ForegroundColor Green
Write-Host "  - V123: Improvement Plan Creator" -ForegroundColor Green
Write-Host "  - V124: EU Information System Interface" -ForegroundColor Green
Write-Host "  - V125: Due Diligence Statement Creator" -ForegroundColor Green
Write-Host "  - V126: Reference Number Generator" -ForegroundColor Green
Write-Host "  - V127: Customs Declaration Support" -ForegroundColor Green
Write-Host "  - V128: Authority Communication Manager" -ForegroundColor Green
Write-Host "=============================================================================" -ForegroundColor Green
