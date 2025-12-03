# =============================================================================
# GreenLang Agents - Build Script (PowerShell)
# =============================================================================
# Builds Docker images for all 4 agents
# Usage: .\scripts\build-agents.ps1 [-Tag "v1.0.0"] [-Push] [-Scan] [-Local] [-Verify]
# =============================================================================

param(
    [string]$Tag = "latest",
    [switch]$Push,
    [switch]$Scan,
    [switch]$Local,     # Build with local tags only (no registry prefix)
    [switch]$Verify     # Verify images after build
)

# Configuration
$Registry = if ($Local) { "greenlang" } elseif ($env:REGISTRY) { $env:REGISTRY } else { "ghcr.io/greenlang" }

# Agents to build (all 4 agents)
$Agents = @(
    @{ Name = "fuel-analyzer"; Path = "generated/fuel_analyzer_agent"; AgentId = "emissions/fuel_analyzer_v1" },
    @{ Name = "carbon-intensity"; Path = "generated/carbon_intensity_v1"; AgentId = "cbam/carbon_intensity_v1" },
    @{ Name = "energy-performance"; Path = "generated/energy_performance_v1"; AgentId = "building/energy_performance_v1" },
    @{ Name = "eudr-compliance"; Path = "generated/eudr_compliance_v1"; AgentId = "regulatory/eudr_compliance_v1" }
)

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Color = switch ($Level) {
        "INFO" { "Green" }
        "WARN" { "Yellow" }
        "ERROR" { "Red" }
        default { "White" }
    }
    Write-Host "[$Level] $Message" -ForegroundColor $Color
}

function Build-BaseImage {
    Write-Log "Building base image..."

    $env:DOCKER_BUILDKIT = "1"

    docker build `
        -t "$Registry/greenlang-base:$Tag" `
        -f docker/base/Dockerfile.base `
        docker/base/

    if ($LASTEXITCODE -ne 0) {
        Write-Log "Failed to build base image" "ERROR"
        exit 1
    }

    if ($Push) {
        Write-Log "Pushing base image..."
        docker push "$Registry/greenlang-base:$Tag"
    }
}

function Build-Agent {
    param(
        [string]$Name,
        [string]$Path
    )

    Write-Log "Building $Name..."

    $env:DOCKER_BUILDKIT = "1"

    docker build `
        -t "$Registry/${Name}:$Tag" `
        -t "$Registry/${Name}:latest" `
        -f "$Path/Dockerfile" `
        .

    if ($LASTEXITCODE -ne 0) {
        Write-Log "Failed to build $Name" "ERROR"
        exit 1
    }

    if ($Scan) {
        Write-Log "Running Trivy security scan for $Name..."
        $trivyExists = Get-Command trivy -ErrorAction SilentlyContinue
        if ($trivyExists) {
            trivy image --severity HIGH,CRITICAL "$Registry/${Name}:$Tag"
        } else {
            Write-Log "Trivy not installed, skipping security scan" "WARN"
        }
    }

    if ($Push) {
        Write-Log "Pushing $Name..."
        docker push "$Registry/${Name}:$Tag"
        docker push "$Registry/${Name}:latest"
    }

    Write-Log "Successfully built $Name"
}

# Main
Write-Host ""
Write-Log "=== GreenLang Agent Build ==="
Write-Log "Registry: $Registry"
Write-Log "Tag: $Tag"
Write-Host ""

# Build base image
Build-BaseImage

# Build each agent
foreach ($Agent in $Agents) {
    Build-Agent -Name $Agent.Name -Path $Agent.Path
}

Write-Host ""
Write-Log "=== Build Complete ==="
Write-Log "Built images:"
foreach ($Agent in $Agents) {
    Write-Host "  - $Registry/$($Agent.Name):$Tag"
}

# Verify images if requested
if ($Verify) {
    Write-Host ""
    Write-Log "=== Verifying Images ==="

    # Verify base image
    Write-Log "Verifying base image..."
    $baseResult = docker image inspect "$Registry/greenlang-base:$Tag" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Base image verified: $Registry/greenlang-base:$Tag" "INFO"
    } else {
        Write-Log "Base image NOT found: $Registry/greenlang-base:$Tag" "ERROR"
    }

    # Verify each agent
    foreach ($Agent in $Agents) {
        Write-Log "Verifying $($Agent.Name)..."
        $result = docker image inspect "$Registry/$($Agent.Name):$Tag" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Agent image verified: $Registry/$($Agent.Name):$Tag" "INFO"

            # Show image size
            $size = docker image inspect "$Registry/$($Agent.Name):$Tag" --format '{{.Size}}' 2>&1
            $sizeMB = [math]::Round($size / 1MB, 2)
            Write-Host "    Size: ${sizeMB} MB"
        } else {
            Write-Log "Agent image NOT found: $Registry/$($Agent.Name):$Tag" "ERROR"
        }
    }

    Write-Host ""
    Write-Log "=== Image Summary ==="
    docker images | Select-String -Pattern "greenlang"
}

Write-Host ""
Write-Log "=== Quick Test Commands ==="
Write-Host ""
Write-Host "# Test fuel-analyzer agent:"
Write-Host "docker run --rm -p 8000:8000 $Registry/fuel-analyzer:$Tag"
Write-Host ""
Write-Host "# Test carbon-intensity agent:"
Write-Host "docker run --rm -p 8001:8000 $Registry/carbon-intensity:$Tag"
Write-Host ""
Write-Host "# Test energy-performance agent:"
Write-Host "docker run --rm -p 8002:8000 $Registry/energy-performance:$Tag"
Write-Host ""
Write-Host "# Test eudr-compliance agent:"
Write-Host "docker run --rm -p 8003:8000 $Registry/eudr-compliance:$Tag"
Write-Host ""
