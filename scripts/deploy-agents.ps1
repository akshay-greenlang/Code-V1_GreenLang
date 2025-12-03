# =============================================================================
# GreenLang Agents - Kubernetes Deployment Script (PowerShell)
# =============================================================================
# Deploys all agents to Kubernetes cluster
# Usage: .\scripts\deploy-agents.ps1 [-Namespace "greenlang-dev"] [-DryRun]
# =============================================================================

param(
    [string]$Namespace = "greenlang-dev",
    [switch]$DryRun
)

# Configuration
$K8sDir = "k8s/agents"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Color = switch ($Level) {
        "INFO" { "Green" }
        "WARN" { "Yellow" }
        "ERROR" { "Red" }
        "SECTION" { "Cyan" }
        default { "White" }
    }
    $Prefix = if ($Level -eq "SECTION") { "[====]" } else { "[$Level]" }
    Write-Host "$Prefix $Message" -ForegroundColor $Color
}

function Test-Prerequisites {
    Write-Log "Checking prerequisites..." "SECTION"

    # Check kubectl
    $kubectlExists = Get-Command kubectl -ErrorAction SilentlyContinue
    if (-not $kubectlExists) {
        Write-Log "kubectl not found. Please install kubectl." "ERROR"
        exit 1
    }

    # Check cluster connection
    try {
        kubectl cluster-info | Out-Null
    } catch {
        Write-Log "Cannot connect to Kubernetes cluster. Check your kubeconfig." "ERROR"
        exit 1
    }

    Write-Log "Prerequisites check passed"
}

function Apply-Manifests {
    $DryRunFlag = if ($DryRun) { "--dry-run=client" } else { "" }

    if ($DryRun) {
        Write-Log "Running in dry-run mode" "WARN"
    }

    Write-Log "Applying Kubernetes manifests..." "SECTION"

    $Manifests = @(
        "namespace.yaml",
        "rbac.yaml",
        "configmap.yaml",
        "services.yaml",
        "deployment-fuel-analyzer.yaml",
        "deployment-carbon-intensity.yaml",
        "deployment-energy-performance.yaml",
        "hpa.yaml"
    )

    foreach ($Manifest in $Manifests) {
        $File = Join-Path $K8sDir $Manifest
        if (Test-Path $File) {
            Write-Log "Applying $Manifest..."
            if ($DryRun) {
                kubectl apply -f $File --dry-run=client
            } else {
                kubectl apply -f $File
            }
        } else {
            Write-Log "File not found: $File" "WARN"
        }
    }
}

function Wait-ForDeployments {
    if ($DryRun) {
        Write-Log "Skipping wait in dry-run mode"
        return
    }

    Write-Log "Waiting for deployments to be ready..." "SECTION"

    $Deployments = @(
        "fuel-analyzer",
        "carbon-intensity",
        "energy-performance"
    )

    foreach ($Deployment in $Deployments) {
        Write-Log "Waiting for $Deployment..."
        kubectl rollout status deployment/$Deployment -n $Namespace --timeout=300s

        if ($LASTEXITCODE -ne 0) {
            Write-Log "Deployment $Deployment failed to become ready" "ERROR"
            return
        }
    }

    Write-Log "All deployments are ready"
}

function Show-DeploymentStatus {
    if ($DryRun) {
        Write-Log "Skipping verification in dry-run mode"
        return
    }

    Write-Log "Verifying deployment..." "SECTION"

    Write-Host ""
    Write-Log "Pods:"
    kubectl get pods -n $Namespace -l app.kubernetes.io/part-of=greenlang-platform

    Write-Host ""
    Write-Log "Services:"
    kubectl get svc -n $Namespace

    Write-Host ""
    Write-Log "HPA:"
    kubectl get hpa -n $Namespace
}

# Main
Write-Host ""
Write-Log "=== GreenLang Agent Deployment ==="
Write-Log "Namespace: $Namespace"
Write-Log "K8s Directory: $K8sDir"
Write-Host ""

Test-Prerequisites
Apply-Manifests
Wait-ForDeployments
Show-DeploymentStatus

Write-Host ""
Write-Log "=== Deployment Complete ==="
Write-Log "To access agents, use kubectl port-forward:"
Write-Host "  kubectl port-forward svc/fuel-analyzer 8001:80 -n $Namespace"
Write-Host "  kubectl port-forward svc/carbon-intensity 8002:80 -n $Namespace"
Write-Host "  kubectl port-forward svc/energy-performance 8003:80 -n $Namespace"
