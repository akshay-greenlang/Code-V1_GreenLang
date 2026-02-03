<#
.SYNOPSIS
    INFRA-001: Deploy GreenLang Kubernetes Production Cluster

.DESCRIPTION
    This script automates the deployment of the GreenLang production infrastructure
    to AWS EKS as specified in PRD-INFRA-001-K8s-Deployment.md

.PARAMETER Environment
    Target environment: dev, staging, or prod (default: prod)

.PARAMETER Action
    Deployment action: plan, apply, destroy, or status (default: plan)

.PARAMETER SkipPrerequisites
    Skip prerequisite checks (not recommended)

.PARAMETER AutoApprove
    Auto-approve terraform apply (use with caution)

.EXAMPLE
    .\Deploy-INFRA001.ps1 -Environment prod -Action plan

.EXAMPLE
    .\Deploy-INFRA001.ps1 -Environment prod -Action apply -AutoApprove

.NOTES
    Version: 1.0.0
    Date: 2026-02-03
    Estimated Cost: ~$4,234/month for production
#>

[CmdletBinding()]
param(
    [ValidateSet('dev', 'staging', 'prod')]
    [string]$Environment = 'prod',

    [ValidateSet('plan', 'apply', 'destroy', 'status', 'init')]
    [string]$Action = 'plan',

    [switch]$SkipPrerequisites,

    [switch]$AutoApprove
)

# Script configuration
$ErrorActionPreference = 'Stop'
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
$TerraformDir = Join-Path $ScriptRoot "terraform\environments\$Environment"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }
function Write-Info { Write-Host $args -ForegroundColor Cyan }

# Banner
function Show-Banner {
    Write-Host ""
    Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║           INFRA-001: GreenLang K8s Production Deploy          ║" -ForegroundColor Cyan
    Write-Host "║                                                               ║" -ForegroundColor Cyan
    Write-Host "║  Environment: $($Environment.PadRight(10))  Action: $($Action.PadRight(10))           ║" -ForegroundColor Cyan
    Write-Host "║  Estimated Monthly Cost: ~`$4,234                              ║" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    $errors = @()

    # Check AWS CLI
    try {
        $awsVersion = aws --version 2>&1
        Write-Success "  [OK] AWS CLI: $awsVersion"
    } catch {
        $errors += "AWS CLI not found. Install from: https://aws.amazon.com/cli/"
    }

    # Check Terraform
    try {
        $tfVersion = terraform version -json 2>&1 | ConvertFrom-Json
        Write-Success "  [OK] Terraform: $($tfVersion.terraform_version)"
    } catch {
        $errors += "Terraform not found. Install from: https://www.terraform.io/downloads"
    }

    # Check kubectl
    try {
        $kubectlVersion = kubectl version --client -o json 2>&1 | ConvertFrom-Json
        Write-Success "  [OK] kubectl: $($kubectlVersion.clientVersion.gitVersion)"
    } catch {
        $errors += "kubectl not found. Install from: https://kubernetes.io/docs/tasks/tools/"
    }

    # Check Helm
    try {
        $helmVersion = helm version --short 2>&1
        Write-Success "  [OK] Helm: $helmVersion"
    } catch {
        $errors += "Helm not found. Install from: https://helm.sh/docs/intro/install/"
    }

    # Check AWS credentials
    try {
        $identity = aws sts get-caller-identity 2>&1 | ConvertFrom-Json
        Write-Success "  [OK] AWS Account: $($identity.Account)"
        Write-Success "  [OK] AWS User: $($identity.Arn)"
        $script:AwsAccountId = $identity.Account
    } catch {
        $errors += "AWS credentials not configured. Run: aws configure"
    }

    if ($errors.Count -gt 0) {
        Write-Error ""
        Write-Error "Prerequisites check failed:"
        foreach ($err in $errors) {
            Write-Error "  - $err"
        }
        Write-Error ""
        return $false
    }

    Write-Success ""
    Write-Success "All prerequisites satisfied!"
    return $true
}

# Update terraform.tfvars with actual AWS Account ID
function Update-TfVars {
    Write-Info "Updating terraform.tfvars with AWS Account ID..."

    $tfvarsPath = Join-Path $TerraformDir "terraform.tfvars"

    if (-not (Test-Path $tfvarsPath)) {
        Write-Error "terraform.tfvars not found at: $tfvarsPath"
        return $false
    }

    $content = Get-Content $tfvarsPath -Raw

    if ($content -match 'ACCOUNT_ID') {
        $newContent = $content -replace 'ACCOUNT_ID', $script:AwsAccountId
        Set-Content -Path $tfvarsPath -Value $newContent
        Write-Success "  [OK] Replaced ACCOUNT_ID with $($script:AwsAccountId)"
    } else {
        Write-Info "  [OK] No ACCOUNT_ID placeholders found (already configured)"
    }

    return $true
}

# Initialize Terraform backend
function Initialize-TerraformBackend {
    Write-Info "Initializing Terraform backend..."

    $initScript = Join-Path $ScriptRoot "terraform\scripts\infra-init.ps1"

    if (Test-Path $initScript) {
        Write-Info "Running backend initialization script..."
        & $initScript -Environment $Environment
    } else {
        Write-Warning "Backend init script not found, running terraform init directly..."
    }

    Push-Location $TerraformDir
    try {
        terraform init -reconfigure
        if ($LASTEXITCODE -ne 0) {
            throw "Terraform init failed"
        }
        Write-Success "  [OK] Terraform initialized"
    } finally {
        Pop-Location
    }

    return $true
}

# Run Terraform plan
function Invoke-TerraformPlan {
    Write-Info "Running Terraform plan..."

    Push-Location $TerraformDir
    try {
        terraform plan -out=tfplan
        if ($LASTEXITCODE -ne 0) {
            throw "Terraform plan failed"
        }
        Write-Success ""
        Write-Success "Plan saved to: tfplan"
        Write-Success "Review the plan above, then run with -Action apply to deploy"
    } finally {
        Pop-Location
    }

    return $true
}

# Run Terraform apply
function Invoke-TerraformApply {
    Write-Info "Running Terraform apply..."

    if (-not $AutoApprove) {
        Write-Warning ""
        Write-Warning "WARNING: This will create AWS resources with estimated cost of ~`$4,234/month"
        Write-Warning ""
        $confirm = Read-Host "Type 'yes' to confirm deployment"
        if ($confirm -ne 'yes') {
            Write-Info "Deployment cancelled"
            return $false
        }
    }

    Push-Location $TerraformDir
    try {
        $applyArgs = @()
        if ($AutoApprove) {
            $applyArgs += "-auto-approve"
        }
        if (Test-Path "tfplan") {
            terraform apply @applyArgs tfplan
        } else {
            terraform apply @applyArgs
        }

        if ($LASTEXITCODE -ne 0) {
            throw "Terraform apply failed"
        }

        Write-Success ""
        Write-Success "Infrastructure deployed successfully!"

    } finally {
        Pop-Location
    }

    return $true
}

# Configure kubectl for EKS
function Set-KubectlConfig {
    Write-Info "Configuring kubectl for EKS access..."

    $clusterName = "greenlang-$Environment-eks"
    $region = "us-east-1"

    aws eks update-kubeconfig --name $clusterName --region $region

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to configure kubectl. EKS cluster may not be ready yet."
        return $false
    }

    Write-Success "  [OK] kubectl configured for $clusterName"

    # Verify connection
    kubectl cluster-info

    return $true
}

# Deploy Kubernetes add-ons
function Install-K8sAddons {
    Write-Info "Installing Kubernetes add-ons..."

    # Add Helm repos
    Write-Info "  Adding Helm repositories..."
    helm repo add eks https://aws.github.io/eks-charts 2>$null
    helm repo add autoscaler https://kubernetes.github.io/autoscaler 2>$null
    helm repo add bitnami https://charts.bitnami.com/bitnami 2>$null
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>$null
    helm repo add grafana https://grafana.github.io/helm-charts 2>$null
    helm repo update

    $clusterName = "greenlang-$Environment-eks"

    # AWS Load Balancer Controller
    Write-Info "  Installing AWS Load Balancer Controller..."
    helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller `
        -n kube-system `
        --set clusterName=$clusterName `
        --wait

    # Cluster Autoscaler
    Write-Info "  Installing Cluster Autoscaler..."
    helm upgrade --install cluster-autoscaler autoscaler/cluster-autoscaler `
        -n kube-system `
        --set autoDiscovery.clusterName=$clusterName `
        --wait

    # Metrics Server
    Write-Info "  Installing Metrics Server..."
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

    Write-Success "  [OK] Kubernetes add-ons installed"

    return $true
}

# Deploy monitoring stack
function Install-MonitoringStack {
    Write-Info "Installing monitoring stack..."

    $monitoringChart = Join-Path $ScriptRoot "infrastructure\monitoring\helm"
    $valuesFile = Join-Path $monitoringChart "values-$Environment.yaml"

    if (-not (Test-Path $valuesFile)) {
        $valuesFile = Join-Path $monitoringChart "values.yaml"
    }

    helm upgrade --install greenlang-monitoring $monitoringChart `
        -f $valuesFile `
        -n monitoring `
        --create-namespace `
        --wait --timeout 10m

    Write-Success "  [OK] Monitoring stack installed"

    return $true
}

# Deploy GreenLang applications
function Install-GreenLangApps {
    Write-Info "Deploying GreenLang applications..."

    $kustomizeDir = Join-Path $ScriptRoot "kustomize\overlays\$Environment"

    if (Test-Path $kustomizeDir) {
        kubectl apply -k $kustomizeDir
    } else {
        Write-Warning "Kustomize overlay not found: $kustomizeDir"
        Write-Info "Deploying from Helm charts instead..."

        $helmChart = Join-Path $ScriptRoot "helm\greenlang-agents"
        $valuesFile = Join-Path $helmChart "values-$Environment.yaml"

        helm upgrade --install greenlang-agents $helmChart `
            -f $valuesFile `
            -n greenlang-agents `
            --create-namespace `
            --wait --timeout 10m
    }

    Write-Success "  [OK] GreenLang applications deployed"

    return $true
}

# Validate deployment
function Test-Deployment {
    Write-Info "Validating deployment..."

    Write-Info "  Checking pods..."
    kubectl get pods -A | Select-String -NotMatch "Running|Completed"

    Write-Info "  Checking services..."
    kubectl get svc -n greenlang

    Write-Info "  Checking ingress..."
    kubectl get ingress -n greenlang

    Write-Info "  Checking nodes..."
    kubectl get nodes

    Write-Success ""
    Write-Success "Deployment validation complete!"

    return $true
}

# Show deployment status
function Get-DeploymentStatus {
    Write-Info "Checking deployment status..."

    Push-Location $TerraformDir
    try {
        Write-Info ""
        Write-Info "Terraform State:"
        terraform state list 2>$null

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "No Terraform state found. Infrastructure may not be deployed."
        }
    } finally {
        Pop-Location
    }

    Write-Info ""
    Write-Info "Kubernetes Status:"
    try {
        kubectl cluster-info 2>$null
        kubectl get nodes 2>$null
        kubectl get pods -A 2>$null | Select-Object -First 20
    } catch {
        Write-Warning "Unable to connect to Kubernetes cluster"
    }

    return $true
}

# Main execution
function Main {
    Show-Banner

    # Check prerequisites
    if (-not $SkipPrerequisites) {
        if (-not (Test-Prerequisites)) {
            exit 1
        }
    }

    switch ($Action) {
        'init' {
            Update-TfVars
            Initialize-TerraformBackend
        }
        'plan' {
            Update-TfVars
            Initialize-TerraformBackend
            Invoke-TerraformPlan
        }
        'apply' {
            Update-TfVars
            Initialize-TerraformBackend
            Invoke-TerraformPlan
            Invoke-TerraformApply
            Set-KubectlConfig
            Install-K8sAddons
            Install-MonitoringStack
            Install-GreenLangApps
            Test-Deployment
        }
        'destroy' {
            Write-Warning "WARNING: This will DESTROY all infrastructure!"
            $confirm = Read-Host "Type 'destroy' to confirm"
            if ($confirm -eq 'destroy') {
                Push-Location $TerraformDir
                terraform destroy
                Pop-Location
            }
        }
        'status' {
            Get-DeploymentStatus
        }
    }

    Write-Host ""
    Write-Success "Done!"
}

# Run main
Main
