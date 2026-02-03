#Requires -Version 5.1
<#
.SYNOPSIS
    GreenLang Infrastructure Initialization Script (PowerShell)

.DESCRIPTION
    This script initializes the AWS infrastructure required for Terraform state
    management and performs the initial terraform init.

    Prerequisites:
    - AWS CLI installed and configured
    - Terraform >= 1.0 installed
    - kubectl installed (for EKS management)

    This script is idempotent - safe to run multiple times.

.PARAMETER Region
    AWS region (default: us-east-1)

.PARAMETER Environment
    Target environment: dev, staging, prod (default: dev)

.PARAMETER SkipTerraformInit
    Skip terraform init after creating backend resources

.EXAMPLE
    .\infra-init.ps1
    Initialize with default settings

.EXAMPLE
    .\infra-init.ps1 -Region "us-west-2"
    Initialize using us-west-2 region

.EXAMPLE
    .\infra-init.ps1 -Environment "prod" -SkipTerraformInit
    Initialize for production, skip terraform init

.NOTES
    Author: GreenLang DevOps Team
    Version: 1.0.0
#>

[CmdletBinding()]
param(
    [Parameter(HelpMessage = "AWS region")]
    [string]$Region = $env:AWS_REGION,

    [Parameter(HelpMessage = "Target environment: dev, staging, prod")]
    [ValidateSet("dev", "staging", "prod")]
    [string]$Environment = "dev",

    [Parameter(HelpMessage = "Skip terraform init after creating backend resources")]
    [switch]$SkipTerraformInit
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
$ErrorActionPreference = "Stop"

# Set default region if not provided
if ([string]::IsNullOrEmpty($Region)) {
    $Region = "us-east-1"
}

$StateBucketName = "greenlang-terraform-state"
$LockTableName = "greenlang-terraform-locks"
$script:AwsAccountId = $null

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Blue
    Write-Host $Message -ForegroundColor White
    Write-Host "============================================================================" -ForegroundColor Blue
    Write-Host ""
}

function Write-Step {
    param([string]$Message)
    Write-Host "[STEP] " -ForegroundColor Cyan -NoNewline
    Write-Host $Message
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor Blue -NoNewline
    Write-Host $Message
}

function Test-CommandExists {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# -----------------------------------------------------------------------------
# Check Prerequisites
# -----------------------------------------------------------------------------
function Test-Prerequisites {
    Write-Header "Checking Prerequisites"

    $hasErrors = $false

    # Check AWS CLI
    Write-Step "Checking AWS CLI..."
    if (Test-CommandExists "aws") {
        $awsVersion = & aws --version 2>&1
        Write-Success "AWS CLI installed: $awsVersion"
    }
    else {
        Write-Error "AWS CLI is not installed"
        Write-Host "  Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        $hasErrors = $true
    }

    # Check Terraform
    Write-Step "Checking Terraform..."
    if (Test-CommandExists "terraform") {
        $tfVersionOutput = & terraform version 2>&1 | Select-Object -First 1
        Write-Success "Terraform installed: $tfVersionOutput"

        # Check minimum version
        $tfVersion = $tfVersionOutput -replace '.*v(\d+\.\d+\.\d+).*', '$1'
        $requiredVersion = [version]"1.0.0"
        try {
            $currentVersion = [version]$tfVersion
            if ($currentVersion -lt $requiredVersion) {
                Write-Warning "Terraform version $tfVersion may be too old. Required: >= $requiredVersion"
            }
        }
        catch {
            Write-Info "Could not parse Terraform version for comparison"
        }
    }
    else {
        Write-Error "Terraform is not installed"
        Write-Host "  Install: https://developer.hashicorp.com/terraform/downloads"
        $hasErrors = $true
    }

    # Check kubectl
    Write-Step "Checking kubectl..."
    if (Test-CommandExists "kubectl") {
        $kubectlVersion = & kubectl version --client --short 2>&1
        if ($LASTEXITCODE -ne 0) {
            $kubectlVersion = & kubectl version --client 2>&1 | Select-Object -First 1
        }
        Write-Success "kubectl installed: $kubectlVersion"
    }
    else {
        Write-Warning "kubectl is not installed (required for EKS management)"
        Write-Host "  Install: https://kubernetes.io/docs/tasks/tools/"
    }

    if ($hasErrors) {
        Write-Error "Prerequisites check failed. Please install required tools and try again."
        exit 1
    }

    Write-Host ""
    Write-Success "All required prerequisites are installed"
}

# -----------------------------------------------------------------------------
# Validate AWS Credentials
# -----------------------------------------------------------------------------
function Test-AwsCredentials {
    Write-Header "Validating AWS Credentials"

    Write-Step "Checking AWS credentials..."

    try {
        $callerIdentity = & aws sts get-caller-identity --output json 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw $callerIdentity
        }

        $identity = $callerIdentity | ConvertFrom-Json

        $script:AwsAccountId = $identity.Account
        $userArn = $identity.Arn
        $userId = $identity.UserId

        Write-Success "AWS credentials are valid"
        Write-Host ""
        Write-Host "  Account ID: $($script:AwsAccountId)"
        Write-Host "  User ARN:   $userArn"
        Write-Host "  User ID:    $userId"
        Write-Host "  Region:     $Region"

        # Check if AWS_ACCOUNT_ID environment variable matches
        if (-not [string]::IsNullOrEmpty($env:AWS_ACCOUNT_ID) -and $env:AWS_ACCOUNT_ID -ne $script:AwsAccountId) {
            Write-Warning "AWS_ACCOUNT_ID ($($env:AWS_ACCOUNT_ID)) does not match authenticated account ($($script:AwsAccountId))"
            Write-Host ""
            $confirm = Read-Host "Use authenticated account ($($script:AwsAccountId))? [Y/n]"
            if ($confirm -eq "n" -or $confirm -eq "N") {
                Write-Error "Account ID mismatch. Please verify your AWS credentials."
                exit 1
            }
        }

        Write-Host ""
        Write-Success "Using AWS Account: $($script:AwsAccountId)"
    }
    catch {
        Write-Error "Failed to validate AWS credentials"
        Write-Host ""
        Write-Host "Please ensure you have valid AWS credentials configured:"
        Write-Host "  Option 1: Run 'aws configure' to set up credentials"
        Write-Host "  Option 2: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
        Write-Host "  Option 3: Use AWS SSO with 'aws sso login'"
        Write-Host "  Option 4: Use an IAM role (for EC2 instances or ECS tasks)"
        Write-Host ""
        Write-Host "Error: $_"
        exit 1
    }
}

# -----------------------------------------------------------------------------
# Create S3 Bucket for Terraform State
# -----------------------------------------------------------------------------
function New-StateBucket {
    Write-Header "Creating S3 Bucket for Terraform State"

    Write-Step "Checking if bucket '$StateBucketName' exists..."

    $bucketExists = $false
    try {
        & aws s3api head-bucket --bucket $StateBucketName 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $bucketExists = $true
        }
    }
    catch {
        $bucketExists = $false
    }

    if ($bucketExists) {
        Write-Success "Bucket '$StateBucketName' already exists"
    }
    else {
        Write-Step "Creating bucket '$StateBucketName'..."

        # Create bucket (different command for us-east-1 vs other regions)
        if ($Region -eq "us-east-1") {
            & aws s3api create-bucket `
                --bucket $StateBucketName `
                --region $Region
        }
        else {
            & aws s3api create-bucket `
                --bucket $StateBucketName `
                --region $Region `
                --create-bucket-configuration LocationConstraint=$Region
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create bucket"
            exit 1
        }

        Write-Success "Bucket '$StateBucketName' created"
    }

    # Enable versioning
    Write-Step "Enabling versioning on bucket..."
    & aws s3api put-bucket-versioning `
        --bucket $StateBucketName `
        --versioning-configuration Status=Enabled

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to enable versioning"
        exit 1
    }
    Write-Success "Versioning enabled"

    # Enable server-side encryption
    Write-Step "Enabling server-side encryption..."
    $encryptionConfig = @{
        Rules = @(
            @{
                ApplyServerSideEncryptionByDefault = @{
                    SSEAlgorithm = "AES256"
                }
                BucketKeyEnabled = $true
            }
        )
    } | ConvertTo-Json -Depth 10 -Compress

    & aws s3api put-bucket-encryption `
        --bucket $StateBucketName `
        --server-side-encryption-configuration $encryptionConfig

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to enable encryption"
        exit 1
    }
    Write-Success "Server-side encryption enabled (AES256)"

    # Block public access
    Write-Step "Blocking public access..."
    $publicAccessConfig = @{
        BlockPublicAcls = $true
        IgnorePublicAcls = $true
        BlockPublicPolicy = $true
        RestrictPublicBuckets = $true
    } | ConvertTo-Json -Compress

    & aws s3api put-public-access-block `
        --bucket $StateBucketName `
        --public-access-block-configuration $publicAccessConfig

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to block public access"
        exit 1
    }
    Write-Success "Public access blocked"

    # Add bucket policy for secure transport
    Write-Step "Adding bucket policy for secure transport..."
    $bucketPolicy = @{
        Version = "2012-10-17"
        Statement = @(
            @{
                Sid = "EnforceSecureTransport"
                Effect = "Deny"
                Principal = "*"
                Action = "s3:*"
                Resource = @(
                    "arn:aws:s3:::$StateBucketName",
                    "arn:aws:s3:::$StateBucketName/*"
                )
                Condition = @{
                    Bool = @{
                        "aws:SecureTransport" = "false"
                    }
                }
            }
        )
    } | ConvertTo-Json -Depth 10 -Compress

    & aws s3api put-bucket-policy `
        --bucket $StateBucketName `
        --policy $bucketPolicy

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to add bucket policy"
        exit 1
    }
    Write-Success "Bucket policy applied (HTTPS only)"

    # Add tags
    Write-Step "Adding tags to bucket..."
    $tagging = @{
        TagSet = @(
            @{ Key = "Project"; Value = "GreenLang" }
            @{ Key = "Purpose"; Value = "Terraform State" }
            @{ Key = "ManagedBy"; Value = "infra-init.ps1" }
        )
    } | ConvertTo-Json -Depth 10 -Compress

    & aws s3api put-bucket-tagging `
        --bucket $StateBucketName `
        --tagging $tagging

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to add tags"
        exit 1
    }
    Write-Success "Tags added"

    Write-Host ""
    Write-Success "S3 bucket '$StateBucketName' is ready for Terraform state"
}

# -----------------------------------------------------------------------------
# Create DynamoDB Table for State Locking
# -----------------------------------------------------------------------------
function New-LockTable {
    Write-Header "Creating DynamoDB Table for State Locking"

    Write-Step "Checking if table '$LockTableName' exists..."

    $tableExists = $false
    $tableStatus = $null

    try {
        $tableStatus = & aws dynamodb describe-table `
            --table-name $LockTableName `
            --region $Region `
            --query 'Table.TableStatus' `
            --output text 2>&1

        if ($LASTEXITCODE -eq 0) {
            $tableExists = $true
        }
    }
    catch {
        $tableExists = $false
    }

    if ($tableExists) {
        Write-Success "Table '$LockTableName' already exists (status: $tableStatus)"

        # Wait for table to be active if it's creating
        if ($tableStatus -eq "CREATING") {
            Write-Step "Waiting for table to become active..."
            & aws dynamodb wait table-exists --table-name $LockTableName --region $Region
            Write-Success "Table is now active"
        }
    }
    else {
        Write-Step "Creating table '$LockTableName'..."

        & aws dynamodb create-table `
            --table-name $LockTableName `
            --attribute-definitions AttributeName=LockID,AttributeType=S `
            --key-schema AttributeName=LockID,KeyType=HASH `
            --billing-mode PAY_PER_REQUEST `
            --region $Region `
            --tags Key=Project,Value=GreenLang Key=Purpose,Value="Terraform State Locking" Key=ManagedBy,Value=infra-init.ps1 `
            | Out-Null

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create DynamoDB table"
            exit 1
        }

        Write-Step "Waiting for table to become active..."
        & aws dynamodb wait table-exists --table-name $LockTableName --region $Region
        Write-Success "Table '$LockTableName' created and active"
    }

    # Enable point-in-time recovery
    Write-Step "Enabling point-in-time recovery..."
    try {
        & aws dynamodb update-continuous-backups `
            --table-name $LockTableName `
            --region $Region `
            --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true `
            2>&1 | Out-Null
        Write-Success "Point-in-time recovery configured"
    }
    catch {
        Write-Info "Point-in-time recovery already enabled or not available"
    }

    Write-Host ""
    Write-Success "DynamoDB table '$LockTableName' is ready for state locking"
}

# -----------------------------------------------------------------------------
# Run Terraform Init
# -----------------------------------------------------------------------------
function Invoke-TerraformInit {
    Write-Header "Running Terraform Init"

    # Determine script directory and terraform environment path
    $scriptDir = Split-Path -Parent $MyInvocation.ScriptName
    if ([string]::IsNullOrEmpty($scriptDir)) {
        $scriptDir = $PSScriptRoot
    }
    if ([string]::IsNullOrEmpty($scriptDir)) {
        $scriptDir = (Get-Location).Path
    }

    $tfEnvDir = Join-Path -Path $scriptDir -ChildPath "..\environments\$Environment"
    $tfEnvDir = [System.IO.Path]::GetFullPath($tfEnvDir)

    if (-not (Test-Path $tfEnvDir)) {
        Write-Error "Terraform environment directory not found: $tfEnvDir"
        Write-Host ""
        Write-Host "Available environments:"
        $envDir = Join-Path -Path $scriptDir -ChildPath "..\environments"
        if (Test-Path $envDir) {
            Get-ChildItem -Path $envDir -Directory | ForEach-Object { Write-Host "  - $($_.Name)" }
        }
        else {
            Write-Host "  (none found)"
        }
        exit 1
    }

    Write-Step "Initializing Terraform in: $tfEnvDir"
    Write-Host ""

    Push-Location $tfEnvDir
    try {
        & terraform init `
            -backend-config="bucket=$StateBucketName" `
            -backend-config="region=$Region" `
            -backend-config="dynamodb_table=$LockTableName" `
            -backend-config="encrypt=true"

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Terraform init failed"
            exit 1
        }

        Write-Host ""
        Write-Success "Terraform initialized successfully for $Environment environment"
    }
    finally {
        Pop-Location
    }
}

# -----------------------------------------------------------------------------
# Print Summary and Next Steps
# -----------------------------------------------------------------------------
function Write-Summary {
    Write-Header "Infrastructure Initialization Complete"

    Write-Host "Summary:"
    Write-Host "  AWS Account:      $($script:AwsAccountId)"
    Write-Host "  AWS Region:       $Region"
    Write-Host "  Environment:      $Environment"
    Write-Host "  State Bucket:     $StateBucketName"
    Write-Host "  Lock Table:       $LockTableName"
    Write-Host ""

    Write-Host "Next Steps:" -ForegroundColor Green
    Write-Host ""
    Write-Host "1. Review the Terraform configuration:"
    Write-Host "   cd deployment\terraform\environments\$Environment"
    Write-Host "   Get-Content terraform.tfvars"
    Write-Host ""
    Write-Host "2. Preview the infrastructure changes:"
    Write-Host "   terraform plan -out=tfplan"
    Write-Host ""
    Write-Host "3. Apply the infrastructure (when ready):"
    Write-Host "   terraform apply tfplan"
    Write-Host ""
    Write-Host "4. Configure kubectl for EKS (after apply):"
    Write-Host "   aws eks update-kubeconfig --name greenlang-$Environment-eks --region $Region"
    Write-Host ""
    Write-Host "5. Deploy applications to Kubernetes:"
    Write-Host "   cd ..\..\kubernetes"
    Write-Host "   kubectl apply -k overlays\$Environment"
    Write-Host ""

    Write-Info "For production deployments, ensure you:"
    Write-Host "  - Review and customize terraform.tfvars"
    Write-Host "  - Set up proper IAM roles and policies"
    Write-Host "  - Configure monitoring and alerting"
    Write-Host "  - Enable AWS CloudTrail for auditing"
    Write-Host "  - Set up AWS Config for compliance"
    Write-Host ""

    Write-Success "GreenLang infrastructure initialization complete!"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
function Main {
    Write-Header "GreenLang Infrastructure Initialization"

    Write-Host "Configuration:"
    Write-Host "  AWS Region:     $Region"
    Write-Host "  Environment:    $Environment"
    Write-Host "  State Bucket:   $StateBucketName"
    Write-Host "  Lock Table:     $LockTableName"
    Write-Host ""

    # Run initialization steps
    Test-Prerequisites
    Test-AwsCredentials
    New-StateBucket
    New-LockTable

    if (-not $SkipTerraformInit) {
        Invoke-TerraformInit
    }
    else {
        Write-Info "Skipping terraform init (-SkipTerraformInit flag set)"
    }

    Write-Summary
}

# Run main
Main
