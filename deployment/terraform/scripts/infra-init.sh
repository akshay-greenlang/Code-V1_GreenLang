#!/bin/bash
# ============================================================================
# GreenLang Infrastructure Initialization Script
# ============================================================================
# This script initializes the AWS infrastructure required for Terraform state
# management and performs the initial terraform init.
#
# Prerequisites:
#   - AWS CLI installed and configured
#   - Terraform >= 1.0 installed
#   - kubectl installed (for EKS management)
#
# Usage:
#   ./infra-init.sh [--region REGION] [--environment ENV] [--skip-init]
#
# Options:
#   --region       AWS region (default: us-east-1)
#   --environment  Target environment: dev, staging, prod (default: dev)
#   --skip-init    Skip terraform init after creating backend resources
#   --help         Show this help message
#
# This script is idempotent - safe to run multiple times.
# ============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
AWS_REGION="${AWS_REGION:-us-east-1}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
SKIP_TERRAFORM_INIT=false

STATE_BUCKET_NAME="greenlang-terraform-state"
LOCK_TABLE_NAME="greenlang-terraform-locks"

# Colors for output (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
fi

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
print_header() {
    echo ""
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

show_help() {
    echo "GreenLang Infrastructure Initialization Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --region REGION        AWS region (default: us-east-1)"
    echo "  --environment ENV      Target environment: dev, staging, prod (default: dev)"
    echo "  --skip-init            Skip terraform init after creating backend resources"
    echo "  --help                 Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  AWS_REGION             AWS region (overridden by --region)"
    echo "  AWS_ACCOUNT_ID         AWS account ID (will prompt if not set)"
    echo "  ENVIRONMENT            Target environment (overridden by --environment)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Initialize with defaults"
    echo "  $0 --region us-west-2                 # Use us-west-2 region"
    echo "  $0 --environment prod --skip-init    # Prod environment, skip terraform init"
    echo ""
}

# -----------------------------------------------------------------------------
# Parse Command Line Arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --skip-init)
            SKIP_TERRAFORM_INIT=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT. Must be one of: dev, staging, prod"
    exit 1
fi

# -----------------------------------------------------------------------------
# Check Prerequisites
# -----------------------------------------------------------------------------
check_prerequisites() {
    print_header "Checking Prerequisites"

    local has_errors=false

    # Check AWS CLI
    print_step "Checking AWS CLI..."
    if command -v aws &> /dev/null; then
        local aws_version
        aws_version=$(aws --version 2>&1 | head -n1)
        print_success "AWS CLI installed: $aws_version"
    else
        print_error "AWS CLI is not installed"
        echo "  Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        has_errors=true
    fi

    # Check Terraform
    print_step "Checking Terraform..."
    if command -v terraform &> /dev/null; then
        local tf_version
        tf_version=$(terraform version -json 2>/dev/null | grep -o '"terraform_version":"[^"]*"' | cut -d'"' -f4 2>/dev/null || terraform version | head -n1)
        print_success "Terraform installed: $tf_version"

        # Check minimum version
        local required_version="1.0.0"
        local current_version
        current_version=$(terraform version -json 2>/dev/null | grep -o '"terraform_version":"[^"]*"' | cut -d'"' -f4 2>/dev/null || terraform version | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        if [ "$(printf '%s\n' "$required_version" "$current_version" | sort -V | head -n1)" != "$required_version" ]; then
            print_warning "Terraform version $current_version may be too old. Required: >= $required_version"
        fi
    else
        print_error "Terraform is not installed"
        echo "  Install: https://developer.hashicorp.com/terraform/downloads"
        has_errors=true
    fi

    # Check kubectl
    print_step "Checking kubectl..."
    if command -v kubectl &> /dev/null; then
        local kubectl_version
        kubectl_version=$(kubectl version --client --short 2>/dev/null || kubectl version --client 2>/dev/null | head -n1)
        print_success "kubectl installed: $kubectl_version"
    else
        print_warning "kubectl is not installed (required for EKS management)"
        echo "  Install: https://kubernetes.io/docs/tasks/tools/"
    fi

    # Check jq (optional but helpful)
    print_step "Checking jq (optional)..."
    if command -v jq &> /dev/null; then
        print_success "jq installed: $(jq --version)"
    else
        print_info "jq is not installed (optional, but recommended)"
        echo "  Install: https://stedolan.github.io/jq/download/"
    fi

    if [ "$has_errors" = true ]; then
        print_error "Prerequisites check failed. Please install required tools and try again."
        exit 1
    fi

    echo ""
    print_success "All required prerequisites are installed"
}

# -----------------------------------------------------------------------------
# Validate AWS Credentials
# -----------------------------------------------------------------------------
validate_aws_credentials() {
    print_header "Validating AWS Credentials"

    print_step "Checking AWS credentials..."

    local caller_identity
    if ! caller_identity=$(aws sts get-caller-identity --output json 2>&1); then
        print_error "Failed to validate AWS credentials"
        echo ""
        echo "Please ensure you have valid AWS credentials configured:"
        echo "  Option 1: Run 'aws configure' to set up credentials"
        echo "  Option 2: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
        echo "  Option 3: Use AWS SSO with 'aws sso login'"
        echo "  Option 4: Use an IAM role (for EC2 instances or ECS tasks)"
        echo ""
        echo "Error: $caller_identity"
        exit 1
    fi

    # Extract account information
    local account_id
    local user_arn
    local user_id

    if command -v jq &> /dev/null; then
        account_id=$(echo "$caller_identity" | jq -r '.Account')
        user_arn=$(echo "$caller_identity" | jq -r '.Arn')
        user_id=$(echo "$caller_identity" | jq -r '.UserId')
    else
        account_id=$(echo "$caller_identity" | grep -oP '"Account":\s*"\K[^"]+')
        user_arn=$(echo "$caller_identity" | grep -oP '"Arn":\s*"\K[^"]+')
        user_id=$(echo "$caller_identity" | grep -oP '"UserId":\s*"\K[^"]+')
    fi

    print_success "AWS credentials are valid"
    echo ""
    echo "  Account ID: $account_id"
    echo "  User ARN:   $user_arn"
    echo "  User ID:    $user_id"
    echo "  Region:     $AWS_REGION"

    # Set AWS_ACCOUNT_ID if not already set
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        export AWS_ACCOUNT_ID="$account_id"
    elif [ "$AWS_ACCOUNT_ID" != "$account_id" ]; then
        print_warning "AWS_ACCOUNT_ID ($AWS_ACCOUNT_ID) does not match authenticated account ($account_id)"
        echo ""
        read -p "Use authenticated account ($account_id)? [Y/n] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            print_error "Account ID mismatch. Please verify your AWS credentials."
            exit 1
        fi
        export AWS_ACCOUNT_ID="$account_id"
    fi

    echo ""
    print_success "Using AWS Account: $AWS_ACCOUNT_ID"
}

# -----------------------------------------------------------------------------
# Create S3 Bucket for Terraform State
# -----------------------------------------------------------------------------
create_state_bucket() {
    print_header "Creating S3 Bucket for Terraform State"

    print_step "Checking if bucket '$STATE_BUCKET_NAME' exists..."

    if aws s3api head-bucket --bucket "$STATE_BUCKET_NAME" 2>/dev/null; then
        print_success "Bucket '$STATE_BUCKET_NAME' already exists"
    else
        print_step "Creating bucket '$STATE_BUCKET_NAME'..."

        # Create bucket (different command for us-east-1 vs other regions)
        if [ "$AWS_REGION" = "us-east-1" ]; then
            aws s3api create-bucket \
                --bucket "$STATE_BUCKET_NAME" \
                --region "$AWS_REGION"
        else
            aws s3api create-bucket \
                --bucket "$STATE_BUCKET_NAME" \
                --region "$AWS_REGION" \
                --create-bucket-configuration LocationConstraint="$AWS_REGION"
        fi

        print_success "Bucket '$STATE_BUCKET_NAME' created"
    fi

    # Enable versioning
    print_step "Enabling versioning on bucket..."
    aws s3api put-bucket-versioning \
        --bucket "$STATE_BUCKET_NAME" \
        --versioning-configuration Status=Enabled
    print_success "Versioning enabled"

    # Enable server-side encryption
    print_step "Enabling server-side encryption..."
    aws s3api put-bucket-encryption \
        --bucket "$STATE_BUCKET_NAME" \
        --server-side-encryption-configuration '{
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "AES256"
                    },
                    "BucketKeyEnabled": true
                }
            ]
        }'
    print_success "Server-side encryption enabled (AES256)"

    # Block public access
    print_step "Blocking public access..."
    aws s3api put-public-access-block \
        --bucket "$STATE_BUCKET_NAME" \
        --public-access-block-configuration '{
            "BlockPublicAcls": true,
            "IgnorePublicAcls": true,
            "BlockPublicPolicy": true,
            "RestrictPublicBuckets": true
        }'
    print_success "Public access blocked"

    # Add bucket policy for secure transport
    print_step "Adding bucket policy for secure transport..."
    local bucket_policy=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "EnforceSecureTransport",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::${STATE_BUCKET_NAME}",
                "arn:aws:s3:::${STATE_BUCKET_NAME}/*"
            ],
            "Condition": {
                "Bool": {
                    "aws:SecureTransport": "false"
                }
            }
        }
    ]
}
EOF
)
    aws s3api put-bucket-policy \
        --bucket "$STATE_BUCKET_NAME" \
        --policy "$bucket_policy"
    print_success "Bucket policy applied (HTTPS only)"

    # Add tags
    print_step "Adding tags to bucket..."
    aws s3api put-bucket-tagging \
        --bucket "$STATE_BUCKET_NAME" \
        --tagging '{
            "TagSet": [
                {"Key": "Project", "Value": "GreenLang"},
                {"Key": "Purpose", "Value": "Terraform State"},
                {"Key": "ManagedBy", "Value": "infra-init.sh"}
            ]
        }'
    print_success "Tags added"

    echo ""
    print_success "S3 bucket '$STATE_BUCKET_NAME' is ready for Terraform state"
}

# -----------------------------------------------------------------------------
# Create DynamoDB Table for State Locking
# -----------------------------------------------------------------------------
create_lock_table() {
    print_header "Creating DynamoDB Table for State Locking"

    print_step "Checking if table '$LOCK_TABLE_NAME' exists..."

    local table_status
    if table_status=$(aws dynamodb describe-table --table-name "$LOCK_TABLE_NAME" --region "$AWS_REGION" --query 'Table.TableStatus' --output text 2>/dev/null); then
        print_success "Table '$LOCK_TABLE_NAME' already exists (status: $table_status)"

        # Wait for table to be active if it's creating
        if [ "$table_status" = "CREATING" ]; then
            print_step "Waiting for table to become active..."
            aws dynamodb wait table-exists --table-name "$LOCK_TABLE_NAME" --region "$AWS_REGION"
            print_success "Table is now active"
        fi
    else
        print_step "Creating table '$LOCK_TABLE_NAME'..."

        aws dynamodb create-table \
            --table-name "$LOCK_TABLE_NAME" \
            --attribute-definitions AttributeName=LockID,AttributeType=S \
            --key-schema AttributeName=LockID,KeyType=HASH \
            --billing-mode PAY_PER_REQUEST \
            --region "$AWS_REGION" \
            --tags Key=Project,Value=GreenLang Key=Purpose,Value="Terraform State Locking" Key=ManagedBy,Value=infra-init.sh \
            > /dev/null

        print_step "Waiting for table to become active..."
        aws dynamodb wait table-exists --table-name "$LOCK_TABLE_NAME" --region "$AWS_REGION"
        print_success "Table '$LOCK_TABLE_NAME' created and active"
    fi

    # Enable point-in-time recovery
    print_step "Enabling point-in-time recovery..."
    aws dynamodb update-continuous-backups \
        --table-name "$LOCK_TABLE_NAME" \
        --region "$AWS_REGION" \
        --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true \
        > /dev/null 2>&1 || print_info "Point-in-time recovery already enabled or not available"
    print_success "Point-in-time recovery configured"

    echo ""
    print_success "DynamoDB table '$LOCK_TABLE_NAME' is ready for state locking"
}

# -----------------------------------------------------------------------------
# Run Terraform Init
# -----------------------------------------------------------------------------
run_terraform_init() {
    print_header "Running Terraform Init"

    # Determine script directory and terraform environment path
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local tf_env_dir="$script_dir/../environments/$ENVIRONMENT"

    if [ ! -d "$tf_env_dir" ]; then
        print_error "Terraform environment directory not found: $tf_env_dir"
        echo ""
        echo "Available environments:"
        ls -1 "$script_dir/../environments/" 2>/dev/null || echo "  (none found)"
        exit 1
    fi

    print_step "Initializing Terraform in: $tf_env_dir"
    echo ""

    cd "$tf_env_dir"

    # Run terraform init with backend configuration
    terraform init \
        -backend-config="bucket=$STATE_BUCKET_NAME" \
        -backend-config="region=$AWS_REGION" \
        -backend-config="dynamodb_table=$LOCK_TABLE_NAME" \
        -backend-config="encrypt=true"

    echo ""
    print_success "Terraform initialized successfully for $ENVIRONMENT environment"
}

# -----------------------------------------------------------------------------
# Print Summary and Next Steps
# -----------------------------------------------------------------------------
print_summary() {
    print_header "Infrastructure Initialization Complete"

    echo "Summary:"
    echo "  AWS Account:      $AWS_ACCOUNT_ID"
    echo "  AWS Region:       $AWS_REGION"
    echo "  Environment:      $ENVIRONMENT"
    echo "  State Bucket:     $STATE_BUCKET_NAME"
    echo "  Lock Table:       $LOCK_TABLE_NAME"
    echo ""

    echo -e "${GREEN}Next Steps:${NC}"
    echo ""
    echo "1. Review the Terraform configuration:"
    echo "   cd deployment/terraform/environments/$ENVIRONMENT"
    echo "   cat terraform.tfvars"
    echo ""
    echo "2. Preview the infrastructure changes:"
    echo "   terraform plan -out=tfplan"
    echo ""
    echo "3. Apply the infrastructure (when ready):"
    echo "   terraform apply tfplan"
    echo ""
    echo "4. Configure kubectl for EKS (after apply):"
    echo "   aws eks update-kubeconfig --name greenlang-$ENVIRONMENT-eks --region $AWS_REGION"
    echo ""
    echo "5. Deploy applications to Kubernetes:"
    echo "   cd ../../kubernetes"
    echo "   kubectl apply -k overlays/$ENVIRONMENT"
    echo ""

    print_info "For production deployments, ensure you:"
    echo "  - Review and customize terraform.tfvars"
    echo "  - Set up proper IAM roles and policies"
    echo "  - Configure monitoring and alerting"
    echo "  - Enable AWS CloudTrail for auditing"
    echo "  - Set up AWS Config for compliance"
    echo ""

    print_success "GreenLang infrastructure initialization complete!"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    print_header "GreenLang Infrastructure Initialization"

    echo "Configuration:"
    echo "  AWS Region:     $AWS_REGION"
    echo "  Environment:    $ENVIRONMENT"
    echo "  State Bucket:   $STATE_BUCKET_NAME"
    echo "  Lock Table:     $LOCK_TABLE_NAME"
    echo ""

    # Run initialization steps
    check_prerequisites
    validate_aws_credentials
    create_state_bucket
    create_lock_table

    if [ "$SKIP_TERRAFORM_INIT" = false ]; then
        run_terraform_init
    else
        print_info "Skipping terraform init (--skip-init flag set)"
    fi

    print_summary
}

# Run main
main
