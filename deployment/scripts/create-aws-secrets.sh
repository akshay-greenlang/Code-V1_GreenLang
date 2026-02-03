#!/bin/bash
#===============================================================================
# GreenLang Platform - AWS Secrets Manager Setup Script
# INFRA-001: Production Secrets Infrastructure
#===============================================================================
#
# This script creates all required secrets in AWS Secrets Manager for the
# GreenLang platform. It uses placeholder values that MUST be replaced with
# actual credentials before deploying to any environment.
#
# Usage:
#   ./create-aws-secrets.sh [environment] [--dry-run]
#
# Arguments:
#   environment  - Target environment: dev, staging, prod (default: dev)
#   --dry-run    - Show commands without executing them
#
# Prerequisites:
#   - AWS CLI v2 installed and configured
#   - Appropriate IAM permissions for Secrets Manager
#   - jq installed for JSON processing
#
# Security Notes:
#   - NEVER commit actual secret values to version control
#   - Use AWS Secrets Manager console or parameter store for sensitive data
#   - Enable automatic rotation where supported
#   - Review KMS key permissions before deployment
#
#===============================================================================

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_DIR="${SCRIPT_DIR}/../secrets"

# Default values
ENVIRONMENT="${1:-dev}"
DRY_RUN=false
AWS_REGION="${AWS_REGION:-us-east-1}"
KMS_KEY_ALIAS="alias/greenlang-secrets-${ENVIRONMENT}"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        dev|staging|prod)
            ENVIRONMENT="$arg"
            shift
            ;;
        *)
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    echo -e "${RED}Error: Invalid environment '$ENVIRONMENT'. Must be dev, staging, or prod.${NC}"
    exit 1
fi

# Secret prefix based on environment
SECRET_PREFIX="greenlang-${ENVIRONMENT}"

#-------------------------------------------------------------------------------
# Helper Functions
#-------------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_aws_command() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} aws $*"
    else
        aws "$@"
    fi
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi

    # Check jq
    if ! command -v jq &> /dev/null; then
        log_error "jq is not installed. Please install it first."
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid."
        exit 1
    fi

    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    log_success "AWS Account: $AWS_ACCOUNT_ID"
    log_success "Region: $AWS_REGION"
    log_success "Environment: $ENVIRONMENT"
}

create_kms_key() {
    log_info "Creating/verifying KMS key for secrets encryption..."

    # Check if key alias exists
    if aws kms describe-key --key-id "$KMS_KEY_ALIAS" --region "$AWS_REGION" &> /dev/null; then
        log_info "KMS key alias $KMS_KEY_ALIAS already exists"
        KMS_KEY_ID=$(aws kms describe-key --key-id "$KMS_KEY_ALIAS" --region "$AWS_REGION" --query 'KeyMetadata.KeyId' --output text)
    else
        log_info "Creating new KMS key..."

        KMS_KEY_ID=$(run_aws_command kms create-key \
            --description "GreenLang Secrets Manager encryption key - ${ENVIRONMENT}" \
            --key-usage ENCRYPT_DECRYPT \
            --key-spec SYMMETRIC_DEFAULT \
            --tags TagKey=Environment,TagValue="${ENVIRONMENT}" TagKey=Application,TagValue=greenlang \
            --region "$AWS_REGION" \
            --query 'KeyMetadata.KeyId' \
            --output text 2>/dev/null || echo "dry-run-key-id")

        # Create alias
        run_aws_command kms create-alias \
            --alias-name "$KMS_KEY_ALIAS" \
            --target-key-id "$KMS_KEY_ID" \
            --region "$AWS_REGION"

        # Enable key rotation
        run_aws_command kms enable-key-rotation \
            --key-id "$KMS_KEY_ID" \
            --region "$AWS_REGION"

        log_success "Created KMS key: $KMS_KEY_ID"
    fi
}

create_secret() {
    local secret_name="$1"
    local secret_description="$2"
    local secret_value="$3"
    local tags="$4"

    log_info "Creating secret: $secret_name"

    # Check if secret already exists
    if aws secretsmanager describe-secret --secret-id "$secret_name" --region "$AWS_REGION" &> /dev/null; then
        log_warning "Secret $secret_name already exists. Updating..."

        run_aws_command secretsmanager update-secret \
            --secret-id "$secret_name" \
            --secret-string "$secret_value" \
            --region "$AWS_REGION"
    else
        run_aws_command secretsmanager create-secret \
            --name "$secret_name" \
            --description "$secret_description" \
            --kms-key-id "$KMS_KEY_ALIAS" \
            --secret-string "$secret_value" \
            --tags "$tags" \
            --region "$AWS_REGION"
    fi

    log_success "Secret created/updated: $secret_name"
}

#-------------------------------------------------------------------------------
# Secret Definitions
# IMPORTANT: Replace placeholder values with actual secrets before running!
#-------------------------------------------------------------------------------

create_database_secret() {
    local secret_name="${SECRET_PREFIX}/database"
    local description="PostgreSQL database credentials and connection configuration"

    # WARNING: Replace these placeholder values with actual credentials!
    local secret_value=$(cat <<EOF
{
    "connection_string": "postgresql://greenlang_admin:REPLACE_WITH_PASSWORD@greenlang-db.cluster-xxx.${AWS_REGION}.rds.amazonaws.com:5432/greenlang_${ENVIRONMENT}",
    "host": "greenlang-db.cluster-xxx.${AWS_REGION}.rds.amazonaws.com",
    "port": "5432",
    "database": "greenlang_${ENVIRONMENT}",
    "username": "greenlang_admin",
    "password": "REPLACE_WITH_SECURE_PASSWORD_MIN_32_CHARS",
    "ssl_mode": "require",
    "pool_size": "20",
    "max_overflow": "40",
    "connection_timeout": "30",
    "read_replica_host": "greenlang-db-reader.cluster-xxx.${AWS_REGION}.rds.amazonaws.com"
}
EOF
)

    local tags="Key=Environment,Value=${ENVIRONMENT} Key=Application,Value=greenlang Key=Component,Value=database Key=ManagedBy,Value=script"

    create_secret "$secret_name" "$description" "$secret_value" "$tags"
}

create_redis_secret() {
    local secret_name="${SECRET_PREFIX}/redis"
    local description="Redis/ElastiCache credentials for caching and session management"

    # WARNING: Replace these placeholder values with actual credentials!
    local secret_value=$(cat <<EOF
{
    "password": "REPLACE_WITH_SECURE_REDIS_PASSWORD_MIN_32_CHARS",
    "auth_token": "REPLACE_WITH_SECURE_AUTH_TOKEN_64_CHARS",
    "host": "greenlang-redis.xxx.ng.0001.${AWS_REGION}.cache.amazonaws.com",
    "port": "6379",
    "ssl_enabled": "true",
    "cluster_mode": "false",
    "sentinel_host": "greenlang-redis-sentinel.xxx.${AWS_REGION}.cache.amazonaws.com",
    "sentinel_port": "26379",
    "database": "0",
    "connection_timeout": "5",
    "socket_timeout": "5",
    "max_connections": "100"
}
EOF
)

    local tags="Key=Environment,Value=${ENVIRONMENT} Key=Application,Value=greenlang Key=Component,Value=cache Key=ManagedBy,Value=script"

    create_secret "$secret_name" "$description" "$secret_value" "$tags"
}

create_api_keys_secret() {
    local secret_name="${SECRET_PREFIX}/api-keys"
    local description="Third-party API keys for AI services and external integrations"

    # WARNING: Replace these placeholder values with actual API keys!
    local secret_value=$(cat <<EOF
{
    "openai_api_key": "sk-REPLACE_WITH_OPENAI_API_KEY",
    "openai_org_id": "org-REPLACE_WITH_ORG_ID",
    "anthropic_api_key": "sk-ant-REPLACE_WITH_ANTHROPIC_API_KEY",
    "pinecone_api_key": "REPLACE_WITH_PINECONE_API_KEY",
    "pinecone_environment": "${AWS_REGION}-aws",
    "pinecone_index_name": "greenlang-embeddings-${ENVIRONMENT}",
    "cohere_api_key": "REPLACE_WITH_COHERE_API_KEY",
    "huggingface_api_key": "hf_REPLACE_WITH_HF_API_KEY",
    "weaviate_api_key": "REPLACE_WITH_WEAVIATE_API_KEY",
    "google_ai_api_key": "REPLACE_WITH_GOOGLE_AI_API_KEY",
    "azure_openai_api_key": "REPLACE_WITH_AZURE_OPENAI_KEY",
    "azure_openai_endpoint": "https://greenlang-${ENVIRONMENT}.openai.azure.com/",
    "sendgrid_api_key": "SG.REPLACE_WITH_SENDGRID_KEY",
    "stripe_api_key": "sk_live_REPLACE_WITH_STRIPE_KEY",
    "stripe_webhook_secret": "whsec_REPLACE_WITH_WEBHOOK_SECRET"
}
EOF
)

    local tags="Key=Environment,Value=${ENVIRONMENT} Key=Application,Value=greenlang Key=Component,Value=api-keys Key=ManagedBy,Value=script Key=Sensitivity,Value=high"

    create_secret "$secret_name" "$description" "$secret_value" "$tags"
}

create_runner_secret() {
    local secret_name="${SECRET_PREFIX}/runner"
    local description="CI/CD runner tokens and container registry credentials"

    # WARNING: Replace these placeholder values with actual tokens!
    local secret_value=$(cat <<EOF
{
    "github_runner_token": "REPLACE_WITH_GITHUB_RUNNER_TOKEN",
    "github_app_id": "REPLACE_WITH_GITHUB_APP_ID",
    "github_app_private_key": "-----BEGIN RSA PRIVATE KEY-----\\nREPLACE_WITH_GITHUB_APP_PRIVATE_KEY\\n-----END RSA PRIVATE KEY-----",
    "github_webhook_secret": "REPLACE_WITH_WEBHOOK_SECRET",
    "ecr_registry_url": "${AWS_ACCOUNT_ID:-ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com",
    "ecr_access_key_id": "REPLACE_WITH_ECR_ACCESS_KEY",
    "ecr_secret_access_key": "REPLACE_WITH_ECR_SECRET_KEY",
    "ghcr_username": "greenlang-bot",
    "ghcr_token": "ghp_REPLACE_WITH_GHCR_TOKEN",
    "dockerhub_username": "greenlang",
    "dockerhub_token": "dckr_pat_REPLACE_WITH_DOCKERHUB_TOKEN",
    "sonarqube_token": "sqp_REPLACE_WITH_SONARQUBE_TOKEN",
    "codecov_token": "REPLACE_WITH_CODECOV_TOKEN",
    "snyk_token": "REPLACE_WITH_SNYK_TOKEN"
}
EOF
)

    local tags="Key=Environment,Value=${ENVIRONMENT} Key=Application,Value=greenlang Key=Component,Value=cicd Key=ManagedBy,Value=script"

    create_secret "$secret_name" "$description" "$secret_value" "$tags"
}

create_tls_secret() {
    local secret_name="${SECRET_PREFIX}/tls"
    local description="TLS/SSL certificates and private keys for secure communications"

    # WARNING: Replace these placeholder values with actual certificates!
    # Note: For production, consider using AWS Certificate Manager instead
    local secret_value=$(cat <<EOF
{
    "domain": "api.greenlang.io",
    "certificate": "-----BEGIN CERTIFICATE-----\\nREPLACE_WITH_TLS_CERTIFICATE\\n-----END CERTIFICATE-----",
    "private_key": "-----BEGIN PRIVATE KEY-----\\nREPLACE_WITH_TLS_PRIVATE_KEY\\n-----END PRIVATE KEY-----",
    "certificate_chain": "-----BEGIN CERTIFICATE-----\\nREPLACE_WITH_INTERMEDIATE_CERTIFICATE\\n-----END CERTIFICATE-----",
    "ca_certificate": "-----BEGIN CERTIFICATE-----\\nREPLACE_WITH_CA_CERTIFICATE\\n-----END CERTIFICATE-----",
    "wildcard_certificate": "-----BEGIN CERTIFICATE-----\\nREPLACE_WITH_WILDCARD_CERTIFICATE\\n-----END CERTIFICATE-----",
    "wildcard_private_key": "-----BEGIN PRIVATE KEY-----\\nREPLACE_WITH_WILDCARD_PRIVATE_KEY\\n-----END PRIVATE KEY-----",
    "internal_ca_certificate": "-----BEGIN CERTIFICATE-----\\nREPLACE_WITH_INTERNAL_CA_CERT\\n-----END CERTIFICATE-----",
    "internal_ca_private_key": "-----BEGIN PRIVATE KEY-----\\nREPLACE_WITH_INTERNAL_CA_KEY\\n-----END PRIVATE KEY-----",
    "mtls_client_certificate": "-----BEGIN CERTIFICATE-----\\nREPLACE_WITH_MTLS_CLIENT_CERT\\n-----END CERTIFICATE-----",
    "mtls_client_key": "-----BEGIN PRIVATE KEY-----\\nREPLACE_WITH_MTLS_CLIENT_KEY\\n-----END PRIVATE KEY-----"
}
EOF
)

    local tags="Key=Environment,Value=${ENVIRONMENT} Key=Application,Value=greenlang Key=Component,Value=tls Key=ManagedBy,Value=script Key=Sensitivity,Value=critical"

    create_secret "$secret_name" "$description" "$secret_value" "$tags"
}

create_jwt_secret() {
    local secret_name="${SECRET_PREFIX}/jwt"
    local description="JWT signing keys and authentication secrets"

    # WARNING: Replace these placeholder values with actual keys!
    # Generate with: openssl rand -hex 32
    local secret_value=$(cat <<EOF
{
    "jwt_secret_key": "REPLACE_WITH_256_BIT_JWT_SECRET_KEY_64_HEX_CHARS",
    "jwt_refresh_secret_key": "REPLACE_WITH_256_BIT_REFRESH_SECRET_KEY",
    "jwt_algorithm": "HS256",
    "jwt_access_token_expire_minutes": "30",
    "jwt_refresh_token_expire_days": "7",
    "rsa_private_key": "-----BEGIN RSA PRIVATE KEY-----\\nREPLACE_WITH_RSA_PRIVATE_KEY\\n-----END RSA PRIVATE KEY-----",
    "rsa_public_key": "-----BEGIN PUBLIC KEY-----\\nREPLACE_WITH_RSA_PUBLIC_KEY\\n-----END PUBLIC KEY-----",
    "encryption_key": "REPLACE_WITH_256_BIT_ENCRYPTION_KEY",
    "session_secret": "REPLACE_WITH_SESSION_SECRET_MIN_32_CHARS"
}
EOF
)

    local tags="Key=Environment,Value=${ENVIRONMENT} Key=Application,Value=greenlang Key=Component,Value=auth Key=ManagedBy,Value=script"

    create_secret "$secret_name" "$description" "$secret_value" "$tags"
}

create_monitoring_secret() {
    local secret_name="${SECRET_PREFIX}/monitoring"
    local description="Monitoring and observability service credentials"

    # WARNING: Replace these placeholder values with actual credentials!
    local secret_value=$(cat <<EOF
{
    "grafana_admin_user": "admin",
    "grafana_admin_password": "REPLACE_WITH_GRAFANA_PASSWORD",
    "prometheus_basic_auth_password": "REPLACE_WITH_PROMETHEUS_PASSWORD",
    "alertmanager_slack_webhook_url": "https://hooks.slack.com/services/REPLACE/WITH/WEBHOOK",
    "pagerduty_integration_key": "REPLACE_WITH_PAGERDUTY_KEY",
    "datadog_api_key": "REPLACE_WITH_DATADOG_API_KEY",
    "datadog_app_key": "REPLACE_WITH_DATADOG_APP_KEY",
    "newrelic_license_key": "REPLACE_WITH_NEWRELIC_LICENSE_KEY",
    "sentry_dsn": "https://REPLACE@oXXXXXX.ingest.sentry.io/XXXXXX",
    "opsgenie_api_key": "REPLACE_WITH_OPSGENIE_API_KEY"
}
EOF
)

    local tags="Key=Environment,Value=${ENVIRONMENT} Key=Application,Value=greenlang Key=Component,Value=monitoring Key=ManagedBy,Value=script"

    create_secret "$secret_name" "$description" "$secret_value" "$tags"
}

create_encryption_secret() {
    local secret_name="${SECRET_PREFIX}/encryption"
    local description="Data encryption keys for at-rest and in-transit encryption"

    # WARNING: Replace these placeholder values with actual keys!
    # Generate with: openssl rand -hex 32
    local secret_value=$(cat <<EOF
{
    "data_encryption_key": "REPLACE_WITH_256_BIT_DEK",
    "kms_key_id": "arn:aws:kms:${AWS_REGION}:${AWS_ACCOUNT_ID:-ACCOUNT_ID}:key/REPLACE-WITH-KMS-KEY-ID",
    "backup_encryption_key": "REPLACE_WITH_BACKUP_ENCRYPTION_KEY",
    "pii_encryption_key": "REPLACE_WITH_PII_ENCRYPTION_KEY",
    "csrd_encryption_key": "REPLACE_WITH_CSRD_ENCRYPTION_KEY",
    "field_level_encryption_key": "REPLACE_WITH_FLE_KEY",
    "key_derivation_salt": "REPLACE_WITH_RANDOM_SALT_32_BYTES"
}
EOF
)

    local tags="Key=Environment,Value=${ENVIRONMENT} Key=Application,Value=greenlang Key=Component,Value=encryption Key=ManagedBy,Value=script Key=Sensitivity,Value=critical"

    create_secret "$secret_name" "$description" "$secret_value" "$tags"
}

#-------------------------------------------------------------------------------
# IAM Policy Creation
#-------------------------------------------------------------------------------

create_iam_policy() {
    log_info "Creating IAM policy for secrets access..."

    local policy_name="greenlang-secrets-access-${ENVIRONMENT}"
    local policy_document=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "GetSecrets",
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue",
                "secretsmanager:DescribeSecret"
            ],
            "Resource": "arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID:-*}:secret:${SECRET_PREFIX}/*"
        },
        {
            "Sid": "DecryptSecrets",
            "Effect": "Allow",
            "Action": [
                "kms:Decrypt",
                "kms:DescribeKey"
            ],
            "Resource": "arn:aws:kms:${AWS_REGION}:${AWS_ACCOUNT_ID:-*}:key/*",
            "Condition": {
                "StringEquals": {
                    "kms:ViaService": "secretsmanager.${AWS_REGION}.amazonaws.com"
                }
            }
        }
    ]
}
EOF
)

    # Check if policy exists
    if aws iam get-policy --policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID:-000000000000}:policy/${policy_name}" &> /dev/null; then
        log_info "IAM policy ${policy_name} already exists"
    else
        run_aws_command iam create-policy \
            --policy-name "$policy_name" \
            --policy-document "$policy_document" \
            --description "Policy to access GreenLang secrets in ${ENVIRONMENT} environment"

        log_success "Created IAM policy: ${policy_name}"
    fi
}

#-------------------------------------------------------------------------------
# Verification
#-------------------------------------------------------------------------------

verify_secrets() {
    log_info "Verifying created secrets..."

    local secrets=(
        "${SECRET_PREFIX}/database"
        "${SECRET_PREFIX}/redis"
        "${SECRET_PREFIX}/api-keys"
        "${SECRET_PREFIX}/runner"
        "${SECRET_PREFIX}/tls"
        "${SECRET_PREFIX}/jwt"
        "${SECRET_PREFIX}/monitoring"
        "${SECRET_PREFIX}/encryption"
    )

    local success_count=0
    local fail_count=0

    for secret in "${secrets[@]}"; do
        if aws secretsmanager describe-secret --secret-id "$secret" --region "$AWS_REGION" &> /dev/null; then
            log_success "Verified: $secret"
            ((success_count++))
        else
            log_error "Missing: $secret"
            ((fail_count++))
        fi
    done

    echo ""
    log_info "Verification Summary:"
    log_success "  Secrets created: $success_count"
    [ $fail_count -gt 0 ] && log_error "  Secrets missing: $fail_count"
}

#-------------------------------------------------------------------------------
# Cleanup Function (for removing secrets)
#-------------------------------------------------------------------------------

cleanup_secrets() {
    log_warning "This will DELETE all GreenLang secrets in the ${ENVIRONMENT} environment!"
    read -p "Are you sure? (type 'yes' to confirm): " confirm

    if [ "$confirm" != "yes" ]; then
        log_info "Cleanup cancelled."
        return
    fi

    local secrets=(
        "${SECRET_PREFIX}/database"
        "${SECRET_PREFIX}/redis"
        "${SECRET_PREFIX}/api-keys"
        "${SECRET_PREFIX}/runner"
        "${SECRET_PREFIX}/tls"
        "${SECRET_PREFIX}/jwt"
        "${SECRET_PREFIX}/monitoring"
        "${SECRET_PREFIX}/encryption"
    )

    for secret in "${secrets[@]}"; do
        log_info "Deleting secret: $secret"
        run_aws_command secretsmanager delete-secret \
            --secret-id "$secret" \
            --force-delete-without-recovery \
            --region "$AWS_REGION" 2>/dev/null || true
    done

    log_success "Cleanup complete!"
}

#-------------------------------------------------------------------------------
# Main Execution
#-------------------------------------------------------------------------------

main() {
    echo ""
    echo "==============================================================================="
    echo "  GreenLang AWS Secrets Manager Setup - INFRA-001"
    echo "==============================================================================="
    echo ""

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN MODE - No changes will be made"
        echo ""
    fi

    check_prerequisites
    echo ""

    # Handle cleanup command
    if [[ "${2:-}" == "--cleanup" ]] || [[ "${1:-}" == "--cleanup" ]]; then
        cleanup_secrets
        exit 0
    fi

    log_info "Creating secrets for environment: ${ENVIRONMENT}"
    echo ""

    # Create KMS key first
    create_kms_key
    echo ""

    # Create all secrets
    log_info "Creating secrets..."
    echo ""

    create_database_secret
    create_redis_secret
    create_api_keys_secret
    create_runner_secret
    create_tls_secret
    create_jwt_secret
    create_monitoring_secret
    create_encryption_secret

    echo ""

    # Create IAM policy
    create_iam_policy
    echo ""

    # Verify if not dry run
    if [ "$DRY_RUN" = false ]; then
        verify_secrets
    fi

    echo ""
    echo "==============================================================================="
    echo "  IMPORTANT: Update placeholder values before deploying to production!"
    echo "==============================================================================="
    echo ""
    log_warning "The secrets have been created with PLACEHOLDER values."
    log_warning "You MUST update them with actual credentials using:"
    echo ""
    echo "  aws secretsmanager update-secret \\"
    echo "    --secret-id ${SECRET_PREFIX}/<secret-name> \\"
    echo "    --secret-string 'actual-secret-json'"
    echo ""
    echo "  Or use the AWS Console: https://console.aws.amazon.com/secretsmanager"
    echo ""
    log_success "Setup complete!"
}

# Run main function
main "$@"
