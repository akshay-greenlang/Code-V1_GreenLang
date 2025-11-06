#!/bin/bash
# Terraform Destroy Script

set -e

echo "=========================================="
echo "Terraform Destroy"
echo "=========================================="

# Check if terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "ERROR: Terraform is not installed"
    exit 1
fi

# Multiple confirmations for safety
echo ""
echo "⚠️  DANGER: This will DESTROY all infrastructure! ⚠️"
echo ""
echo "This action will:"
echo "  - Destroy the EKS cluster"
echo "  - Destroy the RDS database"
echo "  - Destroy the ElastiCache cluster"
echo "  - Delete all S3 buckets (with data)"
echo "  - Remove all networking resources"
echo ""
read -p "Type the environment name to confirm: " env_confirm

CURRENT_ENV=$(terraform workspace show)

if [ "$env_confirm" != "$CURRENT_ENV" ]; then
    echo "ERROR: Environment name doesn't match. Destroy cancelled."
    exit 1
fi

echo ""
read -p "Are you ABSOLUTELY sure? Type 'destroy-everything' to confirm: " final_confirm

if [ "$final_confirm" != "destroy-everything" ]; then
    echo "Destroy cancelled"
    exit 0
fi

# Run destroy
echo ""
echo "Running Terraform destroy..."
terraform destroy

echo ""
echo "=========================================="
echo "Destroy complete!"
echo "=========================================="
