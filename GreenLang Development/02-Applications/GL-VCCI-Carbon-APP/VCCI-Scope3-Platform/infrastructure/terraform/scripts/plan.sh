#!/bin/bash
# Terraform Plan Script

set -e

echo "=========================================="
echo "Terraform Plan"
echo "=========================================="

# Check if terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "ERROR: Terraform is not installed"
    exit 1
fi

# Validate configuration
echo "Validating Terraform configuration..."
terraform validate

if [ $? -ne 0 ]; then
    echo "ERROR: Terraform validation failed"
    exit 1
fi

# Format check
echo "Checking Terraform formatting..."
terraform fmt -check -recursive

# Run plan
echo ""
echo "Running Terraform plan..."
terraform plan -out=tfplan

echo ""
echo "=========================================="
echo "Plan complete!"
echo "=========================================="
echo ""
echo "Review the plan above. To apply:"
echo "  terraform apply tfplan"
echo ""
echo "Or use the apply script:"
echo "  ./scripts/apply.sh"
