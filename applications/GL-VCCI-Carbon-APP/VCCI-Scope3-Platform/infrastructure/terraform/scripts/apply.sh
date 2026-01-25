#!/bin/bash
# Terraform Apply Script

set -e

echo "=========================================="
echo "Terraform Apply"
echo "=========================================="

# Check if terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "ERROR: Terraform is not installed"
    exit 1
fi

# Check if plan file exists
if [ ! -f "tfplan" ]; then
    echo "ERROR: No plan file found. Run ./scripts/plan.sh first"
    exit 1
fi

# Prompt for confirmation
echo ""
echo "WARNING: This will apply changes to your infrastructure!"
echo ""
read -p "Are you sure you want to proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Apply cancelled"
    exit 0
fi

# Apply the plan
echo ""
echo "Applying Terraform plan..."
terraform apply tfplan

# Clean up plan file
rm -f tfplan

echo ""
echo "=========================================="
echo "Apply complete!"
echo "=========================================="
echo ""
echo "To view outputs:"
echo "  terraform output"
echo ""
echo "To configure kubectl:"
echo "  aws eks update-kubeconfig --name <cluster-name> --region us-west-2"
