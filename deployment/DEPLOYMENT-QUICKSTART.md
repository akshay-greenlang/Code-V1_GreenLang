# INFRA-001: Kubernetes Production Deployment Quick Start

**Status:** IaC 100% Complete | AWS Deployment 0% Complete
**Estimated Cost:** ~$4,234/month
**Deployment Time:** 2-4 hours

---

## Prerequisites Checklist

- [ ] AWS CLI installed (`aws --version`)
- [ ] Terraform >= 1.6 installed (`terraform --version`)
- [ ] kubectl installed (`kubectl version --client`)
- [ ] Helm >= 3.0 installed (`helm version`)
- [ ] AWS credentials with admin permissions

---

## Step 1: Configure AWS Credentials

```bash
# Configure AWS CLI with your credentials
aws configure

# Verify credentials work
aws sts get-caller-identity

# Note your AWS Account ID (12-digit number)
# Example output: { "Account": "123456789012", ... }
```

---

## Step 2: Update terraform.tfvars

Replace `ACCOUNT_ID` placeholder in `deployment/terraform/environments/prod/terraform.tfvars`:

```bash
# Get your AWS Account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Your AWS Account ID: $AWS_ACCOUNT_ID"

# Navigate to prod environment
cd deployment/terraform/environments/prod

# Replace all ACCOUNT_ID placeholders (Linux/macOS)
sed -i "s/ACCOUNT_ID/$AWS_ACCOUNT_ID/g" terraform.tfvars

# Windows PowerShell alternative:
# (Get-Content terraform.tfvars) -replace 'ACCOUNT_ID', $AWS_ACCOUNT_ID | Set-Content terraform.tfvars
```

---

## Step 3: Initialize Terraform Backend

```bash
# From repository root
cd deployment/terraform/scripts

# Linux/macOS/WSL
chmod +x infra-init.sh
./infra-init.sh --environment prod

# Windows PowerShell
.\infra-init.ps1 -Environment prod
```

This creates:
- S3 bucket: `greenlang-prod-terraform-state`
- DynamoDB table: `greenlang-prod-terraform-locks`

---

## Step 4: Review Terraform Plan

```bash
cd deployment/terraform/environments/prod

# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Review what will be created
terraform plan -out=tfplan

# IMPORTANT: Review the plan carefully before applying!
```

---

## Step 5: Apply Infrastructure (Incremental)

Deploy in order to manage dependencies:

```bash
# 1. VPC (5 min) - Creates networking foundation
terraform apply -target=module.vpc -auto-approve

# 2. S3 Buckets (2 min)
terraform apply -target=module.s3 -auto-approve

# 3. IAM Roles (2 min)
terraform apply -target=module.iam -auto-approve

# 4. EKS Cluster (15 min) - Creates Kubernetes cluster
terraform apply -target=module.eks -auto-approve

# 5. RDS PostgreSQL (20 min) - Database
terraform apply -target=module.rds -auto-approve

# 6. ElastiCache Redis (10 min) - Cache
terraform apply -target=module.elasticache -auto-approve

# 7. Remaining resources
terraform apply -auto-approve
```

---

## Step 6: Configure kubectl

```bash
# Update kubeconfig for EKS access
aws eks update-kubeconfig --name greenlang-prod-eks --region us-east-1

# Verify connection
kubectl cluster-info
kubectl get nodes
```

---

## Step 7: Deploy Kubernetes Add-ons

```bash
# Add Helm repositories
helm repo add eks https://aws.github.io/eks-charts
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm repo update

# AWS Load Balancer Controller
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=greenlang-prod-eks

# Cluster Autoscaler
helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  -n kube-system \
  --set autoDiscovery.clusterName=greenlang-prod-eks

# Metrics Server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

---

## Step 8: Deploy Monitoring Stack

```bash
cd deployment/helm/greenlang-agents
helm dependency update
helm install monitoring . -f values-prod.yaml -n monitoring --create-namespace
```

---

## Step 9: Deploy GreenLang Applications

```bash
# Using Kustomize
cd deployment/kustomize/overlays/prod
kubectl apply -k .

# Verify deployments
kubectl get pods -n greenlang
kubectl get svc -n greenlang
kubectl get ingress -n greenlang
```

---

## Step 10: Validate Production Readiness

```bash
# Check all pods are running
kubectl get pods -A | grep -v Running | grep -v Completed

# Check services have endpoints
kubectl get endpoints -n greenlang

# Check ingress has external IP
kubectl get ingress -n greenlang

# Test health endpoint (update with actual URL)
curl -s https://api.greenlang.io/health
```

---

## Cost Breakdown

| Service | Monthly Cost |
|---------|--------------|
| EKS Control Plane | $73 |
| EC2 Nodes (System) | $432 |
| EC2 Nodes (API) | $612 |
| EC2 Nodes (Agent) | $680 |
| RDS PostgreSQL | $1,200 |
| ElastiCache Redis | $900 |
| NAT Gateways (3x) | $135 |
| S3 Storage | $12 |
| Data Transfer | $90 |
| CloudWatch | $100 |
| **Total** | **~$4,234** |

---

## Rollback Procedure

If something goes wrong:

```bash
# Destroy all infrastructure (CAUTION: This deletes everything!)
cd deployment/terraform/environments/prod
terraform destroy

# Or target specific modules
terraform destroy -target=module.eks
```

---

## Ralphy Automation (Alternative)

Run the full deployment using Ralphy task automation:

```bash
# Install Ralphy
pip install ralphy

# Run INFRA-001 deployment
ralphy run deployment/terraform/INFRA-001-TASKS.yaml --environment prod
```

---

## Support

- **PRD Document:** `GreenLang Development/05-Documentation/PRD-INFRA-001-K8s-Deployment.md`
- **Terraform Modules:** `deployment/terraform/modules/`
- **Ralphy Config:** `.ralphy/config.yaml`
- **Task Config:** `deployment/terraform/INFRA-001-TASKS.yaml`
