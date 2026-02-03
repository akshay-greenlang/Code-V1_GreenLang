# GreenLang Platform - INFRA-001 Final Deployment Checklist

**Task ID:** INFRA-001
**Version:** 1.0.0
**Last Updated:** 2026-02-03
**Status:** Production Ready

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Steps](#deployment-steps)
3. [Post-Deployment Verification](#post-deployment-verification)
4. [Rollback Procedures](#rollback-procedures)
5. [Emergency Contacts](#emergency-contacts)

---

## Pre-Deployment Checklist

### 1. Environment Prerequisites

#### 1.1 Local Tools
- [ ] **AWS CLI v2.x** - Configured with appropriate credentials
  ```bash
  aws --version  # Should be 2.x
  aws sts get-caller-identity  # Verify credentials
  ```

- [ ] **Terraform >= 1.6.0** - Infrastructure as Code
  ```bash
  terraform version  # Should be >= 1.6.0
  ```

- [ ] **kubectl >= 1.28** - Kubernetes CLI
  ```bash
  kubectl version --client
  ```

- [ ] **Helm >= 3.13** - Kubernetes package manager
  ```bash
  helm version  # Should be v3.13+
  ```

- [ ] **Docker Engine >= 24.0** - Container runtime
  ```bash
  docker --version
  docker compose version
  ```

- [ ] **jq** - JSON processor
  ```bash
  jq --version
  ```

- [ ] **Git** - Version control
  ```bash
  git --version
  ```

#### 1.2 AWS Account Verification
- [ ] Correct AWS account selected
- [ ] IAM permissions verified for:
  - [ ] VPC management
  - [ ] EKS cluster operations
  - [ ] RDS administration
  - [ ] ElastiCache management
  - [ ] S3 bucket operations
  - [ ] IAM role/policy management
  - [ ] Secrets Manager access
  - [ ] CloudWatch Logs

#### 1.3 Terraform State Backend
- [ ] S3 bucket `greenlang-terraform-state` exists
- [ ] DynamoDB table `greenlang-terraform-locks` exists
- [ ] Encryption enabled on state bucket
- [ ] Versioning enabled on state bucket

#### 1.4 Secrets Configuration
- [ ] AWS Secrets Manager secrets created:
  - [ ] `greenlang/{env}/database-credentials`
  - [ ] `greenlang/{env}/redis-auth-token`
  - [ ] `greenlang/{env}/app-secrets`
  - [ ] `greenlang/{env}/github-oidc-token` (if using GitHub Actions)

#### 1.5 Network Requirements
- [ ] VPC CIDR range confirmed (no conflicts)
- [ ] Availability zones available (minimum 3)
- [ ] NAT Gateway Elastic IPs available
- [ ] Route53 hosted zone ready (if using custom domain)

#### 1.6 Container Images
- [ ] Application images built and pushed to ECR/GHCR
- [ ] Image tags verified and documented
- [ ] Security scanning completed on images

---

### 2. Configuration Files

#### 2.1 Terraform Variables
- [ ] `terraform.tfvars` populated for target environment
- [ ] Sensitive values removed from version control
- [ ] Variable validation passed (`terraform validate`)

#### 2.2 Helm Values
- [ ] `values-{env}.yaml` configured for target environment
- [ ] Resource limits appropriate for workload
- [ ] Replica counts set correctly
- [ ] Ingress hosts configured

#### 2.3 Environment Variables
- [ ] `.env` file created from `.env.example`
- [ ] All required API keys set:
  - [ ] `ANTHROPIC_API_KEY`
  - [ ] `OPENAI_API_KEY`
  - [ ] `PINECONE_API_KEY` (if applicable)
- [ ] Database passwords are strong (min 16 chars)
- [ ] JWT secrets generated (`openssl rand -hex 32`)

---

### 3. Approval Gates

#### 3.1 Development Environment
- [ ] No approval required
- [ ] Auto-approve enabled

#### 3.2 Staging Environment
- [ ] Technical lead approval
- [ ] QA sign-off on test results

#### 3.3 Production Environment
- [ ] **Minimum 2 approvers required**
- [ ] Platform Engineering team approval
- [ ] Security team approval
- [ ] Change management ticket created
- [ ] Rollback plan documented and approved
- [ ] Maintenance window scheduled
- [ ] Stakeholders notified

---

## Deployment Steps

### Phase 1: Infrastructure Foundation (Stages 1-3)

#### Step 1.1: Initialize Terraform
```bash
make init ENV=prod
```
**Verification:**
- [ ] Backend initialized successfully
- [ ] Workspace selected/created
- [ ] `terraform validate` passes

#### Step 1.2: Review Terraform Plan
```bash
make plan ENV=prod
```
**Verification:**
- [ ] Plan generated without errors
- [ ] Review resource changes
- [ ] No unexpected destructive changes
- [ ] Cost estimate reviewed

#### Step 1.3: Apply Core Infrastructure (VPC, IAM, S3)
```bash
# These run in parallel
terraform apply -target=module.vpc -auto-approve
terraform apply -target=module.iam -auto-approve
terraform apply -target=module.s3 -auto-approve
```
**Verification:**
- [ ] VPC created with correct CIDR
- [ ] Subnets created across AZs
- [ ] NAT Gateways active
- [ ] IAM roles created
- [ ] S3 buckets created with encryption

---

### Phase 2: Compute and Data Layer (Stages 4-5)

#### Step 2.1: Deploy EKS Cluster
```bash
terraform apply -target=module.eks -auto-approve
```
**Verification:**
- [ ] EKS cluster status: ACTIVE
- [ ] Node groups healthy
- [ ] CoreDNS running
- [ ] kubectl connectivity confirmed

#### Step 2.2: Deploy RDS PostgreSQL
```bash
terraform apply -target=module.rds -auto-approve
```
**Verification:**
- [ ] RDS instance status: available
- [ ] Multi-AZ enabled (prod only)
- [ ] Encryption at rest enabled
- [ ] Connectivity from EKS verified

#### Step 2.3: Deploy ElastiCache Redis
```bash
terraform apply -target=module.elasticache -auto-approve
```
**Verification:**
- [ ] Redis replication group status: available
- [ ] Encryption in transit enabled
- [ ] Encryption at rest enabled
- [ ] Connectivity from EKS verified

---

### Phase 3: Kubernetes Configuration (Stages 6-7)

#### Step 3.1: Configure kubectl
```bash
aws eks update-kubeconfig --name greenlang-prod-eks --region us-east-1
kubectl cluster-info
kubectl get nodes
```
**Verification:**
- [ ] kubectl connected to correct cluster
- [ ] All nodes Ready
- [ ] System pods running

#### Step 3.2: Create Namespaces
```bash
kubectl apply -f deployment/infrastructure/kubernetes/greenlang/base/namespace.yaml
```
**Verification:**
- [ ] `greenlang` namespace created
- [ ] `greenlang-agents` namespace created
- [ ] `monitoring` namespace created
- [ ] `cert-manager` namespace created

#### Step 3.3: Deploy Kubernetes Add-ons (Parallel)
```bash
# Cert Manager
helm upgrade --install cert-manager jetstack/cert-manager \
  --namespace cert-manager --set installCRDs=true --wait

# Ingress Controller
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx --create-namespace --wait

# External Secrets Operator
helm upgrade --install external-secrets external-secrets/external-secrets \
  --namespace external-secrets --create-namespace --wait

# Cluster Autoscaler
helm upgrade --install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system --wait
```
**Verification:**
- [ ] Cert-manager webhook ready
- [ ] Ingress controller running with LoadBalancer
- [ ] External Secrets Operator ready
- [ ] Cluster Autoscaler running

---

### Phase 4: Monitoring Stack (Stage 8)

#### Step 4.1: Deploy Prometheus Stack
```bash
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --wait
```
**Verification:**
- [ ] Prometheus running
- [ ] Alertmanager running
- [ ] Grafana accessible
- [ ] ServiceMonitors created

#### Step 4.2: Deploy Custom Alerts and Dashboards
```bash
kubectl apply -f deployment/monitoring/alerts-unified.yml -n monitoring
kubectl apply -f deployment/kubernetes/prometheus-alerts.yaml -n monitoring
```
**Verification:**
- [ ] Alert rules loaded
- [ ] Dashboards imported
- [ ] No critical alerts firing

---

### Phase 5: Application Deployment (Stage 9)

#### Step 5.1: Create External Secrets
```bash
kubectl apply -f deployment/infrastructure/kubernetes/greenlang/secrets/external-secret.yaml
kubectl wait --for=condition=Ready externalsecret/greenlang-secrets -n greenlang --timeout=120s
```
**Verification:**
- [ ] ExternalSecret synced
- [ ] Kubernetes Secret created
- [ ] Secret values populated

#### Step 5.2: Deploy GreenLang Application
```bash
make deploy-k8s ENV=prod
# OR
helm upgrade --install greenlang deployment/infrastructure/helm/greenlang \
  --namespace greenlang \
  -f deployment/infrastructure/helm/greenlang/values-prod.yaml \
  --wait
```
**Verification:**
- [ ] Executor pods running (min 3)
- [ ] Worker pods running (min 5)
- [ ] Services created
- [ ] Ingress configured

#### Step 5.3: Run Database Migrations
```bash
kubectl run greenlang-migrations \
  --image=greenlang/app:latest \
  --namespace greenlang \
  --restart=Never --rm --attach \
  --command -- python manage.py migrate --noinput
```
**Verification:**
- [ ] Migrations completed successfully
- [ ] Database schema up to date

---

### Phase 6: Validation (Stage 10)

#### Step 6.1: Health Checks
```bash
make validate ENV=prod
```
**Verification:**
- [ ] All health endpoints returning 200
- [ ] Database connectivity confirmed
- [ ] Redis connectivity confirmed
- [ ] Message queue connectivity confirmed

#### Step 6.2: Smoke Tests
```bash
# API availability
curl -sf https://api.greenlang.io/api/v1/health

# Database connectivity
curl -sf https://api.greenlang.io/api/v1/health/db

# Redis connectivity
curl -sf https://api.greenlang.io/api/v1/health/redis
```
**Verification:**
- [ ] API responding
- [ ] All dependencies healthy
- [ ] Latency within SLO (<500ms)

---

## Post-Deployment Verification

### 1. Infrastructure Verification

| Component | Check | Expected | Actual | Status |
|-----------|-------|----------|--------|--------|
| VPC | Status | available | | [ ] |
| EKS | Cluster Status | ACTIVE | | [ ] |
| RDS | Instance Status | available | | [ ] |
| ElastiCache | Cluster Status | available | | [ ] |
| S3 | Buckets Accessible | Yes | | [ ] |

### 2. Application Verification

| Component | Check | Expected | Actual | Status |
|-----------|-------|----------|--------|--------|
| Executor | Ready Replicas | >= 3 | | [ ] |
| Worker | Ready Replicas | >= 5 | | [ ] |
| API Health | HTTP Status | 200 | | [ ] |
| DB Health | Connected | Yes | | [ ] |
| Redis Health | Connected | Yes | | [ ] |

### 3. Monitoring Verification

| Component | Check | Expected | Actual | Status |
|-----------|-------|----------|--------|--------|
| Prometheus | Targets Up | >= 10 | | [ ] |
| Alertmanager | Running | Yes | | [ ] |
| Grafana | Accessible | Yes | | [ ] |
| Critical Alerts | Firing | 0 | | [ ] |

### 4. Security Verification

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| TLS Certificates Valid | Yes | | [ ] |
| Network Policies Applied | Yes | | [ ] |
| Pod Security Standards | Enforced | | [ ] |
| Secrets Encrypted | Yes | | [ ] |
| RDS Encryption | Enabled | | [ ] |
| Redis Encryption | Enabled | | [ ] |

### 5. Performance Baseline

| Metric | SLO | Actual | Status |
|--------|-----|--------|--------|
| API Latency (p50) | < 100ms | | [ ] |
| API Latency (p99) | < 500ms | | [ ] |
| Error Rate | < 0.1% | | [ ] |
| Availability | > 99.9% | | [ ] |

---

## Rollback Procedures

### Rollback Decision Tree

```
Deployment Issue Detected
         |
         v
    Is it Critical?
      /        \
   Yes          No
    |            |
    v            v
Emergency     Standard
Rollback      Rollback
```

### 1. Application Rollback (Fastest - ~5 minutes)

**When to use:** Application bugs, configuration issues, performance degradation

```bash
# Rollback Helm release to previous version
helm rollback greenlang -n greenlang

# Verify rollback
kubectl get pods -n greenlang
kubectl rollout status deployment/greenlang-executor -n greenlang
```

### 2. Kubernetes Add-ons Rollback (~10 minutes)

**When to use:** Add-on issues (ingress, cert-manager, external-secrets)

```bash
# Rollback specific add-on
helm rollback kube-prometheus-stack -n monitoring
helm rollback ingress-nginx -n ingress-nginx
helm rollback external-secrets -n external-secrets
helm rollback cert-manager -n cert-manager
```

### 3. Infrastructure Rollback (~30-60 minutes)

**When to use:** Infrastructure issues, network problems, data layer issues

**WARNING: This is destructive. Requires approval.**

```bash
# Create RDS snapshot before rollback
aws rds create-db-snapshot \
  --db-instance-identifier greenlang-prod-postgres \
  --db-snapshot-identifier greenlang-prod-pre-rollback-$(date +%Y%m%d%H%M%S)

# Rollback Terraform
cd deployment/terraform/environments/prod
terraform plan -destroy -target=module.eks
# Review plan carefully
terraform destroy -target=module.eks -auto-approve
```

### 4. Full Environment Rollback (Last Resort)

**When to use:** Complete environment failure, security breach

```bash
# 1. Create all backups
./scripts/backup-all.sh

# 2. Destroy environment
make destroy ENV=prod

# 3. Restore from backup
./scripts/restore-environment.sh --backup-id <backup-id>
```

### Rollback Checklist

- [ ] Identify the issue and scope
- [ ] Notify stakeholders
- [ ] Create backup/snapshot (if applicable)
- [ ] Execute rollback command
- [ ] Verify rollback successful
- [ ] Run health checks
- [ ] Document incident
- [ ] Schedule post-mortem

---

## Emergency Contacts

### On-Call Rotation
- **Platform Engineering:** #greenlang-platform (Slack)
- **PagerDuty:** https://greenlang.pagerduty.com/schedules/platform

### Escalation Path

| Level | Contact | Response Time |
|-------|---------|---------------|
| L1 | On-call Engineer | 15 minutes |
| L2 | Platform Engineering Lead | 30 minutes |
| L3 | VP of Engineering | 1 hour |
| L4 | CTO | 2 hours |

### External Support
- **AWS Support:** https://console.aws.amazon.com/support
- **Kubernetes Community:** #kubernetes (Slack)

---

## Deployment Sign-Off

### Development
- **Deployed By:** _______________
- **Date:** _______________
- **Version:** _______________

### Staging
- **Deployed By:** _______________
- **Date:** _______________
- **QA Sign-Off:** _______________
- **Version:** _______________

### Production
- **Deployed By:** _______________
- **Date:** _______________
- **Approved By (1):** _______________
- **Approved By (2):** _______________
- **Change Ticket:** _______________
- **Version:** _______________

---

**End of INFRA-001 Final Deployment Checklist**
