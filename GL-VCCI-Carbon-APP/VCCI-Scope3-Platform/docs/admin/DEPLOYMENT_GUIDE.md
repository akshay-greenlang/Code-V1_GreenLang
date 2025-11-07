# GL-VCCI Scope 3 Platform v2.0 - Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Application Deployment](#application-deployment)
5. [Multi-Tenant Configuration](#multi-tenant-configuration)
6. [Post-Deployment Validation](#post-deployment-validation)
7. [Rollback Procedures](#rollback-procedures)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose
This guide provides comprehensive instructions for deploying the GL-VCCI Scope 3 Carbon Intelligence Platform v2.0 in production environments.

### Architecture Overview
- **Platform**: Kubernetes (EKS, GKE, or AKS)
- **Infrastructure as Code**: Terraform
- **Container Registry**: ECR, GCR, or ACR
- **Database**: PostgreSQL 14+ (managed service)
- **Cache**: Redis 7+ (managed service)
- **Message Queue**: RabbitMQ 3.11+
- **Storage**: S3-compatible object storage
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)

### Deployment Models
- **Production**: Multi-AZ, High Availability
- **Staging**: Single-AZ, Production-like
- **Development**: Minimal resources, single instance

### Estimated Deployment Time
- Infrastructure Setup: 2-3 hours
- Application Deployment: 1-2 hours
- Configuration & Validation: 1-2 hours
- **Total**: 4-7 hours

---

## Pre-Deployment Checklist

### 1. Prerequisites

#### Required Tools
```bash
# Verify tool installations
kubectl version --client
terraform version
helm version
aws-cli --version  # or gcloud, az
docker version
jq --version
```

**Minimum Versions**:
- kubectl: 1.25+
- Terraform: 1.5+
- Helm: 3.10+
- Docker: 20.10+

#### Access Requirements
- [ ] Cloud provider admin access (AWS, GCP, or Azure)
- [ ] Container registry push/pull access
- [ ] DNS management access
- [ ] Certificate authority access (for TLS)
- [ ] Secrets management access (AWS Secrets Manager, GCP Secret Manager, or Azure Key Vault)

#### Credential Setup
```bash
# AWS Example
export AWS_PROFILE=production
export AWS_REGION=us-east-1

# GCP Example
gcloud auth application-default login
export GCP_PROJECT=vcci-production
export GCP_REGION=us-central1

# Azure Example
az login
export AZURE_SUBSCRIPTION_ID=<subscription-id>
export AZURE_RESOURCE_GROUP=vcci-production
```

### 2. Infrastructure Planning

#### Sizing Estimates

**Small Deployment (< 10 tenants, < 1M records/month)**
```yaml
Application Servers: 3 nodes x m5.xlarge (4 vCPU, 16GB RAM)
Database: db.r5.xlarge (4 vCPU, 32GB RAM)
Redis: cache.r5.large (2 vCPU, 13GB RAM)
Storage: 500GB
Monthly Cost: ~$3,000
```

**Medium Deployment (10-50 tenants, 1-10M records/month)**
```yaml
Application Servers: 6 nodes x m5.2xlarge (8 vCPU, 32GB RAM)
Database: db.r5.2xlarge (8 vCPU, 64GB RAM) + 1 read replica
Redis: cache.r5.xlarge (4 vCPU, 26GB RAM) - cluster mode
Storage: 2TB
Monthly Cost: ~$8,000
```

**Large Deployment (50+ tenants, 10M+ records/month)**
```yaml
Application Servers: 12 nodes x m5.4xlarge (16 vCPU, 64GB RAM)
Database: db.r5.4xlarge (16 vCPU, 128GB RAM) + 2 read replicas
Redis: cache.r5.2xlarge (8 vCPU, 52GB RAM) - cluster mode
Storage: 10TB
Monthly Cost: ~$25,000
```

#### Network Planning
- [ ] VPC CIDR block allocated (e.g., 10.0.0.0/16)
- [ ] Subnet planning completed:
  - Public subnets for load balancers (3 AZs)
  - Private subnets for applications (3 AZs)
  - Private subnets for databases (3 AZs)
- [ ] NAT Gateway strategy defined
- [ ] VPN or Direct Connect configured (if required)
- [ ] DNS zones created
- [ ] Certificate requests submitted

### 3. Security Preparation

#### Certificates
```bash
# Generate CSR for TLS certificates
openssl req -new -newkey rsa:4096 -nodes \
  -keyout vcci-platform.key \
  -out vcci-platform.csr \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=*.vcci-platform.com"

# Store certificates in secrets manager
aws secretsmanager create-secret \
  --name vcci/production/tls-cert \
  --secret-string file://vcci-platform.crt

aws secretsmanager create-secret \
  --name vcci/production/tls-key \
  --secret-string file://vcci-platform.key
```

#### Encryption Keys
```bash
# Generate data encryption key
openssl rand -base64 32 > data-encryption-key.txt

# Store in secrets manager
aws secretsmanager create-secret \
  --name vcci/production/data-encryption-key \
  --secret-string file://data-encryption-key.txt

# Clean up local file
shred -u data-encryption-key.txt
```

#### Service Accounts
- [ ] Application service account created
- [ ] Database service account created
- [ ] Monitoring service account created
- [ ] Backup service account created
- [ ] Least privilege IAM policies attached

### 4. Configuration Files Preparation

#### Environment Configuration Template
```bash
# Create configuration from template
cat > config/production.env << 'EOF'
# Application
APP_ENV=production
APP_DEBUG=false
APP_LOG_LEVEL=info
APP_VERSION={{ VERSION }}

# API
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=300
API_MAX_REQUEST_SIZE=100MB

# Database
DATABASE_HOST={{ DB_ENDPOINT }}
DATABASE_PORT=5432
DATABASE_NAME=vcci_production
DATABASE_POOL_SIZE=20
DATABASE_POOL_MAX_OVERFLOW=10
DATABASE_SSL_MODE=require

# Redis
REDIS_HOST={{ REDIS_ENDPOINT }}
REDIS_PORT=6379
REDIS_SSL=true
REDIS_MAX_CONNECTIONS=50

# RabbitMQ
RABBITMQ_HOST={{ RABBITMQ_ENDPOINT }}
RABBITMQ_PORT=5672
RABBITMQ_VHOST=/vcci
RABBITMQ_SSL=true

# Storage
S3_BUCKET=vcci-production-data
S3_REGION=us-east-1
S3_ENCRYPTION=AES256

# Multi-Tenancy
TENANT_ISOLATION_LEVEL=strict
TENANT_DATA_ENCRYPTION=true
TENANT_RATE_LIMIT_ENABLED=true

# Security
JWT_ALGORITHM=RS256
SESSION_TIMEOUT=3600
PASSWORD_MIN_LENGTH=12
MFA_ENABLED=true

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
METRICS_INTERVAL=30s

# Logging
LOG_FORMAT=json
LOG_OUTPUT=stdout
LOG_AUDIT_ENABLED=true
LOG_RETENTION_DAYS=90

# Feature Flags
FEATURE_ML_PREDICTIONS=true
FEATURE_ADVANCED_ANALYTICS=true
FEATURE_SUPPLIER_PORTAL=true
FEATURE_API_V2=true
EOF
```

---

## Infrastructure Setup

### 1. Terraform Infrastructure Deployment

#### Directory Structure
```
terraform/
├── environments/
│   ├── production/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── terraform.tfvars
│   │   └── outputs.tf
│   ├── staging/
│   └── development/
├── modules/
│   ├── networking/
│   ├── kubernetes/
│   ├── database/
│   ├── redis/
│   ├── storage/
│   └── monitoring/
└── scripts/
    ├── deploy.sh
    └── destroy.sh
```

#### Main Terraform Configuration

**terraform/environments/production/main.tf**
```hcl
terraform {
  required_version = ">= 1.5.0"

  backend "s3" {
    bucket         = "vcci-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "vcci-terraform-locks"
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = "production"
      Project     = "vcci-scope3-platform"
      ManagedBy   = "terraform"
      Owner       = "platform-team"
    }
  }
}

# Networking Module
module "networking" {
  source = "../../modules/networking"

  environment         = var.environment
  vpc_cidr            = var.vpc_cidr
  availability_zones  = var.availability_zones
  enable_nat_gateway  = true
  enable_vpn_gateway  = var.enable_vpn

  tags = var.common_tags
}

# EKS Cluster Module
module "kubernetes" {
  source = "../../modules/kubernetes"

  cluster_name       = "vcci-production"
  cluster_version    = "1.28"
  vpc_id             = module.networking.vpc_id
  subnet_ids         = module.networking.private_subnet_ids

  node_groups = {
    application = {
      desired_size   = 6
      min_size       = 3
      max_size       = 12
      instance_types = ["m5.2xlarge"]
      capacity_type  = "ON_DEMAND"
      disk_size      = 100

      labels = {
        role = "application"
      }

      taints = []
    }

    ml_workloads = {
      desired_size   = 2
      min_size       = 1
      max_size       = 4
      instance_types = ["c5.4xlarge"]
      capacity_type  = "SPOT"
      disk_size      = 200

      labels = {
        role = "ml-compute"
      }

      taints = [{
        key    = "workload"
        value  = "ml"
        effect = "NoSchedule"
      }]
    }
  }

  enable_irsa            = true
  enable_cluster_autoscaler = true
  enable_metrics_server  = true

  tags = var.common_tags
}

# RDS PostgreSQL Module
module "database" {
  source = "../../modules/database"

  identifier              = "vcci-production"
  engine_version          = "14.9"
  instance_class          = "db.r5.2xlarge"
  allocated_storage       = 500
  max_allocated_storage   = 2000
  storage_encrypted       = true
  kms_key_id             = aws_kms_key.database.arn

  vpc_id                  = module.networking.vpc_id
  subnet_ids              = module.networking.database_subnet_ids

  database_name           = "vcci_production"
  master_username         = "vcci_admin"

  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  create_read_replica    = true
  replica_count          = 1

  performance_insights_enabled = true
  monitoring_interval    = 60

  parameter_group_family = "postgres14"
  parameters = [
    {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements,auto_explain"
    },
    {
      name  = "log_min_duration_statement"
      value = "1000"
    },
    {
      name  = "max_connections"
      value = "500"
    }
  ]

  tags = var.common_tags
}

# ElastiCache Redis Module
module "redis" {
  source = "../../modules/redis"

  cluster_id           = "vcci-production"
  engine_version       = "7.0"
  node_type            = "cache.r5.xlarge"
  num_cache_nodes      = 3

  vpc_id               = module.networking.vpc_id
  subnet_ids           = module.networking.cache_subnet_ids

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token_enabled         = true

  automatic_failover_enabled = true
  multi_az_enabled          = true

  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"

  parameter_group_family = "redis7"
  parameters = [
    {
      name  = "maxmemory-policy"
      value = "allkeys-lru"
    },
    {
      name  = "timeout"
      value = "300"
    }
  ]

  tags = var.common_tags
}

# S3 Storage Module
module "storage" {
  source = "../../modules/storage"

  bucket_name = "vcci-production-data"

  versioning_enabled = true

  lifecycle_rules = [
    {
      id      = "archive-old-data"
      enabled = true

      transition = [{
        days          = 90
        storage_class = "STANDARD_IA"
      }]

      noncurrent_version_transition = [{
        days          = 30
        storage_class = "GLACIER"
      }]
    }
  ]

  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        sse_algorithm     = "aws:kms"
        kms_master_key_id = aws_kms_key.storage.arn
      }
    }
  }

  cors_rules = [{
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST"]
    allowed_origins = ["https://*.vcci-platform.com"]
    max_age_seconds = 3600
  }]

  tags = var.common_tags
}

# Monitoring Module
module "monitoring" {
  source = "../../modules/monitoring"

  cluster_name = module.kubernetes.cluster_name

  prometheus_storage_size = "100Gi"
  grafana_storage_size    = "20Gi"

  alertmanager_config = {
    slack_webhook_url = var.slack_webhook_url
    pagerduty_key     = var.pagerduty_key
  }

  log_retention_days = 90

  tags = var.common_tags
}

# KMS Keys
resource "aws_kms_key" "database" {
  description             = "VCCI Production Database Encryption Key"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = merge(var.common_tags, {
    Name = "vcci-production-database-key"
  })
}

resource "aws_kms_key" "storage" {
  description             = "VCCI Production Storage Encryption Key"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = merge(var.common_tags, {
    Name = "vcci-production-storage-key"
  })
}

# Outputs
output "vpc_id" {
  value = module.networking.vpc_id
}

output "eks_cluster_endpoint" {
  value = module.kubernetes.cluster_endpoint
}

output "database_endpoint" {
  value = module.database.endpoint
}

output "redis_endpoint" {
  value = module.redis.endpoint
}

output "s3_bucket_name" {
  value = module.storage.bucket_name
}
```

#### Variables Configuration

**terraform/environments/production/variables.tf**
```hcl
variable "aws_region" {
  description = "AWS region for infrastructure"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "enable_vpn" {
  description = "Enable VPN gateway"
  type        = bool
  default     = false
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for alerts"
  type        = string
  sensitive   = true
}

variable "pagerduty_key" {
  description = "PagerDuty integration key"
  type        = string
  sensitive   = true
}

variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Environment = "production"
    Project     = "vcci-scope3-platform"
    ManagedBy   = "terraform"
  }
}
```

#### Deployment Script

**terraform/scripts/deploy.sh**
```bash
#!/bin/bash
set -euo pipefail

ENVIRONMENT=${1:-production}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/../environments/${ENVIRONMENT}"

echo "=========================================="
echo "VCCI Platform Infrastructure Deployment"
echo "Environment: ${ENVIRONMENT}"
echo "=========================================="

# Validate prerequisites
echo "Validating prerequisites..."
command -v terraform >/dev/null 2>&1 || { echo "terraform not found"; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "aws-cli not found"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "kubectl not found"; exit 1; }

# Check AWS credentials
aws sts get-caller-identity >/dev/null || { echo "AWS credentials not configured"; exit 1; }

cd "${ENV_DIR}"

# Initialize Terraform
echo "Initializing Terraform..."
terraform init -upgrade

# Validate configuration
echo "Validating Terraform configuration..."
terraform validate

# Plan deployment
echo "Creating deployment plan..."
terraform plan -out=tfplan

# Review plan
echo ""
echo "Review the plan above. Continue with deployment? (yes/no)"
read -r CONFIRM

if [ "${CONFIRM}" != "yes" ]; then
    echo "Deployment cancelled"
    exit 0
fi

# Apply deployment
echo "Applying infrastructure changes..."
terraform apply tfplan

# Save outputs
echo "Saving infrastructure outputs..."
terraform output -json > outputs.json

echo ""
echo "=========================================="
echo "Infrastructure Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Configure kubectl: ./configure-kubectl.sh"
echo "2. Deploy application: ./deploy-application.sh"
echo ""
```

### 2. Kubernetes Configuration

#### Update kubeconfig
```bash
# AWS EKS
aws eks update-kubeconfig \
  --region us-east-1 \
  --name vcci-production \
  --alias vcci-production

# Verify connection
kubectl cluster-info
kubectl get nodes
```

#### Install Core Components

**Install NGINX Ingress Controller**
```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.replicaCount=3 \
  --set controller.service.type=LoadBalancer \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-type"="nlb" \
  --set controller.metrics.enabled=true \
  --set controller.podAnnotations."prometheus\.io/scrape"="true" \
  --set controller.podAnnotations."prometheus\.io/port"="10254"

# Get load balancer endpoint
kubectl get svc -n ingress-nginx ingress-nginx-controller
```

**Install Cert-Manager**
```bash
helm repo add jetstack https://charts.jetstack.io
helm repo update

helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.13.0 \
  --set installCRDs=true

# Create ClusterIssuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-production
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@vcci-platform.com
    privateKeySecretRef:
      name: letsencrypt-production
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

**Install External Secrets Operator**
```bash
helm repo add external-secrets https://charts.external-secrets.io
helm repo update

helm install external-secrets external-secrets/external-secrets \
  --namespace external-secrets \
  --create-namespace \
  --set installCRDs=true

# Create SecretStore for AWS Secrets Manager
cat <<EOF | kubectl apply -f -
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets
            namespace: external-secrets
EOF
```

---

## Application Deployment

### 1. Create Namespace and RBAC

**Create Application Namespace**
```bash
kubectl create namespace vcci-production

# Label namespace for network policies
kubectl label namespace vcci-production \
  name=vcci-production \
  environment=production \
  tenant-isolation=enabled
```

**Create Service Account**
```yaml
# k8s/rbac/serviceaccount.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: vcci-app
  namespace: vcci-production
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT_ID:role/vcci-production-app
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: vcci-app-role
  namespace: vcci-production
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: vcci-app-rolebinding
  namespace: vcci-production
subjects:
- kind: ServiceAccount
  name: vcci-app
  namespace: vcci-production
roleRef:
  kind: Role
  name: vcci-app-role
  apiGroup: rbac.authorization.k8s.io
```

```bash
kubectl apply -f k8s/rbac/serviceaccount.yaml
```

### 2. Create Secrets

**Database Secrets**
```yaml
# k8s/secrets/external-secret-database.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
  namespace: vcci-production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: database-credentials
    creationPolicy: Owner
  data:
  - secretKey: host
    remoteRef:
      key: vcci/production/database
      property: host
  - secretKey: port
    remoteRef:
      key: vcci/production/database
      property: port
  - secretKey: database
    remoteRef:
      key: vcci/production/database
      property: database
  - secretKey: username
    remoteRef:
      key: vcci/production/database
      property: username
  - secretKey: password
    remoteRef:
      key: vcci/production/database
      property: password
```

**Application Secrets**
```yaml
# k8s/secrets/external-secret-app.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: app-secrets
  namespace: vcci-production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: app-secrets
    creationPolicy: Owner
  data:
  - secretKey: jwt-private-key
    remoteRef:
      key: vcci/production/app
      property: jwt-private-key
  - secretKey: jwt-public-key
    remoteRef:
      key: vcci/production/app
      property: jwt-public-key
  - secretKey: data-encryption-key
    remoteRef:
      key: vcci/production/app
      property: data-encryption-key
  - secretKey: api-key-salt
    remoteRef:
      key: vcci/production/app
      property: api-key-salt
```

```bash
kubectl apply -f k8s/secrets/
```

### 3. Create ConfigMaps

**Application Configuration**
```yaml
# k8s/configmaps/app-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: vcci-production
data:
  APP_ENV: "production"
  APP_DEBUG: "false"
  APP_LOG_LEVEL: "info"
  API_PORT: "8000"
  API_WORKERS: "4"
  API_TIMEOUT: "300"
  DATABASE_POOL_SIZE: "20"
  DATABASE_SSL_MODE: "require"
  REDIS_MAX_CONNECTIONS: "50"
  TENANT_ISOLATION_LEVEL: "strict"
  PROMETHEUS_ENABLED: "true"
  LOG_FORMAT: "json"
  FEATURE_ML_PREDICTIONS: "true"
```

```bash
kubectl apply -f k8s/configmaps/app-config.yaml
```

### 4. Deploy Application

**API Deployment**
```yaml
# k8s/deployments/api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vcci-api
  namespace: vcci-production
  labels:
    app: vcci-api
    component: api
    version: v2.0.0
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  selector:
    matchLabels:
      app: vcci-api
  template:
    metadata:
      labels:
        app: vcci-api
        component: api
        version: v2.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: vcci-app
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000

      initContainers:
      - name: wait-for-database
        image: postgres:14-alpine
        command: ['sh', '-c']
        args:
        - |
          until pg_isready -h $(DATABASE_HOST) -p $(DATABASE_PORT) -U $(DATABASE_USER); do
            echo "Waiting for database..."
            sleep 2
          done
        env:
        - name: DATABASE_HOST
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: host
        - name: DATABASE_PORT
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: port
        - name: DATABASE_USER
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: username

      - name: run-migrations
        image: your-registry.com/vcci-platform:v2.0.0
        command: ['python', 'manage.py', 'migrate']
        envFrom:
        - configMapRef:
            name: app-config
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: url

      containers:
      - name: api
        image: your-registry.com/vcci-platform:v2.0.0
        imagePullPolicy: Always

        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP

        envFrom:
        - configMapRef:
            name: app-config

        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        - name: JWT_PRIVATE_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: jwt-private-key
        - name: DATA_ENCRYPTION_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: data-encryption-key

        resources:
          requests:
            cpu: 2000m
            memory: 4Gi
          limits:
            cpu: 4000m
            memory: 8Gi

        livenessProbe:
          httpGet:
            path: /health/live
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /health/ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]

        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
          readOnlyRootFilesystem: true

        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache

      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir: {}

      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - vcci-api
              topologyKey: kubernetes.io/hostname
---
apiVersion: v1
kind: Service
metadata:
  name: vcci-api
  namespace: vcci-production
  labels:
    app: vcci-api
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 80
    targetPort: http
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: metrics
    protocol: TCP
  selector:
    app: vcci-api
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vcci-api-hpa
  namespace: vcci-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vcci-api
  minReplicas: 6
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

**Worker Deployment**
```yaml
# k8s/deployments/worker.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vcci-worker
  namespace: vcci-production
  labels:
    app: vcci-worker
    component: worker
spec:
  replicas: 4
  selector:
    matchLabels:
      app: vcci-worker
  template:
    metadata:
      labels:
        app: vcci-worker
        component: worker
    spec:
      serviceAccountName: vcci-app

      containers:
      - name: worker
        image: your-registry.com/vcci-platform:v2.0.0
        command: ['celery', '-A', 'vcci.celery', 'worker']
        args:
        - '--loglevel=info'
        - '--concurrency=4'
        - '--max-tasks-per-child=1000'

        envFrom:
        - configMapRef:
            name: app-config

        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        - name: RABBITMQ_URL
          valueFrom:
            secretKeyRef:
              name: rabbitmq-credentials
              key: url

        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
```

**Deploy All Components**
```bash
# Apply deployments
kubectl apply -f k8s/deployments/

# Verify deployment
kubectl get deployments -n vcci-production
kubectl get pods -n vcci-production
kubectl get svc -n vcci-production

# Check pod logs
kubectl logs -n vcci-production -l app=vcci-api --tail=100
```

### 5. Configure Ingress

**Ingress Resource**
```yaml
# k8s/ingress/api-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vcci-api-ingress
  namespace: vcci-production
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-production"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/limit-rps: "10"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.vcci-platform.com
    secretName: vcci-api-tls
  rules:
  - host: api.vcci-platform.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vcci-api
            port:
              number: 80
```

```bash
kubectl apply -f k8s/ingress/api-ingress.yaml

# Get ingress details
kubectl get ingress -n vcci-production
kubectl describe ingress vcci-api-ingress -n vcci-production
```

---

## Multi-Tenant Configuration

### 1. Tenant Database Schema Setup

**Create Schema Migration**
```sql
-- migrations/tenant_setup.sql
-- Create tenant-specific schemas
CREATE OR REPLACE FUNCTION create_tenant_schema(tenant_id TEXT)
RETURNS VOID AS $$
BEGIN
  EXECUTE format('CREATE SCHEMA IF NOT EXISTS tenant_%s', tenant_id);
  EXECUTE format('GRANT USAGE ON SCHEMA tenant_%s TO vcci_app', tenant_id);

  -- Create tenant tables
  EXECUTE format('
    CREATE TABLE IF NOT EXISTS tenant_%s.emissions (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      category VARCHAR(50) NOT NULL,
      scope INTEGER NOT NULL,
      amount DECIMAL(15,4) NOT NULL,
      unit VARCHAR(20) NOT NULL,
      source_id VARCHAR(100),
      calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
      metadata JSONB,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )', tenant_id);

  -- Create indexes
  EXECUTE format('
    CREATE INDEX IF NOT EXISTS idx_emissions_category
    ON tenant_%s.emissions(category)', tenant_id);
  EXECUTE format('
    CREATE INDEX IF NOT EXISTS idx_emissions_scope
    ON tenant_%s.emissions(scope)', tenant_id);
  EXECUTE format('
    CREATE INDEX IF NOT EXISTS idx_emissions_calculated_at
    ON tenant_%s.emissions(calculated_at)', tenant_id);

  -- Enable Row Level Security
  EXECUTE format('ALTER TABLE tenant_%s.emissions ENABLE ROW LEVEL SECURITY', tenant_id);

  -- Create RLS policy
  EXECUTE format('
    CREATE POLICY tenant_isolation_policy ON tenant_%s.emissions
    USING (current_setting(''app.current_tenant'') = ''%s'')', tenant_id, tenant_id);
END;
$$ LANGUAGE plpgsql;

-- Create tenant management table
CREATE TABLE IF NOT EXISTS public.tenants (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id VARCHAR(100) UNIQUE NOT NULL,
  name VARCHAR(255) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'active',
  tier VARCHAR(20) NOT NULL DEFAULT 'standard',
  quota_records INTEGER DEFAULT 1000000,
  quota_users INTEGER DEFAULT 100,
  quota_api_calls INTEGER DEFAULT 10000000,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  metadata JSONB
);

-- Create tenant API keys table
CREATE TABLE IF NOT EXISTS public.tenant_api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id VARCHAR(100) REFERENCES public.tenants(tenant_id),
  key_hash VARCHAR(255) NOT NULL,
  name VARCHAR(255) NOT NULL,
  permissions JSONB NOT NULL,
  expires_at TIMESTAMP WITH TIME ZONE,
  last_used_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  revoked_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_tenant_api_keys_hash ON public.tenant_api_keys(key_hash);
```

**Apply Migration**
```bash
# Run migration
kubectl exec -n vcci-production -it deployment/vcci-api -- \
  psql $DATABASE_URL -f /app/migrations/tenant_setup.sql
```

### 2. Tenant Onboarding Script

**create-tenant.sh**
```bash
#!/bin/bash
set -euo pipefail

TENANT_ID=$1
TENANT_NAME=$2
TIER=${3:-standard}

echo "Creating tenant: ${TENANT_ID}"

# Create tenant in database
kubectl exec -n vcci-production -it deployment/vcci-api -- \
  python manage.py create_tenant \
    --tenant-id "${TENANT_ID}" \
    --name "${TENANT_NAME}" \
    --tier "${TIER}"

# Create tenant schema
kubectl exec -n vcci-production -it deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT create_tenant_schema('${TENANT_ID}');"

# Create tenant namespace (optional, for strict isolation)
if [ "${TIER}" == "enterprise" ]; then
  kubectl create namespace "tenant-${TENANT_ID}"
  kubectl label namespace "tenant-${TENANT_ID}" tenant-id="${TENANT_ID}"
fi

# Generate API key
API_KEY=$(kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py generate_api_key --tenant-id "${TENANT_ID}")

echo ""
echo "Tenant created successfully!"
echo "Tenant ID: ${TENANT_ID}"
echo "API Key: ${API_KEY}"
echo ""
echo "Save this API key securely - it cannot be retrieved again."
```

### 3. Resource Quotas

**Configure Tenant Resource Limits**
```yaml
# k8s/resource-quotas/tenant-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-{{ TENANT_ID }}-quota
  namespace: tenant-{{ TENANT_ID }}
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "5"
    services.loadbalancers: "1"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: tenant-{{ TENANT_ID }}-limits
  namespace: tenant-{{ TENANT_ID }}
spec:
  limits:
  - max:
      cpu: "4"
      memory: 8Gi
    min:
      cpu: 100m
      memory: 128Mi
    default:
      cpu: "1"
      memory: 2Gi
    defaultRequest:
      cpu: 500m
      memory: 1Gi
    type: Container
```

---

## Post-Deployment Validation

### 1. Health Checks

**API Health Check**
```bash
#!/bin/bash

API_URL="https://api.vcci-platform.com"

echo "Checking API health..."
curl -f "${API_URL}/health/live" || exit 1
curl -f "${API_URL}/health/ready" || exit 1

echo "API health check passed!"
```

**Database Connection Test**
```bash
kubectl exec -n vcci-production deployment/vcci-api -- \
  python -c "
from sqlalchemy import create_engine
import os
engine = create_engine(os.environ['DATABASE_URL'])
with engine.connect() as conn:
    result = conn.execute('SELECT 1')
    print('Database connection successful!')
"
```

**Redis Connection Test**
```bash
kubectl exec -n vcci-production deployment/vcci-api -- \
  python -c "
import redis
import os
r = redis.from_url(os.environ['REDIS_URL'])
r.ping()
print('Redis connection successful!')
"
```

### 2. Smoke Tests

**Run Smoke Test Suite**
```bash
#!/bin/bash
set -euo pipefail

API_URL="https://api.vcci-platform.com"
API_KEY="your-api-key"

echo "Running smoke tests..."

# Test 1: Authentication
echo "Test 1: Authentication"
AUTH_RESPONSE=$(curl -s -X POST "${API_URL}/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"test123"}')
TOKEN=$(echo $AUTH_RESPONSE | jq -r '.access_token')
echo "✓ Authentication successful"

# Test 2: Create emission record
echo "Test 2: Create emission record"
EMISSION_RESPONSE=$(curl -s -X POST "${API_URL}/api/v1/emissions" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "purchased_goods",
    "scope": 3,
    "amount": 1500.50,
    "unit": "kg_co2e",
    "source_id": "test-001"
  }')
EMISSION_ID=$(echo $EMISSION_RESPONSE | jq -r '.id')
echo "✓ Emission record created: ${EMISSION_ID}"

# Test 3: Retrieve emission record
echo "Test 3: Retrieve emission record"
curl -sf "${API_URL}/api/v1/emissions/${EMISSION_ID}" \
  -H "Authorization: Bearer ${TOKEN}" > /dev/null
echo "✓ Emission record retrieved"

# Test 4: Calculate hotspots
echo "Test 4: Calculate hotspots"
curl -sf "${API_URL}/api/v1/analytics/hotspots" \
  -H "Authorization: Bearer ${TOKEN}" > /dev/null
echo "✓ Hotspot calculation successful"

echo ""
echo "All smoke tests passed!"
```

### 3. Performance Validation

**Load Test**
```bash
# Install k6
curl https://github.com/grafana/k6/releases/download/v0.47.0/k6-v0.47.0-linux-amd64.tar.gz -L | tar xvz

# Run load test
./k6 run - <<EOF
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 10 },
    { duration: '5m', target: 10 },
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  let res = http.get('https://api.vcci-platform.com/health/ready');
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  sleep(1);
}
EOF
```

### 4. Security Validation

**Run Security Scan**
```bash
# Scan container images
trivy image your-registry.com/vcci-platform:v2.0.0

# Check Kubernetes security
kubesec scan k8s/deployments/api.yaml

# Run compliance check
kubectl apply -f https://raw.githubusercontent.com/aquasecurity/kube-bench/main/job.yaml
kubectl logs -f job/kube-bench
```

---

## Rollback Procedures

### 1. Application Rollback

**Quick Rollback (Kubernetes)**
```bash
# Check deployment history
kubectl rollout history deployment/vcci-api -n vcci-production

# Rollback to previous version
kubectl rollout undo deployment/vcci-api -n vcci-production

# Rollback to specific revision
kubectl rollout undo deployment/vcci-api -n vcci-production --to-revision=5

# Monitor rollback
kubectl rollout status deployment/vcci-api -n vcci-production
```

**Database Rollback**
```bash
# List migrations
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py showmigrations

# Rollback to specific migration
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py migrate app_name migration_name

# Example: Rollback to 0012
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py migrate emissions 0012_previous_migration
```

### 2. Infrastructure Rollback

**Terraform Rollback**
```bash
cd terraform/environments/production

# Show state history
terraform state list

# Rollback using previous plan
terraform plan -out=rollback.tfplan -var-file=previous-version.tfvars

# Review and apply
terraform apply rollback.tfplan
```

### 3. Complete System Rollback

**Emergency Rollback Script**
```bash
#!/bin/bash
set -euo pipefail

PREVIOUS_VERSION=$1

echo "=========================================="
echo "EMERGENCY ROLLBACK TO VERSION ${PREVIOUS_VERSION}"
echo "=========================================="

# 1. Scale down current deployment
echo "Scaling down current deployment..."
kubectl scale deployment/vcci-api -n vcci-production --replicas=0

# 2. Rollback database migrations
echo "Rolling back database..."
kubectl exec -n vcci-production deployment/vcci-api -- \
  python manage.py migrate --database=default ${PREVIOUS_VERSION}

# 3. Deploy previous version
echo "Deploying previous version..."
kubectl set image deployment/vcci-api -n vcci-production \
  api=your-registry.com/vcci-platform:${PREVIOUS_VERSION}

# 4. Scale up deployment
echo "Scaling up deployment..."
kubectl scale deployment/vcci-api -n vcci-production --replicas=6

# 5. Wait for rollout
echo "Waiting for rollout to complete..."
kubectl rollout status deployment/vcci-api -n vcci-production

# 6. Verify health
echo "Verifying application health..."
sleep 30
kubectl exec -n vcci-production deployment/vcci-api -- \
  curl -f http://localhost:8000/health/ready

echo ""
echo "Rollback completed successfully!"
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Pods Not Starting

**Symptoms**:
```
kubectl get pods -n vcci-production
NAME                        READY   STATUS             RESTARTS
vcci-api-7d8f9c6b5d-abcde   0/1     CrashLoopBackOff   5
```

**Diagnosis**:
```bash
# Check pod logs
kubectl logs -n vcci-production vcci-api-7d8f9c6b5d-abcde

# Check pod events
kubectl describe pod -n vcci-production vcci-api-7d8f9c6b5d-abcde

# Check previous container logs
kubectl logs -n vcci-production vcci-api-7d8f9c6b5d-abcde --previous
```

**Common Causes**:
1. Missing or incorrect secrets
2. Database connection issues
3. Insufficient resources
4. Image pull errors

**Resolution**:
```bash
# Verify secrets exist
kubectl get secrets -n vcci-production

# Test database connection
kubectl exec -n vcci-production -it deployment/vcci-api -- \
  psql $DATABASE_URL -c "SELECT 1"

# Check resource allocation
kubectl top pods -n vcci-production
```

#### Issue 2: Database Connection Timeout

**Symptoms**:
```
Error: could not connect to server: Connection timed out
```

**Resolution**:
```bash
# Check security groups
aws ec2 describe-security-groups --group-ids sg-xxxxx

# Verify database endpoint
aws rds describe-db-instances --db-instance-identifier vcci-production

# Test connection from pod
kubectl exec -n vcci-production -it deployment/vcci-api -- \
  nc -zv $DATABASE_HOST $DATABASE_PORT
```

#### Issue 3: Certificate Issues

**Symptoms**:
```
Error: certificate has expired or is not yet valid
```

**Resolution**:
```bash
# Check certificate status
kubectl get certificate -n vcci-production

# Describe certificate
kubectl describe certificate vcci-api-tls -n vcci-production

# Force certificate renewal
kubectl delete certificate vcci-api-tls -n vcci-production
kubectl apply -f k8s/ingress/api-ingress.yaml
```

---

## Appendix

### A. Deployment Checklist

**Pre-Deployment**
- [ ] Infrastructure plan reviewed and approved
- [ ] Terraform state backend configured
- [ ] Secrets created in secrets manager
- [ ] DNS records prepared
- [ ] Certificates obtained
- [ ] Backup strategy defined
- [ ] Monitoring alerts configured
- [ ] On-call rotation established

**During Deployment**
- [ ] Infrastructure deployed successfully
- [ ] Kubernetes cluster accessible
- [ ] Core components installed (Ingress, Cert-Manager, etc.)
- [ ] Application deployed
- [ ] Database migrations applied
- [ ] Secrets and configmaps created
- [ ] Health checks passing

**Post-Deployment**
- [ ] Smoke tests passed
- [ ] Load tests passed
- [ ] Security scans completed
- [ ] Monitoring dashboards verified
- [ ] Backup jobs verified
- [ ] Documentation updated
- [ ] Runbooks reviewed
- [ ] Team trained on operations

### B. Contact Information

**Escalation Path**
1. Level 1: Platform Team (platform-team@company.com)
2. Level 2: Engineering Manager (eng-manager@company.com)
3. Level 3: CTO (cto@company.com)

**On-Call**
- PagerDuty: https://company.pagerduty.com
- Slack Channel: #vcci-platform-ops
- Phone: +1-555-0123 (24/7 hotline)

### C. References

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [VCCI Platform Architecture](./ARCHITECTURE.md)
- [Operations Guide](./OPERATIONS_GUIDE.md)
- [Security Guide](./SECURITY_GUIDE.md)

---

**Document Version**: 1.0.0
**Last Updated**: 2025-01-06
**Maintained By**: Platform Engineering Team
