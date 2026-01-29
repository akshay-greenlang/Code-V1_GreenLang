# GL Normalizer Infrastructure

Infrastructure configuration for **GL-FOUND-X-003: GreenLang Unit & Reference Normalizer**.

## Overview

This directory contains production-grade infrastructure configuration for deploying the GreenLang Normalizer system, including:

- **Docker** - Container images and local development environment
- **Kubernetes** - Orchestration manifests with Kustomize overlays
- **Terraform** - AWS infrastructure as code

## Directory Structure

```
infrastructure/
├── docker/
│   ├── Dockerfile.core              # Core library image
│   ├── Dockerfile.service           # API service image
│   ├── Dockerfile.review-console    # Review console web UI
│   ├── docker-compose.yaml          # Local development
│   ├── docker-compose.test.yaml     # CI/CD testing
│   └── init-scripts/                # Database initialization
│
├── kubernetes/
│   ├── base/                        # Base Kubernetes manifests
│   │   ├── namespace.yaml           # Namespace with security policies
│   │   ├── deployment.yaml          # Deployments for service & console
│   │   ├── service.yaml             # Services and Ingress
│   │   ├── configmap.yaml           # Application configuration
│   │   ├── secrets.yaml             # Secrets template
│   │   ├── hpa.yaml                 # Horizontal Pod Autoscaler
│   │   ├── pdb.yaml                 # Pod Disruption Budget
│   │   └── kustomization.yaml       # Base kustomization
│   └── overlays/
│       ├── dev/                     # Development environment
│       ├── staging/                 # Staging environment
│       └── prod/                    # Production environment
│
└── terraform/
    ├── modules/
    │   ├── eks/                     # AWS EKS cluster
    │   ├── rds/                     # PostgreSQL database
    │   ├── kafka/                   # AWS MSK (Kafka)
    │   └── s3/                      # S3 buckets for audit storage
    └── environments/
        ├── dev/                     # Development infrastructure
        └── prod/                    # Production infrastructure
```

## Quick Start

### Local Development

```bash
# Start all services
cd infrastructure/docker
docker compose up -d

# View logs
docker compose logs -f gl-normalizer-service

# Access services
# API: http://localhost:8000
# Review Console: http://localhost:8080
# Kafka UI: http://localhost:8082 (requires profile)
```

### Kubernetes Deployment

```bash
# Deploy to development
kubectl apply -k infrastructure/kubernetes/overlays/dev/

# Deploy to staging
kubectl apply -k infrastructure/kubernetes/overlays/staging/

# Deploy to production
kubectl apply -k infrastructure/kubernetes/overlays/prod/
```

### Terraform Infrastructure

```bash
# Initialize and plan
cd infrastructure/terraform/environments/dev
terraform init
terraform plan

# Apply infrastructure
terraform apply

# Get kubeconfig
aws eks update-kubeconfig --region us-east-1 --name gl-normalizer-dev
```

## Docker Images

### Building Images

```bash
# Build all images
docker compose build

# Build specific image
docker build -f docker/Dockerfile.service -t gl-normalizer-service:latest ../../
```

### Image Specifications

| Image | Base | Size | Purpose |
|-------|------|------|---------|
| `gl-normalizer-core` | python:3.11-slim | ~200MB | Core library |
| `gl-normalizer-service` | python:3.11-slim | ~300MB | REST API |
| `gl-normalizer-review-console` | python:3.11-slim | ~400MB | Web UI |

### Security Features

- Multi-stage builds for minimal attack surface
- Non-root user (UID 1000)
- Read-only root filesystem
- No shell in production images
- Health checks for all containers

## Kubernetes Configuration

### Resource Requirements

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|-----------|----------------|--------------|
| Service (prod) | 500m | 2 | 1Gi | 2Gi |
| Service (dev) | 100m | 500m | 256Mi | 512Mi |
| Console | 100m | 500m | 256Mi | 512Mi |

### Auto-scaling

- **HPA**: Scales based on CPU (70%) and Memory (80%)
- **Min Replicas**: 3 (prod), 1 (dev)
- **Max Replicas**: 20 (prod), 3 (dev)

### High Availability

- Pod Disruption Budget: min 2 available (prod)
- Pod Anti-Affinity: spread across nodes and zones
- Topology Spread: max skew of 1 across zones

## Terraform Modules

### EKS Module

Creates an EKS cluster with:
- Managed node groups (general + spot)
- OIDC provider for IRSA
- VPC CNI, CoreDNS, EBS CSI addons
- IAM roles for GL Normalizer service

### RDS Module

Provisions PostgreSQL with:
- Multi-AZ deployment (prod)
- Encryption at rest (KMS)
- Performance Insights
- Automated backups
- Read replica (optional)

### Kafka Module

Deploys MSK cluster with:
- SASL/SCRAM and SASL/IAM authentication
- TLS encryption in transit
- Prometheus monitoring
- S3 log archival

### S3 Module

Creates S3 buckets for:
- Audit event cold storage
- Vocabulary data
- Access logging

Features:
- Server-side encryption (KMS)
- Lifecycle policies (Glacier transition)
- Object Lock for immutability
- Cross-region replication (DR)

## Environment Configuration

### Development

- Single NAT gateway
- Smaller instance types
- No Multi-AZ for RDS
- Minimal monitoring
- Public EKS API endpoint

### Production

- NAT gateway per AZ
- Production instance types
- Multi-AZ RDS with read replica
- Full monitoring and alerting
- Private EKS API endpoint
- WAF protection
- Cross-region DR replication

## Security

### Network Security

- VPC with private subnets
- Security groups with least privilege
- Network policies in Kubernetes
- TLS everywhere

### Identity & Access

- IRSA for EKS pods
- Secrets Manager for credentials
- RBAC in Kubernetes
- Non-root containers

### Data Protection

- Encryption at rest (KMS)
- Encryption in transit (TLS)
- Object Lock for audit immutability
- Backup retention policies

## Monitoring

### Metrics

- Prometheus scraping enabled
- Custom metrics exposed at `/metrics`
- HPA metrics for scaling

### Logging

- Structured JSON logging
- CloudWatch integration
- Log retention policies

### Alerting

- CloudWatch alarms for AWS resources
- SNS integration for notifications

## Disaster Recovery

### Backup Strategy

| Component | Backup Frequency | Retention |
|-----------|------------------|-----------|
| RDS | Daily snapshots | 30 days |
| S3 | Versioning enabled | 7 years |
| Kafka | Log retention | 30 days |

### Cross-Region Replication

- S3 audit data replicated to DR region
- RDS read replica in separate AZ

## Troubleshooting

### Common Issues

**EKS Node Not Joining**
```bash
# Check node status
kubectl get nodes
kubectl describe node <node-name>
```

**Pod Not Starting**
```bash
# Check events and logs
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

**Database Connection Issues**
```bash
# Verify security group rules
aws ec2 describe-security-groups --group-ids <sg-id>
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Kubernetes readiness
kubectl get pods -l app.kubernetes.io/name=gl-normalizer-service
```

## Contributing

1. Make changes to infrastructure code
2. Run `terraform plan` to preview changes
3. Test in dev environment first
4. Create PR with plan output
5. Apply after approval

## License

Proprietary - GreenLang Platform
