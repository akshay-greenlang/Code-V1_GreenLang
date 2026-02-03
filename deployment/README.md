# GreenLang Platform Deployment

**Version:** 1.0.0
**Task ID:** INFRA-001
**Last Updated:** 2026-02-03
**Status:** Production Ready

---

## Overview

This directory contains the complete deployment infrastructure for the GreenLang platform, supporting three applications (CBAM, CSRD, VCCI) with shared infrastructure including PostgreSQL, Redis, RabbitMQ, and comprehensive monitoring.

```
                           GreenLang Platform Architecture
+--------------------------------------------------------------------------------+
|                                                                                |
|    +---------------------------+  +---------------------------+               |
|    |       CBAM API            |  |       CSRD Web           |               |
|    |       (Port 8001)         |  |       (Port 8002)        |               |
|    +---------------------------+  +---------------------------+               |
|                                                                                |
|    +---------------------------+  +---------------------------+               |
|    |    VCCI Backend           |  |    VCCI Worker           |               |
|    |    (Port 8000)            |  |    (Background)          |               |
|    +---------------------------+  +---------------------------+               |
|                                                                                |
|    +-----------------------------------------------------------------------+  |
|    |                     Shared Infrastructure                              |  |
|    |  +-------------+  +-----------+  +------------+  +---------------+   |  |
|    |  | PostgreSQL  |  |   Redis   |  |  RabbitMQ  |  |   Weaviate    |   |  |
|    |  | (5432)      |  |  (6379)   |  |   (5672)   |  |    (8080)     |   |  |
|    |  +-------------+  +-----------+  +------------+  +---------------+   |  |
|    +-----------------------------------------------------------------------+  |
|                                                                                |
|    +-----------------------------------------------------------------------+  |
|    |                         Monitoring                                     |  |
|    |  +-------------+  +-----------+  +------------+                       |  |
|    |  | Prometheus  |  |  Grafana  |  |  pgAdmin   |                       |  |
|    |  | (9090)      |  |  (3000)   |  |  (5050)    |                       |  |
|    |  +-------------+  +-----------+  +------------+                       |  |
|    +-----------------------------------------------------------------------+  |
|                                                                                |
+--------------------------------------------------------------------------------+
```

---

## Quick Start

### Option 1: Local Development (Docker Compose)

```bash
# Navigate to deployment directory
cd deployment

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and passwords

# Start the platform
./deploy.sh start

# Check status
./deploy.sh health
```

### Option 2: Cloud Deployment (AWS/Kubernetes)

```bash
# Full deployment to development
make deploy-all ENV=dev

# Or use the comprehensive script
./deploy-all.sh dev

# For production (requires approvals)
./deploy-all.sh prod
```

---

## Directory Structure

```
deployment/
|
+-- README.md                       # This file
+-- FINAL-CHECKLIST.md              # Complete deployment checklist
+-- DEPLOYMENT_GUIDE.md             # Detailed deployment guide
+-- QUICK_START.md                  # 5-minute quick start
|
+-- Makefile                        # Make targets for deployment
+-- deploy.sh                       # Docker Compose deployment
+-- deploy-all.sh                   # Full AWS/K8s deployment script
|
+-- .env.example                    # Environment variables template
+-- docker-compose-unified.yml      # Docker Compose configuration
+-- validate_integration.py         # Integration validation script
|
+-- terraform/                      # Infrastructure as Code
|   +-- modules/
|   |   +-- vpc/                    # VPC, subnets, NAT gateways
|   |   +-- eks/                    # EKS cluster and node groups
|   |   +-- rds/                    # PostgreSQL database
|   |   +-- elasticache/            # Redis cache
|   |   +-- s3/                     # S3 buckets
|   |   +-- iam/                    # IAM roles and policies
|   |   +-- vault/                  # HashiCorp Vault (optional)
|   |   +-- keycloak/               # Keycloak IAM (optional)
|   |
|   +-- environments/
|   |   +-- dev/                    # Development environment
|   |   +-- staging/                # Staging environment
|   |   +-- prod/                   # Production environment
|   |
|   +-- scripts/                    # Infrastructure scripts
|   +-- INFRA-001-TASKS.yaml        # Ralphy automation config
|
+-- infrastructure/
|   +-- helm/
|   |   +-- greenlang/              # Helm chart for GreenLang
|   |       +-- Chart.yaml
|   |       +-- values.yaml
|   |       +-- values-dev.yaml
|   |       +-- values-staging.yaml
|   |       +-- values-prod.yaml
|   |       +-- templates/
|   |
|   +-- kubernetes/
|       +-- greenlang/              # Raw Kubernetes manifests
|           +-- base/               # Namespace, RBAC
|           +-- data/               # Redis, PVCs
|           +-- workers/            # Worker deployments
|           +-- monitoring/         # Dashboards
|
+-- kubernetes/                     # Additional K8s manifests
|   +-- manifests/                  # Core manifests
|   +-- Dockerfile                  # K8s-specific Dockerfile
|
+-- docker/                         # Docker configurations
|   +-- Full.Dockerfile
|   +-- Runner.Dockerfile
|   +-- weaviate/                   # Weaviate configuration
|
+-- monitoring/
|   +-- prometheus-unified.yml      # Prometheus configuration
|   +-- alerts-unified.yml          # Alert rules
|   +-- grafana-provisioning/       # Grafana dashboards
|
+-- security/                       # Security configurations
|   +-- audits/                     # Security audit reports
|   +-- scripts/                    # Security scanning scripts
|   +-- README.md                   # Security documentation
|
+-- init/
|   +-- shared_db_schema.sql        # Database initialization
|
+-- sbom/                           # Software Bill of Materials
|
+-- workflows/                      # Sample workflows
```

---

## Deployment Options

### 1. Make Targets (Recommended)

```bash
# Show all available targets
make help

# Initialize Terraform
make init ENV=prod

# Plan infrastructure changes
make plan ENV=prod

# Apply infrastructure
make apply ENV=prod

# Deploy to Kubernetes
make deploy-k8s ENV=prod

# Run validation
make validate ENV=prod

# Destroy (use with caution!)
make destroy ENV=prod

# Full deployment pipeline
make deploy-all ENV=prod
```

### 2. Deploy Script

```bash
# Full deployment with all options
./deploy-all.sh prod

# Dry run (show what would happen)
./deploy-all.sh prod --dry-run

# Skip infrastructure (K8s only)
./deploy-all.sh prod --skip-infra

# Auto-approve (dev/staging only)
./deploy-all.sh dev --auto-approve

# Rollback
./deploy-all.sh prod --rollback
```

### 3. Docker Compose (Local)

```bash
# Start all services
./deploy.sh start

# Stop services
./deploy.sh stop

# View status
./deploy.sh status

# Health checks
./deploy.sh health

# View logs
./deploy.sh logs

# Rebuild images
./deploy.sh build

# Clean up (removes data!)
./deploy.sh clean
```

---

## Environment Configuration

### Required Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `POSTGRES_PASSWORD` | PostgreSQL admin password | Yes |
| `REDIS_PASSWORD` | Redis authentication password | Yes |
| `RABBITMQ_PASSWORD` | RabbitMQ password | Yes |
| `SHARED_SECRET_KEY` | Platform encryption key (32+ chars) | Yes |
| `SHARED_JWT_SECRET` | JWT signing secret (32+ chars) | Yes |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `PINECONE_API_KEY` | Pinecone API key | Optional |

### Generate Secure Secrets

```bash
# Generate random secrets
openssl rand -hex 32  # For SHARED_SECRET_KEY
openssl rand -hex 32  # For SHARED_JWT_SECRET
openssl rand -hex 16  # For POSTGRES_PASSWORD
```

---

## Environments

| Environment | Auto-Approve | Multi-AZ | Replicas | Use Case |
|-------------|--------------|----------|----------|----------|
| dev | Yes | No | 1 | Development/testing |
| staging | Yes | No | 2 | Pre-production validation |
| prod | No (2 approvers) | Yes | 3+ | Production workloads |

---

## Infrastructure Components

### AWS Resources (Terraform)

| Component | Service | Description |
|-----------|---------|-------------|
| Network | VPC | Isolated network with public/private subnets |
| Compute | EKS | Managed Kubernetes cluster |
| Database | RDS PostgreSQL | Multi-AZ relational database |
| Cache | ElastiCache Redis | In-memory cache with replication |
| Storage | S3 | Object storage for artifacts/backups |
| Security | IAM | IRSA for pod-level permissions |
| Secrets | Secrets Manager | Secure secret storage |

### Kubernetes Add-ons

| Component | Purpose |
|-----------|---------|
| cert-manager | TLS certificate management |
| ingress-nginx | Load balancer and ingress |
| external-secrets | AWS Secrets Manager integration |
| cluster-autoscaler | Node auto-scaling |
| kube-prometheus-stack | Monitoring and alerting |

---

## Monitoring & Observability

### Prometheus Metrics

```promql
# Total HTTP requests
sum(rate(http_requests_total[5m])) by (app)

# Request latency p99
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))
```

### Grafana Dashboards

- GreenLang Platform Overview
- Application Performance
- Infrastructure Health
- Database Metrics
- Redis Cache Stats

### Alert Rules

Critical alerts are configured for:
- Pod failures
- High error rates (>1%)
- Database connectivity
- Redis memory usage
- Certificate expiration

---

## Security

### Best Practices Implemented

- [x] Non-root containers
- [x] Read-only root filesystem
- [x] Network policies
- [x] Pod security standards
- [x] Encryption at rest (RDS, S3, ElastiCache)
- [x] Encryption in transit (TLS everywhere)
- [x] IRSA for AWS access
- [x] Secrets in AWS Secrets Manager
- [x] Regular security scanning

### Security Documentation

- `security/README.md` - Security overview
- `security/audits/` - Audit reports
- `security/scripts/` - Scanning scripts

---

## Documentation

### Core Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - deployment overview |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Detailed deployment instructions |
| [QUICK_START.md](QUICK_START.md) | 5-minute quick start guide |

### Architecture and Operations (INFRA-001)

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, components, data flow, security boundaries |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Day-to-day operations, scaling, backup/restore, incident response |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues, debugging commands, log locations, support contacts |

### Infrastructure Guides

| Document | Description |
|----------|-------------|
| [cost-estimation.md](cost-estimation.md) | AWS cost estimates |
| [environment-sizing-guide.md](environment-sizing-guide.md) | Resource sizing guide |
| [platform-disaster-recovery.md](platform-disaster-recovery.md) | Disaster recovery procedures |
| [shared-infrastructure.md](shared-infrastructure.md) | Infrastructure sharing |
| [cross-application-integration.md](cross-application-integration.md) | Application integration |

### Security Documentation

| Document | Description |
|----------|-------------|
| [security/README.md](security/README.md) | Security overview |
| [security/SECURITY_AUDIT_EXECUTIVE_SUMMARY.md](security/SECURITY_AUDIT_EXECUTIVE_SUMMARY.md) | Security audit summary |
| [security/audits/](security/audits/) | Detailed security audit reports |

---

## Troubleshooting

For comprehensive troubleshooting, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

### Quick Diagnostic Commands

```bash
# Platform health check
./deploy.sh health

# Container status
docker-compose -f docker-compose-unified.yml ps

# View logs
docker-compose -f docker-compose-unified.yml logs --tail=100

# Resource usage
docker stats --no-stream
```

### Common Issues

| Issue | Quick Fix |
|-------|-----------|
| Services won't start | Check ports: `lsof -i :8001` |
| Database connection error | Verify PostgreSQL: `docker-compose ps postgres` |
| Out of memory | Increase Docker memory or run `docker system prune` |
| Slow responses | Check database queries and restart applications |

#### Terraform State Lock
```bash
# Force unlock (use with caution)
terraform force-unlock LOCK_ID
```

#### Kubernetes Pods Not Starting
```bash
# Check pod events
kubectl describe pod POD_NAME -n greenlang

# Check logs
kubectl logs POD_NAME -n greenlang

# Check node resources
kubectl top nodes
```

#### Database Connectivity
```bash
# Test from within cluster
kubectl run psql-test --rm -it --image=postgres:14 -- \
  psql -h greenlang-prod-postgres.xxx.rds.amazonaws.com -U greenlang_admin
```

#### Health Check Failures
```bash
# Run validation
make validate ENV=prod

# Check specific health endpoints
curl http://localhost:8001/health  # CBAM
curl http://localhost:8002/health  # CSRD
curl http://localhost:8000/health/live  # VCCI
```

### Getting Help

- **Detailed Troubleshooting:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **Operations Guide:** [docs/OPERATIONS.md](docs/OPERATIONS.md)
- **Logs:** Check `deployment/logs/` directory
- **Diagnostics:** Run `make validate ENV=<env>`
- **Slack:** #greenlang-platform
- **On-call:** PagerDuty schedule

---

## Contributing

1. Create a feature branch
2. Make changes
3. Run `terraform fmt` and `terraform validate`
4. Test in dev environment
5. Create pull request
6. Await review and approval

---

## License

Copyright 2024 GreenLang. All rights reserved.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-03 | Initial INFRA-001 release |

---

**Happy Deploying!**
