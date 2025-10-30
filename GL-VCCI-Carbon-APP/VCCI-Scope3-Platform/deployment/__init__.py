# GL-VCCI Deployment Module
# Infrastructure-as-Code and deployment configurations

"""
VCCI Deployment
===============

Infrastructure-as-Code (IaC) and deployment configurations for production.

Deployment Options:
------------------
1. Kubernetes (Recommended)
   - Helm charts
   - Auto-scaling
   - High availability

2. Docker Compose (Development/Staging)
   - Single-machine deployment
   - Quick setup

Infrastructure Components:
-------------------------
- kubernetes/
  - deployments.yaml: Application deployments
  - services.yaml: Load balancers, networking
  - configmaps.yaml: Configuration
  - secrets.yaml: Encrypted credentials
  - ingress.yaml: Ingress controller

- terraform/
  - main.tf: Infrastructure definition
  - variables.tf: Configuration variables
  - outputs.tf: Outputs (IPs, endpoints)

- docker/
  - Dockerfile: Application container
  - docker-compose.yaml: Multi-container setup

Deployment Stages:
-----------------
1. Development: Local Docker Compose
2. Staging: Kubernetes cluster (single region)
3. Production: Kubernetes cluster (multi-region, HA)

Resources:
---------
- PostgreSQL: RDS Multi-AZ (production)
- Redis: ElastiCache cluster
- Weaviate: Self-hosted on K8s
- S3: Provenance storage

Usage:
------
```bash
# Deploy to staging
kubectl apply -f deployment/kubernetes/staging/

# Deploy to production
kubectl apply -f deployment/kubernetes/production/

# Terraform infrastructure
cd deployment/terraform
terraform init
terraform plan
terraform apply
```
"""

__version__ = "1.0.0"
