# VCCI Scope 3 Carbon Intelligence Platform - Kubernetes Infrastructure

## Overview

This directory contains production-ready Kubernetes manifests for deploying the VCCI Scope 3 Carbon Intelligence Platform as a multi-tenant SaaS application. The infrastructure is designed for high availability, scalability, security, and observability.

## Architecture

### Components

- **API Gateway**: NGINX Ingress with rate limiting, TLS termination, and load balancing
- **Backend API**: FastAPI/Flask application with horizontal autoscaling (3-10 replicas)
- **Workers**: Celery workers for background processing (2-20 replicas)
  - Standard workers for general tasks
  - ML workers with GPU support for machine learning workloads
- **Frontend**: React SPA served by NGINX (2-5 replicas)
- **PostgreSQL**: Highly available database cluster (3 replicas) with replication
- **Redis**: Cluster mode cache (3 replicas) with Sentinel
- **Weaviate**: Vector database for ML/AI workloads (3 replicas)

### Observability Stack

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Dashboards and visualization
- **Fluentd**: Log aggregation (DaemonSet)
- **Jaeger**: Distributed tracing with OpenTelemetry

### Security Features

- **Network Policies**: Strict ingress/egress rules with default deny
- **RBAC**: Role-based access control for all service accounts
- **Pod Security Policies**: Enforce security standards (restricted by default)
- **TLS Certificates**: Automated certificate management with cert-manager
- **Sealed Secrets**: Encrypted secrets management
- **Resource Quotas**: Per-namespace resource limits
- **Multi-tenant Isolation**: Namespace-level tenant separation

## Directory Structure

```
infrastructure/kubernetes/
├── README.md                          # This file
├── base/                              # Base infrastructure
│   ├── namespace.yaml                 # Namespace definitions
│   ├── rbac.yaml                      # RBAC roles and bindings
│   ├── resource-quotas.yaml           # Resource quotas per namespace
│   └── network-policies.yaml          # Network isolation policies
├── applications/                      # Application deployments
│   ├── api-gateway/                   # NGINX API Gateway
│   ├── backend-api/                   # Backend API service
│   ├── worker/                        # Background workers
│   └── frontend/                      # React frontend
├── data/                              # Data layer
│   ├── postgresql/                    # PostgreSQL StatefulSet
│   ├── redis/                         # Redis cluster
│   └── weaviate/                      # Weaviate vector DB
├── observability/                     # Monitoring and logging
│   ├── prometheus/                    # Metrics collection
│   ├── grafana/                       # Dashboards
│   ├── fluentd/                       # Log aggregation
│   └── jaeger/                        # Distributed tracing
├── security/                          # Security resources
│   ├── cert-manager/                  # TLS certificates
│   ├── sealed-secrets/                # Encrypted secrets
│   └── pod-security-policies.yaml     # Pod security standards
└── kustomization/                     # Environment-specific configs
    ├── base/                          # Base kustomization
    └── overlays/
        ├── dev/                       # Development environment
        ├── staging/                   # Staging environment
        └── production/                # Production environment
```

## Prerequisites

### Required Tools

1. **kubectl** (v1.27+)
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

2. **kustomize** (v5.0+)
```bash
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/
```

3. **helm** (v3.12+)
```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

4. **kubeseal** (for sealed secrets)
```bash
wget https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/kubeseal-linux-amd64
sudo install -m 755 kubeseal-linux-amd64 /usr/local/bin/kubeseal
```

### Kubernetes Cluster Requirements

- **Version**: Kubernetes 1.27 or higher
- **Nodes**: Minimum 3 worker nodes
- **Node Pools**:
  - **Compute Pool**: CPU-optimized for API and workers
  - **Memory Pool**: Memory-optimized for databases
  - **GPU Pool**: GPU-enabled for ML workloads (optional)
- **Storage Classes**:
  - `fast-ssd`: High-performance SSD storage (for databases)
  - `standard`: Standard persistent storage
  - `nfs-storage`: Network file system (for shared storage)
- **Ingress Controller**: NGINX Ingress Controller
- **Cert Manager**: For automated TLS certificate management

## Quick Start

### 1. Cluster Setup

#### Option A: AWS EKS
```bash
# Install eksctl
curl --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster with node groups
eksctl create cluster \
  --name vcci-platform \
  --region us-east-1 \
  --version 1.27 \
  --nodegroup-name compute \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed

# Add memory-optimized node group
eksctl create nodegroup \
  --cluster vcci-platform \
  --name memory \
  --node-type r5.xlarge \
  --nodes 2 \
  --nodes-min 2 \
  --nodes-max 5 \
  --node-labels workload-type=memory

# Add GPU node group (optional)
eksctl create nodegroup \
  --cluster vcci-platform \
  --name gpu \
  --node-type p3.2xlarge \
  --nodes 1 \
  --nodes-min 0 \
  --nodes-max 3 \
  --node-labels workload-type=gpu \
  --node-taints nvidia.com/gpu=true:NoSchedule
```

#### Option B: GKE
```bash
gcloud container clusters create vcci-platform \
  --region us-central1 \
  --cluster-version 1.27 \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10 \
  --enable-network-policy \
  --enable-ip-alias

# Add node pools
gcloud container node-pools create memory \
  --cluster vcci-platform \
  --machine-type n1-highmem-4 \
  --num-nodes 2 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 5 \
  --node-labels workload-type=memory
```

### 2. Install Required Controllers

```bash
# Install NGINX Ingress Controller
helm upgrade --install ingress-nginx ingress-nginx \
  --repo https://kubernetes.github.io/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.metrics.enabled=true \
  --set controller.podAnnotations."prometheus\.io/scrape"=true \
  --set controller.podAnnotations."prometheus\.io/port"=10254

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Install sealed-secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Install Prometheus Operator (optional, for ServiceMonitors)
helm install prometheus-operator prometheus-community/kube-prometheus-stack \
  --namespace vcci-observability \
  --create-namespace
```

### 3. Configure Secrets

#### Create Base Secrets

```bash
# Create secrets directory (don't commit this!)
mkdir -p secrets

# Backend API secrets
cat > secrets/backend-secrets.env << EOF
DATABASE_URL=postgresql://vcci:CHANGE_ME@postgresql:5432/vcci
REDIS_URL=redis://:CHANGE_ME@redis:6379/0
WEAVIATE_API_KEY=CHANGE_ME
JWT_SECRET=$(openssl rand -base64 32)
OPENAI_API_KEY=sk-YOUR_KEY_HERE
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
EOF

# PostgreSQL secrets
kubectl create secret generic postgresql-secrets \
  --from-literal=username=vcci \
  --from-literal=password=$(openssl rand -base64 32) \
  --from-literal=exporter-dsn="postgresql://vcci_monitor:$(openssl rand -base64 32)@localhost:5432/vcci?sslmode=disable" \
  --namespace vcci-platform \
  --dry-run=client -o yaml > secrets/postgresql-secrets.yaml

# Seal the secrets
kubeseal -f secrets/postgresql-secrets.yaml -w security/sealed-secrets/postgresql-sealed.yaml

# Redis secrets
kubectl create secret generic redis-secrets \
  --from-literal=password=$(openssl rand -base64 32) \
  --namespace vcci-platform \
  --dry-run=client -o yaml | \
  kubeseal -w security/sealed-secrets/redis-sealed.yaml

# Grafana secrets
kubectl create secret generic grafana-secrets \
  --from-literal=admin-user=admin \
  --from-literal=admin-password=$(openssl rand -base64 32) \
  --namespace vcci-observability \
  --dry-run=client -o yaml | \
  kubeseal -w security/sealed-secrets/grafana-sealed.yaml
```

### 4. Deploy to Development

```bash
# Apply base infrastructure
kubectl apply -k kustomization/base/

# Or use environment-specific overlay
kubectl apply -k kustomization/overlays/dev/

# Verify deployment
kubectl get pods -n vcci-dev
kubectl get services -n vcci-dev
kubectl get ingress -n vcci-dev
```

### 5. Deploy to Production

```bash
# Review the configuration
kubectl kustomize kustomization/overlays/production/

# Apply to production
kubectl apply -k kustomization/overlays/production/

# Verify deployment
kubectl get pods -n vcci-platform
kubectl get services -n vcci-platform
kubectl get ingress -n vcci-platform

# Check HPA status
kubectl get hpa -n vcci-platform

# Check PDB status
kubectl get pdb -n vcci-platform
```

## Multi-Tenant Setup

### Creating a New Tenant

```bash
# Set tenant information
TENANT_ID="acme-corp"
TENANT_NAME="ACME Corporation"
TENANT_TIER="enterprise"

# Create tenant namespace
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: vcci-tenant-${TENANT_ID}
  labels:
    tenant: "${TENANT_ID}"
    tenant-tier: "${TENANT_TIER}"
    app.kubernetes.io/part-of: vcci-scope3-platform
  annotations:
    tenant.vcci.io/id: "${TENANT_ID}"
    tenant.vcci.io/name: "${TENANT_NAME}"
    tenant.vcci.io/tier: "${TENANT_TIER}"
    tenant.vcci.io/created: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF

# Apply resource quotas based on tier
kubectl apply -f base/resource-quotas.yaml --selector=tier=${TENANT_TIER}

# Apply network policies for isolation
kubectl apply -f base/network-policies.yaml -n vcci-tenant-${TENANT_ID}

# Create tenant-specific RBAC
kubectl apply -f base/rbac.yaml -n vcci-tenant-${TENANT_ID}

# Verify tenant setup
kubectl get namespace vcci-tenant-${TENANT_ID} -o yaml
kubectl get resourcequota -n vcci-tenant-${TENANT_ID}
kubectl get networkpolicy -n vcci-tenant-${TENANT_ID}
```

## Configuration Management

### Environment Variables

Use ConfigMaps for non-sensitive configuration:

```bash
# Update platform config
kubectl create configmap platform-config \
  --from-literal=ENVIRONMENT=production \
  --from-literal=LOG_LEVEL=INFO \
  --from-literal=API_VERSION=v1 \
  --namespace vcci-platform \
  --dry-run=client -o yaml | kubectl apply -f -

# Rollout restart to pick up changes
kubectl rollout restart deployment/vcci-backend-api -n vcci-platform
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment vcci-backend-api --replicas=5 -n vcci-platform

# Check HPA status
kubectl get hpa vcci-backend-api-hpa -n vcci-platform

# Adjust HPA
kubectl patch hpa vcci-backend-api-hpa -n vcci-platform \
  --type=merge -p '{"spec":{"minReplicas":5,"maxReplicas":15}}'
```

### Rolling Updates

```bash
# Update image
kubectl set image deployment/vcci-backend-api \
  api=greenlang/vcci-backend-api:v1.1.0 \
  -n vcci-platform

# Watch rollout
kubectl rollout status deployment/vcci-backend-api -n vcci-platform

# Rollback if needed
kubectl rollout undo deployment/vcci-backend-api -n vcci-platform

# Check revision history
kubectl rollout history deployment/vcci-backend-api -n vcci-platform
```

## Monitoring and Observability

### Access Grafana

```bash
# Port-forward to access locally
kubectl port-forward svc/grafana 3000:3000 -n vcci-observability

# Or get the external URL
kubectl get ingress grafana -n vcci-observability
```

### Access Prometheus

```bash
kubectl port-forward svc/prometheus 9090:9090 -n vcci-observability
```

### Access Jaeger

```bash
kubectl port-forward svc/jaeger-all-in-one 16686:16686 -n vcci-observability
```

### View Logs

```bash
# View logs for a specific pod
kubectl logs -f deployment/vcci-backend-api -n vcci-platform

# View logs with labels
kubectl logs -l app=vcci-backend-api -n vcci-platform --tail=100

# View logs from all containers in a pod
kubectl logs -f pod/vcci-backend-api-xxxxx --all-containers -n vcci-platform
```

### Metrics

```bash
# Get pod metrics
kubectl top pods -n vcci-platform

# Get node metrics
kubectl top nodes

# Get HPA metrics
kubectl get hpa -n vcci-platform -w
```

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl get pods -n vcci-platform
kubectl describe pod <pod-name> -n vcci-platform

# Check events
kubectl get events -n vcci-platform --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n vcci-platform --previous
```

#### 2. Network Connectivity Issues

```bash
# Test DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup vcci-backend-api.vcci-platform.svc.cluster.local

# Test connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- curl http://vcci-backend-api.vcci-platform.svc.cluster.local:8000/health

# Check network policies
kubectl get networkpolicy -n vcci-platform
kubectl describe networkpolicy <policy-name> -n vcci-platform
```

#### 3. Certificate Issues

```bash
# Check certificate status
kubectl get certificate -n vcci-platform
kubectl describe certificate vcci-api-cert -n vcci-platform

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Manually trigger certificate renewal
kubectl delete certificaterequest <request-name> -n vcci-platform
```

#### 4. Database Connection Issues

```bash
# Check PostgreSQL status
kubectl get pods -l app=postgresql -n vcci-platform
kubectl logs -l app=postgresql -n vcci-platform

# Test connection
kubectl run -it --rm psql --image=postgres:15-alpine --restart=Never -- \
  psql -h postgresql.vcci-platform.svc.cluster.local -U vcci -d vcci

# Check Redis
kubectl exec -it redis-0 -n vcci-platform -- redis-cli ping
```

### Health Checks

```bash
# Check all health endpoints
kubectl get --raw /api/v1/namespaces/vcci-platform/services/vcci-backend-api:8000/proxy/health/live
kubectl get --raw /api/v1/namespaces/vcci-platform/services/vcci-backend-api:8000/proxy/health/ready
```

## Backup and Disaster Recovery

### PostgreSQL Backup

```bash
# Create backup job
kubectl create job postgresql-backup-$(date +%Y%m%d-%H%M%S) \
  --from=cronjob/postgresql-backup \
  -n vcci-platform

# Verify backup
kubectl logs job/postgresql-backup-XXXXXXX -n vcci-platform
```

### Restore from Backup

```bash
# Scale down applications
kubectl scale deployment vcci-backend-api --replicas=0 -n vcci-platform
kubectl scale deployment vcci-worker --replicas=0 -n vcci-platform

# Restore database
kubectl exec -it postgresql-0 -n vcci-platform -- \
  pg_restore -U vcci -d vcci /backup/vcci-backup-YYYYMMDD.dump

# Scale up applications
kubectl scale deployment vcci-backend-api --replicas=3 -n vcci-platform
kubectl scale deployment vcci-worker --replicas=2 -n vcci-platform
```

## Security Best Practices

1. **Always use RBAC**: Never use cluster-admin for application service accounts
2. **Enable Network Policies**: Implement default deny policies
3. **Use Pod Security Standards**: Enforce restricted or baseline standards
4. **Encrypt Secrets**: Use sealed-secrets or external secret management (AWS Secrets Manager, Vault)
5. **Enable Audit Logging**: Configure Kubernetes audit logs
6. **Regular Updates**: Keep Kubernetes and all components updated
7. **Scan Images**: Use image scanning tools (Trivy, Clair) in CI/CD
8. **Limit Resource Access**: Use resource quotas and limit ranges

## Performance Tuning

### Resource Optimization

```yaml
# Recommended resource settings for production
Backend API:
  requests: cpu=500m, memory=1Gi
  limits: cpu=2000m, memory=4Gi

Workers:
  requests: cpu=1000m, memory=2Gi
  limits: cpu=4000m, memory=8Gi

PostgreSQL:
  requests: cpu=1000m, memory=4Gi
  limits: cpu=4000m, memory=16Gi
```

### HPA Configuration

```yaml
# Aggressive scaling for APIs
minReplicas: 3
maxReplicas: 10
targetCPUUtilization: 70%
scaleDown: stabilizationWindowSeconds: 300
scaleUp: stabilizationWindowSeconds: 60
```

## Cost Optimization

1. **Right-size resources**: Use VPA (Vertical Pod Autoscaler) to recommend sizes
2. **Use spot instances**: For non-critical workloads (workers)
3. **Implement cluster autoscaler**: Scale nodes based on demand
4. **Use local SSD for cache**: Reduce EBS costs for Redis/cache layers
5. **Schedule non-critical jobs**: Run during off-peak hours
6. **Monitor unused resources**: Remove orphaned PVCs and load balancers

## Maintenance

### Cluster Upgrade

```bash
# Check current version
kubectl version

# Drain nodes one at a time
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Upgrade control plane
eksctl upgrade cluster --name vcci-platform --version 1.28

# Upgrade node groups
eksctl upgrade nodegroup --cluster vcci-platform --name compute

# Uncordon nodes
kubectl uncordon <node-name>
```

### Application Updates

```bash
# Update using Kustomize
kubectl apply -k kustomization/overlays/production/

# Monitor rollout
kubectl rollout status deployment/vcci-backend-api -n vcci-platform

# Verify health
kubectl get pods -n vcci-platform -w
```

## Support and Documentation

- **Internal Docs**: https://docs.vcci.greenlang.io
- **Runbooks**: ./runbooks/
- **Architecture**: ./docs/architecture.md
- **API Docs**: https://api.vcci.greenlang.io/docs
- **Support**: support@greenlang.io

## License

Proprietary - GreenLang VCCI Platform
© 2025 GreenLang. All rights reserved.
