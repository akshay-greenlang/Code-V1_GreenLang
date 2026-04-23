# GL-005 CombustionControlAgent - Deployment Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-18
**Agent:** GL-005 CombustionControlAgent
**Type:** Real-time Control System
**SLA:** 99.9% uptime, <100ms latency

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Environment Configuration](#environment-configuration)
6. [Deployment Procedures](#deployment-procedures)
7. [Scaling & High Availability](#scaling--high-availability)
8. [Monitoring & Observability](#monitoring--observability)
9. [Troubleshooting](#troubleshooting)
10. [Rollback Procedures](#rollback-procedures)
11. [Security Considerations](#security-considerations)
12. [Performance Tuning](#performance-tuning)

---

## Overview

GL-005 CombustionControlAgent is a real-time AI-powered combustion control system designed for high-availability industrial applications. This deployment guide covers the complete infrastructure setup for Kubernetes-based production deployments.

### Key Features

- **Real-time Control:** <100ms control loop latency
- **High Availability:** 99.9% uptime SLA with 3-15 pod autoscaling
- **Multi-Protocol Support:** Modbus, OPC UA, MQTT integration
- **AI Optimization:** Claude AI-powered control optimization
- **Security Hardened:** Non-root containers, read-only filesystem, network policies

### Deployment Strategy

- **Infrastructure as Code:** Kustomize-based Kubernetes manifests
- **Multi-Environment:** Dev, Staging, Production overlays
- **Zero-Downtime:** Rolling updates with PodDisruptionBudget
- **Auto-Scaling:** HorizontalPodAutoscaler with custom metrics

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Ingress (NGINX)                                     │  │
│  │  - TLS Termination                                   │  │
│  │  - Rate Limiting (100 req/min)                       │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │                                      │
│  ┌───────────────────▼──────────────────────────────────┐  │
│  │  Service (ClusterIP)                                 │  │
│  │  - gl-005-combustion-control:80                      │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │                                      │
│  ┌───────────────────▼──────────────────────────────────┐  │
│  │  Deployment (GL-005 Pods)                            │  │
│  │  - Replicas: 3-15 (HPA managed)                      │  │
│  │  - Resources: 1-2 CPU, 1-2Gi memory                  │  │
│  │  - Health Checks: Liveness, Readiness, Startup       │  │
│  └──────┬────────┬────────┬────────────────────────────┘  │
│         │        │        │                                │
│  ┌──────▼──┐ ┌──▼──────┐ ┌▼──────────┐                   │
│  │ Pod 1   │ │ Pod 2   │ │ Pod 3     │  ... (scaled)      │
│  │ GL-005  │ │ GL-005  │ │ GL-005    │                   │
│  └─────────┘ └─────────┘ └───────────┘                   │
│       │          │            │                            │
│  ┌────▼──────────▼────────────▼───────────────────────┐  │
│  │  ConfigMap + Secrets                                │  │
│  │  - Application config                               │  │
│  │  - Database/Redis credentials                       │  │
│  │  - API keys (Anthropic, OpenAI)                     │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────▼────┐          ┌───▼────┐          ┌────▼─────┐
    │PostgreSQL│          │ Redis  │          │Prometheus│
    │ Database │          │ Cache  │          │ Metrics  │
    └──────────┘          └────────┘          └──────────┘
```

### Component Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Container Runtime | Docker | Application containerization |
| Orchestration | Kubernetes | Container orchestration |
| Ingress | NGINX | External traffic routing |
| Configuration | Kustomize | Environment-specific configs |
| Monitoring | Prometheus + Grafana | Metrics & visualization |
| Logging | ELK Stack / Loki | Centralized logging |
| Database | PostgreSQL | Persistent data storage |
| Cache | Redis | High-speed caching |
| Secrets | Kubernetes Secrets / Vault | Credentials management |

---

## Prerequisites

### Required Tools

Install the following tools on your deployment machine:

```bash
# kubectl (Kubernetes CLI)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# kustomize (Kubernetes config management)
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/

# Docker (container runtime)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Optional: Helm (Kubernetes package manager)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Kubernetes Cluster Requirements

- **Kubernetes Version:** 1.24+
- **Node Count:** 3+ (for high availability)
- **Node Resources:** 4 CPU, 8Gi memory per node
- **Storage:** Dynamic provisioning (StorageClass)
- **Network:** CNI plugin (Calico, Flannel, etc.)

### Access Requirements

- **Kubernetes Cluster:** Admin access via kubeconfig
- **Container Registry:** Push access to GCR/ECR/DockerHub
- **DNS:** Ability to configure DNS records
- **TLS Certificates:** cert-manager or manual cert provisioning

---

## Quick Start

### 1. Clone Repository

```bash
cd /path/to/GreenLang_2030/agent_foundation/agents/GL-005
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.template .env

# Edit with your values
nano .env
```

### 3. Validate Deployment

```bash
cd deployment/scripts
./validate.sh dev
```

### 4. Deploy to Dev Environment

```bash
./deploy.sh dev
```

### 5. Verify Deployment

```bash
# Check pods
kubectl get pods -n greenlang-dev -l app=gl-005-combustion-control

# Check logs
kubectl logs -n greenlang-dev -l app=gl-005-combustion-control -f

# Port-forward to test
kubectl port-forward -n greenlang-dev svc/gl-005-combustion-control 8000:80

# Test health endpoint
curl http://localhost:8000/api/v1/health
```

---

## Environment Configuration

### Development Environment

**Purpose:** Local development and testing
**Namespace:** `greenlang-dev`
**Domain:** `gl-005-dev.greenlang.io`

**Configuration:**
- **Replicas:** 1
- **Resources:** 250m CPU, 256Mi memory
- **Mock Mode:** Enabled (hardware/sensors)
- **Debug:** Enabled
- **Autoscaling:** Minimal (1-2 replicas)

**Deploy:**
```bash
./deploy.sh dev
```

### Staging Environment

**Purpose:** Pre-production testing
**Namespace:** `greenlang-staging`
**Domain:** `gl-005-staging.greenlang.io`

**Configuration:**
- **Replicas:** 2
- **Resources:** 500m CPU, 512Mi memory
- **Mock Mode:** Disabled
- **Debug:** Disabled
- **Autoscaling:** Moderate (2-6 replicas)

**Deploy:**
```bash
./deploy.sh staging
```

### Production Environment

**Purpose:** Live production system
**Namespace:** `greenlang`
**Domain:** `gl-005.greenlang.io`

**Configuration:**
- **Replicas:** 3 (minimum)
- **Resources:** 1 CPU, 1Gi memory
- **Mock Mode:** Disabled
- **Debug:** Disabled
- **Autoscaling:** Full (3-15 replicas)
- **Safety Checks:** All enabled

**Deploy:**
```bash
./deploy.sh production
```

---

## Deployment Procedures

### Standard Deployment

1. **Validate Configuration**
   ```bash
   cd deployment/scripts
   ./validate.sh production
   ```

2. **Build Docker Image**
   ```bash
   cd ../..
   docker build -f Dockerfile \
     --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
     --build-arg VCS_REF=$(git rev-parse --short HEAD) \
     --build-arg VERSION=1.0.0 \
     -t gcr.io/greenlang/gl-005-combustion-control:1.0.0 .
   ```

3. **Push Image to Registry**
   ```bash
   docker push gcr.io/greenlang/gl-005-combustion-control:1.0.0
   ```

4. **Deploy to Kubernetes**
   ```bash
   cd deployment/scripts
   ./deploy.sh production
   ```

5. **Monitor Rollout**
   ```bash
   kubectl rollout status deployment/gl-005-combustion-control -n greenlang --timeout=5m
   ```

6. **Verify Deployment**
   ```bash
   kubectl get pods -n greenlang -l app=gl-005-combustion-control
   kubectl logs -n greenlang -l app=gl-005-combustion-control -f
   ```

### Manual Deployment (Using Kustomize)

```bash
# Build manifests
kustomize build deployment/kustomize/overlays/production > /tmp/gl-005-prod.yaml

# Review manifests
cat /tmp/gl-005-prod.yaml

# Apply to cluster
kubectl apply -f /tmp/gl-005-prod.yaml

# Watch rollout
kubectl rollout status deployment/gl-005-combustion-control -n greenlang
```

### Dry-Run Deployment

Test deployment without applying changes:

```bash
./deploy.sh production true  # true = dry-run mode
```

---

## Scaling & High Availability

### Horizontal Pod Autoscaler (HPA)

GL-005 uses HPA to automatically scale based on CPU and memory utilization.

**Configuration:**
- **Min Replicas:** 3 (production), 2 (staging), 1 (dev)
- **Max Replicas:** 15 (production), 6 (staging), 2 (dev)
- **Metrics:**
  - CPU: 70% average utilization
  - Memory: 80% average utilization

**Check HPA Status:**
```bash
kubectl get hpa -n greenlang
kubectl describe hpa gl-005-combustion-control-hpa -n greenlang
```

**Manual Scaling:**
```bash
# Scale to specific replica count (overrides HPA temporarily)
kubectl scale deployment gl-005-combustion-control -n greenlang --replicas=5
```

### PodDisruptionBudget (PDB)

Ensures minimum availability during voluntary disruptions (node drains, upgrades).

**Configuration:**
- **Min Available:** 2 pods (ensures 66% availability with 3 replicas)

**Check PDB Status:**
```bash
kubectl get pdb -n greenlang
kubectl describe pdb gl-005-combustion-control-pdb -n greenlang
```

### Resource Quotas

Namespace-level resource limits to prevent exhaustion.

**Production Limits:**
- **CPU:** 32 cores (requests), 64 cores (limits)
- **Memory:** 64Gi (requests), 128Gi (limits)
- **Pods:** 50 maximum

**Check Quota:**
```bash
kubectl get resourcequota -n greenlang
kubectl describe resourcequota gl-005-resource-quota -n greenlang
```

---

## Monitoring & Observability

### Prometheus Metrics

GL-005 exposes Prometheus metrics on port 8001 at `/metrics`.

**Key Metrics:**
- `gl_005_control_loop_latency_seconds` - Control loop execution time
- `gl_005_control_loop_failures_total` - Control loop failure count
- `gl_005_combustion_efficiency_percent` - Current efficiency
- `gl_005_sensor_read_errors_total` - Sensor communication errors
- `http_request_duration_seconds` - HTTP request latency

**Access Metrics:**
```bash
# Port-forward to metrics endpoint
kubectl port-forward -n greenlang svc/gl-005-metrics 8001:8001

# Curl metrics
curl http://localhost:8001/metrics
```

### Grafana Dashboards

Pre-built Grafana dashboards are available in `monitoring/grafana/`.

**Import Dashboards:**
1. Access Grafana UI
2. Import dashboard JSON from `monitoring/grafana/`
3. Select Prometheus data source

**Dashboards:**
- **Executive Dashboard:** High-level KPIs
- **Operations Dashboard:** System health and performance
- **Agent Dashboard:** Control loop metrics
- **Quality Dashboard:** Data quality metrics

### Prometheus Alerts

Critical alerts are defined in `deployment/servicemonitor.yaml`.

**Alert Rules:**
- **GL005Down:** Service is unreachable (critical)
- **GL005HighLatency:** P95 latency > 100ms (critical)
- **GL005ControlLoopFailure:** Control loop failures detected (critical)
- **GL005HighCPU:** CPU usage > 150% (warning)
- **GL005HighMemory:** Memory usage > 85% (warning)

**Check Alerts:**
```bash
# View active alerts
kubectl get prometheusrules -n greenlang
kubectl describe prometheusrule gl-005-alerts -n greenlang
```

### Logging

Application logs are collected via:
- **stdout/stderr:** Captured by Kubernetes
- **ELK Stack / Loki:** Centralized log aggregation

**View Logs:**
```bash
# Recent logs
kubectl logs -n greenlang -l app=gl-005-combustion-control --tail=100

# Follow logs
kubectl logs -n greenlang -l app=gl-005-combustion-control -f

# Logs from specific pod
kubectl logs -n greenlang <pod-name> -f

# Previous crashed pod logs
kubectl logs -n greenlang <pod-name> --previous
```

---

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

**Symptoms:** Pods stuck in `Pending`, `CrashLoopBackOff`, or `ImagePullBackOff`

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n greenlang
kubectl logs <pod-name> -n greenlang
kubectl get events -n greenlang --sort-by='.lastTimestamp'
```

**Solutions:**
- **ImagePullBackOff:** Check image registry credentials, image tag exists
- **CrashLoopBackOff:** Check logs for application errors, env variables
- **Pending:** Check resource quotas, node capacity, PVC availability

#### 2. High Latency / Slow Performance

**Symptoms:** Control loop latency > 100ms, slow API responses

**Diagnosis:**
```bash
# Check resource usage
kubectl top pods -n greenlang -l app=gl-005-combustion-control

# Check HPA status
kubectl get hpa -n greenlang

# Check metrics
curl http://localhost:8001/metrics | grep latency
```

**Solutions:**
- Scale up replicas manually: `kubectl scale deployment gl-005-combustion-control --replicas=10 -n greenlang`
- Increase resource limits in deployment
- Check database/Redis performance
- Review network policies for bottlenecks

#### 3. Database Connection Issues

**Symptoms:** Errors connecting to PostgreSQL

**Diagnosis:**
```bash
kubectl logs <pod-name> -n greenlang | grep -i database
kubectl exec -it <pod-name> -n greenlang -- env | grep DATABASE
```

**Solutions:**
- Verify DATABASE_URL secret is correct
- Check PostgreSQL service is running: `kubectl get svc postgresql -n greenlang`
- Test connectivity: `kubectl exec -it <pod-name> -n greenlang -- curl postgres:5432`

#### 4. Memory Leaks / OOMKilled

**Symptoms:** Pods restarting with OOMKilled status

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n greenlang | grep -A 5 "Last State"
kubectl top pod <pod-name> -n greenlang
```

**Solutions:**
- Increase memory limits: Edit `deployment/kustomize/overlays/production/patches/resource-patch.yaml`
- Review application memory usage patterns
- Enable memory profiling to identify leaks

---

## Rollback Procedures

### Emergency Rollback

If deployment fails or causes issues, rollback to the previous version:

```bash
cd deployment/scripts
./rollback.sh production
```

### Rollback to Specific Revision

```bash
# View rollout history
kubectl rollout history deployment/gl-005-combustion-control -n greenlang

# Rollback to specific revision
./rollback.sh production 3  # Rollback to revision 3
```

### Manual Rollback

```bash
# Undo last rollout
kubectl rollout undo deployment/gl-005-combustion-control -n greenlang

# Rollback to specific revision
kubectl rollout undo deployment/gl-005-combustion-control -n greenlang --to-revision=2
```

### Post-Rollback Verification

```bash
# Check pod status
kubectl get pods -n greenlang -l app=gl-005-combustion-control

# Check logs
kubectl logs -n greenlang -l app=gl-005-combustion-control -f

# Test health endpoint
kubectl exec -it <pod-name> -n greenlang -- curl http://localhost:8000/api/v1/health
```

---

## Security Considerations

### Container Security

- **Non-Root User:** Containers run as UID 1000 (greenlang)
- **Read-Only Filesystem:** Root filesystem is read-only
- **No Privilege Escalation:** `allowPrivilegeEscalation: false`
- **Dropped Capabilities:** All Linux capabilities dropped

### Network Security

- **Network Policies:** Restrict ingress/egress traffic
- **TLS/SSL:** All external traffic encrypted (HTTPS)
- **Rate Limiting:** 100 requests/minute per IP

### Secrets Management

**Best Practices:**
- Use external secret managers (HashiCorp Vault, AWS Secrets Manager)
- Never commit secrets to Git
- Rotate secrets regularly (every 90 days)
- Use RBAC to restrict secret access

**Using External Secrets Operator:**
```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets -n external-secrets-system --create-namespace

# Configure SecretStore (example: AWS Secrets Manager)
kubectl apply -f deployment/external-secrets/
```

### RBAC (Role-Based Access Control)

GL-005 uses a dedicated ServiceAccount with minimal permissions:
- Read-only access to ConfigMaps and Secrets
- Read access to Pods and Services

**Review RBAC:**
```bash
kubectl describe serviceaccount gl-005-service-account -n greenlang
kubectl describe role gl-005-role -n greenlang
```

---

## Performance Tuning

### Resource Optimization

**Recommended Production Resources:**
- **CPU Request:** 1000m (1 core)
- **CPU Limit:** 2000m (2 cores)
- **Memory Request:** 1Gi
- **Memory Limit:** 2Gi

**Adjust Resources:**
Edit `deployment/kustomize/overlays/production/patches/resource-patch.yaml`

### JVM Tuning (if applicable)

For Python applications using Jython or Java components:
```bash
JAVA_OPTS="-Xms1g -Xmx2g -XX:+UseG1GC -XX:MaxGCPauseMillis=100"
```

### Database Connection Pooling

Configure in `.env`:
```
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30
```

### Redis Optimization

```
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_TIMEOUT=5
REDIS_RETRY_ON_TIMEOUT=true
```

---

## Appendix

### File Structure

```
GL-005/
├── deployment/
│   ├── kustomize/
│   │   ├── base/
│   │   │   └── kustomization.yaml
│   │   └── overlays/
│   │       ├── dev/
│   │       ├── staging/
│   │       └── production/
│   ├── scripts/
│   │   ├── deploy.sh
│   │   ├── rollback.sh
│   │   └── validate.sh
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   ├── networkpolicy.yaml
│   ├── servicemonitor.yaml
│   ├── resourcequota.yaml
│   ├── limitrange.yaml
│   └── DEPLOYMENT_GUIDE.md (this file)
├── Dockerfile
├── requirements.txt
├── .env.template
└── ...
```

### Useful Commands

```bash
# Get all resources
kubectl get all -n greenlang -l app=gl-005-combustion-control

# Describe deployment
kubectl describe deployment gl-005-combustion-control -n greenlang

# View events
kubectl get events -n greenlang --sort-by='.lastTimestamp'

# Port-forward to service
kubectl port-forward -n greenlang svc/gl-005-combustion-control 8000:80

# Execute shell in pod
kubectl exec -it <pod-name> -n greenlang -- /bin/bash

# View resource usage
kubectl top pods -n greenlang -l app=gl-005-combustion-control
kubectl top nodes

# Dump all manifests
kubectl get all,configmap,secret,ingress,hpa,pdb -n greenlang -l app=gl-005-combustion-control -o yaml
```

### Support & Contact

- **Documentation:** https://docs.greenlang.ai/agents/GL-005
- **Issues:** https://github.com/greenlang/Code-V1_GreenLang/issues
- **Slack:** #gl-005-support
- **Email:** support@greenlang.io

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-18
**Maintained By:** GreenLang DevOps Team
