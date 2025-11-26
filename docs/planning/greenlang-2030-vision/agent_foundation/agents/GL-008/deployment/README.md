# GL-008 SteamTrapInspector - Kubernetes Deployment

This directory contains the complete Kubernetes deployment infrastructure for GL-008 SteamTrapInspector, an acoustic and thermal inspection system for steam trap monitoring.

## Directory Structure

```
deployment/
├── README.md                    # This file
├── deployment.yaml              # Main deployment manifest (151 lines)
├── service.yaml                 # Service definitions (81 lines)
├── configmap.yaml               # Configuration data (118 lines)
├── secret.yaml                  # External secrets template (66 lines)
├── hpa.yaml                     # Horizontal Pod Autoscaler (62 lines)
├── pdb.yaml                     # Pod Disruption Budget (21 lines)
├── servicemonitor.yaml          # Prometheus monitoring (45 lines)
├── networkpolicy.yaml           # Network policies (112 lines)
├── ingress.yaml                 # NGINX ingress (42 lines)
├── serviceaccount.yaml          # RBAC configuration (58 lines)
├── pvc.yaml                     # Persistent volume claims (44 lines)
└── kustomize/
    ├── base/
    │   └── kustomization.yaml   # Base kustomize config
    └── overlays/
        ├── dev/                 # Development environment
        │   ├── kustomization.yaml
        │   └── patches/
        ├── staging/             # Staging environment
        │   ├── kustomization.yaml
        │   └── patches/
        └── production/          # Production environment
            ├── kustomization.yaml
            └── patches/
```

## Architecture

### Deployment Specifications

- **Replicas**: 3 (base), HPA scaling: 3-10 replicas
- **Resources**:
  - Requests: CPU 1 core, Memory 512Mi
  - Limits: CPU 4 cores, Memory 2Gi
- **Ports**:
  - Application: 9090
  - Metrics: 9091
- **Security**: Non-root user (UID 1000), read-only root filesystem

### Key Features

1. **High Availability**
   - Pod anti-affinity for node distribution
   - Pod Disruption Budget (minAvailable: 2)
   - Zero-downtime rolling updates

2. **Auto-Scaling**
   - HPA based on CPU (70%) and memory (80%)
   - Scale-up: 2 pods per 30s
   - Scale-down: 1 pod per 60s with 5-min cooldown

3. **Health Checks**
   - Liveness probe: /api/v1/health
   - Readiness probe: /api/v1/ready
   - Startup probe: 180s max (ML model loading)

4. **Security**
   - Non-root user execution
   - Read-only root filesystem
   - Network policies (ingress/egress)
   - External Secrets Operator integration

5. **Monitoring**
   - Prometheus ServiceMonitor (15s scrape interval)
   - Custom metrics on port 9091
   - Grafana dashboard integration

6. **Storage**
   - ML models: 10Gi ReadOnlyMany PVC
   - Application data: 20Gi ReadWriteOnce PVC
   - Ephemeral volumes for logs, cache, temp data

## Deployment Instructions

### Prerequisites

1. Kubernetes cluster (v1.24+)
2. kubectl configured
3. kustomize installed
4. External Secrets Operator installed
5. NGINX Ingress Controller
6. Prometheus Operator (for ServiceMonitor)
7. cert-manager (for TLS certificates)

### Deploy to Development

```bash
# Navigate to deployment directory
cd C:\Users\aksha\Code-V1_GreenLang\docs\planning\greenlang-2030-vision\agent_foundation\agents\GL-008\deployment

# Create namespace
kubectl create namespace greenlang-dev

# Deploy using kustomize
kubectl apply -k kustomize/overlays/dev/

# Verify deployment
kubectl get pods -n greenlang-dev -l agent=GL-008
kubectl get svc -n greenlang-dev
kubectl get ingress -n greenlang-dev
```

### Deploy to Staging

```bash
# Create namespace
kubectl create namespace greenlang-staging

# Deploy using kustomize
kubectl apply -k kustomize/overlays/staging/

# Verify deployment
kubectl get pods -n greenlang-staging -l agent=GL-008
kubectl rollout status deployment/staging-gl-008-steam-trap-inspector -n greenlang-staging
```

### Deploy to Production

```bash
# Create namespace
kubectl create namespace greenlang

# Deploy using kustomize
kubectl apply -k kustomize/overlays/production/

# Verify deployment
kubectl get pods -n greenlang -l agent=GL-008
kubectl rollout status deployment/prod-gl-008-steam-trap-inspector -n greenlang

# Check HPA
kubectl get hpa -n greenlang

# Check PDB
kubectl get pdb -n greenlang
```

## Environment Configuration

### Development
- **Replicas**: 1
- **Resources**: CPU 500m-2000m, Memory 256Mi-1Gi
- **HPA**: 1-3 replicas, 80% CPU target
- **Domain**: gl-008-dev.greenlang.io
- **Log Level**: DEBUG

### Staging
- **Replicas**: 2
- **Resources**: CPU 1000m-3000m, Memory 512Mi-1Gi
- **HPA**: 2-6 replicas, 75% CPU target
- **Domain**: gl-008-staging.greenlang.io
- **Log Level**: INFO

### Production
- **Replicas**: 3
- **Resources**: CPU 1000m-4000m, Memory 512Mi-2Gi
- **HPA**: 3-10 replicas, 70% CPU/80% memory target
- **Domain**: gl-008.greenlang.io
- **Log Level**: INFO

## Secrets Management

This deployment uses External Secrets Operator to sync secrets from AWS Secrets Manager.

### Required Secrets

Create the following secrets in AWS Secrets Manager:

```bash
# Database credentials
greenlang/gl-008/database
{
  "url": "postgresql://user:pass@host:5432/greenlang_gl008",
  "username": "gl008_user",
  "password": "secure_password"
}

# Redis credentials
greenlang/gl-008/redis
{
  "url": "redis://host:6379/0"
}

# AI API keys
greenlang/gl-008/ai
{
  "anthropic_api_key": "sk-ant-..."
}

# AWS credentials
greenlang/gl-008/aws
{
  "access_key_id": "AKIA...",
  "secret_access_key": "..."
}

# Alerting credentials
greenlang/gl-008/alerting
{
  "slack_webhook": "https://hooks.slack.com/...",
  "pagerduty_api_key": "..."
}

# Email SMTP
greenlang/gl-008/email
{
  "smtp_password": "..."
}
```

## Monitoring

### Prometheus Metrics

The application exposes metrics on port 9091 at `/metrics`:

- `gl008_inspections_total` - Total inspections performed
- `gl008_failures_detected` - Steam trap failures detected
- `gl008_acoustic_analysis_duration_seconds` - Acoustic analysis duration
- `gl008_thermal_analysis_duration_seconds` - Thermal analysis duration
- `gl008_ml_inference_duration_seconds` - ML inference duration
- `gl008_energy_cost_savings_usd` - Energy cost savings calculated

### Grafana Dashboards

Import the Grafana dashboard from:
```
C:\Users\aksha\Code-V1_GreenLang\docs\planning\greenlang-2030-vision\agent_foundation\agents\GL-008\monitoring\grafana-dashboard.json
```

### Alerts

Alerting rules are defined in:
```
C:\Users\aksha\Code-V1_GreenLang\docs\planning\greenlang-2030-vision\agent_foundation\agents\GL-008\monitoring\prometheus-rules.yaml
```

## Troubleshooting

### Pod Fails to Start

```bash
# Check pod status
kubectl get pods -n greenlang -l agent=GL-008

# View pod logs
kubectl logs -n greenlang -l agent=GL-008 --tail=100

# Describe pod for events
kubectl describe pod -n greenlang <pod-name>

# Check init containers
kubectl logs -n greenlang <pod-name> -c database-ready
kubectl logs -n greenlang <pod-name> -c redis-ready
kubectl logs -n greenlang <pod-name> -c ml-models-loader
```

### Database Connection Issues

```bash
# Check database connectivity from pod
kubectl exec -it -n greenlang <pod-name> -- sh
# Inside pod:
# nc -zv postgresql-service 5432
```

### HPA Not Scaling

```bash
# Check HPA status
kubectl get hpa -n greenlang

# Describe HPA
kubectl describe hpa gl-008-steam-trap-inspector-hpa -n greenlang

# Check metrics-server
kubectl get apiservice v1beta1.metrics.k8s.io
```

### Secrets Not Loading

```bash
# Check External Secret
kubectl get externalsecret -n greenlang

# Check generated Secret
kubectl get secret gl-008-secrets -n greenlang

# Describe External Secret for errors
kubectl describe externalsecret gl-008-secrets -n greenlang
```

## Rollback

```bash
# View rollout history
kubectl rollout history deployment/gl-008-steam-trap-inspector -n greenlang

# Rollback to previous version
kubectl rollout undo deployment/gl-008-steam-trap-inspector -n greenlang

# Rollback to specific revision
kubectl rollout undo deployment/gl-008-steam-trap-inspector -n greenlang --to-revision=2
```

## Scaling

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment gl-008-steam-trap-inspector -n greenlang --replicas=5

# Verify
kubectl get deployment gl-008-steam-trap-inspector -n greenlang
```

### Disable Auto-Scaling

```bash
# Delete HPA to disable auto-scaling
kubectl delete hpa gl-008-steam-trap-inspector-hpa -n greenlang
```

## Network Policies

Network policies restrict traffic to:

**Ingress**:
- NGINX ingress controller (port 9090)
- Prometheus monitoring (port 9091)
- Same namespace pods (inter-pod)

**Egress**:
- DNS resolution (UDP 53)
- PostgreSQL (TCP 5432)
- Redis (TCP 6379)
- HTTPS external APIs (TCP 443)

## Maintenance

### Update Application

```bash
# Update image in kustomize overlay
cd kustomize/overlays/production
# Edit kustomization.yaml to update image tag

# Apply changes
kubectl apply -k .

# Watch rollout
kubectl rollout status deployment/prod-gl-008-steam-trap-inspector -n greenlang
```

### Update ConfigMap

```bash
# Edit ConfigMap
kubectl edit configmap gl-008-config -n greenlang

# Restart pods to pick up changes
kubectl rollout restart deployment/gl-008-steam-trap-inspector -n greenlang
```

### Backup

```bash
# Backup all resources
kubectl get all,configmap,secret,pvc,ingress -n greenlang -l agent=GL-008 -o yaml > gl-008-backup.yaml
```

## Resource Limits

| Resource | Dev | Staging | Production |
|----------|-----|---------|------------|
| CPU Request | 500m | 1000m | 1000m |
| CPU Limit | 2000m | 3000m | 4000m |
| Memory Request | 256Mi | 512Mi | 512Mi |
| Memory Limit | 1Gi | 1Gi | 2Gi |
| Min Replicas | 1 | 2 | 3 |
| Max Replicas | 3 | 6 | 10 |

## Support

For issues or questions:
- DevOps Team: devops@greenlang.io
- Documentation: https://greenlang.io/docs/agents/GL-008
- Runbooks: C:\Users\aksha\Code-V1_GreenLang\docs\planning\greenlang-2030-vision\agent_foundation\agents\GL-008\runbooks
