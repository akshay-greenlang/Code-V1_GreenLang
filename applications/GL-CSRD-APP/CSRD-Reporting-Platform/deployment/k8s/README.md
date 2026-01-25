# Kubernetes Deployment Manifests

This directory contains production-ready Kubernetes manifests for deploying the CSRD/ESRS Digital Reporting Platform.

## 📁 Files Overview

| File | Purpose | Required |
|------|---------|----------|
| `namespace.yaml` | Creates production and staging namespaces | ✅ Yes |
| `configmap.yaml` | Application configuration (non-sensitive) | ✅ Yes |
| `secrets.yaml` | Sensitive credentials and API keys | ✅ Yes |
| `statefulset.yaml` | PostgreSQL, Redis, Weaviate databases | ✅ Yes |
| `service.yaml` | Kubernetes services for networking | ✅ Yes |
| `deployment.yaml` | Main application deployment with HPA & PDB | ✅ Yes |
| `hpa.yaml` | Horizontal Pod Autoscaler configuration | ⚠️ Recommended |
| `ingress.yaml` | External HTTPS access and routing | ⚠️ Recommended |
| `APPLY_ORDER.md` | Step-by-step deployment instructions | 📖 Documentation |

## 🚀 Quick Deploy

```bash
# 1. Create namespace
kubectl apply -f namespace.yaml

# 2. Configure secrets (IMPORTANT: Edit first!)
cp secrets.yaml.example secrets.yaml
# Edit secrets.yaml with actual credentials
kubectl apply -f secrets.yaml

# 3. Apply all manifests
kubectl apply -f configmap.yaml
kubectl apply -f statefulset.yaml
kubectl apply -f service.yaml
kubectl apply -f deployment.yaml
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml

# 4. Verify deployment
kubectl get all -n production
```

## 📋 Prerequisites

### Required
- Kubernetes cluster v1.24+
- kubectl configured
- Minimum 3 worker nodes (for HA)
- 16GB RAM per node
- 100GB storage available

### Recommended
- Ingress controller (nginx, traefik)
- cert-manager for TLS
- Metrics server for HPA
- Prometheus operator for monitoring
- Persistent storage class

### Install Prerequisites

```bash
# Metrics server (for HPA)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# NGINX ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml

# cert-manager (for TLS)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Prometheus operator (for monitoring)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
```

## ⚙️ Configuration

### 1. Secrets Configuration

**CRITICAL**: Never commit actual secrets to Git!

```bash
# Copy example
cp secrets.yaml secrets.yaml

# Edit with actual values
nano secrets.yaml

# Required secrets:
# - database-url
# - redis-url
# - weaviate-url
# - anthropic-api-key
# - encryption-key
# - secret-key
```

Generate secure keys:

```bash
# Secret key for JWT
openssl rand -base64 32

# Encryption key
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 2. ConfigMap Configuration

Edit `configmap.yaml` to adjust:
- Log levels
- Worker counts
- Feature flags
- Performance tuning
- Agent configurations

### 3. Resource Limits

Current settings per pod:
- **Requests**: 1 CPU, 2GB RAM
- **Limits**: 2 CPU, 4GB RAM

Adjust in `deployment.yaml` based on your workload:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### 4. Auto-Scaling Configuration

HPA settings in `hpa.yaml`:
- **Min replicas**: 3
- **Max replicas**: 20
- **CPU target**: 70%
- **Memory target**: 80%

Adjust based on traffic patterns.

### 5. Ingress Configuration

Edit `ingress.yaml` to set:
- Your domain name
- TLS certificate settings
- CORS origins
- Rate limits

Replace `csrd.yourdomain.com` with your actual domain.

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│          Ingress Controller             │
│       (TLS, Routing, Rate Limit)        │
└───────────────┬─────────────────────────┘
                │
    ┌───────────┴──────────┐
    │                      │
┌───▼──────────┐  ┌────────▼────────┐
│  CSRD API    │  │    Monitoring   │
│  (3-20 pods) │  │    (Grafana)    │
│  Auto-scaled │  └─────────────────┘
└───┬──────────┘
    │
    ├─────────────┬──────────────┬─────────────┐
    │             │              │             │
┌───▼──────┐ ┌───▼──────┐ ┌─────▼──────┐ ┌───▼──────┐
│PostgreSQL│ │  Redis   │ │  Weaviate  │ │   PVC    │
│  (1 pod) │ │ (1 pod)  │ │  (1 pod)   │ │ Storage  │
└──────────┘ └──────────┘ └────────────┘ └──────────┘
```

## 📊 Resource Requirements

### Minimum (Development/Staging)
- **Nodes**: 1
- **CPU**: 4 cores
- **Memory**: 8GB
- **Storage**: 50GB

### Recommended (Production)
- **Nodes**: 3+
- **CPU**: 8 cores per node
- **Memory**: 16GB per node
- **Storage**: 200GB SSD

### Per Component

| Component | CPU Request | Memory Request | Storage |
|-----------|-------------|----------------|---------|
| API Pod | 1000m | 2Gi | - |
| PostgreSQL | 1000m | 2Gi | 50Gi |
| Redis | 500m | 512Mi | 10Gi |
| Weaviate | 1000m | 2Gi | 20Gi |
| Prometheus | 500m | 2Gi | 50Gi |
| Grafana | 200m | 512Mi | 10Gi |

## 🔒 Security Best Practices

1. **Secrets Management**
   - Never commit secrets to Git
   - Use Kubernetes secrets or external vaults
   - Rotate credentials every 90 days
   - Use sealed-secrets or SOPS for GitOps

2. **Network Policies**
   - Restrict pod-to-pod communication
   - Limit egress traffic
   - Use namespace isolation

3. **RBAC**
   - Principle of least privilege
   - Service accounts per component
   - Regular access audits

4. **Pod Security**
   - Run as non-root user (UID 1000)
   - Read-only root filesystem where possible
   - Drop all capabilities
   - No privilege escalation

5. **Image Security**
   - Use official base images
   - Scan for vulnerabilities (Trivy, Snyk)
   - Pin image versions (no `latest`)
   - Sign images (Cosign)

## 📈 Monitoring & Alerts

### Metrics Exposed

- Application metrics at `/metrics`
- Prometheus format
- Custom business metrics

### Grafana Dashboards

Access dashboards at `https://monitoring.csrd.yourdomain.com`

Available dashboards:
1. Application Overview
2. Database Performance
3. Cache Hit Rates
4. Kubernetes Cluster
5. Business Metrics (Reports, Calculations)

### Recommended Alerts

```yaml
# High error rate
expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01

# High latency
expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1

# Pod restarts
expr: rate(kube_pod_container_status_restarts_total[15m]) > 0

# HPA at max
expr: kube_horizontalpodautoscaler_status_current_replicas == kube_horizontalpodautoscaler_spec_max_replicas
```

## 🔄 CI/CD Integration

### GitOps with ArgoCD

```bash
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Create application
kubectl apply -f - <<EOF
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: csrd-app
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/akshay-greenlang/Code-V1_GreenLang
    targetRevision: HEAD
    path: GL-CSRD-APP/CSRD-Reporting-Platform/deployment/k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
EOF
```

### GitHub Actions

See `.github/workflows/ci-cd.yml` for automated deployment pipeline.

## 🆘 Troubleshooting

### Check Deployment Status

```bash
# Overview
kubectl get all -n production

# Pods not running
kubectl get pods -n production
kubectl describe pod <pod-name> -n production
kubectl logs <pod-name> -n production

# Services have no endpoints
kubectl get endpoints -n production
kubectl describe service csrd-service -n production

# PVC not bound
kubectl get pvc -n production
kubectl describe pvc <pvc-name> -n production
```

### Common Issues

1. **ImagePullBackOff**: Check image name and registry credentials
2. **CrashLoopBackOff**: Check application logs and environment variables
3. **Pending Pods**: Check resource availability and scheduling constraints
4. **Service Unreachable**: Verify selectors match pod labels

### Debug Commands

```bash
# Exec into pod
kubectl exec -it <pod-name> -n production -- /bin/bash

# Port forward for local access
kubectl port-forward svc/csrd-service 8000:80 -n production

# Check events
kubectl get events --sort-by='.lastTimestamp' -n production

# Resource usage
kubectl top pods -n production
kubectl top nodes
```

## 📚 Additional Resources

- [APPLY_ORDER.md](APPLY_ORDER.md) - Step-by-step deployment guide
- [Main DEPLOYMENT.md](../../DEPLOYMENT.md) - Complete deployment documentation
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [CSRD Platform Documentation](../../README.md)

## 🤝 Support

For issues or questions:
- GitHub Issues: https://github.com/akshay-greenlang/Code-V1_GreenLang/issues
- Email: support@greenlang.com
- Documentation: https://greenlang.com/docs

---

**Production Ready** ✅
**Last Updated**: 2025-11-08
**Version**: 1.0.0
