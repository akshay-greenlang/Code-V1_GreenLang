# GL-CBAM-APP Deployment Guide

## Overview

This guide covers deploying the CBAM Importer Copilot application to Kubernetes using Kustomize for environment-specific configurations.

## Architecture

```
deployment/
├── kustomize/
│   ├── base/                    # Base configuration (shared)
│   │   ├── kustomization.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   ├── ingress.yaml
│   │   └── serviceaccount.yaml
│   └── overlays/               # Environment-specific overrides
│       ├── dev/
│       │   ├── kustomization.yaml
│       │   └── patches/
│       │       ├── replica-patch.yaml
│       │       ├── resource-patch.yaml
│       │       └── ingress-patch.yaml
│       ├── staging/
│       │   ├── kustomization.yaml
│       │   └── patches/
│       │       ├── replica-patch.yaml
│       │       ├── resource-patch.yaml
│       │       └── ingress-patch.yaml
│       └── production/
│           ├── kustomization.yaml
│           └── patches/
│               ├── replica-patch.yaml
│               ├── resource-patch.yaml
│               └── ingress-patch.yaml
├── hpa.yaml                    # Horizontal Pod Autoscaler
├── pdb.yaml                    # Pod Disruption Budget
├── resourcequota.yaml          # Namespace Resource Quota
└── limitrange.yaml             # Default Resource Limits

```

## Environment Configurations

| Environment | Replicas | CPU      | Memory | Hostname                    | Image Tag       |
|-------------|----------|----------|--------|-----------------------------|-----------------|
| Dev         | 3        | 1 core   | 1 GB   | cbam-dev.greenlang.io       | dev-latest      |
| Staging     | 5        | 2 cores  | 2 GB   | cbam-staging.greenlang.io   | staging-latest  |
| Production  | 3-15 (HPA)| 1 core  | 1 GB   | cbam.greenlang.io           | v1.0.0          |

## Prerequisites

1. **Kubernetes Cluster** (v1.24+)
   - kubectl configured with cluster access
   - Sufficient resources (see Resource Requirements)

2. **Required Add-ons**
   - NGINX Ingress Controller
   - cert-manager (for TLS certificates)
   - Metrics Server (for HPA)
   - Prometheus Adapter (optional, for custom metrics)

3. **Install Tools**
   ```bash
   # Install kubectl
   # https://kubernetes.io/docs/tasks/tools/

   # Install kustomize (optional, kubectl has built-in support)
   curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
   ```

## Quick Start

### Development Environment

```bash
# Create namespace
kubectl create namespace gl-cbam-dev

# Create secrets (update with your values)
kubectl create secret generic cbam-api-secrets \
  --from-literal=DATABASE_URL="postgresql://user:pass@cbam-postgres:5432/cbam" \
  --from-literal=REDIS_URL="redis://cbam-redis:6379/0" \
  --from-literal=JWT_SECRET="your-secret-key" \
  -n gl-cbam-dev

# Deploy using Kustomize
kubectl apply -k deployment/kustomize/overlays/dev

# Verify deployment
kubectl get all -n gl-cbam-dev
kubectl get pods -n gl-cbam-dev -w
```

### Staging Environment

```bash
# Create namespace
kubectl create namespace gl-cbam-staging

# Create secrets
kubectl create secret generic cbam-api-secrets \
  --from-literal=DATABASE_URL="postgresql://user:pass@cbam-postgres:5432/cbam" \
  --from-literal=REDIS_URL="redis://cbam-redis:6379/0" \
  --from-literal=JWT_SECRET="your-secret-key" \
  -n gl-cbam-staging

# Deploy
kubectl apply -k deployment/kustomize/overlays/staging

# Verify
kubectl get all -n gl-cbam-staging
```

### Production Environment

```bash
# Create namespace
kubectl create namespace gl-cbam

# Create secrets (use secure secret management)
kubectl create secret generic cbam-api-secrets \
  --from-literal=DATABASE_URL="postgresql://user:pass@cbam-postgres:5432/cbam" \
  --from-literal=REDIS_URL="redis://cbam-redis:6379/0" \
  --from-literal=JWT_SECRET="your-secret-key" \
  -n gl-cbam

# Apply resource governance
kubectl apply -f deployment/resourcequota.yaml
kubectl apply -f deployment/limitrange.yaml
kubectl apply -f deployment/pdb.yaml
kubectl apply -f deployment/hpa.yaml

# Deploy application
kubectl apply -k deployment/kustomize/overlays/production

# Verify
kubectl get all -n gl-cbam
kubectl get hpa -n gl-cbam
kubectl get pdb -n gl-cbam
```

## Validation

### Preview Manifests

```bash
# Development
kubectl kustomize deployment/kustomize/overlays/dev

# Staging
kubectl kustomize deployment/kustomize/overlays/staging

# Production
kubectl kustomize deployment/kustomize/overlays/production
```

### Dry Run

```bash
# Test without applying
kubectl apply -k deployment/kustomize/overlays/production --dry-run=client

# Server-side validation
kubectl apply -k deployment/kustomize/overlays/production --dry-run=server
```

### Verify Resources

```bash
# Check deployment
kubectl get deployment cbam-api -n gl-cbam
kubectl describe deployment cbam-api -n gl-cbam

# Check pods
kubectl get pods -n gl-cbam -l component=backend
kubectl describe pod <pod-name> -n gl-cbam

# Check HPA
kubectl get hpa cbam-api-hpa -n gl-cbam
kubectl describe hpa cbam-api-hpa -n gl-cbam

# Check PDB
kubectl get pdb cbam-api-pdb -n gl-cbam

# Check resource quota
kubectl describe resourcequota cbam-resource-quota -n gl-cbam

# Check ingress
kubectl get ingress -n gl-cbam
```

## Monitoring

### View Logs

```bash
# All pods
kubectl logs -f deployment/cbam-api -n gl-cbam

# Specific pod
kubectl logs -f <pod-name> -n gl-cbam

# Previous pod instance
kubectl logs <pod-name> -n gl-cbam --previous
```

### Check Health

```bash
# Health endpoint
kubectl port-forward svc/cbam-api 8000:8000 -n gl-cbam
curl http://localhost:8000/health

# Readiness endpoint
curl http://localhost:8000/health/ready

# Metrics endpoint
curl http://localhost:8000/metrics
```

### Watch HPA

```bash
# Real-time HPA status
kubectl get hpa cbam-api-hpa -n gl-cbam --watch

# View scaling events
kubectl get events -n gl-cbam --sort-by='.lastTimestamp' | grep HorizontalPodAutoscaler
```

## Scaling

### Manual Scaling

```bash
# Scale to specific replica count
kubectl scale deployment cbam-api --replicas=5 -n gl-cbam

# Note: HPA will override manual scaling in production
```

### Load Testing

```bash
# Generate load to test HPA
kubectl run -i --tty load-generator --rm --image=busybox --restart=Never -n gl-cbam -- /bin/sh -c \
  "while sleep 0.01; do wget -q -O- http://cbam-api:8000/api/v1/health; done"

# Watch pods scale up
kubectl get pods -n gl-cbam -w
```

## Updates and Rollbacks

### Update Deployment

```bash
# Update image
kubectl set image deployment/cbam-api api=ghcr.io/greenlang/gl-cbam-app:v1.1.0 -n gl-cbam

# Or apply updated Kustomize
kubectl apply -k deployment/kustomize/overlays/production
```

### Rollout Status

```bash
# Watch rollout
kubectl rollout status deployment/cbam-api -n gl-cbam

# View rollout history
kubectl rollout history deployment/cbam-api -n gl-cbam
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/cbam-api -n gl-cbam

# Rollback to specific revision
kubectl rollout undo deployment/cbam-api --to-revision=2 -n gl-cbam
```

## Troubleshooting

### Pod Issues

```bash
# Describe pod (shows events)
kubectl describe pod <pod-name> -n gl-cbam

# Get pod logs
kubectl logs <pod-name> -n gl-cbam

# Execute into pod
kubectl exec -it <pod-name> -n gl-cbam -- /bin/sh

# Check resource usage
kubectl top pods -n gl-cbam
```

### Service Issues

```bash
# Check service endpoints
kubectl get endpoints cbam-api -n gl-cbam

# Test connectivity from within cluster
kubectl run -it --rm debug --image=busybox --restart=Never -n gl-cbam -- sh
wget -O- http://cbam-api:8000/health
```

### Ingress Issues

```bash
# Check ingress
kubectl describe ingress cbam-api-ingress -n gl-cbam

# View ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx -f

# Check certificate
kubectl get certificate -n gl-cbam
kubectl describe certificate cbam-api-tls-cert-prod -n gl-cbam
```

### HPA Issues

```bash
# Check HPA status
kubectl describe hpa cbam-api-hpa -n gl-cbam

# Verify metrics server
kubectl get apiservice v1beta1.metrics.k8s.io -o yaml

# Check custom metrics (if using)
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1
```

## Cleanup

### Remove Deployment

```bash
# Development
kubectl delete -k deployment/kustomize/overlays/dev
kubectl delete namespace gl-cbam-dev

# Staging
kubectl delete -k deployment/kustomize/overlays/staging
kubectl delete namespace gl-cbam-staging

# Production
kubectl delete -f deployment/hpa.yaml
kubectl delete -f deployment/pdb.yaml
kubectl delete -k deployment/kustomize/overlays/production
kubectl delete namespace gl-cbam
```

## Security Best Practices

1. **Secrets Management**
   - Use external secret management (Vault, AWS Secrets Manager)
   - Never commit secrets to Git
   - Rotate secrets regularly

2. **RBAC**
   - Create least-privilege service accounts
   - Use namespaces for isolation
   - Enable Pod Security Standards

3. **Network Policies**
   - Restrict pod-to-pod communication
   - Whitelist ingress sources
   - Enable encryption in transit

4. **Image Security**
   - Scan images for vulnerabilities
   - Use signed images
   - Pull from private registries

## Performance Tuning

### Resource Optimization

```bash
# View actual resource usage
kubectl top pods -n gl-cbam

# Adjust requests/limits based on usage
# Update deployment/kustomize/overlays/*/patches/resource-patch.yaml
```

### HPA Tuning

```bash
# Monitor scaling behavior
kubectl get hpa cbam-api-hpa -n gl-cbam --watch

# Adjust thresholds if needed
# Update deployment/hpa.yaml
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/greenlang/gl-cbam-app/issues
- Documentation: https://docs.greenlang.io
- Support: support@greenlang.io
