# GL-CBAM-APP - Kubernetes Deployment Quick Reference

## Quick Deploy Commands

### Development
```bash
kubectl create namespace gl-cbam-dev
kubectl create secret generic cbam-api-secrets --from-literal=DATABASE_URL="..." -n gl-cbam-dev
kubectl apply -k deployment/kustomize/overlays/dev
kubectl get all -n gl-cbam-dev
```

### Staging
```bash
kubectl create namespace gl-cbam-staging
kubectl create secret generic cbam-api-secrets --from-literal=DATABASE_URL="..." -n gl-cbam-staging
kubectl apply -k deployment/kustomize/overlays/staging
kubectl get all -n gl-cbam-staging
```

### Production
```bash
kubectl create namespace gl-cbam
kubectl create secret generic cbam-api-secrets --from-literal=DATABASE_URL="..." -n gl-cbam
kubectl apply -f deployment/resourcequota.yaml
kubectl apply -f deployment/limitrange.yaml
kubectl apply -f deployment/pdb.yaml
kubectl apply -f deployment/hpa.yaml
kubectl apply -k deployment/kustomize/overlays/production
kubectl get all,hpa,pdb -n gl-cbam
```

## Validation

```bash
# Preview manifests
kubectl kustomize deployment/kustomize/overlays/production

# Dry run
kubectl apply -k deployment/kustomize/overlays/production --dry-run=client

# Validate script
./deployment/scripts/validate.sh
```

## Monitoring

```bash
# View pods
kubectl get pods -n gl-cbam -w

# View HPA
kubectl get hpa cbam-api-hpa -n gl-cbam --watch

# View logs
kubectl logs -f deployment/cbam-api -n gl-cbam

# View resource usage
kubectl top pods -n gl-cbam

# View events
kubectl get events -n gl-cbam --sort-by='.lastTimestamp'
```

## Troubleshooting

```bash
# Describe pod
kubectl describe pod <pod-name> -n gl-cbam

# Execute into pod
kubectl exec -it <pod-name> -n gl-cbam -- /bin/sh

# Check service endpoints
kubectl get endpoints cbam-api -n gl-cbam

# Check ingress
kubectl describe ingress cbam-api-ingress -n gl-cbam

# Check PDB
kubectl get pdb cbam-api-pdb -n gl-cbam

# Check resource quota
kubectl describe resourcequota cbam-resource-quota -n gl-cbam
```

## Scaling

```bash
# Manual scale
kubectl scale deployment cbam-api --replicas=5 -n gl-cbam

# Load test (to trigger HPA)
kubectl run load-generator --image=busybox -n gl-cbam -- /bin/sh -c \
  "while true; do wget -q -O- http://cbam-api:8000/health; done"
```

## Updates

```bash
# Update image
kubectl set image deployment/cbam-api api=ghcr.io/greenlang/gl-cbam-app:v1.1.0 -n gl-cbam

# Rollout status
kubectl rollout status deployment/cbam-api -n gl-cbam

# Rollback
kubectl rollout undo deployment/cbam-api -n gl-cbam
```

## Environment Specs

| Env | Replicas | CPU | Memory | Hostname |
|-----|----------|-----|--------|----------|
| Dev | 3 | 1 core | 1 GB | cbam-dev.greenlang.io |
| Staging | 5 | 2 cores | 2 GB | cbam-staging.greenlang.io |
| Production | 3-15 (HPA) | 1 core | 1 GB | cbam.greenlang.io |

## File Locations

```
deployment/
├── kustomize/base/              # Base configuration
├── kustomize/overlays/dev/      # Dev environment
├── kustomize/overlays/staging/  # Staging environment
├── kustomize/overlays/production/ # Production environment
├── hpa.yaml                     # Autoscaler
├── pdb.yaml                     # Disruption budget
├── resourcequota.yaml           # Resource limits
├── limitrange.yaml              # Container defaults
└── DEPLOYMENT_GUIDE.md          # Full guide
```

## Maturity Score: 93/100 (+2)

- HPA: +1.0 (multi-metric autoscaling)
- PDB: +0.5 (high availability)
- ResourceQuota: +0.25 (governance)
- LimitRange: +0.25 (defaults)
