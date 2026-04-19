# GL-VCCI Deployment Quick Reference

## Common Commands

### Deploy

```bash
# Deploy to development
./deployment/scripts/deploy.sh -e dev -v v2.0.0

# Deploy to staging
./deployment/scripts/deploy.sh -e staging -v v2.0.0

# Deploy to production (rolling)
./deployment/scripts/deploy.sh -e prod -v v2.0.0

# Deploy to production (canary)
./deployment/scripts/deploy.sh -e prod -v v2.0.0 -s canary

# Deploy to production (blue-green)
./deployment/scripts/deploy.sh -e prod -v v2.0.0 -s blue-green

# Dry run
./deployment/scripts/deploy.sh -e prod -v v2.0.0 --dry-run
```

### Check Status

```bash
# Get all resources
kubectl get all -n vcci-production

# Get pods
kubectl get pods -n vcci-production

# Get deployments
kubectl get deployments -n vcci-production

# Get HPA status
kubectl get hpa -n vcci-production

# Get PDB status
kubectl get pdb -n vcci-production

# Check pod logs
kubectl logs -f <pod-name> -n vcci-production

# Describe pod
kubectl describe pod <pod-name> -n vcci-production
```

### Rollback

```bash
# Rollback to previous version
./deployment/scripts/rollback.sh prod

# Rollback to specific revision
./deployment/scripts/rollback.sh prod 3

# Check rollout history
kubectl rollout history deployment/vcci-backend-api -n vcci-production
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment vcci-backend-api --replicas=5 -n vcci-production

# Check HPA status
kubectl describe hpa vcci-backend-api-hpa -n vcci-production

# Edit HPA
kubectl edit hpa vcci-backend-api-hpa -n vcci-production
```

### Secrets

```bash
# List secrets
kubectl get secrets -n vcci-production

# Get external secrets status
kubectl get externalsecret -n vcci-production

# Describe external secret
kubectl describe externalsecret vcci-postgres-credentials -n vcci-production

# Force secret refresh
kubectl annotate externalsecret vcci-api-keys force-sync=$(date +%s) -n vcci-production
```

### Network Policies

```bash
# List network policies
kubectl get networkpolicy -n vcci-production

# Describe network policy
kubectl describe networkpolicy vcci-backend-api-netpol -n vcci-production

# Test connectivity
kubectl run test-pod --rm -it --image=nicolaka/netshoot -n vcci-production -- /bin/bash
curl http://vcci-backend-api:8000/health
```

### Istio (if installed)

```bash
# Check Istio injection
kubectl get namespace -L istio-injection

# Enable injection for namespace
kubectl label namespace vcci-production istio-injection=enabled

# Check virtual services
kubectl get virtualservice -n vcci-production

# Check destination rules
kubectl get destinationrule -n vcci-production

# Analyze Istio config
istioctl analyze -n vcci-production

# Open Kiali
istioctl dashboard kiali

# Open Jaeger
istioctl dashboard jaeger
```

### Troubleshooting

```bash
# Get events
kubectl get events -n vcci-production --sort-by='.lastTimestamp'

# Check resource usage
kubectl top nodes
kubectl top pods -n vcci-production

# Exec into pod
kubectl exec -it <pod-name> -n vcci-production -- /bin/bash

# Port forward to service
kubectl port-forward service/vcci-backend-api 8000:8000 -n vcci-production

# Check service endpoints
kubectl get endpoints vcci-backend-api -n vcci-production

# Check PVC status
kubectl get pvc -n vcci-production

# Restart deployment
kubectl rollout restart deployment/vcci-backend-api -n vcci-production
```

### Monitoring

```bash
# Port forward to Prometheus
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# Port forward to Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Get metrics from pod
kubectl exec <pod-name> -n vcci-production -- curl http://localhost:8000/metrics
```

## File Locations

| What | Where |
|------|-------|
| Base manifests | `k8s/base/` |
| Environment configs | `k8s/overlays/{dev,staging,prod}/` |
| HPA config | `k8s/hpa.yaml` |
| PDB config | `k8s/pdb.yaml` |
| Network policies | `k8s/network-policy.yaml` |
| Resource quotas | `k8s/resource-quota.yaml` |
| External secrets | `k8s/external-secrets.yaml` |
| Istio configs | `k8s/istio/` |
| Deployment scripts | `deployment/scripts/` |
| Documentation | `deployment/DEPLOYMENT_GUIDE.md` |

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DOCKER_REGISTRY` | Docker registry URL | `gcr.io/my-project` |
| `VERSION` | Deployment version | `v2.0.0` |
| `KUBECONFIG` | Kubernetes config file | `~/.kube/config` |

## Health Endpoints

| Service | Endpoint | Port |
|---------|----------|------|
| Backend API | `/health/live` | 8000 |
| Backend API | `/health/ready` | 8000 |
| Backend API | `/metrics` | 8000 |
| Worker | Celery inspect | 5555 |

## Resource Limits

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---------|-------------|-----------|----------------|--------------|
| Backend API (prod) | 1000m | 4000m | 2Gi | 8Gi |
| Worker (prod) | 2000m | 8000m | 4Gi | 16Gi |
| PostgreSQL | 2000m | 8000m | 4Gi | 16Gi |
| Redis | 500m | 2000m | 1Gi | 4Gi |

## HPA Thresholds

| Service | Min | Max | CPU | Memory |
|---------|-----|-----|-----|--------|
| Backend API | 3 | 20 | 70% | 80% |
| Worker | 2 | 15 | 75% | 85% |
| Frontend | 2 | 10 | 60% | 70% |

## Common Issues

### Issue: Pods in CrashLoopBackOff
```bash
kubectl logs <pod-name> -n vcci-production --previous
kubectl describe pod <pod-name> -n vcci-production
```

### Issue: Image pull errors
```bash
# Check image name
kubectl describe pod <pod-name> -n vcci-production | grep Image

# Check secrets
kubectl get secrets -n vcci-production
```

### Issue: Service not accessible
```bash
# Check service
kubectl get svc vcci-backend-api -n vcci-production

# Check endpoints
kubectl get endpoints vcci-backend-api -n vcci-production

# Test from inside cluster
kubectl run test --rm -it --image=busybox -n vcci-production -- wget -O- http://vcci-backend-api:8000/health
```

### Issue: HPA not working
```bash
# Check metrics server
kubectl top nodes

# Check HPA status
kubectl describe hpa vcci-backend-api-hpa -n vcci-production

# Check custom metrics
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1
```

## Emergency Procedures

### Complete Rollback
```bash
./deployment/scripts/rollback.sh prod
```

### Scale Down
```bash
kubectl scale deployment vcci-backend-api --replicas=0 -n vcci-production
```

### Restart All
```bash
kubectl rollout restart deployment -n vcci-production
```

### Delete Stuck Resources
```bash
kubectl delete pod <pod-name> --force --grace-period=0 -n vcci-production
```

## Support

- Documentation: `deployment/DEPLOYMENT_GUIDE.md`
- Team Slack: `#vcci-deployment`
- On-call: Check PagerDuty
