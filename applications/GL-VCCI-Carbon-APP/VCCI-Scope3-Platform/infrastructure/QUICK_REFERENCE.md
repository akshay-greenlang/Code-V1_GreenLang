# Kubernetes Infrastructure - Quick Reference Guide
## VCCI Scope 3 Carbon Intelligence Platform

---

## Essential Commands

### Deployment

```bash
# Deploy to development
kubectl apply -k infrastructure/kubernetes/kustomization/overlays/dev/

# Deploy to staging
kubectl apply -k infrastructure/kubernetes/kustomization/overlays/staging/

# Deploy to production
kubectl apply -k infrastructure/kubernetes/kustomization/overlays/production/

# Preview changes (dry-run)
kubectl apply -k infrastructure/kubernetes/kustomization/overlays/production/ --dry-run=client

# Diff before applying
kubectl diff -k infrastructure/kubernetes/kustomization/overlays/production/
```

### Status Checks

```bash
# Check all pods
kubectl get pods -n vcci-platform

# Check services
kubectl get svc -n vcci-platform

# Check ingress
kubectl get ingress -n vcci-platform

# Check HPAs
kubectl get hpa -n vcci-platform

# Check PVCs
kubectl get pvc -n vcci-platform

# Check everything
kubectl get all -n vcci-platform
```

### Logs and Debugging

```bash
# View logs
kubectl logs -f deployment/vcci-backend-api -n vcci-platform

# View logs from all containers
kubectl logs -f deployment/vcci-backend-api --all-containers -n vcci-platform

# View previous logs (after crash)
kubectl logs deployment/vcci-backend-api --previous -n vcci-platform

# Describe pod
kubectl describe pod <pod-name> -n vcci-platform

# Exec into pod
kubectl exec -it <pod-name> -n vcci-platform -- /bin/bash

# Port forward for testing
kubectl port-forward svc/vcci-backend-api 8000:8000 -n vcci-platform
```

### Scaling

```bash
# Manual scale
kubectl scale deployment vcci-backend-api --replicas=5 -n vcci-platform

# Check HPA
kubectl get hpa -n vcci-platform -w

# Describe HPA
kubectl describe hpa vcci-backend-api-hpa -n vcci-platform
```

### Updates and Rollouts

```bash
# Update image
kubectl set image deployment/vcci-backend-api \
  api=greenlang/vcci-backend-api:v1.1.0 \
  -n vcci-platform

# Check rollout status
kubectl rollout status deployment/vcci-backend-api -n vcci-platform

# Rollout history
kubectl rollout history deployment/vcci-backend-api -n vcci-platform

# Rollback
kubectl rollout undo deployment/vcci-backend-api -n vcci-platform

# Rollback to specific revision
kubectl rollout undo deployment/vcci-backend-api --to-revision=2 -n vcci-platform

# Restart deployment
kubectl rollout restart deployment/vcci-backend-api -n vcci-platform
```

---

## File Organization Quick Reference

```
infrastructure/kubernetes/
│
├── README.md                      → Main documentation (627 lines)
│
├── base/                          → Base infrastructure (4 files)
│   ├── namespace.yaml            → Namespaces + tenant template
│   ├── rbac.yaml                 → Service accounts + RBAC
│   ├── resource-quotas.yaml      → Resource limits per tier
│   └── network-policies.yaml     → Network isolation rules
│
├── applications/                  → Application layer (15 files)
│   ├── api-gateway/              → NGINX gateway (5 files)
│   ├── backend-api/              → FastAPI/Flask (4 files)
│   ├── worker/                   → Celery workers (3 files)
│   └── frontend/                 → React SPA (3 files)
│
├── data/                          → Data layer (9 files)
│   ├── postgresql/               → PostgreSQL cluster (4 files)
│   ├── redis/                    → Redis cluster (3 files)
│   └── weaviate/                 → Vector DB (3 files)
│
├── observability/                 → Monitoring (11 files)
│   ├── prometheus/               → Metrics (4 files)
│   ├── grafana/                  → Dashboards (4 files)
│   ├── fluentd/                  → Logs (2 files)
│   └── jaeger/                   → Tracing (2 files)
│
├── security/                      → Security layer (4 files)
│   ├── cert-manager/             → TLS certificates (2 files)
│   ├── sealed-secrets/           → Encrypted secrets (1 file)
│   └── pod-security-policies.yaml → Pod security (1 file)
│
└── kustomization/                 → Environment configs (4 files)
    ├── base/                     → Base kustomization
    └── overlays/
        ├── dev/                  → Development
        ├── staging/              → Staging
        └── production/           → Production
```

---

## Resource Specifications

### Production Configuration

| Component | Replicas | CPU (req/lim) | Memory (req/lim) | Storage |
|-----------|----------|---------------|------------------|---------|
| API Gateway | 3-10 | 500m/2000m | 512Mi/2Gi | - |
| Backend API | 3-10 | 500m/2000m | 1Gi/4Gi | - |
| Workers | 2-20 | 1000m/4000m | 2Gi/8Gi | - |
| Workers (ML) | 1-5 | 2000m/4000m | 8Gi/16Gi | 100Gi |
| Frontend | 2-5 | 100m/500m | 256Mi/1Gi | - |
| PostgreSQL | 3 | 1000m/4000m | 4Gi/16Gi | 100Gi/node |
| Redis | 3 | 500m/2000m | 2Gi/8Gi | 10Gi/node |
| Weaviate | 3 | 2000m/8000m | 8Gi/32Gi | 50Gi/node |

### HPA Targets

| Component | Min | Max | CPU % | Memory % | Custom Metrics |
|-----------|-----|-----|-------|----------|----------------|
| API Gateway | 3 | 10 | 70 | 80 | RPS: 1000, Conn: 500 |
| Backend API | 3 | 10 | 70 | 80 | RPS: 100, Latency: 500ms |
| Workers | 2 | 20 | 75 | 85 | Queue: 100, Runtime: 300s |
| Workers (ML) | 1 | 5 | - | - | GPU: 80%, Queue: 10 |
| Frontend | 2 | 5 | 70 | 80 | - |

---

## Common Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl get pod <pod-name> -n vcci-platform
kubectl describe pod <pod-name> -n vcci-platform

# Check events
kubectl get events -n vcci-platform --sort-by='.lastTimestamp' | tail -20

# Check logs
kubectl logs <pod-name> -n vcci-platform --previous
```

**Common Issues**:
- ImagePullBackOff → Check image name and registry access
- CrashLoopBackOff → Check logs for application errors
- Pending → Check resource availability and node selectors

### Service Not Accessible

```bash
# Check service
kubectl get svc vcci-backend-api -n vcci-platform
kubectl describe svc vcci-backend-api -n vcci-platform

# Check endpoints
kubectl get endpoints vcci-backend-api -n vcci-platform

# Test internal connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://vcci-backend-api.vcci-platform.svc.cluster.local:8000/health
```

### Database Connection Issues

```bash
# Check PostgreSQL pods
kubectl get pods -l app=postgresql -n vcci-platform
kubectl logs -l app=postgresql -n vcci-platform --tail=50

# Test connection
kubectl run -it --rm psql --image=postgres:15-alpine --restart=Never -- \
  psql -h postgresql.vcci-platform.svc.cluster.local -U vcci -d vcci

# Check Redis
kubectl exec -it redis-0 -n vcci-platform -- redis-cli ping
```

### Certificate Issues

```bash
# Check certificates
kubectl get certificate -n vcci-platform
kubectl describe certificate vcci-api-cert -n vcci-platform

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager --tail=100

# Force renewal
kubectl delete certificaterequest -n vcci-platform --all
```

---

## Monitoring Access

### Grafana

```bash
# Get admin password
kubectl get secret grafana-secrets -n vcci-observability -o jsonpath="{.data.admin-password}" | base64 -d

# Port forward
kubectl port-forward svc/grafana 3000:3000 -n vcci-observability

# Access: http://localhost:3000
```

### Prometheus

```bash
# Port forward
kubectl port-forward svc/prometheus 9090:9090 -n vcci-observability

# Access: http://localhost:9090
```

### Jaeger

```bash
# Port forward
kubectl port-forward svc/jaeger-all-in-one 16686:16686 -n vcci-observability

# Access: http://localhost:16686
```

---

## Security Operations

### Create Sealed Secret

```bash
# Create secret YAML
kubectl create secret generic my-secret \
  --from-literal=key=value \
  --dry-run=client -o yaml > secret.yaml

# Seal it
kubeseal -f secret.yaml -w sealed-secret.yaml

# Apply sealed secret
kubectl apply -f sealed-secret.yaml

# Clean up
rm secret.yaml
```

### Rotate Secrets

```bash
# Update secret
kubectl create secret generic vcci-backend-secrets \
  --from-literal=jwt-secret=$(openssl rand -base64 32) \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new secret
kubectl rollout restart deployment/vcci-backend-api -n vcci-platform
```

### Check Security Policies

```bash
# Network policies
kubectl get networkpolicy -n vcci-platform
kubectl describe networkpolicy allow-backend-to-database -n vcci-platform

# Pod security
kubectl get psp
kubectl get psp vcci-restricted -o yaml

# RBAC
kubectl get rolebinding -n vcci-platform
kubectl get clusterrolebinding | grep vcci
```

---

## Backup and Recovery

### PostgreSQL Backup

```bash
# Manual backup
kubectl exec -it postgresql-0 -n vcci-platform -- \
  pg_dump -U vcci -Fc -f /backup/vcci-$(date +%Y%m%d-%H%M%S).dump vcci

# Copy backup locally
kubectl cp vcci-platform/postgresql-0:/backup/vcci-backup.dump ./vcci-backup.dump
```

### PostgreSQL Restore

```bash
# Copy backup to pod
kubectl cp ./vcci-backup.dump vcci-platform/postgresql-0:/backup/restore.dump

# Scale down apps
kubectl scale deployment vcci-backend-api --replicas=0 -n vcci-platform
kubectl scale deployment vcci-worker --replicas=0 -n vcci-platform

# Restore
kubectl exec -it postgresql-0 -n vcci-platform -- \
  pg_restore -U vcci -d vcci -c /backup/restore.dump

# Scale up apps
kubectl scale deployment vcci-backend-api --replicas=3 -n vcci-platform
kubectl scale deployment vcci-worker --replicas=2 -n vcci-platform
```

---

## Multi-Tenant Operations

### Create New Tenant

```bash
# Set variables
export TENANT_ID="acme-corp"
export TENANT_NAME="ACME Corporation"
export TENANT_TIER="enterprise"

# Create namespace
kubectl create namespace vcci-tenant-${TENANT_ID}
kubectl label namespace vcci-tenant-${TENANT_ID} \
  tenant=${TENANT_ID} \
  tenant-tier=${TENANT_TIER}

# Apply quotas
kubectl apply -f base/resource-quotas.yaml \
  -l tier=${TENANT_TIER} \
  -n vcci-tenant-${TENANT_ID}

# Apply network policies
kubectl apply -f base/network-policies.yaml \
  -n vcci-tenant-${TENANT_ID}
```

### Delete Tenant

```bash
export TENANT_ID="acme-corp"

# Backup data first!
# Then delete namespace (cascades all resources)
kubectl delete namespace vcci-tenant-${TENANT_ID}
```

---

## Performance Tuning

### Get Resource Usage

```bash
# Node resources
kubectl top nodes

# Pod resources
kubectl top pods -n vcci-platform

# Sort by CPU
kubectl top pods -n vcci-platform --sort-by=cpu

# Sort by memory
kubectl top pods -n vcci-platform --sort-by=memory
```

### Optimize Resources

```bash
# Check resource limits
kubectl describe deployment vcci-backend-api -n vcci-platform | grep -A 5 "Limits:"

# Update resource limits
kubectl set resources deployment vcci-backend-api \
  -n vcci-platform \
  --requests=cpu=500m,memory=1Gi \
  --limits=cpu=2000m,memory=4Gi
```

---

## Cost Monitoring

### Check Resource Allocation

```bash
# Get resource quotas
kubectl get resourcequota -n vcci-platform
kubectl describe resourcequota vcci-platform-quota -n vcci-platform

# Check PVC usage
kubectl get pvc -n vcci-platform
kubectl describe pvc -n vcci-platform | grep "Used:"

# Check node usage
kubectl get nodes -o custom-columns=NAME:.metadata.name,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory
```

---

## Emergency Procedures

### Complete Platform Restart

```bash
# 1. Scale down all applications
kubectl scale deployment --all --replicas=0 -n vcci-platform

# 2. Wait for pods to terminate
kubectl get pods -n vcci-platform -w

# 3. Restart StatefulSets (optional)
kubectl rollout restart statefulset/postgresql -n vcci-platform
kubectl rollout restart statefulset/redis -n vcci-platform
kubectl rollout restart statefulset/weaviate -n vcci-platform

# 4. Scale up applications
kubectl scale deployment vcci-backend-api --replicas=3 -n vcci-platform
kubectl scale deployment vcci-worker --replicas=2 -n vcci-platform
kubectl scale deployment vcci-api-gateway --replicas=3 -n vcci-platform
kubectl scale deployment vcci-frontend --replicas=2 -n vcci-platform
```

### Disaster Recovery

```bash
# 1. Restore from backup (see Backup section)
# 2. Verify data integrity
# 3. Scale up applications
# 4. Run health checks
# 5. Verify monitoring
# 6. Test end-to-end functionality
```

---

## Support Contacts

- **Platform Team**: platform@greenlang.io
- **Security Team**: security@greenlang.io
- **On-Call**: oncall@greenlang.io
- **Documentation**: https://docs.vcci.greenlang.io

---

## Quick Links

- Grafana: https://monitoring.vcci.greenlang.io
- API Docs: https://api.vcci.greenlang.io/docs
- Frontend: https://app.vcci.greenlang.io
- Status Page: https://status.vcci.greenlang.io

---

**Last Updated**: 2025-01-06
**Version**: 1.0.0
