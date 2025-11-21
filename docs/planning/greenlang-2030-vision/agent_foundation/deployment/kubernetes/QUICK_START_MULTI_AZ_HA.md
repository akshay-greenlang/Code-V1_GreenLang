# Quick Start: Multi-AZ High Availability Deployment

## 1-Minute Overview

**Objective**: Deploy GreenLang AI agents with 9 pods across 3 availability zones for maximum uptime and resilience.

**Key Features**:
- 9 pods (3 per AZ) with hard zone-level anti-affinity
- Auto-scale 9-100 pods based on load
- Network Load Balancer with cross-zone balancing
- Zero-downtime deployments
- Session affinity (3 hours)

---

## Quick Deploy (3 Commands)

```bash
# 1. Apply comprehensive HA deployment
kubectl apply -f deployment-ha.yaml

# 2. Verify pods distributed across zones
kubectl get pods -n greenlang-ai -o wide -L topology.kubernetes.io/zone

# 3. Check HPA and load balancer
kubectl get hpa,svc -n greenlang-ai
```

---

## Or: Update Existing Deployment

```bash
# Update deployment to 9 replicas with zone anti-affinity
kubectl apply -f deployment.yaml

# Update HPA to 9-100 pods
kubectl apply -f hpa.yaml

# Update service to NLB with cross-zone LB
kubectl apply -f service.yaml
```

---

## Validation

Run the automated validation script:

```bash
# Make executable (Linux/Mac)
chmod +x validate-ha-deployment.sh

# Run validation
./validate-ha-deployment.sh
```

**Or validate manually**:

```bash
# 1. Check pod count and distribution
kubectl get pods -n greenlang-ai -l app=greenlang-agent \
  -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,ZONE:".metadata.labels['topology\.kubernetes\.io/zone']"

# 2. Verify 3 pods per zone
kubectl get nodes -L topology.kubernetes.io/zone

# 3. Check HPA status
kubectl get hpa greenlang-agent-hpa -n greenlang-ai

# 4. Get load balancer endpoint
kubectl get svc greenlang-agent-lb -n greenlang-ai
```

---

## Key Configuration Changes

### deployment.yaml
```yaml
spec:
  replicas: 9  # Was: 3

  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:  # Was: preferred
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - greenlang-agent
        topologyKey: topology.kubernetes.io/zone  # Was: kubernetes.io/hostname
```

### hpa.yaml
```yaml
spec:
  minReplicas: 9   # Was: 3
  maxReplicas: 100 # Was: 20
```

### service.yaml
```yaml
metadata:
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "arn:aws:acm:..."

spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800  # 3 hours
```

---

## Expected Outcome

```
ZONE           PODS    STATUS
us-east-1a     3       Running
us-east-1b     3       Running
us-east-1c     3       Running

TOTAL          9       Running
```

**Fault Tolerance**:
- 1 AZ fails → 6 pods remain (66% capacity)
- 2 AZs fail → 3 pods remain (33% capacity)

---

## Health Check Endpoints

Update your application to expose:

```python
# Startup probe - one-time initialization check
@app.get("/startup")
async def startup():
    # Check database connection, load models, etc.
    return {"status": "ready"}

# Liveness probe - is the app alive?
@app.get("/healthz")
async def healthz():
    # Basic health check (process responsive)
    return {"status": "ok"}

# Readiness probe - can accept traffic?
@app.get("/ready")
async def ready():
    # Check dependencies (DB, Redis, etc.)
    return {"status": "ready", "dependencies": "ok"}
```

---

## Troubleshooting

### Pods not distributing across zones

```bash
# Check node zone labels
kubectl get nodes -L topology.kubernetes.io/zone

# If missing, label nodes manually:
kubectl label nodes <node-name> topology.kubernetes.io/zone=us-east-1a
```

### Pods stuck in Pending

```bash
# Check events
kubectl describe pod <pod-name> -n greenlang-ai

# Common issues:
# - Insufficient nodes in a zone (need 3+ nodes per zone)
# - Resource limits too high
# - PVC not available
```

### HPA not scaling

```bash
# Check metrics server
kubectl top pods -n greenlang-ai

# If no metrics, install metrics-server:
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Load balancer pending

```bash
# Check service events
kubectl describe svc greenlang-agent-lb -n greenlang-ai

# Common issues:
# - IAM permissions for ELB creation
# - VPC subnet tags missing
# - Security groups misconfigured
```

---

## Testing Zone Failure

Simulate an availability zone failure:

```bash
# 1. Cordon all nodes in zone us-east-1a
kubectl get nodes -l topology.kubernetes.io/zone=us-east-1a -o name | \
  xargs kubectl cordon

# 2. Delete pods in that zone (they'll reschedule to other zones)
kubectl delete pods -n greenlang-ai -l app=greenlang-agent \
  --field-selector spec.nodeName=<node-in-us-east-1a>

# 3. Verify pods redistribute
kubectl get pods -n greenlang-ai -o wide -L topology.kubernetes.io/zone

# 4. Uncordon nodes when done
kubectl get nodes -l topology.kubernetes.io/zone=us-east-1a -o name | \
  xargs kubectl uncordon
```

**Expected behavior**:
- Pods in failed zone terminate
- New pods start in healthy zones
- Service remains available (6/9 pods running)
- HPA may scale up to compensate

---

## Rollback

If deployment fails:

```bash
# Check rollout status
kubectl rollout status deployment/greenlang-agent -n greenlang-ai

# View rollout history
kubectl rollout history deployment/greenlang-agent -n greenlang-ai

# Rollback to previous version
kubectl rollout undo deployment/greenlang-agent -n greenlang-ai

# Rollback to specific revision
kubectl rollout undo deployment/greenlang-agent -n greenlang-ai --to-revision=2
```

---

## Monitoring Commands

```bash
# Watch pods in real-time
kubectl get pods -n greenlang-ai -w

# Watch HPA scaling
kubectl get hpa -n greenlang-ai -w

# Stream logs from all pods
kubectl logs -f -l app=greenlang-agent -n greenlang-ai --all-containers=true

# Get resource usage
kubectl top pods -n greenlang-ai

# Check pod distribution by zone
kubectl get pods -n greenlang-ai -o json | \
  jq -r '.items[] | "\(.spec.nodeName) \(.status.phase)"' | \
  sort | uniq -c
```

---

## Load Testing

Generate load to trigger autoscaling:

```bash
# Get service endpoint
EXTERNAL_IP=$(kubectl get svc greenlang-agent-lb -n greenlang-ai \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# Run load test (using Apache Bench)
ab -n 10000 -c 100 http://$EXTERNAL_IP/healthz

# Or using hey
hey -z 5m -c 100 http://$EXTERNAL_IP/healthz

# Watch HPA scale up
kubectl get hpa -n greenlang-ai -w
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `deployment.yaml` | Updated deployment with 9 replicas and zone anti-affinity |
| `hpa.yaml` | Updated HPA with 9-100 replicas |
| `service.yaml` | Updated NLB with cross-zone load balancing |
| `deployment-ha.yaml` | Comprehensive HA deployment (includes all resources) |
| `validate-ha-deployment.sh` | Automated validation script |
| `MULTI_AZ_HA_UPGRADE_SUMMARY.md` | Detailed upgrade documentation |
| `QUICK_START_MULTI_AZ_HA.md` | This quick start guide |

---

## Success Criteria Checklist

- [ ] 9 pods running across 3 zones (3 per zone)
- [ ] Hard pod anti-affinity on `topology.kubernetes.io/zone`
- [ ] HPA configured for 9-100 replicas
- [ ] PodDisruptionBudget ensures 6 pods minimum
- [ ] Network Load Balancer with cross-zone balancing
- [ ] Session affinity (ClientIP, 3 hours)
- [ ] Health checks on `/startup`, `/healthz`, `/ready`
- [ ] Zero-downtime updates (maxUnavailable=0)
- [ ] Load balancer external endpoint accessible

---

## Production Checklist

Before deploying to production:

1. **Secrets Management**
   ```bash
   # Create secrets (don't commit to git!)
   kubectl create secret generic greenlang-secrets \
     --from-literal=database-url="postgresql://..." \
     --from-literal=redis-url="redis://..." \
     -n greenlang-ai
   ```

2. **TLS Certificates**
   ```bash
   # Create TLS secret
   kubectl create secret tls greenlang-tls \
     --cert=path/to/tls.crt \
     --key=path/to/tls.key \
     -n greenlang-ai
   ```

3. **Update ACM Certificate ARN** in `service.yaml`:
   ```yaml
   service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "arn:aws:acm:REGION:ACCOUNT:certificate/ID"
   ```

4. **Configure Monitoring**
   - Set up Prometheus scraping
   - Create Grafana dashboards
   - Configure alerting rules

5. **Test in Staging**
   - Deploy to staging environment
   - Run validation script
   - Perform zone failure simulation
   - Load test with realistic traffic

6. **Plan Maintenance Window**
   - Schedule deployment during low-traffic period
   - Have rollback plan ready
   - Monitor metrics during and after deployment

---

## Support

- **Documentation**: See `MULTI_AZ_HA_UPGRADE_SUMMARY.md` for detailed info
- **Validation**: Run `./validate-ha-deployment.sh`
- **Issues**: Check pod events with `kubectl describe pod <name> -n greenlang-ai`

---

**Status**: Ready for deployment
**Estimated deployment time**: 10-15 minutes
**Downtime**: Zero (rolling update)

---
