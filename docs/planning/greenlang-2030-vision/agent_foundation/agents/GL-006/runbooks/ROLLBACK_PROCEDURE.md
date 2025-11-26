# GL-006 HeatRecoveryMaximizer Rollback Procedure

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-006 |
| Codename | HEATRECLAIM |
| Version | 1.0.0 |
| Last Updated | 2024-11-26 |

---

## 1. Overview

This document provides step-by-step procedures for rolling back the GL-006 HeatRecoveryMaximizer agent to a previous version. Rollbacks may be necessary when a new deployment causes issues that cannot be quickly resolved.

---

## 2. Pre-Rollback Checklist

Before initiating a rollback, complete these checks:

- [ ] Confirm the issue is deployment-related (not infrastructure)
- [ ] Verify rollback target version is known good
- [ ] Notify stakeholders of planned rollback
- [ ] Ensure you have cluster access with appropriate permissions
- [ ] Document current state (pods, versions, errors)

---

## 3. Quick Rollback (Emergency)

For emergency situations requiring immediate rollback:

```bash
# Immediate rollback to previous version
kubectl rollout undo deployment gl-006-heatreclaim -n greenlang

# Verify rollback initiated
kubectl rollout status deployment gl-006-heatreclaim -n greenlang

# Check pods are running
kubectl get pods -n greenlang -l app=gl-006-heatreclaim
```

---

## 4. Standard Rollback Procedure

### 4.1 Step 1: Assess Current State

```bash
# Check current deployment status
kubectl get deployment gl-006-heatreclaim -n greenlang -o wide

# Check current image version
kubectl get deployment gl-006-heatreclaim -n greenlang -o jsonpath='{.spec.template.spec.containers[0].image}'

# View deployment history
kubectl rollout history deployment gl-006-heatreclaim -n greenlang
```

Example output:
```
deployment.apps/gl-006-heatreclaim
REVISION  CHANGE-CAUSE
1         Initial deployment
2         Image update to v1.0.1
3         Image update to v1.0.2 (current - problematic)
```

### 4.2 Step 2: Identify Rollback Target

```bash
# View specific revision details
kubectl rollout history deployment gl-006-heatreclaim -n greenlang --revision=2

# Verify the revision has the expected image
kubectl rollout history deployment gl-006-heatreclaim -n greenlang --revision=2 -o jsonpath='{.spec.template.spec.containers[0].image}'
```

### 4.3 Step 3: Execute Rollback

#### Option A: Rollback to Previous Revision

```bash
# Rollback to immediately previous version
kubectl rollout undo deployment gl-006-heatreclaim -n greenlang
```

#### Option B: Rollback to Specific Revision

```bash
# Rollback to specific revision number
kubectl rollout undo deployment gl-006-heatreclaim -n greenlang --to-revision=2
```

#### Option C: Rollback via Image Tag

```bash
# Directly set image to known good version
kubectl set image deployment/gl-006-heatreclaim \
  gl-006-heatreclaim=gcr.io/greenlang/gl-006-heatreclaim:v1.0.1 \
  -n greenlang
```

### 4.4 Step 4: Monitor Rollback

```bash
# Watch rollout progress
kubectl rollout status deployment gl-006-heatreclaim -n greenlang -w

# Watch pod status
kubectl get pods -n greenlang -l app=gl-006-heatreclaim -w

# Check for any errors
kubectl get events -n greenlang --sort-by='.lastTimestamp' | head -20
```

### 4.5 Step 5: Verify Rollback Success

```bash
# Verify new image version
kubectl get deployment gl-006-heatreclaim -n greenlang -o jsonpath='{.spec.template.spec.containers[0].image}'

# Check all pods are running
kubectl get pods -n greenlang -l app=gl-006-heatreclaim

# Verify health endpoint
kubectl exec -it $(kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o jsonpath='{.items[0].metadata.name}') -n greenlang -- curl -s localhost:8000/health

# Check error rate metrics
curl -s localhost:9090/metrics | grep gl006_errors_total
```

---

## 5. Kustomize-Based Rollback

If using Kustomize for deployments:

### 5.1 Rollback via Git Revert

```bash
# Revert to previous kustomization
cd deployment/kustomize/overlays/production

# Update image tag in kustomization.yaml
# images:
#   - name: gcr.io/greenlang/gl-006-heatreclaim
#     newTag: v1.0.1  # Previous stable version

# Apply the reverted configuration
kubectl apply -k .
```

### 5.2 Rollback via Kustomize Edit

```bash
# Change image tag
cd deployment/kustomize/overlays/production
kustomize edit set image gcr.io/greenlang/gl-006-heatreclaim:v1.0.1

# Apply
kubectl apply -k .
```

---

## 6. Database Rollback Considerations

If the deployment included database migrations:

### 6.1 Check Migration Status

```bash
# Check current migration version
kubectl exec -it postgresql-0 -n greenlang -- psql -U greenlang -d gl006_heatreclaim -c "SELECT * FROM alembic_version;"
```

### 6.2 Rollback Migrations (if needed)

```bash
# Run migration rollback
kubectl exec -it <app-pod> -n greenlang -- alembic downgrade -1

# Or rollback to specific version
kubectl exec -it <app-pod> -n greenlang -- alembic downgrade <revision>
```

**WARNING:** Database rollbacks may cause data loss. Always backup before rolling back migrations.

---

## 7. Configuration Rollback

If configuration changes caused the issue:

### 7.1 Rollback ConfigMap

```bash
# Get previous ConfigMap (if backed up)
kubectl get configmap gl-006-heatreclaim-config-backup -n greenlang -o yaml

# Apply previous configuration
kubectl apply -f configmap-backup.yaml

# Restart deployment to pick up changes
kubectl rollout restart deployment gl-006-heatreclaim -n greenlang
```

### 7.2 Edit ConfigMap Directly

```bash
# Edit ConfigMap
kubectl edit configmap gl-006-heatreclaim-config -n greenlang

# Restart to apply
kubectl rollout restart deployment gl-006-heatreclaim -n greenlang
```

---

## 8. Post-Rollback Verification

### 8.1 Functional Verification

```bash
# Test health endpoint
curl http://gl-006.api.greenlang.io/health

# Test ready endpoint
curl http://gl-006.api.greenlang.io/ready

# Run smoke test (if available)
./scripts/smoke-test.sh production
```

### 8.2 Metrics Verification

```bash
# Check error rate is back to normal
curl -s localhost:9090/metrics | grep gl006_errors_total

# Check latency is acceptable
curl -s localhost:9090/metrics | grep gl006_http_request_duration

# Check calculation success rate
curl -s localhost:9090/metrics | grep gl006_calculations_total
```

### 8.3 Log Verification

```bash
# Check for errors in logs
kubectl logs -n greenlang -l app=gl-006-heatreclaim --since=10m | grep -i error

# Verify normal operation
kubectl logs -n greenlang -l app=gl-006-heatreclaim --since=10m | tail -20
```

---

## 9. Communication

### 9.1 Notify Stakeholders

After successful rollback:

```
GL-006 ROLLBACK COMPLETED

Time: [TIMESTAMP]
Previous Version: v1.0.2
Rolled Back To: v1.0.1
Status: Healthy

Reason for Rollback:
[Description of issue that triggered rollback]

Verification:
- Health checks: PASSING
- Error rate: Normal
- Functionality: Verified

Next Steps:
- Root cause analysis scheduled
- Fix will be developed and tested
- Re-deployment planned for [DATE]

Contact: [Your Name]
```

### 9.2 Update Incident Ticket

- Document rollback actions taken
- Note time of rollback
- Record verification results
- Link to root cause analysis

---

## 10. Rollback Troubleshooting

### 10.1 Rollback Stuck

```bash
# Check rollout status
kubectl rollout status deployment gl-006-heatreclaim -n greenlang

# If stuck, check pod events
kubectl describe pods -n greenlang -l app=gl-006-heatreclaim

# Force rollout if needed
kubectl rollout restart deployment gl-006-heatreclaim -n greenlang
```

### 10.2 Pods Not Starting After Rollback

```bash
# Check pod status
kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o wide

# Check pod events
kubectl describe pod <pod-name> -n greenlang

# Check logs
kubectl logs <pod-name> -n greenlang
```

### 10.3 Image Not Found

```bash
# Verify image exists
docker pull gcr.io/greenlang/gl-006-heatreclaim:v1.0.1

# Check image pull secrets
kubectl get secret -n greenlang | grep registry
```

---

## 11. Prevention Measures

To prevent future rollback needs:

1. **Implement Blue-Green Deployments**
   - Maintain two production environments
   - Switch traffic only after verification

2. **Canary Releases**
   - Deploy to small percentage first
   - Monitor before full rollout

3. **Feature Flags**
   - Enable new features gradually
   - Quick disable without deployment

4. **Automated Testing**
   - Run integration tests before deployment
   - Include smoke tests in CI/CD

5. **Configuration Validation**
   - Validate ConfigMaps before apply
   - Use admission controllers

---

## 12. Related Documents

- [INCIDENT_RESPONSE.md](./INCIDENT_RESPONSE.md)
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- [SCALING_GUIDE.md](./SCALING_GUIDE.md)
- [MAINTENANCE.md](./MAINTENANCE.md)

---

*This runbook is maintained by the Platform Team. For updates, contact platform-team@greenlang.io*
