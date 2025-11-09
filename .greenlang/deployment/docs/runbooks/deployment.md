# Deployment Runbook

Quick reference guide for deploying GreenLang-First enforcement system.

## Pre-Deployment Checklist

- [ ] All tests passing in CI/CD
- [ ] IUM score > threshold for target environment
- [ ] Staging deployment successful (for prod)
- [ ] Stakeholders notified
- [ ] Maintenance window scheduled (if required)
- [ ] Rollback plan ready
- [ ] On-call engineer assigned

## Development Deployment

```bash
# 1. Validate configuration
python .greenlang/deployment/deploy.py --env dev --status

# 2. Deploy
python .greenlang/deployment/deploy.py --env dev --component all

# 3. Verify
python .greenlang/deployment/validate.py --env dev

# 4. Test
greenlang ium calculate
pre-commit run --all-files
```

**Duration:** 5-10 minutes
**Rollback:** Automatic on failure

## Staging Deployment

```bash
# 1. Pre-deployment checks
python .greenlang/deployment/deploy.py --env staging --component all --dry-run

# 2. Deploy monitoring first
python .greenlang/deployment/deploy.py --env staging --component monitoring

# 3. Deploy enforcement
python .greenlang/deployment/deploy.py --env staging --component enforcement

# 4. Deploy dashboards
python .greenlang/deployment/deploy.py --env staging --component dashboards

# 5. Comprehensive validation
python .greenlang/deployment/validate.py --env staging --full

# 6. Smoke tests
curl https://opa.greenlang.io/health
curl https://grafana.greenlang.io/api/health
```

**Duration:** 15-20 minutes
**Rollback:** Manual if needed

## Production Deployment

```bash
# 1. Final checks
python .greenlang/deployment/validate.py --env staging --full
greenlang ium calculate --threshold 95

# 2. Notification
# Post to #engineering: "Production deployment starting at [TIME]"

# 3. Take backup
kubectl get all -n greenlang-enforcement -o yaml > backup-$(date +%Y%m%d).yaml

# 4. Deploy with confirmation
python .greenlang/deployment/deploy.py --env production --component all
# Type "CONFIRM" when prompted

# 5. Monitor deployment
watch kubectl get pods -n greenlang-enforcement

# 6. Health checks
python .greenlang/deployment/validate.py --env production --full

# 7. Verification
# - Check dashboards
# - Run test policy evaluation
# - Monitor alerts for 30 minutes

# 8. All-clear notification
# Post to #engineering: "Production deployment complete. All systems operational."
```

**Duration:** 30-45 minutes
**Rollback:** See rollback section

## Rollback Procedure

```bash
# Quick rollback
python .greenlang/deployment/deploy.py --env production --rollback

# Manual rollback
kubectl rollout undo deployment/opa-deployment -n greenlang-enforcement
kubectl rollout undo deployment/prometheus-deployment -n greenlang-enforcement

# Verify
kubectl rollout status deployment/opa-deployment -n greenlang-enforcement
```

## Troubleshooting

### Deployment Stuck

**Symptom:** Pods not reaching Ready state

**Solution:**
```bash
kubectl describe pod <pod-name> -n greenlang-enforcement
kubectl logs <pod-name> -n greenlang-enforcement
```

Common causes:
- Image pull errors
- Resource limits
- ConfigMap missing
- PVC not bound

### Health Check Failing

**Symptom:** validate.py reports failures

**Solution:**
```bash
# Check specific component
kubectl get pods -n greenlang-enforcement -l app=opa
kubectl logs -l app=opa -n greenlang-enforcement --tail=100

# Restart if needed
kubectl rollout restart deployment/opa-deployment -n greenlang-enforcement
```

## Post-Deployment

- [ ] Monitor alerts for 1 hour
- [ ] Check error rates in Grafana
- [ ] Verify metrics collection
- [ ] Update deployment log
- [ ] Document any issues

## Emergency Contacts

- SRE On-Call: Slack @sre-oncall
- DevOps Lead: Slack @devops-lead
- War Room: #incident-response
