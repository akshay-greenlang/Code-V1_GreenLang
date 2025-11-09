# Troubleshooting Runbook

Common issues and resolutions for GreenLang-First enforcement system.

## OPA Issues

### OPA Service Not Responding

**Symptoms:**
- Policy evaluation timeouts
- 503 errors
- PRs blocked

**Diagnosis:**
```bash
kubectl get pods -n greenlang-enforcement -l app=opa
kubectl describe pod <opa-pod> -n greenlang-enforcement
kubectl logs <opa-pod> -n greenlang-enforcement --tail=100
```

**Solutions:**

1. **Restart OPA pods:**
```bash
kubectl rollout restart deployment/opa-deployment -n greenlang-enforcement
```

2. **Check resource limits:**
```bash
kubectl top pods -n greenlang-enforcement -l app=opa
# If CPU/memory maxed out, increase limits
```

3. **Check policy syntax:**
```bash
cd .greenlang/enforcement/opa-policies
opa test . -v
```

4. **Emergency bypass:**
```bash
greenlang config set enforcement.mode permissive
```

### OPA High Latency

**Symptoms:**
- Slow policy evaluations (>100ms)
- Timeouts

**Diagnosis:**
```bash
# Check metrics
curl http://prometheus-service:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket{job="opa"}[5m]))
```

**Solutions:**

1. **Enable caching:**
```bash
greenlang config set opa.cache_enabled true
greenlang config set opa.cache_ttl 600
```

2. **Scale up replicas:**
```bash
kubectl scale deployment opa-deployment -n greenlang-enforcement --replicas=10
```

3. **Optimize policies:**
- Review complex policy rules
- Add indexes where needed
- Simplify logic

## Monitoring Issues

### Dashboards Not Loading

**Symptoms:**
- Grafana shows "No data"
- Blank charts

**Diagnosis:**
```bash
# Check Prometheus
curl http://prometheus-service:9090/-/healthy

# Check if data is being scraped
curl 'http://prometheus-service:9090/api/v1/query?query=up'

# Check Grafana datasource
curl http://grafana-service:3000/api/datasources
```

**Solutions:**

1. **Restart Grafana:**
```bash
kubectl rollout restart deployment/grafana-deployment -n greenlang-enforcement
```

2. **Check datasource config:**
```bash
kubectl get configmap grafana-datasources -n greenlang-enforcement -o yaml
```

3. **Verify Prometheus is scraping:**
```bash
kubectl logs -l app=prometheus -n greenlang-enforcement | grep "scrape"
```

### Alerts Not Firing

**Symptoms:**
- Expected alerts not received
- No Slack notifications

**Diagnosis:**
```bash
# Check AlertManager
curl http://alertmanager-service:9093/-/healthy

# Check alert rules
curl http://prometheus-service:9090/api/v1/rules

# Check AlertManager config
kubectl get configmap alertmanager-config -n greenlang-enforcement -o yaml
```

**Solutions:**

1. **Verify Slack webhook:**
```bash
kubectl get secret alertmanager-secrets -n greenlang-enforcement -o yaml
# Check if slack-webhook-url is set correctly
```

2. **Test alert manually:**
```bash
curl -XPOST http://alertmanager-service:9093/api/v1/alerts -d '[
  {
    "labels": {
      "alertname": "TestAlert",
      "severity": "warning"
    },
    "annotations": {
      "summary": "Test alert"
    }
  }
]'
```

3. **Check inhibit rules:**
- Review inhibit_rules in AlertManager config
- May be suppressing alerts

## Pre-commit Hook Issues

### Hooks Not Running

**Symptoms:**
- Commits succeed without running checks
- No enforcement locally

**Diagnosis:**
```bash
ls -la .git/hooks/pre-commit
cat .git/hooks/pre-commit
pre-commit --version
```

**Solutions:**

1. **Reinstall hooks:**
```bash
pre-commit uninstall
pre-commit install
pre-commit install --hook-type commit-msg
```

2. **Check config:**
```bash
cat .pre-commit-config.yaml
pre-commit run --all-files
```

3. **Manual run:**
```bash
git commit --no-verify  # Bypass if urgent
# Then fix hooks
```

### Hooks Failing with Errors

**Symptoms:**
- "Command not found" errors
- Import errors

**Diagnosis:**
```bash
pre-commit run --all-files --verbose
```

**Solutions:**

1. **Update dependencies:**
```bash
pre-commit clean
pre-commit autoupdate
pre-commit install
```

2. **Check Python environment:**
```bash
which python3
pip3 list | grep greenlang
```

3. **Reinstall greenlang CLI:**
```bash
cd .greenlang/cli
pip3 install -e . --force-reinstall
```

## IUM Score Issues

### IUM Score Below Threshold

**Symptoms:**
- Deployments blocked
- "IUM score too low" errors

**Diagnosis:**
```bash
greenlang ium calculate --verbose
greenlang ium calculate --path ./src/
```

**Solutions:**

1. **Identify violations:**
```bash
greenlang ium calculate --show-violations
```

2. **Fix common issues:**
- Add missing Terraform files
- Document infrastructure with ADRs
- Lint Dockerfiles
- Add Kubernetes manifests

3. **Emergency override (dev only):**
```bash
GREENLANG_OVERRIDE="Reason: hotfix" git commit -m "fix: critical bug"
```

### False Positive Violations

**Symptoms:**
- Valid code flagged as violation
- Incorrect IUM calculation

**Diagnosis:**
```bash
greenlang ium calculate --debug
greenlang lint --verbose
```

**Solutions:**

1. **Report false positive:**
```bash
# Create issue with example
# Tag as "false-positive"
```

2. **Temporary exclusion:**
```bash
# Add to .greenlangignore
echo "path/to/file.py" >> .greenlangignore
```

3. **Update policy:**
```bash
# Fix policy in .greenlang/enforcement/opa-policies/
# Test and deploy
```

## Deployment Issues

### Deployment Stuck

**Symptoms:**
- Pods in Pending state
- Deployment not progressing

**Diagnosis:**
```bash
kubectl get pods -n greenlang-enforcement
kubectl describe pod <pod-name> -n greenlang-enforcement
kubectl get events -n greenlang-enforcement --sort-by='.lastTimestamp'
```

**Solutions:**

1. **Resource issues:**
```bash
kubectl top nodes
kubectl describe nodes
# Scale down or add nodes
```

2. **Image pull errors:**
```bash
kubectl get pods -n greenlang-enforcement -o yaml | grep -i image
# Check image name, registry access
```

3. **PVC issues:**
```bash
kubectl get pvc -n greenlang-enforcement
# Check storage class, provisioner
```

### Rollback Failed

**Symptoms:**
- Rollback command errors
- Still running bad version

**Diagnosis:**
```bash
kubectl rollout history deployment/opa-deployment -n greenlang-enforcement
kubectl get replicasets -n greenlang-enforcement
```

**Solutions:**

1. **Manual rollback:**
```bash
# Find previous revision
kubectl rollout history deployment/opa-deployment -n greenlang-enforcement

# Rollback to specific revision
kubectl rollout undo deployment/opa-deployment -n greenlang-enforcement --to-revision=2
```

2. **Force update:**
```bash
kubectl delete pod -l app=opa -n greenlang-enforcement
```

3. **Emergency restore:**
```bash
kubectl apply -f backup-YYYYMMDD.yaml
```

## Performance Issues

### High CPU Usage

**Symptoms:**
- Pods throttled
- Slow responses

**Diagnosis:**
```bash
kubectl top pods -n greenlang-enforcement
kubectl describe pod <pod-name> -n greenlang-enforcement | grep -A5 Limits
```

**Solutions:**

1. **Increase limits:**
```bash
kubectl patch deployment opa-deployment -n greenlang-enforcement -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "opa",
            "resources": {
              "limits": {
                "cpu": "2000m"
              }
            }
          }
        ]
      }
    }
  }
}'
```

2. **Scale horizontally:**
```bash
kubectl scale deployment opa-deployment -n greenlang-enforcement --replicas=10
```

3. **Profile application:**
```bash
kubectl port-forward <opa-pod> 6060:6060 -n greenlang-enforcement
go tool pprof http://localhost:6060/debug/pprof/profile
```

### High Memory Usage

**Symptoms:**
- OOMKilled pods
- Frequent restarts

**Diagnosis:**
```bash
kubectl top pods -n greenlang-enforcement
kubectl get events -n greenlang-enforcement | grep OOMKilled
```

**Solutions:**

1. **Increase memory limits:**
```bash
kubectl patch deployment opa-deployment -n greenlang-enforcement -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "opa",
            "resources": {
              "limits": {
                "memory": "2Gi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

2. **Check for memory leaks:**
```bash
kubectl logs <pod-name> -n greenlang-enforcement | grep -i memory
```

3. **Enable caching to reduce memory:**
```bash
greenlang config set opa.cache_size 500MB
```

## Network Issues

### Service Unreachable

**Symptoms:**
- Cannot connect to services
- Timeout errors

**Diagnosis:**
```bash
kubectl get svc -n greenlang-enforcement
kubectl get endpoints -n greenlang-enforcement
kubectl get ingress -n greenlang-enforcement
```

**Solutions:**

1. **Check service selector:**
```bash
kubectl get svc opa-service -n greenlang-enforcement -o yaml
kubectl get pods -n greenlang-enforcement --show-labels
```

2. **Test connectivity:**
```bash
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
# From inside pod:
wget -O- http://opa-service:8181/health
```

3. **Check network policies:**
```bash
kubectl get networkpolicies -n greenlang-enforcement
```

## Security Issues

### Authentication Failures

**Symptoms:**
- 401 Unauthorized errors
- Cannot access dashboards

**Diagnosis:**
```bash
kubectl get secrets -n greenlang-enforcement
kubectl logs -l app=grafana -n greenlang-enforcement | grep auth
```

**Solutions:**

1. **Reset credentials:**
```bash
kubectl delete secret grafana-secrets -n greenlang-enforcement
kubectl create secret generic grafana-secrets -n greenlang-enforcement \
  --from-literal=admin-user=admin \
  --from-literal=admin-password=newpassword
```

2. **Check RBAC:**
```bash
kubectl get rolebindings -n greenlang-enforcement
kubectl auth can-i get pods -n greenlang-enforcement
```

## Escalation

If issue persists:

1. **Create incident:** #incident-response
2. **Page on-call:** @sre-oncall
3. **Gather diagnostics:**
```bash
kubectl cluster-info dump > cluster-dump.txt
kubectl get all -n greenlang-enforcement -o yaml > enforcement-state.yaml
```
4. **Enable debug mode:**
```bash
greenlang config set debug.enabled true
```
5. **Engage vendor support** if needed

## Quick Commands Reference

```bash
# Health check everything
python .greenlang/deployment/validate.py --env production --full

# Restart all
kubectl rollout restart deployment -n greenlang-enforcement

# Check logs
kubectl logs -f -l app=opa -n greenlang-enforcement

# Emergency bypass
greenlang config set enforcement.mode permissive

# Status
kubectl get all -n greenlang-enforcement
```
