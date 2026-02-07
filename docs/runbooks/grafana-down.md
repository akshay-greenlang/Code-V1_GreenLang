# Grafana Down

## Alert

**Alert Name:** `GrafanaDown`

**Severity:** Critical

**Threshold:** `up{job="grafana"} == 0`

**Duration:** 2 minutes

**Related Alert:** `GrafanaBackendDBDown` (fires if the PostgreSQL backend is unreachable)

---

## Description

This alert fires when the Grafana server is completely unreachable by Prometheus for more than 2 minutes. Grafana is the primary visualization, dashboarding, and alerting platform for GreenLang Climate OS. A complete outage means:

1. **No dashboard access** -- Engineers cannot view metrics, logs, or traces
2. **No alert evaluation** -- Grafana-managed alert rules stop evaluating
3. **No image rendering** -- Alert notification screenshots and PDF reports fail
4. **No API access** -- Programmatic SDK operations and provisioning are unavailable

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | All Grafana users lose dashboard access |
| **Alerting Impact** | Critical | Grafana-managed alerts stop firing |
| **Data Impact** | None | Underlying data (Prometheus, Loki) unaffected |
| **SLA Impact** | High | Observability SLA (99.9% uptime) breached |
| **Revenue Impact** | Medium | Delayed incident response may prolong outages |

---

## Symptoms

- Grafana web UI returns HTTP 502/503/504 or connection refused
- Prometheus target `grafana` shows `up == 0`
- Grafana-managed alert rules are not being evaluated
- Alert notification emails/Slack messages stop arriving
- Grafana API calls from Python SDK return connection errors

---

## Diagnostic Steps

### Step 1: Check Grafana Pod Status

```bash
# Check pod status in monitoring namespace
kubectl get pods -n monitoring -l app.kubernetes.io/name=grafana

# Check for recent pod restarts
kubectl get pods -n monitoring -l app.kubernetes.io/name=grafana -o wide

# Look for CrashLoopBackOff or OOMKilled
kubectl describe pods -n monitoring -l app.kubernetes.io/name=grafana | grep -A 10 "State:"
```

### Step 2: Check Pod Events

```bash
# Check recent events for Grafana pods
kubectl get events -n monitoring --field-selector involvedObject.kind=Pod --sort-by='.lastTimestamp' | grep grafana

# Check deployment events
kubectl describe deployment -n monitoring grafana
```

### Step 3: Check Grafana Logs

```bash
# Get logs from the current Grafana pod
kubectl logs -n monitoring -l app.kubernetes.io/name=grafana --tail=200

# Get logs from the previous crashed container (if restarting)
kubectl logs -n monitoring -l app.kubernetes.io/name=grafana --previous --tail=200

# Check sidecar container logs (dashboard provisioning)
kubectl logs -n monitoring -l app.kubernetes.io/name=grafana -c grafana-sc-dashboard --tail=100
```

### Step 4: Check Database Connectivity

```bash
# Verify PostgreSQL backend is reachable from within the cluster
kubectl run -n monitoring --rm -it --restart=Never pg-test \
  --image=postgres:15 -- \
  pg_isready -h grafana-db.monitoring.svc -p 5432

# Check Grafana database connection metrics (if Prometheus is up)
# PromQL: grafana_database_conn_open
# PromQL: grafana_database_conn_max
```

### Step 5: Check Resource Usage

```bash
# Check pod resource consumption
kubectl top pods -n monitoring -l app.kubernetes.io/name=grafana

# Check node resource pressure
kubectl top nodes

# Check if resource limits are being hit
kubectl get pods -n monitoring -l app.kubernetes.io/name=grafana \
  -o jsonpath='{.items[*].spec.containers[*].resources}'
```

### Step 6: Check Network and Service

```bash
# Verify Grafana service exists and has endpoints
kubectl get svc -n monitoring grafana
kubectl get endpoints -n monitoring grafana

# Test connectivity from within the cluster
kubectl run -n monitoring --rm -it --restart=Never curl-test \
  --image=curlimages/curl -- \
  curl -s -o /dev/null -w "%{http_code}" http://grafana.monitoring.svc:3000/api/health

# Check network policies
kubectl get networkpolicy -n monitoring | grep grafana
```

### Step 7: Check Persistent Volume

```bash
# Check PVC status
kubectl get pvc -n monitoring | grep grafana

# Check if PV is bound
kubectl describe pvc -n monitoring grafana-storage
```

---

## Resolution Steps

### Scenario 1: Pod is CrashLoopBackOff

**Symptoms:** Pod status shows `CrashLoopBackOff`, logs show startup errors.

**Resolution:**

1. **Check logs for the root cause:**

```bash
kubectl logs -n monitoring -l app.kubernetes.io/name=grafana --previous --tail=500
```

2. **Common log errors and fixes:**
   - `"database is locked"` -- SQLite mode detected; verify PostgreSQL configuration
   - `"failed to connect to database"` -- Check DB host, credentials, network
   - `"secret key not set"` -- Check `GF_SECURITY_SECRET_KEY` secret mount
   - `"plugin failed to start"` -- Disable problematic plugin in grafana.ini

3. **Fix and redeploy:**

```bash
# Edit Helm values and upgrade
helm upgrade grafana deployment/helm/grafana/ \
  -n monitoring \
  -f deployment/helm/grafana/values-prod.yaml

# Or for emergency, fix configmap and restart
kubectl edit configmap -n monitoring grafana-config
kubectl rollout restart deployment -n monitoring grafana
```

### Scenario 2: Pod is OOMKilled

**Symptoms:** Pod events show `OOMKilled`.

**Resolution:**

1. **Increase memory limits immediately:**

```bash
kubectl patch deployment -n monitoring grafana -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "grafana",
            "resources": {
              "limits": {
                "memory": "3Gi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

2. **Investigate the memory cause** (see [GrafanaHighMemory runbook](./grafana-high-memory.md))

### Scenario 3: Database Unreachable

**Symptoms:** Logs show `"failed to connect to database"`, `GrafanaBackendDBDown` is also firing.

**Resolution:**

1. **Check PostgreSQL (Aurora) status:**

```bash
aws rds describe-db-instances \
  --db-instance-identifier grafana-db \
  --query 'DBInstances[0].DBInstanceStatus'
```

2. **Check Kubernetes secret for DB credentials:**

```bash
kubectl get secret -n monitoring grafana-db-credentials -o yaml
```

3. **If credentials rotated, update the secret and restart:**

```bash
kubectl rollout restart deployment -n monitoring grafana
```

### Scenario 4: Simple Pod Restart Needed

**Symptoms:** No obvious error, pod just needs a fresh start.

**Resolution:**

```bash
# Perform a rolling restart
kubectl rollout restart deployment -n monitoring grafana

# Watch the rollout
kubectl rollout status deployment -n monitoring grafana --timeout=300s

# Verify health
kubectl exec -n monitoring \
  $(kubectl get pod -n monitoring -l app.kubernetes.io/name=grafana \
    -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:3000/api/health
```

### Scenario 5: Ingress or Load Balancer Issue

**Symptoms:** Pods are running and healthy, but external access fails.

**Resolution:**

```bash
# Check ingress
kubectl get ingress -n monitoring grafana
kubectl describe ingress -n monitoring grafana

# Check ingress controller pods
kubectl get pods -n ingress-nginx
kubectl logs -n ingress-nginx -l app.kubernetes.io/component=controller --tail=100

# Check TLS certificate
kubectl get certificate -n monitoring grafana-tls
kubectl describe certificate -n monitoring grafana-tls
```

---

## Emergency Actions

If Grafana is completely down and cannot be recovered quickly:

1. **Use direct Prometheus UI** at `http://prometheus.monitoring.svc:9090` for urgent metric queries
2. **Use Alertmanager UI** at `http://alertmanager.monitoring.svc:9093` to check active alerts
3. **Query Loki directly** via `logcli` for log investigation
4. **Notify the team** via Slack `#platform-alerts-critical`

```bash
# Port-forward Prometheus for direct access
kubectl port-forward -n monitoring svc/gl-prometheus-server 9090:9090

# Port-forward Alertmanager
kubectl port-forward -n monitoring svc/gl-prometheus-alertmanager 9093:9093
```

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Alert fires | On-call engineer | 5 minutes |
| L2 | Not resolved in 15 min | Platform team lead | 15 minutes |
| L3 | Database issue or data loss | Platform team + DBA | 30 minutes |
| L4 | Infrastructure failure (EKS/RDS) | SRE + AWS support | 1 hour |

---

## Prevention Measures

1. **High Availability:** Run Grafana with 2+ replicas (production default is 3)
2. **Resource monitoring:** Alert on memory at 75% of limit (GrafanaHighMemory at 1.5GB, limit is 2Gi)
3. **Database health:** Monitor PostgreSQL backend with GrafanaBackendDBDown and GrafanaDBConnectionPoolExhausted
4. **PDB configured:** PodDisruptionBudget ensures at least 1 replica during maintenance
5. **Automated backups:** PostgreSQL (Aurora) with 7-day backup retention
6. **Health checks:** Liveness and readiness probes configured on `/api/health`
7. **Test failover:** Quarterly DR exercise including Grafana recovery

---

## Related Dashboards

| Dashboard | URL |
|-----------|-----|
| Grafana Health | https://grafana.greenlang.io/d/grafana-health |
| Grafana Usage | https://grafana.greenlang.io/d/grafana-usage |
| Kubernetes Cluster | https://grafana.greenlang.io/d/kubernetes-cluster |

---

## Related Alerts

- `GrafanaBackendDBDown` -- PostgreSQL backend connectivity
- `GrafanaHighMemory` -- Memory usage approaching OOM threshold
- `GrafanaAlertingQueueFull` -- Alerting evaluation pipeline backed up
- `GrafanaAPIErrors` -- HTTP 5xx error rate elevated
- `GrafanaDBConnectionPoolExhausted` -- DB connection pool saturation

---

## References

- [Grafana Server API Health Endpoint](https://grafana.com/docs/grafana/latest/developers/http_api/other/#health-api)
- [Grafana Troubleshooting Guide](https://grafana.com/docs/grafana/latest/troubleshooting/)
- [GreenLang Monitoring Architecture](../architecture/prometheus-stack.md)
