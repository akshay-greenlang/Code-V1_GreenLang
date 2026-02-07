# Alerting Service Down

## Alert

**Alert Name:** `AlertingServiceDown`

**Severity:** Critical

**Threshold:** `absent(up{job="alerting-service"} == 1) or sum(up{job="alerting-service"}) == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang Alerting Service are running. The Alerting Service is responsible for:

1. **Receiving alert webhooks** from Prometheus Alertmanager
2. **Routing alerts** to the appropriate notification channels based on severity and team
3. **Delivering notifications** to PagerDuty, Opsgenie, Slack, and Email
4. **Managing escalations** when alerts are not acknowledged within SLA
5. **Tracking MTTA/MTTR** metrics for alert response analytics
6. **Deduplicating alerts** to reduce notification noise

When the Alerting Service is down, ALL notifications stop. Critical infrastructure alerts from Prometheus will fire but will NOT reach any human responder through any channel.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | On-call responders receive zero notifications for any severity |
| **Data Impact** | High | Alert events accumulate in Alertmanager without delivery confirmation |
| **SLA Impact** | Critical | MTTA/MTTR SLAs cannot be met; incident response is blind |
| **Revenue Impact** | High | Production incidents may go unnoticed, leading to extended outages |
| **Compliance Impact** | High | SOC 2 CC7.2 (monitoring) and CC7.3 (incident response) controls at risk |

---

## Symptoms

- `up{job="alerting-service"}` metric returns 0 or is absent
- No pods running in the `greenlang-alerting` namespace
- Alertmanager webhook receiver returning 5xx errors or connection refused
- Slack alert channels are silent despite active Prometheus alerts
- PagerDuty/Opsgenie show no new incidents despite known issues
- `gl_alert_notifications_total` counter has stopped incrementing
- Alertmanager "Silenced" count is zero but no alerts are being delivered

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List alerting service pods
kubectl get pods -n greenlang-alerting -l app=alerting-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang-alerting -l app=alerting-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace
kubectl get events -n greenlang-alerting --sort-by='.lastTimestamp' | tail -30

# Check deployment status
kubectl describe deployment alerting-service -n greenlang-alerting
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas
kubectl get deployment alerting-service -n greenlang-alerting

# Check ReplicaSet status
kubectl get replicaset -n greenlang-alerting -l app=alerting-service

# Check for rollout issues
kubectl rollout status deployment/alerting-service -n greenlang-alerting

# Check HPA status
kubectl get hpa -n greenlang-alerting
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang-alerting -l app=alerting-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang-alerting <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang-alerting -l app=alerting-service --tail=500 | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Check for database connection errors
kubectl logs -n greenlang-alerting -l app=alerting-service --tail=500 | grep -i "database\|postgres\|redis\|connection"
```

### Step 4: Check Dependencies (PostgreSQL, Redis)

```bash
# Verify PostgreSQL connectivity from within the namespace
kubectl run pg-test --rm -it --image=busybox:1.36 -n greenlang-alerting --restart=Never -- \
  sh -c 'nc -zv greenlang-postgresql.database.svc.cluster.local 5432'

# Verify Redis connectivity
kubectl run redis-test --rm -it --image=busybox:1.36 -n greenlang-alerting --restart=Never -- \
  sh -c 'nc -zv greenlang-redis.redis.svc.cluster.local 6379'

# Check if init containers are stuck waiting for dependencies
kubectl describe pod -n greenlang-alerting -l app=alerting-service | grep -A5 "Init Containers"
```

### Step 5: Check Resource Limits and Node Capacity

```bash
# Check if pods are being evicted due to resource pressure
kubectl get events -n greenlang-alerting --field-selector reason=Evicted

# Check node resource availability
kubectl top nodes

# Check if PDB is preventing scheduling
kubectl get pdb -n greenlang-alerting

# Check if resource quota is exhausted
kubectl describe resourcequota -n greenlang-alerting
```

---

## Resolution Steps

### Scenario 1: CrashLoopBackOff (Application Error)

**Symptoms:** Pod status shows CrashLoopBackOff, container exits with non-zero code

**Cause:** Application startup failure due to configuration error, missing secrets, or code bug.

**Resolution:**

1. **Check the crash reason:**

```bash
kubectl describe pod -n greenlang-alerting <pod-name> | grep -A10 "Last State"
kubectl logs -n greenlang-alerting <pod-name> --previous --tail=100
```

2. **If caused by missing secrets:**

```bash
# Verify secrets exist
kubectl get secret alerting-service-secrets -n greenlang-alerting

# Check ESO sync status
kubectl get externalsecrets -n greenlang-alerting

# If secrets are missing, check SSM parameters
aws ssm get-parameters-by-path --path "/gl/prod/alerting/" --query "Parameters[*].Name"
```

3. **If caused by config error:**

```bash
# Validate the configmap
kubectl get configmap alerting-service-config -n greenlang-alerting -o yaml

# Check for YAML syntax errors in the config
kubectl logs -n greenlang-alerting <pod-name> --previous | grep -i "config\|yaml\|parse"
```

4. **Restart the deployment after fixing:**

```bash
kubectl rollout restart deployment/alerting-service -n greenlang-alerting
kubectl rollout status deployment/alerting-service -n greenlang-alerting
```

### Scenario 2: ImagePullBackOff

**Symptoms:** Pod status shows ImagePullBackOff or ErrImagePull

**Cause:** Container image not found, registry authentication failure, or tag mismatch.

**Resolution:**

1. **Check the image and pull errors:**

```bash
kubectl describe pod -n greenlang-alerting <pod-name> | grep -A5 "Events"
```

2. **Verify image exists in registry:**

```bash
# Check current image tag
kubectl get deployment alerting-service -n greenlang-alerting -o jsonpath='{.spec.template.spec.containers[0].image}'

# Verify the image exists
aws ecr describe-images --repository-name greenlang/alerting-service --image-ids imageTag=latest
```

3. **Fix image tag if needed:**

```bash
kubectl set image deployment/alerting-service alerting-service=greenlang/alerting-service:<correct-tag> -n greenlang-alerting
```

### Scenario 3: Database Connectivity Failure

**Symptoms:** Logs show "connection refused" to PostgreSQL, init container stuck

**Cause:** PostgreSQL is down, network policy blocking traffic, or connection string incorrect.

**Resolution:**

1. **Verify PostgreSQL is running:**

```bash
kubectl get pods -n database -l app=postgresql
kubectl get svc -n database | grep postgresql
```

2. **Check network policies:**

```bash
kubectl get networkpolicy -n greenlang-alerting
kubectl describe networkpolicy alerting-service-egress -n greenlang-alerting
```

3. **Test connectivity from a debug pod:**

```bash
kubectl run debug --rm -it --image=postgres:14 -n greenlang-alerting --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" -c "SELECT 1"
```

4. **If PostgreSQL is down, follow PostgreSQL runbook and then restart alerting:**

```bash
# After PostgreSQL is restored:
kubectl rollout restart deployment/alerting-service -n greenlang-alerting
```

---

## Interim Mitigation

While the Alerting Service is being restored, ensure alert delivery is not completely blind:

1. **Check Alertmanager is still receiving alerts:**

```bash
# Access Alertmanager UI
kubectl port-forward -n monitoring svc/gl-prometheus-alertmanager 9093:9093
# Visit http://localhost:9093 to see active alerts
```

2. **Configure Alertmanager direct Slack/PagerDuty if needed:**

If the Alerting Service will be down for an extended period, temporarily configure Alertmanager's native receivers as a fallback (this bypasses the enriched routing and escalation logic).

3. **Monitor Alertmanager webhook errors:**

```promql
# Check Alertmanager webhook delivery failures
sum(rate(alertmanager_notifications_failed_total[5m])) by (integration)
```

---

## Prevention

### Monitoring

- **Dashboard:** Alerting Service (`/d/alerting-service`)
- **Alert:** `AlertingServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="alerting-service"}` (should always be >= 2)
  - `gl_alert_notifications_total` rate (should be non-zero during active alerts)
  - Pod restart count (should be 0)
  - Init container completion time (should be < 30s)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales to 6** for alert storm handling
4. **Resource requests** (250m CPU, 512Mi memory) are sized for normal load

### Configuration Best Practices

- Always validate ConfigMap YAML before applying changes
- Use ESO for secrets rotation to avoid manual secret management
- Test configuration changes in staging before production
- Maintain Alertmanager's native receivers as a fallback channel

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Observability Team
- **Review cadence:** Quarterly or after any P1 alerting incident
- **Related alerts:** `NotificationDeliveryFailing`, `PagerDutyIntegrationDown`
- **Related dashboards:** Alerting Service, Alertmanager Health
