# Batch Job Metrics Stale

## Alert

**Alert Name:** `BatchJobStale`

**Severity:** Warning

**Threshold:** `time() - gl_batch_job_last_success_timestamp > 3600` (1 hour since last success)

**Duration:** 15 minutes

---

## Description

This alert fires when batch job metrics in the PushGateway become stale, indicating:

1. **Job not running** - CronJob or scheduled job hasn't executed
2. **Push failed** - Job ran but couldn't push metrics to PushGateway
3. **PushGateway issues** - PushGateway is down or unreachable
4. **Job failed** - Job ran but failed to complete successfully

Stale metrics mean we lose visibility into critical batch processes like:
- Emission factor updates
- Data synchronization jobs
- Report generation
- Database maintenance tasks

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Stale data, outdated reports |
| **Data Impact** | Medium | Data pipelines may be broken |
| **SLA Impact** | Low-Medium | Depends on job criticality |
| **Revenue Impact** | Low | May affect compliance reporting |

---

## Symptoms

- `gl_batch_job_last_success_timestamp` hasn't updated
- No recent metrics for specific batch jobs
- PushGateway shows old job metrics
- CronJob not creating new pods
- Job pods in Failed or Error state

---

## Diagnostic Steps

### Step 1: Check Batch Job Metrics in PushGateway

```promql
# Last success timestamp for all batch jobs
gl_batch_job_last_success_timestamp

# Calculate staleness
time() - gl_batch_job_last_success_timestamp

# Job duration
gl_batch_job_duration_seconds

# Job errors
increase(gl_batch_job_errors_total[24h])
```

### Step 2: Check PushGateway Status

```promql
# PushGateway is up
up{job="pushgateway"}

# Metrics pushed to PushGateway
push_time_seconds
push_failure_time_seconds
```

```bash
# Check PushGateway pods
kubectl get pods -n monitoring -l app.kubernetes.io/name=pushgateway

# Check PushGateway logs
kubectl logs -n monitoring -l app.kubernetes.io/name=pushgateway --tail=100
```

### Step 3: Check CronJob Status

```bash
# List all CronJobs
kubectl get cronjobs -n greenlang

# Check specific CronJob
kubectl describe cronjob -n greenlang <job-name>

# Check last scheduled time
kubectl get cronjob -n greenlang <job-name> -o jsonpath='{.status.lastScheduleTime}'
```

### Step 4: Check Job Pods

```bash
# List recent jobs
kubectl get jobs -n greenlang --sort-by=.metadata.creationTimestamp

# Check job pod status
kubectl get pods -n greenlang -l job-name=<job-name>

# Get job logs
kubectl logs -n greenlang -l job-name=<job-name>
```

### Step 5: Check Job Configuration

```bash
# View CronJob YAML
kubectl get cronjob -n greenlang <job-name> -o yaml

# Check schedule
kubectl get cronjob -n greenlang <job-name> -o jsonpath='{.spec.schedule}'

# Check suspend status
kubectl get cronjob -n greenlang <job-name> -o jsonpath='{.spec.suspend}'
```

### Step 6: Test PushGateway Connectivity from Job Namespace

```bash
# Create a test pod
kubectl run test-push -n greenlang --image=curlimages/curl --rm -it -- /bin/sh

# Test PushGateway connectivity
curl -v http://pushgateway.monitoring.svc:9091/metrics
```

---

## Resolution Steps

### Scenario 1: CronJob Not Running

**Symptoms:** No recent job pods, CronJob shows old lastScheduleTime

**Resolution:**

1. **Check if CronJob is suspended:**

```bash
kubectl get cronjob -n greenlang <job-name> -o jsonpath='{.spec.suspend}'
```

2. **Resume suspended CronJob:**

```bash
kubectl patch cronjob -n greenlang <job-name> -p '{"spec":{"suspend":false}}'
```

3. **Check CronJob schedule syntax:**

```bash
# Valid cron expressions: "*/5 * * * *" (every 5 min), "0 */4 * * *" (every 4 hours)
kubectl get cronjob -n greenlang <job-name> -o jsonpath='{.spec.schedule}'
```

4. **Manually trigger a job run:**

```bash
kubectl create job --from=cronjob/<job-name> <job-name>-manual-$(date +%s) -n greenlang
```

### Scenario 2: Job Pods Failing

**Symptoms:** Job pods in Error or CrashLoopBackOff

**Resolution:**

1. **Check job pod logs:**

```bash
kubectl logs -n greenlang -l job-name=<job-name> --tail=200
```

2. **Check for resource issues:**

```bash
kubectl describe job -n greenlang <job-name> | grep -A 10 "Resources"
```

3. **Fix common issues:**

```yaml
# Increase resources if OOMKilled
spec:
  template:
    spec:
      containers:
        - name: job
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
```

4. **Check for dependency failures:**

```bash
# Check if required services are available
kubectl exec -n greenlang <job-pod> -- curl -v http://database.greenlang.svc:5432
kubectl exec -n greenlang <job-pod> -- curl -v http://api.greenlang.svc:8080/health
```

### Scenario 3: PushGateway Unreachable

**Symptoms:** Jobs complete but metrics not pushed

**Resolution:**

1. **Check PushGateway service:**

```bash
kubectl get svc -n monitoring pushgateway
kubectl get endpoints -n monitoring pushgateway
```

2. **Check network policies:**

```bash
kubectl get networkpolicy -n greenlang -o yaml | grep -A 20 egress
```

3. **Add network policy for PushGateway access:**

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-pushgateway
  namespace: greenlang
spec:
  podSelector:
    matchLabels:
      app: batch-jobs
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
          podSelector:
            matchLabels:
              app.kubernetes.io/name: pushgateway
      ports:
        - protocol: TCP
          port: 9091
```

4. **Verify connectivity after fix:**

```bash
kubectl run test-push -n greenlang --image=curlimages/curl --rm -it -- \
  curl -X POST http://pushgateway.monitoring.svc:9091/metrics/job/test
```

### Scenario 4: PushGateway Down

**Symptoms:** PushGateway pods not running

**Resolution:**

1. **Check PushGateway deployment:**

```bash
kubectl get deployment -n monitoring pushgateway
kubectl describe deployment -n monitoring pushgateway
```

2. **Restart PushGateway:**

```bash
kubectl rollout restart deployment -n monitoring pushgateway
```

3. **Check PushGateway PVC if persistence enabled:**

```bash
kubectl get pvc -n monitoring | grep pushgateway
kubectl describe pvc -n monitoring pushgateway-data
```

### Scenario 5: Job Code Not Pushing Metrics

**Symptoms:** Job completes successfully but no metrics appear

**Resolution:**

1. **Verify job code uses PushGateway client:**

```python
# Expected pattern in job code
from greenlang.monitoring.pushgateway import BatchJobMetrics

metrics = BatchJobMetrics(
    job_name="my-batch-job",
    pushgateway_url="http://pushgateway.monitoring.svc:9091"
)

with metrics.track_duration():
    # Job logic here
    pass

metrics.push()  # Ensure push is called
```

2. **Add debug logging:**

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Push with debug
try:
    metrics.push()
    logging.info("Metrics pushed successfully")
except Exception as e:
    logging.error(f"Failed to push metrics: {e}")
```

3. **Check environment variables:**

```bash
# Job should have PUSHGATEWAY_URL set
kubectl get cronjob -n greenlang <job-name> -o jsonpath='{.spec.jobTemplate.spec.template.spec.containers[0].env}'
```

### Scenario 6: Stale Metrics Not Cleaned Up

**Symptoms:** Old metrics from deleted jobs still showing

**Resolution:**

1. **Delete stale metrics from PushGateway:**

```bash
# Delete metrics for a specific job
curl -X DELETE http://pushgateway.monitoring.svc:9091/metrics/job/<job_name>

# Delete metrics with specific grouping key
curl -X DELETE http://pushgateway.monitoring.svc:9091/metrics/job/<job_name>/instance/<instance>
```

2. **Enable automatic metric expiration:**

```yaml
# PushGateway doesn't support TTL natively
# Use a CronJob to clean up stale metrics

apiVersion: batch/v1
kind: CronJob
metadata:
  name: pushgateway-cleanup
  namespace: monitoring
spec:
  schedule: "0 * * * *"  # Every hour
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: cleanup
              image: curlimages/curl
              command:
                - /bin/sh
                - -c
                - |
                  # Delete metrics older than 2 hours
                  # This is a simplified example - production would need more logic
                  curl -X DELETE http://pushgateway:9091/metrics/job/old-job
          restartPolicy: Never
```

---

## Job-Specific Runbooks

### Emission Factor Update Job

| Setting | Value |
|---------|-------|
| CronJob Name | `ef-update-job` |
| Schedule | `0 */6 * * *` (every 6 hours) |
| Metric | `gl_batch_job_last_success_timestamp{job_name="ef-update"}` |
| Max Staleness | 8 hours |

### Data Sync Job

| Setting | Value |
|---------|-------|
| CronJob Name | `data-sync-job` |
| Schedule | `0 2 * * *` (daily at 2 AM) |
| Metric | `gl_batch_job_last_success_timestamp{job_name="data-sync"}` |
| Max Staleness | 26 hours |

---

## Emergency Actions

### If Critical Batch Job Is Stale

1. **Manually trigger the job:**

```bash
kubectl create job --from=cronjob/<job-name> <job-name>-emergency-$(date +%s) -n greenlang
```

2. **Monitor job execution:**

```bash
kubectl logs -n greenlang -l job-name=<job-name>-emergency -f
```

3. **Verify metrics pushed:**

```bash
curl http://pushgateway.monitoring.svc:9091/metrics | grep <job_name>
```

---

## Escalation Path

| Level | Condition | Contact |
|-------|-----------|---------|
| L1 | Non-critical job stale | On-call engineer |
| L2 | Critical job stale >2x schedule | Platform team lead |
| L3 | Multiple critical jobs failing | Platform team + Data team |

---

## Prevention

1. **Monitor job freshness:**

```promql
# Alert when job hasn't run in expected time
(time() - gl_batch_job_last_success_timestamp{job_name="critical-job"}) > 7200
```

2. **Set up job completion alerts:**

```yaml
# In job code
metrics.job_status.labels(job_name=self.job_name, status="completed").set(1)
```

3. **Implement retry logic:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=60))
def push_metrics():
    metrics.push()
```

4. **Regular job health checks:**
   - Review CronJob schedules monthly
   - Test job execution in staging
   - Monitor job duration trends

---

## Related Dashboards

| Dashboard | URL |
|-----------|-----|
| Batch Jobs | https://grafana.greenlang.io/d/batch-jobs |
| PushGateway | https://grafana.greenlang.io/d/pushgateway |
| CronJobs | https://grafana.greenlang.io/d/cronjobs |

---

## Related Alerts

- `PushGatewayDown`
- `CronJobFailed`
- `BatchJobDurationHigh`
- `BatchJobErrorsHigh`

---

## References

- [Prometheus PushGateway](https://prometheus.io/docs/instrumenting/pushing/)
- [Kubernetes CronJobs](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
- [GreenLang BatchJobMetrics SDK](../development/metrics-guide.md#batch-jobs)
