# Service Availability Runbook

This runbook covers alerts related to service availability, health checks, and pod stability in the GreenLang platform.

---

## Table of Contents

- [ServiceDown](#servicedown)
- [APIEndpointUnhealthy](#apiendpointunhealthy)
- [HighPodRestartRate](#highpodrestartrate)

---

## ServiceDown

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | ServiceDown |
| **Severity** | Critical |
| **Team** | Platform |
| **Evaluation Interval** | 30s |
| **For Duration** | 1m |

**PromQL Expression:**

```promql
up{job=~"greenlang.*"} == 0
```

### Description

This alert fires when a GreenLang service has been unreachable for more than 1 minute. The `up` metric is a built-in Prometheus metric that indicates whether the last scrape was successful (1) or failed (0).

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | High | Users cannot access affected service functionality |
| **Data Impact** | Medium | Pending calculations may be delayed or lost |
| **SLA Impact** | Critical | Directly impacts 99.9% availability SLA |
| **Revenue Impact** | High | Service disruption affects customer operations |

### Diagnostic Steps

1. **Identify the affected service**

   ```bash
   # Check which service is down from alert labels
   echo "Affected service: {{ $labels.job }}"
   echo "Instance: {{ $labels.instance }}"
   ```

2. **Verify service status in Kubernetes**

   ```bash
   # List pods for the service
   kubectl get pods -n greenlang -l app={{ $labels.job }}

   # Check pod status details
   kubectl describe pod -n greenlang -l app={{ $labels.job }}
   ```

3. **Check recent events**

   ```bash
   # Get recent events in the namespace
   kubectl get events -n greenlang --sort-by='.lastTimestamp' | tail -20
   ```

4. **Review service logs**

   ```bash
   # Get logs from the affected service
   kubectl logs -n greenlang -l app={{ $labels.job }} --tail=100

   # If pod crashed, get previous container logs
   kubectl logs -n greenlang -l app={{ $labels.job }} --previous --tail=100
   ```

5. **Check network connectivity**

   ```bash
   # Verify service endpoint
   kubectl get endpoints -n greenlang {{ $labels.job }}

   # Test connectivity from within cluster
   kubectl run debug --rm -it --image=curlimages/curl -- \
     curl -v http://{{ $labels.job }}.greenlang.svc.cluster.local/health
   ```

6. **Verify dependencies**

   ```bash
   # Check database connectivity
   kubectl exec -n greenlang deploy/{{ $labels.job }} -- \
     pg_isready -h postgres.greenlang.svc.cluster.local

   # Check Redis connectivity
   kubectl exec -n greenlang deploy/{{ $labels.job }} -- \
     redis-cli -h redis.greenlang.svc.cluster.local ping
   ```

### Resolution Steps

#### Scenario 1: Pod in CrashLoopBackOff

```bash
# 1. Check crash reason
kubectl describe pod -n greenlang -l app={{ $labels.job }} | grep -A5 "Last State"

# 2. Check resource limits
kubectl describe pod -n greenlang -l app={{ $labels.job }} | grep -A10 "Limits"

# 3. If OOMKilled, increase memory limits
kubectl patch deployment -n greenlang {{ $labels.job }} \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"memory":"2Gi"}}}]}}}}'

# 4. Force restart if stuck
kubectl rollout restart deployment -n greenlang {{ $labels.job }}
```

#### Scenario 2: Pod not starting (ImagePullBackOff)

```bash
# 1. Check image details
kubectl describe pod -n greenlang -l app={{ $labels.job }} | grep -A5 "Events"

# 2. Verify image exists
docker pull {{ image_name }}

# 3. Check image pull secrets
kubectl get secrets -n greenlang | grep regcred
```

#### Scenario 3: Network issues

```bash
# 1. Restart the service
kubectl rollout restart deployment -n greenlang {{ $labels.job }}

# 2. Check network policies
kubectl get networkpolicies -n greenlang

# 3. Verify service definition
kubectl get svc -n greenlang {{ $labels.job }} -o yaml
```

#### Scenario 4: Dependency failure

```bash
# 1. Check database status
kubectl get pods -n greenlang -l app=postgres

# 2. Check Redis status
kubectl get pods -n greenlang -l app=redis

# 3. If dependency is down, escalate to database team
```

### Post-Resolution

1. **Verify service recovery**

   ```bash
   # Check that the up metric is now 1
   curl -s "http://prometheus:9090/api/v1/query?query=up{job='{{ $labels.job }}'}" | jq .

   # Verify health endpoint
   curl -s http://{{ $labels.job }}.greenlang.svc.cluster.local/health
   ```

2. **Check for data consistency**

   ```bash
   # Verify no pending jobs were lost
   kubectl exec -n greenlang deploy/{{ $labels.job }} -- \
     python -c "from app.monitoring import check_pending_jobs; check_pending_jobs()"
   ```

3. **Document the incident**
   - Record the root cause
   - Update runbook if new failure mode discovered
   - Create post-incident review if SLA was breached

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Platform Team | #platform-oncall Slack | If not resolved in 15 minutes |
| L3 | Platform Lead | Direct page | If not resolved in 30 minutes |
| L4 | Engineering Director | Phone call | If SLA breach imminent (>45 min) |

### Related Dashboards

- [System Health Dashboard](https://grafana.greenlang.io/d/system-health)
- [Service Overview](https://grafana.greenlang.io/d/service-overview)
- [Kubernetes Cluster Status](https://grafana.greenlang.io/d/k8s-cluster)

---

## APIEndpointUnhealthy

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | APIEndpointUnhealthy |
| **Severity** | Critical |
| **Team** | Platform |
| **Evaluation Interval** | 30s |
| **For Duration** | 2m |

**PromQL Expression:**

```promql
probe_success{job="blackbox-http"} == 0
```

### Description

This alert fires when the Blackbox Exporter health check probe fails for an API endpoint for more than 2 minutes. This indicates that the endpoint is not responding to HTTP requests properly, even if the underlying service pod is running.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | High | Users cannot reach the affected endpoint |
| **Data Impact** | Low | Requests are rejected but data is safe |
| **SLA Impact** | Critical | Contributes to availability SLA measurement |
| **Revenue Impact** | High | Blocked API calls affect integrations |

### Diagnostic Steps

1. **Identify the failing endpoint**

   ```bash
   # From alert annotations
   echo "Endpoint: {{ $labels.instance }}"
   ```

2. **Test the endpoint manually**

   ```bash
   # Direct HTTP test
   curl -v -w "\nHTTP Code: %{http_code}\nTime: %{time_total}s\n" \
     "{{ $labels.instance }}/health"

   # Test with timeout
   curl --connect-timeout 5 --max-time 10 "{{ $labels.instance }}/health"
   ```

3. **Check Blackbox Exporter details**

   ```bash
   # Query probe metrics
   curl -s "http://prometheus:9090/api/v1/query?query=probe_http_status_code{instance='{{ $labels.instance }}'}" | jq .

   # Check probe duration
   curl -s "http://prometheus:9090/api/v1/query?query=probe_duration_seconds{instance='{{ $labels.instance }}'}" | jq .
   ```

4. **Review ingress/load balancer**

   ```bash
   # Check ingress status
   kubectl get ingress -n greenlang

   # Check ingress controller logs
   kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx --tail=100
   ```

5. **Check TLS certificate**

   ```bash
   # Verify certificate validity
   echo | openssl s_client -connect {{ $labels.instance }}:443 2>/dev/null | \
     openssl x509 -noout -dates
   ```

### Resolution Steps

#### Scenario 1: Service returning 5xx errors

```bash
# 1. Check application logs for errors
kubectl logs -n greenlang -l app=api-gateway --tail=200 | grep -i error

# 2. Restart the service
kubectl rollout restart deployment -n greenlang api-gateway

# 3. Monitor recovery
watch kubectl get pods -n greenlang -l app=api-gateway
```

#### Scenario 2: TLS/Certificate issues

```bash
# 1. Check certificate expiry
kubectl get secret -n greenlang tls-secret -o jsonpath='{.data.tls\.crt}' | \
  base64 -d | openssl x509 -noout -enddate

# 2. Renew certificate if expired (cert-manager)
kubectl delete certificate -n greenlang api-tls
kubectl apply -f k8s/certificates/api-tls.yaml

# 3. Force cert-manager to renew
kubectl annotate certificate -n greenlang api-tls \
  cert-manager.io/issuer-kind=ClusterIssuer --overwrite
```

#### Scenario 3: Ingress misconfiguration

```bash
# 1. Verify ingress configuration
kubectl get ingress -n greenlang api-ingress -o yaml

# 2. Check backend service
kubectl describe ingress -n greenlang api-ingress

# 3. Reapply ingress if needed
kubectl apply -f k8s/ingress/api-ingress.yaml
```

#### Scenario 4: Load balancer health check failure

```bash
# 1. Check cloud load balancer status (AWS example)
aws elbv2 describe-target-health --target-group-arn <arn>

# 2. Verify security groups
aws ec2 describe-security-groups --group-ids <sg-id>
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Platform Team | #platform-oncall Slack | If not resolved in 10 minutes |
| L3 | Network Team | #network-oncall Slack | If ingress/LB issues suspected |
| L4 | Engineering Director | Phone call | If customer-facing for >20 min |

---

## HighPodRestartRate

### Alert Details

| Property | Value |
|----------|-------|
| **Alert Name** | HighPodRestartRate |
| **Severity** | Warning |
| **Team** | Platform |
| **Evaluation Interval** | 30s |
| **For Duration** | 5m |

**PromQL Expression:**

```promql
increase(kube_pod_container_status_restarts_total{namespace="greenlang"}[1h]) > 5
```

### Description

This alert fires when a pod has restarted more than 5 times in the last hour. Frequent restarts indicate instability that could lead to service degradation or outage.

### Impact Assessment

| Impact Area | Severity | Description |
|-------------|----------|-------------|
| **User Impact** | Medium | Intermittent failures during restarts |
| **Data Impact** | Medium | In-flight requests may be lost during restart |
| **SLA Impact** | Low | Brief interruptions during restart cycles |
| **Revenue Impact** | Low | Degraded experience but not full outage |

### Diagnostic Steps

1. **Identify restart reason**

   ```bash
   # Get restart count and reason
   kubectl get pods -n greenlang {{ $labels.pod }} -o jsonpath='{.status.containerStatuses[*].restartCount}'

   # Check last termination reason
   kubectl get pods -n greenlang {{ $labels.pod }} -o jsonpath='{.status.containerStatuses[*].lastState.terminated.reason}'
   ```

2. **Review pod events**

   ```bash
   # Get pod events
   kubectl describe pod -n greenlang {{ $labels.pod }} | grep -A20 "Events:"
   ```

3. **Check resource usage before crash**

   ```bash
   # Query memory usage
   curl -s "http://prometheus:9090/api/v1/query_range?query=container_memory_usage_bytes{pod='{{ $labels.pod }}'}&start=$(date -d '1 hour ago' +%s)&end=$(date +%s)&step=60" | jq .

   # Query CPU usage
   curl -s "http://prometheus:9090/api/v1/query_range?query=rate(container_cpu_usage_seconds_total{pod='{{ $labels.pod }}'}[5m])&start=$(date -d '1 hour ago' +%s)&end=$(date +%s)&step=60" | jq .
   ```

4. **Review application logs**

   ```bash
   # Get current logs
   kubectl logs -n greenlang {{ $labels.pod }} --tail=200

   # Get previous container logs (before crash)
   kubectl logs -n greenlang {{ $labels.pod }} --previous --tail=200
   ```

5. **Check liveness/readiness probes**

   ```bash
   # Get probe configuration
   kubectl get pod -n greenlang {{ $labels.pod }} -o jsonpath='{.spec.containers[*].livenessProbe}'
   kubectl get pod -n greenlang {{ $labels.pod }} -o jsonpath='{.spec.containers[*].readinessProbe}'
   ```

### Resolution Steps

#### Scenario 1: OOMKilled (Out of Memory)

```bash
# 1. Confirm OOM kill
kubectl describe pod -n greenlang {{ $labels.pod }} | grep -i oomkilled

# 2. Check current limits
kubectl get pod -n greenlang {{ $labels.pod }} -o jsonpath='{.spec.containers[*].resources}'

# 3. Increase memory limits
kubectl patch deployment -n greenlang {{ deployment_name }} \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"memory":"4Gi"},"requests":{"memory":"2Gi"}}}]}}}}'

# 4. Investigate memory leak if recurring
kubectl exec -n greenlang {{ $labels.pod }} -- python -c "import tracemalloc; tracemalloc.start()"
```

#### Scenario 2: Liveness probe failures

```bash
# 1. Check probe configuration
kubectl get deployment -n greenlang {{ deployment_name }} -o yaml | grep -A10 livenessProbe

# 2. Adjust probe timing if needed
kubectl patch deployment -n greenlang {{ deployment_name }} \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","livenessProbe":{"initialDelaySeconds":60,"periodSeconds":30,"timeoutSeconds":10,"failureThreshold":5}}]}}}}'

# 3. Verify health endpoint is fast enough
kubectl exec -n greenlang {{ $labels.pod }} -- \
  curl -w "\nTime: %{time_total}s\n" http://localhost:8000/health
```

#### Scenario 3: Application crash (unhandled exception)

```bash
# 1. Review crash logs for stack trace
kubectl logs -n greenlang {{ $labels.pod }} --previous | grep -A20 "Traceback"

# 2. Check if specific input causing crash
kubectl logs -n greenlang {{ $labels.pod }} --previous | grep -B10 "Traceback"

# 3. If bug identified, notify development team
# Create incident ticket with logs
```

#### Scenario 4: Dependency connection failures

```bash
# 1. Check database connectivity
kubectl exec -n greenlang {{ $labels.pod }} -- \
  pg_isready -h postgres.greenlang.svc.cluster.local

# 2. Check Redis connectivity
kubectl exec -n greenlang {{ $labels.pod }} -- \
  redis-cli -h redis.greenlang.svc.cluster.local ping

# 3. Verify DNS resolution
kubectl exec -n greenlang {{ $labels.pod }} -- \
  nslookup postgres.greenlang.svc.cluster.local
```

### Escalation Path

| Level | Team | Contact | When to Escalate |
|-------|------|---------|------------------|
| L1 | On-Call Engineer | PagerDuty | Initial response |
| L2 | Backend Team | #backend-oncall Slack | If application bug suspected |
| L3 | Platform Team | #platform-oncall Slack | If infrastructure issue |

### Related Alerts

- [ServiceDown](#servicedown) - May fire if restarts become too frequent
- [HighMemoryUsage](./resources.md#highmemoryusage) - Often precedes OOMKilled restarts

---

## Quick Reference Card

| Alert | Severity | First Check | Quick Fix |
|-------|----------|-------------|-----------|
| ServiceDown | Critical | `kubectl get pods -n greenlang` | `kubectl rollout restart deployment` |
| APIEndpointUnhealthy | Critical | `curl -v <endpoint>/health` | Check ingress and TLS |
| HighPodRestartRate | Warning | `kubectl describe pod` | Check OOM/probe config |
