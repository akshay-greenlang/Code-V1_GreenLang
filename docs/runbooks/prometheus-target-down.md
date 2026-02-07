# Prometheus Target Down

## Alert

**Alert Name:** `PrometheusTargetMissing`

**Severity:** Critical

**Threshold:** `up == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when Prometheus cannot successfully scrape a target for 5 minutes. The `up` metric is set to 0 when a scrape fails, indicating:

1. **Target is down** - The pod/service is not running
2. **Network issue** - Network policies blocking access
3. **Wrong endpoint** - ServiceMonitor misconfigured
4. **Timeout** - Target too slow to respond
5. **Authentication failure** - TLS or auth misconfiguration

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | No visibility into service health |
| **Data Impact** | High | Metrics gap during outage |
| **SLA Impact** | Medium | May miss SLA violations |
| **Revenue Impact** | Low-High | Depends on which target is down |

---

## Symptoms

- `up{job="<target_job>"}` metric is 0
- Gaps in Grafana dashboards for affected service
- No alerts firing for the affected service
- ServiceMonitor showing target as down in Prometheus UI

---

## Diagnostic Steps

### Step 1: Identify the Affected Target

```promql
# Find all down targets
up == 0

# Get target details
up{job="<job_name>"} == 0

# Check last scrape time
scrape_duration_seconds{job="<job_name>"}

# Check scrape failures
increase(scrape_samples_scraped{job="<job_name>"}[10m]) == 0
```

### Step 2: Check Prometheus Targets UI

```bash
# Port-forward to Prometheus UI
kubectl port-forward -n monitoring svc/prometheus-server 9090:9090

# Open http://localhost:9090/targets
# Look for targets with "DOWN" state
```

### Step 3: Check Target Pod Status

```bash
# Get pod status for the service
kubectl get pods -n <namespace> -l app=<service_name>

# Describe pod for events
kubectl describe pod -n <namespace> <pod_name>

# Check pod logs
kubectl logs -n <namespace> <pod_name> --tail=100
```

### Step 4: Check Service and Endpoints

```bash
# Check service exists
kubectl get svc -n <namespace> <service_name>

# Check endpoints
kubectl get endpoints -n <namespace> <service_name>

# Describe service for selector
kubectl describe svc -n <namespace> <service_name>
```

### Step 5: Check ServiceMonitor Configuration

```bash
# Get ServiceMonitor
kubectl get servicemonitor -n monitoring <servicemonitor_name> -o yaml

# Verify selector matches service labels
kubectl get svc -n <namespace> <service_name> -o jsonpath='{.metadata.labels}'
```

### Step 6: Check Network Policies

```bash
# List network policies
kubectl get networkpolicy -n <namespace>

# Check if monitoring namespace can access target
kubectl get networkpolicy -n <namespace> -o yaml | grep -A 20 "ingress"
```

### Step 7: Test Connectivity from Prometheus

```bash
# Exec into Prometheus pod
kubectl exec -it -n monitoring prometheus-server-0 -- /bin/sh

# Test connectivity to target
wget -O- http://<service_name>.<namespace>.svc:<port>/metrics

# Or use curl if available
curl -v http://<service_name>.<namespace>.svc:<port>/metrics
```

---

## Resolution Steps

### Scenario 1: Target Pod is Down

**Symptoms:** Pod not running or in CrashLoopBackOff

**Resolution:**

1. **Check pod events:**

```bash
kubectl describe pod -n <namespace> <pod_name>
```

2. **Check pod logs for crash reason:**

```bash
kubectl logs -n <namespace> <pod_name> --previous
```

3. **Restart the deployment:**

```bash
kubectl rollout restart deployment -n <namespace> <deployment_name>
```

4. **If OOMKilled, increase resources:**

```bash
kubectl patch deployment -n <namespace> <deployment_name> -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "<container_name>",
            "resources": {
              "limits": {
                "memory": "1Gi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

### Scenario 2: Network Policy Blocking Access

**Symptoms:** Pod is running but Prometheus cannot reach it

**Resolution:**

1. **Check existing network policies:**

```bash
kubectl get networkpolicy -n <namespace> -o yaml
```

2. **Create or update network policy to allow Prometheus:**

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-prometheus-scrape
  namespace: <namespace>
spec:
  podSelector:
    matchLabels:
      app: <service_name>
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
          podSelector:
            matchLabels:
              app.kubernetes.io/name: prometheus
      ports:
        - protocol: TCP
          port: <metrics_port>
```

3. **Apply network policy:**

```bash
kubectl apply -f network-policy.yaml
```

### Scenario 3: ServiceMonitor Misconfigured

**Symptoms:** Service is up but not being scraped

**Resolution:**

1. **Check ServiceMonitor selector matches service labels:**

```bash
# Get service labels
kubectl get svc -n <namespace> <service_name> -o jsonpath='{.metadata.labels}'

# Compare with ServiceMonitor selector
kubectl get servicemonitor -n monitoring <name> -o jsonpath='{.spec.selector.matchLabels}'
```

2. **Update ServiceMonitor if needed:**

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: <service_name>
  namespace: monitoring
  labels:
    release: prometheus  # Required for Prometheus Operator
spec:
  selector:
    matchLabels:
      app: <service_name>  # Must match service labels
  namespaceSelector:
    matchNames:
      - <namespace>
  endpoints:
    - port: metrics       # Must match service port name
      interval: 15s
      path: /metrics
```

3. **Apply and verify:**

```bash
kubectl apply -f servicemonitor.yaml

# Wait 30 seconds for Prometheus to reload config
kubectl logs -n monitoring prometheus-server-0 | grep "Completed loading"
```

### Scenario 4: Wrong Metrics Endpoint

**Symptoms:** 404 or connection refused on metrics endpoint

**Resolution:**

1. **Verify metrics endpoint on the service:**

```bash
# Port-forward to the service
kubectl port-forward -n <namespace> svc/<service_name> 8080:<port>

# Test metrics endpoint
curl http://localhost:8080/metrics
```

2. **Check if the application exposes metrics:**

```bash
# Common metrics paths
curl http://localhost:8080/metrics
curl http://localhost:8080/actuator/prometheus  # Spring Boot
curl http://localhost:8080/q/metrics            # Quarkus
```

3. **Update ServiceMonitor with correct path:**

```yaml
endpoints:
  - port: http
    path: /actuator/prometheus  # Correct path
    interval: 15s
```

### Scenario 5: Scrape Timeout

**Symptoms:** Scrape fails with timeout error

**Resolution:**

1. **Check current timeout:**

```yaml
endpoints:
  - port: metrics
    interval: 15s
    scrapeTimeout: 10s  # Default is often too short
```

2. **Increase timeout if metrics endpoint is slow:**

```yaml
endpoints:
  - port: metrics
    interval: 30s
    scrapeTimeout: 25s  # Must be less than interval
```

3. **Investigate why metrics endpoint is slow:**
   - Too many metrics being generated
   - Database queries in metrics collection
   - CPU/memory constraints

### Scenario 6: TLS/Authentication Issues

**Symptoms:** Connection errors or 401/403 responses

**Resolution:**

1. **Check if target requires TLS:**

```yaml
endpoints:
  - port: metrics
    scheme: https
    tlsConfig:
      insecureSkipVerify: true  # For testing only
      # Or use proper CA
      ca:
        secret:
          name: target-ca-cert
          key: ca.crt
```

2. **Check if target requires authentication:**

```yaml
endpoints:
  - port: metrics
    basicAuth:
      username:
        name: metrics-auth-secret
        key: username
      password:
        name: metrics-auth-secret
        key: password
```

---

## Emergency Actions

### If Critical Service is Not Being Monitored

1. **Create temporary scrape config:**

```bash
# Edit Prometheus ConfigMap directly
kubectl edit configmap -n monitoring prometheus-server

# Add under scrape_configs:
- job_name: 'emergency-scrape'
  static_configs:
    - targets: ['<service>.<namespace>.svc:<port>']
```

2. **Reload Prometheus:**

```bash
curl -X POST http://prometheus.monitoring.svc:9090/-/reload
```

---

## Escalation Path

| Level | Condition | Contact |
|-------|-----------|---------|
| L1 | Non-critical target down | On-call engineer |
| L2 | Critical service (API, agents) down | Platform team lead |
| L3 | Multiple targets down, potential cluster issue | Platform team + SRE |

---

## Prevention

1. **Standardize metrics endpoints:**
   - All services expose `/metrics` on port 8080
   - Document any exceptions

2. **Include network policies in deployments:**
   - Always allow Prometheus ingress for monitored services

3. **Test ServiceMonitors before merging:**
   - Use `kubectl apply --dry-run=server`
   - Verify selector matches service labels

4. **Monitor scrape success rate:**

```promql
# Alert if scrape success rate drops
(sum(up) / count(up)) < 0.95
```

---

## Related Dashboards

| Dashboard | URL |
|-----------|-----|
| Prometheus Targets | https://grafana.greenlang.io/d/prometheus-targets |
| Service Health | https://grafana.greenlang.io/d/service-health |

---

## Related Alerts

- `PrometheusConfigReloadFailed`
- `PrometheusNotConnectedToAlertmanager`
- `ServiceDown`

---

## References

- [ServiceMonitor Specification](https://github.com/prometheus-operator/prometheus-operator/blob/main/Documentation/api.md#servicemonitor)
- [Prometheus Scraping](https://prometheus.io/docs/prometheus/latest/configuration/configuration/#scrape_config)
- [Kubernetes Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
