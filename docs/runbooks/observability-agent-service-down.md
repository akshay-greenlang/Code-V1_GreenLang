# Observability Agent Service Down

## Alert

**Alert Name:** `ObsAgentDown`

**Severity:** Critical

**Threshold:** `up{job="observability-agent-service"} == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang Observability & Telemetry Agent (AGENT-FOUND-010) are running. The Observability Agent is the self-monitoring and telemetry aggregation layer for the entire GreenLang Climate OS platform. It is responsible for:

1. **Metrics collection** -- Recording and forwarding Prometheus metrics from all GreenLang agents (47+ services) to the monitoring backend, tracked via `gl_obs_metrics_recorded_total`
2. **Distributed tracing** -- Creating and managing OpenTelemetry spans for request tracing across the agent mesh, tracked via `gl_obs_spans_created_total` and `gl_obs_spans_active`
3. **Log ingestion** -- Aggregating structured logs from all services and forwarding them to Loki, tracked via `gl_obs_logs_ingested_total`
4. **Alert evaluation** -- Evaluating alert rules against live metrics and managing alert lifecycle (pending, firing, resolved), tracked via `gl_obs_alerts_evaluated_total` and `gl_obs_alerts_firing`
5. **Health checking** -- Performing deep health checks across all dependent services (Prometheus, Tempo, Loki, Grafana, PostgreSQL, Redis), tracked via `gl_obs_health_checks_total` and `gl_obs_health_status`
6. **SLO compliance tracking** -- Computing SLO compliance ratios and error budget consumption for all GreenLang services, tracked via `gl_obs_slo_compliance_ratio` and `gl_obs_error_budget_remaining`
7. **Dashboard query serving** -- Serving pre-computed dashboard queries for Grafana panels, tracked via `gl_obs_dashboard_queries_total`
8. **Operation duration tracking** -- Recording histogram latency for all observability operations, tracked via `gl_obs_operation_duration_seconds`

When the Observability Agent is down:
- **Platform visibility is lost** and operators cannot monitor the health of GreenLang services
- **Alert evaluation stops** and critical conditions across all agents will not trigger notifications
- **SLO compliance tracking halts** and error budget consumption cannot be measured
- **Distributed tracing is disrupted** and request traces across the agent mesh will have gaps
- **Log aggregation stops** and structured logs will not be forwarded to Loki
- **Dashboard queries return stale data** and Grafana panels will show outdated or missing information
- **Health check coverage is lost** and degraded backend services may go undetected

**Note:** All telemetry data (metrics, traces, logs) is stored in dedicated backends (Prometheus/Thanos, Tempo, Loki) and is not affected by an Observability Agent outage. Prometheus continues to scrape targets independently. Once the agent recovers, it will resume all operations. No historical data is lost during an outage, but real-time alerting and SLO tracking will have gaps.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Platform operators lose real-time visibility into all GreenLang services |
| **Data Impact** | Low | Telemetry backends continue independently; agent state in PostgreSQL persists |
| **SLA Impact** | High | SLO compliance and error budget tracking halted; alerting gaps |
| **Revenue Impact** | Medium | Compliance-sensitive customers require continuous observability guarantees |
| **Compliance Impact** | High | SOC 2 requires continuous monitoring; audit trail gap during outage |
| **Downstream Impact** | Critical | All 47+ GreenLang agents lose centralized alert evaluation and SLO tracking |

---

## Symptoms

- `up{job="observability-agent-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=observability-agent-service`
- `gl_obs_metrics_recorded_total` counter stops incrementing
- `gl_obs_spans_created_total` counter stops incrementing
- `gl_obs_logs_ingested_total` counter stops incrementing
- `gl_obs_alerts_evaluated_total` counter stops incrementing
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /health` is unreachable
- Grafana Observability Agent dashboard shows "No Data" or stale timestamps
- SLO compliance panels show flat lines or missing data points
- Alert notifications stop being generated for other GreenLang services

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List observability agent service pods
kubectl get pods -n greenlang -l app=observability-agent-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=observability-agent-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to the observability agent
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=observability-agent-service | tail -30

# Check deployment status
kubectl describe deployment observability-agent-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment observability-agent-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=observability-agent-service

# Check for rollout issues
kubectl rollout status deployment/observability-agent-service -n greenlang

# Check HPA status (scales 2-10 replicas)
kubectl get hpa -n greenlang -l app=observability-agent-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=observability-agent-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=observability-agent-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for observability-specific errors
kubectl logs -n greenlang -l app=observability-agent-service --tail=500 \
  | grep -i "prometheus\|tempo\|loki\|grafana\|metric\|span\|trace\|alert\|slo"

# Look for database connection errors
kubectl logs -n greenlang -l app=observability-agent-service --tail=500 \
  | grep -i "database\|postgres\|timescale\|connection\|pool\|migration"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage
kubectl top pods -n greenlang -l app=observability-agent-service

# Check if pods were OOMKilled
kubectl get pods -n greenlang -l app=observability-agent-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Check node resource availability
kubectl top nodes
```

### Step 5: Check Database Connectivity

```bash
# Verify PostgreSQL connectivity
kubectl run pg-test --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  pg_isready -h greenlang-db.postgres.svc.cluster.local -p 5432

# Check database connection pool status in logs
kubectl logs -n greenlang -l app=observability-agent-service --tail=200 \
  | grep -i "pool\|connection\|database\|postgres"

# Check if the observability_agent_service schema exists
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_schema='observability_agent_service'
   ORDER BY table_name;"
```

### Step 6: Check Redis Connectivity

```bash
# Verify Redis connectivity
kubectl run redis-test --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 PING

# Check Redis connection in logs
kubectl logs -n greenlang -l app=observability-agent-service --tail=200 \
  | grep -i "redis\|cache\|session"
```

### Step 7: Check Telemetry Backend Connectivity

```bash
# Verify Prometheus connectivity
kubectl run prom-test --rm -it --image=curlimages/curl:latest -n greenlang --restart=Never -- \
  curl -sf http://prometheus-server.monitoring.svc.cluster.local:9090/-/healthy

# Verify Tempo (tracing backend) connectivity
kubectl run tempo-test --rm -it --image=curlimages/curl:latest -n greenlang --restart=Never -- \
  curl -sf http://tempo.monitoring.svc.cluster.local:3200/ready

# Verify Loki (log backend) connectivity
kubectl run loki-test --rm -it --image=curlimages/curl:latest -n greenlang --restart=Never -- \
  curl -sf http://loki.monitoring.svc.cluster.local:3100/ready
```

### Step 8: Check ConfigMap and Secrets

```bash
# Verify the service ConfigMap exists and is valid
kubectl get configmap observability-agent-service-config -n greenlang
kubectl get configmap observability-agent-service-config -n greenlang -o yaml | head -50

# Verify secrets exist
kubectl get secret observability-agent-service-secrets -n greenlang

# Check environment variables are set correctly
kubectl get deployment observability-agent-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env[*].name}' | tr ' ' '\n' | sort
```

### Step 9: Check Network Policies

```bash
# Check network policies affecting the observability agent
kubectl get networkpolicy -n greenlang | grep observability-agent

# Verify the service can reach PostgreSQL
kubectl run net-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-db.postgres.svc.cluster.local 5432'

# Verify the service can reach Prometheus
kubectl run net-test-2 --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv prometheus-server.monitoring.svc.cluster.local 9090'

# Verify upstream services can reach the observability agent
kubectl run net-test-3 --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv observability-agent-service.greenlang.svc.cluster.local 8080'
```

---

## Resolution Steps

### Scenario 1: OOMKilled (Out of Memory)

**Symptoms:** Pod status shows OOMKilled, container exits with code 137.

**Resolution:**

1. Confirm the OOM cause:
```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl get events -n greenlang --field-selector reason=OOMKilling --sort-by='.lastTimestamp'
```

2. Increase memory limits:
```bash
kubectl patch deployment observability-agent-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "observability-agent-service",
            "resources": {
              "limits": {
                "cpu": "1",
                "memory": "1Gi"
              },
              "requests": {
                "cpu": "250m",
                "memory": "512Mi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

3. Verify pods restart successfully:
```bash
kubectl rollout status deployment/observability-agent-service -n greenlang
kubectl get pods -n greenlang -l app=observability-agent-service
```

### Scenario 2: CrashLoopBackOff -- Telemetry Backend Unreachable

**Symptoms:** Pod status shows CrashLoopBackOff, logs show connection errors to Prometheus, Tempo, or Loki.

**Resolution:**

1. Check telemetry backend health:
```bash
kubectl get pods -n monitoring -l app=prometheus
kubectl get pods -n monitoring -l app=tempo
kubectl get pods -n monitoring -l app=loki
```

2. If backends are down, restore them first:
```bash
kubectl rollout restart deployment/prometheus-server -n monitoring
kubectl rollout restart statefulset/tempo -n monitoring
kubectl rollout restart statefulset/loki -n monitoring
```

3. Once backends are healthy, restart the observability agent:
```bash
kubectl rollout restart deployment/observability-agent-service -n greenlang
kubectl rollout status deployment/observability-agent-service -n greenlang
```

### Scenario 3: CrashLoopBackOff -- Database Migration Failure

**Symptoms:** Pod status shows CrashLoopBackOff, init container logs show migration errors.

**Resolution:**

1. Check init container logs:
```bash
kubectl logs -n greenlang <pod-name> -c check-db-migration --tail=100
```

2. Verify database schema:
```bash
kubectl run pg-migration --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT version, description, success FROM flyway_schema_history
   ORDER BY installed_rank DESC LIMIT 5;"
```

3. Restart the deployment after fixing:
```bash
kubectl rollout restart deployment/observability-agent-service -n greenlang
kubectl rollout status deployment/observability-agent-service -n greenlang
```

### Scenario 4: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns.

**Resolution:**

1. Check recent deployment history:
```bash
kubectl rollout history deployment/observability-agent-service -n greenlang
```

2. Rollback to the previous version:
```bash
kubectl rollout undo deployment/observability-agent-service -n greenlang
kubectl rollout status deployment/observability-agent-service -n greenlang
```

3. Verify the rollback resolved the issue:
```bash
kubectl get pods -n greenlang -l app=observability-agent-service
kubectl port-forward -n greenlang svc/observability-agent-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

---

## Post-Incident Steps

### Step 1: Verify Service Health

```bash
# Check all pods are running and ready
kubectl get pods -n greenlang -l app=observability-agent-service

# Check the health endpoint
kubectl port-forward -n greenlang svc/observability-agent-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# Verify the service is being scraped
up{job="observability-agent-service"} == 1

# Verify metrics recording is operational
increase(gl_obs_metrics_recorded_total[5m])

# Verify spans are being created
increase(gl_obs_spans_created_total[5m])

# Verify logs are being ingested
rate(gl_obs_logs_ingested_total[5m])

# Verify alert evaluation is running
increase(gl_obs_alerts_evaluated_total[5m])
```

### Step 3: Verify SLO Tracking Resumed

```bash
# Check SLO compliance via API
curl -s http://localhost:8080/v1/slo/compliance \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check error budget status
curl -s http://localhost:8080/v1/slo/error-budget \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 4: Verify Alert Evaluation Resumed

```bash
# Check active alerts via API
curl -s http://localhost:8080/v1/alerts?state=firing \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

---

## Interim Mitigation

While the Observability Agent is being restored:

1. **Prometheus continues scraping independently.** All Prometheus metric collection continues via direct ServiceMonitor targets. The observability agent adds value by aggregating, correlating, and enriching metrics, but raw scraping is not disrupted.

2. **Alertmanager continues firing for direct Prometheus alerts.** Any PrometheusRule-based alerts (including this one) continue to be evaluated by Prometheus and routed via Alertmanager. Only agent-managed alert evaluations are disrupted.

3. **Traces continue to be exported.** Applications instrumented with OpenTelemetry SDKs continue exporting traces to the OTel Collector. The observability agent enriches and manages traces but is not the sole path.

4. **Loki continues receiving logs.** Grafana Alloy continues shipping logs to Loki. The observability agent adds structured log correlation but is not the sole ingestion path.

5. **SLO tracking has a gap.** SLO compliance ratios and error budget calculations are paused. After recovery, historical data may need to be backfilled.

6. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-foundation` -- engineering response
   - `#platform-oncall` -- on-call engineer
   - `#compliance-ops` -- compliance impact notification (SOC 2 monitoring gap)

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Observability agent down, direct monitoring paths still functional | On-call engineer | Immediate (<5 min) |
| L2 | Observability agent down > 15 minutes, SLO tracking gap growing | Platform team lead + #platform-foundation | 15 minutes |
| L3 | Observability agent down > 30 minutes, compliance impact, alert evaluation gap | Platform team + compliance team + CTO notification | Immediate |
| L4 | Observability agent down due to infrastructure failure affecting telemetry backends | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Observability Agent Health (`/d/obs-agent-svc`)
- **Alert:** `ObsAgentDown` (this alert)
- **Key metrics to watch:**
  - `up{job="observability-agent-service"}` (should always be >= 2)
  - `gl_obs_metrics_recorded_total` rate (should be non-zero)
  - `gl_obs_spans_created_total` rate (should be non-zero during active tracing)
  - `gl_obs_logs_ingested_total` rate (should be >= 1/s)
  - `gl_obs_alerts_evaluated_total` rate (should be non-zero)
  - `gl_obs_health_status` (should be 1)
  - `gl_obs_slo_compliance_ratio` (should be >= 0.999)
  - `gl_obs_error_budget_remaining` (should be > 0.1)
  - Pod restart count (should be 0)
  - Container memory usage vs limit (should stay below 80%)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales from 2 to 10 replicas** based on CPU and memory utilization
4. **Database connection pool** is sized for expected concurrency (default: min 2, max 10)
5. **Rate limit telemetry ingestion** to prevent cardinality explosions from consuming resources

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `ObsAgentDown` | Critical | This alert -- no observability agent pods running |
| `ObsAgentHighErrorRate` | Warning | >5% error rate across operations |
| `ObsAgentHighLatency` | Warning | p99 latency above 1s |
| `ObsAgentHealthDegraded` | Warning | Health status < 1 (degraded) |
| `ObsAgentHealthUnhealthy` | Critical | Health status = 0 (unhealthy) |
| `ObsAgentHighMemoryUsage` | Warning | Memory usage above 80% of limit |
| `ObsAgentPodRestarting` | Warning | >3 restarts in 1 hour |
| `ObsAgentReplicasMismatch` | Warning | Desired replicas != available replicas |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Foundation Team
- **Review cadence:** Quarterly or after any P1 observability agent incident
- **Related runbooks:** [Observability Agent SLO Breach](./observability-agent-slo-breach.md)
