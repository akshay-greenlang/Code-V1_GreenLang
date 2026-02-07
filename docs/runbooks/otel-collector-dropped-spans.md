# OTel Collector Dropped Spans

## Alert

**Alert Name:** `OTelCollectorDroppedSpans`

**Severity:** Warning (escalates to Critical if sustained > 30m)

**Threshold:** `sum(rate(otelcol_exporter_send_failed_spans_total[5m])) > 100` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when the OpenTelemetry Collector is dropping spans from its export pipeline. Dropped spans indicate that trace data is being permanently lost before reaching Grafana Tempo. Common causes include:

1. **Exporter queue full**: The batch exporter's sending queue has reached capacity
2. **Tempo backend unavailable**: Tempo distributors are down or unreachable
3. **Network issues**: DNS resolution failures, firewall changes, or TLS errors
4. **Memory pressure**: Collector is approaching its memory limit and shedding load
5. **Rate limiting**: Tempo is rate-limiting the tenant ingestion

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Trace search returns incomplete results; service maps have gaps |
| **Data Impact** | High | Dropped spans are permanently lost; cannot be recovered |
| **SLA Impact** | Medium | Trace data completeness SLA (99.9%) is at risk |
| **Compliance Impact** | Medium | Compliance agent traces may be affected (100% sampling required) |

---

## Symptoms

- `otelcol_exporter_send_failed_spans_total` counter is incrementing
- `otelcol_exporter_queue_size` near or at `otelcol_exporter_queue_capacity`
- Collector memory usage increasing toward limits
- Gaps in Grafana trace search results
- Service graph panels showing incomplete topology
- Missing spans in compliance agent trace flows

---

## Diagnostic Steps

### Step 1: Check Collector Export Metrics

```promql
# Failed span exports per exporter
sum(rate(otelcol_exporter_send_failed_spans_total[5m])) by (exporter)

# Successful span exports (should be non-zero)
sum(rate(otelcol_exporter_sent_spans_total[5m])) by (exporter)

# Queue utilisation (ratio of size to capacity)
otelcol_exporter_queue_size / otelcol_exporter_queue_capacity

# Receiver accepted vs refused spans
sum(rate(otelcol_receiver_accepted_spans_total[5m])) by (receiver)
sum(rate(otelcol_receiver_refused_spans_total[5m])) by (receiver)
```

### Step 2: Check Collector Pod Health

```bash
# List collector pods
kubectl get pods -n monitoring -l app.kubernetes.io/name=otel-collector

# Check resource usage
kubectl top pods -n monitoring -l app.kubernetes.io/name=otel-collector

# Check memory limits
kubectl get pods -n monitoring -l app.kubernetes.io/name=otel-collector \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].resources.limits.memory}{"\n"}{end}'

# Check for OOMKilled or restarts
kubectl get pods -n monitoring -l app.kubernetes.io/name=otel-collector \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'
```

### Step 3: Check Collector Logs

```bash
# Get recent error logs
kubectl logs -n monitoring -l app.kubernetes.io/name=otel-collector --tail=200 | grep -i "error\|fail\|drop\|refused"

# Check for Tempo connectivity issues
kubectl logs -n monitoring -l app.kubernetes.io/name=otel-collector --tail=500 | grep -i "tempo\|export\|connection\|timeout\|grpc"

# Check for memory limiter activation
kubectl logs -n monitoring -l app.kubernetes.io/name=otel-collector --tail=200 | grep -i "memory_limiter\|dropping\|refusing"

# Check for TLS errors
kubectl logs -n monitoring -l app.kubernetes.io/name=otel-collector --tail=200 | grep -i "tls\|certificate\|handshake"
```

### Step 4: Check Tempo Distributor Health

```promql
# Tempo distributor ingestion rate
sum(rate(tempo_distributor_spans_received_total[5m]))

# Tempo distributor errors
sum(rate(tempo_distributor_ingester_append_failures_total[5m]))

# Tempo rate limiting
sum(rate(tempo_discarded_spans_total{reason="rate_limited"}[5m]))
```

```bash
# Check Tempo distributor pods
kubectl get pods -n tracing -l app.kubernetes.io/component=distributor

# Check Tempo distributor logs
kubectl logs -n tracing -l app.kubernetes.io/component=distributor --tail=200 | grep -i "error\|rate\|limit\|reject"
```

### Step 5: Check Network Connectivity

```bash
# Test connectivity from collector to Tempo distributor
kubectl exec -n monitoring <collector-pod> -- wget -q --spider http://tempo-distributor:4317 && echo "Reachable" || echo "Unreachable"

# Check DNS resolution
kubectl exec -n monitoring <collector-pod> -- nslookup tempo-distributor.tracing.svc.cluster.local

# Check NetworkPolicy
kubectl get networkpolicy -n monitoring -o yaml | grep -A 20 "otel-collector"
```

### Step 6: Check zpages for Pipeline Status

```bash
# Port-forward to zpages endpoint
kubectl port-forward -n monitoring <collector-pod> 55679:55679

# Then browse http://localhost:55679/debug/tracez for pipeline status
# Or use curl:
curl -s http://localhost:55679/debug/pipelinez | python -m json.tool
```

---

## Resolution Steps

### Scenario 1: Exporter Queue Full

**Symptoms:** `otelcol_exporter_queue_size` equals `otelcol_exporter_queue_capacity`

**Resolution:**

1. **Increase queue size (immediate relief):**

```yaml
# In OTel Collector config (configmap.yaml or Helm values)
exporters:
  otlp/tempo:
    sending_queue:
      queue_size: 1000  # Increase from default 500
      num_consumers: 10  # Increase parallel export workers
```

2. **Increase batch size to improve throughput:**

```yaml
processors:
  batch:
    send_batch_size: 2048   # Increase from 1024
    send_batch_max_size: 4096
    timeout: 5s             # Increase from 2s
```

3. **Scale collector replicas:**

```bash
kubectl scale deployment otel-collector -n monitoring --replicas=5
```

### Scenario 2: Tempo Backend Unavailable

**Symptoms:** Collector logs show connection refused/timeout to Tempo

**Resolution:**

1. **Check Tempo distributor status:**

```bash
kubectl get pods -n tracing -l app.kubernetes.io/component=distributor
kubectl logs -n tracing -l app.kubernetes.io/component=distributor --tail=100
```

2. **Restart Tempo distributors if unhealthy:**

```bash
kubectl rollout restart deployment tempo-distributor -n tracing
```

3. **Enable retry with backoff (should already be configured):**

```yaml
exporters:
  otlp/tempo:
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s
```

### Scenario 3: Memory Limiter Dropping Spans

**Symptoms:** Logs show "memory_limiter" processor dropping data

**Resolution:**

1. **Increase collector memory limits:**

```yaml
# In Helm values
resources:
  requests:
    memory: 1Gi
    cpu: 500m
  limits:
    memory: 2Gi
    cpu: "2"
```

2. **Adjust memory limiter thresholds:**

```yaml
processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 1536      # Set to ~75% of memory limit
    spike_limit_mib: 512  # Allow 512MB spike headroom
```

3. **Reduce incoming volume with head sampling:**

```yaml
processors:
  probabilistic_sampler:
    sampling_percentage: 50
```

### Scenario 4: Tempo Rate Limiting

**Symptoms:** `tempo_discarded_spans_total{reason="rate_limited"}` > 0

**Resolution:**

1. **Increase Tempo rate limits:**

```yaml
# In Tempo config
overrides:
  defaults:
    ingestion:
      rate_limit_bytes: 30000000   # Increase to 30MB/s
      burst_size_bytes: 40000000   # 40MB burst
```

2. **Scale Tempo distributors:**

```bash
kubectl scale deployment tempo-distributor -n tracing --replicas=4
```

### Scenario 5: Tail Sampling Consuming Too Much Memory

**Symptoms:** High memory usage, tail_sampling processor holding too many traces

**Resolution:**

1. **Reduce tail sampling trace buffer:**

```yaml
processors:
  tail_sampling:
    decision_wait: 5s       # Reduce from 10s
    num_traces: 25000        # Reduce from 50000
```

2. **Switch high-volume services to head sampling:**

```yaml
processors:
  probabilistic_sampler:
    sampling_percentage: 25
```

---

## Tuning Guide

### Queue Sizing Formula

```
queue_size = (peak_spans_per_sec * decision_wait_sec * 2) / send_batch_size
```

For GreenLang production (5000 spans/sec peak, 10s decision wait, 1024 batch):
```
queue_size = (5000 * 10 * 2) / 1024 = ~97 → round up to 200
```

### Memory Sizing Formula

```
memory_needed = num_traces_in_buffer * avg_trace_size_bytes + batch_queue_bytes
```

For production (50000 traces, ~2KB avg, 500 batch queue of 1024 spans * ~200B):
```
memory = (50000 * 2048) + (500 * 1024 * 200) ≈ 200MB
```

Set memory_limiter to 75% of container limit.

---

## Prevention

### Monitoring

- **Dashboard:** OTel Collector (`/d/otel-collector`)
- **Alerts:** `OTelCollectorDroppedSpans`, `OTelCollectorQueueFull`, `OTelCollectorHighMemory`
- **Key metrics:**
  - `otelcol_exporter_send_failed_spans_total` (should be 0)
  - `otelcol_exporter_queue_size` (should stay < 80% of capacity)
  - Container memory usage (should stay < 75% of limit)
  - `otelcol_processor_dropped_spans` (should be 0)

### Capacity Planning

1. **Size collector replicas** at 2x expected peak throughput for headroom
2. **Set memory limits** at 2x the calculated memory_needed
3. **Use HPA** with target CPU utilisation of 60%
4. **Monitor growth trends** weekly and scale proactively

### Configuration Best Practices

```yaml
# Recommended production OTel Collector settings
processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 1536
    spike_limit_mib: 512
  batch:
    send_batch_size: 1024
    send_batch_max_size: 2048
    timeout: 2s
  tail_sampling:
    decision_wait: 10s
    num_traces: 50000

exporters:
  otlp/tempo:
    sending_queue:
      queue_size: 500
      num_consumers: 10
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
```

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Observability Team
- **Review cadence:** Quarterly or after any P1 tracing incident
- **Related alerts:** `OTelCollectorDown`, `OTelCollectorHighMemory`, `TracePipelineEndToEndLatency`
- **Related dashboards:** OTel Collector, Tracing Overview
