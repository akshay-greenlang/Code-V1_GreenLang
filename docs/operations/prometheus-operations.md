# Prometheus Operations Guide

## Overview

This guide covers day-to-day operations of the GreenLang Prometheus stack, including monitoring health, scaling, backup/restore, capacity planning, and maintenance procedures.

---

## Table of Contents

1. [Daily Operations Checklist](#daily-operations-checklist)
2. [Health Monitoring](#health-monitoring)
3. [Scaling Procedures](#scaling-procedures)
4. [Backup and Restore](#backup-and-restore)
5. [Capacity Planning](#capacity-planning)
6. [Maintenance Windows](#maintenance-windows)
7. [Troubleshooting](#troubleshooting)

---

## Daily Operations Checklist

### Morning Checklist

- [ ] Check Prometheus health dashboard
- [ ] Review overnight alerts
- [ ] Verify all scrape targets are up
- [ ] Check Thanos compactor status
- [ ] Review storage utilization

### Commands

```bash
# Check all monitoring pods
kubectl get pods -n monitoring

# Check Prometheus targets
kubectl port-forward -n monitoring svc/prometheus-server 9090:9090
# Open http://localhost:9090/targets

# Check active alerts
kubectl port-forward -n monitoring svc/alertmanager 9093:9093
# Open http://localhost:9093/#/alerts

# Check Thanos Query stores
kubectl port-forward -n monitoring svc/thanos-query 10902:9090
# Open http://localhost:10902/stores
```

---

## Health Monitoring

### Key Metrics to Watch

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `up{job="prometheus"}` | Prometheus is up | == 0 for 1m |
| `prometheus_tsdb_head_series` | Active time series | > 2M |
| `prometheus_tsdb_head_samples_appended_total` | Ingestion rate | < 1000/s |
| `prometheus_rule_evaluation_failures_total` | Rule failures | > 0 |
| `thanos_compact_halted` | Compactor halted | == 1 |
| `alertmanager_notifications_failed_total` | Notification failures | > 0 |

### Health Dashboard Queries

```promql
# Prometheus availability (last 24h)
avg_over_time(up{job="prometheus"}[24h]) * 100

# Ingestion rate
rate(prometheus_tsdb_head_samples_appended_total[5m])

# Query latency P99
histogram_quantile(0.99, rate(prometheus_engine_query_duration_seconds_bucket[5m]))

# Storage growth rate
deriv(prometheus_tsdb_storage_blocks_bytes[1h])

# Alert delivery success rate
1 - (
  rate(alertmanager_notifications_failed_total[1h]) /
  rate(alertmanager_notifications_total[1h])
) * 100
```

---

## Scaling Procedures

### Horizontal Scaling

#### Scale Prometheus Replicas

```bash
# Check current replicas
kubectl get statefulset -n monitoring prometheus-server

# Scale up (requires Helm values update for persistence)
# Update values.yaml:
#   prometheus:
#     prometheusSpec:
#       replicas: 3

helm upgrade prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring -f values.yaml
```

#### Scale Thanos Query

```bash
# Scale Thanos Query replicas
kubectl scale deployment -n monitoring thanos-query --replicas=3

# Or via Helm
helm upgrade thanos bitnami/thanos -n monitoring \
  --set query.replicaCount=3
```

#### Scale Thanos Store Gateway

```bash
# Scale Store Gateway
kubectl scale statefulset -n monitoring thanos-storegateway --replicas=3
```

### Vertical Scaling

#### Increase Prometheus Memory

```yaml
# values.yaml
prometheus:
  prometheusSpec:
    resources:
      requests:
        memory: 4Gi
      limits:
        memory: 16Gi
```

```bash
helm upgrade prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring -f values.yaml
```

#### Increase Storage

```bash
# For StatefulSets with PVC resize support
kubectl patch pvc prometheus-prometheus-server-0 -n monitoring \
  -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'

# Verify
kubectl get pvc -n monitoring
```

---

## Backup and Restore

### Thanos Handles Long-term Storage

Thanos automatically stores metrics in S3 with configurable retention. This eliminates the need for traditional Prometheus backups.

### S3 Bucket Backup

The Thanos S3 bucket is the source of truth for historical metrics.

**Bucket Lifecycle:**
- Raw data: 30 days
- 5m resolution: 120 days
- 1h resolution: 730 days (2 years)

### Prometheus Configuration Backup

```bash
# Export Prometheus configuration
kubectl get configmap -n monitoring prometheus-server -o yaml > prometheus-config-backup.yaml

# Export PrometheusRules
kubectl get prometheusrule -n monitoring -o yaml > prometheus-rules-backup.yaml

# Export ServiceMonitors
kubectl get servicemonitor -n monitoring -o yaml > servicemonitors-backup.yaml
```

### Alertmanager Configuration Backup

```bash
# Export Alertmanager configuration
kubectl get secret -n monitoring alertmanager-config -o yaml > alertmanager-config-backup.yaml
```

### Restore Procedures

#### Restore Prometheus Configuration

```bash
kubectl apply -f prometheus-config-backup.yaml
kubectl apply -f prometheus-rules-backup.yaml
kubectl apply -f servicemonitors-backup.yaml
```

#### Restore from Thanos S3

Historical data is automatically available through Thanos Store Gateway once:
1. Store Gateway pods are running
2. S3 bucket is accessible
3. Blocks are synced

```bash
# Verify Store Gateway has synced blocks
kubectl logs -n monitoring -l app.kubernetes.io/name=thanos-storegateway | grep "loaded blocks"
```

---

## Capacity Planning

### Current Usage

```promql
# Total time series
prometheus_tsdb_head_series

# Storage used
prometheus_tsdb_storage_blocks_bytes + prometheus_tsdb_wal_storage_size_bytes

# Ingestion rate
rate(prometheus_tsdb_head_samples_appended_total[5m])

# Query rate
rate(prometheus_http_requests_total{handler="/api/v1/query"}[5m])
```

### Growth Projections

#### Memory Planning

```
Required Memory = Base + (Series * 3KB * 2)

Example:
- 500K series: 500,000 * 3KB * 2 = 3GB
- Add 2GB base overhead
- Total: 5GB minimum, 8GB recommended
```

#### Storage Planning

```
Daily Storage = Samples/day * Bytes/sample
              = (Ingestion Rate * 86400) * 1.5 bytes

Example:
- 100K samples/sec: 100,000 * 86400 * 1.5 = 12.96GB/day
- 7-day retention: 90.72GB
- With overhead (2x): 180GB recommended
```

### Scaling Triggers

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Memory Usage | 70% | 85% | Increase memory limits |
| Storage Usage | 70% | 85% | Expand PVC or reduce retention |
| Series Count | 1M | 2M | Add Prometheus replica or drop metrics |
| Query Latency P99 | 10s | 30s | Add Query replicas or recording rules |

---

## Maintenance Windows

### Scheduled Maintenance

| Window | Time (UTC) | Duration | Purpose |
|--------|------------|----------|---------|
| Weekly | Sunday 02:00 | 2 hours | Helm upgrades, config changes |
| Monthly | First Sunday 02:00 | 4 hours | Major version upgrades |

### Pre-Maintenance Checklist

```bash
# 1. Notify stakeholders
# Post in #greenlang-ops about upcoming maintenance

# 2. Create silence for maintenance alerts
kubectl exec -n monitoring alertmanager-0 -- amtool silence add \
  --alertmanager.url=http://localhost:9093 \
  --comment="Scheduled maintenance" \
  --duration=2h \
  'alertname=~"Prometheus.*"'

# 3. Verify backups
kubectl get configmap -n monitoring prometheus-server -o yaml > pre-maintenance-backup.yaml

# 4. Check current state
kubectl get pods -n monitoring
kubectl get pvc -n monitoring
```

### Maintenance Procedures

#### Rolling Restart

```bash
# Restart Prometheus (one at a time)
kubectl rollout restart statefulset -n monitoring prometheus-server

# Wait for healthy
kubectl rollout status statefulset -n monitoring prometheus-server

# Restart Thanos components
kubectl rollout restart deployment -n monitoring thanos-query
kubectl rollout restart statefulset -n monitoring thanos-storegateway
```

#### Helm Upgrade

```bash
# Dry-run first
helm upgrade prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring -f values.yaml --dry-run

# Apply upgrade
helm upgrade prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring -f values.yaml

# Monitor rollout
watch kubectl get pods -n monitoring
```

### Post-Maintenance Checklist

```bash
# 1. Verify all pods running
kubectl get pods -n monitoring

# 2. Check targets
# Open http://localhost:9090/targets

# 3. Verify alerts resolved
# Open http://localhost:9093/#/alerts

# 4. Remove maintenance silence
kubectl exec -n monitoring alertmanager-0 -- amtool silence expire \
  --alertmanager.url=http://localhost:9093 <silence-id>

# 5. Notify stakeholders
# Post in #greenlang-ops that maintenance is complete
```

---

## Troubleshooting

### Common Issues

#### Prometheus High Memory

```bash
# Check memory usage
kubectl top pods -n monitoring -l app.kubernetes.io/name=prometheus

# Check cardinality
kubectl port-forward -n monitoring svc/prometheus-server 9090:9090
# Query: topk(10, count by (__name__)({__name__=~".+"}))

# Solution: Add relabel_configs to drop high-cardinality labels
```

See [Runbook: prometheus-high-memory.md](../runbooks/prometheus-high-memory.md)

#### Thanos Compactor Halted

```bash
# Check compactor logs
kubectl logs -n monitoring -l app.kubernetes.io/name=thanos-compactor

# Check for overlapping blocks
kubectl exec -n monitoring -it <compactor-pod> -- \
  /bin/thanos tools bucket verify --objstore.config-file=/etc/thanos/objstore.yaml

# Solution: Delete overlapping blocks, restart with --wait
```

See [Runbook: thanos-compactor-halted.md](../runbooks/thanos-compactor-halted.md)

#### Alert Notifications Not Delivered

```bash
# Check Alertmanager logs
kubectl logs -n monitoring -l app.kubernetes.io/name=alertmanager

# Test webhook
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test"}' \
  <slack-webhook-url>

# Solution: Verify webhook credentials, check network policies
```

See [Runbook: alertmanager-notifications-failing.md](../runbooks/alertmanager-notifications-failing.md)

### Diagnostic Commands

```bash
# Get Prometheus TSDB status
kubectl exec -n monitoring prometheus-server-0 -- promtool tsdb list

# Check Prometheus configuration
kubectl exec -n monitoring prometheus-server-0 -- cat /etc/prometheus/prometheus.yml

# Verify Thanos stores
kubectl exec -n monitoring -it thanos-query-xxx -- \
  /bin/thanos query --store.sd-dns-interval=30s --http-address=0.0.0.0:10902

# Test alert delivery
kubectl exec -n monitoring alertmanager-0 -- amtool alert add \
  --alertmanager.url=http://localhost:9093 \
  alertname="TestAlert" severity="warning" service="test"
```

---

## Related Documentation

- [Prometheus Stack Architecture](../architecture/prometheus-stack.md)
- [Metrics Developer Guide](../development/metrics-guide.md)
- [Operational Runbooks](../runbooks/README.md)
