# Performance Service Level Objectives (SLOs)

**Version:** 1.0.0
**Last Updated:** 2025-11-09
**Review Cycle:** Quarterly

---

## Overview

This document defines performance Service Level Objectives (SLOs) for GreenLang infrastructure and applications. SLOs are measured over 30-day rolling windows unless otherwise specified.

**SLI = Service Level Indicator** (actual measurement)
**SLO = Service Level Objective** (target)
**SLA = Service Level Agreement** (contractual commitment)

---

## Infrastructure SLOs

### 1. greenlang.intelligence (LLM Services)

| Metric | SLO | Measurement | Alert Threshold |
|--------|-----|-------------|-----------------|
| ChatSession P95 Latency | < 2 seconds | 95th percentile response time | > 2.5 seconds |
| ChatSession P99 Latency | < 5 seconds | 99th percentile response time | > 6 seconds |
| First Token Latency | < 200ms | Time to first token in stream | > 300ms |
| Semantic Cache Hit Rate | > 30% | Hits / (Hits + Misses) | < 20% |
| Embedding Generation | < 100ms | Per embedding latency | > 150ms |
| Error Rate | < 1% | Errors / Total Requests | > 2% |
| Availability | 99.5% | Uptime / Total Time | < 99% |

**Cost SLOs:**
- Monthly LLM costs < budget × 1.1 (10% buffer)
- Cost per 1M tokens < $35 (blended rate)

**Monitoring:**
```python
from greenlang.intelligence import get_llm_metrics

metrics = get_llm_metrics(window_days=30)
assert metrics['p95_latency_ms'] < 2000
assert metrics['cache_hit_rate'] > 0.30
assert metrics['error_rate'] < 0.01
```

### 2. greenlang.cache (Caching Layers)

| Metric | SLO | Measurement | Alert Threshold |
|--------|-----|-------------|-----------------|
| L1 GET Latency P95 | < 100 µs | 95th percentile | > 200 µs |
| L1 SET Latency P95 | < 100 µs | 95th percentile | > 200 µs |
| L2 GET Latency P95 | < 5ms | 95th percentile (Redis) | > 10ms |
| L2 SET Latency P95 | < 10ms | 95th percentile (Redis) | > 20ms |
| L3 GET Latency P95 | < 100ms | 95th percentile (Disk/S3) | > 200ms |
| Overall Hit Rate | > 50% | Across all levels | < 40% |
| Throughput (L1) | > 100K ops/sec | Operations per second | < 80K ops/sec |
| Throughput (L2) | > 10K ops/sec | Operations per second | < 8K ops/sec |
| Availability | 99.95% | Uptime / Total Time | < 99.5% |

**Monitoring:**
```python
from greenlang.cache import get_cache_metrics

metrics = get_cache_metrics()
assert metrics['l1_p95_latency_us'] < 100
assert metrics['l2_p95_latency_ms'] < 5
assert metrics['overall_hit_rate'] > 0.50
```

### 3. greenlang.db (Database)

| Metric | SLO | Measurement | Alert Threshold |
|--------|-----|-------------|-----------------|
| Query P50 Latency | < 50ms | 50th percentile | > 100ms |
| Query P95 Latency | < 100ms | 95th percentile | > 200ms |
| Query P99 Latency | < 500ms | 99th percentile | > 1000ms |
| Connection Acquisition | < 10ms | Pool acquisition time | > 50ms |
| Transaction Commit | < 20ms | Commit latency | > 50ms |
| Pool Utilization | < 80% | Active / Total Connections | > 90% |
| Slow Queries | < 1% | Queries > 1 second | > 5% |
| Error Rate | < 0.1% | Failed queries / Total | > 1% |
| Availability | 99.99% | Database uptime | < 99.9% |

**Monitoring:**
```python
from greenlang.db import get_db_metrics

metrics = get_db_metrics()
assert metrics['p95_latency_ms'] < 100
assert metrics['pool_utilization'] < 0.80
assert metrics['error_rate'] < 0.001
```

### 4. greenlang.services.factor_broker

| Metric | SLO | Measurement | Alert Threshold |
|--------|-----|-------------|-----------------|
| Resolution P50 Latency | < 20ms | 50th percentile | > 50ms |
| Resolution P95 Latency | < 50ms | 95th percentile | > 100ms |
| Resolution P99 Latency | < 100ms | 99th percentile | > 200ms |
| Cache Hit Rate | > 70% | Cached / Total Resolutions | < 60% |
| Accuracy | > 95% | Correct factors / Total | < 90% |
| Data Freshness | < 24 hours | Age of cached factors | > 48 hours |
| Availability | 99.9% | Service uptime | < 99.5% |

**Monitoring:**
```python
from greenlang.services.factor_broker import get_broker_metrics

metrics = get_broker_metrics()
assert metrics['p95_latency_ms'] < 50
assert metrics['cache_hit_rate'] > 0.70
assert metrics['accuracy'] > 0.95
```

### 5. greenlang.services.entity_mdm

| Metric | SLO | Measurement | Alert Threshold |
|--------|-----|-------------|-----------------|
| Matching P95 Latency | < 100ms | Per entity | > 200ms |
| Batch Throughput | > 50/sec | Entities matched per second | < 30/sec |
| Accuracy | > 95% | Correct matches / Total | < 90% |
| Precision | > 90% | True positives / (TP + FP) | < 85% |
| Recall | > 85% | True positives / (TP + FN) | < 80% |
| Availability | 99.9% | Service uptime | < 99% |

---

## Application SLOs

### GL-CBAM-APP

| Metric | SLO | Measurement | Alert Threshold |
|--------|-----|-------------|-----------------|
| Single Shipment P95 | < 1 second | End-to-end processing | > 2 seconds |
| Batch Throughput | > 1000/sec | Records processed per second | < 800/sec |
| Memory (10K records) | < 100 MB | Resident Set Size | > 150 MB |
| Intake Agent P95 | < 100ms | Validation latency | > 200ms |
| Calculator Agent P95 | < 500ms | Calculation latency | > 1000ms |
| Packager Agent P95 | < 200ms | Packaging latency | > 400ms |
| Error Rate | < 0.5% | Failed shipments / Total | > 1% |
| Data Quality | > 99% | Valid outputs / Total | < 98% |

**Critical Path SLOs:**
```
Total P95 = Intake P95 + Calculator P95 + Packager P95
         < 100ms + 500ms + 200ms
         = 800ms (well under 1 second target)
```

**Monitoring:**
```python
from GL_CBAM_APP.monitoring import get_pipeline_metrics

metrics = get_pipeline_metrics()
assert metrics['single_shipment_p95_ms'] < 1000
assert metrics['batch_throughput_per_sec'] > 1000
assert metrics['error_rate'] < 0.005
```

### GL-CSRD-APP

| Metric | SLO | Measurement | Alert Threshold |
|--------|-----|-------------|-----------------|
| Materiality Assessment | < 5 seconds | Complete assessment | > 10 seconds |
| ESRS Calculation | < 3 seconds | All ESRS metrics | > 5 seconds |
| XBRL Generation | < 2 seconds | Full XBRL report | > 4 seconds |
| RAG Retrieval | < 100ms | Document retrieval | > 200ms |
| RAG + LLM | < 1 second | Retrieval + generation | > 2 seconds |
| Full Pipeline | < 10 seconds | End-to-end | > 15 seconds |
| Error Rate | < 1% | Failed assessments / Total | > 2% |
| Regulatory Accuracy | > 98% | Correct interpretations | < 95% |

**Monitoring:**
```python
from GL_CSRD_APP.monitoring import get_csrd_metrics

metrics = get_csrd_metrics()
assert metrics['materiality_p95_ms'] < 5000
assert metrics['full_pipeline_p95_ms'] < 10000
assert metrics['regulatory_accuracy'] > 0.98
```

### GL-VCCI-APP

| Metric | SLO | Measurement | Alert Threshold |
|--------|-----|-------------|-----------------|
| Scope 3 (10K suppliers) | < 60 seconds | Batch calculation | > 120 seconds |
| Scope 3 Throughput | > 100/sec | Suppliers per second | < 80/sec |
| Entity Resolution | > 50/sec | Entities matched per second | < 40/sec |
| Hotspot Analysis | < 1 second | Top 20% identification | > 2 seconds |
| PDF Report Generation | < 5 seconds | Full report | > 10 seconds |
| Excel Report | < 3 seconds | Full report | > 6 seconds |
| Error Rate | < 0.5% | Failed calculations / Total | > 1% |
| Data Confidence | > 85% | High confidence results | < 80% |

**Monitoring:**
```python
from GL_VCCI_APP.monitoring import get_vcci_metrics

metrics = get_vcci_metrics()
assert metrics['scope3_10k_seconds'] < 60
assert metrics['throughput_per_sec'] > 100
assert metrics['data_confidence'] > 0.85
```

---

## Availability SLOs

### Service Tiers

**Tier 1 (Critical):** 99.99% uptime
- Database (primary)
- Cache (L1/L2)

**Tier 2 (Important):** 99.9% uptime
- Factor Broker
- Entity MDM
- LLM Services

**Tier 3 (Standard):** 99.5% uptime
- Applications
- Monitoring
- Reporting

### Downtime Allowances

| SLO | Daily | Monthly | Yearly |
|-----|-------|---------|--------|
| 99.99% | 8.6 seconds | 4.3 minutes | 52.6 minutes |
| 99.9% | 1.4 minutes | 43.2 minutes | 8.8 hours |
| 99.5% | 7.2 minutes | 3.6 hours | 1.8 days |

---

## Error Budget Policy

### Error Budget Calculation

```
Error Budget = (1 - SLO) × Total Requests

Example:
- SLO: 99.9% availability
- Monthly requests: 10M
- Error budget: 0.001 × 10M = 10,000 errors allowed
```

### Error Budget Status

**Healthy (> 50% remaining):**
- Normal operations
- Deploy when ready
- Feature development continues

**Warning (20-50% remaining):**
- Increase monitoring
- Review recent changes
- Defer non-critical deploys

**Critical (< 20% remaining):**
- Freeze feature deploys
- Focus on reliability
- Incident postmortems required
- CTO approval for deploys

### Error Budget Monitoring

```python
from greenlang.monitoring import get_error_budget

budget = get_error_budget(service="cbam_pipeline", window_days=30)

print(f"SLO: {budget['slo']}")
print(f"Total requests: {budget['total_requests']}")
print(f"Errors allowed: {budget['errors_allowed']}")
print(f"Errors actual: {budget['errors_actual']}")
print(f"Budget remaining: {budget['remaining_percent']}%")
print(f"Status: {budget['status']}")  # healthy|warning|critical

assert budget['status'] != 'critical'
```

---

## SLO Compliance Tracking

### Monthly SLO Report

```python
from greenlang.monitoring import generate_slo_report

report = generate_slo_report(month="2025-11")

# Example output:
# {
#   "infrastructure": {
#     "llm_services": {
#       "p95_latency_ms": 1850,
#       "target": 2000,
#       "compliance": True,
#       "trend": "improving"
#     },
#     "cache": {
#       "hit_rate": 0.55,
#       "target": 0.50,
#       "compliance": True,
#       "trend": "stable"
#     }
#   },
#   "applications": {
#     "cbam": {
#       "throughput_per_sec": 1200,
#       "target": 1000,
#       "compliance": True,
#       "trend": "improving"
#     }
#   },
#   "overall_compliance": 0.95  # 95% of SLOs met
# }
```

### SLO Dashboard

Access real-time SLO compliance:
- **URL:** https://metrics.greenlang.ai/slo
- **Grafana:** Dashboard "Performance SLOs"
- **CLI:** `greenlang slo status`

### Alerting

**Alert Routing:**
- **P0 (Critical):** Availability < SLO → PagerDuty → On-call engineer
- **P1 (High):** Latency > 2× SLO → Slack #incidents
- **P2 (Medium):** Error budget < 20% → Slack #performance
- **P3 (Low):** Degraded but within SLO → Email performance@

**Alert Examples:**
```yaml
# Prometheus alerting rules

- alert: LLMLatencyHigh
  expr: llm_p95_latency_ms > 2000
  for: 5m
  labels:
    severity: high
  annotations:
    summary: "LLM P95 latency exceeds SLO"

- alert: CacheHitRateLow
  expr: cache_hit_rate < 0.30
  for: 10m
  labels:
    severity: medium
  annotations:
    summary: "Cache hit rate below SLO"

- alert: ErrorBudgetCritical
  expr: error_budget_remaining_percent < 20
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Error budget critical - freeze deploys"
```

---

## SLO Review Process

### Quarterly Review Checklist

- [ ] Review all SLO compliance (past 90 days)
- [ ] Analyze SLO violations and root causes
- [ ] Adjust SLOs based on:
  - Business requirements changes
  - Infrastructure improvements
  - User feedback
  - Cost considerations
- [ ] Update documentation
- [ ] Communicate changes to stakeholders
- [ ] Update monitoring and alerting

### Stakeholders

- **SLO Owner:** Performance Engineering Team
- **Reviewers:** Engineering Leads, Product Management
- **Approvers:** CTO, VP Engineering
- **Informed:** All engineering teams, Customer Success

---

## Appendix

### Measurement Tools

```bash
# Infrastructure metrics
python tools/profiling/measure_infrastructure.py

# Application metrics
python tools/profiling/measure_applications.py

# Generate SLO report
python tools/reporting/generate_slo_report.py --month 2025-11
```

### Related Documentation

- [Performance Optimization Guide](PERFORMANCE_OPTIMIZATION_GUIDE.md)
- [Performance Troubleshooting](PERFORMANCE_TROUBLESHOOTING.md)
- [Monitoring Dashboard](../greenlang/monitoring/dashboards/performance_detailed.py)
- [Benchmarks](../benchmarks/)

---

**Questions or concerns about SLOs?**
- Slack: #performance-engineering
- Email: performance@greenlang.ai
- Wiki: https://wiki.greenlang.ai/slo
