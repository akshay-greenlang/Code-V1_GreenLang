# -*- coding: utf-8 -*-
# GL-VCCI Monitoring Module
# Observability, metrics, logging, and alerting

"""
VCCI Monitoring & Observability
================================

Comprehensive monitoring, metrics, logging, and alerting for production.

Monitoring Stack:
----------------
1. Metrics: Prometheus + Grafana
   - Custom dashboards
   - SLO tracking
   - Performance metrics

2. Logging: Structured logging (JSON)
   - Application logs
   - Audit logs
   - Access logs

3. Tracing: OpenTelemetry
   - Distributed tracing
   - Request flow visualization
   - Performance bottleneck identification

4. Alerting: PagerDuty + Slack
   - Critical alerts → PagerDuty
   - Warnings → Slack
   - Automated incident creation

Key Metrics:
-----------
Performance:
- API response time (p50, p95, p99)
- Calculation throughput (calculations/sec)
- Data ingestion rate (records/sec)

Reliability:
- Uptime percentage
- Error rate
- Failed calculations

Business:
- Active customers
- Total calculations performed
- Emissions calculated (tCO2e)
- Suppliers engaged

Dashboards:
----------
1. Executive Dashboard
   - High-level KPIs
   - Business metrics
   - Customer health

2. Operations Dashboard
   - System health
   - Performance metrics
   - Error rates

3. Security Dashboard
   - Authentication events
   - Failed login attempts
   - API key usage

SLOs (Service Level Objectives):
--------------------------------
- API availability: 99.9% (43 minutes downtime/month max)
- API response time: <200ms (p95)
- Calculation accuracy: 100% (Tier 1), 95% (Tier 2)
- Data loss: 0% (zero data loss tolerance)

Alerts:
------
Critical (PagerDuty):
- Service down
- Database unreachable
- Critical error spike

Warning (Slack):
- High response times
- Error rate increase
- Resource utilization >80%

Usage:
------
```python
from monitoring import metrics, logger, tracer

# Record metric
metrics.record_calculation(
    category=1,
    tier="tier_1",
    duration_ms=150
)

# Structured logging
logger.info(
    "calculation_completed",
    calculation_id="calc_123",
    category=1,
    emissions_tco2e=1234.56
)

# Distributed tracing
with tracer.start_span("calculate_scope3") as span:
    span.set_attribute("category", 1)
    result = calculate(...)
```
"""

__version__ = "1.0.0"

__all__ = [
    # "metrics",
    # "logger",
    # "tracer",
    # "alert",
]
