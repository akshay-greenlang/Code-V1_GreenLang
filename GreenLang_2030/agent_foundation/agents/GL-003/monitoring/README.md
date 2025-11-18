# GL-003 SteamSystemAnalyzer - Monitoring Infrastructure

## Overview

Comprehensive monitoring and observability infrastructure for GL-003 SteamSystemAnalyzer, providing production-grade metrics, dashboards, and alerting for steam system analysis operations.

## Quick Start

```bash
# 1. Deploy ServiceMonitor
kubectl apply -f ../deployment/servicemonitor.yaml

# 2. Deploy alerting rules
kubectl apply -f alerts/prometheus_rules.yaml
kubectl apply -f alerts/determinism_alerts.yml

# 3. Import Grafana dashboards
kubectl create configmap gl-003-grafana-dashboards \
  --from-file=grafana/ \
  -n greenlang

# 4. Verify metrics
kubectl port-forward -n greenlang svc/gl-003-metrics 8000:8000
curl http://localhost:8000/metrics
```

## Components

### Python Monitoring Files

1. **metrics.py** (650+ lines)
   - 80+ Prometheus metrics
   - Steam system operating metrics (pressure, temperature, flow, condensate)
   - Leak detection and steam trap performance
   - Distribution efficiency tracking
   - Heat loss monitoring
   - Determinism verification metrics
   - Business metrics (cost savings, energy savings)

2. **health_checks.py**
   - Kubernetes liveness/readiness/startup probes
   - Component health checks (database, cache, external APIs)
   - System resource monitoring
   - Detailed health status reporting

3. **determinism_validator.py**
   - Runtime determinism verification
   - AI configuration validation (temperature=0.0, seed=42)
   - Calculation reproducibility checks
   - Provenance hash verification
   - Unseeded random operation detection

4. **feedback_metrics.py**
   - User feedback collection and tracking
   - Net Promoter Score (NPS) calculation
   - Accuracy metrics
   - Experiment tracking

5. **metrics_integration.py**
   - Seamless metrics integration into orchestrator
   - Decorators for automatic tracking
   - Context managers for analysis tracking
   - Background metrics updater

### Alert Rules

1. **prometheus_rules.yaml**
   - CRITICAL alerts (agent unavailable, high error rate, critical leaks)
   - WARNING alerts (performance degradation, low efficiency, moderate leaks)
   - BUSINESS alerts (low savings, high leak costs)
   - SLO alerts (availability, latency, error budget)
   - QUALITY alerts (calculation inconsistency, detection accuracy)

2. **determinism_alerts.yml**
   - Determinism violation alerts
   - Hash verification failures
   - AI configuration compliance
   - Seed propagation issues

### Grafana Dashboards

1. **agent_dashboard.json** - Main operational dashboard
2. **determinism_dashboard.json** - Determinism monitoring
3. **executive_dashboard.json** - Executive summary
4. **feedback_dashboard.json** - User feedback analytics
5. **operations_dashboard.json** - Operations team view
6. **quality_dashboard.json** - Quality metrics

## Key Metrics

### Steam System Metrics

```promql
# Steam pressure
gl_003_steam_pressure_bar{system_id, location, steam_type}

# Steam flow rate
gl_003_steam_flow_rate_kg_hr{system_id, location}

# Condensate return
gl_003_condensate_return_percent{system_id}

# Distribution efficiency
gl_003_distribution_efficiency_percent{system_id}
```

### Leak Detection Metrics

```promql
# Active leaks
gl_003_active_leaks_count{system_id, severity}

# Leak cost impact
gl_003_leak_cost_impact_usd_hr{system_id}

# Leak detection confidence
gl_003_leak_detection_confidence_percent{system_id, leak_id}
```

### Steam Trap Metrics

```promql
# Operational traps
gl_003_steam_trap_operational_count{system_id, trap_type}

# Failed traps
gl_003_steam_trap_failed_count{system_id, trap_type, failure_mode}

# Trap performance
gl_003_steam_trap_performance_score_percent{system_id, trap_id}
```

## Dashboards Access

- **Executive Dashboard**: https://grafana.greenlang.io/d/gl-003-executive
- **Operations Dashboard**: https://grafana.greenlang.io/d/gl-003-operations
- **Agent Dashboard**: https://grafana.greenlang.io/d/gl-003-agent
- **Quality Dashboard**: https://grafana.greenlang.io/d/gl-003-quality

## Alerting

### Severity Levels

| Severity | Response Time | Channel | Examples |
|----------|--------------|---------|----------|
| CRITICAL | Immediate | PagerDuty | Agent down, critical leaks, massive trap failure |
| WARNING | 15 minutes | Slack | Low efficiency, moderate leaks, high heat loss |
| INFO | 1 hour | Email | Low savings, business metrics |

### Alert Destinations

- **CRITICAL**: PagerDuty on-call + Slack #greenlang-critical
- **WARNING**: Slack #greenlang-ops
- **INFO**: Email team@greenlang.io

## Documentation

- **MONITORING.md** - Complete monitoring guide (setup, troubleshooting, runbooks)
- **QUICK_REFERENCE.md** - Quick reference for common queries
- **README.md** - This file (overview and quick start)

## Integration Example

```python
from monitoring.metrics import MetricsCollector
from monitoring.metrics_integration import MetricsIntegration

# Initialize metrics
metrics = MetricsIntegration()

# Update steam system state
metrics.update_steam_system_state(
    system_id="STEAM-001",
    state={
        "pressure_bar": 10.5,
        "temperature_c": 184,
        "flow_rate_kg_hr": 5000,
        "condensate_return_percent": 85,
        "distribution_efficiency_percent": 92,
        "location": "main_header",
        "steam_type": "high_pressure"
    }
)

# Record analysis result
metrics.record_analysis_result(
    system_id="STEAM-001",
    steam_type="high_pressure",
    analysis_type="distribution_efficiency",
    result={
        "improvement_percent": 4.5,
        "cost_savings_usd_hr": 85.50,
        "energy_savings_mwh": 12.3,
        "annual_savings_usd": 749580,
        "annual_energy_savings_mwh": 107784
    }
)
```

## Support

- **Documentation**: https://docs.greenlang.io/gl-003
- **Runbooks**: https://runbooks.greenlang.io/gl-003
- **On-Call**: PagerDuty - GL-003 Escalation Policy
- **Slack**: #greenlang-gl-003
- **Email**: ops@greenlang.io

## SLA

| Metric | Target | Monitoring |
|--------|--------|------------|
| Availability | 99.9% | Prometheus + AlertManager |
| P95 Latency | < 2s | Grafana Operations Dashboard |
| Error Rate | < 0.1% | Prometheus Alerts |
| MTTR | < 15min | Incident tracking |

---

**Version**: 1.0.0
**Last Updated**: 2025-11-17
**Maintained By**: GreenLang DevOps Team
