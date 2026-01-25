# GreenLang Monitoring - Quick Reference Card

**Version:** 1.0.0 | **Updated:** 2025-10-16

---

## 🚀 Quick Start (2 Minutes)

### Automated Integration

```bash
# Single agent
python scripts/add_monitoring_and_changelog.py --agent carbon

# All agents
python scripts/add_monitoring_and_changelog.py --all-agents

# Dry run first
python scripts/add_monitoring_and_changelog.py --agent fuel --dry-run
```

### Manual Integration (4 Lines of Code)

```python
# 1. Import
from templates.agent_monitoring import OperationalMonitoringMixin

# 2. Inherit
class MyAgent(OperationalMonitoringMixin, BaseAgent):

    # 3. Setup
    def __init__(self):
        super().__init__()
        self.setup_monitoring(agent_name="my_agent")

    # 4. Track
    def execute(self, input_data):
        with self.track_execution(input_data) as tracker:
            result = self._do_work(input_data)
            tracker.set_cost(0.08)
            tracker.set_tokens(2500)
            return result
```

---

## 📊 Core API

### Setup

```python
agent.setup_monitoring(
    agent_name="my_agent",
    enable_metrics=True,
    enable_health_checks=True,
    enable_alerting=True,
    max_history=1000,
    alert_callback=my_callback
)
```

### Track Execution

```python
with self.track_execution(input_data) as tracker:
    # Your code here
    tracker.set_cost(0.08)              # USD
    tracker.set_tokens(2500)            # Count
    tracker.increment_ai_calls(1)       # Count
    tracker.increment_tool_calls(3)     # Count
    tracker.set_cache_hit(True)         # Boolean
```

### Health Check

```python
health = agent.health_check()
print(health.status.value)              # healthy/degraded/unhealthy
print(health.metrics['success_rate'])   # 0.0 to 1.0
print(health.uptime_seconds)            # Seconds
```

### Performance Summary

```python
summary = agent.get_performance_summary(window_minutes=60)
print(summary['total_executions'])
print(summary['success_rate'])
print(summary['latency']['avg_ms'])
print(summary['cost']['total_usd'])
```

### Alerts

```python
# Get alerts
alerts = agent.get_alerts(
    severity=AlertSeverity.CRITICAL,
    unresolved_only=True
)

# Resolve alert
agent.resolve_alert(alert_id)

# Set thresholds
agent.set_thresholds(
    latency_ms=2000,
    error_rate=0.01,
    cost_usd=0.25
)
```

### Metrics Export

```python
# Prometheus format
metrics = agent.export_metrics_prometheus()

# Execution history
history = agent.get_execution_history(limit=100)
```

---

## 🔧 Production Setup

### Flask Endpoints

```python
from flask import Flask, jsonify, Response

@app.route('/health')
def health():
    return jsonify(agent.health_check().to_dict())

@app.route('/health/liveness')
def liveness():
    return jsonify({"status": "alive"}), 200

@app.route('/health/readiness')
def readiness():
    health = agent.health_check()
    if health.status == HealthStatus.UNHEALTHY:
        return jsonify({"status": "not ready"}), 503
    return jsonify({"status": "ready"}), 200

@app.route('/metrics')
def metrics():
    return Response(
        agent.export_metrics_prometheus(),
        mimetype='text/plain'
    )
```

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health/liveness
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/readiness
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Prometheus Scraping

```yaml
scrape_configs:
  - job_name: 'greenlang_agents'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

---

## 📈 Metrics Collected

### Performance
- `execution_duration_ms` - Latency (histogram)
- `execution_cost_usd` - Cost per execution (histogram)
- `execution_tokens` - Token usage (histogram)
- `executions_total` - Total executions (counter)
- `cache_hits_total` - Cache hits (counter)

### Health
- `success_rate` - Success percentage (gauge)
- `error_rate` - Error percentage (gauge)
- `avg_latency_ms` - Average latency (gauge)
- `uptime_seconds` - Agent uptime (gauge)

---

## ⚠️ Alert Severity Levels

- **INFO** - Informational, no action needed
- **WARNING** - Attention needed, not critical
- **ERROR** - Error occurred, investigate soon
- **CRITICAL** - Critical issue, immediate action

### Default Thresholds

```python
latency_ms = 5000      # 5 seconds
error_rate = 0.10      # 10%
cost_usd = 1.00        # $1.00
```

---

## 🧪 Testing

```bash
# Run test suite
python templates/test_monitoring_system.py

# Run examples
python templates/example_integration.py

# Verify integration
python scripts/add_monitoring_and_changelog.py --agent carbon --verify-only
```

---

## 📝 Changelog Management

### Update Changelog

```markdown
## [Unreleased]

### Added
- New feature X

### Changed
- Updated Y to Z

### Fixed
- Bug in calculation
```

### Version Numbering

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backward-compatible)
- PATCH: Bug fixes

---

## 🐛 Troubleshooting

### Import Error
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Not Tracking
- Ensure `setup_monitoring()` was called
- Use `with self.track_execution(...)` context manager

### High Memory
```python
agent.setup_monitoring(
    agent_name="my_agent",
    max_history=100  # Reduce from default 1000
)
```

---

## 📚 Documentation

- **Quick Start:** `templates/README.md`
- **Full Guide:** `templates/README_MONITORING.md`
- **Examples:** `templates/example_integration.py`
- **Tests:** `templates/test_monitoring_system.py`

---

## ✅ Checklist

Integration:
- [ ] Import `OperationalMonitoringMixin`
- [ ] Add to class inheritance
- [ ] Call `setup_monitoring()` in `__init__`
- [ ] Wrap `execute()` with `track_execution()`
- [ ] Create CHANGELOG.md

Production:
- [ ] Setup health endpoints
- [ ] Configure Prometheus scraping
- [ ] Set production thresholds
- [ ] Configure alert callbacks
- [ ] Test Kubernetes probes

---

**🎯 Goal:** D11 (Operations) + D12 (Improvement) compliance in < 5 minutes

**📊 Impact:** < 5% performance overhead, 100% visibility

**🔗 Support:** See `templates/README_MONITORING.md` for full documentation
