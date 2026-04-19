# GreenLang Monitoring & Observability System

Production-grade monitoring, alerting, and analytics infrastructure for GreenLang.

## Overview

This comprehensive monitoring system provides:

- **Real-time Dashboards**: 6 Grafana dashboards for IUM, costs, performance, compliance, productivity, and health
- **Automated Alerting**: 15+ alert rules with Slack, email, and PagerDuty integration
- **Analytics API**: REST API for programmatic access to metrics
- **Data Collection**: Background agents running every 5 minutes
- **Automated Reporting**: Weekly summaries, monthly executive reports, and quarterly ROI reports

## Architecture

```
greenlang/monitoring/
├── dashboards/          # Grafana dashboard definitions
│   ├── infrastructure_usage.py
│   ├── cost_savings.py
│   ├── performance.py
│   ├── compliance.py
│   ├── productivity.py
│   └── health.py
├── alerts/              # Alert rules and handlers
│   └── alert_rules.py
├── api/                 # Analytics REST API
│   └── analytics_api.py
├── collectors/          # Data collection agents
│   ├── metrics_collector.py
│   ├── log_aggregator.py
│   ├── violation_scanner.py
│   └── health_checker.py
├── reports/             # Automated report generation
│   └── report_generator.py
├── config.yaml          # Central configuration
└── orchestrator.py      # Main orchestrator
```

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Required services
- Prometheus (http://localhost:9090)
- Grafana (http://localhost:3000)
- Redis (localhost:6379)
- PostgreSQL (localhost:5432)
```

### Configuration

1. Copy and configure `config.yaml`:

```bash
cp config.yaml.example config.yaml
```

2. Set environment variables:

```bash
export GRAFANA_API_KEY="your-grafana-api-key"
export SLACK_WEBHOOK_URL="your-slack-webhook"
export PAGERDUTY_INTEGRATION_KEY="your-pagerduty-key"
export API_KEY_PROD="your-api-key"
```

### Deploy Dashboards

```bash
# Generate and deploy all Grafana dashboards
python orchestrator.py --deploy-dashboards

# Deploy individual dashboard
python dashboards/infrastructure_usage.py
```

### Deploy Alert Rules

```bash
# Export alert rules
python alerts/alert_rules.py

# Import to Prometheus
# Copy prometheus_rules.json to Prometheus alert rules directory
```

### Run Data Collection

```bash
# Run all collectors once
python orchestrator.py --run-once

# Run continuously (production mode)
python orchestrator.py --continuous

# Run individual collector
python collectors/metrics_collector.py
```

### Start Analytics API

```bash
# Start API server
python api/analytics_api.py

# Access API at http://localhost:8080
# API docs at http://localhost:8080/docs
```

## Dashboards

### 1. Infrastructure Usage Dashboard

**File**: `dashboards/infrastructure_usage.py`

**Metrics**:
- Overall IUM gauge (0-100%)
- IUM trend over 7 days
- IUM by application (bar chart)
- IUM by team (bar chart)
- Top 10 files by IUM
- Bottom 10 files needing attention
- Component adoption heatmap
- Custom code hotspots treemap

**URL**: `http://localhost:3000/d/greenlang-ium`

### 2. Cost Savings Dashboard

**File**: `dashboards/cost_savings.py`

**Metrics**:
- Total cost savings (USD)
- ROI percentage
- LLM cost savings trend
- Cache hit rates by service
- Developer time savings (cumulative)
- Infrastructure vs custom development time
- Savings by optimization type

**Refresh**: Every 5 minutes

**URL**: `http://localhost:3000/d/greenlang-cost-savings`

### 3. Performance Monitoring Dashboard

**File**: `dashboards/performance.py`

**Metrics**:
- P95 latency (with 1s alert threshold)
- Error rate (with 1% alert threshold)
- Throughput (requests/second)
- Agent execution times (P50, P95, P99)
- Cache latency by layer (L1/L2/L3)
- Database query performance
- LLM response time distribution

**Refresh**: Every 30 seconds

**URL**: `http://localhost:3000/d/greenlang-performance`

### 4. Compliance & Quality Dashboard

**File**: `dashboards/compliance.py`

**Metrics**:
- IUM compliance for new PRs
- ADR coverage percentage
- Test coverage by application
- Enforcement violations by type
- Pre-commit hook success rate
- PR approval time distribution
- Code review feedback categories
- Security scan results

**URL**: `http://localhost:3000/d/greenlang-compliance`

### 5. Developer Productivity Dashboard

**File**: `dashboards/productivity.py`

**Metrics**:
- Time to first PR (new developers)
- Code reuse percentage
- LOC per developer (infrastructure vs custom)
- Infrastructure contributors leaderboard
- Time saved vs manual implementation
- Bug fix time comparison
- Team velocity (story points)
- Developer achievements

**Gamification**: Leaderboards and achievement badges

**URL**: `http://localhost:3000/d/greenlang-productivity`

### 6. Infrastructure Health Dashboard

**File**: `dashboards/health.py`

**Metrics**:
- Service health status grid
- Service uptime (24h)
- Cache service health (Redis)
- Database connection pool status
- LLM provider availability
- API rate limit utilization
- CPU/Memory/Disk usage
- Network throughput

**Refresh**: Every 30 seconds

**URL**: `http://localhost:3000/d/greenlang-health`

## Alert Rules

Total: **15+ production-ready alert rules**

### Critical Alerts

1. **IUM Below 90%** - Application-level IUM drop
2. **New PR IUM Below 95%** - PR violates IUM target
3. **P95 Latency > 1s** - SLA violation
4. **Error Rate > 1%** - High error rate
5. **Pre-commit Hook Disabled** - Security violation
6. **Custom LLM Wrapper Detected** - Policy violation
7. **Critical Security Vulnerability** - Security issue
8. **Service Down** - Service unavailability
9. **Redis Cache Unavailable** - Cache service down

### Warning Alerts

10. **Semantic Cache Hit Rate < 25%** - Low cache efficiency
11. **Agent Execution Time > 2x Baseline** - Performance degradation
12. **Missing ADR (>100 LOC)** - Compliance issue
13. **Test Coverage < 85%** - Quality issue
14. **High Resource Utilization** - Capacity warning
15. **LLM Provider Rate Limit** - Integration warning

### Alert Channels

- **Slack**: Real-time notifications to #greenlang-alerts
- **Email**: Detailed alerts to team email
- **PagerDuty**: Critical incidents requiring immediate response

## Analytics API

### Endpoints

```
GET  /                                  # API info
GET  /api/ium/overall                   # Overall IUM metrics
GET  /api/ium/by-app/{app_name}        # Application-specific IUM
GET  /api/ium/by-team/{team_name}      # Team-specific IUM
GET  /api/costs/savings                 # Cost savings metrics
GET  /api/performance/metrics           # Performance metrics
GET  /api/compliance/violations         # Compliance violations
GET  /api/leaderboard                   # Developer leaderboard
GET  /api/health                        # Infrastructure health
GET  /metrics                           # Prometheus metrics
```

### Authentication

All endpoints require API key authentication:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/ium/overall
```

### Example Usage

```python
import requests

headers = {"X-API-Key": "greenlang-api-key-prod"}

# Get overall IUM
response = requests.get(
    "http://localhost:8080/api/ium/overall",
    headers=headers
)
ium_data = response.json()
print(f"Overall IUM: {ium_data['overall_ium']}%")

# Get cost savings
response = requests.get(
    "http://localhost:8080/api/costs/savings?period_days=30",
    headers=headers
)
savings = response.json()
print(f"30-day savings: ${savings['total_savings_usd']:,.2f}")
```

## Data Collectors

### Metrics Collector

**File**: `collectors/metrics_collector.py`

**Frequency**: Every 5 minutes

**Collects**:
- IUM scores (application, file, component levels)
- Cost savings (LLM, developer time)
- Cache hit rates
- Performance metrics

**Output**: Prometheus Pushgateway

### Log Aggregator

**File**: `collectors/log_aggregator.py`

**Frequency**: Every 5 minutes

**Processes**:
- Structured JSON logs
- Unstructured text logs
- Performance logs
- Error logs
- Cache logs

**Output**: Elasticsearch

### Violation Scanner

**File**: `collectors/violation_scanner.py`

**Frequency**: Every 1 hour

**Detects**:
- Custom LLM wrappers
- Custom cache implementations
- Missing ADRs
- Hardcoded credentials
- Missing error handling
- Deprecated imports
- Missing test files

**Output**: Violation reports + database

### Health Checker

**File**: `collectors/health_checker.py`

**Frequency**: Every 1 minute

**Checks**:
- HTTP service health
- Redis availability
- PostgreSQL connectivity
- Service response times
- Resource utilization

**Output**: Prometheus metrics

## Automated Reporting

### Weekly Summary (Email)

**Schedule**: Every Monday at 9:00 AM

**Recipients**: platform-team@greenlang.io

**Content**:
- Key metrics (IUM, cost savings, performance)
- Highlights
- Top performers leaderboard
- Action items

**Format**: HTML email

### Monthly Executive Report (PDF)

**Schedule**: 1st of month at 9:00 AM

**Recipients**: executives@greenlang.io

**Content**:
- Executive summary
- Key achievements
- Financial impact
- ROI analysis
- Recommendations

**Format**: PDF

### Quarterly ROI Report (Slides)

**Schedule**: Quarterly

**Recipients**: executives@greenlang.io

**Content**:
- Quarterly cost savings
- ROI percentage
- Developer productivity gains
- Infrastructure adoption
- Next quarter goals

**Format**: JSON (convertible to PowerPoint)

### Incident Post-Mortem

**Trigger**: On-demand

**Content**:
- Incident summary
- Timeline
- Root cause analysis
- Impact assessment
- Resolution steps
- Action items
- Lessons learned

**Format**: Markdown

## Cron Setup

Add to crontab for automated execution:

```cron
# Metrics collection (every 5 minutes)
*/5 * * * * cd /path/to/greenlang && python monitoring/collectors/metrics_collector.py

# Log aggregation (every 5 minutes)
*/5 * * * * cd /path/to/greenlang && python monitoring/collectors/log_aggregator.py

# Health checks (every minute)
* * * * * cd /path/to/greenlang && python monitoring/collectors/health_checker.py

# Violation scanning (every hour)
0 * * * * cd /path/to/greenlang && python monitoring/collectors/violation_scanner.py

# Weekly report (Monday 9 AM)
0 9 * * 1 cd /path/to/greenlang && python monitoring/orchestrator.py --weekly-report

# Monthly report (1st of month, 9 AM)
0 9 1 * * cd /path/to/greenlang && python monitoring/orchestrator.py --monthly-report
```

## Docker Deployment

```dockerfile
# Dockerfile for monitoring system
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY greenlang/monitoring /app/monitoring

CMD ["python", "monitoring/orchestrator.py", "--continuous"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  monitoring:
    build: .
    environment:
      - GRAFANA_API_KEY=${GRAFANA_API_KEY}
      - SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL}
    volumes:
      - ./greenlang:/app/greenlang
    ports:
      - "8080:8080"
    depends_on:
      - prometheus
      - grafana
      - redis
      - postgres

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Metrics Reference

### IUM Metrics

- `greenlang_ium_score{application, team}` - IUM percentage
- `greenlang_file_ium_score{file_path, application}` - Per-file IUM
- `greenlang_component_usage{component, application}` - Component usage count

### Cost Metrics

- `greenlang_cost_savings_usd{application, optimization_type}` - Cost savings
- `greenlang_llm_cost_savings_usd{application, provider}` - LLM savings
- `greenlang_cache_hit_rate{service, cache_type}` - Cache efficiency
- `greenlang_developer_hours_saved{application, feature}` - Time savings

### Performance Metrics

- `greenlang_request_duration_seconds{service, endpoint}` - Request latency
- `greenlang_agent_execution_seconds{agent, application}` - Agent execution time
- `greenlang_cache_latency_seconds{layer, service}` - Cache latency
- `greenlang_request_total{service}` - Request count
- `greenlang_request_errors_total{service, error_type}` - Error count

### Health Metrics

- `greenlang_service_up{service}` - Service availability (1=up, 0=down)
- `greenlang_service_response_time_ms{service}` - Response time

## Troubleshooting

### Dashboards not appearing in Grafana

1. Check Grafana API key permissions
2. Verify Grafana URL in config.yaml
3. Check dashboard JSON syntax
4. Review Grafana logs: `docker logs grafana`

### Metrics not collecting

1. Verify Prometheus Pushgateway is running
2. Check collector logs for errors
3. Verify codebase_path in config.yaml
4. Test individual collector: `python collectors/metrics_collector.py`

### Alerts not firing

1. Check Prometheus alert rules loaded
2. Verify alertmanager configuration
3. Test Slack webhook URL
4. Check alert rule expressions

### API not accessible

1. Verify API is running: `curl http://localhost:8080/`
2. Check API key in headers
3. Review API logs
4. Verify port not in use

## Support

For issues or questions:

- **Documentation**: https://docs.greenlang.io/monitoring
- **Issues**: https://github.com/akshay-greenlang/Code-V1_GreenLang/issues
- **Slack**: #greenlang-monitoring
- **Email**: platform-team@greenlang.io

## License

Proprietary - GreenLang Platform Engineering Team

---

**Created by**: Monitoring & Observability Team
**Last Updated**: 2025-11-09
**Version**: 1.0.0
