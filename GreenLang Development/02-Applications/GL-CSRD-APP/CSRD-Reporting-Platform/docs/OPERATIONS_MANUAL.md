# CSRD Reporting Platform - Operations Manual

**Version:** 1.0.0
**Last Updated:** 2025-10-18
**Target Audience:** DevOps engineers, system administrators, SREs

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Deployment Checklist](#deployment-checklist)
4. [Health Monitoring](#health-monitoring)
5. [Performance Benchmarks](#performance-benchmarks)
6. [Alerting Configuration](#alerting-configuration)
7. [Log Analysis](#log-analysis)
8. [Incident Response](#incident-response)
9. [Maintenance Tasks](#maintenance-tasks)
10. [Runbooks](#runbooks)

---

## Overview

### Purpose

This operations manual provides procedures for deploying, monitoring, and maintaining the CSRD Reporting Platform in production environments.

### Operational Model

```
Production Environment
├── Application Tier
│   ├── CSRD Platform (Python 3.10+)
│   ├── 6 AI Agents (containerized)
│   └── CLI/SDK interfaces
├── Data Tier
│   ├── Input data storage (S3/blob storage)
│   ├── ESRS catalogs (JSON)
│   ├── Compliance rules (YAML)
│   └── Output reports (JSON/XBRL)
├── Monitoring Tier
│   ├── Application logs (structured JSON)
│   ├── Metrics (Prometheus)
│   ├── Traces (OpenTelemetry)
│   └── Alerts (PagerDuty/Slack)
└── External Dependencies
    ├── LLM APIs (OpenAI/Anthropic)
    └── XBRL validator services
```

### Service Level Objectives (SLOs)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Availability** | 99.5% | Uptime during business hours (EU timezone) |
| **Latency (p95)** | < 30 min | Full 6-agent pipeline end-to-end |
| **Latency (p99)** | < 45 min | Full 6-agent pipeline end-to-end |
| **Success Rate** | 99% | Reports generated without critical errors |
| **Data Quality** | 95% | Reports meeting quality threshold (≥80/100) |
| **Compliance Rate** | 98% | Reports passing compliance validation |

### On-Call Responsibilities

**Severity Levels:**
- **P0 (Critical)**: Production down, data loss, compliance failure → Page immediately
- **P1 (High)**: Degraded performance, partial outage → Page during business hours
- **P2 (Medium)**: Non-critical issues, workarounds available → Next business day
- **P3 (Low)**: Cosmetic issues, enhancement requests → Backlog

**On-Call Rotation:**
- Primary: 24/7 on-call
- Secondary: Backup during business hours
- Escalation: Engineering lead → CTO

---

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Load Balancer / API Gateway                 │
│                    (Rate limiting, authentication)               │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CSRD Platform Application                     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Web UI (Optional)                                     │     │
│  │  - Report upload                                       │     │
│  │  - Status monitoring                                   │     │
│  │  - Results download                                    │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  API Layer (FastAPI/Flask)                            │     │
│  │  - POST /api/v1/reports                               │     │
│  │  - GET /api/v1/reports/{id}                           │     │
│  │  - GET /api/v1/reports/{id}/status                    │     │
│  │  - GET /health                                         │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Pipeline Orchestrator (csrd_pipeline.py)             │     │
│  │  - Job queue management                                │     │
│  │  - Agent coordination                                  │     │
│  │  - Progress tracking                                   │     │
│  │  - Error handling                                      │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Agent Layer (6 agents)                                   │  │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──┐ │  │
│  │  │Intake│→│Materi│→│Calcul│→│Aggreg│→│Report│→│Audit│││  │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Data Storage                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Inputs     │  │  Processed   │  │   Outputs    │          │
│  │  (S3/Blob)   │  │   (Redis)    │  │  (S3/Blob)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │   Configs    │  │   Catalogs   │                            │
│  │  (Git/S3)    │  │  (Read-only) │                            │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Observability Stack                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Logging    │  │   Metrics    │  │   Tracing    │          │
│  │ (CloudWatch) │  │(Prometheus)  │  │ (Jaeger)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Infrastructure Components

**Compute:**
- **Application servers**: 2-4 instances (auto-scaling)
- **CPU**: 4-8 cores per instance
- **Memory**: 16-32 GB per instance
- **Storage**: 100 GB SSD per instance

**Storage:**
- **Object storage (S3/Azure Blob)**: 1 TB for inputs/outputs
- **Cache (Redis)**: 16 GB for session state
- **Database (PostgreSQL, optional)**: 50 GB for metadata

**Networking:**
- **Load balancer**: Application Load Balancer (ALB) or equivalent
- **VPC/VNet**: Isolated network with private subnets
- **Security groups**: Restricted ingress/egress

**External Services:**
- **LLM APIs**: OpenAI or Anthropic (materiality assessment)
- **XBRL validator**: Optional third-party validation

### Resource Requirements

**Per Report Processing**:
- **CPU**: 2-4 cores (parallel agent execution)
- **Memory**: 8-16 GB (peak during calculation phase)
- **Storage**: 100-500 MB per report
- **Network**: ~10 MB for LLM API calls (materiality only)

**Scaling Capacity**:
- **Baseline**: 10 concurrent reports
- **Peak**: 50 concurrent reports (auto-scaling)
- **Storage**: 10,000 reports retention (1 year)

---

## Deployment Checklist

### Pre-Deployment

- [ ] **Infrastructure Provisioned**
  - [ ] Compute instances launched
  - [ ] Storage buckets created
  - [ ] Network configured (VPC, subnets, security groups)
  - [ ] Load balancer configured
  - [ ] DNS records created

- [ ] **Dependencies Installed**
  - [ ] Python 3.10+ installed
  - [ ] Required Python packages installed (from requirements.txt)
  - [ ] System packages installed (libxml2, libxslt, etc.)

- [ ] **Configuration Files Deployed**
  - [ ] ESRS data points catalog (data/esrs_data_points.json)
  - [ ] Emission factors database (data/emission_factors.json)
  - [ ] Compliance rules (rules/compliance_rules.yaml)
  - [ ] Data quality rules (rules/data_quality_rules.yaml)
  - [ ] ESRS formulas (rules/esrs_formulas.yaml)
  - [ ] Framework mappings (data/framework_mappings.json)

- [ ] **Secrets Management**
  - [ ] LLM API keys stored in secret manager (AWS Secrets Manager, Azure Key Vault)
  - [ ] Database credentials secured
  - [ ] API authentication tokens generated

- [ ] **Environment Variables Set**
  - [ ] `CSRD_ENV=production`
  - [ ] `CSRD_LOG_LEVEL=INFO`
  - [ ] `CSRD_OUTPUT_DIR=/mnt/outputs`
  - [ ] `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (from secret manager)

### Deployment

- [ ] **Application Deployment**
  - [ ] Code deployed to application servers (blue-green or rolling)
  - [ ] Service restarted with new code
  - [ ] Health checks passing

- [ ] **Smoke Tests**
  - [ ] Health endpoint responding (GET /health)
  - [ ] Sample report processed successfully
  - [ ] All 6 agents executing correctly

- [ ] **Monitoring Setup**
  - [ ] Logs flowing to centralized logging (CloudWatch, Splunk)
  - [ ] Metrics being collected (Prometheus, Datadog)
  - [ ] Dashboards configured
  - [ ] Alerts configured

### Post-Deployment

- [ ] **Verification**
  - [ ] End-to-end test report generated
  - [ ] Performance within SLO targets
  - [ ] No errors in logs
  - [ ] Alerting functional (test alert sent)

- [ ] **Documentation**
  - [ ] Deployment notes captured
  - [ ] Configuration changes documented
  - [ ] Runbooks updated

- [ ] **Communication**
  - [ ] Stakeholders notified of deployment
  - [ ] On-call team briefed
  - [ ] Change ticket closed

---

## Health Monitoring

### Health Check Endpoints

**Basic Health Check**:
```bash
curl http://csrd-platform.example.com/health
```

**Response (Healthy)**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-18T10:00:00Z",
  "version": "1.0.0",
  "checks": {
    "agents": {
      "status": "healthy",
      "details": {
        "intake_agent": "healthy",
        "materiality_agent": "healthy",
        "calculator_agent": "healthy",
        "aggregator_agent": "healthy",
        "reporting_agent": "healthy",
        "audit_agent": "healthy"
      }
    },
    "storage": {
      "status": "healthy",
      "details": {
        "input_storage": "accessible",
        "output_storage": "accessible"
      }
    },
    "external_dependencies": {
      "status": "healthy",
      "details": {
        "llm_api": "reachable",
        "xbrl_validator": "reachable"
      }
    },
    "resources": {
      "status": "healthy",
      "details": {
        "cpu_usage": 45,
        "memory_usage": 62,
        "disk_usage": 38
      }
    }
  }
}
```

**Response (Unhealthy)**:
```json
{
  "status": "unhealthy",
  "timestamp": "2025-10-18T10:00:00Z",
  "version": "1.0.0",
  "checks": {
    "agents": {
      "status": "unhealthy",
      "details": {
        "intake_agent": "healthy",
        "materiality_agent": "degraded",  // LLM API slow
        "calculator_agent": "healthy",
        "aggregator_agent": "healthy",
        "reporting_agent": "unhealthy",  // XBRL validator unreachable
        "audit_agent": "healthy"
      }
    }
  }
}
```

### Component Health Checks

**Agent Health Checks**:

Each agent exposes a health check method:
```python
# IntakeAgent
agent = IntakeAgent()
health = agent.health_check()
# Returns: {"status": "healthy", "last_successful_run": "2025-10-18T09:45:00Z"}

# MaterialityAgent (checks LLM API)
agent = MaterialityAgent(llm_config=config)
health = agent.health_check()
# Returns: {"status": "healthy", "llm_api_reachable": true, "llm_latency_ms": 234}

# CalculatorAgent (validates formulas loaded)
agent = CalculatorAgent()
health = agent.health_check()
# Returns: {"status": "healthy", "formulas_loaded": 156, "emission_factors_loaded": 342}
```

**Storage Health Checks**:
```bash
# Check input storage
aws s3 ls s3://csrd-platform-inputs/ --profile production

# Check output storage
aws s3 ls s3://csrd-platform-outputs/ --profile production

# Check disk space
df -h /mnt/outputs
```

**External Dependency Checks**:
```bash
# Check LLM API (OpenAI)
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Check LLM API (Anthropic)
curl -H "x-api-key: $ANTHROPIC_API_KEY" \
  https://api.anthropic.com/v1/messages
```

### Health Monitoring Schedule

| Check | Frequency | Threshold | Alert |
|-------|-----------|-----------|-------|
| Health endpoint | 30s | 3 consecutive failures | P1 |
| CPU usage | 1m | > 90% for 5m | P2 |
| Memory usage | 1m | > 95% for 5m | P1 |
| Disk usage | 5m | > 90% | P2 |
| LLM API latency | 1m | > 5s p95 | P2 |
| Report success rate | 5m | < 95% over 1h | P1 |

---

## Performance Benchmarks

### Expected Performance

**6-Agent Pipeline (Full Report)**:

| Agent | Expected Time | Memory Peak | Notes |
|-------|---------------|-------------|-------|
| IntakeAgent | 10-30s | 1-2 GB | Depends on input file size |
| MaterialityAgent | 2-5 min | 2-4 GB | LLM API calls (parallel) |
| CalculatorAgent | 30-60s | 4-8 GB | Deterministic calculations |
| AggregatorAgent | 10-20s | 1-2 GB | Framework mapping |
| ReportingAgent | 30-60s | 2-4 GB | XBRL generation |
| AuditAgent | 1-3 min | 2-3 GB | 215+ compliance rules |
| **Total** | **5-12 min** | **8-16 GB** | Typical report |

**Individual Agent Benchmarks**:

```bash
# IntakeAgent (1000 data points)
time csrd validate --input sample_1000.csv
# Expected: ~15 seconds

# CalculatorAgent (150 metrics)
time csrd calculate --input validated_data.json
# Expected: ~45 seconds

# AuditAgent (215 rules)
time csrd audit --report report.json
# Expected: ~2 minutes
```

### Performance Testing

**Load Test Script**:
```bash
#!/bin/bash
# load_test.sh - Concurrent report generation

NUM_CONCURRENT=10
for i in $(seq 1 $NUM_CONCURRENT); do
  csrd run --input data_${i}.csv --company-profile company_${i}.json \
    --output-dir output/${i} --quiet &
done

wait
echo "All $NUM_CONCURRENT reports completed"
```

**Expected Results**:
- 10 concurrent reports: 5-15 minutes total
- 50 concurrent reports: 10-20 minutes total (auto-scaling kicks in)

### Performance Optimization

**Tuning Parameters**:

```python
# Increase parallel processing
config = CSRDConfig(
    parallel_agents=True,  # Run independent agents in parallel
    max_workers=8          # Thread pool size
)

# Skip optional steps for speed
report = csrd_build_report(
    esg_data="data.csv",
    company_profile="company.json",
    skip_materiality=True,  # Skip if not needed (3-5 min savings)
    skip_audit=False        # Always audit
)

# Use smaller LLM model for faster materiality
config = CSRDConfig(
    llm_model="gpt-3.5-turbo"  # Faster than gpt-4o (but less accurate)
)
```

**Caching**:
```bash
# Cache emission factors and formulas in memory
export CSRD_CACHE_ENABLED=true
export CSRD_CACHE_TTL=3600  # 1 hour

# Pre-load catalogs on startup
python -m csrd_platform preload --esrs-catalog --emission-factors
```

---

## Alerting Configuration

### Alert Rules

**Critical Alerts (P0)**:

```yaml
# Prometheus alert rules
groups:
  - name: csrd_critical
    interval: 30s
    rules:
      - alert: CSRDPlatformDown
        expr: up{job="csrd-platform"} == 0
        for: 2m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "CSRD Platform is down"
          description: "Instance {{ $labels.instance }} has been down for more than 2 minutes"
          runbook: "https://docs.example.com/runbooks/platform-down"

      - alert: CSRDHighErrorRate
        expr: rate(csrd_reports_failed_total[5m]) > 0.05
        for: 10m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "High report failure rate"
          description: "{{ $value | humanizePercentage }} of reports failing"
          runbook: "https://docs.example.com/runbooks/high-error-rate"

      - alert: CSRDComplianceFailure
        expr: csrd_compliance_failures_total > 0
        for: 1m
        labels:
          severity: critical
          team: compliance
        annotations:
          summary: "Compliance validation failures detected"
          description: "{{ $value }} reports failed compliance validation"
          runbook: "https://docs.example.com/runbooks/compliance-failure"
```

**Warning Alerts (P1)**:

```yaml
  - name: csrd_warnings
    interval: 1m
    rules:
      - alert: CSRDHighLatency
        expr: histogram_quantile(0.95, rate(csrd_pipeline_duration_seconds_bucket[5m])) > 1800  # 30 min
        for: 10m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High pipeline latency"
          description: "P95 latency is {{ $value | humanizeDuration }}"
          runbook: "https://docs.example.com/runbooks/high-latency"

      - alert: CSRDLowDataQuality
        expr: avg(csrd_data_quality_score) < 80
        for: 30m
        labels:
          severity: warning
          team: data-quality
        annotations:
          summary: "Low data quality scores"
          description: "Average quality score is {{ $value }}"
          runbook: "https://docs.example.com/runbooks/low-data-quality"

      - alert: CSRDHighMemoryUsage
        expr: process_resident_memory_bytes / 1e9 > 28  # 28 GB (out of 32 GB)
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanize }}GB"
          runbook: "https://docs.example.com/runbooks/high-memory"
```

### Alert Channels

**PagerDuty Integration**:
```yaml
# prometheus/alertmanager.yml
receivers:
  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '<PAGERDUTY_SERVICE_KEY>'
        severity: 'critical'

  - name: 'pagerduty-warning'
    pagerduty_configs:
      - service_key: '<PAGERDUTY_SERVICE_KEY>'
        severity: 'warning'

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10m
  repeat_interval: 12h
  receiver: 'pagerduty-warning'
  routes:
    - match:
        severity: critical
      receiver: pagerduty-critical
```

**Slack Integration**:
```yaml
receivers:
  - name: 'slack-csrd-alerts'
    slack_configs:
      - api_url: '<SLACK_WEBHOOK_URL>'
        channel: '#csrd-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

### Alert Testing

```bash
# Send test alert
curl -X POST http://prometheus:9090/api/v1/alerts \
  -d '{"alerts": [{"labels": {"alertname": "TestAlert", "severity": "warning"}}]}'

# Verify alert routing
amtool alert query --alertmanager.url=http://localhost:9093

# Test PagerDuty integration
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H 'Content-Type: application/json' \
  -d '{"routing_key": "<KEY>", "event_action": "trigger", "payload": {"summary": "Test alert", "severity": "warning", "source": "csrd-platform"}}'
```

---

## Log Analysis

### Structured Logging Format

All logs are emitted in JSON format for easy parsing:

```json
{
  "timestamp": "2025-10-18T10:30:45.123Z",
  "level": "INFO",
  "logger": "csrd_platform.agents.intake_agent",
  "message": "Data validation completed",
  "context": {
    "report_id": "CSRD-20241018-001",
    "agent": "IntakeAgent",
    "company_name": "Acme Manufacturing GmbH",
    "total_records": 1234,
    "valid_records": 1210,
    "invalid_records": 24,
    "quality_score": 92.5,
    "processing_time_seconds": 15.3
  },
  "trace_id": "abc123def456",
  "span_id": "789ghi012jkl"
}
```

### Log Levels

| Level | Usage | Examples |
|-------|-------|----------|
| **DEBUG** | Detailed diagnostic info | Variable values, loop iterations |
| **INFO** | Normal operation events | Agent started, report completed |
| **WARNING** | Unexpected but handled | Data quality below threshold, LLM API slow |
| **ERROR** | Errors causing operation failure | Validation failed, calculation error |
| **CRITICAL** | System-level failures | Out of memory, storage unavailable |

**Production Setting**: `LOG_LEVEL=INFO`

### Common Log Queries

**CloudWatch Logs Insights**:

```sql
-- Find failed reports
fields @timestamp, context.report_id, context.error_message
| filter level = "ERROR"
| filter logger = "csrd_platform.pipeline"
| sort @timestamp desc
| limit 20

-- Data quality issues
fields @timestamp, context.report_id, context.quality_score, context.invalid_records
| filter level = "WARNING"
| filter logger = "csrd_platform.agents.intake_agent"
| filter context.quality_score < 80
| sort @timestamp desc

-- Slow reports (> 30 min)
fields @timestamp, context.report_id, context.processing_time_seconds
| filter level = "INFO"
| filter message = "Pipeline completed"
| filter context.processing_time_seconds > 1800
| sort context.processing_time_seconds desc

-- Compliance failures
fields @timestamp, context.report_id, context.compliance_status, context.critical_failures
| filter level = "ERROR"
| filter logger = "csrd_platform.agents.audit_agent"
| filter context.compliance_status = "FAIL"
| sort @timestamp desc
```

**Elasticsearch/Kibana Queries**:

```json
{
  "query": {
    "bool": {
      "must": [
        {"match": {"level": "ERROR"}},
        {"range": {"@timestamp": {"gte": "now-1h"}}}
      ]
    }
  },
  "sort": [{"@timestamp": {"order": "desc"}}]
}
```

### Log Retention

| Environment | Retention | Location |
|-------------|-----------|----------|
| Development | 7 days | Local files |
| Staging | 30 days | CloudWatch/Splunk |
| Production | 90 days | CloudWatch/Splunk (hot), S3 (archive) |

---

## Incident Response

### Incident Severity Matrix

| Severity | Definition | Response Time | Examples |
|----------|------------|---------------|----------|
| **P0** | Production completely down, data loss | 15 min | Platform unreachable, database corrupted |
| **P1** | Major functionality broken, no workaround | 1 hour | All reports failing, compliance always failing |
| **P2** | Partial functionality affected, workaround exists | 4 hours | Single agent failing, slow performance |
| **P3** | Minor issue, cosmetic | Next business day | UI typo, non-critical warning |

### Incident Response Process

**Step 1: Acknowledge (within 15 min for P0/P1)**
```bash
# Check system status
curl http://csrd-platform.example.com/health

# Check recent logs
aws logs tail /aws/csrd-platform --follow

# Check metrics
open https://grafana.example.com/d/csrd-platform

# Acknowledge in PagerDuty
```

**Step 2: Assess**
- What is broken?
- What is the impact? (How many users/reports affected?)
- What is the root cause? (Initial hypothesis)

**Step 3: Communicate**
```markdown
**INCIDENT: CSRD Platform Degraded Performance**

**Status**: Investigating
**Severity**: P1
**Impact**: Reports taking >45 min (expected <30 min)
**Affected**: ~30 reports in queue
**Started**: 2025-10-18 10:30 UTC
**Updates**: Every 30 min

**Current Actions**:
- Checking LLM API latency
- Reviewing application logs
- Monitoring auto-scaling status

**Next Update**: 11:00 UTC
```

**Step 4: Mitigate**
- Implement immediate fix or workaround
- Restore service to acceptable level
- Document temporary measures

**Step 5: Resolve**
- Implement permanent fix
- Verify issue resolved
- Monitor for recurrence

**Step 6: Post-Mortem**
- Document timeline
- Identify root cause
- Action items to prevent recurrence
- Share learnings

### Common Incident Scenarios

#### Scenario 1: Platform Unreachable

**Symptoms**:
- Health check failing
- All requests timing out

**Troubleshooting**:
```bash
# Check application servers
aws ec2 describe-instance-status --instance-ids i-xxxxx

# Check load balancer
aws elbv2 describe-target-health --target-group-arn arn:...

# Check security groups
aws ec2 describe-security-groups --group-ids sg-xxxxx

# Check application logs
aws logs tail /aws/csrd-platform --since 10m
```

**Resolution**:
- Restart application servers
- Check for deployment issues
- Verify network connectivity

#### Scenario 2: High Error Rate

**Symptoms**:
- >5% of reports failing

**Troubleshooting**:
```bash
# Find error patterns
aws logs filter-log-events --log-group-name /aws/csrd-platform \
  --filter-pattern 'ERROR' --start-time $(date -d '1 hour ago' +%s)000

# Check specific agent failures
grep "ERROR" /var/log/csrd-platform/agents/*.log

# Check LLM API status
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

**Resolution**:
- Identify failing agent
- Check external dependencies (LLM API, XBRL validator)
- Review recent code changes
- Rollback if needed

#### Scenario 3: Slow Performance

**Symptoms**:
- Reports taking >45 min (p99)

**Troubleshooting**:
```bash
# Check resource usage
top -b -n 1 | head -20
free -h
df -h

# Check LLM API latency
time curl -X POST https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]}'

# Check concurrent report count
redis-cli LLEN csrd:report_queue

# Check auto-scaling status
aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names csrd-platform-asg
```

**Resolution**:
- Scale out application servers
- Optimize slow agents
- Check LLM API quotas
- Consider materiality caching

---

## Maintenance Tasks

### Daily Maintenance

**Health Check Review** (10 min):
```bash
# Check health dashboard
open https://grafana.example.com/d/csrd-platform-health

# Review error logs
aws logs filter-log-events --log-group-name /aws/csrd-platform \
  --filter-pattern 'ERROR' --start-time $(date -d '24 hours ago' +%s)000 | jq '.events | length'

# Check report success rate
aws cloudwatch get-metric-statistics \
  --namespace CSRD \
  --metric-name ReportSuccessRate \
  --start-time $(date -d '24 hours ago' --iso-8601) \
  --end-time $(date --iso-8601) \
  --period 3600 \
  --statistics Average
```

**Disk Space Cleanup** (5 min):
```bash
# Check disk usage
df -h /mnt/outputs

# Clean up old reports (>90 days)
find /mnt/outputs -type f -mtime +90 -delete

# Archive to S3
aws s3 sync /mnt/outputs/archive/ s3://csrd-platform-archive/ --storage-class GLACIER
```

### Weekly Maintenance

**Performance Review** (30 min):
```bash
# Review p95/p99 latency trends
open https://grafana.example.com/d/csrd-platform-performance

# Review top 10 slowest reports
aws logs insights query \
  --log-group-name /aws/csrd-platform \
  --query-string 'fields @timestamp, context.report_id, context.processing_time_seconds | filter message = "Pipeline completed" | sort context.processing_time_seconds desc | limit 10' \
  --start-time $(date -d '7 days ago' +%s) \
  --end-time $(date +%s)

# Review data quality trends
aws cloudwatch get-metric-statistics \
  --namespace CSRD \
  --metric-name AverageDataQualityScore \
  --start-time $(date -d '7 days ago' --iso-8601) \
  --end-time $(date --iso-8601) \
  --period 86400 \
  --statistics Average
```

**Dependency Updates** (1 hour):
```bash
# Check for Python package updates
pip list --outdated

# Update non-breaking dependencies
pip install --upgrade <package>

# Run regression tests
pytest tests/ --cov=csrd_platform

# Deploy to staging
./deploy.sh staging
```

### Monthly Maintenance

**Capacity Planning** (2 hours):
- Review usage trends
- Forecast storage needs
- Plan auto-scaling adjustments
- Review cost optimization opportunities

**Security Updates** (2 hours):
- Apply OS security patches
- Update Python packages with CVE fixes
- Rotate API keys and credentials
- Review access logs for anomalies

**Disaster Recovery Test** (4 hours):
- Test backup restoration
- Validate failover procedures
- Update runbooks based on findings

---

## Runbooks

### Runbook: Platform Down

**Trigger**: Health check failing for >2 min

**Steps**:

1. **Verify outage**
   ```bash
   curl http://csrd-platform.example.com/health
   # Expected: Timeout or 503 error
   ```

2. **Check application servers**
   ```bash
   aws ec2 describe-instance-status --instance-ids i-xxxxx i-yyyyy
   # Look for: StatusCheck failures
   ```

3. **Check load balancer**
   ```bash
   aws elbv2 describe-target-health --target-group-arn arn:...
   # Look for: unhealthy targets
   ```

4. **Check application logs**
   ```bash
   aws logs tail /aws/csrd-platform --since 10m --follow
   # Look for: ERROR or CRITICAL messages
   ```

5. **Restart application**
   ```bash
   # Option 1: Graceful restart
   systemctl restart csrd-platform

   # Option 2: Force restart
   aws ec2 reboot-instances --instance-ids i-xxxxx i-yyyyy
   ```

6. **Verify recovery**
   ```bash
   curl http://csrd-platform.example.com/health
   # Expected: 200 OK with "status": "healthy"
   ```

7. **Post-incident**
   - Document timeline
   - Identify root cause
   - Update monitoring if needed

---

### Runbook: High Memory Usage

**Trigger**: Memory usage >95% for >5 min

**Steps**:

1. **Identify memory hog**
   ```bash
   ps aux --sort=-%mem | head -10
   # Look for: csrd-platform processes
   ```

2. **Check for memory leaks**
   ```bash
   # Get process ID
   PID=$(pgrep -f csrd-platform)

   # Monitor memory over time
   watch -n 5 "ps -p $PID -o %mem,rss,vsz"
   ```

3. **Check concurrent reports**
   ```bash
   redis-cli LLEN csrd:report_queue
   # High queue = high memory usage
   ```

4. **Temporary mitigation**
   ```bash
   # Option 1: Clear queue (if safe)
   redis-cli DEL csrd:report_queue

   # Option 2: Scale out
   aws autoscaling set-desired-capacity \
     --auto-scaling-group-name csrd-platform-asg \
     --desired-capacity 6
   ```

5. **Long-term fix**
   - Optimize agent memory usage
   - Increase instance memory
   - Implement memory limits per report

---

### Runbook: LLM API Outage

**Trigger**: MaterialityAgent failing with API errors

**Steps**:

1. **Verify LLM API status**
   ```bash
   # OpenAI
   curl https://status.openai.com/api/v2/status.json

   # Anthropic
   curl https://status.anthropic.com/api/v2/status.json
   ```

2. **Check API quota**
   ```bash
   # Check usage
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/usage?date=$(date +%Y-%m-%d)
   ```

3. **Implement fallback**
   ```python
   # Option 1: Use alternative provider
   config = CSRDConfig(
       llm_provider="anthropic",  # Switch from OpenAI to Anthropic
       llm_model="claude-3-sonnet-20240229"
   )

   # Option 2: Skip materiality (if urgent)
   report = csrd_build_report(
       esg_data="data.csv",
       company_profile="company.json",
       skip_materiality=True  # Use pre-existing materiality
   )
   ```

4. **Communicate impact**
   - Notify users of materiality assessment unavailability
   - Provide ETA for restoration
   - Offer manual materiality assessment option

---

**End of Operations Manual**

For more information, see:
- [User Guide](USER_GUIDE.md) - How to use the platform
- [API Reference](API_REFERENCE.md) - Technical documentation
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Installation and setup
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
- [Compliance Guide](COMPLIANCE_GUIDE.md) - CSRD/ESRS compliance requirements
