# GL-011 FUELCRAFT Grafana Dashboards

## Overview

This directory contains three comprehensive Grafana dashboards for monitoring GL-011 FUELCRAFT, GreenLang's fuel procurement and optimization intelligence agent. These dashboards provide real-time visibility into operational performance, business KPIs, and executive-level strategic metrics.

## Dashboard Suite

### 1. Operations Dashboard
**File:** `operations_dashboard.json`
**UID:** `gl011-operations`
**Audience:** DevOps Engineers, SREs, Technical Operators
**Refresh:** 30 seconds
**Time Range:** Last 6 hours (default)

**Purpose:** Real-time technical monitoring of system health, API performance, integration status, and error tracking.

**Key Features:**
- Agent health status with traffic light indicators
- Request rate and latency percentiles (P50/P95/P99)
- CPU, memory, and thread monitoring
- Cache hit rate tracking (target: >85%)
- API call latency heatmaps
- Integration health table (ERP, market data, storage)
- Queue depth and concurrency metrics
- Calculator performance by type
- Provenance tracking counters
- Alert history and recent error logs
- Network traffic in/out
- Log volume breakdown by severity level

**Panels:** 20 operational monitoring panels

---

### 2. Business KPI Dashboard
**File:** `business_kpi_dashboard.json`
**UID:** `gl011-business-kpi`
**Audience:** Operations Managers, Procurement Teams, Business Analysts
**Refresh:** 1 minute
**Time Range:** Last 24 hours (default)

**Purpose:** Track fuel costs, savings, carbon emissions, supplier performance, and optimization effectiveness.

**Key Features:**
- Total fuel cost and cost savings vs baseline
- Carbon emissions and reduction metrics
- Energy delivered (MWh) and fuel efficiency
- Current fuel blend composition (pie chart)
- Fuel cost by type (stacked bar chart)
- Multi-fuel price trend analysis
- Inventory levels by fuel type
- Optimization recommendations applied
- Contract utilization by supplier
- Supplier performance scorecard
- Procurement order tracking
- Fuel quality violation alerts
- Emissions compliance status
- Cost per MWh trends
- ROI waterfall chart

**Panels:** 18 business-focused KPI panels

---

### 3. Executive Dashboard
**File:** `executive_dashboard.json`
**UID:** `gl011-executive`
**Audience:** Executives, C-Suite, Senior Stakeholders
**Refresh:** 5 minutes
**Time Range:** Last 30 days (default)

**Purpose:** High-level strategic overview for decision-makers, focusing on business impact and compliance.

**Key Features:**
- Monthly cost savings (big number)
- Year-to-date carbon reduction
- System uptime percentage (target: >99%)
- Cost optimization score (0-100)
- Environmental performance score (0-100)
- Critical alert count
- 90-day fuel cost trend (sparkline)
- Top 5 cost drivers table
- Compliance status grid (fuel quality, emissions, contracts)
- Optimization success rate
- Integration health summary
- Recent achievements tracker (savings, emissions avoided, efficiencies)

**Panels:** 12 executive-level strategic panels

---

## Quick Start

### Prerequisites

1. **Grafana 8.0+** installed and running
2. **Prometheus** data source configured
3. **GL-011 FUELCRAFT agent** running and exposing metrics
4. **Network access** to Prometheus from Grafana

### Installation Steps

#### Method 1: Grafana UI Import

1. Log in to Grafana
2. Navigate to **Dashboards** > **Import**
3. Click **Upload JSON file**
4. Select one of the dashboard files:
   - `operations_dashboard.json`
   - `business_kpi_dashboard.json`
   - `executive_dashboard.json`
5. Select your Prometheus data source
6. Click **Import**
7. Repeat for remaining dashboards

#### Method 2: API Import (Automated)

```bash
#!/bin/bash

GRAFANA_URL="http://localhost:3000"
GRAFANA_API_KEY="your-api-key-here"

for dashboard in operations_dashboard business_kpi_dashboard executive_dashboard; do
  curl -X POST "${GRAFANA_URL}/api/dashboards/db" \
    -H "Authorization: Bearer ${GRAFANA_API_KEY}" \
    -H "Content-Type: application/json" \
    -d @"${dashboard}.json"
done
```

#### Method 3: Provisioning (Infrastructure as Code)

Create a provisioning file: `/etc/grafana/provisioning/dashboards/gl011.yml`

```yaml
apiVersion: 1

providers:
  - name: 'GL-011 FUELCRAFT'
    orgId: 1
    folder: 'GreenLang Agents'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards/gl011
```

Copy dashboard JSON files to `/var/lib/grafana/dashboards/gl011/`

Restart Grafana:
```bash
sudo systemctl restart grafana-server
```

---

## Configuration

### Data Source Setup

1. Navigate to **Configuration** > **Data Sources**
2. Add **Prometheus** data source
3. Configure connection:
   - **URL:** `http://prometheus:9090` (or your Prometheus endpoint)
   - **Access:** Server (default) or Browser
   - **Scrape interval:** 15s (recommended)
4. Click **Save & Test**

### Template Variables

All dashboards use the following template variables:

**agent_id** (Required)
- **Type:** Query
- **Data Source:** Prometheus
- **Query:** `label_values(gl011_agent_health_status, agent_id)`
- **Default:** `gl011-prod-01`
- **Description:** Select which GL-011 agent instance to monitor

**fuel_type** (Optional, Business KPI & Operations)
- **Type:** Query
- **Data Source:** Prometheus
- **Query:** `label_values(gl011_fuel_cost_total, fuel_type)`
- **Multi-select:** Yes
- **Include All:** Yes
- **Description:** Filter metrics by fuel type (Natural Gas, Coal, Oil, Biomass, etc.)

**time_range** (Operations Dashboard)
- **Type:** Custom
- **Options:** 1h, 3h, 6h, 12h, 24h
- **Default:** 6h
- **Description:** Quick time range selector

### Alert Configuration

#### Step 1: Configure Notification Channels

Navigate to **Alerting** > **Notification channels** and add:

**Email Notifications**
```yaml
Type: Email
Addresses: devops@greenlang.io, ops-team@greenlang.io
Send on all alerts: Yes
```

**Slack Notifications**
```yaml
Type: Slack
Webhook URL: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
Channel: #gl011-alerts
Username: GL-011 Monitor
```

**PagerDuty (Critical Only)**
```yaml
Type: PagerDuty
Integration Key: YOUR_PAGERDUTY_KEY
Severity: Critical
Auto resolve: Yes
```

#### Step 2: Import Alert Rules

Create Prometheus alert rules file: `/etc/prometheus/rules/gl011_alerts.yml`

```yaml
groups:
  - name: gl011_critical_alerts
    interval: 30s
    rules:
      - alert: GL011AgentDown
        expr: gl011_agent_health_status{agent_id="gl011-prod-01"} == 0
        for: 1m
        labels:
          severity: critical
          agent: GL-011
        annotations:
          summary: "GL-011 FUELCRAFT agent is unhealthy"
          description: "Agent {{ $labels.agent_id }} health status is DOWN for more than 1 minute."

      - alert: GL011HighErrorRate
        expr: rate(gl011_errors_total{agent_id="gl011-prod-01"}[5m]) > 10
        for: 5m
        labels:
          severity: warning
          agent: GL-011
        annotations:
          summary: "High error rate detected on GL-011"
          description: "Error rate is {{ $value }} errors/sec (threshold: 10 errors/sec)."

      - alert: GL011LowCacheHitRate
        expr: gl011_cache_hit_rate{agent_id="gl011-prod-01"} < 0.70
        for: 10m
        labels:
          severity: warning
          agent: GL-011
        annotations:
          summary: "GL-011 cache hit rate below 70%"
          description: "Cache hit rate is {{ $value | humanizePercentage }} (target: >85%)."

      - alert: GL011ComplianceViolation
        expr: gl011_emissions_compliance_status{agent_id="gl011-prod-01"} == 0
        for: 0m
        labels:
          severity: critical
          agent: GL-011
        annotations:
          summary: "CRITICAL: GL-011 emissions compliance violated"
          description: "Emissions compliance status is NON-COMPLIANT. Immediate action required."

      - alert: GL011HighLatency
        expr: histogram_quantile(0.95, rate(gl011_request_duration_seconds_bucket{agent_id="gl011-prod-01"}[5m])) > 5
        for: 5m
        labels:
          severity: warning
          agent: GL-011
        annotations:
          summary: "GL-011 P95 latency exceeds 5 seconds"
          description: "P95 request latency is {{ $value }}s (threshold: 5s)."

      - alert: GL011IntegrationDown
        expr: gl011_integration_health{agent_id="gl011-prod-01"} == 0
        for: 2m
        labels:
          severity: critical
          agent: GL-011
        annotations:
          summary: "GL-011 integration {{ $labels.integration_type }} is DOWN"
          description: "Integration {{ $labels.integration_type }} has been unhealthy for 2+ minutes."
```

Reload Prometheus configuration:
```bash
curl -X POST http://localhost:9090/-/reload
```

---

## Usage Guide

### For DevOps Engineers (Operations Dashboard)

**Daily Tasks:**
1. Check agent health status (Panel 1.1) - should be GREEN
2. Monitor request rate and latency (Panels 1.2, 1.3)
3. Review active alerts (Panel 1.4) - should be 0
4. Check error rate trends (Panel 2.1)
5. Verify integration health (Panel 3.2) - all integrations UP

**Incident Response:**
1. If alert fires, check **Alert History** (Panel 6.1)
2. Review **Recent Errors** (Panel 6.4) for root cause
3. Check **API Call Latency Heatmap** (Panel 3.1) for performance degradation
4. Monitor **Queue Depth** (Panel 4.1) for backlog
5. Review **Log Volume** (Panel 6.2) for anomalies

**Performance Tuning:**
- Monitor **Cache Hit Rate** (Panel 2.5) - optimize if <85%
- Review **Calculator Performance** (Panel 5.1) - identify slow calculators
- Check **CPU/Memory Usage** (Panels 2.2, 2.3) - scale if consistently high

---

### For Business Analysts (Business KPI Dashboard)

**Daily Tasks:**
1. Review **Total Fuel Cost** (Panel 1.1) vs budget
2. Check **Cost Savings vs Baseline** (Panel 1.2) - track ROI
3. Monitor **Fuel Efficiency** (Panel 2.2) - target >85%
4. Review **Current Fuel Blend** (Panel 2.3) - ensure compliance with strategy
5. Check **Supplier Performance** (Panel 5.2) - identify top performers

**Weekly Review:**
1. Analyze **Fuel Price Trends** (Panel 3.2) - forecast future costs
2. Review **Inventory Levels** (Panel 4.1) - prevent stockouts
3. Check **Contract Utilization** (Panel 5.1) - optimize contracts
4. Review **Fuel Quality Violations** (Panel 6.2) - address supplier issues
5. Verify **Emissions Compliance** (Panel 7.1) - ensure regulatory adherence

**Monthly Reporting:**
1. Export **Fuel Cost by Type** (Panel 3.1) for management reports
2. Calculate **Cost per MWh Trend** (Panel 7.2) - track unit economics
3. Review **Optimization Recommendations Applied** (Panel 4.2) - measure system adoption
4. Prepare **ROI Tracker** (Panel 7.3) for stakeholder presentations

---

### For Executives (Executive Dashboard)

**Weekly Review:**
1. Check **Monthly Cost Savings** (Panel 1.1) - track financial impact
2. Review **YTD Carbon Reduction** (Panel 1.2) - measure environmental progress
3. Verify **System Uptime** (Panel 1.3) - ensure reliability (target: >99%)
4. Monitor **Critical Alerts** (Panel 2.2) - should be 0

**Monthly Board Meetings:**
1. Present **Cost Optimization Score** (Panel 1.4) - overall system effectiveness
2. Share **Environmental Performance** (Panel 2.1) - ESG reporting
3. Review **Top 5 Cost Drivers** (Panel 3.1) - strategic focus areas
4. Show **Compliance Status Grid** (Panel 3.2) - regulatory adherence
5. Highlight **Recent Achievements** (Panel 4.2) - celebrate wins

**Quarterly Strategy:**
1. Analyze **90-Day Fuel Cost Trend** (Panel 2.3) - identify patterns
2. Review **Optimization Success Rate** (Panel 3.3) - assess system maturity
3. Check **Integration Health Summary** (Panel 4.1) - technology stack status

---

## Customization

### Adding Custom Panels

1. Click **Add panel** in dashboard edit mode
2. Select visualization type (Time series, Stat, Gauge, Table, etc.)
3. Enter PromQL query (see metric list below)
4. Configure display options (legend, axes, thresholds)
5. Save dashboard

### Modifying Thresholds

Example: Change cache hit rate threshold from 85% to 90%

1. Edit Panel 2.5 (Operations Dashboard)
2. Navigate to **Field > Thresholds**
3. Change values:
   - Red: 0 - 0.7
   - Yellow: 0.7 - 0.9
   - Green: 0.9 - 1.0
4. Update panel title to reflect new target
5. Save dashboard

### Creating Custom Variables

1. Navigate to **Dashboard settings** > **Variables**
2. Click **Add variable**
3. Configure:
   - **Name:** `environment`
   - **Type:** Query
   - **Data Source:** Prometheus
   - **Query:** `label_values(gl011_agent_health_status, environment)`
4. Save variable
5. Use in queries: `gl011_agent_health_status{environment="$environment"}`

---

## Metrics Reference

### Operations Dashboard Metrics

```promql
# System Health
gl011_agent_health_status{agent_id}
gl011_cpu_usage_percent{agent_id}
gl011_memory_usage_bytes{agent_id, memory_type}
gl011_thread_count{agent_id}
gl011_cache_hit_rate{agent_id}

# Request Metrics
gl011_requests_total{agent_id, endpoint, status}
gl011_request_duration_seconds_bucket{agent_id, endpoint, le}
gl011_errors_total{agent_id, error_type, severity}

# Integration Metrics
gl011_integration_health{agent_id, integration_type}
gl011_queue_depth{agent_id, queue_name}

# Network & Logs
gl011_network_bytes_in{agent_id}
gl011_network_bytes_out{agent_id}
gl011_log_entries_total{agent_id, log_level}
```

### Business KPI Metrics

```promql
# Financial
gl011_fuel_cost_total{agent_id, fuel_type}
gl011_cost_savings_vs_baseline{agent_id, fuel_type}
gl011_fuel_price_per_unit{agent_id, fuel_type, unit}

# Environmental
gl011_carbon_emissions_kg{agent_id, fuel_type}
gl011_carbon_reduction_vs_baseline_kg{agent_id, fuel_type}

# Efficiency
gl011_energy_delivered_mwh{agent_id, fuel_type}
gl011_fuel_efficiency_percent{agent_id, fuel_type}
gl011_inventory_level_units{agent_id, fuel_type}

# Suppliers & Contracts
gl011_supplier_performance_score{agent_id, supplier_name}
gl011_contract_utilization_percent{agent_id, contract_id}
```

### Executive Metrics

```promql
# KPIs
gl011_monthly_cost_savings{agent_id}
gl011_ytd_carbon_reduction_kg{agent_id}
gl011_system_uptime_percent{agent_id}
gl011_cost_optimization_score{agent_id}

# Compliance
gl011_fuel_quality_compliance{agent_id}
gl011_emissions_compliance{agent_id}
gl011_contract_compliance{agent_id}
```

For complete metric specifications, see `DASHBOARD_SPECIFICATIONS.md`.

---

## Troubleshooting

### Issue: "No data" in panels

**Symptoms:** Panels show "No data" or empty graphs

**Causes:**
1. GL-011 agent not running or not exposing metrics
2. Prometheus not scraping GL-011 metrics endpoint
3. Incorrect `agent_id` template variable
4. Firewall blocking Prometheus scraping

**Solutions:**
```bash
# 1. Verify GL-011 is running
curl http://gl011-agent:8080/metrics

# 2. Check Prometheus targets
curl http://prometheus:9090/api/v1/targets | grep gl011

# 3. Verify metrics are scraped
curl http://prometheus:9090/api/v1/query?query=gl011_agent_health_status

# 4. Check template variable
# In Grafana, click agent_id dropdown - should show gl011-prod-01

# 5. Test connectivity
telnet prometheus 9090
```

---

### Issue: Slow dashboard loading

**Symptoms:** Dashboard takes >10 seconds to load

**Causes:**
1. Too many data points queried
2. High-cardinality metrics
3. Large time range (>7 days on Operations dashboard)
4. Prometheus query overload

**Solutions:**
```bash
# 1. Reduce time range
# Operations: Use 1h or 3h instead of 24h
# Business: Use 24h instead of 7d

# 2. Limit query range vectors
# Change [1h] to [5m] in rate() queries

# 3. Use recording rules
# In Prometheus, pre-aggregate metrics:
groups:
  - name: gl011_recording_rules
    interval: 30s
    rules:
      - record: gl011:fuel_cost:sum
        expr: sum(gl011_fuel_cost_total)

# 4. Enable query caching
# In Grafana datasource settings:
# Cache timeout: 60 seconds
```

---

### Issue: Incorrect metric values

**Symptoms:** Values don't match expected ranges (e.g., cost shows 0.05 instead of $50,000)

**Causes:**
1. Wrong metric unit (bytes vs GB, percentunit vs percent)
2. Missing aggregation function (sum, avg, rate)
3. Incorrect label filters

**Solutions:**
```promql
# 1. Check metric units in panel settings
# Unit: currencyUSD (not short)
# Unit: percent (not percentunit if metric is 0-100 scale)

# 2. Verify aggregation
# Wrong: gl011_fuel_cost_total
# Right: sum(gl011_fuel_cost_total{agent_id="$agent_id"})

# 3. Add label filters
# Wrong: gl011_fuel_cost_total
# Right: gl011_fuel_cost_total{agent_id="$agent_id", fuel_type=~"$fuel_type"}

# 4. Use instant queries for current values
# Format: instant = true (for tables and stats)
```

---

### Issue: Template variables not working

**Symptoms:** Dropdown shows "No options" or queries fail with undefined variable

**Causes:**
1. Data source not configured correctly
2. Metric doesn't exist or has no labels
3. Variable query syntax error

**Solutions:**
```bash
# 1. Test variable query manually
curl 'http://prometheus:9090/api/v1/label/agent_id/values'

# 2. Verify metric exists
curl 'http://prometheus:9090/api/v1/query?query=gl011_agent_health_status' | jq

# 3. Check variable configuration
# Query should be: label_values(gl011_agent_health_status, agent_id)
# NOT: gl011_agent_health_status{agent_id}

# 4. Refresh variable
# Click variable dropdown > Refresh icon
```

---

### Issue: Alerts not firing

**Symptoms:** Expected alerts don't trigger despite threshold breach

**Causes:**
1. Alert rules not loaded in Prometheus
2. Notification channels misconfigured
3. Alert evaluation interval too long
4. Query returns no data

**Solutions:**
```bash
# 1. Check Prometheus alert rules
curl http://prometheus:9090/api/v1/rules | jq '.data.groups[] | select(.name=="gl011_critical_alerts")'

# 2. Verify alerts are pending/firing
curl http://prometheus:9090/api/v1/alerts | grep GL011

# 3. Test notification channel
# In Grafana: Alerting > Notification channels > Test

# 4. Check alert query in Prometheus UI
# Navigate to http://prometheus:9090/graph
# Paste: gl011_agent_health_status{agent_id="gl011-prod-01"} == 0
```

---

## Performance Tuning

### Optimize Query Performance

**Use Recording Rules for Expensive Queries**

```yaml
# /etc/prometheus/rules/gl011_recording_rules.yml
groups:
  - name: gl011_performance
    interval: 30s
    rules:
      # Pre-aggregate total fuel cost
      - record: gl011:fuel_cost:sum
        expr: sum(gl011_fuel_cost_total)

      # Pre-calculate P95 latency
      - record: gl011:request_latency:p95
        expr: histogram_quantile(0.95, rate(gl011_request_duration_seconds_bucket[5m]))

      # Pre-aggregate cost by fuel type
      - record: gl011:fuel_cost_by_type:sum
        expr: sum by (fuel_type) (gl011_fuel_cost_by_type)
```

Use in dashboards:
```promql
# Instead of: sum(gl011_fuel_cost_total{agent_id="$agent_id"})
# Use: gl011:fuel_cost:sum{agent_id="$agent_id"}
```

**Reduce Cardinality**

```promql
# High cardinality (slow):
gl011_supplier_performance_score

# Reduced cardinality (fast):
gl011_supplier_performance_score{supplier_name=~"Top5Supplier.*"}

# Or use topk():
topk(5, gl011_supplier_performance_score)
```

---

## Backup & Recovery

### Export Dashboards

**Via UI:**
1. Open dashboard
2. Click **Settings** (gear icon)
3. Click **JSON Model**
4. Copy JSON
5. Save to file

**Via API:**
```bash
# Export all GL-011 dashboards
GRAFANA_URL="http://localhost:3000"
API_KEY="your-api-key"

for uid in gl011-operations gl011-business-kpi gl011-executive; do
  curl -H "Authorization: Bearer ${API_KEY}" \
    "${GRAFANA_URL}/api/dashboards/uid/${uid}" | jq '.dashboard' \
    > "${uid}_backup_$(date +%Y%m%d).json"
done
```

### Version Control Integration

```bash
# Initialize Git repo for dashboards
cd /var/lib/grafana/dashboards/gl011
git init
git add *.json
git commit -m "Initial GL-011 dashboard backup"

# Add remote
git remote add origin https://github.com/greenlang/grafana-dashboards.git
git push -u origin main

# Set up automated backups (cron job)
echo "0 2 * * * cd /var/lib/grafana/dashboards/gl011 && git add . && git commit -m 'Automated backup $(date)' && git push" | crontab -
```

---

## Security Best Practices

### Dashboard Access Control

**Role-Based Access:**
```yaml
# /etc/grafana/ldap.toml (LDAP integration)
[[servers.group_mappings]]
group_dn = "cn=devops,ou=groups,dc=greenlang,dc=io"
org_role = "Editor"
grafana_admin = false

[[servers.group_mappings]]
group_dn = "cn=executives,ou=groups,dc=greenlang,dc=io"
org_role = "Viewer"
grafana_admin = false
```

**Folder Permissions:**
1. Create folder: **GreenLang Agents**
2. Move GL-011 dashboards to folder
3. Set permissions:
   - DevOps team: Editor
   - Business team: Editor (Business KPI only)
   - Executive team: Viewer (Executive only)

### Data Privacy

**Redact Sensitive Data:**
```promql
# Hide actual costs in non-production environments
label_replace(
  gl011_fuel_cost_total{environment!="production"},
  "masked_cost",
  "REDACTED",
  "cost",
  ".*"
)
```

**Audit Logging:**
```ini
# /etc/grafana/grafana.ini
[log]
mode = console file
level = info

[log.console]
format = json

[security]
admin_user = admin
admin_password = ${GRAFANA_ADMIN_PASSWORD}
disable_gravatar = true
cookie_secure = true
cookie_samesite = strict
```

---

## Support & Maintenance

### Regular Maintenance Tasks

**Weekly:**
- [ ] Review dashboard load times (should be <3 seconds)
- [ ] Check for Grafana plugin updates
- [ ] Verify alert notification delivery

**Monthly:**
- [ ] Update dashboard thresholds based on performance trends
- [ ] Review and optimize slow PromQL queries
- [ ] Backup dashboards to Git repository
- [ ] Review user access permissions

**Quarterly:**
- [ ] Audit metric cardinality (use `prometheus_tsdb_symbol_table_size_bytes`)
- [ ] Review and sunset unused metrics
- [ ] Update dashboard documentation
- [ ] Conduct user training sessions

### Getting Help

**Internal Support:**
- GreenLang DevOps Team: devops@greenlang.io
- Slack: #gl011-support
- Confluence: https://wiki.greenlang.io/gl011/monitoring

**External Resources:**
- Grafana Documentation: https://grafana.com/docs/grafana/latest/
- Prometheus Query Examples: https://prometheus.io/docs/prometheus/latest/querying/examples/
- PromQL Cheat Sheet: https://promlabs.com/promql-cheat-sheet/

**Filing Issues:**
1. Check existing issues: https://github.com/greenlang/gl011/issues
2. Create new issue with template:
   - Dashboard affected
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots
   - Grafana/Prometheus versions

---

## Changelog

### Version 1.0.0 (2025-12-01)
- Initial release of GL-011 FUELCRAFT dashboard suite
- 3 dashboards: Operations, Business KPI, Executive
- 50 total panels across all dashboards
- Comprehensive metric coverage
- Alert integration
- Template variable support
- Multi-tenancy via agent_id variable

### Planned Enhancements (v1.1.0)
- Machine learning anomaly detection overlays
- Predictive cost forecasting panels
- What-if scenario modeling
- Mobile-optimized layouts
- Custom report scheduling

---

## License

Copyright 2025 GreenLang Foundation
Licensed under Apache 2.0

---

## Authors

**GreenLang DevOps Team**
- Lead DevOps Engineer: GL-DevOpsEngineer
- Monitoring Specialist: Prometheus/Grafana SME
- Business Analyst: KPI Definition Lead

**Contributors:**
- GL-011 FUELCRAFT Development Team
- GreenLang Operations Team
- Procurement & Finance Teams

---

## Appendix

### Complete Metric List

See `DASHBOARD_SPECIFICATIONS.md` for full metric catalog with descriptions, units, and label dimensions.

### Sample PromQL Queries

See `DASHBOARD_SPECIFICATIONS.md` Section "Prometheus Metric Requirements" for 50+ example queries.

### Dashboard Screenshots

Available in `docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-011/monitoring/screenshots/` (to be generated).

---

**Last Updated:** 2025-12-01
**Document Version:** 1.0.0
**Dashboard Version:** 1.0.0
