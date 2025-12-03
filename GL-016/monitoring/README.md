# GL-016 WATERGUARD - Monitoring Documentation

## Overview

This directory contains comprehensive monitoring and observability infrastructure for GL-016 WATERGUARD, the intelligent boiler water treatment optimization agent. The monitoring system provides real-time visibility into water chemistry, chemical optimization, agent health, and operational performance.

## Table of Contents

- [Architecture](#architecture)
- [Dashboard Descriptions](#dashboard-descriptions)
- [Alert Explanations](#alert-explanations)
- [Integration Instructions](#integration-instructions)
- [Metrics Reference](#metrics-reference)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Best Practices](#best-practices)

## Architecture

### Monitoring Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    GL-016 WATERGUARD Agent                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Water      │  │  Chemical    │  │    Agent     │      │
│  │  Chemistry   │  │ Optimization │  │   Health     │      │
│  │   Module     │  │    Module    │  │   Monitor    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│                    Metrics Exporter                          │
│                   (Prometheus Format)                        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Prometheus    │
                    │   (Scraper &    │
                    │  Time Series)   │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
         ┌─────────────┐          ┌─────────────┐
         │  Alertmanager│          │   Grafana   │
         │  (Alerting)  │          │ (Visualization)│
         └──────┬──────┘          └─────────────┘
                │
                ▼
         ┌─────────────┐
         │   PagerDuty │
         │    Email    │
         │    Slack    │
         └─────────────┘
```

### Data Flow

1. **Metrics Collection**: WATERGUARD agent exposes metrics endpoint at `/metrics`
2. **Scraping**: Prometheus scrapes metrics every 15-30 seconds
3. **Storage**: Time-series data stored in Prometheus with 90-day retention
4. **Alerting**: Prometheus evaluates alert rules and sends to Alertmanager
5. **Visualization**: Grafana queries Prometheus for dashboard display
6. **Notification**: Alertmanager routes alerts to appropriate channels

## Dashboard Descriptions

### 1. Comprehensive Overview Dashboard
**File**: `grafana/waterguard_dashboard.json`
**UID**: `waterguard-overview`
**Refresh**: 30 seconds

#### Purpose
Primary operational dashboard providing comprehensive real-time view of all WATERGUARD systems across multiple boilers.

#### Panels

##### Water Chemistry Overview
- **pH Level**: Real-time pH with control limits (target: 11.0-11.8)
- **Conductivity**: Electrical conductivity in µS/cm (limit: < 3000)
- **TDS**: Total Dissolved Solids in ppm (limit: < 2000)
- **Dissolved Oxygen**: Critical corrosion parameter in ppb (target: < 5)

##### Blowdown Metrics
- **Blowdown Rate**: Percentage of feedwater blown down (typical: 4-8%)
- **Cycles of Concentration**: Efficiency metric (target: 12-25 cycles)
- **Heat Loss**: Energy lost through blowdown in MMBtu/hr

##### Chemical Dosing
- **Dosing Rates**: Real-time chemical feed rates (phosphate, sulfite, polymer, caustic)
- **Tank Levels**: Chemical inventory levels with low-level alerts (< 20%)

##### Scale/Corrosion Risk
- **Scale Risk Index**: Predictive index 0-100 (safe: < 70)
- **Corrosion Risk Index**: Predictive index 0-100 (safe: < 70)
- **Prediction**: 24-hour scale formation probability

##### Compliance Status
- **Status Indicator**: Binary compliant/non-compliant
- **Violations Table**: Active parameter violations

##### Agent Health
- **Latency**: p99 request latency (target: < 100ms)
- **Error Rate**: Errors per second (target: < 1)
- **Throughput**: Requests per second processed

##### Cost Savings
- **Water Savings**: Daily gallons saved
- **Chemical Savings**: Daily cost reduction
- **Energy Savings**: Daily MMBtu conserved
- **Total Savings**: Aggregate daily cost savings

##### Historical Trends
- **30-Day Cost Trend**: Rolling savings visualization

##### Alert Status
- **Active Alerts Table**: All firing alerts with severity

#### Variables
- `boiler_id`: Filter by specific boiler (multi-select)
- `facility`: Filter by facility location
- `datasource`: Prometheus datasource selection

#### Use Cases
- Real-time operations monitoring
- Multi-boiler facility overview
- Executive performance dashboards
- Shift handoff briefings

---

### 2. Water Chemistry Trends Dashboard
**File**: `grafana/water_chemistry_trends.json`
**UID**: `waterguard-chemistry`
**Refresh**: 1 minute

#### Purpose
Detailed chemical parameter trending for water treatment specialists and chemists.

#### Panels

##### pH Monitoring
- **pH Trending**: Actual vs setpoint vs control limits
- **Current pH Stat**: Large numerical display with color coding
- **pH Gauge**: Visual gauge showing position within range

##### Conductivity & TDS
- **Conductivity Trending**: Real-time with setpoint and alarm limit
- **TDS Trending**: Total dissolved solids with limits

##### Silica & Phosphate
- **Silica Concentration**: Critical for turbine protection (< 150 ppm)
- **Phosphate Residual**: Scale protection indicator (target: 15-35 ppm)

##### Dissolved Oxygen & Scavenger
- **DO Level**: Critical corrosion parameter
- **Sulfite Residual**: Oxygen scavenger effectiveness

##### Hardness & Alkalinity
- **Total Hardness**: Calcium/magnesium content
- **Total Alkalinity**: Buffer capacity

##### Chloride & Sulfate
- **Chloride**: Corrosion accelerator (< 400 ppm)
- **Sulfate**: Monitored for balance

##### Metal Content
- **Iron**: Corrosion product indicator (< 0.1 ppm)
- **Copper**: System corrosion indicator (< 0.05 ppm)

#### Variables
- `boiler_id`: Single boiler selection
- `facility`: Facility location
- `datasource`: Prometheus datasource

#### Use Cases
- Water treatment optimization
- Trend analysis for chemists
- Root cause investigation
- Regulatory compliance documentation
- Laboratory result validation

---

### 3. Chemical Optimization Dashboard
**File**: `grafana/chemical_optimization.json`
**UID**: `waterguard-chemical-optimization`
**Refresh**: 5 minutes

#### Purpose
Financial and efficiency analysis for chemical treatment programs.

#### Panels

##### Chemical Consumption Overview
- **Consumption Trends**: Rate of chemical usage by type
- **Consumption Statistics**: Mean, max, current usage

##### Cost Analysis
- **Total Chemical Cost**: Period cost aggregate
- **Cost per Unit Steam**: Efficiency metric ($/1000 lb steam)
- **Cost Reduction**: Percentage vs baseline
- **Total Savings**: Dollar savings in period

##### Cost Trending
- **Cost per Unit Steam Trend**: Actual vs baseline over time

##### Dosing Efficiency
- **Chemical Efficiency**: Percentage efficiency by chemical type
- **Average Efficiency Gauges**: Visual efficiency indicators

##### Tank Inventory Management
- **Current Tank Levels**: Bar chart of all chemical tanks
- **Inventory Details Table**: Gallons remaining by chemical
- **Tank Level Trends**: Historical level tracking

##### Reorder Predictions
- **Reorder Forecast Table**: Days until reorder needed by chemical
  - Color coded: Red (< 7 days), Yellow (7-14 days), Green (> 14 days)

##### ROI Tracking
- **Monthly ROI**: Return on investment percentage
- **Payback Period**: Days to recover agent costs
- **Net Savings**: Total savings minus operating costs
- **Cumulative ROI Trend**: Long-term value visualization

#### Variables
- `boiler_id`: Multi-select boiler filter
- `facility`: Facility location
- `timerange`: Analysis period (1h, 24h, 7d, 30d)
- `datasource`: Prometheus datasource

#### Use Cases
- Financial reporting
- Chemical vendor negotiations
- Procurement planning
- ROI justification
- Budget forecasting
- Sustainability reporting

## Alert Explanations

### Critical Alerts

#### WaterTreatmentPHCriticalHigh
**Threshold**: pH > 12.0 for 2 minutes
**Severity**: Critical
**Impact**: Risk of caustic embrittlement and stress corrosion cracking

**What It Means**: Boiler water has excessive alkalinity, creating conditions for caustic attack on metal surfaces under stress.

**Immediate Actions**:
1. Reduce caustic feed immediately
2. Verify pH probe accuracy
3. Check blowdown operation
4. Contact water treatment specialist

**Root Causes**:
- Excessive caustic dosing
- Failed pH probe
- Blowdown system malfunction
- Condensate contamination with caustic

---

#### WaterTreatmentPHCriticalLow
**Threshold**: pH < 10.5 for 2 minutes
**Severity**: Critical
**Impact**: Risk of acidic corrosion and metal loss

**What It Means**: Boiler water lacks sufficient alkalinity to prevent acidic corrosion.

**Immediate Actions**:
1. Increase caustic feed immediately
2. Verify pH probe accuracy
3. Check for acid contamination
4. Review feedwater treatment

**Root Causes**:
- Insufficient caustic dosing
- Failed pH probe
- Acid contamination
- CO2 ingress from condensate

---

#### WaterTreatmentDissolvedOxygenCritical
**Threshold**: DO > 7 ppb for 5 minutes
**Severity**: Critical
**Impact**: Accelerated oxygen pitting corrosion

**What It Means**: Dissolved oxygen in boiler water will cause localized pitting corrosion.

**Immediate Actions**:
1. Increase oxygen scavenger (sulfite) feed
2. Check deaerator operation
3. Inspect for air in-leakage
4. Review feedwater heater performance

**Root Causes**:
- Deaerator malfunction
- Insufficient oxygen scavenger
- Air leaks in feedwater system
- Feedwater heater bypass

---

#### WaterTreatmentSilicaCritical
**Threshold**: Silica > 150 ppm for 5 minutes
**Severity**: Critical
**Impact**: Silica deposition on turbine blades

**What It Means**: High silica will volatilize with steam and deposit on turbine blades, reducing efficiency.

**Immediate Actions**:
1. Increase blowdown rate
2. Verify blowdown control
3. Check makeup water silica
4. Review condensate quality

**Root Causes**:
- Inadequate blowdown
- High makeup water silica
- Contaminated condensate
- Blowdown control failure

---

#### WaterTreatmentPhosphateResidualLow
**Threshold**: Phosphate < 10 ppm for 5 minutes
**Severity**: Critical
**Impact**: Loss of scale protection

**What It Means**: Insufficient phosphate to precipitate hardness, allowing scale formation.

**Immediate Actions**:
1. Increase phosphate feed
2. Check phosphate pump operation
3. Verify chemical tank level
4. Review feedwater hardness

**Root Causes**:
- Insufficient phosphate dosing
- Phosphate pump failure
- Empty chemical tank
- Phosphate hideout (precipitation in deposits)

---

#### WaterTreatmentConductivityCritical
**Threshold**: Conductivity > 3000 µS/cm for 5 minutes
**Severity**: Critical
**Impact**: High TDS increases carryover and foaming risk

**What It Means**: Total dissolved solids too high, risking steam contamination.

**Immediate Actions**:
1. Increase blowdown rate
2. Verify blowdown control
3. Check conductivity probe calibration
4. Review makeup water quality

**Root Causes**:
- Inadequate blowdown
- Blowdown control failure
- Poor makeup water quality
- Condensate contamination

---

#### WaterTreatmentAgentHealthCritical
**Threshold**: Error rate > 5 errors/sec for 2 minutes
**Severity**: Critical
**Impact**: Loss of automated control

**What It Means**: WATERGUARD agent is experiencing serious operational issues.

**Immediate Actions**:
1. Review agent error logs
2. Check SCADA connectivity
3. Verify database connectivity
4. Switch to manual control if necessary
5. Contact GL-016 support

**Root Causes**:
- SCADA connection lost
- Database issues
- Network problems
- Software bugs
- Resource exhaustion

---

#### WaterGuardSCADAConnectionLost
**Threshold**: Connection status = 0 for 2 minutes
**Severity**: Critical
**Impact**: No real-time data or control

**What It Means**: Agent cannot communicate with SCADA system to read sensors or send commands.

**Immediate Actions**:
1. Switch to manual control
2. Verify SCADA system status
3. Check network connectivity
4. Review credentials and authentication
5. Check firewall rules
6. Test OPC/Modbus connection

**Root Causes**:
- SCADA system down
- Network outage
- Firewall blocking
- Credential expiration
- Protocol configuration error

---

### Warning Alerts

#### WaterTreatmentPHWarningHigh/Low
**Threshold**: pH 11.8-12.0 (high) or 10.5-11.0 (low) for 5 minutes
**Severity**: Warning
**Action**: Review dosing rate, monitor trend

---

#### WaterTreatmentDissolvedOxygenWarning
**Threshold**: DO 5-7 ppb for 10 minutes
**Severity**: Warning
**Action**: Review oxygen scavenger dosing

---

#### WaterTreatmentChemicalTankLow
**Threshold**: Tank level < 20% for 15 minutes
**Severity**: Warning
**Action**: Schedule chemical delivery

---

#### WaterTreatmentBlowdownInefficient
**Threshold**: Blowdown rate > 10% for 30 minutes
**Severity**: Warning
**Action**: Review blowdown control strategy

---

#### WaterTreatmentScaleRiskElevated
**Threshold**: Scale risk index 70-85 for 15 minutes
**Severity**: Warning
**Action**: Review hardness, alkalinity, phosphate

---

#### WaterGuardAgentLatencyWarning
**Threshold**: p99 latency 100-200ms for 5 minutes
**Severity**: Warning
**Action**: Monitor performance, investigate if continues

---

## Integration Instructions

### Prerequisites

- Prometheus 2.30+
- Grafana 8.0+
- Alertmanager 0.23+ (optional, for alert routing)
- Network access from Prometheus to WATERGUARD agent

### Step 1: Configure WATERGUARD Agent Metrics Export

Edit WATERGUARD agent configuration:

```yaml
# config/waterguard.yaml
metrics:
  enabled: true
  port: 9090
  path: /metrics
  update_interval: 15s
```

Restart the agent:

```bash
sudo systemctl restart waterguard-agent
```

Verify metrics endpoint:

```bash
curl http://localhost:9090/metrics
```

### Step 2: Configure Prometheus

Add WATERGUARD scrape configuration to `prometheus.yml`:

```yaml
global:
  scrape_interval: 30s
  evaluation_interval: 30s

scrape_configs:
  - job_name: 'waterguard'
    static_configs:
      - targets:
          - 'boiler-01.facility.com:9090'
          - 'boiler-02.facility.com:9090'
          - 'boiler-03.facility.com:9090'
        labels:
          facility: 'main-plant'
          environment: 'production'

    # For dynamic service discovery (Kubernetes)
    # kubernetes_sd_configs:
    #   - role: pod
    #     namespaces:
    #       names:
    #         - waterguard
    # relabel_configs:
    #   - source_labels: [__meta_kubernetes_pod_label_app]
    #     regex: waterguard-agent
    #     action: keep

  # Additional scrape configs for multi-facility deployments
  - job_name: 'waterguard-north-plant'
    static_configs:
      - targets:
          - 'boiler-n1.north.com:9090'
          - 'boiler-n2.north.com:9090'
        labels:
          facility: 'north-plant'
          environment: 'production'
```

Reload Prometheus configuration:

```bash
curl -X POST http://localhost:9090/-/reload
# OR
sudo systemctl reload prometheus
```

Verify scrape targets:

```bash
# Check targets are UP
curl http://localhost:9090/api/v1/targets
```

### Step 3: Load Alert Rules

Copy alert rule files to Prometheus:

```bash
sudo cp monitoring/alerts/*.yaml /etc/prometheus/rules/
```

Update `prometheus.yml` to include rules:

```yaml
rule_files:
  - "/etc/prometheus/rules/water_treatment_alerts.yaml"
  - "/etc/prometheus/rules/agent_health_alerts.yaml"
```

Validate rules syntax:

```bash
promtool check rules /etc/prometheus/rules/*.yaml
```

Reload Prometheus:

```bash
sudo systemctl reload prometheus
```

Verify rules loaded:

```bash
curl http://localhost:9090/api/v1/rules
```

### Step 4: Configure Alertmanager (Optional)

Create `alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

route:
  receiver: 'default'
  group_by: ['alertname', 'facility', 'boiler_id']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h

  routes:
    # Critical alerts to PagerDuty
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true

    # All alerts to Slack
    - receiver: 'slack'

receivers:
  - name: 'default'
    email_configs:
      - to: 'ops@company.com'
        from: 'alerts@company.com'
        smarthost: 'smtp.company.com:587'
        auth_username: 'alerts@company.com'
        auth_password: 'PASSWORD'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ .GroupLabels.alertname }} on {{ .GroupLabels.boiler_id }}'

  - name: 'slack'
    slack_configs:
      - channel: '#waterguard-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}'
        color: '{{ if eq .Status "firing" }}danger{{ else }}good{{ end }}'
```

Start Alertmanager:

```bash
sudo systemctl start alertmanager
sudo systemctl enable alertmanager
```

Configure Prometheus to send to Alertmanager:

```yaml
# prometheus.yml
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - 'localhost:9093'
```

### Step 5: Import Grafana Dashboards

#### Option A: Manual Import via UI

1. Login to Grafana
2. Navigate to Dashboards → Import
3. Click "Upload JSON file"
4. Select `grafana/waterguard_dashboard.json`
5. Select Prometheus datasource
6. Click Import
7. Repeat for other dashboards

#### Option B: Automated Import via API

```bash
#!/bin/bash
GRAFANA_URL="http://grafana.company.com"
GRAFANA_API_KEY="YOUR_API_KEY"

for dashboard in monitoring/grafana/*.json; do
  curl -X POST \
    -H "Authorization: Bearer $GRAFANA_API_KEY" \
    -H "Content-Type: application/json" \
    -d @"$dashboard" \
    "$GRAFANA_URL/api/dashboards/db"
done
```

#### Option C: Provisioning (Recommended for Automation)

Create provisioning file `/etc/grafana/provisioning/dashboards/waterguard.yaml`:

```yaml
apiVersion: 1

providers:
  - name: 'WATERGUARD'
    orgId: 1
    folder: 'GL-016 WATERGUARD'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards/waterguard
```

Copy dashboard files:

```bash
sudo mkdir -p /var/lib/grafana/dashboards/waterguard
sudo cp monitoring/grafana/*.json /var/lib/grafana/dashboards/waterguard/
sudo chown -R grafana:grafana /var/lib/grafana/dashboards/waterguard
```

Restart Grafana:

```bash
sudo systemctl restart grafana-server
```

### Step 6: Configure Datasources

Ensure Prometheus datasource is configured in Grafana:

```yaml
# /etc/grafana/provisioning/datasources/prometheus.yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    editable: false
```

### Step 7: Verification

Verify complete setup:

```bash
# Check Prometheus scraping WATERGUARD
curl http://localhost:9090/api/v1/query?query=water_treatment_ph_value

# Check alert rules loaded
curl http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[] | select(.name | contains("WaterTreatment"))'

# Check Grafana dashboards
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
  http://grafana.company.com/api/search?query=waterguard

# Trigger test alert (set low threshold temporarily)
# Watch alert in Prometheus: http://localhost:9090/alerts
```

## Metrics Reference

### Water Chemistry Metrics

| Metric Name | Type | Description | Unit | Labels |
|-------------|------|-------------|------|--------|
| `water_treatment_ph_value` | Gauge | Current pH value | pH units | boiler_id, facility |
| `water_treatment_ph_setpoint` | Gauge | Target pH setpoint | pH units | boiler_id, facility |
| `water_treatment_ph_upper_limit` | Gauge | Upper control limit | pH units | boiler_id, facility |
| `water_treatment_ph_lower_limit` | Gauge | Lower control limit | pH units | boiler_id, facility |
| `water_treatment_conductivity_microsiemens` | Gauge | Electrical conductivity | µS/cm | boiler_id, facility |
| `water_treatment_tds_ppm` | Gauge | Total dissolved solids | ppm | boiler_id, facility |
| `water_treatment_dissolved_oxygen_ppb` | Gauge | Dissolved oxygen content | ppb | boiler_id, facility |
| `water_treatment_silica_ppm` | Gauge | Silica concentration | ppm SiO₂ | boiler_id, facility |
| `water_treatment_phosphate_residual_ppm` | Gauge | Phosphate residual | ppm PO₄ | boiler_id, facility |
| `water_treatment_sulfite_residual_ppm` | Gauge | Sulfite residual | ppm SO₃ | boiler_id, facility |
| `water_treatment_total_hardness_ppm` | Gauge | Total hardness | ppm CaCO₃ | boiler_id, facility |
| `water_treatment_total_alkalinity_ppm` | Gauge | Total alkalinity | ppm CaCO₃ | boiler_id, facility |
| `water_treatment_chloride_ppm` | Gauge | Chloride concentration | ppm Cl | boiler_id, facility |
| `water_treatment_sulfate_ppm` | Gauge | Sulfate concentration | ppm SO₄ | boiler_id, facility |
| `water_treatment_iron_ppm` | Gauge | Iron content | ppm Fe | boiler_id, facility |
| `water_treatment_copper_ppm` | Gauge | Copper content | ppm Cu | boiler_id, facility |

### Blowdown Metrics

| Metric Name | Type | Description | Unit | Labels |
|-------------|------|-------------|------|--------|
| `water_treatment_blowdown_rate_percent` | Gauge | Current blowdown rate | % | boiler_id, facility |
| `water_treatment_cycles_of_concentration` | Gauge | Cycles of concentration | cycles | boiler_id, facility |
| `water_treatment_blowdown_heat_loss_mmbtu_per_hour` | Gauge | Heat loss from blowdown | MMBtu/hr | boiler_id, facility |

### Chemical Dosing Metrics

| Metric Name | Type | Description | Unit | Labels |
|-------------|------|-------------|------|--------|
| `water_treatment_chemical_dosing_rate_ppm` | Gauge | Chemical feed rate | ppm | boiler_id, facility, chemical |
| `water_treatment_chemical_tank_level_percent` | Gauge | Tank level percentage | % | boiler_id, facility, chemical |
| `water_treatment_chemical_tank_level_gallons` | Gauge | Tank level volume | gallons | boiler_id, facility, chemical |
| `water_treatment_chemical_consumption_gallons` | Counter | Cumulative consumption | gallons | boiler_id, facility, chemical |
| `water_treatment_chemical_cost_usd` | Counter | Cumulative chemical cost | USD | boiler_id, facility, chemical |
| `water_treatment_chemical_reorder_days` | Gauge | Days until reorder | days | boiler_id, facility, chemical |

### Risk & Prediction Metrics

| Metric Name | Type | Description | Unit | Labels |
|-------------|------|-------------|------|--------|
| `water_treatment_scale_risk_index` | Gauge | Scale formation risk | 0-100 | boiler_id, facility |
| `water_treatment_corrosion_risk_index` | Gauge | Corrosion risk | 0-100 | boiler_id, facility |
| `water_treatment_scale_prediction_probability` | Gauge | Scale prediction | 0-1 | boiler_id, facility, horizon |

### Compliance Metrics

| Metric Name | Type | Description | Unit | Labels |
|-------------|------|-------------|------|--------|
| `water_treatment_compliance_status` | Gauge | Compliance status | 0=non-compliant, 1=compliant | boiler_id, facility |
| `water_treatment_compliance_violations` | Gauge | Number of violations | count | boiler_id, facility, parameter |

### Cost Metrics

| Metric Name | Type | Description | Unit | Labels |
|-------------|------|-------------|------|--------|
| `water_treatment_water_saved_gallons` | Counter | Cumulative water saved | gallons | boiler_id, facility |
| `water_treatment_chemical_savings_usd` | Counter | Chemical cost savings | USD | boiler_id, facility |
| `water_treatment_energy_savings_mmbtu` | Counter | Energy savings | MMBtu | boiler_id, facility |
| `water_treatment_total_cost_savings_usd` | Counter | Total cost savings | USD | boiler_id, facility |
| `water_treatment_chemical_cost_per_1000lb_steam_usd` | Gauge | Cost per unit steam | USD | boiler_id, facility |
| `water_treatment_agent_operating_cost_usd` | Counter | Agent operating cost | USD | boiler_id, facility |

### Agent Health Metrics

| Metric Name | Type | Description | Unit | Labels |
|-------------|------|-------------|------|--------|
| `water_treatment_agent_latency_seconds` | Histogram | Request latency | seconds | boiler_id, facility |
| `water_treatment_agent_errors_total` | Counter | Total errors | count | boiler_id, facility, error_type |
| `water_treatment_agent_requests_total` | Counter | Total requests | count | boiler_id, facility |
| `water_treatment_agent_memory_usage_bytes` | Gauge | Memory usage | bytes | instance |
| `water_treatment_agent_cpu_usage_seconds_total` | Counter | CPU time | seconds | instance |
| `water_treatment_agent_uptime_seconds` | Gauge | Agent uptime | seconds | instance |
| `water_treatment_scada_connection_status` | Gauge | SCADA connection | 0=down, 1=up | boiler_id, scada_system |
| `water_treatment_erp_connection_status` | Gauge | ERP connection | 0=down, 1=up | boiler_id, erp_system |
| `water_treatment_database_connection_status` | Gauge | Database connection | 0=down, 1=up | database_name |

## Troubleshooting Guide

### Issue: No Data in Grafana Dashboards

**Symptoms**: Dashboards show "No Data" or empty panels

**Diagnosis**:
```bash
# 1. Check Prometheus can reach WATERGUARD agent
curl http://BOILER-HOST:9090/metrics

# 2. Check Prometheus scrape targets
curl http://prometheus:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job == "waterguard")'

# 3. Query metrics directly in Prometheus
curl http://prometheus:9090/api/v1/query?query=water_treatment_ph_value

# 4. Check Grafana datasource
curl -H "Authorization: Bearer $API_KEY" http://grafana:3000/api/datasources
```

**Solutions**:
- Verify WATERGUARD agent is running: `systemctl status waterguard-agent`
- Check firewall allows port 9090: `sudo firewall-cmd --list-ports`
- Verify Prometheus configuration: `promtool check config prometheus.yml`
- Test Grafana-Prometheus connection in Grafana UI

---

### Issue: Alerts Not Firing

**Symptoms**: Expected alerts not appearing in Alertmanager or Grafana

**Diagnosis**:
```bash
# 1. Check alert rules loaded
curl http://prometheus:9090/api/v1/rules

# 2. Check alert evaluation
curl http://prometheus:9090/api/v1/alerts

# 3. Validate rule syntax
promtool check rules /etc/prometheus/rules/*.yaml

# 4. Check Alertmanager config
amtool config show
```

**Solutions**:
- Verify alert rule syntax is correct
- Check alert `for` duration hasn't prevented firing
- Ensure Alertmanager is running and configured in Prometheus
- Review alert inhibition and silencing rules

---

### Issue: High Agent Latency

**Symptoms**: `WaterGuardAgentLatencyWarning` or `WaterGuardAgentLatencyCritical` alerts

**Diagnosis**:
```bash
# 1. Check agent logs
sudo journalctl -u waterguard-agent -n 100 --no-pager

# 2. Check system resources
top -p $(pgrep waterguard-agent)
free -h
df -h

# 3. Check network latency
ping SCADA-HOST
ping DATABASE-HOST

# 4. Profile agent
curl http://localhost:9090/debug/pprof/profile?seconds=30 > profile.out
```

**Solutions**:
- Restart agent if memory leak suspected: `systemctl restart waterguard-agent`
- Optimize database queries (check slow query log)
- Reduce SCADA polling frequency if network constrained
- Scale up compute resources if consistently high

---

### Issue: SCADA Connection Lost

**Symptoms**: `WaterGuardSCADAConnectionLost` alert, no real-time data

**Diagnosis**:
```bash
# 1. Test network connectivity
ping SCADA-HOST
telnet SCADA-HOST OPC-PORT

# 2. Check SCADA system status
# (SCADA-specific commands)

# 3. Review agent SCADA logs
sudo journalctl -u waterguard-agent -g "SCADA" -n 50

# 4. Check credentials
# Review agent config for SCADA credentials
```

**Solutions**:
- Verify SCADA system is operational
- Check firewall rules allow OPC/Modbus traffic
- Validate SCADA credentials haven't expired
- Review SCADA configuration in agent config file
- Test OPC/Modbus connection manually with client tool

---

### Issue: Metric Gaps or Missing Data

**Symptoms**: Gaps in time-series data, intermittent "No Data"

**Diagnosis**:
```bash
# 1. Check Prometheus scrape success rate
rate(prometheus_target_scrapes_exceeded_sample_limit_total[5m])
rate(prometheus_target_scrapes_sample_duplicate_timestamp_total[5m])

# 2. Check agent uptime
water_treatment_agent_uptime_seconds

# 3. Review Prometheus logs
sudo journalctl -u prometheus -n 100 --no-pager

# 4. Check disk space
df -h /var/lib/prometheus
```

**Solutions**:
- Increase Prometheus storage if full
- Increase scrape timeout if agent slow
- Adjust sample limit if exceeding default
- Check for agent restarts causing gaps

---

### Issue: Dashboard Variables Not Populating

**Symptoms**: Boiler ID or facility dropdowns empty

**Diagnosis**:
```bash
# 1. Query label values
curl 'http://prometheus:9090/api/v1/label/boiler_id/values'
curl 'http://prometheus:9090/api/v1/label/facility/values'

# 2. Check metrics have required labels
curl 'http://prometheus:9090/api/v1/query?query=water_treatment_ph_value' | jq '.data.result[].metric'
```

**Solutions**:
- Ensure metrics include `boiler_id` and `facility` labels
- Verify Grafana datasource configured correctly
- Check dashboard variable regex filters
- Refresh dashboard variables manually in Grafana

---

### Issue: False Positive Alerts

**Symptoms**: Alerts firing inappropriately

**Diagnosis**:
- Review alert `for` duration (may be too short)
- Check threshold values in alert rules
- Verify sensor accuracy (pH probe calibration, etc.)
- Review recent changes to alert rules

**Solutions**:
- Adjust alert thresholds based on operational experience
- Increase `for` duration to filter transients
- Add additional conditions to alert expression
- Calibrate sensors if suspect inaccuracy

---

### Issue: Chemical Tank Level Incorrect

**Symptoms**: Tank gauge shows wrong level, false low-level alerts

**Diagnosis**:
```bash
# 1. Query current tank levels
curl 'http://prometheus:9090/api/v1/query?query=water_treatment_chemical_tank_level_percent'

# 2. Compare to physical inspection
# (Visual tank level check)

# 3. Review level sensor logs
sudo journalctl -u waterguard-agent -g "tank_level" -n 50
```

**Solutions**:
- Calibrate level sensor
- Check sensor wiring and connections
- Verify tank capacity configured correctly in agent
- Review level calculation logic if custom

## Best Practices

### Dashboard Usage

1. **Real-Time Monitoring**
   - Use Overview Dashboard for shift operations
   - Set appropriate refresh rate (30s-1m)
   - Enable browser notifications for Grafana alerts
   - Create facility-specific dashboard copies with preset filters

2. **Trend Analysis**
   - Use Water Chemistry Dashboard for long-term trending
   - Set time range to 7-30 days for pattern recognition
   - Export data to CSV for offline analysis
   - Create annotations for operational changes

3. **Financial Reporting**
   - Use Chemical Optimization Dashboard monthly
   - Compare month-over-month with time shift
   - Export ROI metrics for management reports
   - Track payback period progress

### Alert Management

1. **Alert Tuning**
   - Review and adjust thresholds quarterly
   - Document threshold changes in Git
   - Test alerts after tuning with known conditions
   - Balance sensitivity vs alert fatigue

2. **Alert Response**
   - Document standard operating procedures for each alert
   - Train operators on alert interpretation
   - Use runbook links for troubleshooting guidance
   - Review alert response times in post-mortems

3. **Alert Routing**
   - Route critical alerts to on-call rotation
   - Send warnings to operator dashboard only
   - Group alerts by boiler_id to reduce noise
   - Use Alertmanager silences during maintenance

### Metric Collection

1. **Label Hygiene**
   - Keep cardinality low (avoid high-cardinality labels)
   - Use consistent label names across all metrics
   - Document label meanings and valid values
   - Avoid labels that change frequently

2. **Metric Naming**
   - Follow Prometheus naming conventions
   - Use base unit (seconds, bytes, not ms, KB)
   - Include unit in metric name (_seconds, _bytes, _total)
   - Namespace with `water_treatment_` prefix

3. **Retention**
   - Set Prometheus retention to 90 days minimum
   - Use remote write for long-term storage (1+ year)
   - Consider downsampling for extended retention
   - Backup Prometheus data regularly

### Performance Optimization

1. **Query Optimization**
   - Use recording rules for expensive queries
   - Limit time range for high-cardinality queries
   - Use `rate()` over `irate()` for smoother graphs
   - Aggregate before arithmetic operations

2. **Dashboard Performance**
   - Limit panels per dashboard (< 30)
   - Use appropriate refresh intervals
   - Avoid `*` in queries (be specific)
   - Use dashboard variables for filtering

3. **Scalability**
   - Use Prometheus federation for large deployments
   - Implement Thanos/Cortex for multi-cluster
   - Shard by facility or region if needed
   - Monitor Prometheus resource usage

### Security

1. **Access Control**
   - Use Grafana authentication (LDAP/OAuth)
   - Implement role-based access control
   - Restrict Prometheus API access
   - Use TLS for all connections

2. **Secrets Management**
   - Store credentials in environment variables
   - Use secrets management (Vault, AWS Secrets Manager)
   - Rotate API keys regularly
   - Never commit credentials to Git

3. **Network Security**
   - Restrict metrics endpoint to Prometheus IP
   - Use firewall rules for service isolation
   - Implement VPN for remote access
   - Regular security audits

### Operational Excellence

1. **Change Management**
   - Version control all configs (Git)
   - Test changes in non-production first
   - Document changes in commit messages
   - Use CI/CD for configuration deployment

2. **Documentation**
   - Keep runbooks up-to-date
   - Document alert response procedures
   - Create troubleshooting flowcharts
   - Record lessons learned from incidents

3. **Continuous Improvement**
   - Review dashboards quarterly
   - Conduct alert retrospectives
   - Gather operator feedback
   - Benchmark against industry standards

4. **Disaster Recovery**
   - Backup Prometheus data regularly
   - Document recovery procedures
   - Test recovery process annually
   - Maintain redundant monitoring where critical

## Support

### Getting Help

- **Documentation**: https://docs.greenlang.io/gl-016
- **Runbooks**: https://runbooks.greenlang.io/gl-016
- **Support Email**: support@greenlang.io
- **Emergency On-Call**: PagerDuty escalation

### Reporting Issues

When reporting monitoring issues, include:

1. Affected boiler ID(s) and facility
2. Time range of issue
3. Relevant metric queries
4. Screenshots of dashboards
5. Alert history
6. Agent logs excerpt

### Contributing

To contribute improvements to monitoring:

1. Fork repository
2. Create feature branch
3. Make changes with descriptive commits
4. Test thoroughly in non-production
5. Submit pull request with documentation

---

**Version**: 1.0.0
**Last Updated**: 2025-12-02
**Maintained By**: GreenLang Platform Team
**License**: Proprietary - GreenLang Inc.
