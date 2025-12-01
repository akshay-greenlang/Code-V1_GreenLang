# GL-013 PREDICTMAINT - Grafana Dashboard Specifications

## Overview

This document specifies the Grafana dashboard design for GL-013 PREDICTMAINT, providing comprehensive visualization of predictive maintenance metrics, equipment health, failure predictions, and operational performance.

**Dashboard ID:** `gl013-predictive-maintenance`
**Version:** 1.0.0
**Author:** GL-MonitoringEngineer
**Last Updated:** 2024-12-01

---

## Dashboard Architecture

### Design Principles

1. **Progressive Disclosure**: Overview panels at top, details below
2. **Actionable Insights**: Clear thresholds and color coding
3. **Real-time Monitoring**: Auto-refresh with 30-second intervals
4. **Mobile Responsive**: Optimized for operations center and mobile devices
5. **Standards Compliance**: ISO 10816 zones clearly indicated

### Dashboard Hierarchy

```
GL-013 Predictive Maintenance Dashboard
|
+-- Row 1: Executive Summary (4 panels)
|   +-- Total Equipment Monitored
|   +-- Critical Equipment Count
|   +-- Average Health Index
|   +-- Cost Savings (YTD)
|
+-- Row 2: Equipment Health Overview (3 panels)
|   +-- Health Index Distribution
|   +-- Equipment Health Heatmap
|   +-- Health Trend (7-day)
|
+-- Row 3: Failure Prediction (3 panels)
|   +-- RUL Distribution
|   +-- Failure Probability Timeline
|   +-- Upcoming Failures (Table)
|
+-- Row 4: Vibration Analysis (4 panels)
|   +-- Vibration Zone Distribution
|   +-- Vibration Velocity Trend
|   +-- Bearing Fault Indicators
|   +-- Spectrum Analysis
|
+-- Row 5: Temperature Monitoring (3 panels)
|   +-- Temperature Overview
|   +-- Thermal Life Consumption
|   +-- Temperature Alerts
|
+-- Row 6: Anomaly Detection (3 panels)
|   +-- Active Anomalies
|   +-- Anomaly Score Trend
|   +-- Anomaly Distribution
|
+-- Row 7: Maintenance Schedule (3 panels)
|   +-- Upcoming Maintenance
|   +-- Maintenance Backlog
|   +-- Cost Savings Analysis
|
+-- Row 8: System Performance (4 panels)
|   +-- Operation Latency
|   +-- Cache Performance
|   +-- Connector Status
|   +-- Error Rates
```

---

## Panel Specifications

### Row 1: Executive Summary

#### Panel 1.1: Total Equipment Monitored
- **Type:** Stat
- **Query:** `count(gl013_equipment_health_index)`
- **Color Mode:** Value
- **Graph Mode:** None
- **Thresholds:**
  - Base: Blue (#1890FF)
- **Unit:** none
- **Decimals:** 0
- **Description:** Total number of equipment assets being monitored

#### Panel 1.2: Critical Equipment Count
- **Type:** Stat
- **Query:** `count(gl013_equipment_health_index < 25)`
- **Color Mode:** Background
- **Graph Mode:** Area
- **Thresholds:**
  - 0: Green (#52C41A)
  - 1: Yellow (#FAAD14)
  - 3: Orange (#FA8C16)
  - 5: Red (#F5222D)
- **Unit:** none
- **Description:** Equipment with health index below critical threshold (25%)

#### Panel 1.3: Average Health Index
- **Type:** Gauge
- **Query:** `avg(gl013_equipment_health_index)`
- **Min:** 0
- **Max:** 100
- **Thresholds:**
  - 0-25: Red (#F5222D) - Critical
  - 25-50: Orange (#FA8C16) - Poor
  - 50-70: Yellow (#FAAD14) - Fair
  - 70-90: Light Green (#73D13D) - Good
  - 90-100: Green (#52C41A) - Excellent
- **Unit:** percent
- **Decimals:** 1
- **Description:** Fleet-wide average equipment health

#### Panel 1.4: Cost Savings (YTD)
- **Type:** Stat
- **Query:** `sum(gl013_maintenance_cost_savings_usd_total)`
- **Color Mode:** Value
- **Thresholds:**
  - Base: Green (#52C41A)
- **Unit:** currencyUSD
- **Decimals:** 0
- **Prefix:** $
- **Description:** Year-to-date cost savings from predictive maintenance

---

### Row 2: Equipment Health Overview

#### Panel 2.1: Health Index Distribution
- **Type:** Bar Gauge
- **Query:**
  ```promql
  count(gl013_equipment_health_index >= 90) # Excellent
  count(gl013_equipment_health_index >= 70 and gl013_equipment_health_index < 90) # Good
  count(gl013_equipment_health_index >= 50 and gl013_equipment_health_index < 70) # Fair
  count(gl013_equipment_health_index >= 25 and gl013_equipment_health_index < 50) # Poor
  count(gl013_equipment_health_index < 25) # Critical
  ```
- **Orientation:** Horizontal
- **Display Mode:** Gradient
- **Colors:**
  - Excellent: #52C41A
  - Good: #73D13D
  - Fair: #FAAD14
  - Poor: #FA8C16
  - Critical: #F5222D
- **Description:** Distribution of equipment across health categories

#### Panel 2.2: Equipment Health Heatmap
- **Type:** Heatmap
- **Query:** `gl013_equipment_health_index`
- **Y-Axis:** equipment_id
- **Color Scheme:** RdYlGn (inverted for lower = red)
- **Cell Display:** Show values
- **Legend:** Show
- **Description:** Visual overview of all equipment health status

#### Panel 2.3: Health Trend (7-day)
- **Type:** Time Series
- **Query:** `avg(gl013_equipment_health_index)`
- **Time Range:** Last 7 days
- **Line Width:** 2
- **Fill Opacity:** 20
- **Thresholds:**
  - 25: Red line (Critical)
  - 50: Yellow line (Warning)
  - 70: Green line (Good)
- **Description:** Average health index trend over past week

---

### Row 3: Failure Prediction

#### Panel 3.1: RUL Distribution
- **Type:** Histogram
- **Query:** `gl013_equipment_rul_days`
- **Bucket Size:** 7 (days)
- **X-Axis:** Days remaining
- **Color Zones:**
  - 0-7: Red (#F5222D)
  - 7-30: Orange (#FA8C16)
  - 30-90: Yellow (#FAAD14)
  - 90+: Green (#52C41A)
- **Description:** Distribution of equipment by remaining useful life

#### Panel 3.2: Failure Probability Timeline
- **Type:** Time Series
- **Queries:**
  ```promql
  avg(gl013_failure_probability_30d) by (equipment_type)
  avg(gl013_failure_probability_90d) by (equipment_type)
  ```
- **Time Range:** Last 30 days
- **Display:** Lines with points
- **Threshold Lines:**
  - 0.5: Warning
  - 0.8: Critical
- **Description:** Failure probability trends by equipment type

#### Panel 3.3: Upcoming Failures (Table)
- **Type:** Table
- **Query:**
  ```promql
  topk(10, gl013_failure_probability_30d)
  ```
- **Columns:**
  - Equipment ID
  - Equipment Type
  - RUL (days)
  - Failure Probability (30d)
  - Health Index
  - Status (color-coded)
- **Sort:** By failure probability (descending)
- **Row Colors:** Based on severity
- **Description:** Top 10 equipment at highest failure risk

---

### Row 4: Vibration Analysis

#### Panel 4.1: Vibration Zone Distribution
- **Type:** Pie Chart
- **Queries:**
  ```promql
  count(gl013_vibration_zone == 1) # Zone A
  count(gl013_vibration_zone == 2) # Zone B
  count(gl013_vibration_zone == 3) # Zone C
  count(gl013_vibration_zone == 4) # Zone D
  ```
- **Colors:**
  - Zone A (Good): #52C41A
  - Zone B (Acceptable): #73D13D
  - Zone C (Alert): #FAAD14
  - Zone D (Danger): #F5222D
- **Legend Position:** Right
- **Description:** Distribution of equipment across ISO 10816 zones

#### Panel 4.2: Vibration Velocity Trend
- **Type:** Time Series
- **Query:** `gl013_vibration_velocity_mm_s`
- **Time Range:** Last 24 hours
- **Y-Axis:** mm/s RMS
- **Threshold Lines:**
  - Zone A/B boundary: Green dashed
  - Zone B/C boundary: Yellow dashed
  - Zone C/D boundary: Red dashed
- **Multi-series:** By equipment_id
- **Description:** Real-time vibration velocity trends

#### Panel 4.3: Bearing Fault Indicators
- **Type:** Bar Chart
- **Queries:**
  ```promql
  gl013_bearing_fault_frequency_energy{fault_type="BPFO"}
  gl013_bearing_fault_frequency_energy{fault_type="BPFI"}
  gl013_bearing_fault_frequency_energy{fault_type="BSF"}
  gl013_bearing_fault_frequency_energy{fault_type="FTF"}
  ```
- **Orientation:** Horizontal
- **Group By:** equipment_id
- **Threshold:** 0.5 (fault indication threshold)
- **Description:** Energy levels at bearing fault frequencies

#### Panel 4.4: Spectrum Analysis
- **Type:** Time Series (or Custom Visualization)
- **Query:** `gl013_vibration_spectrum_dominant_freq_hz`
- **Y-Axis:** Frequency (Hz)
- **Annotations:** Bearing fault frequencies marked
- **Description:** Dominant frequencies in vibration spectrum

---

### Row 5: Temperature Monitoring

#### Panel 5.1: Temperature Overview
- **Type:** Table
- **Query:** `gl013_temperature_celsius`
- **Columns:**
  - Equipment ID
  - Location
  - Temperature (C)
  - Delta from Ambient
  - Status
- **Cell Colors:** Based on temperature thresholds
- **Sparklines:** 1-hour trend
- **Description:** Current temperature readings for all sensors

#### Panel 5.2: Thermal Life Consumption
- **Type:** Bar Gauge
- **Query:** `gl013_thermal_life_consumed_percent`
- **Orientation:** Horizontal
- **Min:** 0
- **Max:** 100
- **Thresholds:**
  - 0-60: Green
  - 60-80: Yellow
  - 80-100: Red
- **Show Values:** Percentage
- **Description:** Thermal life consumption by equipment

#### Panel 5.3: Temperature Alerts
- **Type:** Stat Panel (repeating)
- **Query:** `count(gl013_temperature_celsius > 100)`
- **Title:** High Temp Alerts
- **Color Mode:** Background
- **Thresholds:**
  - 0: Green
  - 1: Yellow
  - 3: Red
- **Description:** Count of temperature threshold exceedances

---

### Row 6: Anomaly Detection

#### Panel 6.1: Active Anomalies
- **Type:** Stat (repeating by severity)
- **Queries:**
  ```promql
  sum(gl013_anomalies_active{severity="low"})
  sum(gl013_anomalies_active{severity="medium"})
  sum(gl013_anomalies_active{severity="high"})
  sum(gl013_anomalies_active{severity="critical"})
  ```
- **Layout:** 4 stats in row
- **Colors:**
  - Low: Blue
  - Medium: Yellow
  - High: Orange
  - Critical: Red
- **Description:** Count of active anomalies by severity

#### Panel 6.2: Anomaly Score Trend
- **Type:** Time Series
- **Query:** `gl013_anomaly_score`
- **Time Range:** Last 24 hours
- **Y-Axis:** Score (0-1)
- **Threshold Lines:**
  - 0.7: Yellow (Warning)
  - 0.9: Red (Critical)
- **Multi-series:** By equipment_id
- **Description:** Anomaly score evolution over time

#### Panel 6.3: Anomaly Distribution
- **Type:** Pie Chart
- **Query:** `sum by (anomaly_type) (increase(gl013_anomalies_detected_total[24h]))`
- **Legend:** Bottom
- **Show Values:** Count and percentage
- **Description:** Distribution of anomaly types in last 24 hours

---

### Row 7: Maintenance Schedule

#### Panel 7.1: Upcoming Maintenance
- **Type:** Table
- **Query:** `gl013_maintenance_lead_time_days`
- **Columns:**
  - Equipment ID
  - Maintenance Type
  - Days Until Due
  - Priority
  - Status
- **Sort:** By lead time (ascending)
- **Row Colors:** By urgency
- **Filter:** Lead time < 30 days
- **Description:** Maintenance tasks due within 30 days

#### Panel 7.2: Maintenance Backlog
- **Type:** Bar Chart
- **Queries:**
  ```promql
  sum(gl013_maintenance_tasks_active{urgency="routine"})
  sum(gl013_maintenance_tasks_active{urgency="planned"})
  sum(gl013_maintenance_tasks_active{urgency="urgent"})
  sum(gl013_maintenance_tasks_overdue)
  ```
- **Orientation:** Vertical
- **Stack:** True
- **Colors:**
  - Routine: Blue
  - Planned: Green
  - Urgent: Orange
  - Overdue: Red
- **Description:** Current maintenance task distribution

#### Panel 7.3: Cost Savings Analysis
- **Type:** Time Series
- **Queries:**
  ```promql
  sum(increase(gl013_maintenance_cost_savings_usd_total[1d])) by (savings_type)
  sum(increase(gl013_maintenance_downtime_prevented_hours_total[1d])) * 5000
  ```
- **Time Range:** Last 30 days
- **Y-Axis:** USD
- **Stacked:** True
- **Description:** Daily cost savings from predictive maintenance

---

### Row 8: System Performance

#### Panel 8.1: Operation Latency
- **Type:** Time Series
- **Queries:**
  ```promql
  histogram_quantile(0.50, rate(gl013_operation_latency_seconds_bucket[5m]))
  histogram_quantile(0.95, rate(gl013_operation_latency_seconds_bucket[5m]))
  histogram_quantile(0.99, rate(gl013_operation_latency_seconds_bucket[5m]))
  ```
- **Legend:** P50, P95, P99
- **Y-Axis:** Seconds
- **Threshold:** 10s (SLO)
- **Description:** Operation latency percentiles

#### Panel 8.2: Cache Performance
- **Type:** Gauge
- **Query:** `gl013_cache_hit_rate`
- **Min:** 0
- **Max:** 100
- **Thresholds:**
  - 0-50: Red
  - 50-70: Yellow
  - 70-100: Green
- **Unit:** percent
- **Description:** Cache hit rate effectiveness

#### Panel 8.3: Connector Status
- **Type:** State Timeline
- **Query:** `gl013_connector_status`
- **States:**
  - 0 (Disconnected): Red
  - 1 (Connected): Green
  - 2 (Degraded): Yellow
- **Group By:** connector_type
- **Description:** Integration connector health over time

#### Panel 8.4: Error Rates
- **Type:** Time Series
- **Query:**
  ```promql
  sum(rate(gl013_operations_total{status="failure"}[5m])) /
  sum(rate(gl013_operations_total[5m])) * 100
  ```
- **Y-Axis:** Percentage
- **Threshold:** 1% (SLO)
- **Alert:** When > 5%
- **Description:** Operation error rate percentage

---

## Variables (Dashboard Filters)

### Variable: equipment_type
- **Type:** Query
- **Query:** `label_values(gl013_equipment_health_index, equipment_type)`
- **Multi-select:** Yes
- **Include All:** Yes
- **Description:** Filter by equipment type

### Variable: equipment_id
- **Type:** Query
- **Query:** `label_values(gl013_equipment_health_index{equipment_type=~"$equipment_type"}, equipment_id)`
- **Multi-select:** Yes
- **Include All:** Yes
- **Dependent:** On equipment_type
- **Description:** Filter by specific equipment

### Variable: time_range
- **Type:** Interval
- **Values:** 15m, 1h, 6h, 12h, 24h, 7d, 30d
- **Default:** 24h
- **Description:** Time range for trend panels

### Variable: refresh_interval
- **Type:** Interval
- **Values:** 10s, 30s, 1m, 5m, Off
- **Default:** 30s
- **Description:** Dashboard auto-refresh interval

---

## Annotations

### Annotation: Maintenance Events
- **Query:** `changes(gl013_maintenance_tasks_scheduled_total[5m]) > 0`
- **Color:** Blue
- **Icon:** Wrench
- **Description:** Marks when maintenance tasks are scheduled

### Annotation: Alert Firing
- **Data Source:** Alertmanager
- **Filter:** alertname=~"gl013_.*"
- **Color:** Red
- **Description:** Shows when alerts fire

### Annotation: Anomaly Detection
- **Query:** `gl013_anomaly_score > 0.7`
- **Color:** Orange
- **Description:** Marks significant anomaly detections

---

## Alert Integration

### Linked Alerts
The dashboard includes links to relevant alerts:
- Equipment Health alerts
- RUL critical alerts
- Vibration zone violations
- Temperature exceedances
- Anomaly detections

### Alert Panel
- Shows active alerts for filtered equipment
- Links to alert rules and runbooks
- Silencing options for maintenance windows

---

## Access and Permissions

### Folder Structure
```
Dashboards/
  GreenLang/
    GL-013 PREDICTMAINT/
      Main Dashboard
      Vibration Deep Dive
      Thermal Analysis
      Equipment Detail
```

### Permissions
- **Viewers:** Read-only access
- **Editors:** Can modify thresholds and layouts
- **Admins:** Full access including data source changes

---

## Deployment Notes

### Data Source Requirements
- Prometheus (primary metrics)
- Alertmanager (alert status)
- Loki (optional, for log correlation)

### Performance Optimization
- Use `$__rate_interval` for rate queries
- Limit cardinality with label filters
- Use recording rules for expensive queries

### Backup and Versioning
- Dashboard JSON stored in Git
- Changes tracked through provisioning
- Rollback available through version history

---

## Revision History

| Version | Date       | Author              | Changes                    |
|---------|------------|---------------------|----------------------------|
| 1.0.0   | 2024-12-01 | GL-MonitoringEngineer | Initial specification      |
