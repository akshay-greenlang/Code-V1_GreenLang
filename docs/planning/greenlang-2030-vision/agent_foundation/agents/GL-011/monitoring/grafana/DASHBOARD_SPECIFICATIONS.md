# GL-011 FUELCRAFT Grafana Dashboard Specifications

## Overview

This document provides comprehensive specifications for the three Grafana dashboards designed for GL-011 FUELCRAFT, the fuel procurement and optimization intelligence agent. These dashboards provide operational, business, and executive-level insights into fuel procurement, cost optimization, and environmental performance.

---

## Dashboard Architecture

### Dashboard Hierarchy

```
GL-011 FUELCRAFT Monitoring
├── Operations Dashboard (Technical Focus)
│   ├── System Health & Performance
│   ├── API & Integration Monitoring
│   └── Error Tracking & Diagnostics
│
├── Business KPI Dashboard (Operational Focus)
│   ├── Cost & Savings Metrics
│   ├── Fuel Management & Efficiency
│   └── Supplier & Contract Performance
│
└── Executive Dashboard (Strategic Focus)
    ├── High-Level KPIs
    ├── Compliance Status
    └── Achievement Tracking
```

---

## 1. Operations Dashboard

**File:** `operations_dashboard.json`
**UID:** `gl011-operations`
**Refresh Rate:** 30 seconds
**Default Time Range:** Last 6 hours

### Purpose
Real-time operational monitoring for DevOps teams, SREs, and technical operators to track system health, performance, and integration status.

### Panel Specifications (20 Panels)

#### Row 1: System Health Overview (y=0)

**Panel 1.1: Agent Health Status**
- **Type:** Stat (gauge)
- **Position:** (0,0) - 4x4
- **Metric:** `gl011_agent_health_status{agent_id="$agent_id"}`
- **Thresholds:**
  - 0 = RED (UNHEALTHY)
  - 1 = GREEN (HEALTHY)
- **Display:** Background color mode

**Panel 1.2: Request Rate**
- **Type:** Time series
- **Position:** (4,0) - 10x8
- **Metric:** `rate(gl011_requests_total{agent_id="$agent_id"}[5m])`
- **Legend:** `{{endpoint}} - {{status}}`
- **Unit:** requests per second (reqps)
- **Aggregations:** mean, lastNotNull, max

**Panel 1.3: Request Latency (P50/P95/P99)**
- **Type:** Time series
- **Position:** (14,0) - 10x8
- **Metrics:**
  - P50: `histogram_quantile(0.50, rate(gl011_request_duration_seconds_bucket[5m]))`
  - P95: `histogram_quantile(0.95, rate(gl011_request_duration_seconds_bucket[5m]))`
  - P99: `histogram_quantile(0.99, rate(gl011_request_duration_seconds_bucket[5m]))`
- **Unit:** seconds (s)
- **Legend:** P50, P95, P99

**Panel 1.4: Active Alerts**
- **Type:** Stat
- **Position:** (0,4) - 4x4
- **Metric:** `sum(gl011_active_alerts{agent_id="$agent_id"})`
- **Thresholds:**
  - 0 = GREEN
  - 1-4 = YELLOW
  - 5+ = RED

#### Row 2: Performance Metrics (y=8)

**Panel 2.1: Error Rate**
- **Type:** Time series
- **Position:** (0,8) - 12x8
- **Metric:** `rate(gl011_errors_total{agent_id="$agent_id"}[5m])`
- **Legend:** `{{error_type}} - {{severity}}`
- **Unit:** operations per second (ops)

**Panel 2.2: CPU Usage**
- **Type:** Gauge
- **Position:** (12,8) - 6x4
- **Metric:** `gl011_cpu_usage_percent{agent_id="$agent_id"}`
- **Unit:** percent (%)
- **Thresholds:**
  - 0-50% = GREEN
  - 50-80% = YELLOW
  - 80-100% = RED

**Panel 2.3: Memory Usage**
- **Type:** Gauge
- **Position:** (18,8) - 6x4
- **Metric:** `gl011_memory_usage_bytes{agent_id="$agent_id",memory_type="rss"}`
- **Unit:** bytes
- **Thresholds:**
  - 0-2GB = GREEN
  - 2-4GB = YELLOW
  - 4GB+ = RED

**Panel 2.4: Thread Count**
- **Type:** Time series
- **Position:** (12,12) - 6x6
- **Metric:** `gl011_thread_count{agent_id="$agent_id"}`
- **Legend:** Threads

**Panel 2.5: Cache Hit Rate**
- **Type:** Gauge
- **Position:** (18,12) - 6x6
- **Metric:** `gl011_cache_hit_rate{agent_id="$agent_id"}`
- **Unit:** percentunit (0-1 scale)
- **Thresholds:**
  - <0.7 = RED
  - 0.7-0.85 = YELLOW
  - >0.85 = GREEN
- **Target:** >85%

#### Row 3: Integration & API Performance (y=16)

**Panel 3.1: API Call Latency Heatmap**
- **Type:** Heatmap
- **Position:** (0,16) - 12x8
- **Metric:** `rate(gl011_request_duration_seconds_bucket{agent_id="$agent_id"}[5m])`
- **Color Scheme:** Spectral
- **Format:** heatmap

**Panel 3.2: Integration Health Table**
- **Type:** Table
- **Position:** (12,18) - 12x8
- **Metric:** `gl011_integration_health{agent_id="$agent_id"}`
- **Columns:**
  - Integration Type (ERP, Market Data, Storage)
  - Status (UP/DOWN with color coding)
- **Value Mapping:**
  - 0 = RED (DOWN)
  - 1 = GREEN (UP)

#### Row 4: Queue & Concurrency (y=24)

**Panel 4.1: Queue Depth**
- **Type:** Time series
- **Position:** (0,24) - 8x8
- **Metric:** `gl011_queue_depth{agent_id="$agent_id"}`
- **Legend:** `{{queue_name}}`
- **Thresholds:**
  - <100 = GREEN
  - 100-500 = YELLOW
  - >500 = RED

**Panel 4.2: Concurrent Requests**
- **Type:** Time series
- **Position:** (8,24) - 8x8
- **Metric:** `gl011_concurrent_requests{agent_id="$agent_id"}`
- **Legend:** Concurrent Requests

**Panel 4.3: Optimization Execution Time**
- **Type:** Heatmap/Histogram
- **Position:** (16,24) - 8x8
- **Metric:** `rate(gl011_optimization_execution_seconds_bucket[5m])`
- **Legend:** `{{optimization_type}}`
- **Color Scheme:** RdYlGn

#### Row 5: Calculator & Provenance (y=32)

**Panel 5.1: Calculator Performance**
- **Type:** Bar Gauge
- **Position:** (0,32) - 12x8
- **Metric:** `avg by (calculator_type) (gl011_calculator_performance_seconds)`
- **Legend:** `{{calculator_type}}`
- **Unit:** seconds (s)
- **Orientation:** Horizontal

**Panel 5.2: Provenance Tracking**
- **Type:** Stat
- **Position:** (12,32) - 12x8
- **Metric:** `sum(increase(gl011_provenance_records_created{agent_id="$agent_id"}[24h]))`
- **Display:** Total records tracked in last 24 hours

#### Row 6: Logs & Diagnostics (y=40)

**Panel 6.1: Alert History**
- **Type:** Table
- **Position:** (0,40) - 12x8
- **Metric:** `gl011_alerts_history{agent_id="$agent_id"}`
- **Columns:**
  - Alert Name
  - Severity (color-coded: critical=RED, warning=YELLOW, info=BLUE)
  - Timestamp
- **Sort:** Descending by timestamp

**Panel 6.2: Log Volume by Level**
- **Type:** Pie Chart
- **Position:** (12,40) - 6x8
- **Metric:** `sum by (log_level) (rate(gl011_log_entries_total[5m]))`
- **Legend:** `{{log_level}}`
- **Display:** Percentage breakdown

**Panel 6.3: Network Traffic In/Out**
- **Type:** Time series
- **Position:** (18,40) - 6x8
- **Metrics:**
  - Inbound: `rate(gl011_network_bytes_in[5m])`
  - Outbound: `rate(gl011_network_bytes_out[5m])`
- **Unit:** bytes per second (Bps)

**Panel 6.4: Recent Errors**
- **Type:** Table
- **Position:** (0,48) - 24x8
- **Metric:** `gl011_recent_errors{agent_id="$agent_id"}`
- **Columns:**
  - Timestamp
  - Error Type
  - Error Message
  - Endpoint
- **Limit:** Last 50 errors

### Template Variables

1. **agent_id**
   - Type: Query
   - Data Source: Prometheus
   - Query: `label_values(gl011_agent_health_status, agent_id)`
   - Default: `gl011-prod-01`

2. **time_range**
   - Type: Custom
   - Options: 1h, 3h, 6h (default), 12h, 24h

3. **fuel_type**
   - Type: Query
   - Data Source: Prometheus
   - Query: `label_values(gl011_fuel_cost_total, fuel_type)`
   - Multi-select: Yes
   - Include All: Yes

### Alerting Integration

- Annotations enabled for all critical alerts
- Alert history visible in Panel 6.1
- Active alert count in Panel 1.4

---

## 2. Business KPI Dashboard

**File:** `business_kpi_dashboard.json`
**UID:** `gl011-business-kpi`
**Refresh Rate:** 1 minute
**Default Time Range:** Last 24 hours

### Purpose
Business-focused monitoring for operations managers, procurement teams, and business analysts to track cost savings, fuel efficiency, and supplier performance.

### Panel Specifications (18 Panels)

#### Row 1: Financial Overview (y=0)

**Panel 1.1: Total Fuel Cost**
- **Type:** Stat (Big Number)
- **Position:** (0,0) - 6x6
- **Metric:** `sum(gl011_fuel_cost_total{agent_id="$agent_id", fuel_type=~"$fuel_type"})`
- **Unit:** USD currency
- **Display:** Large value with trend area graph

**Panel 1.2: Cost Savings vs Baseline**
- **Type:** Stat (Big Number)
- **Position:** (6,0) - 6x6
- **Metric:** `sum(gl011_cost_savings_vs_baseline{agent_id="$agent_id"})`
- **Unit:** USD currency
- **Thresholds:**
  - <$0 = RED (losing money)
  - $0-$50k = YELLOW
  - >$50k = GREEN
- **Display:** Large value with % change indicator

**Panel 1.3: Carbon Emissions**
- **Type:** Stat (Big Number)
- **Position:** (12,0) - 6x6
- **Metric:** `sum(gl011_carbon_emissions_kg{agent_id="$agent_id"})`
- **Unit:** kg CO2

**Panel 1.4: Carbon Reduction vs Baseline**
- **Type:** Stat (Big Number)
- **Position:** (18,0) - 6x6
- **Metric:** `sum(gl011_carbon_reduction_vs_baseline_kg{agent_id="$agent_id"})`
- **Unit:** kg CO2
- **Thresholds:**
  - <0 = RED
  - 0-10,000 kg = YELLOW
  - >10,000 kg = GREEN

#### Row 2: Energy & Efficiency (y=6)

**Panel 2.1: Energy Delivered**
- **Type:** Stat
- **Position:** (0,6) - 8x6
- **Metric:** `sum(gl011_energy_delivered_mwh{agent_id="$agent_id"})`
- **Unit:** MWh (megawatt-hours)

**Panel 2.2: Fuel Efficiency**
- **Type:** Gauge
- **Position:** (8,6) - 8x6
- **Metric:** `avg(gl011_fuel_efficiency_percent{agent_id="$agent_id"})`
- **Unit:** percent (%)
- **Thresholds:**
  - <70% = RED
  - 70-85% = YELLOW
  - >85% = GREEN
- **Target:** >85%

**Panel 2.3: Current Fuel Blend**
- **Type:** Pie Chart
- **Position:** (16,6) - 8x6
- **Metric:** `sum by (fuel_type) (gl011_fuel_blend_ratio{agent_id="$agent_id"})`
- **Legend:** `{{fuel_type}}`
- **Display:** Name and percentage

#### Row 3: Cost Analysis (y=12)

**Panel 3.1: Fuel Cost by Type (Stacked)**
- **Type:** Time series (Stacked Bars)
- **Position:** (0,12) - 12x10
- **Metric:** `sum by (fuel_type) (gl011_fuel_cost_by_type{agent_id="$agent_id"})`
- **Legend:** `{{fuel_type}}`
- **Unit:** USD
- **Stacking:** Normal
- **Fill Opacity:** 80%
- **Aggregations:** sum, mean

**Panel 3.2: Fuel Price Trends**
- **Type:** Time series (Multi-line)
- **Position:** (12,12) - 12x10
- **Metric:** `gl011_fuel_price_per_unit{agent_id="$agent_id", fuel_type=~"$fuel_type"}`
- **Legend:** `{{fuel_type}} - {{unit}}`
- **Unit:** USD
- **Aggregations:** mean, lastNotNull, max

#### Row 4: Inventory & Optimization (y=22)

**Panel 4.1: Inventory Levels**
- **Type:** Bar Gauge
- **Position:** (0,22) - 12x8
- **Metric:** `gl011_inventory_level_units{agent_id="$agent_id", fuel_type=~"$fuel_type"}`
- **Legend:** `{{fuel_type}}`
- **Unit:** units
- **Thresholds:**
  - <10,000 = RED (low inventory)
  - 10,000-50,000 = YELLOW
  - >50,000 = GREEN
- **Orientation:** Horizontal

**Panel 4.2: Optimization Recommendations Applied**
- **Type:** Stat
- **Position:** (12,22) - 12x8
- **Metric:** `sum(increase(gl011_optimization_recommendations_applied[24h]))`
- **Display:** Count over last 24 hours

#### Row 5: Contracts & Suppliers (y=30)

**Panel 5.1: Contract Utilization**
- **Type:** Gauge (Multiple)
- **Position:** (0,30) - 12x8
- **Metric:** `gl011_contract_utilization_percent{agent_id="$agent_id"}`
- **Legend:** `{{contract_id}} - {{supplier}}`
- **Unit:** percentunit (0-1 scale)
- **Thresholds:**
  - <0.7 = RED (underutilized)
  - 0.7-0.85 = YELLOW
  - >0.85 = GREEN
- **Target:** >85%

**Panel 5.2: Supplier Performance**
- **Type:** Table
- **Position:** (12,30) - 12x8
- **Metrics:**
  - Performance Score: `gl011_supplier_performance_score`
  - Total Spend: `gl011_supplier_total_spend`
- **Columns:**
  - Supplier Name
  - Fuel Type
  - Performance Score (color-coded: <70=RED, 70-85=YELLOW, >85=GREEN)
  - Total Spend (USD)
- **Sort:** Descending by Performance Score

#### Row 6: Procurement & Quality (y=38)

**Panel 6.1: Procurement Orders**
- **Type:** Time series
- **Position:** (0,38) - 12x8
- **Metric:** `sum by (order_type) (rate(gl011_procurement_orders_total[1h]))`
- **Legend:** `{{order_type}}`
- **Aggregations:** sum, mean

**Panel 6.2: Fuel Quality Violations**
- **Type:** Table
- **Position:** (12,38) - 12x8
- **Metric:** `gl011_fuel_quality_violations{agent_id="$agent_id"}`
- **Columns:**
  - Timestamp
  - Fuel Type
  - Violation Type
  - Severity (color-coded: critical=RED, warning=YELLOW, minor=BLUE)
- **Sort:** Descending by timestamp

#### Row 7: Compliance & ROI (y=46)

**Panel 7.1: Emissions Compliance Status**
- **Type:** Stat (Traffic Light)
- **Position:** (0,46) - 8x6
- **Metric:** `gl011_emissions_compliance_status{agent_id="$agent_id"}`
- **Value Mapping:**
  - 0 = RED (NON-COMPLIANT)
  - 1 = GREEN (COMPLIANT)
- **Display:** Background color mode

**Panel 7.2: Cost per MWh Trend**
- **Type:** Time series
- **Position:** (8,46) - 8x6
- **Metric:** `gl011_cost_per_mwh{agent_id="$agent_id", fuel_type=~"$fuel_type"}`
- **Legend:** `{{fuel_type}}`
- **Unit:** USD/MWh

**Panel 7.3: ROI Tracker (Waterfall Chart)**
- **Type:** Time series (Bars)
- **Position:** (16,46) - 8x6
- **Metric:** `gl011_roi_waterfall{agent_id="$agent_id"}`
- **Legend:** `{{category}}`
- **Unit:** USD
- **Display:** Waterfall-style bar chart showing cost breakdown

### Template Variables

1. **agent_id** (same as Operations Dashboard)
2. **fuel_type** (same as Operations Dashboard)

---

## 3. Executive Dashboard

**File:** `executive_dashboard.json`
**UID:** `gl011-executive`
**Refresh Rate:** 5 minutes
**Default Time Range:** Last 30 days

### Purpose
High-level strategic monitoring for executives, C-suite, and stakeholders to track overall performance, compliance, and business impact.

### Panel Specifications (12 Panels)

#### Row 1: Key Performance Indicators (y=0)

**Panel 1.1: Monthly Cost Savings**
- **Type:** Stat (Big Number)
- **Position:** (0,0) - 6x8
- **Metric:** `sum(gl011_monthly_cost_savings{agent_id="$agent_id"})`
- **Unit:** USD
- **Thresholds:**
  - <$0 = RED
  - $0-$100k = YELLOW
  - >$100k = GREEN
- **Display:** Extra large value (size: 40)
- **Graph Mode:** Area trend

**Panel 1.2: YTD Carbon Reduction**
- **Type:** Stat (Big Number)
- **Position:** (6,0) - 6x8
- **Metric:** `sum(gl011_ytd_carbon_reduction_kg{agent_id="$agent_id"})`
- **Unit:** kg CO2
- **Thresholds:**
  - <0 = RED
  - 0-50,000 kg = YELLOW
  - >50,000 kg = GREEN
- **Display:** Extra large value (size: 40)
- **Graph Mode:** Area trend

**Panel 1.3: System Uptime**
- **Type:** Gauge
- **Position:** (12,0) - 6x8
- **Metric:** `avg(gl011_system_uptime_percent{agent_id="$agent_id"})`
- **Unit:** percentunit (0-1 scale)
- **Thresholds:**
  - <0.95 = RED
  - 0.95-0.99 = YELLOW
  - >0.99 = GREEN
- **Target:** >99%
- **Display:** Large gauge (size: 40)

**Panel 1.4: Cost Optimization Score**
- **Type:** Gauge
- **Position:** (18,0) - 6x8
- **Metric:** `gl011_cost_optimization_score{agent_id="$agent_id"}`
- **Unit:** Score (0-100)
- **Thresholds:**
  - 0-50 = RED
  - 50-75 = YELLOW
  - 75-90 = GREEN
  - 90-100 = DARK GREEN
- **Min:** 0
- **Max:** 100
- **Display:** Large gauge (size: 40)

#### Row 2: Performance & Alerts (y=8)

**Panel 2.1: Environmental Performance**
- **Type:** Gauge
- **Position:** (0,8) - 6x6
- **Metric:** `gl011_environmental_performance_score{agent_id="$agent_id"}`
- **Unit:** Score (0-100)
- **Thresholds:**
  - 0-50 = RED
  - 50-75 = YELLOW
  - 75-90 = GREEN
  - 90-100 = DARK GREEN

**Panel 2.2: Critical Alerts**
- **Type:** Stat
- **Position:** (6,8) - 6x6
- **Metric:** `sum(gl011_critical_alerts_count{agent_id="$agent_id"})`
- **Thresholds:**
  - 0 = GREEN
  - 1-4 = YELLOW
  - 5+ = RED
- **Display:** Background color mode

**Panel 2.3: Fuel Cost Trend (90 days)**
- **Type:** Time series (Sparkline)
- **Position:** (12,8) - 12x6
- **Metric:** `sum(gl011_fuel_cost_trend_90d{agent_id="$agent_id"})`
- **Unit:** USD
- **Display:** Minimal sparkline (no legend)
- **Purpose:** Quick trend visualization

#### Row 3: Cost Drivers & Compliance (y=14)

**Panel 3.1: Top 5 Cost Drivers**
- **Type:** Table
- **Position:** (0,14) - 12x10
- **Metric:** `topk(5, gl011_cost_driver_impact{agent_id="$agent_id"})`
- **Columns:**
  - Cost Driver
  - Category
  - Fuel Type
  - Cost Impact (USD, color-coded background)
- **Sort:** Descending by Cost Impact
- **Transformation:** Organize to rename columns

**Panel 3.2: Compliance Status (Traffic Lights)**
- **Type:** Stat (Multiple)
- **Position:** (12,14) - 6x10
- **Metrics:**
  - Fuel Quality: `gl011_fuel_quality_compliance`
  - Emissions: `gl011_emissions_compliance`
  - Contracts: `gl011_contract_compliance`
- **Value Mapping:**
  - 0 = RED (NON-COMPLIANT)
  - 1 = GREEN (COMPLIANT)
- **Display:** Vertical layout, background color mode
- **Legend:** Custom labels (Fuel Quality, Emissions, Contracts)

**Panel 3.3: Optimization Success Rate**
- **Type:** Gauge
- **Position:** (18,14) - 6x10
- **Metric:** `avg(gl011_optimization_success_rate{agent_id="$agent_id"})`
- **Unit:** percentunit (0-1 scale)
- **Thresholds:**
  - <0.7 = RED
  - 0.7-0.85 = YELLOW
  - >0.85 = GREEN
- **Target:** >85%

#### Row 4: Integrations & Achievements (y=24)

**Panel 4.1: Integration Health Summary**
- **Type:** Table (Status Grid)
- **Position:** (0,24) - 12x8
- **Metric:** `gl011_integration_status{agent_id="$agent_id"}`
- **Columns:**
  - Integration Name
  - Type
  - Status (color-coded: 0=RED/DOWN, 1=GREEN/UP)
- **Display:** Center-aligned status with background color

**Panel 4.2: Recent Achievements**
- **Type:** Table
- **Position:** (12,24) - 12x8
- **Metric:** `gl011_recent_achievements{agent_id="$agent_id"}`
- **Columns:**
  - Timestamp
  - Achievement Type
  - Savings (USD, color-coded)
  - Emissions Avoided (kg CO2, color-coded)
  - Efficiency Gain (%, color-coded)
- **Sort:** Descending by Timestamp
- **Purpose:** Highlight wins for stakeholders

### Template Variables

1. **agent_id** (same as previous dashboards)

### Executive-Specific Features

- **Simplified Metrics:** Only high-level KPIs, no technical details
- **Longer Time Range:** 30-day default for strategic view
- **Lower Refresh Rate:** 5 minutes (executives don't need real-time)
- **Color-Coded Compliance:** Immediate visual status (red/yellow/green)
- **Achievement Tracking:** Positive reinforcement of system value

---

## Common Dashboard Features

### Color Schemes

**Thresholds (Standard across all dashboards):**
- **RED:** Critical issues, non-compliance, poor performance (<70%)
- **YELLOW:** Warning state, moderate performance (70-85%)
- **GREEN:** Good performance, compliant, target met (85-95%)
- **DARK GREEN:** Excellent performance, exceeding targets (>95%)

**Heatmaps:**
- **Spectral:** For latency/performance heatmaps (blue=fast, red=slow)
- **RdYlGn:** For optimization efficiency (red=inefficient, green=efficient)

### Annotations

- All dashboards support Grafana alert annotations
- Critical alerts displayed as red vertical lines
- Warning alerts displayed as yellow vertical lines
- Deployment markers (optional, via external annotations API)

### Time Range Controls

| Dashboard | Default | Selectable Options |
|-----------|---------|-------------------|
| Operations | 6 hours | 1h, 3h, 6h, 12h, 24h |
| Business KPI | 24 hours | 6h, 12h, 24h, 7d, 30d |
| Executive | 30 days | 7d, 30d, 90d, 1y |

### Refresh Rates

| Dashboard | Default | Selectable Options |
|-----------|---------|-------------------|
| Operations | 30s | 10s, 30s, 1m, 5m, 15m, 30m, 1h |
| Business KPI | 1m | 30s, 1m, 5m, 15m, 30m, 1h |
| Executive | 5m | 1m, 5m, 15m, 30m, 1h, 2h |

### Cross-Dashboard Navigation

- All dashboards tagged with `GL-011`
- Dashboard links panel in top-right corner
- Variables preserved during navigation
- Time range synced across dashboards

---

## Prometheus Metric Requirements

### Operations Metrics

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
gl011_concurrent_requests{agent_id}

# Integration Metrics
gl011_integration_health{agent_id, integration_type}
gl011_queue_depth{agent_id, queue_name}
gl011_optimization_execution_seconds_bucket{agent_id, optimization_type, le}

# Calculator & Provenance
gl011_calculator_performance_seconds{agent_id, calculator_type}
gl011_provenance_records_created{agent_id}

# Alerts & Logs
gl011_active_alerts{agent_id, severity}
gl011_alerts_history{agent_id, alert_name, severity}
gl011_log_entries_total{agent_id, log_level}

# Network
gl011_network_bytes_in{agent_id}
gl011_network_bytes_out{agent_id}

# Errors
gl011_recent_errors{agent_id, error_type, endpoint}
```

### Business KPI Metrics

```promql
# Financial
gl011_fuel_cost_total{agent_id, fuel_type}
gl011_cost_savings_vs_baseline{agent_id, fuel_type}
gl011_fuel_cost_by_type{agent_id, fuel_type}
gl011_fuel_price_per_unit{agent_id, fuel_type, unit}

# Environmental
gl011_carbon_emissions_kg{agent_id, fuel_type}
gl011_carbon_reduction_vs_baseline_kg{agent_id, fuel_type}

# Energy & Efficiency
gl011_energy_delivered_mwh{agent_id, fuel_type}
gl011_fuel_efficiency_percent{agent_id, fuel_type}
gl011_fuel_blend_ratio{agent_id, fuel_type}

# Inventory & Optimization
gl011_inventory_level_units{agent_id, fuel_type}
gl011_optimization_recommendations_applied{agent_id}

# Contracts & Suppliers
gl011_contract_utilization_percent{agent_id, contract_id, supplier}
gl011_supplier_performance_score{agent_id, supplier_name, fuel_type}
gl011_supplier_total_spend{agent_id, supplier_name}

# Procurement & Quality
gl011_procurement_orders_total{agent_id, order_type, fuel_type}
gl011_fuel_quality_violations{agent_id, fuel_type, violation_type, severity}

# Compliance & ROI
gl011_emissions_compliance_status{agent_id}
gl011_cost_per_mwh{agent_id, fuel_type}
gl011_roi_waterfall{agent_id, category}
```

### Executive Metrics

```promql
# High-Level KPIs
gl011_monthly_cost_savings{agent_id}
gl011_ytd_carbon_reduction_kg{agent_id}
gl011_system_uptime_percent{agent_id}
gl011_cost_optimization_score{agent_id}
gl011_environmental_performance_score{agent_id}

# Alerts
gl011_critical_alerts_count{agent_id}

# Trends
gl011_fuel_cost_trend_90d{agent_id}

# Cost Analysis
gl011_cost_driver_impact{agent_id, cost_driver, category, fuel_type}

# Compliance
gl011_fuel_quality_compliance{agent_id}
gl011_emissions_compliance{agent_id}
gl011_contract_compliance{agent_id}

# Performance
gl011_optimization_success_rate{agent_id}

# Status
gl011_integration_status{agent_id, integration_name, integration_type}

# Achievements
gl011_recent_achievements{agent_id, achievement_type, savings_usd, emissions_avoided_kg, efficiency_gain_percent}
```

---

## Alert Rules Integration

### Critical Alerts (P0)

```yaml
- alert: GL011AgentDown
  expr: gl011_agent_health_status == 0
  for: 1m
  annotations:
    summary: "GL-011 agent is unhealthy"

- alert: GL011HighErrorRate
  expr: rate(gl011_errors_total[5m]) > 10
  for: 5m
  annotations:
    summary: "High error rate detected"

- alert: GL011ComplianceViolation
  expr: gl011_emissions_compliance_status == 0
  for: 0m
  annotations:
    summary: "CRITICAL: Emissions compliance violated"
```

### Warning Alerts (P1)

```yaml
- alert: GL011LowCacheHitRate
  expr: gl011_cache_hit_rate < 0.7
  for: 10m
  annotations:
    summary: "Cache hit rate below 70%"

- alert: GL011HighLatency
  expr: histogram_quantile(0.95, rate(gl011_request_duration_seconds_bucket[5m])) > 5
  for: 5m
  annotations:
    summary: "P95 latency exceeds 5 seconds"

- alert: GL011LowContractUtilization
  expr: gl011_contract_utilization_percent < 0.7
  for: 1h
  annotations:
    summary: "Contract underutilized (<70%)"
```

---

## Access Control & Permissions

### Role-Based Dashboard Access

| Dashboard | Roles | Permissions |
|-----------|-------|-------------|
| Operations | DevOps, SRE, Developers | View, Edit |
| Business KPI | Procurement Managers, Operations | View, Edit |
| Executive | Executives, C-Suite, Stakeholders | View Only |

### Data Security

- All dashboards use template variable `agent_id` for multi-tenancy
- Fuel type filtering available to restrict view by fuel category
- Sensitive cost data only visible to authorized roles
- Audit logs for dashboard access (Grafana Enterprise feature)

---

## Performance Optimization

### Query Optimization

- Use `rate()` and `increase()` over raw counters
- Apply `[5m]` or `[1h]` range vectors for smoothing
- Use `topk()` to limit high-cardinality queries
- Avoid `*` wildcards in metric names

### Dashboard Performance

- Limit time series per panel to <20 for readability
- Use `instant: true` for table queries to reduce data transfer
- Set appropriate refresh rates (don't over-poll)
- Use query result caching where possible

### Cardinality Management

- Limit label values (e.g., max 20 fuel types, 10 suppliers)
- Use aggregations (`sum by`, `avg by`) to reduce series count
- Drop high-cardinality labels in recording rules

---

## Deployment & Maintenance

### Initial Setup

1. Import dashboard JSON files via Grafana UI or API
2. Configure Prometheus data source
3. Verify template variables resolve correctly
4. Set up alert channels (email, Slack, PagerDuty)
5. Configure RBAC permissions

### Version Control

- Store dashboard JSON in Git repository
- Use `uid` field for dashboard tracking
- Increment `version` field on updates
- Document changes in commit messages

### Regular Maintenance

- **Weekly:** Review alert thresholds based on performance trends
- **Monthly:** Update dashboard panels based on user feedback
- **Quarterly:** Audit metric retention and cardinality
- **Annually:** Major dashboard redesign based on business evolution

---

## Troubleshooting

### Common Issues

**Issue:** "No data" in panels
**Solution:** Verify Prometheus scraping GL-011 metrics, check `agent_id` variable

**Issue:** Slow dashboard load
**Solution:** Reduce time range, add query time limits, optimize PromQL queries

**Issue:** Incorrect values
**Solution:** Check metric units (bytes vs GB, percentunit vs percent), verify aggregations

**Issue:** Missing template variables
**Solution:** Re-import dashboard, ensure Prometheus data source configured

---

## Future Enhancements

### Planned Features (Q1 2026)

- **Machine Learning Insights:** Anomaly detection overlays on time series
- **Predictive Analytics:** Forecasted fuel costs and carbon emissions
- **What-If Scenarios:** Interactive cost modeling panels
- **Mobile Dashboards:** Optimized layouts for tablet/phone viewing
- **Custom Reports:** Scheduled PDF exports for stakeholders

### Experimental Features

- **AR/VR Dashboards:** 3D visualization of fuel blend optimization
- **Voice Alerts:** Alexa/Google Assistant integration for critical alerts
- **Natural Language Queries:** Ask Grafana questions in plain English

---

## Conclusion

These three dashboards provide comprehensive monitoring coverage for GL-011 FUELCRAFT across technical operations, business KPIs, and executive strategy. They leverage Prometheus metrics, Grafana visualization best practices, and GreenLang's DevOps standards to deliver actionable insights at every organizational level.

For questions or support, contact the GreenLang DevOps team or refer to the accompanying `README.md`.
