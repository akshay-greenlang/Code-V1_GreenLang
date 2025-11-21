# GL-001 Grafana Dashboard Specifications

## Overview

This document specifies the 5 core Grafana dashboards for GL-001 ProcessHeatOrchestrator. Each dashboard is optimized for specific stakeholder needs.

---

## 1. Master Orchestrator Dashboard

**File**: `master_orchestrator_dashboard.json`
**Audience**: DevOps, SRE
**Refresh**: 10s
**Time Range**: Last 1 hour (default)

### Panels

#### Row 1: Health & Availability
- **Orchestrator Status** (Stat)
  - Query: `gl_001_orchestrator_health_status`
  - Thresholds: 0=red, 1=green
  - Display: Current state with icon

- **Uptime** (Stat)
  - Query: `gl_001_orchestrator_uptime_seconds / 3600`
  - Unit: hours
  - Display: Time since last restart

- **Current State** (Stat)
  - Query: `gl_001_orchestrator_state`
  - Value Mappings: 0=INIT, 1=READY, 2=EXECUTING, 3=ERROR, 4=RECOVERING, 5=TERMINATED
  - Color: State-based

- **30-Day Availability** (Gauge)
  - Query: `100 * avg_over_time(up{job="gl-001-orchestrator"}[30d])`
  - Min: 99.0, Max: 100.0
  - Target: 99.9%

#### Row 2: Request Metrics
- **Request Rate** (Graph)
  - Query: `sum(rate(gl_001_orchestration_requests_total[5m])) by (orchestration_type)`
  - Legend: By orchestration type
  - Y-axis: Requests/sec

- **Request Latency (P50, P95, P99)** (Graph)
  - Queries:
    - P50: `histogram_quantile(0.50, sum(rate(gl_001_orchestration_duration_seconds_bucket[5m])) by (le))`
    - P95: `histogram_quantile(0.95, sum(rate(gl_001_orchestration_duration_seconds_bucket[5m])) by (le))`
    - P99: `histogram_quantile(0.99, sum(rate(gl_001_orchestration_duration_seconds_bucket[5m])) by (le))`
  - Y-axis: Seconds

- **Error Rate** (Graph)
  - Query: `100 * sum(rate(gl_001_orchestration_requests_total{status="failure"}[5m])) / sum(rate(gl_001_orchestration_requests_total[5m]))`
  - Y-axis: Percentage
  - Threshold: 1% warning, 5% critical

#### Row 3: Resource Utilization
- **CPU Usage** (Graph)
  - Query: `gl_001_system_cpu_usage_percent`
  - Y-axis: Percentage
  - Thresholds: 80% warning, 90% critical

- **Memory Usage** (Graph)
  - Query: `gl_001_system_memory_usage_bytes{type="rss"} / 1024 / 1024 / 1024`
  - Y-axis: GB
  - Thresholds: 6GB warning, 8GB critical

- **Cache Performance** (Graph)
  - Queries:
    - Hit Rate: `100 * sum(rate(gl_001_calculation_cache_hits_total[5m])) / (sum(rate(gl_001_calculation_cache_hits_total[5m])) + sum(rate(gl_001_calculation_cache_misses_total[5m])))`
    - Cache Size: `gl_001_calculation_cache_size`
  - Dual Y-axes

#### Row 4: Active Operations
- **Active Orchestrations** (Stat)
  - Query: `sum(gl_001_orchestrator_state{state_name="EXECUTING"})`
  - Display: Current count

- **Task Queue Depth** (Graph)
  - Query: `sum(gl_001_task_queue_depth) by (priority)`
  - Legend: By priority (critical, high, medium, low)
  - Stacked area

- **Recent Errors** (Table)
  - Query: `topk(10, rate(gl_001_orchestration_requests_total{status="failure"}[5m]))`
  - Columns: Orchestration Type, Error Rate, Count

---

## 2. Multi-Plant Dashboard

**File**: `multi_plant_dashboard.json`
**Audience**: Plant Managers, Operations
**Refresh**: 30s
**Time Range**: Last 4 hours (default)

### Panels

#### Row 1: Fleet Overview
- **Active Plants** (Stat)
  - Query: `gl_001_active_plants_count`
  - Display: Current count with trend

- **Fleet Efficiency** (Gauge)
  - Query: `gl_001_aggregate_thermal_efficiency_percent`
  - Min: 60, Max: 100
  - Target: 85%

- **Total Heat Generation** (Stat)
  - Query: `sum(gl_001_plant_heat_generation_mw)`
  - Unit: MW
  - Color: Green gradient

- **Total Heat Losses** (Stat)
  - Query: `sum(gl_001_plant_heat_losses_mw)`
  - Unit: MW
  - Color: Red gradient

#### Row 2: Plant Health Matrix
- **Plant Health Status** (Status Map)
  - Query: `gl_001_plant_health_status`
  - Display: Heatmap by plant_id and plant_name
  - Colors: 0=red (unhealthy), 1=green (healthy)

- **Plant Efficiency Heatmap** (Heatmap)
  - Query: `gl_001_plant_thermal_efficiency_percent`
  - X-axis: Time
  - Y-axis: Plant Name
  - Color Scale: 60% (red) → 100% (green)

#### Row 3: Plant Performance
- **Plant Efficiency Ranking** (Bar Gauge)
  - Query: `topk(10, gl_001_plant_thermal_efficiency_percent)`
  - Orientation: Horizontal
  - Sort: Descending

- **Capacity Utilization** (Bar Gauge)
  - Query: `gl_001_plant_capacity_utilization_percent`
  - Thresholds: <60% red, 60-80% yellow, >80% green

- **Heat Generation vs Demand** (Graph)
  - Queries:
    - Generation: `sum(gl_001_plant_heat_generation_mw) by (plant_id)`
    - Demand: `sum(gl_001_plant_heat_demand_mw) by (plant_id)`
  - Display: Stacked by plant

#### Row 4: Heat Losses Breakdown
- **Losses by Type** (Pie Chart)
  - Query: `sum(gl_001_plant_heat_losses_mw) by (loss_type)`
  - Types: distribution, radiation, flue_gas

- **Losses by Plant** (Graph)
  - Query: `gl_001_plant_heat_losses_mw`
  - Legend: By plant_id
  - Stacked area

#### Row 5: Cross-Plant Integration
- **Cross-Plant Heat Transfer** (Sankey Diagram)
  - Query: `gl_001_cross_plant_heat_transfer_mw`
  - Nodes: source_plant, destination_plant
  - Flow width: Transfer MW

- **Inter-Plant Flow Matrix** (Table)
  - Query: `gl_001_cross_plant_heat_transfer_mw > 0`
  - Columns: Source, Destination, MW, Efficiency

---

## 3. Sub-Agent Coordination Dashboard

**File**: `subagent_coordination_dashboard.json`
**Audience**: DevOps, Agent Developers
**Refresh**: 15s
**Time Range**: Last 1 hour (default)

### Panels

#### Row 1: Agent Fleet Status
- **Active Sub-Agents** (Stat)
  - Query: `gl_001_active_subagents_count`
  - Display: Current / Total (99)
  - Percentage: `100 * (gl_001_active_subagents_count / 99)`

- **Agent Availability** (Gauge)
  - Query: `100 * (gl_001_active_subagents_count / 99)`
  - Min: 90, Max: 100
  - Thresholds: <95% warning, <90% critical

- **Unhealthy Agents** (Stat)
  - Query: `count(gl_001_subagent_health_status == 0)`
  - Color: Red if > 0

#### Row 2: Agent Health Matrix (GL-002 to GL-100)
- **Agent Health Heatmap** (Status Map)
  - Query: `gl_001_subagent_health_status`
  - Display: Grid by agent_id
  - Colors: 0=red, 1=green
  - Layout: 10x10 grid (agents 002-100)

#### Row 3: Agent Performance
- **Agent Response Times** (Heatmap)
  - Query: `histogram_quantile(0.95, sum(rate(gl_001_subagent_response_time_seconds_bucket[5m])) by (le, agent_id))`
  - X-axis: Time
  - Y-axis: Agent ID
  - Color Scale: <1s green → >5s red

- **Slowest Agents (P95)** (Bar Gauge)
  - Query: `topk(10, histogram_quantile(0.95, sum(rate(gl_001_subagent_response_time_seconds_bucket[5m])) by (le, agent_id)))`
  - Orientation: Horizontal
  - Sort: Descending

#### Row 4: Task Coordination
- **Task Delegation Rate** (Graph)
  - Query: `sum(rate(gl_001_tasks_delegated_total[5m])) by (task_category)`
  - Legend: By category
  - Stacked area

- **Task Completion Rate** (Graph)
  - Query: `sum(rate(gl_001_tasks_completed_total[5m])) by (status)`
  - Legend: success, failure, timeout
  - Stacked area

- **Task Success Rate** (Gauge)
  - Query: `100 * sum(rate(gl_001_tasks_completed_total{status="success"}[5m])) / sum(rate(gl_001_tasks_completed_total[5m]))`
  - Min: 90, Max: 100
  - Target: 99%

#### Row 5: Message Queues
- **Queue Depths by Agent** (Graph)
  - Query: `gl_001_subagent_message_queue_depth`
  - Display: Line per agent (top 20)
  - Threshold: 50 messages warning

- **Agents with Highest Queue** (Table)
  - Query: `topk(20, gl_001_subagent_message_queue_depth)`
  - Columns: Agent ID, Agent Type, Queue Depth, Plant ID

#### Row 6: Coordination Failures
- **Failure Rate by Agent** (Graph)
  - Query: `rate(gl_001_subagent_coordination_failures_total[5m])`
  - Legend: By agent_id and failure_type
  - Y-axis: Failures/sec

- **Top Failing Agents** (Table)
  - Query: `topk(10, sum(rate(gl_001_subagent_coordination_failures_total[5m])) by (agent_id, failure_type))`
  - Columns: Agent ID, Failure Type, Rate

---

## 4. Thermal Efficiency Dashboard

**File**: `thermal_efficiency_dashboard.json`
**Audience**: Energy Managers, Plant Engineers
**Refresh**: 1m
**Time Range**: Last 24 hours (default)

### Panels

#### Row 1: Enterprise-Wide Efficiency
- **Fleet Thermal Efficiency** (Gauge)
  - Query: `gl_001_aggregate_thermal_efficiency_percent`
  - Min: 60, Max: 100
  - Zones: <75% red, 75-85% yellow, >85% green

- **Efficiency Trend (7 days)** (Graph)
  - Query: `gl_001_aggregate_thermal_efficiency_percent`
  - Time range override: 7d
  - Trend line

- **Total Heat Generated** (Stat)
  - Query: `sum(gl_001_aggregate_heat_generation_mw)`
  - Unit: MW

- **Total Heat Demand** (Stat)
  - Query: `sum(gl_001_aggregate_heat_demand_mw)`
  - Unit: MW

#### Row 2: Plant-Level Efficiency
- **Plant Efficiency Comparison** (Bar Gauge)
  - Query: `gl_001_plant_thermal_efficiency_percent`
  - Orientation: Horizontal
  - Sorted by value
  - Thresholds: <75% red, 75-85% yellow, >85% green

- **Efficiency Distribution** (Histogram)
  - Query: `gl_001_plant_thermal_efficiency_percent`
  - Buckets: 10% intervals (60-70, 70-80, 80-90, 90-100)
  - X-axis: Efficiency %
  - Y-axis: Plant count

#### Row 3: Efficiency Components
- **Boiler Efficiency by Plant** (Graph)
  - Query: `gl_001_plant_boiler_efficiency_percent`
  - Legend: By plant_name
  - Y-axis: Percentage

- **Heat Recovery Efficiency** (Graph)
  - Query: `gl_001_plant_heat_recovery_efficiency_percent`
  - Legend: By plant_name
  - Y-axis: Percentage

- **Distribution Efficiency** (Graph)
  - Query: `gl_001_plant_distribution_efficiency_percent`
  - Legend: By plant_name
  - Y-axis: Percentage

#### Row 4: Optimization Impact
- **Efficiency Improvements** (Graph)
  - Query: `rate(gl_001_efficiency_improvement_percent_sum[1h]) / rate(gl_001_efficiency_improvement_percent_count[1h])`
  - Legend: By plant_id
  - Y-axis: Percentage improvement

- **Optimization Events** (Graph)
  - Query: `sum(rate(gl_001_efficiency_optimization_events_total[5m])) by (optimization_type)`
  - Legend: By type
  - Stacked bar

#### Row 5: Energy Balance
- **Energy Balance Closure** (Graph)
  - Query: `gl_001_energy_balance_closure_percent`
  - Legend: By plant_id
  - Target line: 95%
  - Thresholds: <90% red

- **Balance Error** (Graph)
  - Query: `gl_001_energy_balance_error_mw`
  - Legend: By plant_id
  - Y-axis: MW imbalance

- **Energy Inputs vs Outputs** (Graph)
  - Queries:
    - Inputs: `sum(gl_001_energy_input_mw) by (energy_type)`
    - Outputs: `sum(gl_001_energy_output_mw) by (output_type)`
  - Dual Y-axes
  - Stacked area

---

## 5. Operations Dashboard

**File**: `operations_dashboard.json`
**Audience**: 24/7 Operations Team
**Refresh**: 15s
**Time Range**: Last 1 hour (default)

### Panels

#### Row 1: System Health Overview
- **Orchestrator Status** (Big Number)
  - Query: `gl_001_orchestrator_health_status`
  - Display: "HEALTHY" or "UNHEALTHY"
  - Full-screen mode: Green/Red background

- **Active Alerts** (Stat)
  - Query: `count(ALERTS{alertname=~"GL001.*", alertstate="firing"})`
  - Color: 0=green, >0=red
  - Link to Alertmanager

- **Critical Alarms** (Stat)
  - Query: `sum(gl_001_scada_alarms_active{severity="critical"})`
  - Color: 0=green, >0=red

#### Row 2: Integration Health
- **SCADA Connections** (Status Map)
  - Query: `gl_001_scada_connection_status`
  - Display: Matrix by plant_id and scada_system
  - Colors: 0=red, 1=green

- **ERP Integration** (Status Map)
  - Query: `gl_001_erp_connection_status`
  - Display: By erp_system
  - Colors: 0=red, 1=green

- **Data Quality Score** (Gauge)
  - Query: `avg(gl_001_scada_data_quality_percent)`
  - Min: 80, Max: 100
  - Target: 95%

#### Row 3: Real-Time Operations
- **Active Plants** (World Map or Site Map)
  - Query: `gl_001_plant_health_status`
  - Display: Geographic locations with health markers
  - Tooltip: Plant name, efficiency, status

- **Current Heat Generation** (Graph)
  - Query: `sum(gl_001_plant_heat_generation_mw) by (plant_id)`
  - Display: Real-time stacked area
  - Auto-refresh: 10s

#### Row 4: Task Queues & Performance
- **Task Queue Depths** (Bar Gauge)
  - Query: `sum(gl_001_task_queue_depth) by (priority)`
  - Categories: critical, high, medium, low
  - Orientation: Horizontal

- **Processing Rate** (Stat)
  - Query: `sum(rate(gl_001_tasks_completed_total[1m]))`
  - Unit: Tasks/sec
  - Trend sparkline

#### Row 5: Recent Activity
- **Recent Errors** (Logs Panel)
  - Query: `{job="gl-001-orchestrator"} |= "ERROR"`
  - Limit: Last 20
  - Auto-scroll

- **Recent Optimizations** (Table)
  - Query: `topk(10, rate(gl_001_efficiency_optimization_events_total[5m]))`
  - Columns: Plant, Type, Status, Rate

#### Row 6: Key Performance Indicators
- **Fleet Efficiency** (Stat)
  - Query: `gl_001_aggregate_thermal_efficiency_percent`
  - Size: Large
  - Trend: 24h

- **Cost Savings (Today)** (Stat)
  - Query: `sum(gl_001_total_cost_savings_usd_hr) * 24`
  - Unit: USD
  - Prefix: $

- **CO2 Reduction (Today)** (Stat)
  - Query: `sum(gl_001_emissions_co2_tons_hr) * 24`
  - Unit: tons
  - Color: Green

---

## Dashboard Import Instructions

### Option 1: Via Grafana UI
1. Login to Grafana
2. Navigate to Dashboards → Import
3. Upload JSON file or paste JSON content
4. Select Prometheus data source
5. Click Import

### Option 2: Via ConfigMap (Kubernetes)
```bash
kubectl create configmap gl-001-grafana-dashboards \
  --from-file=grafana/ \
  --namespace=greenlang

kubectl label configmap gl-001-grafana-dashboards \
  grafana_dashboard=1 \
  --namespace=greenlang
```

### Option 3: Via Grafana API
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @master_orchestrator_dashboard.json \
  http://grafana.greenlang.io/api/dashboards/db
```

## Variables & Templating

All dashboards support these template variables:

- **$plant_id** - Multi-select dropdown of plant IDs
- **$agent_id** - Multi-select dropdown of sub-agent IDs (GL-002 to GL-100)
- **$time_range** - Global time range selector
- **$refresh_interval** - Auto-refresh interval (10s, 30s, 1m, 5m)
- **$orchestration_type** - Filter by orchestration type

Example variable query:
```promql
label_values(gl_001_plant_health_status, plant_id)
```

## Annotations

Enable annotations for:
- Deployment events
- Configuration changes
- Optimization events
- Alert firing/resolution

Example annotation query:
```promql
ALERTS{alertname=~"GL001.*"}
```

---

**Note**: Full JSON dashboard files can be generated using Grafana's export feature or via the GreenLang dashboard generator tool.

**Contact**: devops@greenlang.io for custom dashboard requests
