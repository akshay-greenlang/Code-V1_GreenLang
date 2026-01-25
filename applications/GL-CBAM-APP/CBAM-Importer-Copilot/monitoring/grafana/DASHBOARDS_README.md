# CBAM Importer Copilot - Grafana Dashboards Documentation

## Overview

This directory contains 5 comprehensive Grafana dashboards designed to achieve +1 maturity point in the CBAM Importer Copilot monitoring system. These dashboards provide deep insights into agent performance, data quality, compliance, emissions analytics, and business KPIs.

**Sprint 3+4 Target:** Increase maturity score from 91/100 to 94/100
**Monitoring Contribution:** +1 point (from enhanced observability)

---

## Dashboard Inventory

### 1. **cbam_agent_performance.json**
- **Lines:** ~410
- **Panels:** 17 panels
- **Refresh:** 30 seconds
- **Default Time Range:** Last 24 hours

#### Purpose
Monitors individual agent performance for the three CBAM agents: Intake, Calculator, and Packager.

#### Key Metrics Visualized
- **Agent Execution Time:**
  - P50, P95, P99 percentiles for each agent
  - Separate graphs for Intake, Calculator, and Packager agents

- **Success & Failure Rates:**
  - Individual success rates for each agent (target: >99%)
  - Color-coded thresholds (green >99%, yellow >95%, red <95%)

- **Throughput:**
  - Shipments processed per second by each agent
  - Processing speed in milliseconds per record

- **Resource Usage:**
  - CPU usage percentage (0-100%)
  - Memory usage in megabytes
  - Thresholds for resource alerts

- **Failure Analysis:**
  - Failures broken down by type and agent
  - Total failure count with trend analysis

#### Key Prometheus Queries
```promql
# Agent execution time (P95)
histogram_quantile(0.95, sum(rate(cbam_agent_duration_seconds_bucket{agent="intake"}[5m])) by (le))

# Agent success rate
sum(rate(cbam_agent_executions_total{agent="calculator",status="success"}[5m])) / sum(rate(cbam_agent_executions_total{agent="calculator"}[5m])) * 100

# Agent throughput
sum(rate(cbam_agent_shipments_processed_total{agent="packager"}[5m]))

# Agent CPU usage
cbam_agent_cpu_usage_percent{agent="intake"}
```

#### Template Variables
- `environment`: Filter by deployment environment
- `agent`: Filter by specific agent(s) (multi-select)

---

### 2. **cbam_data_quality.json**
- **Lines:** ~470
- **Panels:** 21 panels
- **Refresh:** 1 minute
- **Default Time Range:** Last 24 hours

#### Purpose
Monitors data quality across validation, enrichment, and completeness metrics.

#### Key Metrics Visualized
- **Overall Data Quality:**
  - Overall quality score (calculated as 1 - error_rate)
  - Target: >98% quality score

- **Validation Errors:**
  - Error rate trends over time
  - Distribution by error code (pie chart)
  - Top 10 error codes (bar gauge)
  - Errors by category (stacked time series)

- **Supplier Data Quality:**
  - Quality score heatmap by supplier
  - Count of suppliers achieving >95% quality

- **CN Code Enrichment:**
  - Success rate of CN code lookups
  - Success/failure/not-found trends

- **Shipment Rejections:**
  - Rejection reasons breakdown (donut chart)
  - Rejection rate over time

- **Data Completeness:**
  - Field-level completeness metrics (bar gauge)
  - Overall completeness score
  - Critical fields: CN code, origin country, supplier ID, emissions data

#### Key Prometheus Queries
```promql
# Overall data quality score
(1 - (sum(rate(cbam_validation_errors_total[1h])) / sum(rate(cbam_records_processed_total[1h])))) * 100

# Validation errors by code
sum by (error_code) (increase(cbam_validation_errors_total[24h]))

# CN code enrichment success rate
sum(rate(cbam_cn_code_enrichment_total{status="success"}[5m])) / sum(rate(cbam_cn_code_enrichment_total[5m])) * 100

# Field completeness
cbam_field_completeness_percent{field="cn_code"}
```

#### Template Variables
- `environment`: Filter by deployment environment
- `error_category`: Filter by error category (multi-select)
- `supplier`: Filter by supplier ID (multi-select)

---

### 3. **cbam_compliance.json**
- **Lines:** ~410
- **Panels:** 21 panels
- **Refresh:** 5 minutes
- **Default Time Range:** Last 7 days

#### Purpose
Monitors EU CBAM compliance requirements including rule violations, complex goods cap, and reporting deadlines.

#### Key Metrics Visualized
- **Rule Violations:**
  - Total violations (last 7 days)
  - Violations by rule ID (bar gauge)
  - Violations by severity (pie chart)
  - Violation trends over time

- **Complex Goods Monitoring:**
  - Gauge showing percentage (20% regulatory cap)
  - Color thresholds: green <15%, yellow 15-18%, orange 18-20%, red >20%
  - Quarterly trend analysis

- **Calculation Methods:**
  - Distribution: supplier actual vs EU default values
  - Supplier actual data usage rate (higher is better)

- **Quarterly Reporting:**
  - Days until reporting deadline (with color alerts)
  - Quarter completion percentage
  - Reports ready for submission count

- **Compliance Validation:**
  - Pass/fail/warning rates
  - Validation trends over time

- **EU Member State Coverage:**
  - Import volume by member state (bar gauge)
  - Count of member states served

#### Key Prometheus Queries
```promql
# Total rule violations
sum(increase(cbam_rule_violations_total[7d]))

# Complex goods percentage
sum(cbam_complex_goods_total{period="current_quarter"}) / sum(cbam_total_imports{period="current_quarter"}) * 100

# Supplier actual data usage
sum(increase(cbam_emissions_calculations_total{calculation_method="supplier_actual"}[7d])) / sum(increase(cbam_emissions_calculations_total[7d])) * 100

# Days until deadline
cbam_days_until_deadline{deadline_type="quarterly_report"}
```

#### Template Variables
- `environment`: Filter by deployment environment
- `member_state`: Filter by EU member state (multi-select)

#### Annotations
- Deployment events
- 7-day deadline warning alerts

---

### 4. **cbam_emissions_analysis.json**
- **Lines:** ~470
- **Panels:** 23 panels
- **Refresh:** 5 minutes
- **Default Time Range:** Last 30 days

#### Purpose
Comprehensive carbon emissions analytics across products, countries, suppliers, and calculation methods.

#### Key Metrics Visualized
- **Total Emissions Overview:**
  - Current quarter total (tCO2)
  - Last 30 days total
  - Average emission intensity (tCO2/ton)
  - Calculation rate (tCO2/sec)

- **Emissions Trends:**
  - Daily emissions trend
  - Cumulative emissions

- **By Product Group:**
  - Distribution pie chart (cement, steel, aluminum, fertilizers, electricity, hydrogen)
  - Top 10 products by emissions

- **By Origin Country:**
  - Top 10 countries (bar gauge)
  - Trends for top countries over time

- **By Supplier:**
  - Top 20 suppliers by emissions
  - Total suppliers tracked
  - Average emissions per supplier

- **Calculation Method Analysis:**
  - Volume distribution (supplier actual vs EU default)
  - Method trends over time

- **Emission Intensity:**
  - Intensity by product group (tCO2/ton)
  - Intensity trends over time

#### Key Prometheus Queries
```promql
# Total emissions (current quarter)
sum(cbam_emissions_total_tco2{period="current_quarter"})

# Emissions by product group
sum by (product_group) (cbam_emissions_total_tco2)

# Top 10 countries by emissions
topk(10, sum by (origin_country) (cbam_emissions_total_tco2))

# Emission intensity by product
sum by (product_group) (cbam_emissions_total_tco2) / sum by (product_group) (cbam_product_quantity_tons)
```

#### Template Variables
- `environment`: Filter by deployment environment
- `product_group`: Filter by CBAM product group (multi-select)
- `origin_country`: Filter by origin country (multi-select)
- `supplier`: Filter by supplier ID (multi-select)

---

### 5. **cbam_business_kpis.json**
- **Lines:** ~360
- **Panels:** 22 panels
- **Refresh:** 30 seconds
- **Default Time Range:** Last 7 days

#### Purpose
Executive-level business KPIs and operational metrics for stakeholder visibility.

#### Key Metrics Visualized
- **Executive Summary:**
  - Overall success rate (target: >95%)
  - Shipments processed (last 24h)
  - EU importers served
  - Reports generated (current quarter)

- **Throughput Metrics:**
  - Daily throughput (successful vs failed)
  - Current throughput rate (shipments/sec)
  - Total shipments processed (cumulative)

- **Pipeline Performance:**
  - Execution time trends (P50, P95, P99)
  - Average pipeline duration
  - SLA compliance thresholds

- **API Performance:**
  - API latency by endpoint (P95, P99)
  - API request rate by endpoint
  - Performance threshold alerts

- **Error Metrics:**
  - Error rate trend (target: <5%)
  - Total errors (last 24h)
  - Error distribution

- **Business Summary:**
  - Quarterly report generation trends
  - Average reports per importer
  - Platform uptime percentage (7-day)

#### Key Prometheus Queries
```promql
# Overall success rate
sum(rate(cbam_pipeline_executions_total{status="success"}[1h])) / sum(rate(cbam_pipeline_executions_total[1h])) * 100

# Current throughput rate
sum(rate(cbam_shipments_processed_total[5m]))

# Pipeline P95 duration
histogram_quantile(0.95, sum(rate(cbam_pipeline_duration_seconds_bucket[5m])) by (le))

# API latency P99
histogram_quantile(0.99, sum(rate(cbam_api_latency_seconds_bucket[5m])) by (le, endpoint))

# Error rate
sum(rate(cbam_errors_total[5m])) / sum(rate(cbam_pipeline_executions_total[5m])) * 100
```

#### Template Variables
- `environment`: Filter by deployment environment
- `importer`: Filter by importer ID (multi-select)

#### Annotations
- Deployment events
- High error rate alerts (>5%)

---

## Installation Instructions

### Prerequisites
- Grafana 8.0+ installed and running
- Prometheus data source configured in Grafana
- CBAM Importer Copilot application instrumented with metrics

### Step 1: Import Dashboards

#### Option A: Via Grafana UI
1. Log in to Grafana
2. Navigate to **Dashboards → Import**
3. Click **Upload JSON file**
4. Select one of the dashboard JSON files
5. Choose the Prometheus datasource from the dropdown
6. Click **Import**
7. Repeat for all 5 dashboards

#### Option B: Via Grafana API
```bash
# Set your Grafana credentials
GRAFANA_URL="http://localhost:3000"
GRAFANA_API_KEY="your-api-key"

# Import all dashboards
for dashboard in cbam_*.json; do
  curl -X POST \
    -H "Authorization: Bearer $GRAFANA_API_KEY" \
    -H "Content-Type: application/json" \
    -d @"$dashboard" \
    "$GRAFANA_URL/api/dashboards/db"
done
```

#### Option C: Via Provisioning (Recommended for Production)
1. Copy dashboard JSON files to Grafana provisioning directory:
   ```bash
   cp cbam_*.json /etc/grafana/provisioning/dashboards/
   ```

2. Create or update provisioning YAML:
   ```yaml
   # /etc/grafana/provisioning/dashboards/cbam.yaml
   apiVersion: 1
   providers:
     - name: 'CBAM Dashboards'
       orgId: 1
       folder: 'CBAM'
       type: file
       disableDeletion: false
       updateIntervalSeconds: 30
       allowUiUpdates: true
       options:
         path: /etc/grafana/provisioning/dashboards/
         foldersFromFilesStructure: false
   ```

3. Restart Grafana:
   ```bash
   systemctl restart grafana-server
   ```

### Step 2: Configure Prometheus Datasource

Ensure Prometheus datasource is configured with the correct URL:

```yaml
# /etc/grafana/provisioning/datasources/prometheus.yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

### Step 3: Verify Installation

1. Navigate to **Dashboards → Browse**
2. Look for the "CBAM" folder or search for dashboards:
   - CBAM Agent Performance Dashboard
   - CBAM Data Quality Dashboard
   - CBAM Compliance Dashboard
   - CBAM Emissions Analysis Dashboard
   - CBAM Business KPIs Dashboard
3. Open each dashboard and verify panels are loading data

---

## Dashboard Usage Guide

### For Operations Teams

**Primary Dashboard:** `cbam_agent_performance.json`
- Monitor agent health and performance in real-time
- Investigate performance degradations
- Track resource utilization trends
- Alert on agent failures

**Secondary Dashboard:** `cbam_data_quality.json`
- Monitor validation error rates
- Identify problematic data sources
- Track enrichment success rates

### For Data Quality Teams

**Primary Dashboard:** `cbam_data_quality.json`
- Analyze validation error patterns
- Monitor supplier data quality
- Track data completeness metrics
- Investigate rejection reasons

**Secondary Dashboard:** `cbam_compliance.json`
- Ensure regulatory compliance
- Monitor rule violation trends

### For Compliance Teams

**Primary Dashboard:** `cbam_compliance.json`
- Track rule violations and remediation
- Monitor complex goods percentage (20% cap)
- Track reporting deadlines
- Ensure calculation method compliance

**Secondary Dashboard:** `cbam_emissions_analysis.json`
- Verify emissions data accuracy
- Audit calculation methods

### For Business Stakeholders

**Primary Dashboard:** `cbam_business_kpis.json`
- Monitor overall platform health
- Track throughput and success rates
- View business metrics (importers, reports)
- Monitor API performance and uptime

**Secondary Dashboard:** `cbam_emissions_analysis.json`
- Understand emissions trends
- Analyze by product, country, supplier

### For Sustainability Analysts

**Primary Dashboard:** `cbam_emissions_analysis.json`
- Deep-dive into emissions data
- Analyze by product group, country, supplier
- Track emission intensity trends
- Monitor calculation method usage

---

## Metrics Reference

### Core Metrics Used Across Dashboards

#### Pipeline Metrics
```promql
cbam_pipeline_executions_total{status="success|failed"}
cbam_pipeline_duration_seconds_bucket
cbam_pipeline_active
```

#### Agent Metrics
```promql
cbam_agent_duration_seconds_bucket{agent="intake|calculator|packager"}
cbam_agent_executions_total{agent, status}
cbam_agent_shipments_processed_total{agent}
cbam_agent_cpu_usage_percent{agent}
cbam_agent_memory_usage_bytes{agent}
cbam_agent_failures_total{agent, failure_type}
cbam_agent_ms_per_record{agent, quantile}
```

#### Validation & Quality Metrics
```promql
cbam_validation_errors_total{error_code, error_category, error_description}
cbam_validation_results_total{result}
cbam_supplier_data_quality_score{supplier_id}
cbam_cn_code_enrichment_total{status}
cbam_shipment_rejections_total{rejection_reason}
cbam_field_completeness_percent{field}
```

#### Compliance Metrics
```promql
cbam_rule_violations_total{rule_id, severity, rule_description}
cbam_complex_goods_total{period}
cbam_total_imports{period}
cbam_emissions_calculations_total{calculation_method}
cbam_days_until_deadline{deadline_type}
cbam_quarter_completion_percent{period}
cbam_compliance_validations_total{result}
cbam_imports_total{member_state}
```

#### Emissions Metrics
```promql
cbam_emissions_total_tco2{product_group, origin_country, supplier_id, calculation_method}
cbam_emissions_calculated_tco2
cbam_product_quantity_tons{product_group}
```

#### Business & API Metrics
```promql
cbam_shipments_processed_total{status}
cbam_reports_generated_total{period, quarter}
cbam_api_latency_seconds_bucket{endpoint}
cbam_api_requests_total{endpoint}
cbam_errors_total{error_type}
cbam_records_processed_total{status}
up{job="cbam-importer-copilot"}
```

---

## Alerting Configuration

### Recommended Alerts

Each dashboard includes built-in threshold visualizations. To enable actual alerting, configure the following in Prometheus Alertmanager:

#### Critical Alerts
```yaml
# Low success rate
- alert: CBAMPipelineSuccessRateLow
  expr: |
    sum(rate(cbam_pipeline_executions_total{status="success"}[5m])) /
    sum(rate(cbam_pipeline_executions_total[5m])) * 100 < 95
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "CBAM pipeline success rate below 95%"

# High error rate
- alert: CBAMErrorRateHigh
  expr: |
    sum(rate(cbam_errors_total[5m])) /
    sum(rate(cbam_pipeline_executions_total[5m])) * 100 > 5
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "CBAM error rate above 5%"

# Complex goods approaching 20% cap
- alert: CBAMComplexGoodsNearCap
  expr: |
    sum(cbam_complex_goods_total{period="current_quarter"}) /
    sum(cbam_total_imports{period="current_quarter"}) * 100 > 18
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "Complex goods approaching 20% regulatory cap"
```

#### Warning Alerts
```yaml
# High API latency
- alert: CBAMAPILatencyHigh
  expr: |
    histogram_quantile(0.95,
      sum(rate(cbam_api_latency_seconds_bucket[5m])) by (le, endpoint)
    ) > 1
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "CBAM API P95 latency above 1 second"

# Approaching reporting deadline
- alert: CBAMReportingDeadlineNear
  expr: cbam_days_until_deadline{deadline_type="quarterly_report"} < 7
  labels:
    severity: warning
  annotations:
    summary: "CBAM quarterly reporting deadline in less than 7 days"
```

---

## Customization Guide

### Adding New Panels

1. Open the dashboard JSON file
2. Increment the highest panel `id` by 1
3. Add panel configuration to `panels` array:
   ```json
   {
     "id": 24,
     "type": "graph",
     "title": "My New Panel",
     "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
     "gridPos": {"x": 0, "y": 73, "w": 12, "h": 8},
     "targets": [
       {
         "expr": "your_prometheus_query",
         "refId": "A",
         "legendFormat": "{{label}}"
       }
     ]
   }
   ```

### Adding Template Variables

Add to the `templating.list` array:
```json
{
  "name": "my_variable",
  "type": "query",
  "datasource": "${DS_PROMETHEUS}",
  "query": "label_values(cbam_metric, label_name)",
  "refresh": 1,
  "multi": true,
  "includeAll": true,
  "allValue": ".*"
}
```

### Modifying Thresholds

Update the `thresholds` section in `fieldConfig`:
```json
"thresholds": {
  "mode": "absolute",
  "steps": [
    {"value": 0, "color": "green"},
    {"value": 80, "color": "yellow"},
    {"value": 90, "color": "red"}
  ]
}
```

---

## Troubleshooting

### Dashboards Show "No Data"

**Cause:** Prometheus datasource not configured or metrics not being scraped

**Solution:**
1. Verify Prometheus is running: `curl http://prometheus:9090/-/healthy`
2. Check if CBAM metrics exist: `curl http://prometheus:9090/api/v1/label/__name__/values | grep cbam`
3. Verify scrape configuration in `prometheus.yml`
4. Check Grafana datasource settings

### Panels Show Errors

**Cause:** Invalid PromQL queries or missing labels

**Solution:**
1. Test query directly in Prometheus UI
2. Verify metric names and label names
3. Check for typos in dashboard JSON
4. Ensure time range has data

### Template Variables Not Populating

**Cause:** Prometheus datasource variable not set or metrics missing labels

**Solution:**
1. Verify `${DS_PROMETHEUS}` is configured
2. Check if metrics have the required labels
3. Test label_values query in Prometheus

### Performance Issues

**Cause:** Too many panels or expensive queries

**Solution:**
1. Increase refresh interval
2. Optimize PromQL queries (use recording rules)
3. Limit time range for expensive queries
4. Use `topk()` to limit result sets

---

## Maintenance

### Regular Tasks

**Weekly:**
- Review dashboard performance
- Check for missing data points
- Validate alert thresholds

**Monthly:**
- Review and update panel queries for optimization
- Add new metrics as application evolves
- Archive or remove obsolete panels

**Quarterly:**
- Review dashboard usage analytics
- Gather user feedback
- Plan dashboard enhancements

### Version Control

All dashboard JSON files should be committed to version control:

```bash
git add monitoring/grafana/*.json
git commit -m "chore: update CBAM Grafana dashboards"
git push
```

---

## Support & Contribution

### Reporting Issues

If you encounter issues with the dashboards:

1. Check this README for troubleshooting steps
2. Verify Prometheus metrics are being scraped
3. Test queries directly in Prometheus UI
4. Open an issue with:
   - Dashboard name and panel ID
   - Error message or screenshot
   - Prometheus query being used
   - Expected vs actual behavior

### Contributing Enhancements

To contribute improvements:

1. Create a branch: `git checkout -b feature/dashboard-enhancement`
2. Modify dashboard JSON files
3. Test in Grafana UI
4. Update this README if adding new metrics or panels
5. Submit pull request with description of changes

---

## Maturity Score Impact

### Current Score: 91/100
### Target Score: 94/100
### Monitoring Contribution: +1 point

These dashboards contribute to the monitoring maturity score through:

1. **Comprehensive Coverage** - All critical system components monitored
2. **Multi-dimensional Analysis** - Agent, data quality, compliance, emissions, business views
3. **Proactive Alerting** - Threshold-based visualizations for early warning
4. **Stakeholder Visibility** - Executive and operational dashboards
5. **Compliance Monitoring** - EU CBAM regulatory requirement tracking

### Complementary Improvements for +3 Points Total

**Already Achieved (+1 from these dashboards)**

**Recommended for +2 Additional Points:**
- Implement Prometheus recording rules for expensive queries
- Set up automated dashboard testing (snapshot testing)
- Create dashboard-as-code deployment pipeline
- Implement SLO tracking with SLI metrics
- Add distributed tracing visualization (Jaeger/Tempo integration)

---

## Dashboard Screenshots

(Note: Screenshots should be added after deployment showing key panels)

### Agent Performance Dashboard
- Agent execution time graphs
- Success rate statistics
- Resource usage trends

### Data Quality Dashboard
- Validation error distribution
- Supplier quality heatmap
- Data completeness metrics

### Compliance Dashboard
- Rule violation breakdown
- Complex goods gauge
- Deadline countdown

### Emissions Analysis Dashboard
- Emissions by product group
- Top countries by emissions
- Calculation method distribution

### Business KPIs Dashboard
- Executive summary stats
- Throughput trends
- API performance metrics

---

## Additional Resources

- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [EU CBAM Regulation](https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en)
- [Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/best-practices-for-creating-dashboards/)

---

**Version:** 1.0.0
**Last Updated:** 2025-11-18
**Maintained By:** CBAM Monitoring Team
**Contact:** monitoring@greenlang.com
