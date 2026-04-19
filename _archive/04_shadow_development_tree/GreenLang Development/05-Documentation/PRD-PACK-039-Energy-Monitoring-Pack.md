# PRD-PACK-039: Energy Monitoring Pack

**Pack ID:** PACK-039-energy-monitoring
**Category:** Energy Efficiency Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-23
**Prerequisite:** None (standalone; enhanced with PACK-031 Industrial Energy Audit, PACK-032 Building Energy Assessment, PACK-036 Utility Analysis, and PACK-038 Peak Shaving if present)

---

## 1. Executive Summary

### 1.1 Problem Statement

Energy monitoring is the foundation of all energy management programs, yet most organizations lack comprehensive, real-time visibility into their energy consumption patterns. ISO 50001:2018 mandates that organizations "determine energy performance indicators (EnPIs) appropriate for measuring and monitoring energy performance" and "establish an energy measurement plan," but implementation remains fragmented:

1. **Fragmented data sources**: Facilities typically have 5-15 separate metering systems (utility meters, submeters, BMS, SCADA, IoT sensors, manual readings) with no unified data platform. Each system uses different protocols (Modbus, BACnet, MQTT, OPC-UA, Pulse), different intervals (1-sec to monthly), and different units. Without automated data fusion, energy managers spend 60-70% of their time on data collection and reconciliation rather than analysis.

2. **Missing submetering coverage**: While utility meters capture whole-building consumption, most facilities submeter less than 30% of their electrical loads. Without submetering to the system level (HVAC, lighting, process, plug loads), organizations cannot identify where energy is consumed, detect waste, or attribute costs to departments or tenants. ASHRAE Guideline 14-2014 recommends submetering major end uses to within 80% of total consumption.

3. **No real-time anomaly detection**: Energy waste from equipment malfunctions, scheduling errors, simultaneous heating/cooling, and after-hours operation can persist for weeks or months before discovery. Manual bill review catches anomalies only at billing frequency (monthly), costing 5-15% of total energy spend. Real-time anomaly detection using statistical process control, CUSUM, and machine learning can identify waste within minutes.

4. **Inadequate EnPI tracking**: ISO 50001 requires Energy Performance Indicators that normalize consumption for relevant variables (weather, production, occupancy). Most organizations track only raw kWh, which conflates weather effects, production changes, and efficiency improvements. Proper EnPIs require regression-based normalization (IPMVP Option C) with rolling baselines, significance testing, and uncertainty quantification.

5. **Poor energy cost allocation**: Multi-tenant buildings, campus facilities, and manufacturing plants need accurate cost allocation by department, tenant, process, or product. Without interval-level submetering and tariff-aware allocation, organizations use simplistic area-based splits that create perverse incentives and tenant disputes.

6. **Missing automated reporting**: Energy managers produce weekly, monthly, and annual reports manually, consuming 10-20 hours per month. Automated report generation with configurable templates, scheduled distribution, and exception-based alerting can reduce reporting effort by 80% while improving data freshness and consistency.

7. **Insufficient alarm management**: Most BMS alarm systems generate excessive false alarms (50-200 per day), leading to alarm fatigue and missed genuine faults. Intelligent alarm management with suppression rules, correlation analysis, priority classification, and escalation protocols can reduce actionable alarms by 80% while ensuring critical events receive immediate attention.

8. **No energy budgeting**: Organizations rarely set energy budgets by department, building, or process. Without budget-vs-actual tracking with variance analysis, there is no accountability for energy consumption and no mechanism to drive behavioral change.

### 1.2 Solution Overview

PACK-039 is the **Energy Monitoring Pack** -- the ninth pack in the "Energy Efficiency Packs" category. It provides a comprehensive energy monitoring and targeting (M&T) platform covering real-time data acquisition from multiple metering sources, automated data validation and gap-filling, statistical anomaly detection, ISO 50001 EnPI tracking with regression normalization, multi-tenant cost allocation, energy budgeting with variance analysis, automated report generation, and intelligent alarm management.

The pack includes 10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets covering the complete energy monitoring lifecycle from meter installation through continuous performance tracking.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / BMS Approach | PACK-039 Energy Monitoring Pack |
|-----------|----------------------|--------------------------------|
| Data sources | Single BMS/utility | Unified multi-source (BMS, AMI, SCADA, IoT, manual) |
| Coverage | Whole-building only | System-level submetering (80%+ coverage) |
| Anomaly detection | Monthly bill review | Real-time statistical process control (CUSUM, EWMA) |
| EnPI tracking | Raw kWh only | Regression-normalized EnPIs per ISO 50001 |
| Cost allocation | Area-based split | Interval-level tariff-aware allocation |
| Reporting | Manual (10-20 hrs/month) | Automated with configurable templates |
| Alarm management | Raw BMS alarms (50-200/day) | Intelligent (suppression, correlation, escalation) |
| Budgeting | No energy budgets | Budget-vs-actual with variance analysis |
| Audit trail | None | SHA-256 provenance, full calculation lineage |

### 1.4 Regulatory and Standards Framework

| Standard/Regulation | Relevance |
|---------------------|-----------|
| ISO 50001:2018 | Energy management system - EnPI, energy baseline, measurement plan |
| ISO 50006:2014 | Measuring energy performance using EnPIs and EnBs |
| ISO 50015:2014 | Measurement and verification of energy performance |
| IPMVP Core Concepts 2022 | M&V methodology (Option C: whole facility regression) |
| ASHRAE Guideline 14-2014 | Measurement of energy, demand, and water savings |
| EU Energy Efficiency Directive 2023/1791 | Mandatory energy audits and monitoring |
| EU Energy Performance of Buildings Directive 2024/1275 | Building energy monitoring requirements |
| CSRD/ESRS E1 | Climate disclosure - energy consumption reporting |
| GHG Protocol | Energy data for Scope 1/2 emissions calculation |
| IEC 61968/61970 | CIM for energy data exchange |

---

## 2. Technical Architecture

### 2.1 Pack Structure

```
packs/energy-efficiency/PACK-039-energy-monitoring/
├── __init__.py
├── pack.yaml
├── engines/
│   ├── __init__.py
│   ├── meter_registry_engine.py        # Engine 1: Meter asset registry
│   ├── data_acquisition_engine.py      # Engine 2: Multi-source data acquisition
│   ├── data_validation_engine.py       # Engine 3: Automated data validation
│   ├── anomaly_detection_engine.py     # Engine 4: Statistical anomaly detection
│   ├── enpi_engine.py                  # Engine 5: EnPI calculation & tracking
│   ├── cost_allocation_engine.py       # Engine 6: Multi-tenant cost allocation
│   ├── budget_engine.py               # Engine 7: Energy budgeting & variance
│   ├── alarm_engine.py                # Engine 8: Intelligent alarm management
│   ├── dashboard_engine.py            # Engine 9: Real-time dashboards
│   └── monitoring_reporting_engine.py  # Engine 10: Automated report generation
├── workflows/
│   ├── __init__.py
│   ├── meter_setup_workflow.py         # Workflow 1: Meter registration & config
│   ├── data_collection_workflow.py     # Workflow 2: Automated data collection
│   ├── anomaly_response_workflow.py    # Workflow 3: Anomaly investigation
│   ├── enpi_tracking_workflow.py       # Workflow 4: EnPI monitoring cycle
│   ├── cost_allocation_workflow.py     # Workflow 5: Period-end cost allocation
│   ├── budget_review_workflow.py       # Workflow 6: Budget variance review
│   ├── reporting_workflow.py           # Workflow 7: Scheduled report generation
│   └── full_monitoring_workflow.py     # Workflow 8: Full M&T lifecycle
├── templates/
│   ├── __init__.py
│   ├── meter_inventory_report.py       # Template 1: Meter registry report
│   ├── energy_consumption_report.py    # Template 2: Consumption dashboard
│   ├── anomaly_report.py              # Template 3: Anomaly investigation
│   ├── enpi_performance_report.py     # Template 4: EnPI tracking report
│   ├── cost_allocation_report.py      # Template 5: Cost allocation report
│   ├── budget_variance_report.py      # Template 6: Budget vs actual
│   ├── alarm_summary_report.py        # Template 7: Alarm management report
│   ├── utility_bill_report.py         # Template 8: Bill validation report
│   ├── executive_summary_report.py    # Template 9: Executive summary
│   └── iso50001_compliance_report.py  # Template 10: ISO 50001 compliance
├── integrations/
│   ├── __init__.py
│   ├── pack_orchestrator.py            # Integration 1: Pipeline orchestrator
│   ├── mrv_bridge.py                   # Integration 2: MRV emissions bridge
│   ├── data_bridge.py                  # Integration 3: DATA agent bridge
│   ├── meter_protocol_bridge.py        # Integration 4: Modbus/BACnet/MQTT/OPC-UA
│   ├── ami_bridge.py                   # Integration 5: Smart meter AMI data
│   ├── bms_bridge.py                   # Integration 6: BMS trend data
│   ├── iot_sensor_bridge.py            # Integration 7: IoT sensor platforms
│   ├── pack036_bridge.py              # Integration 8: PACK-036 Utility Analysis
│   ├── pack038_bridge.py              # Integration 9: PACK-038 Peak Shaving
│   ├── health_check.py                 # Integration 10: System health verification
│   ├── setup_wizard.py                 # Integration 11: Configuration wizard
│   └── alert_bridge.py                 # Integration 12: Alert notifications
├── config/
│   ├── __init__.py
│   └── presets/
│       ├── __init__.py
│       ├── commercial_office.yaml
│       ├── manufacturing.yaml
│       ├── retail_chain.yaml
│       ├── hospital.yaml
│       ├── university_campus.yaml
│       ├── data_center.yaml
│       ├── multi_tenant.yaml
│       └── industrial_process.yaml
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_meter_registry_engine.py
    ├── test_data_acquisition_engine.py
    ├── test_data_validation_engine.py
    ├── test_anomaly_detection_engine.py
    ├── test_enpi_engine.py
    ├── test_cost_allocation_engine.py
    ├── test_budget_engine.py
    ├── test_alarm_engine.py
    ├── test_dashboard_engine.py
    ├── test_monitoring_reporting_engine.py
    ├── test_workflows.py
    └── test_integrations.py
```

### 2.2 Agent Dependencies

| Agent Layer | Agents Used | Purpose |
|-------------|-------------|---------|
| MRV | MRV-001 (Stationary Combustion), MRV-009 (Location-Based), MRV-010 (Market-Based) | Scope 1/2 from metered energy data |
| DATA | DATA-001 (PDF), DATA-002 (Excel/CSV), DATA-003 (ERP), DATA-010 (Quality Profiler), DATA-014 (Gap Filler), DATA-016 (Freshness Monitor) | Data intake, quality, gap-filling |
| FOUND | FOUND-001 to FOUND-010 | Pipeline foundation |
| Packs | PACK-036 (Utility Analysis), PACK-038 (Peak Shaving) | Rate structures, demand analysis |

---

## 3. Engine Specifications

### 3.1 Engine 1: Meter Registry Engine (`meter_registry_engine.py`)
Meter asset management covering registration, hierarchy (site > building > floor > system > circuit), calibration tracking, CT/PT ratio management, virtual meter calculation, and data channel configuration.
- Enums: `MeterType` (REVENUE/CHECK/SUBMETER/VIRTUAL/TEMPORARY), `MeterProtocol` (MODBUS_RTU/MODBUS_TCP/BACNET_IP/BACNET_MSTP/MQTT/OPCUA/PULSE/MANUAL), `EnergyType` (ELECTRICITY/NATURAL_GAS/STEAM/CHILLED_WATER/HOT_WATER/FUEL_OIL/PROPANE/DISTRICT_HEAT), `MeterStatus` (ACTIVE/INACTIVE/CALIBRATION_DUE/FAULT/DECOMMISSIONED), `ChannelType` (KW/KWH/KVAR/KVARH/KVA/VOLTAGE/CURRENT/PF/THERM/M3/FLOW/TEMPERATURE/PRESSURE)

### 3.2 Engine 2: Data Acquisition Engine (`data_acquisition_engine.py`)
Multi-source data collection with protocol abstraction, polling schedules, buffer management, timestamp alignment, and data normalization to common units.
- Key calculations: Pulse-to-engineering conversion (pulse count x multiplier), CT/PT ratio application, cumulative-to-interval conversion (delta calculation), timestamp alignment to standard intervals, unit normalization

### 3.3 Engine 3: Data Validation Engine (`data_validation_engine.py`)
Automated data quality validation with 12 check types, gap detection, spike filtering, stuck-value detection, and quality scoring per ASHRAE Guideline 14.
- Checks: Range (min/max), spike (rate-of-change), stuck value (zero variance), gap detection, meter rollover, negative consumption, sum check (submeter vs main), phase balance, power factor range, timestamp continuity, duplicate detection, completeness scoring

### 3.4 Engine 4: Anomaly Detection Engine (`anomaly_detection_engine.py`)
Statistical anomaly detection using CUSUM, EWMA, Z-score, IQR, regression residual, and schedule-based methods.
- Key calculations: CUSUM (Cumulative Sum control chart), EWMA (Exponentially Weighted Moving Average), Modified Z-score (median absolute deviation), Tukey IQR outlier detection, Regression residual analysis (actual vs predicted), Schedule comparison (occupied vs unoccupied baselines), Weather-normalized anomaly (CDD/HDD adjustment)

### 3.5 Engine 5: EnPI Engine (`enpi_engine.py`)
ISO 50001 Energy Performance Indicator calculation with regression-based normalization, significance testing, and CUSUM tracking.
- Key calculations: Simple ratio EnPI (kWh/unit), regression EnPI (multivariate OLS), energy baseline (EnB) with relevant variables (HDD, CDD, production, occupancy), CUSUM energy savings tracking, significance testing (F-test, t-test, R²), adjustment factors for non-routine changes, rolling baseline updates, uncertainty quantification per ISO 50015

### 3.6 Engine 6: Cost Allocation Engine (`cost_allocation_engine.py`)
Interval-level energy cost allocation with tariff-aware calculation, tenant billing, department charging, and product costing.
- Key calculations: Interval cost = interval_kWh x blended_rate + interval_kW_contribution x demand_charge_share, Tenant allocation by metered consumption, Virtual meter allocation (parent minus sum of children), Tax and surcharge allocation, Common area allocation methods (area-weighted, headcount, fixed split), Reconciliation to utility bill total

### 3.7 Engine 7: Budget Engine (`budget_engine.py`)
Energy budget creation, tracking, and variance analysis with weather normalization and rolling forecasts.
- Key calculations: Budget creation (baseline year x target reduction), Weather-normalized budget (degree-day adjustment), Variance analysis (volume, rate, weather, efficiency components), Cumulative variance tracking, Rolling forecast (trend + seasonality), Budget alert thresholds (warning at 90%, critical at 100%)

### 3.8 Engine 8: Alarm Engine (`alarm_engine.py`)
Intelligent alarm management with suppression, correlation, priority classification, and escalation.
- Key calculations: Alarm priority scoring (severity x impact x frequency), Alarm suppression rules (shelter, delay, deadband, time-window), Correlation analysis (temporal proximity, causal chain), False alarm rate tracking, Mean-time-to-acknowledge (MTTA), Mean-time-to-resolve (MTTR), Alarm rationalization scoring (ISA 18.2 lifecycle)

### 3.9 Engine 9: Dashboard Engine (`dashboard_engine.py`)
Real-time dashboard generation with KPIs, widgets, trend visualization, heatmaps, and Sankey diagrams.
- Panels: Energy consumption (real-time + trend), Cost tracking (budget vs actual), EnPI performance (target vs actual with CUSUM), Anomaly status (active alerts map), Submeter breakdown (treemap/Sankey), Weather correlation (scatter + regression), Load profile (24-hour overlay), Alarm summary (priority distribution)

### 3.10 Engine 10: Monitoring Reporting Engine (`monitoring_reporting_engine.py`)
Automated report generation with configurable schedules, exception-based triggers, and multi-format output.
- Reports: Daily energy summary, Weekly performance review, Monthly utility bill analysis, Quarterly EnPI report, Annual energy review, ISO 50001 management review, Exception report (anomaly triggered), Custom ad-hoc reports

---

## 4. Database Migrations (V306-V315)

| Migration | Description | Key Tables |
|-----------|-------------|------------|
| V306 | Meter registry and hierarchy | `em_meters`, `em_meter_channels`, `em_meter_hierarchy`, `em_calibration_records`, `em_virtual_meters` |
| V307 | Data acquisition and storage | `em_interval_data`, `em_acquisition_schedules`, `em_data_buffers`, `em_protocol_configs` |
| V308 | Data validation and quality | `em_validation_rules`, `em_validation_results`, `em_quality_scores`, `em_data_corrections` |
| V309 | Anomaly detection | `em_anomalies`, `em_anomaly_rules`, `em_investigation_records`, `em_baselines` |
| V310 | EnPI tracking | `em_enpi_definitions`, `em_enpi_values`, `em_energy_baselines`, `em_cusum_tracking`, `em_regression_models` |
| V311 | Cost allocation | `em_cost_allocations`, `em_tenant_accounts`, `em_allocation_rules`, `em_billing_periods` |
| V312 | Energy budgeting | `em_budgets`, `em_budget_periods`, `em_variance_records`, `em_forecasts` |
| V313 | Alarm management | `em_alarms`, `em_alarm_rules`, `em_alarm_acknowledgments`, `em_escalation_configs` |
| V314 | Dashboards and reports | `em_dashboard_configs`, `em_report_schedules`, `em_report_outputs`, `em_kpi_definitions` |
| V315 | Views, indexes, audit trail, seed data | Materialized views, audit trail, seeded meter types, alarm templates |

---

## 5. Testing Strategy

- **Unit tests:** 850+ test functions across 12 test files
- **Parametrize expansions:** 400+ for multi-scenario coverage
- **Total expanded tests:** 1,250+ test cases
- **Coverage target:** 85%+ line coverage

---

## 6. Glossary

| Term | Definition |
|------|-----------|
| AMI | Advanced Metering Infrastructure |
| BACnet | Building Automation and Control Network protocol |
| CUSUM | Cumulative Sum control chart for detecting persistent shifts |
| EnB | Energy Baseline per ISO 50001 |
| EnPI | Energy Performance Indicator per ISO 50001 |
| EWMA | Exponentially Weighted Moving Average |
| HDD/CDD | Heating/Cooling Degree Days |
| M&T | Monitoring and Targeting |
| Modbus | Serial communication protocol for industrial devices |
| MQTT | Message Queuing Telemetry Transport (IoT protocol) |
| OPC-UA | Open Platform Communications Unified Architecture |
| SPC | Statistical Process Control |
| Virtual Meter | Calculated meter derived from physical meter readings |
