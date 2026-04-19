# GreenLang Climate OS
## Complete Agent Specifications (402 Canonical Agents)

**Version:** 1.0
**Date:** January 26, 2026
**Source:** GreenLang Agent Catalog v3

---

## Overview

This document provides complete specifications for all 402 canonical agents in the GreenLang Climate OS. These agents are organized into 11 layers and can scale to 100,000+ deployable variants through the Agent Family & Variant System.

---

# Layer 1: Foundation & Governance (10 Agents)

## GL-FOUND-X-001: GreenLang Orchestrator

### Basic Information
| Attribute | Value |
|-----------|-------|
| **Agent ID** | GL-FOUND-X-001 |
| **Layer** | Foundation & Governance |
| **Sector** | Cross-cutting |
| **Domain** | Platform runtime / pipeline orchestration |
| **Maturity Target** | MVP |
| **Family** | OrchestrationFamily |
| **Est. Variants** | 10,000 |

### Functionality
**Purpose:** Plans and executes multi-agent pipelines; manages dependency graph, retries, timeouts, and handoffs; enforces deterministic run metadata for auditability.

**Primary Users:** Platform engineers, solution architects

### Inputs & Outputs
| Direction | Data |
|-----------|------|
| **Key Inputs** | Pipeline YAML, agent registry, run configuration, credentials/permissions |
| **Key Outputs** | Execution plan, run logs, step-level artifacts, status and lineage |

### Technical Details
| Attribute | Value |
|-----------|-------|
| **Methods/Tools** | DAG orchestration, policy checks, observability hooks |
| **Dependencies** | OPS+DATA agents, audit trail |
| **Frequency** | Per run |

### Variant Dimensions
| Dimension | Applicable |
|-----------|------------|
| Facility_Count | Yes |

---

## GL-FOUND-X-002: Schema Compiler & Validator

### Basic Information
| Attribute | Value |
|-----------|-------|
| **Agent ID** | GL-FOUND-X-002 |
| **Layer** | Foundation & Governance |
| **Sector** | Cross-cutting |
| **Domain** | Schemas / data contracts |
| **Maturity Target** | MVP |
| **Family** | SchemaFamily |
| **Est. Variants** | 1,500 |

### Functionality
**Purpose:** Validates input payloads against GreenLang schemas; pinpoints missing fields, unit inconsistencies, and invalid ranges; emits machine-fixable error hints.

**Primary Users:** Developers, data engineers

### Inputs & Outputs
| Direction | Data |
|-----------|------|
| **Key Inputs** | YAML/JSON inputs, schema version, validation rules |
| **Key Outputs** | Validation report, normalized payload, fix suggestions |

### Technical Details
| Attribute | Value |
|-----------|-------|
| **Methods/Tools** | Schema validation, rule engines, linting |
| **Frequency** | Per run |

### Variant Dimensions
| Dimension | Applicable |
|-----------|------------|
| Asset_Type | Yes |
| Industry_Subsector | Yes |

---

## GL-FOUND-X-003: Unit & Reference Normalizer

### Basic Information
| Attribute | Value |
|-----------|-------|
| **Agent ID** | GL-FOUND-X-003 |
| **Layer** | Foundation & Governance |
| **Sector** | Cross-cutting |
| **Domain** | Units / conversions / reference data |
| **Maturity Target** | MVP |
| **Family** | NormalizationFamily |
| **Est. Variants** | 1,800 |

### Functionality
**Purpose:** Normalizes units, converts to canonical units, and standardizes naming for fuels, processes, and materials; maintains consistent reference IDs.

**Primary Users:** Developers, analysts

### Inputs & Outputs
| Direction | Data |
|-----------|------|
| **Key Inputs** | Raw measurements, unit metadata, reference tables |
| **Key Outputs** | Canonical measurements, conversion audit log |

### Technical Details
| Attribute | Value |
|-----------|-------|
| **Methods/Tools** | Unit conversion, entity resolution, controlled vocabularies |
| **Dependencies** | Schema Validator |
| **Frequency** | Per run |

### Variant Dimensions
| Dimension | Applicable |
|-----------|------------|
| Fuel_Type | Yes |
| Asset_Type | Yes |

---

## GL-FOUND-X-004: Assumptions Registry Agent

### Basic Information
| Attribute | Value |
|-----------|-------|
| **Agent ID** | GL-FOUND-X-004 |
| **Layer** | Foundation & Governance |
| **Sector** | Cross-cutting |
| **Domain** | Assumptions governance |
| **Maturity Target** | v1 |
| **Family** | GovernanceFamily |
| **Est. Variants** | 3,000 |

### Functionality
**Purpose:** Stores, versions, and retrieves assumptions (emission factors, efficiencies, load factors, baselines); forces explicit assumption selection and change logging.

**Primary Users:** Sustainability leads, auditors

### Inputs & Outputs
| Direction | Data |
|-----------|------|
| **Key Inputs** | Assumption catalog, scenario settings, jurisdiction |
| **Key Outputs** | Assumption set manifest, diff reports, reproducibility bundle |

### Technical Details
| Attribute | Value |
|-----------|-------|
| **Methods/Tools** | Version control patterns, config management |
| **Dependencies** | Emission Factor Library |
| **Frequency** | Per scenario/run |

### Variant Dimensions
| Dimension | Applicable |
|-----------|------------|
| Geography | Yes |
| Reporting_Standard | Yes |

---

## GL-FOUND-X-005: Citations & Evidence Agent

### Basic Information
| Attribute | Value |
|-----------|-------|
| **Agent ID** | GL-FOUND-X-005 |
| **Layer** | Foundation & Governance |
| **Sector** | Cross-cutting |
| **Domain** | Evidence packaging |
| **Maturity Target** | v1 |
| **Family** | EvidenceFamily |
| **Est. Variants** | 12 |

### Functionality
**Purpose:** Attaches sources, evidence files, and calculation notes to outputs; creates an evidence map tying every KPI to inputs and rules.

**Primary Users:** Sustainability teams, auditors, partners

### Inputs & Outputs
| Direction | Data |
|-----------|------|
| **Key Inputs** | Calculation artifacts, source documents, external citations |
| **Key Outputs** | Evidence bundle, citation index, provenance graph |

### Technical Details
| Attribute | Value |
|-----------|-------|
| **Methods/Tools** | Document linking, hashing, metadata injection |
| **Frequency** | Per run |

### Variant Dimensions
| Dimension | Applicable |
|-----------|------------|
| Reporting_Standard | Yes |

---

## GL-FOUND-X-006: Access & Policy Guard Agent

### Basic Information
| Attribute | Value |
|-----------|-------|
| **Agent ID** | GL-FOUND-X-006 |
| **Layer** | Foundation & Governance |
| **Sector** | Cross-cutting |
| **Domain** | Security and policy enforcement |
| **Maturity Target** | v1 |
| **Family** | PolicyGuardFamily |
| **Est. Variants** | 200 |

### Functionality
**Purpose:** Enforces tooling policy, PII minimization, role-based permissions, and data-residency rules at runtime; blocks non-compliant calls.

**Primary Users:** Security teams, compliance officers

### Inputs & Outputs
| Direction | Data |
|-----------|------|
| **Key Inputs** | User context, request payload, policy ruleset |
| **Key Outputs** | Allow/deny decision, audit event, redaction manifest |

### Technical Details
| Attribute | Value |
|-----------|-------|
| **Methods/Tools** | OPA policies, attribute-based access control |
| **Dependencies** | Agent Registry |
| **Frequency** | Per request |

### Variant Dimensions
| Dimension | Applicable |
|-----------|------------|
| Geography | Yes |

---

## GL-FOUND-X-007: Versioned Agent Registry Agent

### Basic Information
| Attribute | Value |
|-----------|-------|
| **Agent ID** | GL-FOUND-X-007 |
| **Layer** | Foundation & Governance |
| **Sector** | Cross-cutting |
| **Domain** | Agent lifecycle management |
| **Maturity Target** | MVP |
| **Family** | RegistryFamily |
| **Est. Variants** | 1 |

### Functionality
**Purpose:** Maintains catalog of all agents, versions, health status, and capabilities; supports semantic versioning and blue-green deployment.

**Primary Users:** DevOps, platform team

### Inputs & Outputs
| Direction | Data |
|-----------|------|
| **Key Inputs** | Agent definitions, version tags, deployment status |
| **Key Outputs** | Agent catalog, version graph, deprecation alerts |

### Technical Details
| Attribute | Value |
|-----------|-------|
| **Methods/Tools** | Service registry patterns, health probes |
| **Frequency** | Continuous |

---

## GL-FOUND-X-008: Run Reproducibility & Replay Agent

### Basic Information
| Attribute | Value |
|-----------|-------|
| **Agent ID** | GL-FOUND-X-008 |
| **Layer** | Foundation & Governance |
| **Sector** | Cross-cutting |
| **Domain** | Reproducibility and audit |
| **Maturity Target** | v1 |
| **Family** | ReplayFamily |
| **Est. Variants** | 1 |

### Functionality
**Purpose:** Allows any historical run to be replayed exactly; locks versions, inputs, and assumptions and verifies bit-identical outputs.

**Primary Users:** Auditors, QA teams

### Inputs & Outputs
| Direction | Data |
|-----------|------|
| **Key Inputs** | Run ID, artifact store reference |
| **Key Outputs** | Replay execution, diff report, pass/fail |

### Technical Details
| Attribute | Value |
|-----------|-------|
| **Methods/Tools** | Artifact hashing, deterministic replay |
| **Dependencies** | Orchestrator, Assumptions Registry |
| **Frequency** | On demand |

---

## GL-FOUND-X-009: QA Test Harness Agent

### Basic Information
| Attribute | Value |
|-----------|-------|
| **Agent ID** | GL-FOUND-X-009 |
| **Layer** | Foundation & Governance |
| **Sector** | Cross-cutting |
| **Domain** | Quality assurance |
| **Maturity Target** | MVP |
| **Family** | TestFamily |
| **Est. Variants** | 100 |

### Functionality
**Purpose:** Runs test suites against agent outputs and flags regressions; supports unit, integration, and boundary tests.

**Primary Users:** QA engineers, developers

### Inputs & Outputs
| Direction | Data |
|-----------|------|
| **Key Inputs** | Test definitions, expected outputs, agent under test |
| **Key Outputs** | Test report, coverage metrics, failure root cause |

### Technical Details
| Attribute | Value |
|-----------|-------|
| **Methods/Tools** | Test frameworks, snapshot testing, fuzzing |
| **Frequency** | Per deployment |

---

## GL-FOUND-X-010: Observability & Telemetry Agent

### Basic Information
| Attribute | Value |
|-----------|-------|
| **Agent ID** | GL-FOUND-X-010 |
| **Layer** | Foundation & Governance |
| **Sector** | Cross-cutting |
| **Domain** | Platform monitoring |
| **Maturity Target** | MVP |
| **Family** | ObservabilityFamily |
| **Est. Variants** | 10 |

### Functionality
**Purpose:** Collects metrics, logs, and traces for agent runs; powers dashboards and anomaly detection for platform health.

**Primary Users:** SREs, platform engineers

### Inputs & Outputs
| Direction | Data |
|-----------|------|
| **Key Inputs** | Agent telemetry, run context |
| **Key Outputs** | Metrics, dashboards, alerts |

### Technical Details
| Attribute | Value |
|-----------|-------|
| **Methods/Tools** | OpenTelemetry, Prometheus, Grafana |
| **Frequency** | Continuous |

---

# Layer 2: Data & Connectors (50 Agents)

## Data Intake Agents (12)

| Agent ID | Agent Name | Purpose | Est. Variants |
|----------|------------|---------|---------------|
| GL-DATA-X-001 | PDF & Invoice Extractor | OCR + layout analysis for utility bills, invoices | 1,500 |
| GL-DATA-X-002 | Excel/CSV Normalizer | Spreadsheet ingestion, header detection, schema mapping | 1,200 |
| GL-DATA-X-003 | ERP Finance Connector | SAP/Oracle/Workday GL, cost centers, POs | 3,000 |
| GL-DATA-X-004 | SCADA/BMS/IoT Connector | Building/industrial operational data streaming | 5,000 |
| GL-DATA-X-005 | Fleet Telematics Connector | GPS, fuel, mileage from fleet APIs | 2,000 |
| GL-DATA-X-006 | Utility Tariff & Grid Factor Agent | Grid emission factors and utility rates | 12,000 |
| GL-DATA-X-007 | Supplier Portal Scraper | Extract sustainability data from supplier portals | 1,000 |
| GL-DATA-X-008 | API Gateway Agent | Unified REST/GraphQL interface | 100 |
| GL-DATA-X-009 | Real-Time Event Processor | Streaming data with windowing | 500 |
| GL-DATA-X-010 | Document Classification Agent | Route documents by type | 200 |
| GL-DATA-X-011 | Multi-Language OCR Agent | 30+ language document processing | 900 |
| GL-DATA-X-012 | Email Attachment Processor | Extract data from email attachments | 300 |

## Data Quality Agents (10)

| Agent ID | Agent Name | Purpose | Est. Variants |
|----------|------------|---------|---------------|
| GL-DATA-X-013 | Data Quality Profiler | Completeness, consistency, timeliness | 500 |
| GL-DATA-X-014 | Duplicate Detection Agent | Identify and merge duplicates | 200 |
| GL-DATA-X-015 | Missing Value Imputer | Fill missing values with audit | 300 |
| GL-DATA-X-016 | Outlier Detection Agent | Flag statistical outliers | 200 |
| GL-DATA-X-017 | Time Series Gap Filler | Interpolate missing time series | 400 |
| GL-DATA-X-018 | Cross-Source Reconciliation | Reconcile across sources | 500 |
| GL-DATA-X-019 | Data Freshness Monitor | Track data age | 100 |
| GL-DATA-X-020 | Schema Migration Agent | Schema evolution handling | 200 |
| GL-DATA-X-021 | Data Lineage Tracker | Field-level lineage | 100 |
| GL-DATA-X-022 | Validation Rule Engine | Business rule validation | 500 |

## Sector-Specific Connectors (28)

| Agent ID | Agent Name | Sector | Est. Variants |
|----------|------------|--------|---------------|
| GL-DATA-IND-001 | Manufacturing ERP Connector | Industrial | 2,000 |
| GL-DATA-IND-002 | Process Historian Connector | Industrial | 1,500 |
| GL-DATA-IND-003 | CMMS/Maintenance Connector | Industrial | 800 |
| GL-DATA-BLD-001 | BMS Protocol Gateway | Buildings | 3,000 |
| GL-DATA-BLD-002 | Energy Star Portfolio Sync | Buildings | 500 |
| GL-DATA-BLD-003 | Smart Meter Collector | Buildings | 2,000 |
| GL-DATA-TRN-001 | Fleet Management API | Transport | 1,500 |
| GL-DATA-TRN-002 | EV Charging Network | Transport | 800 |
| GL-DATA-TRN-003 | Route Optimization Feed | Transport | 600 |
| GL-DATA-AGR-001 | Precision Ag Connector | Agriculture | 1,000 |
| GL-DATA-AGR-002 | Weather Station Collector | Agriculture | 500 |
| GL-DATA-AGR-003 | Satellite Imagery Processor | Agriculture | 300 |
| GL-DATA-FIN-001 | Bloomberg/Reuters Feed | Finance | 200 |
| GL-DATA-FIN-002 | Carbon Registry Connector | Finance | 100 |
| GL-DATA-FIN-003 | ESG Rating Connector | Finance | 150 |
| GL-DATA-UTL-001 | Grid Operator Connector | Utilities | 1,000 |
| GL-DATA-UTL-002 | REC/Certificate Registry | Utilities | 200 |
| GL-DATA-UTL-003 | Demand Response Platform | Utilities | 300 |
| GL-DATA-SUP-001 | Supplier Questionnaire | Supply Chain | 500 |
| GL-DATA-SUP-002 | Spend Data Categorizer | Supply Chain | 1,000 |
| GL-DATA-SUP-003 | CDP Supply Chain Data | Supply Chain | 200 |
| GL-DATA-REG-001 | EU CBAM Registry | Regulatory | 100 |
| GL-DATA-REG-002 | EU Taxonomy Database | Regulatory | 50 |
| GL-DATA-REG-003 | SEC EDGAR Filing | Regulatory | 100 |
| GL-DATA-REG-004 | EUDR Traceability | Regulatory | 200 |
| GL-DATA-GEO-001 | GIS/Mapping Connector | Geospatial | 500 |
| GL-DATA-GEO-002 | Climate Hazard Database | Geospatial | 300 |
| GL-DATA-GEO-003 | Deforestation Satellite | Geospatial | 200 |

---

# Layer 3: MRV / Accounting (45 Agents)

## Scope 1 Agents (8)

| Agent ID | Agent Name | Purpose | Dims | Est. Variants |
|----------|------------|---------|------|---------------|
| GL-MRV-X-001 | Stationary Combustion Agent | Fuel combustion in stationary equipment | Fuel×Asset×Standard | 21,600 |
| GL-MRV-X-002 | Refrigerants & F-Gas Agent | Refrigerant leakage emissions | Geo×Asset | 15,000 |
| GL-MRV-X-003 | Mobile Combustion Agent | Fleet/vehicle direct emissions | Vehicle×Fuel×Geo | 12,000 |
| GL-MRV-X-004 | Process Emissions Agent | Non-combustion industrial emissions | Industry×Standard | 8,000 |
| GL-MRV-X-005 | Fugitive Emissions Agent | Pipeline, tank leaks | Asset×Geo | 5,000 |
| GL-MRV-X-006 | Land Use Emissions Agent | Land use change emissions | Geo×Soil | 3,000 |
| GL-MRV-X-007 | Waste Treatment Emissions | On-site waste treatment | Asset×Geo | 2,000 |
| GL-MRV-X-008 | Agricultural Emissions Agent | Livestock, fertilizer, manure | Crop×Geo | 6,000 |

## Scope 2 Agents (5)

| Agent ID | Agent Name | Purpose | Dims | Est. Variants |
|----------|------------|---------|------|---------------|
| GL-MRV-X-020 | Scope 2 Location-Based | Grid-average factors | Utility×Standard | 12,000 |
| GL-MRV-X-021 | Scope 2 Market-Based | RECs, PPAs, residual mix | Geo×Standard | 3,000 |
| GL-MRV-X-022 | Steam/Heat Purchase Agent | Purchased steam/heat | Asset×Geo | 1,500 |
| GL-MRV-X-023 | Cooling Purchase Agent | District cooling | Asset×Geo | 1,000 |
| GL-MRV-X-024 | Dual Reporting Reconciliation | Location vs market reconciliation | Standard | 500 |

## Scope 3 Category Agents (15)

| Agent ID | Agent Name | Category | Dims | Est. Variants |
|----------|------------|----------|------|---------------|
| GL-MRV-S3-001 | Purchased Goods & Services | Cat 1 | Supplier×Standard | 5,000 |
| GL-MRV-S3-002 | Capital Goods Agent | Cat 2 | Asset×Standard | 2,000 |
| GL-MRV-S3-003 | Fuel & Energy Activities | Cat 3 | Fuel×Geo | 3,000 |
| GL-MRV-S3-004 | Upstream Transportation | Cat 4 | Vehicle×Geo | 4,000 |
| GL-MRV-S3-005 | Waste Generated Agent | Cat 5 | Asset×Geo | 2,500 |
| GL-MRV-S3-006 | Business Travel Agent | Cat 6 | Vehicle×Geo | 3,000 |
| GL-MRV-S3-007 | Employee Commuting Agent | Cat 7 | Geo×Vehicle | 2,000 |
| GL-MRV-S3-008 | Upstream Leased Assets | Cat 8 | Asset×Geo | 1,000 |
| GL-MRV-S3-009 | Downstream Transportation | Cat 9 | Vehicle×Geo | 2,500 |
| GL-MRV-S3-010 | Processing of Sold Products | Cat 10 | Industry×Geo | 1,500 |
| GL-MRV-S3-011 | Use of Sold Products Agent | Cat 11 | Asset×Geo×Scenario | 3,000 |
| GL-MRV-S3-012 | End-of-Life Treatment | Cat 12 | Asset×Geo | 2,000 |
| GL-MRV-S3-013 | Downstream Leased Assets | Cat 13 | Asset×Geo | 800 |
| GL-MRV-S3-014 | Franchises Agent | Cat 14 | Industry×Geo | 500 |
| GL-MRV-S3-015 | Investments Agent | Cat 15 | Industry×Geo | 1,000 |

## Cross-Cutting MRV Agents (17)

| Agent ID | Agent Name | Purpose | Est. Variants |
|----------|------------|---------|---------------|
| GL-MRV-X-040 | Scope 3 Category Mapper | Map spend/PO/BOM to categories | 720 |
| GL-MRV-X-041 | Uncertainty & Data Quality | Quantify uncertainty ranges | 12 |
| GL-MRV-X-042 | Audit Trail & Lineage | Immutable lineage creation | 12 |
| GL-MRV-X-043 | Emission Factor Selector | Select appropriate factors | 3,000 |
| GL-MRV-X-044 | Activity Data Validator | Validate activity data | 500 |
| GL-MRV-X-045 | Method Selection Advisor | Recommend calculation methods | 100 |
| GL-MRV-X-046 | Consolidation & Rollup | Aggregate across entities | 500 |
| GL-MRV-X-047 | Base Year Recalculation | Handle structural changes | 100 |
| GL-MRV-X-048 | Biogenic Carbon Tracker | Track biogenic CO2 | 200 |
| GL-MRV-X-049 | Carbon Removal Accounting | Account removals/offsets | 100 |
| GL-MRV-X-050 | Market Instrument Tracker | Track RECs, credits | 200 |
| GL-MRV-X-051 | GWP Conversion Agent | Apply GWP factors | 50 |
| GL-MRV-X-052 | Variance Analysis Agent | Year-over-year variance | 100 |
| GL-MRV-X-053 | Intensity Metric Calculator | Per-revenue, per-product | 500 |
| GL-MRV-X-054 | Benchmark Comparison Agent | Industry benchmarks | 500 |
| GL-MRV-X-055 | Data Gap Estimator | Estimate missing data | 300 |
| GL-MRV-X-056 | Multi-Standard Reporter | GHG/ISO/CSRD outputs | 36 |

---

# Layer 4: Decarbonization Planning (55 Agents)

## Abatement Analysis Agents (15)

| Agent ID | Agent Name | Purpose | Est. Variants |
|----------|------------|---------|---------------|
| GL-PLAN-X-001 | Abatement Option Generator | Generate reduction options | 5,000 |
| GL-PLAN-X-002 | MACC Curve Builder | Marginal abatement cost curves | 2,000 |
| GL-PLAN-X-003 | Technology Readiness Assessor | Evaluate TRL and maturity | 1,000 |
| GL-PLAN-X-004 | Electrification Feasibility | Assess electrification potential | 3,000 |
| GL-PLAN-X-005 | Fuel Switching Analyzer | Evaluate alternative fuels | 2,000 |
| GL-PLAN-X-006 | Energy Efficiency Identifier | Find efficiency opportunities | 5,000 |
| GL-PLAN-X-007 | Process Optimization Agent | Identify process improvements | 3,000 |
| GL-PLAN-X-008 | Renewable Energy Planner | Plan RE procurement | 2,000 |
| GL-PLAN-X-009 | Carbon Capture Evaluator | Assess CCS/CCU potential | 500 |
| GL-PLAN-X-010 | Nature-Based Solutions | Plan NBS interventions | 1,000 |
| GL-PLAN-X-011 | Circular Economy Agent | Circular opportunities | 1,500 |
| GL-PLAN-X-012 | Supply Chain Decarbonizer | Supplier engagement | 2,000 |
| GL-PLAN-X-013 | Behavior Change Planner | Employee/consumer programs | 500 |
| GL-PLAN-X-014 | Offset Strategy Agent | Develop offset portfolio | 500 |
| GL-PLAN-X-015 | Innovation Pipeline Agent | Emerging technologies | 300 |

## Roadmap & Target Agents (12)

| Agent ID | Agent Name | Purpose | Est. Variants |
|----------|------------|---------|---------------|
| GL-PLAN-X-020 | Science-Based Target Calculator | SBTi-aligned targets | 1,000 |
| GL-PLAN-X-021 | Pathway Scenario Modeler | Decarbonization pathways | 500 |
| GL-PLAN-X-022 | Interim Target Designer | Set interim milestones | 300 |
| GL-PLAN-X-023 | Budget Allocation Agent | Carbon budget allocation | 200 |
| GL-PLAN-X-024 | Phasing & Sequencing Agent | Implementation sequence | 500 |
| GL-PLAN-X-025 | Dependency Mapper | Project dependencies | 200 |
| GL-PLAN-X-026 | Risk-Adjusted Planner | Implementation risk adjustment | 300 |
| GL-PLAN-X-027 | Resource Requirement Agent | Human/capital resources | 500 |
| GL-PLAN-X-028 | Stakeholder Impact Analyzer | Stakeholder impacts | 200 |
| GL-PLAN-X-029 | Change Management Planner | Organizational change | 100 |
| GL-PLAN-X-030 | Progress Tracking Agent | Track vs. roadmap | 100 |
| GL-PLAN-X-031 | Adaptive Replanning Agent | Adjust based on actuals | 100 |

## Sector-Specific Planning Agents (28)

| Agent ID | Agent Name | Sector | Focus | Est. Variants |
|----------|------------|--------|-------|---------------|
| GL-PLAN-IND-001 | Industrial Heat Decarb | Industrial | Process heat | 5,000 |
| GL-PLAN-IND-002 | Boiler Replacement Advisor | Industrial | Boiler upgrades | 3,000 |
| GL-PLAN-IND-003 | Heat Recovery Optimizer | Industrial | Waste heat | 2,000 |
| GL-PLAN-IND-004 | Compressed Air Optimizer | Industrial | Compressed air | 1,500 |
| GL-PLAN-IND-005 | Motor & Drive Optimizer | Industrial | Motors/VFDs | 2,000 |
| GL-PLAN-BLD-001 | Building Retrofit Planner | Buildings | Deep retrofits | 5,000 |
| GL-PLAN-BLD-002 | HVAC Upgrade Advisor | Buildings | HVAC systems | 3,000 |
| GL-PLAN-BLD-003 | Envelope Optimization | Buildings | Insulation | 2,000 |
| GL-PLAN-BLD-004 | Lighting Upgrade Planner | Buildings | LED conversion | 1,500 |
| GL-PLAN-BLD-005 | Building Electrification | Buildings | Heat pump transition | 2,500 |
| GL-PLAN-TRN-001 | Fleet Electrification | Transport | EV transition | 3,000 |
| GL-PLAN-TRN-002 | Charging Infrastructure | Transport | Depot charging | 2,000 |
| GL-PLAN-TRN-003 | Route Optimization | Transport | Efficiency routing | 1,500 |
| GL-PLAN-TRN-004 | Alternative Fuel Planner | Transport | Hydrogen, biofuels | 1,000 |
| GL-PLAN-TRN-005 | Modal Shift Analyzer | Transport | Rail/water shift | 500 |
| GL-PLAN-AGR-001 | Regenerative Ag Planner | Agriculture | Soil carbon | 1,500 |
| GL-PLAN-AGR-002 | Livestock Emissions | Agriculture | Enteric, manure | 1,000 |
| GL-PLAN-AGR-003 | Precision Fertilizer | Agriculture | N2O reduction | 800 |
| GL-PLAN-AGR-004 | Agroforestry Planner | Agriculture | Tree integration | 500 |
| GL-PLAN-ENR-001 | On-Site Renewables | Energy | Solar, wind | 2,000 |
| GL-PLAN-ENR-002 | PPA/VPPA Structurer | Energy | RE procurement | 500 |
| GL-PLAN-ENR-003 | Battery Storage Optimizer | Energy | Energy storage | 1,000 |
| GL-PLAN-ENR-004 | Microgrid Designer | Energy | Resilient power | 500 |
| GL-PLAN-SUP-001 | Supplier Engagement | Supply Chain | Scope 3 suppliers | 2,000 |
| GL-PLAN-SUP-002 | Product Redesign Agent | Supply Chain | Low-carbon design | 1,500 |
| GL-PLAN-SUP-003 | Logistics Optimization | Supply Chain | Transport emissions | 1,000 |
| GL-PLAN-WAT-001 | Water Efficiency Planner | Water | Water-energy nexus | 500 |
| GL-PLAN-WST-001 | Waste Reduction Planner | Waste | Zero waste | 500 |

---

# Layers 5-11: Summary Tables

## Layer 5: Climate Risk & Adaptation (40 Agents)

| Category | Agents | Purpose | Est. Variants |
|----------|--------|---------|---------------|
| Physical Risk | 15 | Flood, heat, wildfire, drought, etc. | 15,000 |
| Transition Risk | 10 | Policy, market, technology | 5,000 |
| Adaptation | 15 | Resilience planning | 10,000 |

## Layer 6: Finance & Investment (45 Agents)

| Category | Agents | Purpose | Est. Variants |
|----------|--------|---------|---------------|
| Project Finance | 12 | TCO, NPV, payback | 10,000 |
| Carbon Markets | 10 | ETS, VCM, credits | 5,000 |
| Green Finance | 13 | Bonds, loans, taxonomy | 10,000 |
| Portfolio Finance | 10 | Enterprise finance | 15,000 |

## Layer 7: Procurement & Delivery (35 Agents)

| Category | Agents | Purpose | Est. Variants |
|----------|--------|---------|---------------|
| Procurement Planning | 10 | RFP, specs, vendors | 8,000 |
| Project Delivery | 12 | Schedule, budget, QA | 5,000 |
| Supplier Management | 13 | Onboarding, performance | 7,000 |

## Layer 8: Policy & Standards (30 Agents)

| Category | Agents | Purpose | Est. Variants |
|----------|--------|---------|---------------|
| Regulatory Mapping | 15 | CSRD, CBAM, SEC, etc. | 5,000 |
| Standards Intelligence | 15 | Gap analysis, deadlines | 10,000 |

## Layer 9: Reporting & Assurance (45 Agents)

| Category | Agents | Purpose | Est. Variants |
|----------|--------|---------|---------------|
| Disclosure Generation | 15 | XBRL, PDF, dashboards | 10,000 |
| Audit & Assurance | 15 | Evidence, verification | 5,000 |
| Stakeholder Comms | 15 | Investor, employee, media | 10,000 |

## Layer 10: Operations & Monitoring (30 Agents)

| Category | Agents | Purpose | Est. Variants |
|----------|--------|---------|---------------|
| Real-Time Optimization | 15 | Energy, demand response | 10,000 |
| Performance Tracking | 15 | KPIs, alerts, trends | 10,000 |

## Layer 11: Developer Tools (17 Agents)

| Category | Agents | Purpose | Est. Variants |
|----------|--------|---------|---------------|
| SDK & Development | 9 | CLI, Python, TypeScript | 500 |
| Deployment & Ops | 8 | Docker, K8s, CI/CD | 500 |

---

# Grand Total

| Layer | Agents | Est. Variants |
|-------|--------|---------------|
| Foundation & Governance | 10 | 15,623 |
| Data & Connectors | 50 | 25,450 |
| MRV / Accounting | 45 | 119,918 |
| Decarbonization Planning | 55 | 79,800 |
| Climate Risk & Adaptation | 40 | 30,000 |
| Finance & Investment | 45 | 40,000 |
| Procurement & Delivery | 35 | 20,000 |
| Policy & Standards | 30 | 15,000 |
| Reporting & Assurance | 45 | 25,000 |
| Operations & Monitoring | 30 | 20,000 |
| Developer Tools | 17 | 1,000 |
| **TOTAL** | **402** | **~392,000** |

---

*End of Agent Specifications Document*
