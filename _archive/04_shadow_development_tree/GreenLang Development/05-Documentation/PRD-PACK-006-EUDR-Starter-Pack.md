# PRD-PACK-006: EUDR Starter Pack

**Document ID**: PRD-PACK-006
**Version**: 1.0
**Status**: DRAFT
**Author**: GreenLang Product Team
**Created**: 2026-03-14
**Last Updated**: 2026-03-14
**Approval**: Pending

---

## 1. Executive Summary

PACK-006 EUDR Starter Pack is the first-tier compliance solution for the EU Deforestation Regulation (EUDR, Regulation (EU) 2023/1115). It provides everything a first-time EUDR operator needs to achieve compliance: supplier onboarding, geolocation validation, risk assessment, Due Diligence Statement (DDS) generation, and submission to the EU Information System.

The pack leverages 18 of the 40 existing EUDR agents, the GL-EUDR-APP v1.0 platform, and 16 shared foundation/data agents for a total of **34 agents**. All components are already built and production-ready; PACK-006 provides the orchestration, configuration, workflows, templates, and integration wiring to bundle them into a deployable compliance solution.

**Regulation**: EU Deforestation Regulation (EU) 2023/1115
**Commodities**: All 7 (cattle, cocoa, coffee, oil palm, rubber, soya, wood) + derived products per Annex I
**Target**: First-time EUDR operators, SMEs, mid-market importers
**Agents**: 34 (18 EUDR-specific + 3 data intake + 3 data quality + 10 foundation)

---

## 2. Background & Motivation

### 2.1 Regulatory Context

The EUDR entered into force on June 29, 2023, prohibiting the placing on the EU market of products linked to deforestation or forest degradation after December 31, 2020 (the cutoff date). Key obligations:

1. **Due Diligence System** (Article 8): Three-phase process — information gathering, risk assessment, risk mitigation
2. **Due Diligence Statement** (Article 4): Mandatory DDS before placing goods on EU market
3. **Information Requirements** (Article 9): Geolocation (coordinates/polygons), product description, quantity, supplier, country of production
4. **Risk Assessment** (Articles 10-11): Country benchmarking, supplier verification, satellite cross-checking
5. **Risk Mitigation** (Article 12): Measures to reduce non-negligible risk to negligible
6. **EU Information System** (Article 33): Electronic submission of DDS with reference numbers
7. **7 Commodities**: Cattle, cocoa, coffee, oil palm, rubber, soya, wood + derived products

### 2.2 Tiering Strategy

| Tier | Pack | Agents | Target Users | Key Differentiators |
|------|------|--------|-------------|---------------------|
| **Starter** | **PACK-006** | **34** | First-time operators, SMEs | Basic DD, DDS generation, country risk, Tier 1-2 suppliers |
| Professional | Future | ~55 | Mid-market, complex supply chains | Satellite monitoring, protected areas, indigenous rights, audit management |
| Enterprise | Future | ~65+ | Multinationals, multi-tenant | Blockchain, continuous monitoring, customs integration, grievance mechanisms |

### 2.3 Target Users

1. **First-time EUDR operators** needing guided compliance setup
2. **SMEs** (<250 employees) qualifying for simplified due diligence (Article 13)
3. **Mid-market importers** (250-500 employees) with standard due diligence
4. **Commodity traders** importing any of the 7 EUDR commodities
5. **Compliance officers** managing EUDR obligations across supply chains

---

## 3. Goals & Objectives

### 3.1 Primary Goals

1. **DDS Generation**: Generate compliant Due Diligence Statements per Annex II requirements
2. **Geolocation Compliance**: Validate and manage plot geolocations per Article 9
3. **Risk Assessment**: Implement three-level risk scoring (country + supplier + commodity)
4. **EU IS Submission**: Submit DDS to EU Information System and track reference numbers
5. **Supply Chain Visibility**: Map Tier 1-2 suppliers with chain of custody tracking

### 3.2 Success Metrics

- DDS generation accuracy: 99.9% compliant with Annex II schema
- Geolocation validation: <5 seconds per coordinate set
- Risk assessment: <30 seconds per supplier-commodity pair
- Supplier onboarding: 500 suppliers/minute bulk import
- EU IS submission: <60 seconds per DDS

---

## 4. Technical Architecture

### 4.1 Pack Structure

```
PACK-006-eudr-starter/
├── pack.yaml                          # Pack manifest (34 agents, standalone)
├── README.md                          # Documentation
├── config/
│   ├── __init__.py
│   ├── pack_config.py                 # EUDRStarterConfig
│   ├── presets/
│   │   ├── large_operator.yaml        # Full standard DD, all 7 commodities
│   │   ├── mid_market.yaml            # Standard DD, configurable commodities
│   │   ├── sme.yaml                   # Simplified DD (Article 13)
│   │   └── first_time.yaml            # Guided mode with tutorials
│   ├── sectors/
│   │   ├── palm_oil.yaml              # Palm oil importers
│   │   ├── timber_wood.yaml           # Wood/timber importers
│   │   ├── cocoa_coffee.yaml          # Cocoa and coffee importers
│   │   ├── soy_cattle.yaml            # Soy and cattle importers
│   │   └── rubber.yaml                # Rubber importers
│   └── demo/
│       ├── demo_config.yaml
│       ├── demo_suppliers.json        # 10 sample suppliers across 5 commodities
│       └── demo_plots.geojson         # 20 sample plot geolocations
├── engines/
│   ├── __init__.py
│   ├── dds_assembly_engine.py         # DDS composition from upstream agents
│   ├── geolocation_engine.py          # Coordinate/polygon validation pipeline
│   ├── risk_scoring_engine.py         # Multi-source weighted risk aggregation
│   ├── commodity_classification_engine.py  # CN code mapping, Annex I coverage
│   ├── supplier_compliance_engine.py  # Supplier DD status tracking
│   ├── cutoff_date_engine.py          # Dec 31, 2020 cutoff verification
│   └── policy_compliance_engine.py    # EUDR compliance rule enforcement
├── workflows/
│   ├── __init__.py
│   ├── dds_generation.py              # 6-phase DDS workflow
│   ├── supplier_onboarding.py         # 4-phase supplier setup
│   ├── quarterly_compliance_review.py # 3-phase quarterly review
│   ├── data_quality_baseline.py       # 3-phase data quality check
│   ├── risk_reassessment.py           # 3-phase periodic risk update
│   └── bulk_import.py                 # 3-phase bulk data import
├── templates/
│   ├── __init__.py
│   ├── dds_standard_report.py         # Standard DDS per Annex II
│   ├── dds_simplified_report.py       # Simplified DDS (Article 13)
│   ├── compliance_dashboard.py        # Real-time compliance KPIs
│   ├── supplier_risk_report.py        # Per-supplier risk summary
│   ├── country_risk_matrix.py         # Country x commodity risk matrix
│   ├── geolocation_report.py          # Plot verification with maps
│   └── executive_summary.py           # Board-level EUDR status
├── integrations/
│   ├── __init__.py
│   ├── pack_orchestrator.py           # 8-phase EUDR execution pipeline
│   ├── eudr_app_bridge.py             # Bridge to GL-EUDR-APP v1.0
│   ├── traceability_bridge.py         # Bridge to EUDR Traceability Connector
│   ├── satellite_bridge.py            # Bridge to Deforestation Satellite Connector
│   ├── gis_bridge.py                  # Bridge to GIS/Mapping Connector
│   ├── eu_information_system_bridge.py # EU IS API integration
│   ├── setup_wizard.py                # 8-step guided EUDR setup
│   └── health_check.py                # 14-category health verification
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_manifest.py
    ├── test_config.py
    ├── test_dds_assembly.py
    ├── test_geolocation.py
    ├── test_risk_scoring.py
    ├── test_commodity.py
    ├── test_supplier_compliance.py
    ├── test_cutoff_date.py
    ├── test_policy_compliance.py
    ├── test_workflows.py
    ├── test_templates.py
    ├── test_integrations.py
    ├── test_demo.py
    └── test_e2e.py
```

### 4.2 Agent Composition

**18 EUDR-Specific Agents (Starter Tier)**:
- Supply Chain Traceability: EUDR-001, 002, 006, 007, 008, 009, 012, 015 (8 agents)
- Risk Assessment: EUDR-016, 017, 018 (3 agents)
- Due Diligence Core: EUDR-026, 027, 028 (3 agents)
- Due Diligence Workflow: EUDR-030, 036, 037, 038 (4 agents)

**3 Data Intake Agents**:
- AGENT-DATA-001 (PDF & Invoice Extractor)
- AGENT-DATA-002 (Excel/CSV Normalizer)
- AGENT-DATA-005 (EUDR Traceability Connector)

**3 Data Quality Agents**:
- AGENT-DATA-010 (Data Quality Profiler)
- AGENT-DATA-011 (Duplicate Detection)
- AGENT-DATA-019 (Validation Rule Engine)

**10 Foundation Agents**:
- AGENT-FOUND-001 through 010 (Orchestrator, Schema, Units, Assumptions, Citations, Access, Registry, Reproducibility, QA, Observability)

### 4.3 Dependency Graph

```
PACK-006 EUDR Starter
├── bridges: GL-EUDR-APP v1.0 (31 Python + 47 TS/TSX files)
├── bridges: EUDR Traceability Connector (17 files, ~10.3K lines)
├── bridges: Deforestation Satellite Connector (15 files, ~9.2K lines)
├── bridges: GIS/Mapping Connector (15 files, ~9.9K lines)
├── bridges: greenlang/agents/eudr/ (18 of 40 agents)
├── bridges: greenlang/agents/data/ (6 agents)
├── bridges: greenlang/agents/foundation/ (10 agents)
├── migrations: V082, V089-V103, V104-V106, V114-V116, V118, V124-V126
└── config: applications/GL-EUDR-APP/.../config/eudr_config.yaml
```

---

## 5. Engine Specifications

### 5.1 DDS Assembly Engine (`dds_assembly_engine.py`)

**Purpose**: Compose Due Diligence Statements from upstream agent data per Annex II requirements.

**Capabilities**:
- **Article 9 Data Assembly**: Aggregate product description, HS/CN codes, quantity, supplier details, country of production, geolocation data
- **Geolocation Formatting**: Format coordinates to 6-decimal WGS84, polygons to GeoJSON
- **Risk Summary Integration**: Pull risk scores from EUDR-016/017/018 assessments
- **Mitigation Reference**: Link any mitigation measures applied
- **Annex II Compliance Check**: Validate all required DDS fields populated
- **Standard vs Simplified**: Generate appropriate DDS type based on country risk (Article 13)
- **Provenance Trail**: SHA-256 hash chain linking DDS to all source evidence
- **Batch Assembly**: Generate multiple DDS for bulk shipments

### 5.2 Geolocation Engine (`geolocation_engine.py`)

**Purpose**: Validate, normalize, and manage plot geolocation data per Article 9.

**Capabilities**:
- **Coordinate Validation**: Verify WGS84 format, 6-decimal precision, land-based coordinates
- **Polygon Verification**: Validate GeoJSON polygons (closed rings, valid topology, reasonable area)
- **Area Calculation**: Compute plot area in hectares using geodetic calculations
- **Overlap Detection**: Check for overlapping plots within supplier portfolio
- **Country Determination**: Reverse geocode coordinates to country of production
- **Cutoff Date Linkage**: Link geolocation to pre/post December 31, 2020 status
- **Batch Validation**: Process 1,000+ coordinates per minute
- **Format Normalization**: Accept DD, DMS, UTM; normalize to decimal degrees WGS84

### 5.3 Risk Scoring Engine (`risk_scoring_engine.py`)

**Purpose**: Multi-source weighted risk aggregation per Articles 10-11.

**Capabilities**:
- **Country Risk** (weight 35%): Article 29 benchmarking (low/standard/high), deforestation rates, governance indices
- **Supplier Risk** (weight 25%): Certification status, compliance history, documentation completeness, engagement score
- **Commodity Risk** (weight 20%): Commodity-specific deforestation correlation, global supply chain risk, market dynamics
- **Document Risk** (weight 20%): Certificate validity, permit authenticity, documentation gaps
- **Composite Score**: Weighted aggregation producing 0-100 risk score with LOW/MEDIUM/HIGH/CRITICAL classification
- **Risk Factor Breakdown**: Drill-down into individual risk contributors
- **Country Database**: 200+ countries with deforestation risk scores, governance indices, EUDR benchmarking classification
- **Threshold Management**: Configurable thresholds for risk classification

### 5.4 Commodity Classification Engine (`commodity_classification_engine.py`)

**Purpose**: CN code mapping and Annex I coverage for all 7 EUDR commodities.

**Capabilities**:
- **7 Commodity Categories**: Cattle, cocoa, coffee, oil palm, rubber, soya, wood
- **CN Code Database**: All Annex I CN codes (~400+ codes across 7 commodities)
- **HS-to-CN Mapping**: Convert 6-digit HS codes to 8-digit EU CN codes
- **Derived Product Classification**: Determine if a product is a derived product per Annex I
- **Commodity Identification**: Auto-classify products from descriptions and trade codes
- **Multi-Commodity Shipments**: Handle mixed-commodity consignments
- **Annual CN Update**: Support annual nomenclature changes

### 5.5 Supplier Compliance Engine (`supplier_compliance_engine.py`)

**Purpose**: Track supplier due diligence status and compliance readiness.

**Capabilities**:
- **Supplier Registry**: Manage supplier profiles with EUDR-relevant data
- **DD Status Tracking**: Track each supplier's due diligence status (not started → in progress → complete → verified)
- **Certification Tracking**: Monitor FSC, RSPO, PEFC, Rainforest Alliance, UTZ certifications
- **Data Completeness Scoring**: Score supplier data completeness (0-100) per Article 9 requirements
- **Engagement Campaigns**: Track data request/response cycles with suppliers
- **Risk-Based Prioritization**: Prioritize supplier engagement based on risk scores
- **Compliance Calendar**: Track DDS submission deadlines per supplier

### 5.6 Cutoff Date Engine (`cutoff_date_engine.py`)

**Purpose**: Verify deforestation-free status against December 31, 2020 cutoff date.

**Capabilities**:
- **Cutoff Date Verification**: Check if land was forested on or after December 31, 2020
- **Historical Land Use**: Cross-reference with available historical data
- **Evidence Collection**: Gather evidence supporting deforestation-free status
- **Temporal Analysis**: Analyze satellite imagery dates relative to cutoff
- **Declaration Support**: Generate cutoff compliance declaration text for DDS
- **Exemption Handling**: Handle products from pre-cutoff cleared land

### 5.7 Policy Compliance Engine (`policy_compliance_engine.py`)

**Purpose**: Enforce EUDR compliance rules across all data and processes.

**Capabilities**:
- **45 Compliance Rules**: Covering Articles 3, 4, 8-13, 29
- **Rule Categories**: GEOLOCATION (8 rules), COMMODITY (6 rules), SUPPLIER (7 rules), RISK (6 rules), DDS (8 rules), DOCUMENTATION (5 rules), CUTOFF (5 rules)
- **Weighted Compliance Score**: Overall EUDR compliance score (0-100)
- **Per-Rule Assessment**: Individual pass/fail with remediation guidance
- **Simplified DD Rules**: Special rules for Article 13 (low-risk country simplified DD)
- **Penalty Risk**: Estimate penalty exposure for compliance gaps
- **Audit Trail**: Log all compliance checks for verifiability

---

## 6. Workflow Specifications

### 6.1 DDS Generation Workflow (`dds_generation.py`)

**6-phase workflow (17 working days typical)**:
1. **SupplierOnboarding** (5 days): Collect supplier profiles, geolocation, certificates via EUDR-001, 008, DATA-001, DATA-002
2. **GeolocationCollection** (3 days): Validate coordinates/polygons via EUDR-002, 006, 007; normalize to WGS84
3. **DocumentCollection** (3 days): Gather and authenticate certificates, permits via EUDR-012, DATA-001
4. **RiskAssessment** (2 days): Run country + supplier + commodity risk via EUDR-016, 017, 018, 028
5. **DDSGeneration** (2 days): Assemble DDS from all data via EUDR-030, 037, 038; validate per Annex II
6. **ReviewAndSubmit** (2 days): Human review, approval, submit to EU IS via EUDR-036; track reference number

### 6.2 Supplier Onboarding Workflow (`supplier_onboarding.py`)

**4-phase per-supplier setup**:
1. **DataIntake**: Import supplier data (CSV/Excel/manual) via DATA-001, DATA-002, DATA-005
2. **SupplierProfiling**: Profile supplier, validate data quality via EUDR-008, 017, DATA-010
3. **GeolocationSetup**: Collect and validate plot geolocations via EUDR-002, 006, 007
4. **InitialRiskScoring**: Calculate initial risk scores via EUDR-016, 018

### 6.3 Quarterly Compliance Review (`quarterly_compliance_review.py`)

**3-phase quarterly cycle**:
1. **DataRefresh**: Update supplier data, certifications, country risk via DATA-001, DATA-002
2. **RiskRecalculation**: Recalculate all risk scores via EUDR-016, 017, 018
3. **ComplianceReporting**: Generate compliance reports and dashboards via EUDR-030, GL-EUDR-APP

### 6.4 Data Quality Baseline (`data_quality_baseline.py`)

**3-phase data quality assessment**:
1. **Profiling**: Profile all geolocation and supplier data via DATA-010, DATA-011
2. **Validation**: Apply EUDR validation rules via DATA-019
3. **Remediation**: Generate remediation plan for data quality issues

### 6.5 Risk Reassessment (`risk_reassessment.py`)

**3-phase periodic risk update**:
1. **DataCollection**: Gather updated country, supplier, and commodity data
2. **RiskRecalculation**: Recalculate all risk scores with updated data
3. **AlertGeneration**: Generate alerts for risk changes exceeding thresholds

### 6.6 Bulk Import Workflow (`bulk_import.py`)

**3-phase bulk data import**:
1. **FileProcessing**: Parse uploaded files (CSV/Excel/JSON/GeoJSON)
2. **ValidationAndEnrichment**: Validate, deduplicate, enrich data; flag errors
3. **Integration**: Load validated data into EUDR system; generate import report

---

## 7. Template Specifications

### 7.1 DDS Standard Report (`dds_standard_report.py`)
Standard Due Diligence Statement per Annex II: product info, geolocation, risk assessment, mitigation measures, operator declaration, reference number.

### 7.2 DDS Simplified Report (`dds_simplified_report.py`)
Simplified DDS for products from low-risk countries per Article 13: reduced information requirements, streamlined risk assessment.

### 7.3 Compliance Dashboard (`compliance_dashboard.py`)
Real-time compliance KPIs: total suppliers, DDS submitted, risk distribution, certification coverage, data quality scores, upcoming deadlines.

### 7.4 Supplier Risk Report (`supplier_risk_report.py`)
Per-supplier risk assessment: composite score breakdown, certification status, geolocation verification, data completeness, engagement history.

### 7.5 Country Risk Matrix (`country_risk_matrix.py`)
Country x commodity risk matrix: Article 29 benchmarking, deforestation rates, governance indices, supplier concentration.

### 7.6 Geolocation Report (`geolocation_report.py`)
Plot verification report: coordinate validation results, polygon analysis, area calculations, overlap detection, country determination, cutoff date status.

### 7.7 Executive Summary (`executive_summary.py`)
Board-level EUDR status: overall compliance score, DDS submission status, risk exposure, key action items, regulatory deadline tracker.

---

## 8. Integration Specifications

### 8.1 Pack Orchestrator (`pack_orchestrator.py`)

8-phase EUDR Starter execution pipeline:
1. **HealthCheck**: Run 14-category health verification
2. **ConfigurationLoading**: Load EUDRStarterConfig with preset/sector overlays
3. **DataIntake**: Import supplier and geolocation data via data agents
4. **GeolocationValidation**: Validate all coordinates/polygons via geolocation engine
5. **RiskAssessment**: Calculate multi-source risk scores via risk scoring engine
6. **DDSAssembly**: Generate Due Diligence Statements via DDS assembly engine
7. **ComplianceCheck**: Verify all compliance rules via policy compliance engine
8. **Reporting**: Render all templates, update dashboards

### 8.2 EUDR App Bridge (`eudr_app_bridge.py`)
Bridge to GL-EUDR-APP v1.0 backend API: supplier routes, plot routes, DDS routes, pipeline routes, risk routes, dashboard routes.

### 8.3 Traceability Bridge (`traceability_bridge.py`)
Bridge to EUDR Traceability Connector: plot registry, chain of custody, batch traceability, compliance verification.

### 8.4 Satellite Bridge (`satellite_bridge.py`)
Bridge to Deforestation Satellite Connector: imagery acquisition, forest change detection, alert aggregation, compliance reporting.

### 8.5 GIS Bridge (`gis_bridge.py`)
Bridge to GIS/Mapping Connector: coordinate transformation, boundary resolution, spatial analysis, land cover classification.

### 8.6 EU Information System Bridge (`eu_information_system_bridge.py`)
Bridge to EU EUDR Information System: DDS submission, reference number retrieval, status tracking, amendment handling.

### 8.7 Setup Wizard (`setup_wizard.py`)
8-step guided EUDR setup:
1. Select commodity focus (1-7 commodities)
2. Choose company size preset
3. Configure geolocation settings
4. Set risk assessment thresholds
5. Configure EU IS connection
6. Import initial supplier data
7. Run demo with sample data
8. Health check and readiness verification

### 8.8 Health Check (`health_check.py`)
14-category health verification:
1. Configuration validity
2. DDS Assembly Engine
3. Geolocation Engine
4. Risk Scoring Engine
5. Commodity Classification Engine
6. Supplier Compliance Engine
7. Cutoff Date Engine
8. Policy Compliance Engine
9. EUDR App Bridge
10. Traceability Bridge
11. Satellite Bridge
12. GIS Bridge
13. EU IS Bridge
14. Demo Data Availability

---

## 9. Configuration Model

### 9.1 EUDRStarterConfig

```python
class EUDRStarterConfig(BaseModel):
    pack_id: str = "PACK-006-eudr-starter"
    version: str = "1.0.0"
    tier: str = "starter"

    # Core
    operator: OperatorConfig
    commodities: List[CommodityConfig]
    geolocation: GeolocationConfig
    risk_assessment: RiskAssessmentConfig

    # DDS
    dds: DDSConfig
    eu_information_system: EUISConfig

    # Supply Chain
    supply_chain: SupplyChainConfig
    supplier: SupplierConfig

    # Compliance
    compliance: ComplianceConfig
    cutoff: CutoffDateConfig

    # Operations
    reporting: ReportingConfig
    demo: DemoConfig
```

### 9.2 Sub-Configurations

- **OperatorConfig**: company_name, eori_number, registration_country, operator_type (OPERATOR/TRADER), company_size (SME/MID_MARKET/LARGE)
- **CommodityConfig**: commodity_type (7 enums), cn_codes list, high_risk_origins, certification_schemes
- **GeolocationConfig**: coordinate_precision (6 decimals), polygon_max_vertices, area_unit (HECTARES), crs (WGS84)
- **RiskAssessmentConfig**: country_weight (0.35), supplier_weight (0.25), commodity_weight (0.20), document_weight (0.20), thresholds (low/medium/high/critical)
- **DDSConfig**: dds_type (STANDARD/SIMPLIFIED), auto_generate, review_required, template_version
- **EUISConfig**: api_url, sandbox_mode, auth_credentials, submission_retry_count
- **SupplyChainConfig**: max_tier_depth (2 for starter), chain_of_custody_model (IDENTITY_PRESERVED/SEGREGATED/MASS_BALANCE)
- **SupplierConfig**: bulk_import_limit, engagement_auto_reminders, data_completeness_threshold
- **ComplianceConfig**: rules_enabled list, compliance_score_threshold, simplified_dd_enabled
- **CutoffDateConfig**: cutoff_date (2020-12-31), evidence_required, temporal_buffer_days
- **ReportingConfig**: dashboard_refresh_interval, report_formats (PDF/HTML/JSON), auto_generate_quarterly
- **DemoConfig**: enabled, demo_suppliers_count, demo_plots_count

---

## 10. EUDR Commodities Reference

### 10.1 Seven Commodities with CN Codes

| Commodity | Key CN Codes | High-Risk Origins | Certifications |
|-----------|-------------|-------------------|----------------|
| Cattle | 0102, 0201-0202, 4101-4107, 4301 | BRA, ARG, PRY, BOL, COL | None standard |
| Cocoa | 1801-1806 | CIV, GHA, CMR, NGA, IDN | Rainforest Alliance, UTZ, Fairtrade |
| Coffee | 0901 | BRA, VNM, COL, IDN, ETH | 4C, Rainforest Alliance, UTZ |
| Oil Palm | 1511, 1513, 3823 | IDN, MYS, PNG, COL, NGA | RSPO, ISCC |
| Rubber | 4001, 4005, 4011-4013 | THA, IDN, VNM, CIV, MYS | FSC, PEFC |
| Soya | 1201, 1507, 2304 | BRA, ARG, USA, PRY, BOL | RTRS, ProTerra |
| Wood | 4401-4421, 4501-4504, 9401-9403 | BRA, COD, IDN, MYS, PNG | FSC, PEFC |

### 10.2 Country Risk Database (200+ countries)

Article 29 benchmarking classification:
- **High Risk**: BRA, COD, COG, IDN, MYS, PNG, MMR, LAO, KHM, BOL, PRY, GHA, CIV, CMR, NGA, LBR, SLE, MDG, MOZ, TZA
- **Standard Risk**: All countries not classified as high or low risk
- **Low Risk**: EU member states, CAN, AUS, NZL, JPN, KOR, SGP, CHE, NOR, ISL, GBR, USA (select states)

---

## 11. Test Plan

### 11.1 Test Categories

| Category | Tests | Focus |
|----------|-------|-------|
| Manifest | 15 | pack.yaml validation, agent listing, commodity coverage |
| Config | 45 | EUDRStarterConfig, all sub-configs, presets, sectors |
| DDS Assembly | 25 | Standard/simplified DDS, Annex II compliance, batch assembly |
| Geolocation | 25 | Coordinates, polygons, area, overlap, country, normalization |
| Risk Scoring | 25 | Country/supplier/commodity/document risk, composite, thresholds |
| Commodity | 15 | CN codes, 7 commodities, derived products, classification |
| Supplier Compliance | 15 | Status tracking, certification, completeness, engagement |
| Cutoff Date | 10 | Dec 31 2020 verification, evidence, exemptions |
| Policy Compliance | 20 | 45 rules, categories, scoring, simplified DD |
| Workflows | 30 | All 6 workflows end-to-end, phase-level |
| Templates | 25 | All 7 templates in md/html/json |
| Integrations | 25 | All 8 integrations, bridge connectivity |
| Demo | 10 | Setup wizard, demo data, demo execution |
| E2E | 15 | Full pipeline from intake to DDS submission |
| **Total** | **300** | |

### 11.2 Test Patterns

- All tests use `sys.path.insert(0, os.path.dirname(__file__))` for conftest imports
- Pydantic BaseModel for all test fixtures
- SHA-256 provenance hashing on all outputs
- No external API dependencies (all mocked/stubbed)
- conftest.py provides shared fixtures: sample suppliers, plots, DDS, risk data

---

## 12. Assets Leveraged

### 12.1 EUDR Agents (18 of 40)

| Category | Agents | Files | Lines |
|----------|--------|-------|-------|
| Supply Chain Traceability | EUDR-001, 002, 006, 007, 008, 009, 012, 015 | ~229 | ~245K |
| Risk Assessment | EUDR-016, 017, 018 | ~92 | ~79K |
| Due Diligence Core | EUDR-026, 027, 028 | ~64 | ~38K |
| Due Diligence Workflow | EUDR-030, 036, 037, 038 | ~56 | ~31K |

### 12.2 GL-EUDR-APP v1.0
- Backend: 31 Python files, ~18.8K lines
- Frontend: 47 TS/TSX files, ~12.7K lines
- Config: eudr_config.yaml (275 lines)
- Tests: 11 test files

### 12.3 Data Connectors
- EUDR Traceability Connector: 17 files, ~10.3K lines
- Deforestation Satellite Connector: 15 files, ~9.2K lines
- GIS/Mapping Connector: 15 files, ~9.9K lines

### 12.4 Foundation Agents
- All 10 AGENT-FOUND agents (orchestrator through observability)

### 12.5 Data Agents
- DATA-001 (PDF/Invoice), DATA-002 (Excel/CSV), DATA-005 (EUDR Traceability)
- DATA-010 (Quality Profiler), DATA-011 (Duplicate Detection), DATA-019 (Validation Rules)

---

## 13. Presets

### 13.1 By Company Size

- **large_operator**: Full standard DD, all 7 commodities, all 18 EUDR agents, comprehensive reporting
- **mid_market**: Standard DD, configurable commodities (1-7), core agents, standard reporting
- **sme**: Simplified DD (Article 13 for low-risk), reduced quality gates, essential agents only
- **first_time**: Guided mode with step-by-step wizard, pre-populated examples, extended help text

### 13.2 By Commodity Focus

- **palm_oil**: Palm oil focus (IDN, MYS, PNG), RSPO certification tracking, plantation boundaries
- **timber_wood**: Wood/timber focus (BRA, COD, IDN), FSC/PEFC certification, logging concessions
- **cocoa_coffee**: Cocoa and coffee (CIV, GHA, BRA), cooperative mapping, smallholder geolocation
- **soy_cattle**: Soy and cattle (BRA, ARG, PRY), ranch/farm boundaries, Cerrado/Amazon monitoring
- **rubber**: Rubber focus (THA, IDN, VNM), smallholder networks, processing chain tracking

---

## 14. Build Results

*To be completed after development.*
