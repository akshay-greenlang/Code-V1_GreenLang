# PRD-PACK-049: GHG Multi-Site Management Pack

## 1. Overview

### 1.1 Pack Identity
- **Pack ID**: PACK-049
- **Name**: GHG Multi-Site Management Pack
- **Category**: GHG Accounting Packs (Solution Packs)
- **Tier**: Enterprise
- **Version**: 1.0.0
- **Status**: Production Ready
- **Date**: March 2026

### 1.2 Purpose
PACK-049 provides a comprehensive multi-site GHG inventory management system for organisations operating across multiple facilities, regions, and legal entities. It enables facility-level data collection, site registry management, organisational boundary definition (equity share / operational control / financial control), regional emission factor assignment, multi-site consolidation with intra-group elimination, shared-services allocation, internal portfolio benchmarking across sites, submission completeness tracking, site-level data quality scoring, and consolidated multi-site reporting.

### 1.3 Problem Statement
Large organisations with 10 to 10,000+ sites face several challenges in GHG accounting:
1. **Decentralised data collection** - Sites operate independently with varying data maturity
2. **Organisational boundary complexity** - Subsidiaries, JVs, franchises, and leased assets require different consolidation approaches
3. **Regional emission factor variance** - Grid factors, fuel factors, and climate zones differ by geography
4. **Shared services allocation** - Corporate offices, shared logistics, and co-generation require fair allocation
5. **Submission completeness** - Tracking which sites have reported and which gaps remain
6. **Data quality heterogeneity** - Site-level data quality varies from measured (Tier 3) to estimated (Tier 1)
7. **Internal benchmarking** - Comparing sites to identify best practices and laggards
8. **Consolidation accuracy** - Avoiding double-counting and ensuring correct equity/control adjustments

### 1.4 Distinction from Existing Components
| Component | Scope | PACK-049 Adds |
|-----------|-------|---------------|
| PACK-041 `organizational_boundary_engine` | Basic entity/facility definition | Comprehensive site lifecycle, facility classification taxonomy, site characteristics management |
| PACK-044 `consolidation_management_engine` | Entity-level consolidation | Facility-level consolidation, shared services allocation, landlord/tenant splits |
| PACK-043 `multi_entity_boundary_engine` | Scope 3 entity boundaries | All-scope facility-level management with regional factor assignment |
| GL-MRV-X-010 `inventory_boundary` | Boundary inclusion/exclusion | Full site registry with lifecycle (commissioning through decommissioning) |
| GL-MRV-X-008 `consolidation_rollup` | Multi-entity rollup | Site-level granularity with allocation engines and quality scoring |

PACK-049 provides the **orchestration layer** that ties all these components together into a complete multi-site management workflow.

### 1.5 Regulatory Basis
- **GHG Protocol Corporate Standard** - Chapter 3: Setting Organisational Boundaries; Chapter 4: Setting Operational Boundaries
- **GHG Protocol Scope 2 Guidance** - Location-based vs market-based by facility
- **ISO 14064-1:2018** - Clause 5: Organisational boundaries; Clause 6: Reporting boundaries
- **EU CSRD / ESRS E1** - Consolidated GHG reporting across value chain
- **US SEC Climate Rules** - Entity-level GHG disclosure for registrants
- **California SB 253** - Facility-level emissions reporting
- **PCAF Global Standard** - Data quality scoring per asset/facility
- **GRI 305** - Site-level and consolidated GHG reporting
- **IFRS S2** - Climate disclosure at reporting entity level
- **EPA Mandatory Reporting Rule (40 CFR 98)** - Facility-level GHG reporting
- **SECR (UK)** - Energy and carbon reporting per legal entity
- **NGER (Australia)** - Facility-level GHG reporting

### 1.6 Pack Dependencies
| Dependency | Relationship | Required |
|-----------|-------------|----------|
| PACK-041 | Scope 1-2 emissions per facility | Yes |
| PACK-042 | Scope 3 basic per site | No |
| PACK-043 | Scope 3 enterprise + multi-entity boundary | No |
| PACK-044 | Inventory management, consolidation | No |
| PACK-045 | Base year per site | No |
| PACK-046 | Intensity metrics per site | No |
| PACK-047 | External benchmark per site | No |
| PACK-048 | Assurance prep per site | No |
| AGENT-MRV | 30 MRV agents for emissions calculation | Yes |
| AGENT-DATA | 20 data agents for data collection | Yes |
| AGENT-FOUND | FOUND-003 (Normaliser), FOUND-004 (Assumptions), FOUND-005 (Citations) | Yes |

---

## 2. Architecture

### 2.1 Component Overview
| Component | Count | Description |
|-----------|-------|-------------|
| Engines | 10 | Core calculation and management engines |
| Workflows | 8 | Multi-phase orchestrated processes |
| Templates | 10 | Report generation templates |
| Integrations | 12 | External system bridges |
| Configuration | 1 | Pydantic v2 config with 18 enums, 15 sub-configs |
| Presets | 8 | Industry-specific configuration presets |
| DB Migrations | 10 | PostgreSQL schema (V406-V415) |
| Tests | 18 | Unit, integration, E2E test files |

### 2.2 Engine Specifications

#### Engine 1: SiteRegistryEngine (~1,200 lines)
Comprehensive facility registry with classification, characteristics, and lifecycle management.
- **Facility Types**: 20+ types (MANUFACTURING, OFFICE, WAREHOUSE, RETAIL, DATA_CENTER, LABORATORY, HOSPITAL, HOTEL, RESTAURANT, SCHOOL, UNIVERSITY, GOVERNMENT, MILITARY, AIRPORT, PORT, MINE, REFINERY, POWER_PLANT, FARM, OTHER)
- **Facility Lifecycle**: PLANNED -> COMMISSIONING -> OPERATIONAL -> UNDER_RENOVATION -> TEMPORARILY_CLOSED -> DECOMMISSIONING -> DECOMMISSIONED
- **Characteristics**: Floor area (m2), headcount, operating hours, production output, production unit, grid region, climate zone
- **Classification Taxonomy**: By type, sector (GICS/NACE), geography (country/region/city), business unit, legal entity
- **Grouping**: Custom site groups for portfolio management
- **Temporal Tracking**: Acquisition date, commissioning date, renovation dates, decommissioning date
- **Provenance**: SHA-256 hash on all registry changes

#### Engine 2: SiteDataCollectionEngine (~1,100 lines)
Decentralised data collection with site-level submission templates and validation.
- **Collection Templates**: Energy (electricity, gas, fuel), waste, water, refrigerants, transport, process emissions
- **Data Entry Modes**: Manual entry, spreadsheet upload (XLSX/CSV), API push, ERP connector
- **Validation Rules**: Range checks, YoY variance checks, unit validation, completeness checks
- **Submission Workflow**: DRAFT -> SUBMITTED -> UNDER_REVIEW -> APPROVED -> REJECTED -> RESUBMITTED
- **Period Management**: Monthly, quarterly, annual collection periods
- **Delegation**: Site-level data stewards with role-based access
- **Estimation Support**: Proxy data, extrapolation, benchmark-based estimation with quality flags

#### Engine 3: SiteBoundaryEngine (~1,000 lines)
Organisational boundary definition with consolidation approach per facility.
- **Consolidation Approaches**: Equity share, operational control, financial control (per GHG Protocol Ch. 3)
- **Ownership Mapping**: Legal entity -> facility ownership chain with equity percentages
- **Boundary Changes**: Acquisition, divestiture, merger, restructure, JV formation/dissolution
- **Time-Weighted Consolidation**: Pro-rata consolidation for mid-year structural changes
- **Exclusion Management**: De minimis, not relevant, data unavailable, below threshold with justifications
- **Materiality Assessment**: Facility-level materiality scoring against corporate threshold (default 5%)
- **Annual Boundary Lock**: Freeze boundary definition for reporting period

#### Engine 4: RegionalFactorEngine (~1,000 lines)
Regional emission factor assignment with tiered lookup and grid region management.
- **Factor Tiers**: Facility-specific (Tier 3) > National (Tier 2) > Regional (Tier 1) > IPCC default
- **Factor Sources**: IPCC 2006/2019, DEFRA, EPA eGRID, IEA, UBA, ADEME, ISPRA, Ecoinvent, Supplier-specific
- **Grid Factors**: Location-based country/region grid EFs, market-based residual mix / supplier-specific
- **Temporal Factors**: Annual updates, vintage year tracking, forward projection
- **Climate Zone Assignment**: HDD/CDD by facility location for weather normalisation
- **Currency/Unit Normalisation**: PPP conversion for spend-based, unit harmonisation across regions
- **Factor Override**: Manual override with audit trail and justification

#### Engine 5: SiteConsolidationEngine (~1,200 lines)
Multi-site to corporate-level consolidation with equity/control adjustments.
- **Consolidation Methods**: Equity share (proportional), operational control (100%/0%), financial control (100%/0%)
- **Scope Coverage**: Scope 1, Scope 2 location-based, Scope 2 market-based, Scope 3 (15 categories)
- **Intra-Group Elimination**: Electricity transfers, steam transfers, internal transport, waste transfers, product transfers
- **Equity Adjustments**: Proportional allocation for JVs, associates, minority interests
- **Completeness Checks**: Missing sites, partial data, estimation fill rates
- **Reconciliation**: Bottom-up (sum of sites) vs top-down (corporate total) variance analysis
- **Restatement**: Handle base-year restatement when sites added/removed

#### Engine 6: SiteAllocationEngine (~900 lines)
Shared services allocation, landlord/tenant splits, and co-generation allocation.
- **Allocation Methods**: Floor area, headcount, revenue, production output, energy consumption, custom formula
- **Shared Services**: Corporate HQ, shared IT, shared logistics, central labs
- **Landlord/Tenant Split**: Whole building vs tenant-only, common area allocation, sub-metering support
- **Co-Generation**: CHP allocation between electricity and heat outputs (efficiency method, energy content method)
- **District Heating/Cooling**: Allocation of district system emissions to connected facilities
- **Virtual Power Purchase**: Allocation of VPPA certificates across sites
- **Allocation Hierarchy**: Direct measurement > sub-metering > allocation formula > pro-rata

#### Engine 7: SiteComparisonEngine (~900 lines)
Internal portfolio benchmarking across sites with KPIs and best practice identification.
- **KPI Metrics**: Emissions intensity (tCO2e/m2, tCO2e/FTE, tCO2e/unit, tCO2e/revenue)
- **Peer Grouping**: By facility type, size band, climate zone, geography, business unit
- **Statistical Analysis**: Mean, median, percentiles (P10/P25/P75/P90), standard deviation, IQR
- **Ranking**: Site ranking by KPI with normalised scoring
- **Best Practice Identification**: Top-quartile sites, improvement leaders, consistent performers
- **Gap Analysis**: Site-level gap to best practice, estimated reduction potential
- **Trend Tracking**: Year-over-year improvement rates per site

#### Engine 8: SiteCompletionEngine (~850 lines)
Submission tracking, completeness monitoring, gap detection, and deadline management.
- **Completeness Scoring**: By site, scope, data category, period
- **Submission Tracker**: Status dashboard per site per period (not started, in progress, submitted, approved)
- **Gap Detection**: Missing sites, missing scopes, missing months, missing data categories
- **Estimation Coverage**: Percentage of data estimated vs measured per site
- **Deadline Management**: Collection deadlines, review deadlines, reporting deadlines with escalation
- **Reminder Engine**: Automated reminders at configurable intervals (14d, 7d, 3d, 1d before deadline)
- **Coverage Metrics**: % of sites reporting, % of emissions covered, % of floor area covered

#### Engine 9: SiteQualityEngine (~850 lines)
Site-level data quality scoring, estimation hierarchy, and improvement tracking.
- **Quality Framework**: PCAF data quality scores 1-5 per data point
- **Quality Dimensions**: Accuracy, completeness, consistency, timeliness, methodology, documentation
- **Estimation Hierarchy**: Direct measurement > calculation > estimation > extrapolation > proxy
- **Quality Heatmap**: Site x scope quality matrix with colour coding
- **Improvement Tracking**: Quality score progression over time per site
- **Remediation Plans**: Prioritised actions to improve site-level data quality
- **Aggregated Quality**: Weighted corporate-level quality score based on emissions contribution

#### Engine 10: MultiSiteReportingEngine (~1,100 lines)
Site-level and consolidated reports with multi-format export and drill-down dashboards.
- **Report Types**: Site portfolio dashboard, site detail report, consolidation report, boundary report, allocation report, comparison report, completion report, quality report, trend report, regulatory report
- **Export Formats**: Markdown, HTML, JSON, CSV, XBRL (ESRS E1)
- **Drill-Down**: Corporate -> legal entity -> business unit -> region -> site
- **Aggregation Levels**: Group, entity, business unit, country, region, site
- **Visualisation Hooks**: Chart data structures for Grafana/frontend rendering
- **Multi-Year Trending**: Year-over-year comparisons at all aggregation levels
- **Provenance**: SHA-256 hash on all report outputs

### 2.3 Workflow Specifications

#### Workflow 1: SiteRegistrationWorkflow (5 phases)
`SiteDiscovery -> Classification -> CharacteristicsCapture -> BoundaryAssignment -> Activation`

#### Workflow 2: DataCollectionWorkflow (5 phases)
`PeriodSetup -> TemplateDistribution -> SiteSubmission -> ValidationReview -> Approval`

#### Workflow 3: BoundaryDefinitionWorkflow (5 phases)
`EntityMapping -> OwnershipChain -> ConsolidationApproach -> MaterialityCheck -> BoundaryLock`

#### Workflow 4: ConsolidationWorkflow (5 phases)
`SiteDataGather -> EliminationCheck -> EquityAdjust -> Reconcile -> ConsolidatedTotal`

#### Workflow 5: AllocationWorkflow (4 phases)
`SharedServiceID -> AllocationMethodSelect -> Calculate -> Verify`

#### Workflow 6: SiteComparisonWorkflow (5 phases)
`PeerGroupBuild -> KPICalculate -> Rank -> GapAnalysis -> BestPracticeReport`

#### Workflow 7: QualityImprovementWorkflow (5 phases)
`QualityAssess -> GapIdentify -> RemediationPlan -> Implementation -> Verification`

#### Workflow 8: FullMultiSitePipelineWorkflow (8 phases)
End-to-end multi-site management: `Registration -> Collection -> Boundary -> Consolidation -> Allocation -> Comparison -> Quality -> Reporting`

### 2.4 Template Specifications

| # | Template | Description |
|---|----------|-------------|
| 1 | SitePortfolioDashboard | All-sites overview with KPIs, map data, status |
| 2 | SiteDetailReport | Individual site drill-down with emissions breakdown |
| 3 | ConsolidationReport | Corporate-level consolidated emissions |
| 4 | BoundaryDefinitionReport | Organisational boundary documentation |
| 5 | RegionalFactorReport | Regional EF assignment matrix |
| 6 | AllocationReport | Shared services allocation breakdown |
| 7 | SiteComparisonReport | Cross-site benchmarking league table |
| 8 | DataCollectionStatusReport | Submission tracker dashboard |
| 9 | DataQualityReport | Site-level quality heatmap |
| 10 | MultiSiteTrendReport | Year-over-year by site |

### 2.5 Integration Specifications

| # | Integration | Description |
|---|------------|-------------|
| 1 | PackOrchestrator | 10-phase DAG pipeline coordinator |
| 2 | MRVBridge | 30 AGENT-MRV agents for emissions calculation per site |
| 3 | DataBridge | AGENT-DATA agents for data collection |
| 4 | Pack041Bridge | Scope 1-2 organisational boundary and facility data |
| 5 | Pack042043Bridge | Scope 3 multi-entity boundaries |
| 6 | Pack044Bridge | Inventory management and consolidation |
| 7 | Pack045Bridge | Base year per site |
| 8 | Pack046047Bridge | Intensity metrics and benchmark per site |
| 9 | FoundationBridge | FOUND-003 normaliser, FOUND-004 assumptions, FOUND-005 citations |
| 10 | HealthCheck | 20-category system verification |
| 11 | SetupWizard | 8-step guided configuration |
| 12 | AlertBridge | Deadline, completeness, quality, boundary, allocation alerts |

### 2.6 Configuration

**18 Enums**: FacilityType, FacilityLifecycle, ConsolidationApproach, OwnershipType, CollectionPeriodType, SubmissionStatus, DataEntryMode, AllocationMethod, LandlordTenantSplit, CogenerationType, FactorTier, FactorSource, QualityDimension, QualityScore, ComparisonKPI, ReportType, ExportFormat, AlertType

**15 Sub-Configs**: SiteRegistryConfig, DataCollectionConfig, BoundaryConfig, RegionalFactorConfig, ConsolidationConfig, AllocationConfig, ComparisonConfig, CompletionConfig, QualityConfig, ReportingConfig, SecurityConfig, PerformanceConfig, IntegrationConfig, AlertConfig, MigrationConfig

**8 Presets**: corporate_general, manufacturing, retail_chain, real_estate, financial_services, logistics, healthcare, public_sector

### 2.7 Database Schema

- **Schema**: `ghg_multisite`
- **Table Prefix**: `gl_ms_`
- **Migrations**: V406-V415 (10 files)

| Migration | Content |
|-----------|---------|
| V406 | Core schema (configurations, reporting_periods) |
| V407 | Site registry (sites, site_characteristics, site_groups, site_group_members) |
| V408 | Data collection (collection_rounds, site_submissions, submission_data, validation_results) |
| V409 | Boundary (boundary_definitions, entity_ownership, boundary_inclusions, boundary_changes) |
| V410 | Regional factors (factor_assignments, factor_overrides, grid_regions, climate_zones) |
| V411 | Consolidation (consolidation_runs, site_totals, elimination_entries, reconciliation_results) |
| V412 | Allocation (allocation_configs, allocation_runs, allocation_results) |
| V413 | Comparison + quality (site_kpis, site_rankings, quality_scores, quality_dimensions) |
| V414 | Completion (completion_status, submission_tracker, reminders, deadlines) |
| V415 | Views, indexes, seed data (facility types, allocation methods, quality dimensions) |

---

## 3. Technical Specifications

### 3.1 Zero-Hallucination Compliance
- All consolidation calculations use Python `Decimal` with `ROUND_HALF_UP`
- No LLM used in any calculation path
- SHA-256 provenance hash on every result
- Deterministic allocation formulas from GHG Protocol
- Regional factors from authoritative databases only (IPCC, DEFRA, EPA, IEA)

### 3.2 Security
- JWT RS256 authentication via SEC-001
- RBAC authorisation via SEC-002 with permissions: `multisite:read`, `multisite:write`, `multisite:admin`, `multisite:site_steward`
- AES-256-GCM encryption at rest via SEC-003
- TLS 1.3 in transit via SEC-004
- Full audit trail via SEC-005
- Row-level security (RLS) tenant isolation on all tables

### 3.3 Performance Targets
- Site registry operations: <1 second per 1,000 sites
- Data collection template generation: <5 seconds
- Consolidation run: <60 seconds for 1,000 sites
- Allocation calculation: <30 seconds for 500 sites
- Site comparison: <15 seconds for 1,000 sites
- Full pipeline: <5 minutes for 1,000 sites
- Report generation: <10 seconds per format

### 3.4 Testing
- 18 test files (conftest + 10 engine + workflows + integrations + templates + config + e2e)
- Target: 700+ test functions
- 100% pass rate
- Coverage target: 95%+
- E2E scenarios: 100+
- Determinism verified

---

## 4. Preset Specifications

### 4.1 corporate_general
Default multi-site configuration for general corporate use. Operational control approach, monthly collection, floor area allocation.

### 4.2 manufacturing
Multi-plant manufacturing with production output KPIs, process emissions, co-generation, Tier 3 factors preferred.

### 4.3 retail_chain
High site count (100-10,000+), standardised store format, electricity-dominant, headcount allocation.

### 4.4 real_estate
Property portfolio with landlord/tenant splits, floor area intensity, climate zone normalisation, CRREM alignment.

### 4.5 financial_services
Bank branch network, office-dominant, FTE intensity, financial control approach, PCAF quality scoring.

### 4.6 logistics
Warehouses and distribution centres, throughput-based KPIs, transport allocation, fuel-dominant.

### 4.7 healthcare
Hospital network, bed-day KPIs, medical waste, refrigerant-heavy, 24/7 operations.

### 4.8 public_sector
Government buildings, energy performance certificates, floor area benchmarking, public reporting obligations.

---

## 5. Implementation Notes

### 5.1 Leverages Existing Components
- PACK-041 `organizational_boundary_engine` for base entity/facility structures
- PACK-044 `consolidation_management_engine` for entity-level consolidation patterns
- PACK-041 `emission_factor_manager_engine` for factor tiering logic
- GL-MRV-X-010 `inventory_boundary` for facility type taxonomy
- GL-MRV-X-008 `consolidation_rollup` for intercompany elimination patterns

### 5.2 Extends (Does Not Duplicate)
PACK-049 provides the **multi-site management orchestration layer**. It does not re-implement emission calculations (deferred to MRV agents), factor databases (deferred to PACK-041), or entity consolidation logic (deferred to PACK-044). Instead, it provides:
- Site lifecycle management (registry, classification, grouping)
- Decentralised collection workflows
- Regional factor assignment (not factor database itself)
- Allocation of shared services (new capability)
- Internal portfolio benchmarking (new capability)
- Completeness and quality management (new capability)
- Multi-site consolidated reporting with drill-down (new capability)

### 5.3 Standard Patterns
- Pydantic v2 configuration with `from_preset()`, `from_yaml()`, `merge()`, `validate()`
- Lazy import in `__init__.py` files with `_loaded_*` tracking
- Standard helpers: `_utcnow()`, `_new_uuid()`, `_compute_hash()`, `_MODULE_VERSION`
- `from __future__ import annotations` in all files
- SHA-256 provenance hash on every calculation result
- Flyway-compatible PostgreSQL migrations with RLS
