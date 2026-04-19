# PRD-PACK-041: Scope 1-2 Complete Pack

**Pack ID:** PACK-041-scope-1-2-complete
**Category:** GHG Accounting Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-24
**Prerequisite:** None (standalone; enhanced when PACK-031/032/033 Energy Efficiency Packs and PACK-021/022/023 Net Zero Packs are present)

---

## 1. Executive Summary

### 1.1 Problem Statement

Organizations across all sectors face mandatory and voluntary greenhouse gas (GHG) emissions reporting requirements under frameworks including the GHG Protocol Corporate Standard, EU Corporate Sustainability Reporting Directive (CSRD/ESRS E1), California SB 253 Climate Corporate Data Accountability Act, SEC Climate Disclosure Rule, CDP Climate Change questionnaire, ISO 14064-1, SBTi Corporate Net-Zero Standard, and national reporting programs (UK SECR, Australia NGER, Japan TCFD-aligned). Scope 1 (direct) and Scope 2 (purchased energy) emissions form the foundation of every corporate GHG inventory, yet organizations struggle with eight persistent challenges:

1. **Fragmented source coverage**: Scope 1 emissions span 8 distinct source categories -- stationary combustion (boilers, furnaces, generators), mobile combustion (fleet vehicles, forklifts, aviation), process emissions (cement, chemicals, metals), fugitive emissions (gas distribution, equipment leaks), refrigerant & F-gas losses (HVAC, refrigeration, fire suppression), land use change, waste treatment (on-site incineration, wastewater), and agricultural emissions (enteric fermentation, manure, cropland). Most organizations capture only 2-3 of these categories, missing 15-40% of their Scope 1 footprint. Industrial facilities routinely omit fugitive emissions and process emissions; commercial buildings miss refrigerant losses; agricultural operations undercount soil N2O and enteric CH4.

2. **Scope 2 dual reporting confusion**: The GHG Protocol Scope 2 Guidance (2015) requires dual reporting of location-based and market-based Scope 2 emissions. Organizations must simultaneously track grid-average emission factors (location-based) and contractual instruments including PPAs, RECs, GOs, I-RECs, green tariffs, and supplier-specific factors (market-based). Many organizations report only one method, apply incorrect residual mix factors, double-count renewable energy certificates, or fail to properly allocate contractual instruments across facilities -- resulting in material misstatement and regulatory non-compliance.

3. **Multi-gas complexity**: Beyond CO2, GHG inventories must include CH4, N2O, HFCs, PFCs, SF6, and NF3 per the Kyoto Protocol gases. Each gas requires conversion to CO2-equivalents using Global Warming Potential (GWP) values from AR4, AR5, or AR6. Different frameworks mandate different GWP assessment reports (ESRS requires AR6; CDP accepts AR5 or AR6; EPA requires AR4 for specific programs), creating reconciliation complexity. Process and agricultural emissions involve non-CO2 gases that dominate the footprint (e.g., enteric fermentation is >95% CH4).

4. **Multi-tier methodology**: IPCC Guidelines provide Tier 1 (default emission factors), Tier 2 (country-specific factors), and Tier 3 (facility-specific measurement) methodologies. Organizations must select appropriate tiers for each source category based on data availability, materiality, and regulatory requirements. Incorrect tier selection leads to either insufficient accuracy (Tier 1 for material sources) or unjustified precision claims (Tier 3 without proper measurement infrastructure).

5. **Emission factor management**: A typical multi-site organization requires hundreds of emission factors from 10+ authoritative sources (IPCC 2006/2019, DEFRA, EPA, UBA, ADEME, ISPRA, IEA, national electricity registries, supplier-specific data). Factors must be matched by fuel type, geography, year, and methodology tier. Organizations manually curate spreadsheet-based factor databases that become outdated, contain transcription errors, and lack provenance tracking -- the #1 cause of audit findings in GHG verification.

6. **Uncertainty quantification gaps**: ISO 14064-1 Clause 7.3.3 requires uncertainty assessment. GHG Protocol Chapter 7 provides guidance on Monte Carlo simulation and analytical error propagation. Yet most organizations report point estimates without uncertainty bounds, making it impossible to assess whether year-over-year changes represent real emission reductions or measurement noise. Verification bodies increasingly flag absent uncertainty analysis as a conformity gap.

7. **Audit trail deficiencies**: GHG inventories undergo third-party verification (ISO 14064-3) and assurance (ISAE 3410 limited/reasonable assurance). Verifiers require complete audit trails from raw activity data through emission factor selection, calculation methodology, GWP conversion, and final reported figures. Spreadsheet-based inventories typically lack version control, calculation provenance, data source documentation, and assumption traceability -- extending verification timelines by 2-4 weeks and increasing assurance fees by 20-40%.

8. **Cross-framework reconciliation**: Organizations report to multiple frameworks simultaneously (GHG Protocol + ESRS + CDP + SBTi + national programs). Each framework has specific boundary requirements, consolidation approaches (equity share vs. operational control vs. financial control), emission factor preferences, and disclosure formats. Reconciling a single GHG inventory across 3-5 frameworks manually requires 80-200 hours of specialist effort per reporting cycle.

### 1.2 Solution Overview

PACK-041 is the **Scope 1-2 Complete Pack** -- the first pack in the "GHG Accounting Packs" category. It provides a unified, end-to-end solution for calculating, verifying, and reporting all Scope 1 and Scope 2 GHG emissions by orchestrating all 13 existing MRV agents (8 Scope 1 + 5 Scope 2) into an integrated pack with consolidated workflows, cross-source reconciliation, multi-framework reporting, and complete audit trail generation.

The pack orchestrates:
- **8 Scope 1 agents**: Stationary Combustion (MRV-001), Refrigerants & F-Gas (MRV-002), Mobile Combustion (MRV-003), Process Emissions (MRV-004), Fugitive Emissions (MRV-005), Land Use (MRV-006), Waste Treatment (MRV-007), Agricultural (MRV-008)
- **5 Scope 2 agents**: Location-Based (MRV-009), Market-Based (MRV-010), Steam/Heat Purchase (MRV-011), Cooling Purchase (MRV-012), Dual Reporting Reconciliation (MRV-013)

The pack adds 10 pack-level engines, 8 workflows, 10 templates, 12 integrations, and 8 presets that provide:
- Organizational boundary definition (equity share, operational control, financial control)
- Source category completeness scanning and materiality assessment
- Consolidated emission factor management across all sources
- Cross-source double-counting prevention and reconciliation
- Multi-framework compliance mapping (GHG Protocol, ESRS E1, CDP, ISO 14064, SBTi, SEC, SB 253)
- Uncertainty aggregation using Monte Carlo and analytical methods
- Complete SHA-256 provenance chain from raw data to final disclosure
- Year-over-year trend analysis with base year recalculation
- Verification-ready audit packages per ISO 14064-3 and ISAE 3410

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Spreadsheet Approach | PACK-041 Scope 1-2 Complete Pack |
|-----------|-------------------------------|-----------------------------------|
| Source category coverage | 2-3 of 8 Scope 1 categories (40-60% coverage) | All 8 Scope 1 + 5 Scope 2 categories (100% coverage) |
| Time to complete inventory | 4-12 weeks per reporting cycle | <2 days per reporting cycle (10-30x faster) |
| Cost per inventory cycle | EUR 30,000-100,000 (consultant + internal labor) | EUR 3,000-8,000 per cycle (10x reduction) |
| Emission factor accuracy | Manual lookup with transcription errors (5-15% error rate) | Automated lookup from 10+ authoritative sources (0% transcription error) |
| Scope 2 dual reporting | Often single method only; incorrect residual mix | Full dual reporting with automated instrument allocation |
| Multi-gas handling | CO2-only or incorrect GWP conversion | 7 Kyoto gases with AR4/AR5/AR6 GWP configurable |
| Uncertainty quantification | Point estimates only (no uncertainty) | Monte Carlo + analytical propagation with 95% CI |
| Audit trail | Spreadsheet-based, incomplete provenance | SHA-256 provenance chain, ISAE 3410-ready audit package |
| Multi-framework reporting | Manual reconciliation (80-200 hours) | Automated mapping to 7+ frameworks (<4 hours) |
| Year-over-year analysis | Manual base year recalculation | Automated base year recalculation per GHG Protocol |
| Verification readiness | 2-4 weeks additional prep for verifiers | Verification-ready audit package generated automatically |
| Double-counting prevention | Ad hoc checks prone to errors | Systematic cross-source reconciliation with automated flags |

### 1.4 Scope 1-2 Category Overview

**Scope 1 -- Direct Emissions (8 categories):**

| # | Category | Agent | Key Emission Sources | Primary Gases |
|---|----------|-------|---------------------|---------------|
| 1 | Stationary Combustion | MRV-001 | Boilers, furnaces, generators, heaters, turbines | CO2, CH4, N2O |
| 2 | Refrigerants & F-Gas | MRV-002 | HVAC chillers, refrigeration, fire suppression, switchgear | HFCs, PFCs, SF6, NF3 |
| 3 | Mobile Combustion | MRV-003 | Fleet vehicles, forklifts, aviation, marine vessels | CO2, CH4, N2O |
| 4 | Process Emissions | MRV-004 | Cement clinker, chemical reactions, metals processing | CO2, CH4, N2O, PFCs |
| 5 | Fugitive Emissions | MRV-005 | Natural gas distribution, coal mining, equipment leaks | CH4, CO2 |
| 6 | Land Use Change | MRV-006 | Deforestation, land conversion, soil carbon | CO2, CH4, N2O |
| 7 | Waste Treatment | MRV-007 | On-site incineration, wastewater treatment, composting | CO2, CH4, N2O |
| 8 | Agricultural | MRV-008 | Enteric fermentation, manure management, cropland N2O | CH4, N2O |

**Scope 2 -- Purchased Energy (5 categories):**

| # | Category | Agent | Key Sources | Methodology |
|---|----------|-------|-------------|-------------|
| 9 | Location-Based Electricity | MRV-009 | Grid electricity using average grid EFs | GHG Protocol Scope 2 Guidance |
| 10 | Market-Based Electricity | MRV-010 | Contractual instruments (PPAs, RECs, GOs, green tariffs) | GHG Protocol Scope 2 Guidance |
| 11 | Steam/Heat Purchase | MRV-011 | District heating, steam from CHP, industrial heat | Supplier EF or default |
| 12 | Cooling Purchase | MRV-012 | District cooling, chilled water networks | Supplier EF or default |
| 13 | Dual Reporting Reconciliation | MRV-013 | Location vs. market-based comparison | GHG Protocol requirement |

### 1.5 Target Users

**Primary:**
- Corporate sustainability managers responsible for annual GHG inventory preparation
- Environmental compliance officers reporting under mandatory disclosure regimes (CSRD, SB 253, SEC, SECR, NGER)
- Energy managers tracking Scope 1 fuel consumption and Scope 2 electricity emissions
- Sustainability consultants preparing GHG inventories for multiple clients
- GHG verification bodies performing third-party audits per ISO 14064-3

**Secondary:**
- CFOs and investor relations teams preparing climate-related financial disclosures (TCFD, ISSB S2)
- SBTi target owners tracking progress toward science-based emission reduction targets
- CDP respondents preparing annual Climate Change questionnaire responses
- ESG rating analysts requiring standardized emissions data
- Board-level sustainability committees reviewing organizational carbon footprint
- Supply chain managers requiring upstream Scope 1-2 data from suppliers

### 1.6 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Source category coverage | 100% of applicable Scope 1-2 categories | Categories assessed / total applicable categories |
| Inventory completion time | <2 days (vs. 4-12 weeks manual) | Time from data intake to verified inventory |
| Emission factor accuracy | 0% transcription error; within 2% of authoritative source | Validated against DEFRA/EPA/IEA published values |
| Scope 2 dual reporting compliance | 100% GHG Protocol Scope 2 Guidance conformance | Automated compliance check against 84 requirements |
| Uncertainty quantification | 95% CI on all material sources | Monte Carlo simulation on all Scope 1-2 totals |
| Audit finding rate | <2 findings per verification (vs. 8-15 industry average) | Third-party ISO 14064-3 verification results |
| Multi-framework mapping | 7+ frameworks from single inventory | Automated mapping to GHG Protocol, ESRS E1, CDP, ISO 14064, SBTi, SEC, SB 253 |
| Base year recalculation accuracy | 100% GHG Protocol conformance | Validated against manual recalculation test cases |
| Year-over-year variance analysis | Automated detection of >5% changes | Flagging of significant changes with root cause categorization |
| Customer NPS | >55 | Net Promoter Score survey |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| GHG Protocol Corporate Standard | WRI/WBCSD (2015, revised) | Core methodology for Scope 1-2 emissions accounting; organizational boundary, operational boundary, quantification, reporting |
| GHG Protocol Scope 2 Guidance | WRI/WBCSD (2015) | Location-based and market-based dual reporting; contractual instrument hierarchy; quality criteria |
| IPCC 2006 Guidelines (2019 Refinement) | IPCC | Tier 1/2/3 emission factors, calculation methodologies for all source categories |
| ISO 14064-1:2018 | Quantification of GHG emissions and removals | Organizational boundary, emission sources, quantification principles, uncertainty, quality management |
| ISO 14064-3:2019 | Specification for validation and verification of GHG statements | Verification-ready audit trail requirements |
| EU CSRD / ESRS E1 | Regulation 2023/2772, EFRAG (2023) | E1-4 (GHG emissions), E1-5 (energy), E1-6 (Scope 1/2/3 breakdown), transition plans |
| ISAE 3410 | IAASB (2012) | Assurance engagements on GHG statements; limited and reasonable assurance requirements |

### 2.2 Regulatory Disclosure Frameworks

| Framework | Reference | Pack Relevance |
|-----------|-----------|----------------|
| California SB 253 | Climate Corporate Data Accountability Act (2023) | Scope 1-2 reporting for entities >$1B revenue; CARB-verified data by 2026-2027 |
| SEC Climate Disclosure | SEC Final Rule (2024) | Scope 1-2 for large accelerated filers; phase-in 2025-2028 |
| UK SECR | Companies (Directors' Report) and Limited Liability Partnerships (Energy and Carbon Report) Regulations 2018 | Scope 1-2 for UK quoted companies and large unquoted companies |
| Australia NGER | National Greenhouse and Energy Reporting Act 2007 | Scope 1-2 for facilities exceeding thresholds |
| CDP Climate Change | CDP (2024) | C6 GHG emissions, C7 energy breakdown, C4 targets (Scope 1-2) |
| SBTi Corporate Framework | SBTi (2024) | Scope 1-2 near-term and long-term targets; base year tracking |
| TCFD / ISSB S2 | IFRS S2 (2023) | Scope 1-2 emissions disclosure; metrics and targets |

### 2.3 Supporting Technical Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| GHG Protocol Uncertainty Guidance | WRI/WBCSD Chapter 7 | Monte Carlo simulation, analytical propagation |
| EPA Mandatory Greenhouse Gas Reporting Rule | 40 CFR Part 98 | US-specific emission factors and methods |
| DEFRA Conversion Factors | UK Government (annual) | UK-specific emission factors for all fuel types |
| IEA Emission Factors | IEA (annual) | Country-specific electricity grid emission factors |
| EEA Monitoring Mechanism | EU Regulation 525/2013 | EU member state grid emission factors |
| AR4/AR5/AR6 GWP values | IPCC Assessment Reports | Global warming potentials for GHG-to-CO2e conversion |
| API Compendium 2009 | American Petroleum Institute | Oil and gas sector emission factors |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 10 | GHG inventory orchestration, consolidation, and reporting engines |
| Workflows | 8 | Multi-phase inventory preparation, verification, and disclosure workflows |
| Templates | 10 | Report, dashboard, and disclosure templates |
| Integrations | 12 | MRV agent bridges, data connectors, compliance mappers |
| Presets | 8 | Sector and framework-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `organizational_boundary_engine.py` | Defines the organizational boundary per GHG Protocol Chapter 3: equity share approach (proportional emissions based on % equity ownership), operational control approach (100% of operations where company has operational control), or financial control approach (100% where company has financial control). Maps legal entities, facilities, JVs, subsidiaries, and leased assets to the selected consolidation approach. Handles boundary changes for M&A, divestitures, and structural reorganizations with automatic base year recalculation triggers. |
| 2 | `source_completeness_engine.py` | Scans all 8 Scope 1 categories and 5 Scope 2 categories against the organizational boundary to identify which source categories are applicable, material (>1% of total inventory or required by regulation), and covered by available data. Performs gap analysis showing missing source categories, insufficient data quality, and recommended data collection actions. Ensures 100% source category completeness per ISO 14064-1 Clause 5.2.4 and GHG Protocol Chapter 4. Implements materiality thresholds configurable by regulation (ESRS: include all material sources; SBTi: 95% coverage). |
| 3 | `emission_factor_manager_engine.py` | Consolidated emission factor management across all 13 MRV agents. Maintains a unified factor database drawing from IPCC 2006/2019, DEFRA (annual), EPA (eGRID, 40 CFR 98), UBA (DE), ADEME (FR), ISPRA (IT), IEA (electricity), and supplier-specific data. Provides factor selection logic based on fuel type, geography, year, tier level, and regulatory preference. Tracks factor provenance (source, version, publication date, access date). Supports factor override with audit trail. Validates factor consistency across agents to prevent double-counting or inconsistent factor application. |
| 4 | `scope1_consolidation_engine.py` | Aggregates Scope 1 emissions from all 8 MRV agents (001-008) into a consolidated Scope 1 total. Applies organizational boundary proportions (equity share %, operational control 100%). Performs per-gas aggregation (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3) with configurable GWP conversion (AR4, AR5, AR6). Detects and resolves double-counting between overlapping categories (e.g., on-site waste incineration could appear in both stationary combustion and waste treatment). Produces facility-level, entity-level, and organization-level Scope 1 rollups with full breakdown by source category and gas. |
| 5 | `scope2_consolidation_engine.py` | Aggregates Scope 2 emissions from all 5 Scope 2 agents (009-013) into consolidated location-based and market-based totals. Manages contractual instrument allocation hierarchy per GHG Protocol Scope 2 Guidance: (1) energy attribute certificates, (2) direct contracts/PPAs, (3) supplier-specific EFs, (4) residual mix, (5) grid average. Prevents double-counting of renewable energy claims. Reconciles location-based vs. market-based totals with variance analysis. Produces dual-reported Scope 2 figures at facility, entity, and organization levels. |
| 6 | `uncertainty_aggregation_engine.py` | Aggregates uncertainty estimates from all 13 MRV agents into organization-level uncertainty bounds. Supports two methods: (1) Analytical propagation using quadrature (root-sum-of-squares) for independent sources, with correlation handling for dependent sources; (2) Monte Carlo simulation (10,000+ iterations) sampling from per-source uncertainty distributions to produce organization-level 95% confidence intervals. Identifies the top contributors to overall uncertainty and recommends data quality improvements for maximum uncertainty reduction. Outputs uncertainty breakdown by scope, source category, and facility. |
| 7 | `base_year_recalculation_engine.py` | Implements GHG Protocol base year recalculation per Chapter 5. Triggers recalculation when: (1) structural changes exceed significance threshold (M&A, divestitures >5% of base year emissions), (2) methodology changes (emission factor updates, tier upgrades, calculation corrections), (3) source category additions or removals, (4) error corrections. Tracks base year emissions for each source category and facility. Applies adjustments maintaining like-for-like comparability. Produces recalculated base year inventory with full audit trail showing each adjustment and its justification. |
| 8 | `trend_analysis_engine.py` | Year-over-year emission trend analysis comparing current reporting year to base year and previous years. Decomposes changes into contributing factors: activity level changes (production volume, headcount, floor area), emission intensity changes (efficiency improvements, fuel switching), structural changes (M&A, divestitures), and methodology changes (factor updates, tier changes). Calculates both absolute and intensity metrics (tCO2e/revenue, tCO2e/FTE, tCO2e/m2, tCO2e/unit produced). Produces Kaya identity decomposition for energy-related emissions. Flags statistically significant changes using uncertainty bounds. |
| 9 | `compliance_mapping_engine.py` | Maps the consolidated GHG inventory to the disclosure requirements of 7+ frameworks simultaneously. For each framework, validates boundary completeness, methodology conformance, required breakdowns (by gas, by scope, by geography, by source category), uncertainty requirements, base year requirements, and format specifications. Generates framework-specific gap analysis with remediation actions. Produces compliance readiness scores (0-100) per framework. Maintains a regulatory requirement database with 500+ individual requirements mapped across all supported frameworks. |
| 10 | `inventory_reporting_engine.py` | Generates comprehensive GHG inventory reports and verification packages. Produces: (1) Internal management report with executive summary, detailed emissions tables, trend analysis, and action items; (2) External disclosure reports formatted for each target framework (ESRS E1, CDP, SBTi, SEC); (3) Verification package per ISO 14064-3 including data sources, calculation methodologies, emission factor provenance, uncertainty analysis, base year reconciliation, and completeness statement; (4) Data export in GHG Protocol reporting template format, XBRL (for SEC), and CSV. All outputs include SHA-256 provenance hash chain. |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `boundary_definition_workflow.py` | 4: EntityMapping -> BoundarySelection -> SourceIdentification -> MaterialityAssessment | Establish organizational boundary and identify all applicable emission source categories |
| 2 | `data_collection_workflow.py` | 4: DataRequirements -> DataIngestion -> QualityAssessment -> GapResolution | Systematic data collection across all Scope 1-2 categories with quality validation |
| 3 | `scope1_calculation_workflow.py` | 4: SourceCategoryRouting -> AgentExecution -> ResultConsolidation -> CrossSourceReconciliation | Execute all 8 Scope 1 agent calculations and consolidate results |
| 4 | `scope2_calculation_workflow.py` | 4: InstrumentCollection -> DualMethodExecution -> AllocationReconciliation -> DualReportGeneration | Execute dual-method Scope 2 calculation with instrument allocation |
| 5 | `inventory_consolidation_workflow.py` | 4: Scope1Aggregation -> Scope2Aggregation -> UncertaintyPropagation -> TotalInventoryGeneration | Consolidate all scopes into total inventory with uncertainty bounds |
| 6 | `verification_preparation_workflow.py` | 4: AuditTrailCompilation -> ProvenanceVerification -> CompletenessCheck -> VerificationPackageGeneration | Prepare ISO 14064-3 verification-ready audit package |
| 7 | `disclosure_generation_workflow.py` | 4: FrameworkMapping -> TemplatePopulation -> ComplianceValidation -> OutputGeneration | Generate multi-framework disclosure reports from single inventory |
| 8 | `full_inventory_workflow.py` | 8: BoundarySetup -> DataCollection -> Scope1Calc -> Scope2Calc -> Consolidation -> TrendAnalysis -> Verification -> Disclosure | Complete end-to-end GHG inventory workflow |

### 3.4 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `ghg_inventory_report.py` | MD, HTML, PDF, JSON | Complete GHG inventory report with Scope 1-2 breakdowns by category, gas, facility, and entity |
| 2 | `scope1_detailed_report.py` | MD, HTML, PDF, JSON | Detailed Scope 1 report with per-category emission tables, methodology descriptions, and emission factor citations |
| 3 | `scope2_dual_report.py` | MD, HTML, PDF, JSON | Scope 2 dual-method report showing location-based and market-based results with contractual instrument details |
| 4 | `emission_factor_registry.py` | MD, HTML, PDF, JSON | Complete emission factor registry with source, version, value, geography, and provenance for every factor used |
| 5 | `uncertainty_analysis_report.py` | MD, HTML, PDF, JSON | Uncertainty analysis with Monte Carlo results, sensitivity analysis, top uncertainty contributors, and data quality recommendations |
| 6 | `trend_analysis_report.py` | MD, HTML, PDF, JSON | Year-over-year trend analysis with decomposition, intensity metrics, base year comparison, and SBTi trajectory |
| 7 | `verification_package.py` | MD, HTML, PDF, JSON | ISO 14064-3 verification package with data provenance, calculation audit trail, completeness statement, and methodology documentation |
| 8 | `executive_summary_report.py` | MD, HTML, PDF, JSON | 2-4 page executive summary: total Scope 1-2 emissions, key changes, compliance status, recommended actions |
| 9 | `compliance_dashboard.py` | MD, HTML, JSON | Multi-framework compliance dashboard showing readiness scores, gap analysis, and remediation actions per framework |
| 10 | `esrs_e1_disclosure.py` | MD, HTML, PDF, JSON, XBRL | ESRS E1 climate change disclosure template with all required datapoints (E1-4, E1-5, E1-6) pre-populated from inventory |

### 3.5 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 12-phase DAG pipeline: BoundarySetup -> DataIngestion -> Scope1-StationaryCombustion -> Scope1-Refrigerants -> Scope1-MobileCombustion -> Scope1-OtherCategories -> Scope2-DualMethod -> Consolidation -> UncertaintyAggregation -> TrendAnalysis -> ComplianceMapping -> ReportGeneration. Parallel execution for independent Scope 1 categories. Retry with exponential backoff, SHA-256 provenance chain, phase-level caching. |
| 2 | `mrv_scope1_bridge.py` | Routes to all 8 Scope 1 agents (MRV-001 through MRV-008). Provides unified interface for triggering calculations, collecting results, and monitoring execution status. Handles agent-specific configuration passing (fuel types, equipment registries, fleet data, process parameters). Translates pack-level organizational boundary into agent-level facility scope. |
| 3 | `mrv_scope2_bridge.py` | Routes to all 5 Scope 2 agents (MRV-009 through MRV-013). Manages dual-method execution, instrument allocation coordination between MRV-010 and MRV-013, and steam/heat/cooling consolidation from MRV-011 and MRV-012. Ensures consistent grid emission factor application across location-based calculations. |
| 4 | `data_bridge.py` | Routes to DATA agents: DATA-001 (PDF extraction for invoices/fuel receipts), DATA-002 (Excel/CSV for utility bills and meter data), DATA-003 (ERP for procurement and fuel purchase data), DATA-004 (API for real-time meter feeds), DATA-010 (Data Quality Profiler), DATA-013 (Outlier Detection for activity data), DATA-014 (Time Series Gap Filler for consumption data), DATA-018 (Data Lineage Tracker). |
| 5 | `foundation_bridge.py` | Routes to Foundation agents: FOUND-001 (Orchestrator for DAG execution), FOUND-002 (Schema validation), FOUND-003 (Unit normalization for fuel units, energy units, mass units), FOUND-004 (Assumptions Registry for default values), FOUND-005 (Citations for emission factor sourcing), FOUND-006 (Access control), FOUND-008 (Reproducibility verification), FOUND-010 (Observability). |
| 6 | `energy_efficiency_bridge.py` | Integration with Energy Efficiency Packs (PACK-031 through PACK-040). Imports energy audit findings, building assessments, quick wins, ISO 50001 data, benchmark results, utility analysis, and M&V verified savings. Links emission reductions from energy efficiency measures to GHG inventory changes, enabling verified reporting of Scope 1-2 reductions from energy efficiency programs. |
| 7 | `net_zero_bridge.py` | Integration with Net Zero Packs (PACK-021 through PACK-030). Provides Scope 1-2 baseline data for net-zero pathway planning. Imports target trajectories and tracks actual emission reductions against SBTi commitments. Bi-directional: net-zero packs set reduction targets; inventory pack reports actual performance against targets. |
| 8 | `erp_connector.py` | Enterprise resource planning system integration for automated activity data extraction: fuel purchase volumes from procurement, fleet mileage from transport management, electricity consumption from facility management, refrigerant purchases from maintenance records, production volumes from manufacturing execution systems. Supports SAP, Oracle, Microsoft Dynamics, and generic REST/CSV interfaces. |
| 9 | `utility_data_bridge.py` | Utility bill and meter data integration: automated extraction of electricity consumption (kWh), gas consumption (therms/m3), steam/heat/cooling consumption from utility providers. Supports manual upload (PDF/Excel), automated API feeds, smart meter data, and sub-metering aggregation. Handles unit conversion, estimated billing correction, and multi-site consolidation. |
| 10 | `health_check.py` | 22-category system verification covering all 10 engines, 8 workflows, 13 MRV agent connectivity, database connectivity, cache status, Foundation agent bridges, DATA agent bridges, energy efficiency pack bridges, net-zero pack bridges, ERP connectivity, utility data freshness, and authentication/authorization. |
| 11 | `setup_wizard.py` | 8-step guided inventory configuration: (1) organizational structure (legal entities, facilities, JVs), (2) consolidation approach selection (equity share, operational control, financial control), (3) source category materiality assessment, (4) data source registration (ERP, utility accounts, meters), (5) emission factor preferences (source hierarchy, GWP selection), (6) reporting framework selection, (7) base year configuration, (8) reporting preferences (intensity metrics, breakdowns). |
| 12 | `alert_bridge.py` | Alert and notification integration: data collection deadline reminders, emission factor update notifications, base year recalculation triggers, compliance deadline warnings, verification schedule reminders, anomaly detection when emissions deviate >10% from expected, and data quality deterioration alerts. Supports email, SMS, webhook, and in-app notification channels. |

### 3.6 Presets

| # | Preset | Sector | Key Characteristics |
|---|--------|--------|---------------------|
| 1 | `corporate_office.yaml` | Commercial Office | Scope 1: primarily stationary combustion (heating) and refrigerants (HVAC); Scope 2: electricity-dominant (60-80% of footprint). Simple inventory with 2-3 Scope 1 categories. Focus on Scope 2 market-based optimization through RE procurement. Typical: 50-500 tCO2e Scope 1, 200-5,000 tCO2e Scope 2. |
| 2 | `manufacturing.yaml` | Manufacturing (general) | Scope 1: stationary combustion (process heat), process emissions, mobile (forklifts/trucks), fugitive. Scope 2: electricity for motors, compressed air, lighting. Multi-category Scope 1 with potential process emissions. Typical: 5,000-500,000 tCO2e. Production-normalized intensity metrics. |
| 3 | `energy_utility.yaml` | Energy / Utilities | Scope 1: dominant stationary combustion (power generation), fugitive (gas distribution). Scope 2: minimal (self-generated). Highest absolute emissions. EPA 40 CFR 98 compliance. CEMS data integration. Typical: 100,000-10,000,000 tCO2e. |
| 4 | `transport_logistics.yaml` | Transport & Logistics | Scope 1: dominant mobile combustion (fleet), refrigerants (refrigerated transport). Scope 2: warehouse/depot electricity. Fleet-centric inventory with vehicle-by-vehicle tracking. Typical: 10,000-1,000,000 tCO2e. Distance-normalized metrics. |
| 5 | `food_agriculture.yaml` | Food & Agriculture | Scope 1: agricultural emissions (enteric, manure, cropland), stationary combustion, refrigerants (cold chain). Scope 2: processing electricity. Agricultural emissions require IPCC Tier 1/2 methodology. Typical: 5,000-500,000 tCO2e. Land-area normalized metrics. |
| 6 | `real_estate.yaml` | Real Estate / Property | Scope 1: stationary combustion (heating), refrigerants. Scope 2: electricity, district heating/cooling. Portfolio approach with per-building and per-m2 metrics. GRESB-aligned reporting. Typical: 1,000-100,000 tCO2e. Floor-area intensity metrics. |
| 7 | `healthcare.yaml` | Healthcare | Scope 1: stationary combustion (heating/sterilization), refrigerants (medical/HVAC), mobile (ambulances), waste treatment (medical waste incineration). Scope 2: 24/7 electricity for critical systems. NHSF Net Zero requirements. Typical: 5,000-200,000 tCO2e. Per-bed and per-patient-day metrics. |
| 8 | `sme_simplified.yaml` | SME (any sector, <250 employees) | Simplified 5-engine flow: boundary (auto-detect), Scope 1 (stationary + mobile only), Scope 2 (electricity only), consolidation, reporting. Pre-populated emission factors. Guided walkthrough. Skip agricultural/process/fugitive unless indicated. Typical: 50-5,000 tCO2e. Reduced data requirements. |

---

## 4. Engine Specifications

### 4.1 Engine 1: Organizational Boundary Engine

**Purpose:** Define the organizational boundary per GHG Protocol Chapter 3, determining which facilities, subsidiaries, JVs, and leased assets are included and at what percentage.

**Consolidation Approaches:**

| Approach | Rule | GHG Protocol Ref | Typical Use |
|----------|------|-------------------|-------------|
| Equity Share | Include emissions proportional to equity ownership % | Chapter 3.1 | JVs, partial ownership |
| Operational Control | 100% of operations where company has operational control | Chapter 3.2 | Most common; required by ESRS |
| Financial Control | 100% where company has financial control (consolidates per accounting rules) | Chapter 3.3 | Aligns with financial reporting |

**Entity Classification:**

| Entity Type | Equity Share Treatment | Operational Control Treatment |
|-------------|----------------------|------------------------------|
| Wholly-owned subsidiary | 100% | 100% |
| Majority-owned subsidiary | Ownership % | 100% (if operational control) |
| Joint venture | Ownership % | 100% if operator; 0% if not |
| Joint operation | Ownership % | 100% if operator; 0% if not |
| Associate | Ownership % | 0% (no operational control) |
| Leased asset (finance lease) | 100% | Depends on control terms |
| Leased asset (operating lease) | 0% (lessor includes) | 100% if operational control |
| Franchise | 0% (franchisee includes) | Depends on franchise terms |

**Boundary Change Handling:**

| Change Type | Action | Base Year Impact |
|-------------|--------|-----------------|
| Acquisition (>5% of base year) | Add acquired emissions | Recalculate base year |
| Divestiture (>5% of base year) | Remove divested emissions | Recalculate base year |
| Merger | Combine both entities' emissions | Recalculate base year |
| Organic growth | Include in current year | No base year change |
| Outsourcing/insourcing | Shift between Scope 1 and Scope 3 | Recalculate if material |

**Key Models:**
- `OrganizationStructure` - Legal entities, facilities, ownership percentages, control relationships
- `BoundaryDefinition` - Consolidation approach, included entities with percentages, excluded entities with justification
- `BoundaryChange` - Change type, effective date, affected entities, emission impact, base year recalculation flag
- `FacilityProfile` - Facility ID, name, location, sector, applicable source categories, data sources

**Non-Functional Requirements:**
- Boundary calculation for 1,000 entities: <30 seconds
- Boundary change impact assessment: <60 seconds
- Reproducibility: bit-perfect (SHA-256 verified)

### 4.2 Engine 2: Source Completeness Engine

**Purpose:** Ensure 100% source category coverage per ISO 14064-1 and GHG Protocol requirements.

**Completeness Scanning:**

| Scan Type | Method | Output |
|-----------|--------|--------|
| Category applicability | Map facility type + sector to expected source categories | Applicable/not-applicable per category |
| Data availability | Check available data sources per facility per category | Available/partial/missing data status |
| Materiality assessment | Estimate emissions per category using sector benchmarks | Material (>1% or regulatory requirement) / immaterial |
| Gap analysis | Compare applicable+material vs. data-available | Gap report with remediation actions |

**Sector-Category Mapping:**

| Sector | Stationary | Mobile | Process | Fugitive | Refrigerant | Land Use | Waste | Ag | Scope 2 |
|--------|-----------|--------|---------|----------|------------|----------|-------|----|---------|
| Office | Yes | Maybe | No | No | Yes | No | No | No | Yes |
| Manufacturing | Yes | Yes | Maybe | Maybe | Yes | No | Maybe | No | Yes |
| Energy | Yes | Maybe | No | Yes | Maybe | No | No | No | Maybe |
| Transport | Maybe | Yes | No | No | Yes | No | No | No | Yes |
| Agriculture | Yes | Yes | No | No | Yes | Maybe | Maybe | Yes | Yes |
| Healthcare | Yes | Yes | No | No | Yes | No | Yes | No | Yes |

**Key Models:**
- `CompletenessInput` - Organization structure, facility profiles, available data sources, sector classifications
- `CompletenessResult` - Per-category applicability, materiality, data availability, gap analysis, remediation actions
- `MaterialityAssessment` - Category, estimated emissions (tCO2e), percentage of total, materiality classification, regulatory requirement flag
- `DataGap` - Category, facility, missing data type, data quality score, remediation action, priority

### 4.3 Engine 3: Emission Factor Manager Engine

**Purpose:** Unified emission factor management across all 13 MRV agents.

**Factor Source Hierarchy (default):**

| Priority | Source | Coverage | Update Frequency |
|----------|--------|----------|-----------------|
| 1 | Facility-specific measurement (CEMS, metering) | Site-specific | Continuous |
| 2 | Supplier-specific data (invoices, certificates) | Supplier-specific | Annual |
| 3 | Country-specific (DEFRA, UBA, ADEME, ISPRA, EPA) | National | Annual |
| 4 | Regional (IEA, EEA, eGRID) | Regional | Annual |
| 5 | IPCC default (2006/2019 Guidelines) | Global | Periodic |

**Factor Database Coverage:**

| Factor Category | Count | Sources | Example |
|----------------|-------|---------|---------|
| Stationary combustion (fuel EFs) | 200+ | IPCC, DEFRA, EPA, UBA | Natural gas: 56.1 kgCO2/GJ (IPCC) |
| Mobile combustion (fuel + vehicle EFs) | 300+ | DEFRA, EPA, UBA | Diesel car: 0.168 kgCO2/km (DEFRA) |
| Electricity grid (location-based) | 500+ | IEA, eGRID, EEA, national | Germany grid: 0.385 kgCO2/kWh (UBA 2024) |
| Electricity residual mix (market-based) | 200+ | AIB, national registries | EU residual mix: 0.420 kgCO2/kWh (AIB) |
| Refrigerant GWP | 100+ | AR4, AR5, AR6 | R-410A: 2088 (AR6) |
| Process emission factors | 150+ | IPCC, EPA, sector | Cement clinker: 0.525 tCO2/t clinker |
| Agricultural factors | 100+ | IPCC, national | Dairy cattle enteric: 128 kgCH4/head/yr (IPCC Tier 1) |
| Steam/heat/cooling | 50+ | DEFRA, national | District heat: 0.170 kgCO2/kWh (DEFRA) |

**GWP Conversion:**

| Gas | AR4 (2007) | AR5 (2014) | AR6 (2021) |
|-----|-----------|-----------|-----------|
| CO2 | 1 | 1 | 1 |
| CH4 | 25 | 28 | 27.9 (fossil) / 27.0 (biogenic) |
| N2O | 298 | 265 | 273 |
| SF6 | 22,800 | 23,500 | 25,200 |
| NF3 | 17,200 | 16,100 | 17,400 |

**Key Models:**
- `EmissionFactorRequest` - Fuel/activity type, geography, year, tier, preferred source
- `EmissionFactor` - Value (kgCO2e/unit), gas breakdown (CO2, CH4, N2O), source, version, provenance hash
- `FactorOverride` - Original factor, override value, justification, approver, effective date
- `FactorConsistencyCheck` - Cross-agent factor comparison, discrepancy flags, resolution

### 4.4 Engine 4: Scope 1 Consolidation Engine

**Purpose:** Aggregate Scope 1 emissions from all 8 MRV agents into a consolidated organizational total.

**Consolidation Logic:**

```
Scope1_total = Sum over all facilities f:
  boundary_pct(f) * Sum over categories c in [Stationary, Mobile, Process, Fugitive, Refrigerant, LandUse, Waste, Ag]:
    Sum over gases g in [CO2, CH4, N2O, HFCs, PFCs, SF6, NF3]:
      activity_data(f,c,g) * emission_factor(f,c,g) * GWP(g)
```

**Double-Counting Prevention:**

| Overlap Risk | Categories | Resolution |
|-------------|-----------|------------|
| On-site waste incineration | Stationary Combustion + Waste Treatment | Assign to Waste Treatment if waste is primary fuel; Stationary if energy is primary purpose |
| CHP systems | Stationary Combustion + Steam/Heat | Assign fuel to Stationary; allocate outputs via CHP allocation method |
| Biogas from waste | Waste Treatment + Stationary Combustion | Assign methane capture to Waste; combustion to Stationary |
| Fleet fuel | Mobile Combustion + Stationary (generators) | Classify by equipment type: vehicles = Mobile, generators = Stationary |

**Key Models:**
- `Scope1Input` - Per-facility, per-category results from all 8 agents, boundary percentages, GWP selection
- `Scope1Result` - Consolidated total (tCO2e), breakdown by category, gas, facility, entity; double-counting flags
- `CategoryResult` - Category name, agent source, emissions per gas, total CO2e, methodology tier, uncertainty
- `ConsolidationAuditTrail` - Per-facility, per-category contributions, boundary adjustments, double-counting resolutions

### 4.5 Engine 5: Scope 2 Consolidation Engine

**Purpose:** Aggregate Scope 2 emissions with dual-method reporting and instrument allocation.

**Instrument Allocation Hierarchy (GHG Protocol Scope 2 Guidance):**

| Priority | Instrument Type | Description |
|----------|----------------|-------------|
| 1 | Energy Attribute Certificates | Bundled GOs/RECs/I-RECs matching consumption |
| 2 | Direct contracts (PPAs) | Power purchase agreements with specified generation |
| 3 | Supplier-specific EFs | Utility-disclosed generation mix |
| 4 | Residual mix | Grid mix minus allocated certificates |
| 5 | Grid average (location-based only) | National/regional average grid factor |

**Dual Reporting Structure:**

```
Scope2_location = Sum over facilities f:
  boundary_pct(f) * Sum over energy_types e:
    consumption(f,e) * grid_average_EF(location(f), e)

Scope2_market = Sum over facilities f:
  boundary_pct(f) * Sum over energy_types e:
    (consumption_covered_by_instruments(f,e) * instrument_EF(f,e)) +
    (consumption_uncovered(f,e) * residual_mix_EF(location(f), e))
```

**Key Models:**
- `Scope2Input` - Per-facility consumption data, instrument portfolio, grid factors, residual mix factors
- `Scope2Result` - Location-based total, market-based total, per-facility dual results, instrument allocation detail
- `InstrumentAllocation` - Instrument type, volume (MWh), EF, facility allocation, remaining unallocated
- `DualReportingReconciliation` - Location-based vs. market-based comparison, variance, driver analysis

### 4.6 Engine 6: Uncertainty Aggregation Engine

**Purpose:** Organization-level uncertainty quantification per GHG Protocol Chapter 7 and ISO 14064-1 Clause 7.3.3.

**Analytical Method (Quadrature):**

```
U_total = sqrt(Sum over sources i: (emissions_i * u_i)^2) / total_emissions * 100%

Where:
  emissions_i = emissions from source i (tCO2e)
  u_i = relative uncertainty for source i (%)
  total_emissions = sum of all emissions
```

**Monte Carlo Method:**

```
For iteration k = 1 to N (N >= 10,000):
  For each source i:
    sample activity_data_i from distribution(activity_data_i, u_activity_i)
    sample emission_factor_i from distribution(EF_i, u_EF_i)
    emissions_i_k = activity_data_i * emission_factor_i * GWP
  total_k = sum(emissions_i_k)

95% CI = [percentile(total, 2.5%), percentile(total, 97.5%)]
```

**Typical Uncertainty Ranges:**

| Source Category | Activity Data Uncertainty | Emission Factor Uncertainty | Combined |
|----------------|--------------------------|----------------------------|----------|
| Stationary combustion (metered fuel) | 1-3% | 1-5% | 2-6% |
| Mobile combustion (fuel card data) | 2-5% | 3-5% | 4-7% |
| Refrigerants (purchase records) | 10-30% | 5-10% | 11-32% |
| Electricity (utility bills) | 1-2% | 5-30% (grid EF) | 5-30% |
| Process emissions | 5-15% | 10-30% | 11-34% |
| Fugitive emissions | 20-50% | 50-200% | 54-206% |

**Key Models:**
- `UncertaintyInput` - Per-source emissions, activity data uncertainty, EF uncertainty, distribution types
- `UncertaintyResult` - Analytical CI, Monte Carlo CI, top contributors, data quality improvement recommendations
- `MonteCarloResult` - Distribution histogram, percentiles (5%, 25%, 50%, 75%, 95%), convergence check
- `UncertaintyContributor` - Source category, contribution to total uncertainty, recommended improvement

### 4.7 Engine 7: Base Year Recalculation Engine

**Purpose:** Maintain comparability across reporting years per GHG Protocol Chapter 5.

**Recalculation Triggers:**

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Acquisition | >5% of base year emissions (configurable) | Add acquired entity's base year emissions |
| Divestiture | >5% of base year emissions (configurable) | Remove divested entity's base year emissions |
| Methodology change | Any material change | Recalculate affected sources with new methodology |
| Error correction | Any identified error | Correct and document |
| Source category change | Addition or removal | Add/remove category in base year |

**Recalculation Process:**

```
1. Identify trigger event and affected scope
2. Retrieve original base year data for affected entities/categories
3. Apply structural adjustment (add/remove/modify)
4. Recalculate base year total with adjustment
5. Document adjustment: original value, adjusted value, trigger, justification
6. Update all year-over-year comparisons
7. Generate recalculation audit trail
```

**Key Models:**
- `BaseYearConfig` - Base year, significance threshold, original emissions per scope/category/facility
- `RecalculationTrigger` - Trigger type, affected entities, emission impact estimate, justification
- `RecalculationResult` - Original base year, recalculated base year, adjustments applied, audit trail
- `BaseYearComparison` - Year, original emissions, base year emissions, absolute change, percentage change

### 4.8 Engine 8: Trend Analysis Engine

**Purpose:** Year-over-year emission trend analysis with decomposition.

**Kaya Identity Decomposition:**

```
CO2 = Population * (GDP/Population) * (Energy/GDP) * (CO2/Energy)
    = Population * Affluence * Energy_Intensity * Carbon_Intensity

For corporate context:
CO2 = Activity * Activity_Intensity * Energy_Intensity * Emission_Intensity

Where:
  Activity = production volume, revenue, headcount, floor area
  Activity_Intensity = unit activity per business metric
  Energy_Intensity = energy per unit activity
  Emission_Intensity = emissions per unit energy
```

**Change Decomposition:**

| Factor | Description | Example |
|--------|-------------|---------|
| Activity level change | Change in production, occupancy, fleet size | +10% production = +10% expected emissions |
| Emission intensity change | Efficiency improvements, fuel switching | LED lighting reduces kWh/m2 |
| Structural change | M&A, outsourcing, boundary changes | Acquisition of new facility |
| Methodology change | Factor updates, tier upgrades | Updated grid EF reduces Scope 2 |
| Weather normalization | Degree-day correction for heating/cooling | Mild winter reduces heating emissions |

**Intensity Metrics:**

| Metric | Unit | Applicability |
|--------|------|---------------|
| tCO2e / EUR million revenue | Carbon intensity per revenue | All sectors |
| tCO2e / FTE | Carbon intensity per employee | Office, services |
| tCO2e / m2 floor area | Carbon intensity per area | Real estate, retail |
| tCO2e / unit produced | Carbon intensity per product | Manufacturing |
| tCO2e / tonne-km | Carbon intensity per transport | Logistics |
| tCO2e / patient-day | Carbon intensity per service | Healthcare |

**Key Models:**
- `TrendInput` - Multi-year emission data, activity data, base year, intensity denominators
- `TrendResult` - Year-over-year changes (absolute, %), decomposition factors, intensity trends, SBTi alignment
- `DecompositionResult` - Per-factor contribution to year-over-year change (tCO2e and %)
- `IntensityMetric` - Metric name, numerator (tCO2e), denominator (activity), value, trend

### 4.9 Engine 9: Compliance Mapping Engine

**Purpose:** Map consolidated inventory to 7+ disclosure framework requirements.

**Framework Requirement Database (500+ requirements):**

| Framework | Requirement Count | Key Requirements |
|-----------|------------------|------------------|
| GHG Protocol | 85 | Boundary, completeness, consistency, transparency, accuracy, base year |
| ESRS E1 | 72 | E1-4 targets, E1-5 energy, E1-6 Scope 1/2/3 breakdown, transition plan |
| CDP Climate Change | 95 | C1-C15 modules; C6 emissions, C7 energy, C4 targets |
| ISO 14064-1 | 60 | Clauses 5-9: boundary, quantification, reporting, quality management |
| SBTi | 45 | Target setting, Scope 1-2 coverage, base year, progress tracking |
| SEC Climate Disclosure | 40 | S-K 1500-1507; Scope 1-2 for accelerated filers |
| SB 253 (California) | 35 | Scope 1-2-3 for >$1B revenue; CARB verification |

**Compliance Scoring:**

```
Score_framework = (requirements_met / total_requirements) * 100

Classification:
  >= 95%: "Compliant - Ready for submission"
  >= 80%: "Substantially compliant - Minor gaps"
  >= 60%: "Partially compliant - Material gaps"
  < 60%:  "Non-compliant - Significant remediation needed"
```

**Key Models:**
- `ComplianceInput` - Consolidated inventory, reporting year, selected frameworks, organizational details
- `ComplianceResult` - Per-framework score, gap analysis, remediation actions, submission readiness
- `FrameworkRequirement` - Requirement ID, framework, description, data field, validation rule, mandatory/optional
- `ComplianceGap` - Framework, requirement, current status, gap description, remediation action, priority

### 4.10 Engine 10: Inventory Reporting Engine

**Purpose:** Generate comprehensive reports and verification packages.

**Report Types:**

| Report | Audience | Length | Frequency |
|--------|----------|--------|-----------|
| Executive Summary | Board, C-suite | 2-4 pages | Annual + quarterly |
| GHG Inventory Report | Sustainability team, verifiers | 30-60 pages | Annual |
| Scope 1 Detailed | Technical team, verifiers | 20-40 pages | Annual |
| Scope 2 Dual Report | Technical team, verifiers | 15-25 pages | Annual |
| Verification Package | External verifiers (ISO 14064-3) | 50-100 pages | Annual |
| ESRS E1 Disclosure | CSRD reporting | Per EFRAG template | Annual |
| CDP Response | CDP submission | Per CDP template | Annual |
| Compliance Dashboard | Management | 5-10 pages | Quarterly |

**Verification Package Contents (ISO 14064-3):**

| Section | Contents |
|---------|----------|
| Organization description | Legal structure, boundaries, consolidation approach |
| GHG inventory summary | Total Scope 1-2 by category, gas, facility |
| Methodology | Per-category calculation method, tier selection justification |
| Emission factor provenance | Every EF used: value, source, version, access date, SHA-256 hash |
| Activity data evidence | Data sources, collection methods, quality assessment |
| Uncertainty analysis | Per-source and aggregate uncertainty with methodology |
| Base year | Base year emissions, recalculations applied, justification |
| Completeness statement | Source categories assessed, excluded categories with justification |
| Calculation audit trail | Full SHA-256 provenance chain from raw data to reported values |
| Quality management | Data quality procedures, review process, improvement plan |

**Key Models:**
- `ReportInput` - Consolidated inventory, trend data, compliance scores, uncertainty results, verification data
- `ReportOutput` - Generated report content in requested formats, provenance hash, generation metadata
- `VerificationPackage` - All sections listed above with cross-references and provenance chain
- `DisclosureOutput` - Framework-specific formatted output (ESRS XBRL, CDP XML, SBTi template)

---

## 5. Workflow Specifications

### 5.1 Workflow 1: Boundary Definition Workflow

**Purpose:** Establish organizational boundary and identify all applicable emission source categories.

**Phase 1: Entity Mapping**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Import organizational structure | Legal entity list, ownership percentages, control relationships | Organization structure model | <10 min |
| 1.2 | Map facilities to entities | Facility registry, entity assignments | Facility-entity mapping | <5 min |
| 1.3 | Classify entity types | Entity characteristics, accounting treatment | Entity classification (subsidiary, JV, associate, etc.) | <5 min |

**Phase 2: Boundary Selection**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Select consolidation approach | Regulatory requirements, user preference | Equity share / operational control / financial control | <2 min |
| 2.2 | Calculate inclusion percentages | Entity classifications, ownership %, control assessment | Per-entity inclusion percentage | <5 min |
| 2.3 | Handle boundary exceptions | Leased assets, franchises, JVs | Exception treatment with justification | <10 min |

**Phase 3: Source Identification**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Map sector to source categories | Facility sector classifications | Applicable source category list | <2 min (auto) |
| 3.2 | Assess data availability | Available data sources per facility | Data availability matrix | <5 min (auto) |
| 3.3 | Identify applicable MRV agents | Source categories vs. available agents | Agent activation list (which of 13 agents to run) | <1 min (auto) |

**Phase 4: Materiality Assessment**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Estimate per-category emissions | Sector benchmarks, facility profiles | Preliminary emission estimates | <5 min (auto) |
| 4.2 | Apply materiality thresholds | Estimated emissions, regulatory requirements | Material/immaterial classification | <2 min (auto) |
| 4.3 | Generate completeness report | All assessments | Completeness report with gaps and actions | <3 min (auto) |

**Acceptance Criteria:**
- [ ] All 4 phases execute sequentially with data passing
- [ ] Consolidation approach correctly classifies all entity types
- [ ] Source category mapping covers all 13 categories across all facility types
- [ ] Materiality assessment uses configurable thresholds (default 1%, regulatory override)
- [ ] Total workflow duration <45 minutes for 100 entities

### 5.2 Workflow 2: Data Collection Workflow

**Purpose:** Systematic data collection across all Scope 1-2 categories with quality validation.

**Phases:** DataRequirements -> DataIngestion -> QualityAssessment -> GapResolution

**Acceptance Criteria:**
- [ ] Data requirements generated per source category and facility
- [ ] Multiple ingestion formats supported (PDF, Excel, CSV, API, ERP)
- [ ] Quality assessment scores each data source (0-100)
- [ ] Gap resolution provides specific remediation actions
- [ ] Total workflow duration <4 hours for 50 facilities

### 5.3 Workflow 3: Scope 1 Calculation Workflow

**Purpose:** Execute all 8 Scope 1 agent calculations and consolidate results.

**Phases:** SourceCategoryRouting -> AgentExecution -> ResultConsolidation -> CrossSourceReconciliation

**Phase 2: Agent Execution (Parallel)**

| Agent | Trigger Condition | Input | Output |
|-------|-------------------|-------|--------|
| MRV-001 Stationary Combustion | Fuel consumption data available | Fuel type, quantity, equipment | tCO2e by gas |
| MRV-002 Refrigerants & F-Gas | Refrigerant data available | Gas type, charge, leakage | tCO2e by HFC/PFC/SF6 |
| MRV-003 Mobile Combustion | Fleet/vehicle data available | Vehicle type, fuel, distance | tCO2e by gas |
| MRV-004 Process Emissions | Process data available | Process type, production volume | tCO2e by gas |
| MRV-005 Fugitive Emissions | Fugitive source data available | Source type, activity data | tCO2e by gas |
| MRV-006 Land Use Change | Land use data available | Land type, area, change | tCO2e by gas |
| MRV-007 Waste Treatment | On-site waste data available | Waste type, treatment method | tCO2e by gas |
| MRV-008 Agricultural | Agricultural data available | Livestock, crops, fertilizer | tCO2e by gas |

**Acceptance Criteria:**
- [ ] All applicable Scope 1 agents execute in parallel where possible
- [ ] Results consolidated with per-gas breakdown
- [ ] Double-counting detection and resolution applied
- [ ] Total workflow duration <2 hours for 50 facilities, 5 categories

### 5.4 Workflow 4: Scope 2 Calculation Workflow

**Purpose:** Execute dual-method Scope 2 calculation with instrument allocation.

**Phases:** InstrumentCollection -> DualMethodExecution -> AllocationReconciliation -> DualReportGeneration

**Acceptance Criteria:**
- [ ] Location-based and market-based calculated independently
- [ ] Instrument allocation follows GHG Protocol hierarchy
- [ ] No double-counting of renewable energy claims
- [ ] Dual reporting reconciliation with variance analysis
- [ ] Total workflow duration <1 hour for 50 facilities

### 5.5 Workflow 5: Inventory Consolidation Workflow

**Purpose:** Consolidate all scopes into total inventory with uncertainty bounds.

**Phases:** Scope1Aggregation -> Scope2Aggregation -> UncertaintyPropagation -> TotalInventoryGeneration

**Acceptance Criteria:**
- [ ] Scope 1 total matches sum of individual category results
- [ ] Scope 2 dual totals correctly aggregated
- [ ] Uncertainty propagated using both analytical and Monte Carlo methods
- [ ] Total inventory includes all breakdowns (scope, category, gas, facility, entity)
- [ ] Total workflow duration <30 minutes

### 5.6 Workflow 6: Verification Preparation Workflow

**Purpose:** Prepare ISO 14064-3 verification-ready audit package.

**Phases:** AuditTrailCompilation -> ProvenanceVerification -> CompletenessCheck -> VerificationPackageGeneration

**Acceptance Criteria:**
- [ ] Every calculation has SHA-256 provenance hash
- [ ] Every emission factor has documented provenance (source, version, date)
- [ ] Completeness statement covers all material source categories
- [ ] Verification package contains all ISO 14064-3 required elements
- [ ] Package generated in <30 minutes

### 5.7 Workflow 7: Disclosure Generation Workflow

**Purpose:** Generate multi-framework disclosure reports from single inventory.

**Phases:** FrameworkMapping -> TemplatePopulation -> ComplianceValidation -> OutputGeneration

**Acceptance Criteria:**
- [ ] All 7+ frameworks supported
- [ ] Per-framework gap analysis with remediation actions
- [ ] ESRS E1 XBRL output generated
- [ ] CDP questionnaire sections pre-populated
- [ ] SBTi progress report generated
- [ ] Total workflow duration <2 hours for all frameworks

### 5.8 Workflow 8: Full Inventory Workflow

**Purpose:** Complete end-to-end GHG inventory from boundary definition through disclosure.

**Phases:** BoundarySetup -> DataCollection -> Scope1Calc -> Scope2Calc -> Consolidation -> TrendAnalysis -> Verification -> Disclosure

**Acceptance Criteria:**
- [ ] All 8 phases execute sequentially with full data handoff
- [ ] Total workflow duration <2 days for typical organization (50 facilities)
- [ ] All outputs include SHA-256 provenance chain
- [ ] Final deliverables include all 10 template outputs
- [ ] Audit package passes automated completeness check

---

## 6. Template Specifications

### 6.1 Template 1: GHG Inventory Report

**Purpose:** Complete GHG inventory report with all Scope 1-2 breakdowns.

**Sections:**
- Executive summary: total Scope 1, Scope 2 (location-based), Scope 2 (market-based), combined total
- Organizational boundary description and consolidation approach
- Scope 1 breakdown by category (8 categories), gas (7 gases), facility, and entity
- Scope 2 breakdown by method (location/market), energy type, facility, and entity
- Year-over-year comparison with base year (absolute and intensity)
- Uncertainty analysis summary with 95% CI
- Methodology notes per source category (tier selection, emission factor sources)
- Completeness statement with excluded sources and justification
- Data quality summary and improvement plan

**Output Formats:** MD, HTML, PDF, JSON

### 6.2 Template 2: Scope 1 Detailed Report

**Purpose:** Detailed breakdown of all 8 Scope 1 emission categories.

**Sections:**
- Per-category detailed tables (activity data, emission factors, calculated emissions per gas)
- Equipment/asset-level detail for stationary combustion, mobile fleet, refrigerant equipment
- Process emission calculation methodologies with reaction equations
- Fugitive emission estimation methods (EPA AP-42, emission factor approach, mass balance)
- Land use change carbon stock calculations
- Agricultural emission factors (enteric, manure, cropland, rice)
- Cross-category reconciliation (double-counting resolution documentation)
- Emission factor citations with full provenance

**Output Formats:** MD, HTML, PDF, JSON

### 6.3 Template 3: Scope 2 Dual Report

**Purpose:** Comprehensive Scope 2 dual-method reporting.

**Sections:**
- Location-based electricity emissions by facility with grid emission factors used
- Market-based electricity emissions with instrument allocation detail
- Steam/heat/cooling purchase emissions by supplier and facility
- Contractual instrument portfolio (PPAs, RECs, GOs, green tariffs) with allocation
- Residual mix factor application for uncovered consumption
- Location vs. market-based comparison and variance analysis
- Renewable energy procurement impact quantification
- Quality criteria compliance per GHG Protocol Scope 2 Guidance

**Output Formats:** MD, HTML, PDF, JSON

### 6.4-6.10 Templates 4-10

(Additional templates for Emission Factor Registry, Uncertainty Analysis, Trend Analysis, Verification Package, Executive Summary, Compliance Dashboard, and ESRS E1 Disclosure follow the same detailed specification pattern as above.)

---

## 7. Integration Specifications

### 7.1 Pack Orchestrator

**12-Phase DAG Pipeline:**

```
Phase 1:  BoundarySetup ──────────────────────┐
Phase 2:  DataIngestion ───────────────────────┤
Phase 3:  Scope1-Stationary ──┐                │
Phase 4:  Scope1-Refrigerants ┤                │
Phase 5:  Scope1-Mobile ──────┤ (parallel)     │ (sequential)
Phase 6:  Scope1-Other ───────┘                │
Phase 7:  Scope2-DualMethod ───────────────────┤
Phase 8:  Consolidation ──────────────────────┤
Phase 9:  UncertaintyAggregation ─────────────┤
Phase 10: TrendAnalysis ──────────────────────┤
Phase 11: ComplianceMapping ──────────────────┤
Phase 12: ReportGeneration ────────────────────┘
```

**Configuration:**
- Phase-level retry with exponential backoff (max 3 retries, 30s/60s/120s delays)
- SHA-256 provenance hash chain between phases
- Phase-level result caching (invalidated on input change)
- Parallel execution for independent Scope 1 categories (Phases 3-6)
- Conditional phases: skip agricultural if not applicable, skip process if not applicable

---

## 8. Database Migrations (V326-V335)

| Migration | Purpose |
|-----------|---------|
| V326 | Core schema: organizations, boundaries, entities, facilities, source categories |
| V327 | Emission factor registry: unified factor storage, provenance, overrides |
| V328 | Scope 1 consolidation: per-category results, gas breakdowns, double-counting flags |
| V329 | Scope 2 consolidation: dual-method results, instrument allocations, reconciliation |
| V330 | Uncertainty analysis: Monte Carlo results, analytical results, contributor rankings |
| V331 | Base year management: base year data, recalculation triggers, adjustment history |
| V332 | Trend analysis: multi-year data, decomposition results, intensity metrics |
| V333 | Compliance mapping: framework requirements, compliance scores, gap analysis |
| V334 | Reporting: report metadata, generation history, verification packages |
| V335 | Views, indexes, RLS policies, seed data, materialized views for dashboards |

---

## 9. Non-Functional Requirements

### 9.1 Performance

| Operation | Target | Conditions |
|-----------|--------|------------|
| Boundary definition (100 entities) | <30 seconds | Standard configuration |
| Single-facility Scope 1 calculation (all categories) | <5 minutes | All 8 agents, standard data |
| Organization-wide inventory (50 facilities) | <2 hours | All categories, parallel execution |
| Uncertainty Monte Carlo (10,000 iterations) | <10 minutes | 50 facilities, 5 categories |
| Multi-framework compliance mapping | <15 minutes | 7 frameworks, 500+ requirements |
| Full report generation (all 10 templates) | <30 minutes | Complete inventory |
| Full end-to-end workflow | <2 days | 50 facilities, all categories |

### 9.2 Accuracy

| Requirement | Target |
|-------------|--------|
| Emission factor transcription accuracy | 100% (automated lookup) |
| GWP conversion accuracy | 100% (deterministic lookup) |
| Financial consolidation accuracy | 100% (Decimal arithmetic) |
| Year-over-year comparison consistency | 100% (base year recalculation) |
| Uncertainty quantification | Within 1% of reference Monte Carlo |

### 9.3 Security

- All data encrypted at rest (AES-256-GCM)
- TLS 1.3 for all API communications
- RBAC with 20+ permissions for inventory management
- Audit logging of all data access and modifications
- PII detection and redaction in reports
- Row-level security per tenant

### 9.4 Compliance

- Zero-hallucination: all calculations are deterministic arithmetic with published factors
- Bit-perfect reproducibility: same inputs produce identical outputs (SHA-256 verified)
- Full audit trail: every calculation traceable from raw data to reported value
- Multi-framework: single inventory serves 7+ disclosure frameworks
- Verification-ready: ISAE 3410 and ISO 14064-3 compliant audit packages

---

## 10. Testing Strategy

### 10.1 Test Categories

| Category | Count Target | Coverage |
|----------|-------------|----------|
| Unit tests (per engine) | 500+ | All calculation paths, edge cases, error handling |
| Integration tests | 100+ | Cross-engine data flow, MRV agent integration |
| Workflow tests | 80+ | All 8 workflows, phase transitions, error recovery |
| Template tests | 60+ | All 10 templates, all output formats |
| Compliance tests | 100+ | Framework requirement validation, regulatory formulas |
| End-to-end tests | 30+ | Full inventory scenarios by sector preset |
| Performance tests | 20+ | Timing targets, resource utilization |
| **Total** | **850+** | **85%+ code coverage** |

### 10.2 Reference Test Cases

| Test Case | Expected Result | Validation |
|-----------|----------------|------------|
| Natural gas combustion: 1,000,000 m3 at 0.03869 GJ/m3 | 2,167.2 tCO2e (DEFRA 2024) | Cross-reference DEFRA calculator |
| Grid electricity: 10,000 MWh at 0.385 kgCO2/kWh (DE) | 3,850 tCO2e | Cross-reference UBA |
| R-410A refrigerant: 100 kg loss at GWP 2088 (AR6) | 208.8 tCO2e | Cross-reference IPCC AR6 |
| Fleet diesel: 500,000 km at 0.168 kgCO2/km | 84.0 tCO2e | Cross-reference DEFRA |
| Dual Scope 2: 50% REC coverage | Location total, market 50% reduction | Manual calculation |

---

## 11. Glossary

| Term | Definition |
|------|-----------|
| AR4/AR5/AR6 | IPCC Assessment Reports (4th 2007, 5th 2014, 6th 2021) providing GWP values |
| CO2e | Carbon dioxide equivalent - common unit using GWP conversion |
| CEMS | Continuous Emission Monitoring System |
| EF | Emission Factor - ratio of emissions to activity data |
| GO | Guarantee of Origin (EU renewable energy certificate) |
| GWP | Global Warming Potential - relative warming effect of a GHG vs CO2 |
| ISAE 3410 | Assurance standard for GHG statements |
| IPMVP | International Performance M&V Protocol |
| PPA | Power Purchase Agreement |
| REC | Renewable Energy Certificate |
| Residual Mix | Grid emission factor minus allocated certificates |
| RLS | Row-Level Security (database access control) |
| SBTi | Science Based Targets initiative |
| Tier 1/2/3 | IPCC methodology levels (default/country-specific/facility-specific) |
| WTT | Well-to-Tank (upstream fuel emissions) |

---

## 12. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-24 | GreenLang Product Team | Initial PRD for PACK-041 Scope 1-2 Complete Pack |
