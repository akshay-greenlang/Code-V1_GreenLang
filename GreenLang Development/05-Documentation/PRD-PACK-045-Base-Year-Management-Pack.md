# PRD-PACK-045: Base Year Management Pack

**Pack ID:** PACK-045-base-year
**Category:** GHG Accounting Packs
**Tier:** Enterprise
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-24
**Prerequisite:** PACK-041 Scope 1-2 Complete Pack (required); enhanced with PACK-042 Scope 3 Starter Pack, PACK-043 Scope 3 Complete Pack, and PACK-044 Inventory Management Pack when present

---

## 1. Executive Summary

### 1.1 Problem Statement

Every GHG emissions reduction target is anchored to a base year -- the historical reference point against which progress is measured. While selecting a base year may seem straightforward, managing it over time is one of the most complex and error-prone aspects of corporate GHG accounting. The GHG Protocol Corporate Standard (Chapter 5) and GHG Protocol Scope 3 Standard (Chapter 5) establish detailed base year management requirements, yet organizations face ten persistent challenges that no existing tool adequately addresses:

1. **Arbitrary base year selection**: Organizations frequently select base years based on data availability ("we have 2019 data") rather than systematic evaluation of data quality, representativeness, organizational stability, and regulatory alignment. A poorly chosen base year -- one affected by unusual events (COVID-19, plant shutdowns, acquisitions in progress) -- undermines the credibility of all subsequent emissions comparisons. ISO 14064-1:2018 Clause 5.2 requires documented justification for base year selection; most organizations have none.

2. **Incomplete base year inventories**: The base year inventory must be the highest-quality emissions dataset an organization maintains, as every future comparison depends on its accuracy. Yet base year inventories frequently have gaps -- missing Scope 3 categories, estimated rather than measured activity data, outdated emission factors, and inconsistent consolidation approaches. When organizations later improve data quality for current-year reporting, the base year comparison becomes apple-to-oranges. Without systematic base year inventory preservation including source-level emissions, methodology documentation, and emission factor provenance, year-over-year comparisons are unreliable.

3. **Missing or informal recalculation policies**: GHG Protocol Chapter 5 requires organizations to develop and document a base year recalculation policy that specifies which triggers require recalculation and what significance thresholds apply. Most organizations either have no policy (recalculation decisions are ad hoc) or have a one-paragraph policy that lacks specificity. Without structured policy configuration covering all trigger types (acquisitions, divestitures, methodology changes, error corrections, boundary changes, outsourcing/insourcing), organizations make inconsistent recalculation decisions that undermine time-series comparability. SBTi requires a base year recalculation policy aligned with the GHG Protocol, and verifiers under ISAE 3410 specifically test policy application.

4. **Undetected recalculation triggers**: Organizational changes that should trigger base year recalculation often go undetected by the GHG reporting team. Acquisitions are completed by M&A teams, methodology changes are implemented by consultants, and boundary changes occur through operational restructuring -- all without systematically notifying the base year management process. Without automated trigger detection integrating with organizational change feeds, ERP systems, and methodology version tracking, significant triggers are missed, resulting in non-compliant time-series reporting.

5. **Subjective significance assessment**: When a trigger is detected, the GHG Protocol requires assessment of whether the impact is "significant" enough to warrant recalculation. The protocol suggests thresholds (commonly 5% for individual triggers, 10% cumulative) but organizations lack structured methodology for quantifying impact. Without quantitative significance testing including sensitivity analysis, Monte Carlo simulation for uncertain impacts, and cumulative tracking across multiple triggers within a reporting period, significance decisions are subjective and inconsistently applied.

6. **Error-prone adjustment calculations**: Base year adjustments require recalculating historical emissions as if the current organizational structure or methodology had always been in place. For acquisitions, this means pro-rata time-weighting of acquired entity emissions. For methodology changes, it means restating historical calculations using updated factors. For divestitures, it means removing divested emissions from all historical periods. These calculations are complex, involve multiple entities and time periods, and are highly susceptible to errors. Without structured adjustment calculation with full audit trails, approval workflows, and automated propagation across all historical periods, adjustments introduce errors that persist in the time series.

7. **Broken time-series comparability**: Even with correct adjustments, maintaining comparability across a multi-year time series is challenging. Emission factor vintages change, organizational boundaries shift, methodology tiers are upgraded, and reporting scope expands. Without systematic time-series consistency validation that checks for boundary continuity, methodology consistency, completeness evolution, and emission factor vintage alignment, organizations publish trend data that mixes incomparable measurements.

8. **Disconnected target tracking**: Emissions reduction targets (SBTi, internal, regulatory) are set relative to the base year but tracked in separate systems from base year management. When the base year is recalculated, targets must be rebased, but this linkage is manual and error-prone. Without integrated target tracking that automatically propagates base year adjustments to target calculations, recalculates required reduction pathways (linear, compound, absolute, intensity), and tracks progress against adjusted baselines, organizations report incorrect target progress.

9. **Inadequate audit evidence**: Base year management decisions -- selection rationale, recalculation triggers, significance assessments, adjustment calculations, policy application -- are among the most scrutinized areas in third-party verification under ISO 14064-3 and ISAE 3410. Verifiers require complete evidence packages: trigger documentation, significance calculations, adjustment methodology, approval records, and before/after comparisons. Without structured audit trail generation with evidence packaging, approval workflow records, and verification support, organizations spend 40-80 hours preparing base year evidence for each verification engagement.

10. **Fragmented multi-framework reporting**: Organizations must report base year information across multiple frameworks with different requirements: GHG Protocol (Chapter 5 disclosure), ESRS E1 (paragraphs 34-38, base year and recalculation disclosure), CDP Climate Change (C5.1-C5.2 base year emissions and recalculations), SBTi (Section 7, base year requirements for target validation), SEC Climate Rules (Item 1504, base year disclosure), and California SB 253 (base year reference requirements). Each framework requires specific data fields, formats, and contextual information. Without multi-framework reporting that maps base year data to each framework's requirements, organizations maintain separate reporting processes for each framework.

### 1.2 Solution Overview

PACK-045 is the **Base Year Management Pack** -- the fifth pack in the "GHG Accounting Packs" category. While PACK-041 calculates Scope 1-2 emissions, PACK-042/043 calculate Scope 3 emissions, and PACK-044 manages inventory governance, PACK-045 provides the comprehensive base year lifecycle management layer that ensures time-series integrity, target alignment, and multi-framework compliance.

The pack provides:
- **Multi-criteria base year selection** with scoring across data quality, representativeness, organizational stability, strategic alignment, and regulatory compliance dimensions
- **Complete base year inventory preservation** with source-level emissions, methodology documentation, emission factor provenance, and boundary definition snapshotting for Scope 1, 2, and 3
- **Configurable recalculation policy management** with pre-built policy templates (GHG Protocol default, SBTi strict, SEC compliant, CDP aligned) and custom policy builder
- **Automated recalculation trigger detection** integrating with M&A feeds, methodology version tracking, error correction workflows, boundary change notifications, and outsourcing/insourcing events
- **Quantitative significance assessment** per GHG Protocol Chapter 5 with configurable thresholds (individual 5%, cumulative 10%), sensitivity analysis, and cumulative tracking
- **Structured adjustment calculation** with pro-rata time-weighting, like-for-like methodology restatement, equity share adjustments, and automatic propagation across all historical periods
- **Time-series consistency validation** ensuring boundary continuity, methodology consistency, completeness evolution, emission factor vintage alignment, and normalization factor coherence
- **Integrated target tracking** with SBTi pathway alignment (1.5°C at 4.2%/yr, Well-Below 2°C at 2.5%/yr), automatic target rebasing on base year adjustment, and multi-target portfolio management
- **Audit trail and verification support** with ISAE 3410 evidence packages, multi-level approval workflows, digital sign-off, and verification-ready documentation generation
- **Multi-framework reporting** generating base year disclosures for GHG Protocol, ESRS E1, CDP Climate Change, SBTi, SEC Climate Rules, and California SB 253

The pack includes **10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets** covering the complete base year lifecycle from selection through multi-framework reporting.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Spreadsheet Approach | PACK-045 Base Year Management Pack |
|-----------|-------------------------------|--------------------------------------|
| Base year selection | Ad hoc ("pick the year with best data") | Multi-criteria scoring across 5+ dimensions with documented rationale |
| Inventory preservation | Static copy of spreadsheet | Source-level snapshot with methodology, factors, and boundary documentation |
| Recalculation policy | 1-paragraph document (if any) | Configurable policy engine with 4 pre-built templates and custom rules |
| Trigger detection | Manual notification (often missed) | Automated detection from M&A, methodology, error, boundary change feeds |
| Significance testing | Gut feel or simple percentage | Quantitative assessment with sensitivity analysis and cumulative tracking |
| Adjustment calculation | Manual spreadsheet recalculation | Structured calculation with pro-rata, like-for-like, equity share methods |
| Time-series consistency | Manual review of year-over-year changes | Automated 10-dimension consistency validation with normalization |
| Target tracking | Separate spreadsheet, manual update | Integrated tracking with automatic rebasing on base year adjustment |
| Audit evidence | 40-80 hours manual preparation | One-click evidence package generation with approval trails |
| Multi-framework reporting | Separate process per framework | Single source generating all 6+ framework-specific disclosures |

### 1.4 Regulatory Basis

| Regulation/Standard | Relevant Provisions | PACK-045 Coverage |
|---------------------|---------------------|-------------------|
| GHG Protocol Corporate Standard (2004, rev 2015) | Chapter 5: Base year selection, recalculation triggers, significance thresholds | Full implementation of all Ch 5 requirements |
| GHG Protocol Scope 3 Standard (2011) | Chapter 5: Scope 3 base year, recalculation for value chain changes | Scope 3-specific trigger detection and adjustment |
| ISO 14064-1:2018 | Clause 5.2: Base year selection and justification | Documented selection with multi-criteria scoring |
| ESRS E1 (Delegated Act 2023/2772) | Paragraphs 34-38: Base year disclosure, recalculation triggers | ESRS E1-specific reporting template |
| SBTi Corporate Manual (2023) / Criteria v5.1 | Section 7: Base year requirements, recalculation policy | SBTi-strict policy template, pathway alignment tracking |
| SEC Climate Disclosure Rule (33-11275) | Item 1504: Base year GHG emissions disclosure | SEC-compliant reporting template |
| California SB 253 | Base year reference for annual emissions reporting | SB 253-specific disclosure template |
| ISAE 3410 | Assurance of GHG statements including base year | Evidence package generation for verification |
| CDP Climate Change 2024 | C5.1-C5.2: Base year emissions, recalculations | CDP-formatted base year response generation |

---

## 2. Technical Architecture

### 2.1 Pack Structure

```
packs/ghg-accounting/PACK-045-base-year/
├── __init__.py                          # Pack metadata and exports
├── pack.yaml                            # Pack manifest
├── config/
│   ├── __init__.py                      # Config exports
│   ├── pack_config.py                   # Pydantic v2 configuration (14 enums, 15 sub-configs)
│   ├── demo/
│   │   └── __init__.py                  # Demo configuration
│   └── presets/
│       ├── __init__.py                  # Presets module
│       ├── corporate_office.yaml        # Office-based organizations
│       ├── manufacturing.yaml           # Heavy manufacturing
│       ├── energy_utility.yaml          # Energy/utility companies
│       ├── transport_logistics.yaml     # Transport & logistics
│       ├── food_agriculture.yaml        # Food & agriculture
│       ├── real_estate.yaml             # Real estate portfolios
│       ├── healthcare.yaml              # Healthcare sector
│       └── sme_simplified.yaml          # SME simplified approach
├── engines/
│   ├── __init__.py                      # Engine imports with try/except
│   ├── base_year_selection_engine.py    # Engine 1: Multi-criteria selection
│   ├── base_year_inventory_engine.py    # Engine 2: Inventory preservation
│   ├── recalculation_policy_engine.py   # Engine 3: Policy management
│   ├── recalculation_trigger_engine.py  # Engine 4: Trigger detection
│   ├── significance_assessment_engine.py # Engine 5: Significance testing
│   ├── base_year_adjustment_engine.py   # Engine 6: Adjustment calculation
│   ├── time_series_consistency_engine.py # Engine 7: Consistency validation
│   ├── target_tracking_engine.py        # Engine 8: Target progress tracking
│   ├── base_year_audit_engine.py        # Engine 9: Audit trail & verification
│   └── base_year_reporting_engine.py    # Engine 10: Multi-framework reporting
├── workflows/
│   ├── __init__.py                      # Workflow imports
│   ├── base_year_establishment_workflow.py    # 5-phase establishment
│   ├── recalculation_assessment_workflow.py   # 4-phase trigger assessment
│   ├── recalculation_execution_workflow.py    # 5-phase adjustment execution
│   ├── target_rebasing_workflow.py            # 4-phase target update
│   ├── audit_verification_workflow.py         # 4-phase audit support
│   ├── annual_review_workflow.py              # 4-phase annual review
│   ├── merger_acquisition_workflow.py         # 5-phase M&A handling
│   └── full_base_year_pipeline_workflow.py    # 10-phase end-to-end
├── templates/
│   ├── __init__.py                      # Template registry
│   ├── base_year_selection_report.py    # Selection rationale report
│   ├── inventory_summary_report.py      # Inventory snapshot report
│   ├── recalculation_trigger_report.py  # Trigger assessment report
│   ├── adjustment_detail_report.py      # Adjustment detail report
│   ├── time_series_dashboard.py         # Consistency dashboard
│   ├── target_progress_report.py        # Target tracking report
│   ├── audit_trail_report.py            # Audit evidence report
│   ├── policy_compliance_report.py      # Policy compliance report
│   ├── merger_acquisition_report.py     # M&A impact report
│   └── executive_summary_report.py      # Executive summary
├── integrations/
│   ├── __init__.py                      # Integration imports
│   ├── pack_orchestrator.py             # 10-phase DAG orchestrator
│   ├── pack041_bridge.py                # PACK-041 Scope 1-2 data
│   ├── pack042_bridge.py                # PACK-042 Scope 3 Starter data
│   ├── pack043_bridge.py                # PACK-043 Scope 3 Complete data
│   ├── pack044_bridge.py                # PACK-044 Inventory Management
│   ├── mrv_bridge.py                    # MRV agent integration (30 agents)
│   ├── data_bridge.py                   # DATA agent integration
│   ├── foundation_bridge.py             # Foundation agent integration
│   ├── erp_connector.py                 # SAP/Oracle/Dynamics ERP connector
│   ├── notification_bridge.py           # Multi-channel notification
│   ├── health_check.py                  # 20-category health verification
│   └── setup_wizard.py                  # 8-step configuration wizard
└── tests/
    ├── __init__.py                      # Test suite init
    ├── conftest.py                      # Shared fixtures
    ├── test_config.py                   # Configuration tests
    ├── test_base_year_selection_engine.py
    ├── test_base_year_inventory_engine.py
    ├── test_recalculation_policy_engine.py
    ├── test_recalculation_trigger_engine.py
    ├── test_significance_assessment_engine.py
    ├── test_base_year_adjustment_engine.py
    ├── test_time_series_consistency_engine.py
    ├── test_target_tracking_engine.py
    ├── test_base_year_audit_engine.py
    ├── test_base_year_reporting_engine.py
    ├── test_workflows.py
    ├── test_integrations.py
    ├── test_manifest.py
    └── e2e/
        ├── __init__.py
        └── test_full_pipeline.py
```

### 2.2 Database Schema (V366-V375)

All tables use the `gl_by_` prefix with PostgreSQL RLS policies.

| Migration | Description | Key Tables |
|-----------|-------------|------------|
| V366 | Core schema | gl_by_base_years, gl_by_selection_criteria, gl_by_configuration |
| V367 | Inventory | gl_by_inventory_snapshots, gl_by_source_emissions |
| V368 | Recalculation policy | gl_by_recalculation_policies, gl_by_policy_versions |
| V369 | Triggers | gl_by_triggers, gl_by_trigger_status |
| V370 | Significance | gl_by_significance_assessments, gl_by_sensitivity_results |
| V371 | Adjustments | gl_by_adjustment_packages, gl_by_adjustment_lines |
| V372 | Time series | gl_by_yearly_data, gl_by_consistency_findings |
| V373 | Targets | gl_by_targets, gl_by_progress_tracking |
| V374 | Audit | gl_by_audit_trail, gl_by_approvals, gl_by_verifications |
| V375 | Views & indexes | Materialized views, composite indexes, seed data |

---

## 3. Engine Specifications

### 3.1 Engine 1: BaseYearSelectionEngine

**Purpose**: Multi-criteria base year candidate evaluation and scoring.

**Key Models**: CandidateYear, SelectionWeights, SelectionResult
**Key Methods**: `evaluate_candidate()`, `score_candidates()`, `recommend_base_year()`

**Scoring Dimensions**:
- Data quality score (completeness, accuracy, timeliness)
- Representativeness score (typical operations, no anomalies)
- Organizational stability score (no major M&A during year)
- Strategic alignment score (SBTi target year, regulatory requirement)
- Regulatory compliance score (framework-specific requirements)

**Formulas**:
```
weighted_score = Σ(dimension_score × dimension_weight) / Σ(weights)
confidence = min(dimension_scores) / max(dimension_scores)
```

### 3.2 Engine 2: BaseYearInventoryEngine

**Purpose**: Complete Scope 1+2+3 emissions inventory preservation with source-level detail.

**Key Models**: SourceEmission, ScopeTotal, BaseYearInventory
**Key Methods**: `create_inventory_snapshot()`, `validate_completeness()`, `compute_scope_totals()`

**Preservation Scope**: Source-level emissions, activity data, emission factors with vintage, methodology documentation, boundary definition, consolidation approach, uncertainty estimates.

### 3.3 Engine 3: RecalculationPolicyEngine

**Purpose**: Configurable recalculation policy management with framework-specific defaults.

**Key Models**: PolicyType, ThresholdConfig, TriggerRule
**Pre-built Policies**:
- GHG_PROTOCOL_DEFAULT: 5% individual / 10% cumulative thresholds
- SBTI_STRICT: 5% individual / 5% cumulative (stricter per SBTi Criteria v5.1)
- SEC_COMPLIANT: Aligned with SEC Climate Disclosure Rule materiality
- CDP_ALIGNED: CDP C5.2 recalculation disclosure requirements
- CUSTOM: User-configurable rules per trigger type

### 3.4 Engine 4: RecalculationTriggerEngine

**Purpose**: Automated detection of events requiring base year recalculation assessment.

**Key Models**: TriggerType, DetectionMethod, EntityChange, MethodologyChange, ErrorCorrection, BoundaryChange
**Trigger Types**: ACQUISITION, DIVESTITURE, MERGER, METHODOLOGY_CHANGE, ERROR_CORRECTION, BOUNDARY_CHANGE, OUTSOURCING, INSOURCING, SOURCE_CATEGORY_CHANGE
**Detection Methods**: ERP feed monitoring, manual entry, API webhook, scheduled scan

### 3.5 Engine 5: SignificanceAssessmentEngine

**Purpose**: Quantitative significance testing per GHG Protocol Chapter 5.

**Key Models**: SignificanceMethod, TriggerAssessment, CumulativeAssessment, SensitivityResult
**Core Formula**:
```python
significance_pct = abs(emission_impact) / base_year_total * Decimal("100")
is_significant = significance_pct >= threshold_pct
```

**Methods**: Individual trigger assessment, cumulative assessment within period, sensitivity analysis with parameter variation.

### 3.6 Engine 6: BaseYearAdjustmentEngine

**Purpose**: Structured base year adjustment calculation and propagation.

**Key Models**: AdjustmentType, AdjustmentLine, AdjustmentPackage
**Adjustment Methods**:
- **Pro-rata time-weighting**: For mid-year acquisitions/divestitures, weight emissions by fraction of year
- **Like-for-like restatement**: For methodology changes, restate historical data using updated methodology
- **Equity share adjustment**: For ownership changes, recalculate historical emissions at new equity share
- **Automatic propagation**: Apply adjustments across all historical periods consistently

**Formula** (pro-rata):
```python
adjusted_emissions = acquired_annual_emissions * (days_in_reporting_period / days_in_year)
```

### 3.7 Engine 7: TimeSeriesConsistencyEngine

**Purpose**: Multi-dimensional time-series comparability validation.

**Key Models**: ConsistencyStatus, InconsistencyType, YearData, TrendPoint
**Consistency Dimensions**:
1. Boundary continuity (same organizational boundary across years)
2. Methodology consistency (same calculation approaches)
3. Completeness evolution (same source categories)
4. Emission factor vintage alignment (same factor sets or documented changes)
5. Consolidation approach consistency (equity share, operational control, financial control)
6. Normalization factor coherence (intensity metrics use consistent denominators)

### 3.8 Engine 8: TargetTrackingEngine

**Purpose**: Base year-anchored emissions reduction target tracking with SBTi pathway alignment.

**Key Models**: TargetType, SBTiAmbition, EmissionsTarget, ProgressPoint, ReductionAttribution
**SBTi Pathways**:
- 1.5°C: 4.2% per year compounding reduction
- Well-Below 2°C: 2.5% per year compounding reduction
- Net-Zero: Per SBTi Net-Zero Standard v1.0

**Formula** (compounding reduction):
```python
target_emissions(year) = base_year_emissions * (1 - annual_rate) ** (year - base_year)
```

**Automatic rebasing**: When base year is adjusted, all linked targets automatically recalculate required reduction pathways.

### 3.9 Engine 9: BaseYearAuditEngine

**Purpose**: Audit trail management and ISAE 3410 verification support.

**Key Models**: AuditEventType, VerificationLevel, AuditEntry, VerificationPackage, AuditTrail
**Audit Events**: Selection, trigger detection, significance assessment, adjustment calculation, policy application, approval, review, verification, reporting
**Evidence Packages**: Structured documentation bundles with trigger evidence, significance calculations, adjustment workpapers, approval records, and before/after comparisons.

### 3.10 Engine 10: BaseYearReportingEngine

**Purpose**: Multi-framework base year disclosure generation.

**Key Models**: ReportingFramework, OutputFormat, ReportSection, BaseYearReport, MultiFrameworkReport
**Supported Frameworks**:
- GHG Protocol (Chapter 5 disclosure tables)
- ESRS E1 (paragraphs 34-38 data points)
- CDP Climate Change (C5.1, C5.2 responses)
- SBTi (Section 7 base year documentation)
- SEC Climate Rules (Item 1504 base year disclosure)
- California SB 253 (base year reference tables)

---

## 4. Workflow Specifications

### 4.1 Base Year Establishment Workflow
5-phase: CandidateAssessment → DataQualityCheck → BaseYearSelection → InventorySnapshot → DocumentationGeneration

### 4.2 Recalculation Assessment Workflow
4-phase: TriggerDetection → SignificanceTesting → PolicyCompliance → RecommendationGeneration

### 4.3 Recalculation Execution Workflow
5-phase: AdjustmentCalculation → ImpactValidation → ApprovalCollection → AdjustmentApplication → AuditRecording

### 4.4 Target Rebasing Workflow
4-phase: ImpactAssessment → TargetRecalculation → StakeholderNotification → TargetUpdate

### 4.5 Audit Verification Workflow
4-phase: EvidenceCollection → CompletenessCheck → PackageGeneration → VerificationSupport

### 4.6 Annual Review Workflow
4-phase: PolicyReview → TriggerScan → ConsistencyCheck → ReportGeneration

### 4.7 Merger & Acquisition Workflow
5-phase: EntityIdentification → EmissionQuantification → ProRataCalculation → SignificanceTesting → AdjustmentExecution

### 4.8 Full Base Year Pipeline Workflow
10-phase end-to-end orchestrator: Selection → Inventory → PolicySetup → TriggerDetection → SignificanceAssessment → AdjustmentCalculation → ConsistencyValidation → TargetTracking → AuditTrail → Reporting

---

## 5. Integration Architecture

### 5.1 Pack Bridges
- **PACK-041 Bridge**: Import Scope 1-2 calculation results, emission factors, activity data
- **PACK-042 Bridge**: Import Scope 3 Starter category results
- **PACK-043 Bridge**: Import Scope 3 Complete enterprise data
- **PACK-044 Bridge**: Inventory management period and versioning data

### 5.2 Agent Bridges
- **MRV Bridge**: All 30 MRV calculation agents for recalculation execution
- **DATA Bridge**: Data quality, lineage, and validation agents
- **Foundation Bridge**: Orchestrator, schema validation, assumptions registry

### 5.3 External Connectors
- **ERP Connector**: SAP, Oracle, Microsoft Dynamics for organizational change detection
- **Notification Bridge**: Email, Slack, Teams, webhook notifications
- **Health Check**: 20-category operational health verification
- **Setup Wizard**: 8-step guided configuration

---

## 6. Preset Configurations

| Preset | Sector | Key Characteristics |
|--------|--------|---------------------|
| corporate_office | Office/Professional Services | Scope 2 dominant, simple boundary, annual review |
| manufacturing | Heavy Manufacturing | Scope 1 dominant, complex sources, quarterly monitoring |
| energy_utility | Energy/Utilities | Scope 1+3 critical, regulatory scrutiny, monthly monitoring |
| transport_logistics | Transport & Logistics | Mobile combustion focus, fleet changes, semi-annual review |
| food_agriculture | Food & Agriculture | Land use/agricultural, seasonal patterns, Scope 3 heavy |
| real_estate | Real Estate | Portfolio changes, multi-building, asset-level tracking |
| healthcare | Healthcare | Refrigerants, medical gases, regulatory compliance focus |
| sme_simplified | SME | Simplified approach, minimal triggers, annual review only |

---

## 7. Quality Assurance

### 7.1 Test Coverage
- **Target**: 900+ tests across all engines, workflows, integrations, and templates
- **Unit tests**: Each engine independently tested with boundary conditions, edge cases, and error handling
- **Integration tests**: Cross-engine workflow validation
- **E2E tests**: Full pipeline execution with realistic multi-year scenarios
- **Manifest tests**: YAML manifest validation and structural integrity

### 7.2 Zero-Hallucination Guarantees
- All calculations use Python Decimal arithmetic with ROUND_HALF_UP
- No LLM in any calculation path
- Every result includes SHA-256 provenance hash
- Deterministic output for identical inputs (bit-perfect reproducibility)

### 7.3 Regulatory Compliance
- GHG Protocol Chapter 5: Full implementation of selection, triggers, significance, recalculation
- ISO 14064-1 Clause 5.2: Documented selection with multi-criteria justification
- ISAE 3410: Evidence package generation for third-party verification
- SBTi Criteria v5.1: Strict recalculation policy with pathway alignment
- ESRS E1 paragraphs 34-38: Base year disclosure data points
- SEC Climate Rules Item 1504: Base year disclosure requirements
- CDP Climate Change C5.1-C5.2: Base year response generation
- California SB 253: Base year reference requirements

---

## 8. Dependencies

### 8.1 Required
- PACK-041 Scope 1-2 Complete Pack (Scope 1+2 base year emissions data)

### 8.2 Optional (Enhanced)
- PACK-042 Scope 3 Starter Pack (Scope 3 category-level data)
- PACK-043 Scope 3 Complete Pack (Enterprise Scope 3 with full category breakdown)
- PACK-044 Inventory Management Pack (Inventory governance and period management)

### 8.3 Platform Dependencies
- Python 3.11+
- Pydantic v2
- PostgreSQL 15+ with TimescaleDB
- GreenLang Foundation Agents (orchestrator, schema, assumptions)
- GreenLang MRV Agents (for recalculation execution)

---

## 9. Deployment

### 9.1 Database Migrations
- V366-V375: 10 migration files with `gl_by_` prefix tables
- RLS policies for multi-tenant isolation
- Materialized views for performance optimization
- Composite indexes for query efficiency
- Seed data for emission factors and framework mappings

### 9.2 Configuration
- 8 sector-specific presets for quick deployment
- Setup wizard for guided configuration
- Health check for operational validation

---

## 10. Relationship to Other GHG Accounting Packs

| Pack | Focus | Relationship to PACK-045 |
|------|-------|--------------------------|
| PACK-041 | Scope 1-2 calculation | Provides base year emissions data (required dependency) |
| PACK-042 | Scope 3 starter | Provides Scope 3 category data for base year inventory |
| PACK-043 | Scope 3 complete | Provides enterprise Scope 3 data with category detail |
| PACK-044 | Inventory management | Provides governance layer; PACK-045 manages base year lifecycle specifically |
| **PACK-045** | **Base year management** | **Manages base year selection, recalculation, adjustment, target tracking, audit, and reporting across all scopes** |

---

*Generated by GreenLang Platform Team. All calculations are zero-hallucination, bit-perfect reproducible, and SHA-256 hashed for audit assurance.*
