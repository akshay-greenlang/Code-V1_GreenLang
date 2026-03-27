# PRD-PACK-050: GHG Consolidation Pack

**Pack ID:** PACK-050-ghg-consolidation
**Category:** GHG Accounting Packs
**Tier:** Enterprise
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-27
**Prerequisite:** PACK-041 Scope 1-2 Complete Pack (required); enhanced with PACK-042/043 Scope 3 Packs, PACK-044 Inventory Management, PACK-045 Base Year Management, PACK-048 Assurance Prep, PACK-049 Multi-Site Management

---

## 1. Executive Summary

### 1.1 Problem Statement

Multi-entity corporate groups -- conglomerates, holding companies, multinational corporations, private equity portfolios, joint venture partnerships, and real estate funds -- must produce a single consolidated GHG inventory that accurately reflects the group's total greenhouse gas footprint. The GHG Protocol Corporate Standard (Chapter 3) requires organizations to define organizational boundaries and consolidate emissions from all entities within those boundaries. Yet corporate GHG consolidation remains one of the most error-prone and labor-intensive aspects of GHG accounting, creating persistent challenges:

1. **Organizational boundary ambiguity**: The GHG Protocol offers three consolidation approaches (equity share, operational control, financial control), each producing materially different consolidated totals. A conglomerate with 50+ subsidiaries, 15 joint ventures, and 20 associates may report emissions that differ by 30-60% depending on the approach selected. Most organizations lack a systematic methodology to evaluate all three approaches, assess their impact, and select the most appropriate one for their stakeholder requirements (CSRD mandates operational control; SEC prefers financial control; SBTi accepts either equity share or control). Without rigorous approach evaluation, organizations make ad hoc boundary decisions that create inconsistencies across reporting frameworks.

2. **Multi-tier ownership complexity**: Modern corporate structures involve multi-tier holding chains where a parent owns 80% of Sub-A, which owns 60% of Sub-B, which owns 40% of JV-C. The effective equity share in JV-C is 80% x 60% x 40% = 19.2%. Calculating effective ownership through 3-5 tier holding structures with cross-holdings, preferred equity, convertible instruments, and voting rights vs. economic interest splits requires systematic chain resolution. Manual calculations in spreadsheets routinely produce errors of 5-15% in equity allocation, which propagate directly to consolidated emissions totals. Circular ownership structures (cross-holdings between subsidiaries) require iterative algebraic resolution that spreadsheets cannot handle.

3. **Intercompany double-counting**: Within a corporate group, entity A may sell electricity to entity B (A records Scope 1 from generation; B records Scope 2 from purchase). Entity C may send waste to entity D's treatment facility (C records Scope 3 Category 5; D records Scope 1 from treatment). Without systematic intercompany elimination, these intra-group transfers are double-counted in the consolidated inventory. For vertically integrated groups (energy companies, industrial conglomerates), intercompany double-counting can inflate consolidated emissions by 10-25%. Current practice relies on manual identification of intra-group transfers through email surveys, which typically captures less than 60% of actual transfers.

4. **M&A boundary disruption**: Acquisitions, divestitures, mergers, and demergers change the organizational boundary mid-year. The GHG Protocol (Chapter 5) requires base year recalculation when structural changes exceed 10% of base year emissions, but determining whether a threshold is breached requires accurate pro-rata allocation. An acquisition completed on July 15th requires 169/365 (46.3%) of the acquired entity's annual emissions to be added to the current year and 100% added to the restated base year. Most organizations either include 100% (overstating current year) or 0% (understating it), and fail to perform required base year restatement entirely.

5. **Consolidation timing and completeness**: In a group with 100+ entities across 30 countries, collecting complete GHG data from all entities within the reporting deadline is a major operational challenge. Entities submit data at different times, in different formats, with different levels of completeness. Late submissions, resubmissions, and corrections create version control chaos. Without a systematic data collection and completeness tracking system, consolidated inventories routinely omit 5-15% of entities, with no visibility into which entities are missing or what their estimated contribution would be.

6. **Scope reclassification at group level**: When entity-level inventories are consolidated, certain emission categories change scope. If subsidiary A generates electricity and sells it to subsidiary B within the group, at entity level A reports Scope 1 and B reports Scope 2. At consolidated group level under operational control, if both A and B are within the boundary, the electricity generation is Scope 1 for the group (generation) and the internal transfer is eliminated (not Scope 2). This scope reclassification logic is complex, varies by consolidation approach, and is almost never performed correctly in manual processes.

7. **Multi-framework reporting divergence**: A single consolidated GHG inventory must serve multiple reporting frameworks with different requirements: CSRD/ESRS E1 requires operational control and includes full value chain; SEC requires financial control at the registrant level; CDP accepts either approach; SBTi requires consistency with the chosen target boundary; IFRS S2 follows the financial reporting entity; GRI 305 allows any approach with disclosure. Producing framework-specific consolidated outputs from a single underlying dataset requires a mapping layer that most organizations lack, leading to inconsistent numbers across reports and costly reconciliation exercises during assurance.

8. **Audit trail fragmentation**: External assurance of consolidated GHG inventories requires a complete audit trail from entity-level source data through every consolidation step to the final consolidated totals. Assurance providers need to verify ownership percentages, boundary inclusion/exclusion decisions, intercompany eliminations, M&A adjustments, and scope reclassifications. When consolidation is performed in spreadsheets, the audit trail is fragmented across dozens of files, email chains, and manual override notes. Limited assurance engagements typically spend 40-60% of their time reconstructing the consolidation logic rather than verifying the underlying data.

### 1.2 Solution Overview

PACK-050 is the **GHG Consolidation Pack** -- the tenth pack in the "GHG Accounting Packs" category. While PACK-041 through PACK-049 handle entity-level and site-level GHG accounting, PACK-050 provides the corporate group consolidation layer that rolls up entity-level inventories into a single consolidated corporate GHG inventory per the GHG Protocol Corporate Standard Chapter 3.

**PACK-050 is fundamentally different from:**
- **PACK-041** (Scope 1-2 Complete): Calculates emissions for a single entity
- **PACK-049** (Multi-Site Management): Manages physical facility data collection within a single legal entity or across sites
- **PACK-050** handles **legal entity consolidation at the corporate group level**, resolving ownership chains, applying consolidation approaches, eliminating intercompany transactions, handling M&A events, and producing multi-framework consolidated reports

The pack provides a corporate entity registry with full hierarchy management (subsidiaries, JVs, associates, divisions, SPVs), multi-tier ownership chain resolution with equity/control assessment, GHG Protocol Chapter 3 organizational boundary consolidation across all three approaches (equity share, operational control, financial control), intercompany elimination for intra-group energy/waste/product transfers, M&A event handling with pro-rata allocation and base year restatement, consolidation adjustments with approval workflows, multi-framework consolidated reporting (CSRD, SEC, CDP, GRI, SBTi, IFRS S2, PCAF), and a complete audit trail for assurance readiness.

The pack includes 10 engines, 8 workflows, 10 templates, 13 integrations, and 8 presets covering the complete corporate GHG consolidation lifecycle from entity mapping through assured consolidated disclosure.

Every calculation is **zero-hallucination** (deterministic lookups, ownership arithmetic, and rule-based consolidation only -- no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Spreadsheet Approach | PACK-050 GHG Consolidation Pack |
|-----------|-------------------------------|----------------------------------|
| Entity registry | Ad hoc spreadsheet lists, no hierarchy | Structured registry with parent-child hierarchy, lifecycle, metadata (LEI, ISIN, jurisdiction) |
| Ownership resolution | Manual calculation, single-tier only | Multi-tier chain resolution through 5+ holding levels, cross-holdings, iterative algebra |
| Consolidation approach | Single approach, applied inconsistently | All three approaches (equity, operational, financial) calculated simultaneously |
| Approach comparison | Not performed | Side-by-side impact analysis showing difference across all three approaches |
| Intercompany elimination | Manual email survey (<60% capture) | Automated transfer register with matching, verification, and full elimination log |
| M&A handling | Ad hoc, often incorrect | Systematic pro-rata allocation, base year restatement triggers, organic vs. structural separation |
| Multi-framework output | Separate manual preparation per framework | Single consolidation with automated mapping to CSRD, SEC, CDP, GRI, SBTi, IFRS S2 |
| Completeness tracking | Email reminders, manual tracking | Automated entity-level submission tracking with gap estimation and escalation |
| Audit trail | Fragmented across spreadsheets | SHA-256 provenance chain, step-by-step consolidation log, assurance-ready documentation |
| Consolidation time | 4-12 weeks for 100+ entities | <1 day after entity data collected (50x faster) |
| Error rate | 15-30% of consolidated total affected | <0.1% (deterministic, validated, reproducible) |

### 1.4 Consolidation Approach Definitions

| Approach | GHG Protocol Definition | Inclusion Rule | Typical Use Case |
|----------|------------------------|----------------|-----------------|
| Equity Share | Account for emissions proportional to equity ownership percentage | Include X% of entity emissions where X = equity share | Financial institutions, PE funds, IFRS-aligned reporting |
| Operational Control | Account for 100% of emissions from operations over which the company has operational control | 100% if operational control, 0% if not | CSRD/ESRS E1, most industrial corporates, SBTi preferred |
| Financial Control | Account for 100% of emissions from operations over which the company has financial control | 100% if financial control, 0% if not | SEC climate rules, US-listed companies |

### 1.5 Entity Types

| Entity Type | Ownership Range | Consolidation Treatment |
|-------------|----------------|------------------------|
| Wholly-owned subsidiary | 100% | 100% under all three approaches |
| Majority-owned subsidiary | >50% | 100% under control approaches; equity % under equity share |
| Joint Venture (JV) | Typically 50/50 or other splits | Equity % under equity share; 100% or 0% under control (depends on JV agreement) |
| Associate | 20-50% | Equity % under equity share; typically 0% under control approaches |
| Minority investment | <20% | Equity % under equity share; 0% under control approaches |
| Special Purpose Vehicle (SPV) | Variable | Depends on control assessment; may be off-balance-sheet |
| Franchise | 0% equity typically | 0% under equity share; 100% under operational control if franchisor controls operations |
| Division / Business Unit | N/A (internal) | Always 100% (not a separate legal entity) |

### 1.6 Target Users

**Primary:**
- Group sustainability officers responsible for consolidated corporate GHG reporting
- Corporate finance teams performing GHG consolidation aligned with financial consolidation
- Group-level ESG controllers managing multi-entity GHG data collection and reporting
- M&A integration teams handling boundary changes from acquisitions and divestitures

**Secondary:**
- External assurance providers verifying consolidated GHG inventories
- Board-level sustainability committees reviewing consolidated group emissions
- Investor relations teams preparing consolidated climate disclosures (CSRD, SEC, CDP)
- SBTi target-setting teams defining group-level science-based targets
- Private equity ESG teams consolidating portfolio company emissions
- Regulatory compliance officers (CSRD, SEC, California SB 253)

### 1.7 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to consolidate (100+ entities) | <8 hours after entity data collected (vs. 4-12 weeks manual) | Time from last entity submission to consolidated totals |
| Ownership chain accuracy | 100% match with legal/financial records | Cross-validated against corporate secretarial records and financial consolidation |
| Intercompany elimination completeness | >95% of intra-group transfers identified and eliminated | Validated against treasury/intercompany ledger and reconciliation |
| M&A pro-rata accuracy | 100% match with day-count calculation | Cross-validated against financial consolidation pro-rata dates |
| Base year restatement accuracy | 100% compliance with GHG Protocol Chapter 5 | Verified against GHG Protocol worked examples |
| Multi-framework consistency | Zero unexplained differences across frameworks | Reconciliation report showing all differences with methodology explanations |
| Consolidation audit trail completeness | 100% of steps traceable | Assurance provider can reconstruct every consolidation step from log |
| Entity data completeness | >98% of in-scope entities reporting | Submission tracking dashboard with gap estimation |
| Equity share calculation accuracy | Within 0.01% of financial consolidation | Cross-validated against IFRS/GAAP equity method calculations |
| Customer NPS | >50 | Net Promoter Score survey |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| GHG Protocol Corporate Standard | WRI/WBCSD (2015), Chapter 3 | Organizational boundary setting: equity share, operational control, financial control approaches. Core methodology for corporate GHG consolidation |
| GHG Protocol Corporate Standard | WRI/WBCSD (2015), Chapter 5 | Base year recalculation policy for structural changes (acquisitions, divestitures, mergers). Significance threshold (typically 10%) for triggering restatement |
| ISO 14064-1:2018 | Clause 5: Organizational boundaries | Requires organization to define boundaries, identify facilities/operations, determine consolidation approach. Aligned with GHG Protocol approaches |
| EU CSRD / ESRS E1 | Delegated Regulation 2023/2772 | E1-6: GHG emissions reported at consolidated group level using operational control. Requires Scope 1, 2, 3 for the reporting entity per financial consolidation scope |
| US SEC Climate Rules | 17 CFR Parts 210, 229, 230, 232, 239, 249 | GHG emissions at the registrant level, aligned with financial reporting entity. Financial control approach implied |

### 2.2 Supporting Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| IFRS S2 Climate Disclosures | ISSB (2023) | Climate disclosure at the general-purpose financial reporting entity level. Consolidated GHG reporting consistent with financial consolidation |
| GRI 305: Emissions | GRI (2016, updated 2024) | GRI 305-1/2/3 at consolidated organizational level. Permits any consolidation approach with disclosure |
| CDP Climate Change | CDP (2024/2025) | C0.3 organizational boundary, C1.1 consolidation approach, C6 emissions by scope at group level |
| SBTi Corporate Framework | SBTi (2024) | Group-level science-based targets. Boundary must cover at least 95% of Scope 1+2 emissions. Equity share or control approach |
| PCAF Global Standard | PCAF (2022, 3rd edition) | Financed emissions consolidation for financial institutions. Attribution by outstanding amount / total equity+debt |
| GHG Protocol Scope 2 Guidance | WRI/WBCSD (2015) | Dual reporting (location-based and market-based) at consolidated level. Intercompany energy transfer treatment |
| TCFD Recommendations | FSB/TCFD (2017) | Metrics and targets at the organizational level consistent with financial reporting boundaries |
| IAS 27 / IFRS 10 | IASB | Financial consolidation standards that inform GHG consolidation entity boundary and control assessment |
| IAS 28 | IASB | Associates and joint ventures under the equity method -- parallels GHG equity share approach |

### 2.3 Regulatory Consolidation Requirements by Framework

| Framework | Required Approach | Scope Coverage | Entity Boundary | Key Requirement |
|-----------|------------------|----------------|-----------------|-----------------|
| CSRD/ESRS E1 | Operational control | Scope 1, 2, 3 | Financial consolidation scope + upstream/downstream value chain | Mandatory for EU large companies and listed SMEs from 2025 |
| SEC Climate | Financial control (implied) | Scope 1, 2 (Scope 3 if material) | SEC registrant entity | Phased-in for large accelerated filers |
| CDP | Any (disclose which) | Scope 1, 2, 3 | Self-selected boundary | Annual questionnaire |
| SBTi | Equity share or control | Scope 1, 2 (95%+), Scope 3 (67%+) | Target boundary covering 95% of S1+S2 | Near-term and long-term targets |
| GRI 305 | Any (disclose which) | Scope 1, 2, 3 | Self-selected boundary | Biennial or annual reporting |
| IFRS S2 | Consistent with financial | Scope 1, 2, 3 | General-purpose financial reporting entity | Effective for annual periods beginning 2025+ |
| PCAF | Attribution factor | Scope 1, 2, 3 of investees | Loans and investments portfolio | Financial institution-specific |
| California SB 253 | TBD (expected operational) | Scope 1, 2, 3 | Entity with $1B+ revenue in California | Effective 2026 (Scope 1, 2), 2027 (Scope 3) |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 10 | Corporate entity management, ownership resolution, consolidation calculation, and audit engines |
| Workflows | 8 | Multi-phase orchestration workflows for consolidation lifecycle |
| Templates | 10 | Report, dashboard, and disclosure templates |
| Integrations | 13 | Agent, pack, data, and system bridges |
| Presets | 8 | Corporate-structure-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `entity_registry_engine.py` | Corporate entity hierarchy management: subsidiaries, JVs, associates, divisions, SPVs. Entity lifecycle (active, dormant, divested, acquired). Entity metadata (legal name, jurisdiction, ISIN, LEI, ownership chain). Parent-child hierarchy with unlimited nesting depth. Entity classification by type, sector, geography, and business unit. |
| 2 | `ownership_structure_engine.py` | Equity chain resolution through multi-tier holding structures. Control assessment (operational control vs. financial control) per GHG Protocol criteria. Ownership percentage calculation with direct and indirect holdings. Multi-tier effective equity computation (e.g., 80% x 60% x 40% = 19.2%). Cross-holding iterative resolution. Minority interest identification. JV partner mapping. Voting rights vs. economic interest differentiation. |
| 3 | `boundary_consolidation_engine.py` | GHG Protocol Chapter 3 organizational boundary consolidation. Three simultaneous approaches: equity share, operational control, financial control. Materiality threshold screening (configurable, default 1% of group emissions). Boundary inclusion/exclusion decisions with justification. Boundary lock and versioning. Change management with approval workflows. Side-by-side approach impact comparison. |
| 4 | `equity_share_engine.py` | Equity share approach calculations. Proportional allocation of entity emissions based on effective equity ownership percentage. Multi-tier equity chain resolution through unlimited holding levels. JV equity split with partner attribution. Associate emissions (20-50% ownership) proportional inclusion. Minority investment (<20%) proportional inclusion. Currency of equity vs. control distinction. Reconciliation to financial equity method. |
| 5 | `control_approach_engine.py` | Operational control and financial control approach implementation. Binary 100%/0% inclusion logic based on control assessment. Operational control criteria: authority to introduce and implement operating policies, authority to make key decisions about operations. Financial control criteria: ability to direct the financial and operating policies of the operation with a view to gaining economic benefits. Franchise boundary decisions. Outsourced operations boundary decisions. Leased asset boundary decisions per GHG Protocol. |
| 6 | `intercompany_elimination_engine.py` | Double-counting elimination for intra-group transfers. Energy transfers: electricity, steam, heat, cooling sold between group entities (Scope 1 seller reclassified, Scope 2 buyer eliminated). Waste transfers: waste sent from one group entity to another for treatment. Product flows: intermediate products transferred between manufacturing entities. Transfer register with counterparty matching and verification. Elimination journal with full audit trail. Net group emissions after all eliminations. |
| 7 | `acquisition_divestiture_engine.py` | M&A event handling per GHG Protocol Chapter 5. Acquisition date pro-rata allocation (day-count basis, actual/365). Divestiture removal with pro-rata allocation. Base year recalculation trigger assessment (structural change > significance threshold). Organic vs. structural emissions growth separation. Merger integration (combining two entities into one). Demerger handling (splitting one entity into two). Asset acquisition vs. share acquisition distinction. Restated base year calculation with full audit trail. |
| 8 | `consolidation_adjustment_engine.py` | Manual adjustments, reclassifications, corrections, and late submissions. Adjustment categories: methodology change, error correction, scope reclassification, timing adjustment, data quality upgrade, estimation replacement with actuals. Adjustment approval workflow (preparer -> reviewer -> approver). Adjustment magnitude limits (configurable, auto-escalation above threshold). Adjustment audit trail with before/after values and justification. Reversal capability for incorrect adjustments. |
| 9 | `group_reporting_engine.py` | Consolidated group GHG report generation. Multi-framework output mapped from single consolidated dataset: CSRD/ESRS E1, CDP Climate Change, GRI 305, TCFD Metrics, SEC Climate, SBTi progress, IFRS S2, PCAF. Consolidated Scope 1/2/3 totals with entity-level breakdown. Year-over-year trends with organic vs. structural growth separation. Intensity metrics (revenue, FTE, production-based). Consolidation waterfall (entity sum -> eliminations -> adjustments -> consolidated total). |
| 10 | `consolidation_audit_engine.py` | Complete audit trail for all consolidation steps. Reconciliation: sum of entity emissions vs. consolidated total with variance explanation. Entity-level completeness verification. Boundary inclusion/exclusion decision log. Ownership percentage verification against corporate records. Elimination completeness check. Adjustment log with approval status. Sign-off tracking (entity-level, regional, group). Assurance-ready documentation package. Provenance chain (SHA-256) from entity source data through every consolidation step to final totals. |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `entity_mapping_workflow.py` | 5: EntityDiscovery -> OwnershipChainResolution -> ControlAssessment -> MaterialityScreening -> RegistryLock | Map all group entities, resolve ownership chains, assess control, screen for materiality, and lock the entity registry for the reporting period |
| 2 | `boundary_selection_workflow.py` | 4: ApproachEvaluation -> ImpactAnalysis -> StakeholderApproval -> BoundaryLock | Evaluate all three consolidation approaches, analyze impact differences, obtain stakeholder approval, and lock the boundary definition |
| 3 | `entity_data_collection_workflow.py` | 5: EntityAssignment -> DataRequestDistribution -> SubmissionCollection -> ValidationReview -> GapResolution | Assign reporting responsibilities, distribute data requests, collect submissions, validate data quality, and resolve gaps |
| 4 | `consolidation_execution_workflow.py` | 6: DataGathering -> EquityAdjustment -> ControlAdjustment -> IntercompanyElimination -> AdjustmentApplication -> ConsolidatedTotal | Execute the full consolidation calculation from entity data through equity/control adjustments, eliminations, manual adjustments, to final consolidated totals |
| 5 | `elimination_workflow.py` | 4: TransferIdentification -> MatchingVerification -> EliminationCalculation -> ReconciliationCheck | Identify intra-group transfers, match counterparties, calculate eliminations, and verify reconciliation |
| 6 | `mna_adjustment_workflow.py` | 5: EventCapture -> BoundaryImpactAssessment -> ProRataCalculation -> BaseYearRestatement -> DisclosureGeneration | Capture M&A events, assess boundary impact, calculate pro-rata allocations, restate base year if required, and generate disclosure text |
| 7 | `group_reporting_workflow.py` | 4: DataAggregation -> FrameworkMapping -> ReportGeneration -> QualityAssurance | Aggregate consolidated data, map to framework-specific requirements, generate reports, and perform quality assurance |
| 8 | `full_consolidation_pipeline_workflow.py` | 8: EntityMapping -> BoundarySelection -> DataCollection -> EliminationPrep -> ConsolidationExecution -> MnAAdjustment -> GroupReporting -> AuditFinalization | End-to-end orchestration of the complete consolidation lifecycle |

### 3.4 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `consolidated_ghg_report.py` | MD, HTML, PDF, JSON | Complete group-level consolidated GHG inventory with Scope 1/2/3 totals, entity breakdown, consolidation approach disclosure, and methodology notes |
| 2 | `entity_breakdown_report.py` | MD, HTML, PDF, JSON | Per-entity emission contributions showing each entity's gross emissions and its consolidated contribution (after equity/control adjustment) |
| 3 | `ownership_structure_report.py` | MD, HTML, PDF, JSON | Corporate structure visualization showing parent-child hierarchy, ownership percentages, effective equity, and control assessments |
| 4 | `equity_share_report.py` | MD, HTML, PDF, JSON | Equity share approach calculation details: per-entity equity percentage, effective equity through chain, proportional emissions, and reconciliation |
| 5 | `elimination_log_report.py` | MD, HTML, PDF, JSON | Intercompany elimination details showing all identified intra-group transfers, counterparty matching, elimination amounts, and scope reclassifications |
| 6 | `mna_impact_report.py` | MD, HTML, PDF, JSON | M&A event impact analysis showing acquisition/divestiture dates, pro-rata allocations, base year restatement calculations, and organic vs. structural growth |
| 7 | `scope_breakdown_report.py` | MD, HTML, PDF, JSON | Consolidated Scope 1/2/3 breakdown by category, gas type, geography, business unit, and entity with waterfall from gross to net |
| 8 | `trend_analysis_report.py` | MD, HTML, PDF, JSON | Year-over-year consolidated emission trends with organic vs. structural change separation, intensity metric trends, and target trajectory |
| 9 | `regulatory_disclosure_report.py` | MD, HTML, PDF, JSON | Multi-framework regulatory disclosure template with pre-populated data for CSRD/ESRS E1, CDP, GRI 305, SEC, SBTi, IFRS S2 |
| 10 | `consolidation_dashboard.py` | MD, HTML, JSON | Interactive consolidation dashboard with entity completeness tracker, consolidation waterfall, elimination summary, and sign-off status |

### 3.5 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 10-phase DAG pipeline: EntityRegistrySetup -> OwnershipResolution -> BoundaryDefinition -> DataCollection -> TransferIdentification -> ConsolidationCalculation -> EliminationExecution -> AdjustmentProcessing -> ReportGeneration -> AuditFinalization. Retry with exponential backoff, SHA-256 provenance chain, phase-level caching. |
| 2 | `mrv_bridge.py` | Routes to all 30 AGENT-MRV agents for per-entity emission calculations. MRV agents provide entity-level Scope 1, 2, 3 emissions that PACK-050 consolidates. Bi-directional: entity emissions flow in, consolidated factors and eliminations flow back. |
| 3 | `data_bridge.py` | Routes to 20 AGENT-DATA agents: DATA-002 (Excel/CSV for entity data submissions), DATA-001 (PDF extraction for corporate structure documents), DATA-003 (ERP/Finance for equity data and intercompany ledgers), DATA-010 (Data Quality Profiler), DATA-015 (Cross-Source Reconciliation for entity data verification). |
| 4 | `pack041_bridge.py` | PACK-041 Scope 1-2 Complete integration: imports per-entity Scope 1 and Scope 2 emissions (both location-based and market-based). Primary data source for consolidation. Shares entity registry and emission factor configuration. |
| 5 | `pack042_043_bridge.py` | PACK-042/043 Scope 3 integration: imports per-entity Scope 3 emissions across all 15 categories. Categories 1-8 (upstream) and 9-15 (downstream) per entity flow into consolidated Scope 3 totals. Cross-references intercompany transfers against Scope 3 categories for elimination. |
| 6 | `pack044_bridge.py` | PACK-044 Inventory Management integration: imports entity-level inventory metadata (reporting periods, data quality tiers, completeness scores). Ensures consolidation uses consistent inventory periods across entities. |
| 7 | `pack045_bridge.py` | PACK-045 Base Year Management integration: imports entity-level base year data for base year restatement during M&A events. Shares restatement triggers and significance thresholds. Ensures consolidated base year is restated consistently. |
| 8 | `pack048_bridge.py` | PACK-048 Assurance Prep integration: feeds consolidated audit trail and documentation into assurance preparation workflow. Provides consolidation-specific evidence packages for limited and reasonable assurance engagements. |
| 9 | `pack049_bridge.py` | PACK-049 Multi-Site Management integration: imports site-level data aggregated to entity totals. Sites roll up to entities; entities roll up to consolidated group. Shares site registry and facility classification. |
| 10 | `foundation_bridge.py` | AGENT-FOUND integration: FOUND-003 (Unit Normalizer for emission unit harmonization across entities), FOUND-004 (Assumptions Registry for consolidation assumptions), FOUND-005 (Citations for regulatory references and emission factor sources). |
| 11 | `health_check.py` | 20-category system verification covering all 10 engines, 8 workflows, database connectivity, cache status, MRV bridge, DATA bridge, all pack bridges, authentication, authorization, provenance system, and storage. |
| 12 | `setup_wizard.py` | 8-step guided configuration: group profile, entity hierarchy import, ownership data import, consolidation approach selection, intercompany transfer configuration, M&A event history, reporting framework selection, and output preferences. |
| 13 | `alert_bridge.py` | Alert and notification integration: entity submission deadline reminders, completeness gap alerts, M&A boundary change notifications, consolidation variance warnings, sign-off reminders, and assurance timeline alerts. Supports email, SMS, webhook, and in-app channels. |

### 3.6 Presets

| # | Preset | Corporate Structure | Key Characteristics |
|---|--------|---------------------|---------------------|
| 1 | `corporate_conglomerate.yaml` | Multi-subsidiary conglomerate (100+ entities) | Deep hierarchy (4-5 tiers), multiple sectors, global operations, complex intercompany transfers, frequent M&A. All 10 engines enabled. Operational control approach primary. |
| 2 | `financial_holding.yaml` | Financial holding company (banks, insurance, asset managers) | Equity share approach primary (PCAF alignment), portfolio of financial investments, associate-heavy structure, financed emissions consolidation, regulatory capital alignment. |
| 3 | `jv_partnership.yaml` | Joint venture heavy structure | Multiple JVs with different partners, split operational control, complex equity chains, JV-specific elimination rules, partner-level reporting. Both equity and control approaches calculated. |
| 4 | `multinational.yaml` | Multi-national corporation with regional subsidiaries | Regional holding structures (EMEA, APAC, Americas), country-level subsidiaries, regional consolidation layers, multi-currency equity, cross-border intercompany transfers. |
| 5 | `private_equity.yaml` | PE portfolio company consolidation | Portfolio of 10-50 companies, varying ownership percentages, frequent acquisitions/exits, fund-level consolidation, LP reporting requirements, SFDR alignment. Equity share approach primary. |
| 6 | `real_estate_fund.yaml` | REIT / real estate fund | Property portfolio consolidation, landlord-tenant emission splits, CRREM pathway alignment, asset-level granularity, GRESB reporting, location-based Scope 2 emphasis. |
| 7 | `public_company.yaml` | Listed public company (SEC + CSRD) | Dual reporting (SEC financial control + CSRD operational control), investor-grade assurance requirements, quarterly emissions tracking aligned with financial quarters, proxy statement climate disclosure. |
| 8 | `sme_group.yaml` | SME group (simplified consolidation) | 5-20 entities, single-tier hierarchy, simplified elimination (minimal intercompany), no M&A complexity, single consolidation approach, streamlined reporting. 6 engines enabled (skip M&A, simplify elimination). |

---

## 4. Engine Specifications

### 4.1 Engine 1: Entity Registry Engine

**Purpose:** Comprehensive corporate entity hierarchy management with lifecycle tracking, metadata management, and classification.

**Entity Data Model:**

| Field | Type | Description |
|-------|------|-------------|
| `entity_id` | UUID | Unique identifier |
| `legal_name` | str | Full legal entity name |
| `trading_name` | str | Trading / brand name (optional) |
| `entity_type` | enum | SUBSIDIARY, JV, ASSOCIATE, DIVISION, SPV, FRANCHISE, HOLDING, PARENT |
| `jurisdiction` | str | Country/state of incorporation (ISO 3166-1 alpha-2) |
| `lei` | str | Legal Entity Identifier (20-char, optional) |
| `isin` | str | International Securities Identification Number (optional) |
| `registration_number` | str | Local company registration number |
| `parent_entity_id` | UUID | Parent entity in hierarchy (null for ultimate parent) |
| `hierarchy_level` | int | Depth in hierarchy tree (0 = ultimate parent) |
| `lifecycle_status` | enum | ACTIVE, DORMANT, ACQUIRED, DIVESTED, MERGED, LIQUIDATED, PLANNED |
| `effective_date` | date | Date entity entered current status |
| `sector_gics` | str | GICS sector classification |
| `sector_nace` | str | NACE sector classification |
| `country` | str | Primary operating country |
| `region` | str | Geographic region (EMEA, APAC, Americas, etc.) |
| `business_unit` | str | Internal business unit assignment |
| `reporting_currency` | str | Entity reporting currency (ISO 4217) |
| `fiscal_year_end` | str | Fiscal year end date (MM-DD) |
| `consolidation_group` | str | Financial consolidation group (if different from GHG) |

**Hierarchy Operations:**

| Operation | Description | Complexity |
|-----------|-------------|------------|
| `build_hierarchy_tree()` | Construct full parent-child tree from flat entity list | O(n) |
| `get_descendants(entity_id)` | Return all children, grandchildren, etc. of an entity | O(n) tree traversal |
| `get_ancestors(entity_id)` | Return parent chain up to ultimate parent | O(d) where d = depth |
| `get_siblings(entity_id)` | Return all entities sharing same parent | O(1) lookup |
| `move_entity(entity_id, new_parent_id)` | Re-parent an entity (M&A restructuring) | O(1) + validation |
| `validate_hierarchy()` | Check for cycles, orphans, and integrity | O(n) |
| `count_by_level()` | Count entities at each hierarchy level | O(n) |

**Lifecycle State Machine:**

```
PLANNED -> ACTIVE -> DORMANT -> ACTIVE (reactivation)
PLANNED -> ACTIVE -> DIVESTED
PLANNED -> ACTIVE -> MERGED (into another entity)
PLANNED -> ACTIVE -> LIQUIDATED
ACQUIRED -> ACTIVE (new acquisition enters as active)
```

**Key Models:**
- `EntityInput` - Legal name, entity type, jurisdiction, parent, ownership data, sector, geography
- `EntityRecord` - Complete entity record with hierarchy position, lifecycle, metadata
- `HierarchyTree` - Tree structure with parent-child relationships, depth tracking, traversal methods
- `EntityLifecycleEvent` - Status change record with effective date, reason, approver

**Non-Functional Requirements:**
- Entity registry operations (CRUD): <100 milliseconds
- Hierarchy tree construction (1,000 entities): <2 seconds
- Hierarchy validation: <5 seconds for 1,000 entities
- Reproducibility: bit-perfect (SHA-256 verified)

### 4.2 Engine 2: Ownership Structure Engine

**Purpose:** Resolve multi-tier ownership chains, calculate effective equity percentages, and assess operational/financial control for each entity.

**Ownership Data Model:**

| Field | Type | Description |
|-------|------|-------------|
| `owner_entity_id` | UUID | Entity that owns the stake |
| `owned_entity_id` | UUID | Entity being owned |
| `direct_equity_pct` | Decimal | Direct equity ownership percentage (0.00-100.00) |
| `voting_rights_pct` | Decimal | Voting rights percentage (may differ from equity) |
| `economic_interest_pct` | Decimal | Economic interest percentage (for preferred equity) |
| `instrument_type` | enum | ORDINARY_SHARES, PREFERENCE_SHARES, CONVERTIBLE, PARTNERSHIP_INTEREST, MEMBERSHIP_INTEREST |
| `effective_date` | date | Date ownership became effective |
| `end_date` | date | Date ownership ended (null if current) |
| `source_document` | str | Reference to ownership evidence (shareholder register, JV agreement) |

**Multi-Tier Equity Resolution:**

```
Effective_Equity(Parent, Entity) = Product of direct equity percentages along ownership chain

Example:
  Parent owns 80% of Sub-A
  Sub-A owns 60% of Sub-B
  Sub-B owns 40% of JV-C

  Effective_Equity(Parent, Sub-A) = 80%
  Effective_Equity(Parent, Sub-B) = 80% x 60% = 48%
  Effective_Equity(Parent, JV-C) = 80% x 60% x 40% = 19.2%

For multiple ownership paths (Parent owns Sub-A and Sub-D; both own stakes in Sub-E):
  Effective_Equity(Parent, Sub-E) = Sum of (Effective_Equity along each path)
  Path 1: Parent -> Sub-A -> Sub-E = 80% x 30% = 24%
  Path 2: Parent -> Sub-D -> Sub-E = 70% x 20% = 14%
  Total Effective_Equity(Parent, Sub-E) = 24% + 14% = 38%
```

**Cross-Holding Resolution:**

When entities own shares in each other (circular ownership), effective equity must be resolved iteratively:

```
If A owns x% of B and B owns y% of A:
  Effective_A_in_B = x / (1 - x*y)  (geometric series convergence)
  Effective_B_in_A = y / (1 - x*y)

For complex cross-holdings: iterative matrix resolution until convergence (delta < 0.001%)
```

**Control Assessment Criteria:**

| Control Type | Criteria (GHG Protocol) | Assessment Method |
|-------------|------------------------|-------------------|
| Operational Control | Organization has full authority to introduce and implement its operating policies at the operation | Checklist: (1) appoints management, (2) sets operating policies, (3) directs day-to-day operations |
| Financial Control | Organization has ability to direct the financial and operating policies of the operation with a view to gaining economic benefits | Checklist: (1) majority voting rights, (2) right to appoint/remove majority of board, (3) right to cast majority of votes at board, (4) bears majority of risks and rewards |

**Control Decision Matrix:**

| Scenario | Operational Control | Financial Control |
|----------|-------------------|-------------------|
| 100% subsidiary | Yes | Yes |
| >50% subsidiary, parent manages | Yes | Yes |
| >50% subsidiary, independent management | Maybe (assess policies) | Yes |
| 50/50 JV, shared operations | Assess JV agreement | Assess JV agreement |
| 50/50 JV, one partner operates | Operating partner: Yes | Assess financial policies |
| <50% associate, no control | No | No |
| Franchise, franchisor controls operations | Franchisor: Yes | Franchisor: Usually no |
| Outsourced operation | Depends on policy control | Depends on financial direction |

**Key Models:**
- `OwnershipStake` - Direct equity, voting rights, economic interest between two entities
- `EffectiveEquity` - Calculated effective equity from ultimate parent to each entity through all paths
- `ControlAssessment` - Operational and financial control determination with criteria checklist and justification
- `OwnershipChain` - Full resolution of ownership path from parent to entity with intermediate percentages

**Non-Functional Requirements:**
- Effective equity calculation (100 entities, 5-tier max): <5 seconds
- Cross-holding iterative resolution: convergence within 50 iterations (delta < 0.001%)
- Control assessment: <1 second per entity
- Full ownership structure resolution (500 entities): <30 seconds

### 4.3 Engine 3: Boundary Consolidation Engine

**Purpose:** Apply GHG Protocol Chapter 3 organizational boundary consolidation across all three approaches simultaneously, with materiality screening and boundary lock management.

**Consolidation Logic:**

```
For each entity in hierarchy:
  Equity_Share_Emissions = Entity_Emissions * Effective_Equity_Pct / 100

  Operational_Control_Emissions = Entity_Emissions * 1.0 if has_operational_control else 0.0

  Financial_Control_Emissions = Entity_Emissions * 1.0 if has_financial_control else 0.0
```

**Materiality Screening:**

```
Entity is material if:
  Entity_Emissions > Materiality_Threshold * Group_Total_Emissions

Where:
  Materiality_Threshold = configurable (default 1% of group total)

Immaterial entities may be excluded with disclosure, subject to:
  Sum of excluded entities < 5% of group total (GHG Protocol recommendation)
```

**Boundary Versioning:**

| Version Event | Trigger | Action |
|--------------|---------|--------|
| Initial boundary | First consolidation setup | Create version 1.0 |
| Entity addition | Acquisition, new subsidiary | Create new version, document addition |
| Entity removal | Divestiture, liquidation | Create new version, document removal |
| Control change | JV restructuring, management change | Create new version, reassess control |
| Approach change | Stakeholder decision | Create new version, recalculate all |
| Annual refresh | Start of reporting period | Confirm or update boundary |

**Approach Impact Comparison:**

```
For each of the three approaches, calculate:
  - Total Scope 1 (tCO2e)
  - Total Scope 2 location-based (tCO2e)
  - Total Scope 2 market-based (tCO2e)
  - Total Scope 3 (tCO2e)
  - Number of entities included
  - Percentage of group revenue covered
  - Percentage of group FTE covered

Present side-by-side comparison to support approach selection decision.
```

**Key Models:**
- `BoundaryDefinition` - Approach selected, entity inclusion list, materiality threshold, effective date, version
- `EntityBoundaryStatus` - Per-entity: included/excluded, approach-specific treatment, justification
- `ApproachComparison` - Side-by-side totals for all three approaches with delta analysis
- `BoundaryChangeLog` - Version history with change reason, approver, effective date

**Non-Functional Requirements:**
- Boundary calculation (all three approaches, 500 entities): <30 seconds
- Approach comparison: <10 seconds
- Boundary versioning: <1 second per version operation
- Materiality screening (500 entities): <5 seconds

### 4.4 Engine 4: Equity Share Engine

**Purpose:** Implement the equity share consolidation approach with proportional allocation, multi-tier chain resolution, and reconciliation to financial equity method.

**Equity Share Calculation:**

```
Consolidated_Emissions_Equity = Sum over all entities of:
  Entity_Scope1 * Effective_Equity_Pct / 100
  + Entity_Scope2_Location * Effective_Equity_Pct / 100
  + Entity_Scope2_Market * Effective_Equity_Pct / 100
  + Entity_Scope3 * Effective_Equity_Pct / 100

Where Effective_Equity_Pct is calculated by ownership_structure_engine
through all ownership paths.
```

**Entity Type Treatment:**

| Entity Type | Equity Share Treatment | Example |
|-------------|----------------------|---------|
| Wholly-owned subsidiary | 100% of all scope emissions | Parent owns 100% of Sub: include 100% |
| Majority subsidiary | Equity % of all scope emissions | Parent owns 75% of Sub: include 75% |
| Joint Venture | Each partner's equity % | 50/50 JV: each partner includes 50% |
| Associate (20-50%) | Equity % of all scope emissions | Parent owns 30% of Associate: include 30% |
| Minority (<20%) | Equity % of all scope emissions | Parent owns 10% of investment: include 10% |

**JV Partner Attribution:**

```
For a JV with partners A (60%) and B (40%):
  A's share of JV emissions = JV_Total_Emissions * 60%
  B's share of JV emissions = JV_Total_Emissions * 40%

  Verification: A_share + B_share = JV_Total_Emissions (100%)
```

**Reconciliation to Financial Equity Method:**

| Financial Treatment | GHG Equity Share Treatment | Alignment |
|--------------------|---------------------------|-----------|
| Full consolidation (100%) | Equity % (may be <100%) | Different -- financial includes 100%, GHG includes equity % |
| Equity method (IAS 28) | Equity % | Aligned -- both use ownership percentage |
| Proportional consolidation (IAS 31 legacy) | Equity % | Aligned -- both use ownership percentage |
| Fair value (IFRS 9) | Equity % | Different -- financial at fair value, GHG at equity % of emissions |

**Key Models:**
- `EquityShareInput` - Entity list with effective equity percentages and per-entity emissions by scope
- `EquityShareResult` - Per-entity equity-adjusted emissions, consolidated totals by scope, partner attribution for JVs
- `EquityReconciliation` - Comparison of GHG equity share with financial consolidation equity amounts
- `PartnerAttribution` - For JVs: each partner's share of emissions with percentages

**Non-Functional Requirements:**
- Equity share calculation (500 entities): <10 seconds
- JV partner attribution: <1 second per JV
- Reconciliation report: <5 seconds
- Decimal precision: 6 decimal places for percentages, 2 for tCO2e

### 4.5 Engine 5: Control Approach Engine

**Purpose:** Implement operational control and financial control consolidation approaches with binary 100%/0% inclusion logic and systematic control assessment.

**Control Approach Calculation:**

```
Consolidated_Emissions_Control = Sum over all entities where has_control == True:
  Entity_Scope1 * 1.0 (100% inclusion)
  + Entity_Scope2_Location * 1.0
  + Entity_Scope2_Market * 1.0
  + Entity_Scope3 * 1.0

Entities where has_control == False: 0% inclusion (excluded entirely)
```

**Operational Control Assessment Checklist:**

| Criterion | Weight | Assessment |
|-----------|--------|------------|
| Authority to introduce operating policies | 30% | Yes / No / Partial |
| Authority to implement operating policies | 25% | Yes / No / Partial |
| Authority to make key operational decisions | 20% | Yes / No / Partial |
| Day-to-day operational management | 15% | Yes / No / Partial |
| Ability to set environmental policies | 10% | Yes / No / Partial |

```
Operational_Control_Score = Weighted sum of criteria (Yes=1, Partial=0.5, No=0)
Has_Operational_Control = Score >= 0.5 (configurable threshold)
```

**Financial Control Assessment Checklist:**

| Criterion | Weight | Assessment |
|-----------|--------|------------|
| Majority of voting rights (>50%) | 30% | Yes / No |
| Right to appoint/remove majority of board | 25% | Yes / No |
| Right to cast majority of votes at board meetings | 20% | Yes / No |
| Bears majority of risks and rewards | 25% | Yes / No |

```
Financial_Control_Score = Weighted sum of criteria (Yes=1, No=0)
Has_Financial_Control = Score >= 0.5 (configurable threshold)
```

**Special Entity Treatment:**

| Scenario | Operational Control | Financial Control | Notes |
|----------|-------------------|-------------------|-------|
| Franchise (franchisor) | Include if franchisor controls operations | Usually exclude (no equity) | GHG Protocol: disclose approach |
| Franchise (franchisee) | Include own operations | Include own operations | Franchisee reports its own emissions |
| Outsourced manufacturing | Include if company controls process | Exclude if contractor bears risk | Assess on case-by-case basis |
| Leased asset (lessee, operating lease) | Include if lessee controls use | Exclude (lessor has financial control) | GHG Protocol: follow lease type |
| Leased asset (lessee, finance lease) | Include if lessee controls use | Include (finance lease = control) | Treat as owned for financial control |
| SPV (consolidated) | Depends on operating structure | Include if consolidating entity | Follow financial consolidation |
| SPV (off-balance-sheet) | Assess operational role | Exclude | May need disclosure |

**Key Models:**
- `ControlAssessmentInput` - Entity ID, operational criteria responses, financial criteria responses, supporting documentation
- `ControlAssessmentResult` - Per-entity: operational control (yes/no/score), financial control (yes/no/score), justification
- `ControlApproachResult` - Consolidated totals under operational control and financial control with entity inclusion list
- `SpecialEntityDecision` - Franchise/lease/outsourced entity treatment decision with rationale

**Non-Functional Requirements:**
- Control assessment (per entity): <500 milliseconds
- Control approach consolidation (500 entities): <10 seconds
- Assessment audit trail: complete decision log with criteria responses

### 4.6 Engine 6: Intercompany Elimination Engine

**Purpose:** Identify and eliminate double-counted emissions from intra-group transfers of energy, waste, and products.

**Transfer Types:**

| Transfer Type | Seller Treatment (Entity Level) | Buyer Treatment (Entity Level) | Group Elimination |
|--------------|-------------------------------|-------------------------------|-------------------|
| Electricity (internal generation) | Scope 1 (fuel combustion for generation) | Scope 2 (purchased electricity) | Eliminate buyer's Scope 2; keep seller's Scope 1 as group Scope 1 |
| Steam / heat (internal CHP) | Scope 1 (fuel for CHP) | Scope 2 (purchased steam/heat) | Eliminate buyer's Scope 2; keep seller's Scope 1 |
| Cooling (internal chiller plant) | Scope 1 or 2 (energy for cooling) | Scope 2 (purchased cooling) | Eliminate buyer's Scope 2; keep seller's Scope 1/2 |
| Waste (intra-group treatment) | Scope 3 Cat 5 (waste disposal) | Scope 1 (waste treatment) | Eliminate sender's Scope 3 Cat 5; keep treater's Scope 1 |
| Intermediate products | Scope 3 Cat 1 (purchased goods) | Scope 1/2 (production) | Eliminate buyer's Scope 3 Cat 1 for intra-group portion |
| Transport (intra-group logistics) | Scope 3 Cat 4/9 (transport) | Scope 1 (own fleet) | Eliminate shipper's Scope 3; keep fleet's Scope 1 |

**Transfer Register:**

| Field | Type | Description |
|-------|------|-------------|
| `transfer_id` | UUID | Unique transfer identifier |
| `seller_entity_id` | UUID | Entity providing the energy/waste/product |
| `buyer_entity_id` | UUID | Entity receiving the energy/waste/product |
| `transfer_type` | enum | ELECTRICITY, STEAM, HEAT, COOLING, WASTE, PRODUCT, TRANSPORT |
| `quantity` | Decimal | Transfer quantity (kWh, tonnes, m3, etc.) |
| `unit` | str | Unit of measure |
| `transfer_period` | str | Reporting period (YYYY or YYYY-MM) |
| `seller_scope` | str | Scope classification at seller entity level |
| `buyer_scope` | str | Scope classification at buyer entity level |
| `elimination_amount_tco2e` | Decimal | tCO2e to be eliminated |
| `matched_status` | enum | IDENTIFIED, MATCHED, VERIFIED, ELIMINATED |
| `evidence_ref` | str | Reference to supporting documentation |

**Matching and Verification:**

```
For each identified transfer:
  1. Match seller transfer record to buyer transfer record
  2. Verify quantities match within tolerance (default 5%)
  3. If mismatch: flag for manual resolution
  4. If matched: calculate elimination amount in tCO2e
  5. Apply elimination to consolidated totals
  6. Log elimination with full audit trail
```

**Scope Reclassification at Group Level:**

```
Under Operational Control approach:
  If both seller and buyer are within the boundary:
    - Seller's Scope 1 remains Scope 1 at group level
    - Buyer's Scope 2 from this transfer is ELIMINATED (not reclassified)
    - Net effect: group Scope 1 includes generation; group Scope 2 excludes internal purchase

Under Equity Share approach:
  - Seller's contribution: Equity_Pct * Scope 1 from generation
  - Buyer's contribution: Equity_Pct * Scope 2 from purchase
  - Elimination: remove the double-counted portion
  - Elimination_Amount = min(Seller_Equity_Scope1, Buyer_Equity_Scope2) for the transfer
```

**Key Models:**
- `TransferRecord` - Seller, buyer, type, quantity, period, scope classifications
- `TransferMatch` - Matched pair of seller/buyer records with variance analysis
- `EliminationEntry` - Transfer ID, elimination amount by scope, elimination type, verification status
- `EliminationSummary` - Total eliminations by type, scope impact, net consolidated adjustment

**Non-Functional Requirements:**
- Transfer matching (1,000 transfer records): <30 seconds
- Elimination calculation: <5 seconds for 500 matched pairs
- Matching tolerance: configurable (default 5% quantity variance)
- Audit trail: complete log of every elimination decision

### 4.7 Engine 7: Acquisition & Divestiture Engine

**Purpose:** Handle M&A events per GHG Protocol Chapter 5, including pro-rata allocation, base year restatement triggers, and organic vs. structural growth separation.

**M&A Event Types:**

| Event Type | Description | Current Year Impact | Base Year Impact |
|-----------|-------------|--------------------|-----------------|
| Acquisition (share purchase) | Parent acquires equity in existing entity | Add pro-rata emissions from acquisition date | Add 100% to base year if significant |
| Acquisition (asset purchase) | Parent acquires specific assets/operations | Add pro-rata emissions from transfer date | Add to base year if significant |
| Divestiture | Parent sells equity or assets | Remove pro-rata emissions from divestiture date | Remove from base year if significant |
| Merger | Two entities combine into one | Combine emissions; may change entity structure | Restate base year to include combined entity |
| Demerger | One entity splits into two or more | Allocate emissions to successor entities | Restate base year if boundary changes |
| Internal restructuring | Entities reorganized within group | No impact on consolidated total (wash) | No base year restatement (no boundary change) |

**Pro-Rata Allocation:**

```
Pro_Rata_Factor = Days_In_Boundary / Days_In_Year

For acquisition on July 15 in a non-leap year:
  Days_In_Boundary = 365 - 195 = 170 days (July 15 to Dec 31)
  Pro_Rata_Factor = 170 / 365 = 0.4658

Current_Year_Addition = Acquired_Entity_Annual_Emissions * Pro_Rata_Factor

For divestiture on March 31:
  Days_In_Boundary = 90 days (Jan 1 to Mar 31)
  Pro_Rata_Factor = 90 / 365 = 0.2466

Current_Year_Removal = Divested_Entity_Annual_Emissions * Pro_Rata_Factor
```

**Base Year Restatement Trigger (GHG Protocol Chapter 5):**

```
Structural_Change_Pct = ABS(Structural_Change_Emissions) / Base_Year_Emissions * 100

If Structural_Change_Pct > Significance_Threshold (default 10%):
  TRIGGER base year restatement
  Restated_Base_Year = Original_Base_Year + Acquisition_Emissions - Divestiture_Emissions

If Structural_Change_Pct <= Significance_Threshold:
  No restatement required (disclose the structural change)
```

**Organic vs. Structural Growth Separation:**

```
Total_Change = Current_Year_Emissions - Prior_Year_Emissions

Structural_Change = Sum of:
  + Acquisition pro-rata additions
  - Divestiture pro-rata removals
  +/- Merger/demerger impacts

Organic_Change = Total_Change - Structural_Change

Organic_Change_Pct = Organic_Change / (Prior_Year_Emissions + Structural_Change) * 100
```

**Key Models:**
- `MnAEvent` - Event type, entity involved, effective date, equity percentage acquired/divested, estimated annual emissions
- `ProRataCalculation` - Event, days in boundary, pro-rata factor, pro-rata emissions by scope
- `BaseYearRestatement` - Original base year, structural changes, restated base year, restatement trigger assessment
- `OrganicStructuralSplit` - Total change, structural component, organic component, growth rates

**Non-Functional Requirements:**
- Pro-rata calculation: <1 second per event
- Base year restatement: <10 seconds (including all historical events)
- Growth separation: <5 seconds
- Day-count accuracy: exact day count (actual/365 or actual/366 for leap years)

### 4.8 Engine 8: Consolidation Adjustment Engine

**Purpose:** Manage manual adjustments, reclassifications, corrections, and late submissions with approval workflows and audit trails.

**Adjustment Categories:**

| Category | Description | Example | Approval Level |
|----------|-------------|---------|----------------|
| Error correction | Fix calculation or data entry error | Entity reported 1,000 tCO2e instead of 10,000 | Reviewer (1-level) |
| Methodology change | Change in emission factor or calculation method | Switch from Tier 1 to Tier 2 for an entity | Approver (2-level) |
| Scope reclassification | Move emissions between scopes | Reclassify CHP output from Scope 1 to Scope 2 | Approver (2-level) |
| Late submission | Entity data received after initial consolidation | Entity Z submits data 2 weeks late | Reviewer (1-level) |
| Data quality upgrade | Replace estimate with actual measurement | Replace estimated electricity with metered data | Reviewer (1-level) |
| Timing adjustment | Align entity fiscal year to group reporting period | Entity with March year-end aligned to December | Approver (2-level) |
| Restatement | Restate prior period for comparability | Base year restatement for acquisition | Director (3-level) |

**Adjustment Workflow:**

```
Preparer creates adjustment:
  - Select category
  - Specify entity, scope, amount (before and after)
  - Provide justification
  - Attach supporting documentation

Reviewer reviews:
  - Verify calculation
  - Check documentation
  - Approve or reject with comments

Approver approves (for 2-level and above):
  - Verify materiality
  - Check policy compliance
  - Final approval or reject

System applies:
  - Update consolidated totals
  - Log adjustment with full audit trail
  - Recalculate affected reports
```

**Adjustment Controls:**

| Control | Rule | Action |
|---------|------|--------|
| Magnitude limit (auto) | Adjustment > 5% of entity total | Auto-escalate to 2-level approval |
| Magnitude limit (manual override) | Adjustment > 10% of entity total | Require director approval + written justification |
| Frequency limit | >3 adjustments per entity per period | Alert to group controller |
| Net impact disclosure | Total adjustments > 2% of consolidated total | Require disclosure in consolidated report |
| Reversal tracking | Reversed adjustments | Log original and reversal; track net effect |

**Key Models:**
- `AdjustmentRequest` - Category, entity, scope, before value, after value, justification, supporting docs
- `AdjustmentApproval` - Request ID, reviewer decision, approver decision, comments, timestamps
- `AdjustmentImpact` - Effect on consolidated totals by scope, percentage impact, disclosure requirement
- `AdjustmentAuditLog` - Complete history of all adjustments with before/after values and approval chain

**Non-Functional Requirements:**
- Adjustment application: <2 seconds (including recalculation of consolidated totals)
- Approval workflow: <500 milliseconds per workflow step
- Audit log query: <1 second for full adjustment history per entity
- Concurrent adjustments: support 50 simultaneous adjustments without conflict

### 4.9 Engine 9: Group Reporting Engine

**Purpose:** Generate consolidated group GHG reports mapped to multiple regulatory and voluntary frameworks from a single consolidated dataset.

**Consolidated Output Structure:**

```
Group Consolidated GHG Inventory:
  Scope 1 Total (tCO2e)
    - By GHG gas: CO2, CH4, N2O, HFCs, PFCs, SF6, NF3
    - By category: Stationary combustion, Mobile combustion, Process, Fugitive, Refrigerant
    - By entity
    - By geography (country, region)
    - By business unit

  Scope 2 Total - Location-Based (tCO2e)
    - By source: Electricity, Steam, Heat, Cooling
    - By entity, geography, business unit

  Scope 2 Total - Market-Based (tCO2e)
    - By source: Electricity, Steam, Heat, Cooling
    - By entity, geography, business unit

  Scope 3 Total (tCO2e) [if Scope 3 data available from PACK-042/043]
    - By category (1-15)
    - By entity, geography, business unit

  Eliminations (tCO2e)
    - By transfer type
    - Net impact on each scope

  Adjustments (tCO2e)
    - By adjustment category
    - Net impact on each scope

  Consolidated Total = Scope 1 + Scope 2 + Scope 3 - Eliminations +/- Adjustments
```

**Multi-Framework Mapping:**

| Framework | Required Data Points | PACK-050 Source |
|-----------|---------------------|-----------------|
| CSRD/ESRS E1 | E1-6: Scope 1, 2 (location + market), 3 by category; E1-4: GHG reduction targets | Consolidated totals by scope; target tracking engine |
| SEC Climate | Scope 1, 2 by registrant; Scope 3 if material; GHG intensity | Consolidated totals; intensity metrics |
| CDP Climate Change | C6.1-C6.5: Scope 1/2/3 by activity, country, business division | Full dimensional breakdown |
| GRI 305 | 305-1 (Scope 1), 305-2 (Scope 2), 305-3 (Scope 3), 305-4 (intensity), 305-5 (reduction) | Consolidated totals; intensity; YoY reduction |
| SBTi | Total Scope 1+2 (boundary must cover 95%); Scope 3 (67% of categories) | Coverage analysis; target progress |
| IFRS S2 | Scope 1, 2, 3 for reporting entity; industry-specific metrics | Consolidated totals; SASB industry metrics |
| PCAF | Scope 1, 2, 3 per asset class; data quality score (1-5) | Portfolio attribution; quality scoring |
| TCFD | Scope 1, 2 mandatory; Scope 3 if appropriate; carbon intensity | Consolidated totals; intensity; scenario analysis linkage |

**Consolidation Waterfall:**

```
Step 1: Entity-Level Sum
  Sum of all entity-level emissions = Gross Entity Total

Step 2: Equity/Control Adjustment
  Apply consolidation approach = Adjusted Total
  Delta = Gross - Adjusted (entities partially included or excluded)

Step 3: Intercompany Eliminations
  Remove double-counted transfers = Post-Elimination Total
  Delta = Adjusted - Post-Elimination

Step 4: Manual Adjustments
  Apply corrections, reclassifications = Final Consolidated Total
  Delta = Post-Elimination - Final

Waterfall: Gross -> Equity/Control -> Eliminations -> Adjustments -> Consolidated
```

**Intensity Metrics:**

```
Revenue_Intensity = Consolidated_Emissions / Group_Revenue (tCO2e / EUR million)
FTE_Intensity = Consolidated_Emissions / Group_FTE (tCO2e / FTE)
Production_Intensity = Consolidated_Emissions / Production_Output (tCO2e / unit)

All intensity metrics calculated for Scope 1, 2, 1+2, and 1+2+3 separately.
Year-over-year intensity trend with organic vs. structural separation.
```

**Key Models:**
- `ConsolidatedReport` - Complete consolidated inventory with all scope breakdowns, dimensional analysis, and methodology notes
- `FrameworkOutput` - Framework-specific disclosure data mapped from consolidated dataset
- `ConsolidationWaterfall` - Step-by-step reconciliation from entity sum to consolidated total
- `IntensityMetrics` - Revenue, FTE, and production intensity by scope with trends

**Non-Functional Requirements:**
- Report generation (500 entities, all frameworks): <5 minutes
- Framework mapping: <30 seconds per framework
- Waterfall calculation: <10 seconds
- Intensity metrics: <5 seconds
- Export formats: MD, HTML, PDF, JSON

### 4.10 Engine 10: Consolidation Audit Engine

**Purpose:** Provide a complete, assurance-ready audit trail for the entire consolidation process with reconciliation, completeness checks, and sign-off tracking.

**Audit Trail Components:**

| Component | Content | Verification |
|-----------|---------|-------------|
| Entity registry | Complete list of entities with lifecycle status and hierarchy | Cross-reference to corporate secretarial records |
| Ownership data | All ownership stakes with effective dates and source documents | Cross-reference to shareholder registers |
| Boundary decisions | Inclusion/exclusion for each entity under selected approach | Justification documented per entity |
| Entity-level emissions | Per-entity Scope 1/2/3 with data quality indicators | Source data traceable via PACK-041/042/043 provenance |
| Equity adjustments | Effective equity calculation for each entity | Math verified against ownership chain |
| Intercompany eliminations | Each elimination with transfer evidence and counterparty match | Transfer register with matched records |
| M&A adjustments | Pro-rata calculations with day-count verification | Event dates cross-referenced to legal completion dates |
| Manual adjustments | All adjustments with approval chain and justification | Approval workflow records |
| Consolidated totals | Final totals with reconciliation to entity sum | Waterfall reconciliation (delta = 0) |

**Reconciliation Check:**

```
Reconciliation:
  Entity_Sum = Sum of all entity-level emissions (before consolidation)
  Equity_Adjustment = Entity_Sum - Equity_Adjusted_Total
  Elimination_Adjustment = Sum of all intercompany eliminations
  Manual_Adjustment = Sum of all approved adjustments

  Expected_Consolidated = Entity_Sum - Equity_Adjustment - Elimination_Adjustment +/- Manual_Adjustment
  Actual_Consolidated = Calculated consolidated total

  Reconciliation_Variance = ABS(Expected - Actual)
  PASS if Reconciliation_Variance < 0.01 tCO2e (rounding tolerance)
  FAIL if variance exceeds tolerance
```

**Completeness Checks:**

| Check | Criteria | Status |
|-------|----------|--------|
| Entity coverage | All active entities have submitted data | Count submitted / count active |
| Scope coverage | All required scopes reported per entity | Scope 1/2 mandatory; Scope 3 per config |
| Period coverage | All entities report for full reporting period | Months reported / months required |
| Elimination coverage | All known intercompany transfers have elimination entries | Transfers matched / transfers identified |
| Approval coverage | All adjustments have required approvals | Approved / total adjustments |
| Sign-off coverage | All required sign-offs obtained | Signed / required |

**Sign-Off Tracking:**

| Level | Signer | Scope | Required Before |
|-------|--------|-------|-----------------|
| Entity-level | Entity sustainability contact | Entity emissions data accuracy | Consolidation execution |
| Regional | Regional sustainability director | Regional subtotal accuracy | Group consolidation |
| Group | Group sustainability officer | Consolidated total accuracy | External reporting |
| Executive | CFO or CEO | Consolidated report for disclosure | Public filing |

**Key Models:**
- `AuditTrail` - Complete log of all consolidation steps with timestamps, actors, and provenance hashes
- `ReconciliationReport` - Entity sum to consolidated total reconciliation with variance analysis
- `CompletenessReport` - Entity, scope, period, elimination, approval, and sign-off coverage metrics
- `SignOffRecord` - Signer, scope, status (pending/signed/rejected), timestamp, comments

**Non-Functional Requirements:**
- Audit trail generation: <2 minutes for complete consolidation (500 entities)
- Reconciliation check: <30 seconds
- Completeness report: <30 seconds
- SHA-256 provenance chain: every calculation step hashed and chained
- Immutable audit log: append-only, no deletion or modification of historical entries

---

## 5. Workflow Specifications

### 5.1 Workflow 1: Entity Mapping Workflow

**Purpose:** Map all group entities, resolve ownership chains, assess control, screen for materiality, and lock the entity registry for the reporting period.

**Phase 1: Entity Discovery**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Import entity list from corporate records | Subsidiary register, JV agreements, investment register | Draft entity list with basic metadata | <30 minutes |
| 1.2 | Validate entity data completeness | Draft entity list | Completeness score per entity, gap report | <5 minutes (auto) |
| 1.3 | Classify entities by type | Entity metadata | Entity type assignment (subsidiary, JV, associate, etc.) | <10 minutes |
| 1.4 | Build parent-child hierarchy | Entity list with parent references | Hierarchy tree with depth levels | <2 minutes (auto) |

**Phase 2: Ownership Chain Resolution**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Import ownership data | Shareholder registers, JV agreements, investment records | Direct ownership stakes per entity pair | <30 minutes |
| 2.2 | Calculate effective equity | Direct ownership stakes, hierarchy tree | Effective equity percentage for each entity from ultimate parent | <5 seconds (auto) |
| 2.3 | Resolve cross-holdings | Cross-holding relationships | Iterative resolution of circular ownership | <10 seconds (auto) |
| 2.4 | Validate ownership totals | Effective equity results | Verification that ownership sums are consistent (no >100% allocation) | <2 seconds (auto) |

**Phase 3: Control Assessment**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Complete operational control checklist | Per-entity criteria responses | Operational control determination (yes/no) per entity | <2 minutes per entity |
| 3.2 | Complete financial control checklist | Per-entity criteria responses | Financial control determination (yes/no) per entity | <2 minutes per entity |
| 3.3 | Review special entities | Franchises, leases, SPVs, outsourced operations | Special entity treatment decisions with justification | <30 minutes |
| 3.4 | Document control assessment rationale | Assessment results | Control decision log with supporting evidence | <5 minutes (auto) |

**Phase 4: Materiality Screening**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Estimate entity emissions (where not yet reported) | Entity metadata, sector benchmarks, revenue | Estimated annual emissions per entity | <10 minutes (auto) |
| 4.2 | Calculate materiality percentage | Estimated emissions, group total estimate | Materiality percentage per entity | <1 minute (auto) |
| 4.3 | Apply materiality threshold | Materiality percentages, threshold (default 1%) | Material/immaterial classification per entity | <1 minute (auto) |
| 4.4 | Verify immaterial exclusion cap | Sum of immaterial entities | Verify sum < 5% of group total | <1 minute (auto) |

**Phase 5: Registry Lock**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 5.1 | Review entity registry for completeness | Complete entity registry with ownership and control | Review summary report | <15 minutes |
| 5.2 | Obtain registry approval | Review summary, approver identity | Approval record with timestamp | <5 minutes |
| 5.3 | Lock registry version | Approved registry | Immutable locked version with SHA-256 hash | <1 minute (auto) |
| 5.4 | Generate registry report | Locked registry | Entity registry report with hierarchy visualization | <2 minutes (auto) |

**Acceptance Criteria:**
- [ ] All 5 phases execute sequentially with data passing between phases
- [ ] Entity hierarchy validated for cycles and orphans before proceeding
- [ ] Effective equity calculated through unlimited holding tiers
- [ ] Cross-holdings resolved iteratively (convergence < 0.001%)
- [ ] Control assessment documented for every entity
- [ ] Materiality screening applied with configurable threshold
- [ ] Registry locked with SHA-256 hash and version number
- [ ] Total workflow duration < 4 hours for 200 entities

### 5.2 Workflow 2: Boundary Selection Workflow

**Purpose:** Evaluate all three consolidation approaches, analyze impact differences, obtain stakeholder approval, and lock the boundary definition.

**Phase 1: Approach Evaluation**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Calculate equity share consolidation | Entity emissions, effective equity | Equity share consolidated totals | <30 seconds (auto) |
| 1.2 | Calculate operational control consolidation | Entity emissions, control assessments | Operational control consolidated totals | <30 seconds (auto) |
| 1.3 | Calculate financial control consolidation | Entity emissions, control assessments | Financial control consolidated totals | <30 seconds (auto) |

**Phase 2: Impact Analysis**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Generate approach comparison | Three approach results | Side-by-side comparison table with deltas | <5 minutes (auto) |
| 2.2 | Assess framework alignment | Approach results, framework requirements | Framework compatibility matrix | <2 minutes (auto) |
| 2.3 | Evaluate practical implications | Approach comparison, data availability | Practicality assessment (data collection burden per approach) | <10 minutes |

**Phase 3: Stakeholder Approval**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Present approach comparison to stakeholders | Impact analysis, framework compatibility | Stakeholder presentation package | <30 minutes |
| 3.2 | Record approach selection decision | Stakeholder decision, rationale | Approach selection record with justification | <10 minutes |

**Phase 4: Boundary Lock**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Define boundary per selected approach | Approach selection, entity list | Per-entity inclusion/exclusion with approach-specific treatment | <5 minutes (auto) |
| 4.2 | Lock boundary version | Boundary definition | Immutable locked version with SHA-256 hash | <1 minute (auto) |
| 4.3 | Generate boundary report | Locked boundary | Boundary definition report with entity list and approach disclosure | <2 minutes (auto) |

**Acceptance Criteria:**
- [ ] All three approaches calculated simultaneously for comparison
- [ ] Impact analysis quantifies difference between approaches in tCO2e and percentage
- [ ] Framework compatibility clearly identifies which approach(es) each framework accepts
- [ ] Boundary locked with version control and SHA-256 provenance
- [ ] Total workflow duration < 2 hours

### 5.3 Workflow 3: Entity Data Collection Workflow

**Purpose:** Assign reporting responsibilities, distribute data requests, collect entity-level emissions data, validate quality, and resolve gaps.

**Phase 1: Entity Assignment**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Assign data owners per entity | Entity registry, organizational contacts | Entity-to-contact assignment map | <30 minutes |
| 1.2 | Define data requirements per entity | Boundary definition, entity type, scope coverage requirements | Per-entity data request specification | <15 minutes |
| 1.3 | Set submission deadlines | Reporting calendar, entity count | Per-entity submission deadline schedule | <10 minutes |

**Phase 2: Data Request Distribution**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Generate data request templates | Per-entity requirements, PACK-041 templates | Customized data request packages | <10 minutes (auto) |
| 2.2 | Distribute requests to data owners | Request packages, contact list | Distribution log with delivery confirmation | <5 minutes (auto) |
| 2.3 | Set reminder schedule | Deadlines, escalation policy | Automated reminder schedule (7-day, 3-day, 1-day) | <2 minutes (auto) |

**Phase 3: Submission Collection**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Receive entity submissions | Data submissions (via API, upload, or manual entry) | Raw entity emission data in staging | Ongoing |
| 3.2 | Track submission status | Submissions received vs. expected | Completeness dashboard (submitted / outstanding / overdue) | Real-time |
| 3.3 | Send reminders and escalations | Outstanding submissions, escalation policy | Automated reminders to data owners and escalation to managers | Automated |

**Phase 4: Validation Review**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Validate data quality per entity | Entity submissions | Per-entity validation report (range checks, YoY variance, unit verification) | <5 minutes per entity (auto) |
| 4.2 | Flag anomalies | Validation results | Anomaly list with severity (critical, warning, info) | <2 minutes (auto) |
| 4.3 | Return data for correction | Flagged anomalies | Correction request to data owner with specific issues | <5 minutes |

**Phase 5: Gap Resolution**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 5.1 | Identify remaining gaps | Submission status, validation results | Gap list with estimated impact on consolidated total | <5 minutes (auto) |
| 5.2 | Apply estimation for missing entities | Gap list, entity metadata, sector benchmarks | Estimated emissions for gap entities (flagged as estimated) | <10 minutes (auto) |
| 5.3 | Finalize entity dataset | All submissions + estimates | Complete entity emission dataset ready for consolidation | <5 minutes (auto) |

**Acceptance Criteria:**
- [ ] All in-scope entities receive data requests with clear requirements and deadlines
- [ ] Real-time completeness dashboard shows submission status
- [ ] Automated reminders sent at configurable intervals
- [ ] Data quality validation catches range errors, unit errors, and YoY anomalies
- [ ] Gap estimation uses defensible methodology with clear flagging
- [ ] Total workflow adaptable to 50-500 entity groups

### 5.4 Workflow 4: Consolidation Execution Workflow

**Purpose:** Execute the full consolidation calculation from entity data through all adjustment layers to final consolidated totals.

**Phase 1: Data Gathering**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Load validated entity emission data | Complete entity dataset | In-memory entity emission matrix (entities x scopes x categories) | <30 seconds |
| 1.2 | Load ownership and boundary data | Locked registry, locked boundary | Consolidation parameters per entity | <10 seconds |

**Phase 2: Equity Adjustment**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Apply equity share adjustment | Entity emissions, effective equity percentages | Equity-share-adjusted emissions per entity | <10 seconds (auto) |
| 2.2 | Calculate equity share consolidated totals | Adjusted emissions | Equity share Scope 1/2/3 totals | <5 seconds (auto) |

**Phase 3: Control Adjustment**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Apply operational control filter | Entity emissions, operational control flags | Operational control consolidated totals | <5 seconds (auto) |
| 3.2 | Apply financial control filter | Entity emissions, financial control flags | Financial control consolidated totals | <5 seconds (auto) |

**Phase 4: Intercompany Elimination**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Load transfer register | Verified intercompany transfers | Transfer data for elimination | <10 seconds |
| 4.2 | Calculate eliminations | Transfer register, consolidation approach | Elimination amounts by scope and type | <30 seconds (auto) |
| 4.3 | Apply eliminations to consolidated totals | Pre-elimination totals, elimination amounts | Post-elimination consolidated totals | <5 seconds (auto) |

**Phase 5: Adjustment Application**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 5.1 | Load approved adjustments | Approved adjustment records | Adjustment amounts by scope | <5 seconds |
| 5.2 | Apply adjustments | Post-elimination totals, adjustments | Post-adjustment consolidated totals | <5 seconds (auto) |

**Phase 6: Consolidated Total**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 6.1 | Finalize consolidated totals | Post-adjustment totals | Final consolidated Scope 1/2/3 totals | <2 seconds (auto) |
| 6.2 | Generate consolidation waterfall | All intermediate totals | Step-by-step waterfall from entity sum to consolidated total | <5 seconds (auto) |
| 6.3 | Run reconciliation check | Waterfall, final totals | Reconciliation pass/fail with variance | <5 seconds (auto) |
| 6.4 | Calculate provenance hash | All inputs, intermediate results, final totals | SHA-256 hash chain for complete consolidation | <10 seconds (auto) |

**Acceptance Criteria:**
- [ ] All 6 phases execute sequentially with correct data handoff
- [ ] Equity share, operational control, and financial control all calculated
- [ ] Intercompany eliminations correctly applied per consolidation approach
- [ ] Reconciliation check passes (variance < 0.01 tCO2e)
- [ ] SHA-256 provenance hash chain covers all steps
- [ ] Total workflow duration < 5 minutes for 500 entities

### 5.5 Workflow 5: Elimination Workflow

**Purpose:** Identify, match, verify, and execute intercompany eliminations with reconciliation.

**Phase 1: Transfer Identification**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Import intercompany ledger data | ERP intercompany transaction records, energy billing records | Raw transfer list | <15 minutes |
| 1.2 | Classify transfers by type | Raw transfers | Classified transfers (electricity, steam, waste, product, transport) | <5 minutes (auto) |
| 1.3 | Filter to in-scope entities | Classified transfers, boundary definition | In-scope transfers requiring elimination | <2 minutes (auto) |

**Phase 2: Matching Verification**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Match seller-buyer pairs | In-scope transfers | Matched pairs with quantity comparison | <10 minutes (auto) |
| 2.2 | Flag unmatched transfers | Matching results | Unmatched transfer list for investigation | <2 minutes (auto) |
| 2.3 | Resolve mismatches | Unmatched transfers, counterparty data | Resolved or excluded transfers with justification | <30 minutes (manual) |

**Phase 3: Elimination Calculation**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Calculate elimination amounts | Matched transfers, emission factors | tCO2e elimination per transfer | <5 minutes (auto) |
| 3.2 | Apply scope reclassification | Elimination amounts, consolidation approach | Scope-level impact (which scope reduced, by how much) | <2 minutes (auto) |

**Phase 4: Reconciliation Check**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Verify elimination completeness | Eliminations vs. known intercompany activity | Completeness percentage and gap analysis | <5 minutes (auto) |
| 4.2 | Cross-check with financial intercompany elimination | GHG eliminations vs. financial intercompany eliminations | Consistency report | <10 minutes |
| 4.3 | Generate elimination log | All eliminations | Complete elimination log with audit trail | <2 minutes (auto) |

**Acceptance Criteria:**
- [ ] All known intercompany transfers identified and classified
- [ ] Counterparty matching verifies quantities within 5% tolerance
- [ ] Unmatched transfers flagged and resolved or excluded with justification
- [ ] Elimination amounts calculated using appropriate emission factors
- [ ] Scope reclassification applied correctly per consolidation approach
- [ ] Reconciliation check against financial intercompany elimination performed

### 5.6 Workflow 6: M&A Adjustment Workflow

**Purpose:** Handle acquisition and divestiture events with pro-rata allocation, base year assessment, and disclosure generation.

**Phase 1: Event Capture**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Record M&A event | Event type, entity, effective date, equity % | M&A event record | <10 minutes |
| 1.2 | Gather entity emission data | Acquired/divested entity annual emissions | Entity emission profile | <30 minutes |

**Phase 2: Boundary Impact Assessment**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Assess boundary change | M&A event, current boundary | Updated boundary with new entity status | <5 minutes (auto) |
| 2.2 | Calculate structural change percentage | Entity emissions vs. base year total | Structural change as % of base year | <2 minutes (auto) |
| 2.3 | Determine restatement trigger | Structural change %, significance threshold | Restatement required (yes/no) with justification | <1 minute (auto) |

**Phase 3: Pro-Rata Calculation**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Calculate pro-rata factor | Event effective date, reporting period | Pro-rata factor (days/365) | <1 second (auto) |
| 3.2 | Calculate pro-rata emissions | Entity annual emissions, pro-rata factor | Pro-rata emissions to add/remove from current year | <5 seconds (auto) |

**Phase 4: Base Year Restatement**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Calculate restated base year (if triggered) | Original base year, entity annual emissions | Restated base year totals | <30 seconds (auto) |
| 4.2 | Calculate organic vs. structural split | Current year, prior year, M&A events | Organic change and structural change separated | <10 seconds (auto) |
| 4.3 | Document restatement rationale | Restatement calculation, GHG Protocol Chapter 5 reference | Restatement disclosure text | <5 minutes |

**Phase 5: Disclosure Generation**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 5.1 | Generate M&A impact report | Pro-rata calculations, restatement results | M&A impact disclosure document | <5 minutes (auto) |
| 5.2 | Update consolidated totals | M&A adjustments | Revised consolidated totals incorporating M&A events | <2 minutes (auto) |

**Acceptance Criteria:**
- [ ] All M&A event types supported (acquisition, divestiture, merger, demerger)
- [ ] Pro-rata allocation uses exact day-count (actual/365 or actual/366)
- [ ] Base year restatement triggered correctly per significance threshold
- [ ] Organic vs. structural growth clearly separated
- [ ] Disclosure text generated with GHG Protocol Chapter 5 references
- [ ] Total workflow duration < 2 hours per M&A event

### 5.7 Workflow 7: Group Reporting Workflow

**Purpose:** Aggregate consolidated data, map to framework-specific requirements, generate reports, and perform quality assurance.

**Phase 1: Data Aggregation**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 1.1 | Load consolidated totals | Final consolidated emissions | Aggregation-ready dataset | <10 seconds |
| 1.2 | Load dimensional data | Entity registry, geography, business units | Dimensional breakdown data | <10 seconds |
| 1.3 | Load historical data | Prior period consolidated totals | Trend comparison dataset | <10 seconds |

**Phase 2: Framework Mapping**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 2.1 | Map to CSRD/ESRS E1 | Consolidated data, ESRS data point requirements | ESRS E1-6 pre-populated disclosure | <30 seconds (auto) |
| 2.2 | Map to CDP Climate Change | Consolidated data, CDP question mapping | CDP C6 pre-populated responses | <30 seconds (auto) |
| 2.3 | Map to GRI 305 | Consolidated data, GRI indicator mapping | GRI 305-1 through 305-5 disclosures | <30 seconds (auto) |
| 2.4 | Map to SEC Climate | Consolidated data, SEC line item mapping | SEC climate disclosure pre-populated | <30 seconds (auto) |
| 2.5 | Map to SBTi / IFRS S2 / PCAF | Consolidated data, framework mappings | Framework-specific outputs | <30 seconds each (auto) |

**Phase 3: Report Generation**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 3.1 | Generate consolidated GHG report | All template inputs | Complete consolidated report (all 10 templates) | <5 minutes (auto) |
| 3.2 | Export in required formats | Generated reports | MD, HTML, PDF, JSON outputs | <3 minutes (auto) |

**Phase 4: Quality Assurance**

| Step | Action | Input | Output | Duration |
|------|--------|-------|--------|----------|
| 4.1 | Cross-check framework outputs | Framework-specific outputs | Consistency verification across frameworks | <5 minutes (auto) |
| 4.2 | Verify YoY trends | Current and prior period data | Trend anomaly detection (>20% change flagged) | <2 minutes (auto) |
| 4.3 | Generate QA report | QA check results | Quality assurance summary with pass/fail per check | <2 minutes (auto) |

**Acceptance Criteria:**
- [ ] All configured frameworks receive mapped outputs
- [ ] Cross-framework consistency verified (same underlying data)
- [ ] All 10 templates generated in all required formats
- [ ] YoY trend analysis flags significant changes for review
- [ ] Total workflow duration < 30 minutes

### 5.8 Workflow 8: Full Consolidation Pipeline Workflow

**Purpose:** End-to-end orchestration of the complete consolidation lifecycle from entity mapping through audit finalization.

**8 Phases:**

```
Phase 1: Entity Mapping (Workflow 1)
  |
Phase 2: Boundary Selection (Workflow 2)
  |
Phase 3: Data Collection (Workflow 3)
  |
Phase 4: Elimination Preparation (Workflow 5)
  |
Phase 5: Consolidation Execution (Workflow 4)
  |
Phase 6: M&A Adjustment [conditional] (Workflow 6)
  [if M&A events exist in reporting period]
  |
Phase 7: Group Reporting (Workflow 7)
  |
Phase 8: Audit Finalization
  - Generate complete audit trail
  - Run all reconciliation checks
  - Collect sign-offs
  - Generate assurance documentation package
  - Lock consolidated results with SHA-256 hash
```

**Acceptance Criteria:**
- [ ] All 8 phases orchestrated with correct dependencies
- [ ] Phase 6 (M&A) conditional -- skipped if no M&A events
- [ ] Full provenance chain from entity source data to final consolidated totals
- [ ] Audit package generated with all supporting documentation
- [ ] Consolidated results locked and immutable after audit finalization
- [ ] Total pipeline duration < 2 days for 200+ entity group (including data collection)

---

## 6. Template Specifications

### 6.1 Template 1: Consolidated GHG Report

**Purpose:** Complete group-level consolidated GHG inventory with full methodology disclosure.

**Sections:**
- Executive summary: group profile, consolidation approach, reporting period, key totals
- Organizational boundary: approach selected, entity coverage, materiality exclusions
- Scope 1 consolidated: by gas, by category, by entity, by geography, by business unit
- Scope 2 consolidated: location-based and market-based, by source, by entity, by geography
- Scope 3 consolidated (if available): by category (1-15), by entity, by geography
- Intercompany eliminations: summary of eliminated amounts by type
- Consolidation waterfall: entity sum through eliminations and adjustments to final total
- Year-over-year comparison: organic vs. structural change
- Intensity metrics: revenue, FTE, production-based intensities
- Methodology notes: emission factors, data quality, estimation methods
- Assurance statement reference

**Output Formats:** MD, HTML, PDF, JSON

### 6.2 Template 2: Entity Breakdown Report

**Purpose:** Per-entity emission contributions showing entity-level detail within the consolidated inventory.

**Sections:**
- Entity ranking by emissions (largest to smallest)
- Per-entity detail: entity name, type, equity %, control status, Scope 1/2/3, contribution percentage
- Entity vs. consolidated contribution comparison (gross vs. equity-adjusted)
- Geographic heatmap data (emissions by country/region)
- Business unit breakdown
- Data quality indicator per entity (measured, calculated, estimated)
- Missing entity analysis (entities excluded or not yet reported)

**Output Formats:** MD, HTML, PDF, JSON

### 6.3 Template 3: Ownership Structure Report

**Purpose:** Corporate structure visualization with ownership chain details.

**Sections:**
- Hierarchy tree visualization data (parent-child with ownership percentages)
- Effective equity table (entity, direct equity, effective equity, ownership path)
- Control assessment summary (entity, operational control, financial control)
- Entity classification matrix (type, jurisdiction, sector, region)
- Cross-holding relationships (if any)
- Changes from prior period (new entities, divested entities, ownership changes)

**Output Formats:** MD, HTML, PDF, JSON

### 6.4 Template 4: Equity Share Report

**Purpose:** Detailed equity share approach calculations and reconciliation.

**Sections:**
- Per-entity equity share calculation: entity emissions x effective equity %
- Equity chain detail: ownership path from parent to each entity with step-by-step multiplication
- JV partner attribution: each partner's share of JV emissions
- Associate and minority investment contributions
- Reconciliation: sum of all equity shares vs. consolidated total
- Comparison to financial equity method totals
- Sensitivity analysis: how consolidated total changes with +/-5% equity changes

**Output Formats:** MD, HTML, PDF, JSON

### 6.5 Template 5: Elimination Log Report

**Purpose:** Complete intercompany elimination documentation for audit.

**Sections:**
- Transfer register: all identified intra-group transfers with type, quantity, counterparties
- Matching status: matched, unmatched, resolved, excluded
- Elimination journal: per-transfer elimination amount in tCO2e with scope impact
- Scope reclassification log: scope changes at group level
- Elimination summary: total eliminations by type and scope
- Unmatched transfer investigation: unresolved transfers with estimated impact
- Reconciliation to financial intercompany eliminations

**Output Formats:** MD, HTML, PDF, JSON

### 6.6 Template 6: M&A Impact Report

**Purpose:** Document M&A event impacts on consolidated inventory and base year.

**Sections:**
- M&A event summary: type, entity, effective date, equity percentage
- Pro-rata calculation detail: day-count, pro-rata factor, emissions added/removed
- Base year restatement assessment: structural change %, significance threshold, trigger decision
- Restated base year (if applicable): original vs. restated with variance
- Organic vs. structural growth analysis: total change decomposed into organic and structural
- Impact on intensity metrics: how M&A affects revenue/FTE/production intensities
- Disclosure text: GHG Protocol Chapter 5 compliant disclosure for annual report

**Output Formats:** MD, HTML, PDF, JSON

### 6.7 Template 7: Scope Breakdown Report

**Purpose:** Detailed Scope 1/2/3 breakdown of consolidated emissions.

**Sections:**
- Scope 1 breakdown: by GHG gas (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3), by source category, by entity, by country
- Scope 2 location-based breakdown: by energy source (electricity, steam, heat, cooling), by entity, by grid region
- Scope 2 market-based breakdown: by energy source, by contractual instrument (grid average, RE certificates, PPAs)
- Scope 3 breakdown: by category (1-15), by entity, top contributing categories
- Total consolidated breakdown: scope 1 + 2 + 3 pie chart and trend
- GHG gas composition: CO2-equivalent contribution by gas type
- Geographic distribution: emissions by country and region
- Consolidation approach impact: how breakdown differs between equity share and control approaches

**Output Formats:** MD, HTML, PDF, JSON

### 6.8 Template 8: Trend Analysis Report

**Purpose:** Year-over-year consolidated emission trends with growth decomposition.

**Sections:**
- Multi-year consolidated totals (up to 10 years where data available)
- Year-over-year change: absolute tCO2e change and percentage change
- Organic vs. structural decomposition: separate organic growth from M&A impact
- Scope-level trends: Scope 1, 2, 3 trends separately
- Intensity metric trends: revenue, FTE, production intensity over time
- Target trajectory: actual vs. target pathway (SBTi, internal targets)
- Base year restatement history: all restatements with reasons
- Forecast: extrapolation of current trend (linear and compound)

**Output Formats:** MD, HTML, PDF, JSON

### 6.9 Template 9: Regulatory Disclosure Report

**Purpose:** Pre-populated multi-framework regulatory disclosure template.

**Sections:**
- CSRD/ESRS E1 disclosure: E1-4 (targets), E1-5 (energy), E1-6 (Scope 1/2/3), E1-7 (removals)
- CDP Climate Change: C0 (introduction), C1 (governance), C6 (emissions), C7 (energy)
- GRI 305: 305-1 (Scope 1), 305-2 (Scope 2), 305-3 (Scope 3), 305-4 (intensity), 305-5 (reduction)
- SEC Climate: Scope 1, 2 by segment, Scope 3 materiality assessment
- SBTi: target summary, progress against near-term and long-term targets
- IFRS S2: industry-specific metrics, Scope 1/2/3, transition plan linkage
- Framework reconciliation: explanation of differences between framework outputs
- Methodology index: emission factors, data sources, calculation methods referenced in each disclosure

**Output Formats:** MD, HTML, PDF, JSON

### 6.10 Template 10: Consolidation Dashboard

**Purpose:** Interactive dashboard for real-time consolidation monitoring and management.

**Dashboard Panels:**

| Panel | Content | Update Frequency |
|-------|---------|-----------------|
| Entity Completeness | Submission status by entity: submitted / outstanding / overdue | Real-time |
| Consolidation Status | Current phase in consolidation pipeline with progress percentage | Real-time |
| Scope Summary | Consolidated Scope 1/2/3 totals with comparison to prior period | On-demand |
| Consolidation Waterfall | Visual waterfall from entity sum to consolidated total | On-demand |
| Elimination Summary | Total eliminations by type with percentage of gross emissions | On-demand |
| Entity Ranking | Top 20 entities by consolidated contribution | On-demand |
| Geographic Map | Emissions by country/region with color coding | On-demand |
| Sign-Off Tracker | Sign-off status by level (entity, regional, group, executive) | Real-time |
| Adjustment Log | Recent adjustments with approval status | Real-time |
| Data Quality | Per-entity data quality scores with overall group quality metric | On-demand |

**Output Formats:** MD, HTML, JSON

---

## 7. Integration Specifications

### 7.1 Integration 1: Pack Orchestrator

**Purpose:** Master orchestration pipeline for all PACK-050 engines.

**DAG Pipeline (10 phases):**

```
Phase 1: Entity Registry Setup (entity_registry_engine)
  |
Phase 2: Ownership Resolution (ownership_structure_engine)
  |
Phase 3: Boundary Definition (boundary_consolidation_engine)
  |
Phase 4: Data Collection (entity_data_collection_workflow via pack bridges)
  |
Phase 5: Transfer Identification (intercompany_elimination_engine -- identification phase)
  |
Phase 6: Consolidation Calculation (equity_share_engine + control_approach_engine)
  |
Phase 7: Elimination Execution (intercompany_elimination_engine -- elimination phase)
  |
Phase 8: Adjustment Processing [conditional] (consolidation_adjustment_engine)
  [if pending adjustments exist]
  |
Phase 9: Report Generation (group_reporting_engine + all templates)
  |
Phase 10: Audit Finalization (consolidation_audit_engine)
```

**Orchestrator Features:**
- Conditional phase execution (Phase 8 skipped if no adjustments pending)
- Retry with exponential backoff (max 3 retries per phase, base delay 5 seconds)
- SHA-256 provenance chain across all phases (each phase output hashed, hash included in next phase input)
- Phase-level caching (skip re-execution if inputs unchanged since last run)
- Progress tracking with percentage completion per phase
- Error isolation (failed optional phase does not block required phases)
- Parallel execution where dependencies allow (Phase 2 and Phase 3 can overlap partially)

### 7.2 Integration 2: MRV Bridge

**Purpose:** Connect to all 30 AGENT-MRV agents for per-entity emission calculations.

**Data Flow:**
- MRV agents calculate per-entity Scope 1, 2, 3 emissions
- PACK-050 consumes entity-level emission totals as input for consolidation
- Key MRV agents:
  - MRV-001 through MRV-008: Scope 1 categories per entity
  - MRV-009/010: Scope 2 (location and market-based) per entity
  - MRV-011/012: Scope 2 (steam, cooling) per entity
  - MRV-013: Dual reporting per entity
  - MRV-014 through MRV-028: Scope 3 categories per entity
  - MRV-029: Category mapper
  - MRV-030: Audit trail and lineage

**Bi-directional:**
- MRV -> PACK-050: entity-level emissions flow into consolidation
- PACK-050 -> MRV: consolidated boundary decisions inform which entities MRV agents should calculate for; elimination decisions affect scope classification

### 7.3 Integration 3: Data Bridge

**Purpose:** Route entity data through AGENT-DATA agents for quality assurance.

**Data Pipeline:**
- Entity financial data -> DATA-002 (Excel/CSV Normalizer) -> standardized entity data
- Corporate structure documents -> DATA-001 (PDF Extractor) -> ownership and entity metadata
- ERP intercompany ledger -> DATA-003 (ERP/Finance Connector) -> intercompany transfer data
- All entity data -> DATA-010 (Data Quality Profiler) -> per-entity quality score
- Cross-entity data -> DATA-015 (Cross-Source Reconciliation) -> reconciled entity dataset

### 7.4 Integration 4: PACK-041 Bridge

**Purpose:** Integration with PACK-041 Scope 1-2 Complete Pack.

**Key Functions:**
- Import per-entity Scope 1 emissions (stationary combustion, mobile combustion, process, fugitive, refrigerant)
- Import per-entity Scope 2 emissions (location-based and market-based, electricity, steam, heat, cooling)
- Share entity registry to ensure consistent entity definitions
- Share emission factor configuration for consistency across entities
- Cross-reference: verify entity emissions totals match between PACK-041 output and PACK-050 input

### 7.5 Integration 5: PACK-042/043 Bridge

**Purpose:** Integration with PACK-042 Scope 3 Starter and PACK-043 Scope 3 Complete.

**Key Functions:**
- Import per-entity Scope 3 emissions across all 15 upstream and downstream categories
- Cross-reference intercompany transfers against Scope 3 categories for elimination identification
- Import Scope 3 data quality tiers per entity per category
- Share entity boundary decisions that affect which Scope 3 categories are relevant

### 7.6 Integration 6: PACK-044 Bridge

**Purpose:** Integration with PACK-044 Inventory Management Pack.

**Key Functions:**
- Import entity-level inventory metadata (reporting periods, completeness, data tiers)
- Ensure consolidation uses consistent inventory periods across entities
- Import inventory configuration (GHGs covered, scope coverage, de minimis thresholds)
- Share consolidated inventory totals back to PACK-044 for inventory management

### 7.7 Integration 7: PACK-045 Bridge

**Purpose:** Integration with PACK-045 Base Year Management Pack.

**Key Functions:**
- Import entity-level base year emissions for base year restatement during M&A
- Import base year policy (significance threshold, recalculation triggers)
- Share M&A-triggered restatement calculations for base year update
- Ensure consolidated base year is restated consistently across all entities

### 7.8 Integration 8: PACK-048 Bridge

**Purpose:** Integration with PACK-048 Assurance Prep Pack.

**Key Functions:**
- Feed consolidated audit trail into assurance preparation workflow
- Provide consolidation-specific evidence packages:
  - Entity registry with ownership verification
  - Boundary definition with approach justification
  - Elimination log with transfer evidence
  - M&A calculations with pro-rata verification
  - Adjustment log with approval chain
  - Reconciliation report (entity sum to consolidated total)
- Support both limited and reasonable assurance engagement scopes

### 7.9 Integration 9: PACK-049 Bridge

**Purpose:** Integration with PACK-049 Multi-Site Management Pack.

**Key Functions:**
- Import site-level data aggregated to entity totals (sites roll up to legal entities)
- Share entity registry to ensure site-to-entity mapping is consistent
- Import site-level data quality scores for entity-level quality assessment
- Coordinate shared services allocation (PACK-049) with intercompany elimination (PACK-050)
- Distinguish site consolidation (PACK-049) from entity consolidation (PACK-050)

### 7.10 Integration 10: Foundation Bridge

**Purpose:** AGENT-FOUND integration for core platform services.

**Agent Connections:**
- FOUND-003 (Unit & Reference Normalizer): harmonize emission units across entities (different entities may report in tCO2e, kgCO2e, MtCO2e)
- FOUND-004 (Assumptions Registry): register and track consolidation assumptions (e.g., assumed emission factors for estimated entities, JV control assumptions)
- FOUND-005 (Citations & Evidence Agent): manage regulatory citations (GHG Protocol Ch 3, Ch 5, ISO 14064-1 Cl 5) and emission factor source references

### 7.11 Integration 11: Health Check

**Purpose:** System verification for all PACK-050 components.

**20 Verification Categories:**

| # | Category | Checks |
|---|----------|--------|
| 1 | Engine availability | All 10 engines respond to health ping |
| 2 | Workflow availability | All 8 workflows respond to health ping |
| 3 | Template availability | All 10 templates generate test output |
| 4 | Database connectivity | PostgreSQL connection, migration status (V416-V425) |
| 5 | Redis cache | Cache connectivity and response time |
| 6 | MRV bridge | Connection to 30 MRV agents |
| 7 | Data bridge | Connection to DATA agents (001, 002, 003, 010, 015) |
| 8 | Foundation bridge | Connection to FOUND agents (003, 004, 005) |
| 9 | PACK-041 bridge | Connection to PACK-041 engines (if deployed) |
| 10 | PACK-042/043 bridge | Connection to PACK-042/043 engines (if deployed) |
| 11 | PACK-044 bridge | Connection to PACK-044 engines (if deployed) |
| 12 | PACK-045 bridge | Connection to PACK-045 engines (if deployed) |
| 13 | PACK-048 bridge | Connection to PACK-048 engines (if deployed) |
| 14 | PACK-049 bridge | Connection to PACK-049 engines (if deployed) |
| 15 | Entity registry | Registry loaded with entity count verification |
| 16 | Authentication | JWT RS256 token issuance/validation |
| 17 | Authorization | RBAC permission checks for all 6 roles |
| 18 | Provenance | SHA-256 hash generation/verification |
| 19 | Audit log | Append-only audit log write/read test |
| 20 | Disk/memory | Storage <80% capacity, memory <80% ceiling |

### 7.12 Integration 12: Setup Wizard

**Purpose:** Guided 8-step configuration for new consolidation deployments.

**Steps:**

| Step | Configuration | Inputs Required |
|------|--------------|-----------------|
| 1. Group Profile | Group name, ultimate parent entity, reporting period, fiscal year | Basic group information |
| 2. Entity Hierarchy Import | Entity list with parent-child relationships (CSV, Excel, or API) | Corporate structure data |
| 3. Ownership Data Import | Equity stakes, voting rights, economic interests per entity pair | Shareholder register data |
| 4. Consolidation Approach | Primary approach selection (equity/operational/financial), secondary approach for comparison | Approach decision |
| 5. Intercompany Transfer Config | Transfer types to track, matching tolerance, elimination rules | Intercompany policy |
| 6. M&A Event History | Historical acquisitions, divestitures, mergers with dates and equity % | M&A transaction records |
| 7. Framework Selection | Which frameworks to generate output for (CSRD, CDP, GRI, SEC, SBTi, IFRS S2) | Reporting requirements |
| 8. Output Preferences | Report formats, distribution list, dashboard preferences, sign-off workflow | Output configuration |

### 7.13 Integration 13: Alert Bridge

**Purpose:** Alert and notification integration for consolidation timeline management.

**Alert Types:**

| Alert | Trigger | Channel | Audience |
|-------|---------|---------|----------|
| Entity submission deadline | Deadline approaching (14-day, 7-day, 1-day) | Email, in-app | Entity data owner |
| Submission overdue | Entity has not submitted by deadline | Email, SMS | Entity data owner + regional director |
| Completeness threshold | Group completeness drops below target (e.g., <90% at 30 days before reporting) | Email | Group sustainability officer |
| M&A boundary change | New acquisition/divestiture event recorded | Email, in-app | Group sustainability officer, M&A team |
| Consolidation variance | Consolidated total differs >10% from prior period | Email | Group controller |
| Elimination mismatch | Intercompany transfer quantities mismatch >5% between counterparties | Email, in-app | Group controller |
| Sign-off reminder | Sign-off deadline approaching (7-day, 3-day) | Email | Designated signer |
| Assurance timeline | Assurance engagement milestone approaching | Email | Assurance coordinator |

---

## 8. Preset Specifications

### 8.1 Preset 1: Corporate Conglomerate

**Corporate Structure:** Multi-subsidiary conglomerate with 100+ entities across multiple sectors
**Primary Approach:** Operational control

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| Hierarchy depth | 4-5 tiers |
| Typical entity count | 100-500 |
| Primary approach | Operational control |
| Secondary approach | Equity share (for comparison) |
| Intercompany complexity | High (energy, waste, products, logistics) |
| M&A frequency | 3-10 events per year |
| Framework output | CSRD, CDP, GRI, SBTi |
| Materiality threshold | 0.5% (lower due to many small entities) |
| Key challenge | Volume of entities, multiple sectors, complex intercompany |

### 8.2 Preset 2: Financial Holding

**Corporate Structure:** Financial holding company (banks, insurance, asset managers)
**Primary Approach:** Equity share

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| Hierarchy depth | 2-3 tiers |
| Typical entity count | 50-200 |
| Primary approach | Equity share |
| Secondary approach | Financial control |
| Intercompany complexity | Low (mainly shared services, limited physical transfers) |
| M&A frequency | 5-15 events per year (portfolio activity) |
| Framework output | CSRD, CDP, PCAF, TCFD, IFRS S2 |
| PCAF-specific | Financed emissions attribution by asset class |
| Key challenge | Large number of financial investments, PCAF compliance, varying data quality |

### 8.3 Preset 3: JV Partnership

**Corporate Structure:** Organization with multiple joint ventures and partnerships
**Primary Approach:** Equity share and operational control (both required)

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| Hierarchy depth | 2-3 tiers |
| Typical entity count | 20-80 |
| Primary approach | Both equity share and operational control calculated |
| JV count | 5-30 JVs with different partners |
| Intercompany complexity | Medium (JV-to-parent transfers, shared operations) |
| M&A frequency | 1-5 events per year |
| Framework output | CSRD, CDP, GRI |
| Key challenge | JV control assessment ambiguity, partner data coordination |

### 8.4 Preset 4: Multinational

**Corporate Structure:** Multi-national corporation with regional holding companies and country-level subsidiaries
**Primary Approach:** Operational control

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| Hierarchy depth | 3-4 tiers (parent -> region -> country -> local entity) |
| Typical entity count | 50-300 |
| Primary approach | Operational control |
| Regional structure | EMEA, APAC, Americas, with regional consolidation layers |
| Intercompany complexity | High (cross-border intercompany, transfer pricing, shared services) |
| M&A frequency | 2-8 events per year |
| Framework output | CSRD, SEC, CDP, GRI, SBTi, IFRS S2 |
| Multi-currency | Equity expressed in multiple currencies (converted to group currency) |
| Key challenge | Regional variation in data quality, cross-border transfers, dual SEC+CSRD |

### 8.5 Preset 5: Private Equity

**Corporate Structure:** PE fund with portfolio companies at varying ownership levels
**Primary Approach:** Equity share (PCAF attribution)

| Parameter | Value |
|-----------|-------|
| Engines enabled | 8 (exclude complex intercompany, simplified elimination) |
| Hierarchy depth | 2 tiers (fund -> portfolio companies) |
| Typical entity count | 10-50 portfolio companies |
| Primary approach | Equity share |
| Portfolio turnover | High (3-8 acquisitions/exits per year) |
| Intercompany complexity | Low (independent portfolio companies) |
| M&A frequency | High (core to PE business model) |
| Framework output | SFDR, PCAF, CDP, SBTi (portfolio-level) |
| LP reporting | Limited partner ESG reporting requirements |
| Key challenge | Rapid portfolio changes, varying data maturity across portfolio companies |

### 8.6 Preset 6: Real Estate Fund

**Corporate Structure:** REIT or real estate fund with property portfolio
**Primary Approach:** Operational control (landlord-controlled properties)

| Parameter | Value |
|-----------|-------|
| Engines enabled | 8 (exclude complex JV logic unless JV properties exist) |
| Hierarchy depth | 2-3 tiers (fund -> SPV -> property) |
| Typical entity count | 20-200 (one SPV per property is common) |
| Primary approach | Operational control |
| Landlord-tenant split | Scope 1/2 split between landlord and tenant per lease type |
| Intercompany complexity | Low (limited inter-property transfers) |
| M&A frequency | 3-10 property acquisitions/disposals per year |
| Framework output | CSRD, GRESB, CDP, CRREM, GRI |
| Key challenge | Landlord-tenant boundary, property-level vs. SPV-level, GRESB reporting |

### 8.7 Preset 7: Public Company

**Corporate Structure:** Listed public company with SEC and/or CSRD obligations
**Primary Approach:** Operational control (CSRD) + Financial control (SEC)

| Parameter | Value |
|-----------|-------|
| Engines enabled | All 10 |
| Hierarchy depth | 3-4 tiers |
| Typical entity count | 50-200 |
| Primary approach | Dual: operational control for CSRD + financial control for SEC |
| Quarterly tracking | Emissions tracked quarterly aligned with financial quarters |
| Intercompany complexity | Medium to high |
| M&A frequency | 1-5 events per year |
| Framework output | CSRD, SEC, CDP, GRI, SBTi, IFRS S2, TCFD |
| Assurance requirement | Limited assurance (moving to reasonable) per CSRD and SEC |
| Key challenge | Dual approach calculation, quarterly cadence, investor-grade assurance |

### 8.8 Preset 8: SME Group

**Corporate Structure:** Small-to-medium enterprise group with 5-20 entities
**Primary Approach:** Operational control (simplified)

| Parameter | Value |
|-----------|-------|
| Engines enabled | 6 (entity_registry, ownership, boundary, equity_share, group_reporting, audit) |
| Hierarchy depth | 1-2 tiers |
| Typical entity count | 5-20 |
| Primary approach | Operational control |
| Intercompany complexity | Minimal (simplified elimination or none) |
| M&A frequency | 0-1 events per year |
| Framework output | GRI, CDP (SME version) |
| Data collection | Simplified templates, reduced data requirements |
| Key challenge | Limited sustainability staffing, need for simplicity |

---

## 9. Database Migrations

Inherits platform migrations V001-V415. Pack-specific migrations:

| Migration | Table | Purpose |
|-----------|-------|---------|
| V416__pack050_entity_registry_001 | `ghg_consol_entities`, `ghg_consol_entity_hierarchy`, `ghg_consol_entity_lifecycle`, `ghg_consol_entity_metadata` | Corporate entity registry with hierarchy tree, lifecycle state machine events, and extended metadata (LEI, ISIN, jurisdiction, sector, geography, business unit) |
| V417__pack050_ownership_002 | `ghg_consol_ownership_stakes`, `ghg_consol_effective_equity`, `ghg_consol_control_assessments`, `ghg_consol_cross_holdings` | Direct ownership stakes between entities, calculated effective equity through multi-tier chains, operational and financial control assessment records with criteria checklists, and cross-holding iterative resolution data |
| V418__pack050_boundary_003 | `ghg_consol_boundary_definitions`, `ghg_consol_entity_boundary_status`, `ghg_consol_approach_comparisons`, `ghg_consol_boundary_versions` | Organizational boundary definitions per approach, per-entity inclusion/exclusion status with justification, side-by-side approach comparison results, and boundary version history with lock records |
| V419__pack050_consolidation_004 | `ghg_consol_entity_emissions`, `ghg_consol_equity_share_results`, `ghg_consol_control_approach_results`, `ghg_consol_consolidated_totals` | Per-entity emission data (Scope 1/2/3 by category, gas, source), equity share proportional allocation results, control approach binary inclusion results, and final consolidated totals by scope with dimensional breakdowns |
| V420__pack050_elimination_005 | `ghg_consol_transfer_register`, `ghg_consol_transfer_matches`, `ghg_consol_elimination_entries`, `ghg_consol_elimination_summaries` | Intercompany transfer register (seller, buyer, type, quantity), counterparty matching records with variance analysis, per-transfer elimination entries with scope impact, and period-level elimination summaries |
| V421__pack050_mna_006 | `ghg_consol_mna_events`, `ghg_consol_prorata_calculations`, `ghg_consol_base_year_restatements`, `ghg_consol_organic_structural_splits` | M&A event records (type, entity, date, equity %), pro-rata day-count calculations with emissions allocation, base year restatement calculations with trigger assessment, and organic vs. structural growth separation |
| V422__pack050_adjustments_007 | `ghg_consol_adjustment_requests`, `ghg_consol_adjustment_approvals`, `ghg_consol_adjustment_impacts`, `ghg_consol_adjustment_audit_log` | Manual adjustment requests with category and justification, multi-level approval workflow records, per-adjustment impact on consolidated totals, and complete adjustment audit trail |
| V423__pack050_reporting_008 | `ghg_consol_reports`, `ghg_consol_framework_outputs`, `ghg_consol_waterfalls`, `ghg_consol_intensity_metrics` | Generated consolidated report records, framework-specific output mappings (CSRD, SEC, CDP, GRI, SBTi, IFRS S2, PCAF), consolidation waterfall step-by-step records, and intensity metric calculations |
| V424__pack050_audit_009 | `ghg_consol_audit_trails`, `ghg_consol_reconciliations`, `ghg_consol_completeness_checks`, `ghg_consol_signoffs` | Immutable append-only audit trail log with SHA-256 provenance, reconciliation check records (entity sum vs. consolidated total), entity/scope/period completeness verification, and multi-level sign-off tracking |
| V425__pack050_config_010 | `ghg_consol_configurations`, `ghg_consol_presets`, `ghg_consol_schedules`, `ghg_consol_notifications` | Pack runtime configuration (approach, thresholds, enabled engines), preset assignments per group, data collection and reporting schedules, and notification/alert configuration per user role |

**Table Prefix:** `ghg_consol_` (GHG Consolidation)

**Row-Level Security (RLS):**
- All tables have `group_id` column for group-level access control
- Entity-level RLS: users see only entities assigned to their role (entity data owners see their entities only)
- Regional RLS: regional directors see entities in their region
- Group-level RLS: group sustainability officer sees all entities in the group
- Admin role bypasses RLS for cross-group administration
- External assurance users have read-only access to assigned consolidation records

**Indexes:**
- Composite indexes on `(group_id, reporting_period)` for period-based queries
- Composite indexes on `(group_id, entity_id)` for entity-specific queries
- GIN indexes on JSONB columns for flexible metadata (entity attributes, framework output data)
- Partial indexes on `lifecycle_status = 'ACTIVE'` for active-entity-only queries
- B-tree indexes on `entity_id`, `parent_entity_id`, `transfer_id` for foreign key joins
- Unique index on `(owner_entity_id, owned_entity_id, effective_date)` for ownership stake uniqueness
- Full-text search index on `ghg_consol_entities.legal_name` for entity search

---

## 10. Security & Compliance

### 10.1 Authentication & Authorization

- JWT RS256 authentication via SEC-001
- RBAC with 6 roles specific to consolidation:

| Role | Description |
|------|-------------|
| `group_sustainability_officer` | Full access to consolidated inventory, all entities, all reports |
| `regional_director` | Access to entities in assigned region(s), regional sub-consolidation |
| `entity_data_owner` | Access to assigned entity data entry, submission, and entity-level reports |
| `group_controller` | Access to adjustments, eliminations, reconciliation, and consolidated reports |
| `external_assurance` | Read-only access to assigned consolidation audit trail and evidence |
| `admin` | Full system administration including user management and configuration |

**RBAC Permission Matrix:**

| Permission | group_sustainability_officer | regional_director | entity_data_owner | group_controller | external_assurance | admin |
|------------|-----|-----|-----|-----|-----|-----|
| Manage entity registry | Yes | No | No | No | No | Yes |
| Import ownership data | Yes | No | No | Yes | No | Yes |
| Define boundary | Yes | No | No | No | No | Yes |
| Submit entity data | Yes | Yes (region) | Yes (entity) | No | No | Yes |
| View entity data | Yes | Yes (region) | Yes (entity) | Yes | Yes (assigned) | Yes |
| Run consolidation | Yes | No | No | Yes | No | Yes |
| Manage eliminations | Yes | No | No | Yes | No | Yes |
| Manage adjustments | Yes | No | No | Yes | No | Yes |
| Approve adjustments | Yes | No | No | No | No | Yes |
| Generate reports | Yes | Yes (region) | No | Yes | No | Yes |
| View consolidated reports | Yes | Yes (region) | No | Yes | Yes (assigned) | Yes |
| Manage M&A events | Yes | No | No | Yes | No | Yes |
| Sign-off (entity level) | No | No | Yes (entity) | No | No | Yes |
| Sign-off (regional) | No | Yes (region) | No | No | No | Yes |
| Sign-off (group) | Yes | No | No | No | No | Yes |
| View audit trail | Yes | Yes (region) | Yes (entity) | Yes | Yes (assigned) | Yes |
| Manage users | No | No | No | No | No | Yes |
| System configuration | No | No | No | No | No | Yes |

### 10.2 Data Protection

- AES-256-GCM encryption at rest for all entity data, ownership data, financial data, and consolidation results
- TLS 1.3 for data in transit between all components
- SHA-256 provenance hashing on all consolidation outputs (entity emissions, equity calculations, eliminations, adjustments, consolidated totals)
- Full audit trail per SEC-005 (who changed what, when, with provenance chain)
- Ownership and equity data encrypted via Vault (SEC-006) -- commercially sensitive
- Entity financial data (revenue, FTE) encrypted via Vault
- Read-only mode for external assurance users (no data modification, no deletion)
- Data retention: minimum 7 years for consolidation records, 10 years for base year and restatement records
- Immutable audit log: append-only storage, no deletion or modification of historical entries

---

## 11. Performance Requirements

| Metric | Target |
|--------|--------|
| Entity registry operations (CRUD, 1,000 entities) | <2 seconds |
| Hierarchy tree construction (1,000 entities) | <2 seconds |
| Effective equity calculation (500 entities, 5-tier max) | <5 seconds |
| Cross-holding iterative resolution | Convergence < 50 iterations |
| Control assessment batch (500 entities) | <30 seconds |
| Boundary calculation (all 3 approaches, 500 entities) | <30 seconds |
| Approach comparison | <10 seconds |
| Equity share consolidation (500 entities) | <10 seconds |
| Control approach consolidation (500 entities) | <10 seconds |
| Intercompany transfer matching (1,000 transfers) | <30 seconds |
| Elimination calculation (500 matched pairs) | <5 seconds |
| M&A pro-rata calculation (per event) | <1 second |
| Base year restatement (including all historical events) | <10 seconds |
| Consolidation execution (full pipeline, 500 entities) | <5 minutes |
| Report generation (all 10 templates, all frameworks) | <5 minutes |
| Reconciliation check | <30 seconds |
| Audit trail generation | <2 minutes |
| Dashboard refresh | <3 seconds |
| Memory ceiling | 4096 MB |
| Cache hit target | 70% |
| Max entities per group | 5,000 |
| Max groups | 100 |
| Max intercompany transfers | 10,000 per period |
| Max M&A events per period | 50 |
| Max concurrent consolidations | 20 |

---

## 12. Agent Dependencies

### 12.1 MRV Agents (30)

All 30 AGENT-MRV agents are available via `mrv_bridge.py`:
- **MRV-001 to MRV-008**: Scope 1 per entity (stationary, mobile, process, fugitive, refrigerant, land use, waste, agricultural)
- **MRV-009 to MRV-013**: Scope 2 per entity (location-based, market-based, steam, cooling, dual reporting)
- **MRV-014 to MRV-028**: Scope 3 per entity (Categories 1-15)
- **MRV-029**: Category mapper (aligns entity-level categories across entities)
- **MRV-030**: Audit trail and lineage (entity-level provenance feeds into consolidation provenance)

### 12.2 Data Agents (20)

All 20 AGENT-DATA agents via `data_bridge.py`, with primary relevance for:
- **DATA-001 PDF Extractor**: Corporate structure documents, ownership certificates
- **DATA-002 Excel/CSV Normalizer**: Entity emission data submissions, ownership tables
- **DATA-003 ERP/Finance Connector**: Intercompany ledger data, equity data, financial consolidation data
- **DATA-010 Data Quality Profiler**: Per-entity data quality scoring
- **DATA-015 Cross-Source Reconciliation**: Reconciling entity submissions against financial data

### 12.3 Foundation Agents (10)

All 10 AGENT-FOUND agents for orchestration, schema validation, unit normalization (harmonize emission units across entities), assumptions registry (consolidation assumptions), citations (GHG Protocol Ch 3/5, ISO 14064-1), access control, and provenance tracking.

### 12.4 Pack Dependencies

| Dependency | Relationship | Required |
|-----------|-------------|----------|
| PACK-041 | Per-entity Scope 1-2 emissions (primary data source) | Yes |
| PACK-042 | Per-entity Scope 3 starter emissions | No (enhances Scope 3 consolidation) |
| PACK-043 | Per-entity Scope 3 complete emissions | No (enhances Scope 3 consolidation) |
| PACK-044 | Inventory management metadata | No (enhances data quality tracking) |
| PACK-045 | Base year per entity (for M&A restatement) | No (enhances base year handling) |
| PACK-046 | Intensity metrics per entity | No (enhances entity-level intensity) |
| PACK-047 | Benchmark data per entity | No (enhances entity comparison) |
| PACK-048 | Assurance prep for consolidated inventory | No (enhances assurance readiness) |
| PACK-049 | Multi-site data aggregated to entity totals | No (enhances site-to-entity rollup) |

### 12.5 Application Dependencies

- **GL-GHG-APP**: GHG inventory for emission factor sourcing and consolidated reporting
- **GL-CSRD-APP**: ESRS E1 consolidated climate disclosure
- **GL-CDP-APP**: CDP consolidated climate response
- **GL-ISO14064-APP**: ISO 14064-1 consolidated GHG quantification
- **GL-CBAM-APP**: CBAM-relevant entity-level emission data

---

## 13. File Structure

```
packs/ghg-accounting/PACK-050-ghg-consolidation/
  __init__.py
  pack.yaml
  config/
    __init__.py
    pack_config.py
    demo/
      __init__.py
      demo_config.yaml
    presets/
      __init__.py
      corporate_conglomerate.yaml
      financial_holding.yaml
      jv_partnership.yaml
      multinational.yaml
      private_equity.yaml
      real_estate_fund.yaml
      public_company.yaml
      sme_group.yaml
  engines/
    __init__.py
    entity_registry_engine.py
    ownership_structure_engine.py
    boundary_consolidation_engine.py
    equity_share_engine.py
    control_approach_engine.py
    intercompany_elimination_engine.py
    acquisition_divestiture_engine.py
    consolidation_adjustment_engine.py
    group_reporting_engine.py
    consolidation_audit_engine.py
  workflows/
    __init__.py
    entity_mapping_workflow.py
    boundary_selection_workflow.py
    entity_data_collection_workflow.py
    consolidation_execution_workflow.py
    elimination_workflow.py
    mna_adjustment_workflow.py
    group_reporting_workflow.py
    full_consolidation_pipeline_workflow.py
  templates/
    __init__.py
    consolidated_ghg_report.py
    entity_breakdown_report.py
    ownership_structure_report.py
    equity_share_report.py
    elimination_log_report.py
    mna_impact_report.py
    scope_breakdown_report.py
    trend_analysis_report.py
    regulatory_disclosure_report.py
    consolidation_dashboard.py
  integrations/
    __init__.py
    pack_orchestrator.py
    mrv_bridge.py
    data_bridge.py
    pack041_bridge.py
    pack042_043_bridge.py
    pack044_bridge.py
    pack045_bridge.py
    pack048_bridge.py
    pack049_bridge.py
    foundation_bridge.py
    health_check.py
    setup_wizard.py
    alert_bridge.py
  tests/
    __init__.py
    conftest.py
    test_manifest.py
    test_config.py
    test_entity_registry_engine.py
    test_ownership_structure_engine.py
    test_boundary_consolidation_engine.py
    test_equity_share_engine.py
    test_control_approach_engine.py
    test_intercompany_elimination_engine.py
    test_acquisition_divestiture_engine.py
    test_consolidation_adjustment_engine.py
    test_group_reporting_engine.py
    test_consolidation_audit_engine.py
    test_workflows.py
    test_templates.py
    test_integrations.py
    test_presets.py
    test_e2e.py
    test_orchestrator.py
```

---

## 14. Testing Requirements

| Test Type | Coverage Target | Scope |
|-----------|-----------------|-------|
| Unit Tests | >90% line coverage | All 10 engines, all config models, all presets |
| Workflow Tests | >85% | All 8 workflows with synthetic corporate group data |
| Template Tests | 100% | All 10 templates in 3+ formats (MD, HTML, JSON, PDF where applicable) |
| Integration Tests | >80% | All 13 integrations with mock agents and pack bridges |
| E2E Tests | Core happy path | Full pipeline from entity registry through consolidated report |
| Entity Registry Tests | 100% | CRUD, hierarchy, lifecycle, validation, search |
| Ownership Tests | 100% | Multi-tier equity, cross-holdings, convergence, control assessment |
| Boundary Tests | 100% | All three approaches, materiality screening, boundary versioning |
| Equity Share Tests | 100% | Proportional allocation, JV partner attribution, reconciliation |
| Control Tests | 100% | Operational and financial control checklists, special entities |
| Elimination Tests | 100% | Transfer matching, elimination calculation, scope reclassification, reconciliation |
| M&A Tests | 100% | Pro-rata allocation (all event types), base year restatement triggers, organic/structural split |
| Adjustment Tests | 100% | All adjustment categories, approval workflows, magnitude controls |
| Reporting Tests | 100% | All framework outputs, waterfall calculation, intensity metrics |
| Audit Tests | 100% | Reconciliation checks, completeness verification, sign-off tracking, SHA-256 chain |
| Preset Tests | 100% | All 8 corporate structure presets with representative scenarios |
| Manifest Tests | 100% | pack.yaml validation, component counts, version |

**Test Count Target:** 800+ tests (60-80 per engine, 40-50 workflow, 30-40 integration, 20-30 E2E)

**Known-Value Validation Sets:**
- 30 equity chain resolution calculations validated against financial consolidation equity method
- 25 consolidation calculations (all 3 approaches) validated against GHG Protocol worked examples
- 20 intercompany elimination scenarios validated against financial intercompany elimination logic
- 20 M&A pro-rata calculations validated against day-count reference tables
- 15 base year restatement calculations validated against GHG Protocol Chapter 5 worked examples
- 15 cross-holding iterative resolution cases validated against algebraic solutions
- 10 multi-framework output cases validated against published framework templates
- 10 reconciliation test cases (entity sum to consolidated total, variance = 0)
- 5 Scope reclassification scenarios validated against GHG Protocol guidance

---

## 15. Release Plan

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| Phase 1 | PRD Approval | 2026-03-27 |
| Phase 2 | Engine implementation (10 engines) | 2026-03-28 to 2026-03-31 |
| Phase 3 | Workflow implementation (8 workflows) | 2026-03-31 to 2026-04-02 |
| Phase 4 | Template implementation (10 templates) | 2026-04-02 to 2026-04-04 |
| Phase 5 | Integration implementation (13 integrations) | 2026-04-04 to 2026-04-06 |
| Phase 6 | Test suite (800+ tests) | 2026-04-06 to 2026-04-09 |
| Phase 7 | Database migrations (V416-V425) | 2026-04-09 |
| Phase 8 | Documentation & Release | 2026-04-10 |

---

## 16. Appendix: GHG Protocol Chapter 3 Consolidation Decision Tree

### Decision Tree for Organizational Boundary

```
Step 1: Identify all entities in which the company has an equity interest
  |
Step 2: For each entity, determine:
  a. Equity ownership percentage
  b. Whether company has operational control
  c. Whether company has financial control
  |
Step 3: Select consolidation approach:
  Option A: Equity Share -> include proportional emissions for all entities
  Option B: Operational Control -> include 100% of operationally controlled entities
  Option C: Financial Control -> include 100% of financially controlled entities
  |
Step 4: Apply materiality screening:
  - Exclude entities below materiality threshold (if sum < 5% of total)
  - Document exclusions with justification
  |
Step 5: Identify intercompany transfers:
  - Energy transfers between in-scope entities
  - Waste transfers between in-scope entities
  - Product transfers between in-scope entities
  |
Step 6: Apply consolidation:
  - Sum entity emissions (per chosen approach)
  - Eliminate intercompany double-counting
  - Apply manual adjustments
  - Calculate consolidated total
  |
Step 7: Verify and document:
  - Reconcile entity sum to consolidated total
  - Document all boundary decisions
  - Generate audit trail
  - Obtain sign-offs
```

### Entity Type Quick Reference

| Entity Type | Equity Share | Operational Control | Financial Control |
|-------------|-------------|-------------------|-------------------|
| 100% subsidiary | 100% | Usually Yes | Usually Yes |
| 80% subsidiary | 80% | Usually Yes | Usually Yes |
| 60% subsidiary | 60% | Usually Yes | Usually Yes |
| 51% subsidiary | 51% | Assess | Usually Yes |
| 50/50 JV (you operate) | 50% | Yes | Assess |
| 50/50 JV (partner operates) | 50% | No | Assess |
| 40% JV | 40% | Assess | Usually No |
| 30% associate | 30% | No | No |
| 10% investment | 10% | No | No |
| Franchise (you are franchisor) | 0% | Assess (may be Yes) | No |
| Outsourced operation | 0% | Assess | No |

---

*End of PRD-PACK-050*
