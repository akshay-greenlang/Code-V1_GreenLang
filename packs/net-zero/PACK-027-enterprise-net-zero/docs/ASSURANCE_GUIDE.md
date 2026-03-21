# PACK-027 Enterprise Net Zero Pack -- Assurance Guide

**Pack ID:** PACK-027-enterprise-net-zero
**Version:** 1.0.0
**Date:** 2026-03-19
**Author:** GreenLang Platform Engineering

---

## Table of Contents

1. [Introduction](#introduction)
2. [Assurance Standards Overview](#assurance-standards-overview)
3. [ISO 14064-3 Preparation](#iso-14064-3-preparation)
4. [Workpaper Templates](#workpaper-templates)
5. [Evidence Requirements](#evidence-requirements)
6. [Pre-Assurance Control Testing](#pre-assurance-control-testing)
7. [Provider Selection](#provider-selection)
8. [Engagement Management](#engagement-management)
9. [Timeline Planning](#timeline-planning)
10. [Management Assertion Letter](#management-assertion-letter)
11. [Post-Assurance Activities](#post-assurance-activities)
12. [Regulatory Assurance Requirements](#regulatory-assurance-requirements)

---

## Introduction

External assurance of GHG emissions is increasingly mandatory for large enterprises. Under CSRD, enterprises must obtain limited assurance from FY2025, transitioning to reasonable assurance by 2028. The SEC Climate Disclosure Rule requires attestation for large accelerated filers. California SB 253 mandates third-party verification.

PACK-027 is designed to reduce assurance engagement effort from 200-400 hours to under 80 hours by providing:

- **15 pre-formatted audit workpapers** in Big 4 format
- **Automated pre-assurance control testing** (reconciliation, analytical review, sample testing)
- **SHA-256 provenance hashing** on all calculations for integrity verification
- **Complete audit trail** with cryptographic chaining
- **Management assertion letter** template
- **Evidence index** linking every reported figure to source data

---

## Assurance Standards Overview

### Applicable Standards

| Standard | Full Name | Applicability |
|----------|-----------|---------------|
| **ISO 14064-3:2019** | Specification for GHG verification and validation | ISO 14064-1 GHG statements |
| **ISAE 3410** | Assurance Engagements on GHG Statements | GHG statements per GHG Protocol |
| **ISAE 3000 (Revised)** | Assurance on Non-Financial Information | Sustainability disclosures (CSRD, SEC) |
| **AA1000AS v3** | Stakeholder Assurance Standard | Voluntary sustainability reporting |

### Assurance Levels

| Level | Standard | Conclusion Form | Procedures | Confidence Level |
|-------|----------|-----------------|-----------|-----------------|
| **Limited Assurance** | ISAE 3410 / ISAE 3000 | "Nothing has come to our attention..." (negative) | Inquiry, analytical procedures, limited testing | Moderate |
| **Reasonable Assurance** | ISAE 3410 / ISAE 3000 | "In our opinion, the GHG statement is fairly stated..." (positive) | Detailed testing, corroboration, site visits, confirmations | High |
| **ISO 14064-3 Verification** | ISO 14064-3:2019 | Verification statement with level of assurance | Verification plan, evidence gathering, assessment | Per engagement |

### Regulatory Requirements

| Regulation | Assurance Level | Effective | Scope |
|------------|----------------|-----------|-------|
| CSRD (EU) | Limited assurance | FY2025 | Sustainability disclosures incl. GHG |
| CSRD (EU) | Reasonable assurance | FY2028 (expected) | Same scope, higher rigor |
| SEC Climate Rule (US) | Attestation (LAF) | FY2025 | Scope 1+2 |
| California SB 253 | Third-party verification | FY2026 (S1+S2) | Scope 1+2+3 |
| ISSB S2 | Varies by jurisdiction | Varies | Climate disclosures |

---

## ISO 14064-3 Preparation

### Verification Principles

| Principle | Description | PACK-027 Support |
|-----------|-------------|-----------------|
| Independence | Verifier must be independent of the reporting entity | Auditor role with read-only access |
| Ethical conduct | Verifier acts with integrity and objectivity | Segregation of duties enforced |
| Fair presentation | GHG statement represents a true and fair view | DQ scoring, reconciliation, analytical review |
| Due professional care | Appropriate diligence in verification procedures | Structured workpapers, evidence index |
| Competence | Verifier has appropriate GHG knowledge | Workpapers designed for GHG-specialist and generalist auditors |

### Materiality Threshold

PACK-027 calculates materiality automatically:

```python
from workflows.external_assurance_workflow import ExternalAssuranceWorkflow

workflow = ExternalAssuranceWorkflow(config=config)

materiality = workflow.calculate_materiality(
    total_emissions_tco2e=baseline.grand_total_tco2e,
    materiality_pct=5.0,  # Standard: 5% of total emissions
)

print(f"Total emissions: {materiality.total_tco2e:,.0f} tCO2e")
print(f"Materiality threshold: {materiality.threshold_tco2e:,.0f} tCO2e")
print(f"Performance materiality (75%): {materiality.performance_tco2e:,.0f} tCO2e")
print(f"Trivial threshold (5% of materiality): {materiality.trivial_tco2e:,.0f} tCO2e")
```

---

## Workpaper Templates

### 15 Pre-Formatted Workpapers

PACK-027 generates 15 audit workpapers in Big 4 format:

| # | Workpaper | Content | Auditor Use |
|---|-----------|---------|-------------|
| WP-100 | Engagement Overview | Scope, boundary, criteria, materiality threshold, team | Planning and scoping |
| WP-200 | Organizational Boundary | Entity hierarchy, consolidation approach, ownership %, control assessments | Boundary verification |
| WP-300 | Scope 1 Detail | Per-source emissions with calculation methodology, emission factors, activity data | Substantive testing |
| WP-400 | Scope 2 Detail | Location-based and market-based with grid factors, contractual instruments | Substantive testing |
| WP-500 | Scope 3 Detail | Per-category with methodology, data quality scores, materiality justification | Substantive testing |
| WP-600 | Emission Factors | Full register of all emission factors used with source, version, vintage | Factor verification |
| WP-700 | Data Quality Assessment | Per-category DQ scoring against GHG Protocol hierarchy | Data quality evaluation |
| WP-800 | Base Year Recalculation | Trigger assessment, significance calculation, recalculated values | Consistency testing |
| WP-900 | Consolidation Reconciliation | Entity-to-group reconciliation, intercompany eliminations | Consolidation verification |
| WP-1000 | Calculation Trace | Step-by-step calculation for 60 sampled items | Recalculation testing |
| WP-1100 | Provenance Hashes | SHA-256 hashes for all inputs, outputs, and intermediate calculations | Integrity verification |
| WP-1200 | Control Documentation | Data collection controls, approval workflows, change management | Control testing |
| WP-1300 | Management Assertion | Management representation letter template | Management responsibilities |
| WP-1400 | Prior Year Comparison | Year-over-year analysis with variance explanations for >5% changes | Analytical review |
| WP-1500 | Findings Register | Open issues from pre-assurance testing with severity and remediation | Issue tracking |

### Generating Workpapers

```python
from workflows.external_assurance_workflow import ExternalAssuranceWorkflow

workflow = ExternalAssuranceWorkflow(config=config)

# Generate all 15 workpapers
workpapers = workflow.generate_workpapers(
    baseline=baseline,
    current_year=annual_result,
    assurance_level="limited",  # or "reasonable"
    provider_format="big4_standard",  # or specific: "deloitte", "ey", "kpmg", "pwc"
    sample_size=60,
)

# Export workpapers
for wp in workpapers.papers:
    print(f"  {wp.reference}: {wp.title}")
    print(f"    Pages: {wp.page_count}")
    print(f"    Status: {wp.status}")

# Export to PDF/XLSX
workpapers.export(format="pdf", output_dir="/workpapers/2025/")
workpapers.export(format="xlsx", output_dir="/workpapers/2025/")
```

### Workpaper Detail: WP-1000 Calculation Trace

The calculation trace workpaper provides step-by-step recalculation for sampled items:

```python
# Sample selection for calculation trace
sample = workflow.select_sample(
    population=baseline.all_line_items,
    sample_size=60,
    method="stratified_random",
    strata=[
        {"scope": "scope1", "count": 15},
        {"scope": "scope2", "count": 10},
        {"scope": "scope3", "count": 35},
    ],
)

# Each sample item includes:
for item in sample.items:
    print(f"\nSample {item.number}: {item.description}")
    print(f"  Entity: {item.entity_name}")
    print(f"  Scope: {item.scope}")
    print(f"  Category: {item.category}")
    print(f"  Activity data: {item.activity_value} {item.activity_unit}")
    print(f"  Emission factor: {item.ef_value} {item.ef_unit}")
    print(f"  Factor source: {item.ef_source} ({item.ef_vintage})")
    print(f"  Calculation: {item.activity_value} x {item.ef_value} = {item.result_tco2e}")
    print(f"  Input hash: {item.input_hash}")
    print(f"  Output hash: {item.output_hash}")
    print(f"  Recalculation match: {item.recalculation_match}")
```

---

## Evidence Requirements

### Evidence Hierarchy

| Evidence Type | Reliability | Example | Auditor Weight |
|--------------|-------------|---------|---------------|
| External confirmation | Highest | Utility bill from supplier, CDP supplier response | Primary evidence |
| Third-party calculation | High | Grid emission factor from IEA, DEFRA | Supporting evidence |
| Internal system data | Medium | ERP extraction, meter reading | Subject to testing |
| Internal estimate | Lower | Spend-based proxy, industry average | Subject to scrutiny |
| Management assertion | Lowest | Verbal explanation without supporting data | Requires corroboration |

### Evidence Index

PACK-027 generates a complete evidence index linking every reported figure to its source:

```python
evidence_index = workflow.generate_evidence_index(
    baseline=baseline,
)

# For each reported figure:
for entry in evidence_index.entries:
    print(f"\n{entry.reported_figure}: {entry.value:,.0f} tCO2e")
    print(f"  Source data: {entry.source_type} ({entry.source_reference})")
    print(f"  Emission factor: {entry.ef_source} v{entry.ef_version}")
    print(f"  Calculation: {entry.calculation_description}")
    print(f"  Provenance hash: {entry.provenance_hash}")
    print(f"  Evidence file: {entry.evidence_file_path}")
```

### Document Retention

| Document Type | Retention Period | Storage Location |
|--------------|-----------------|-----------------|
| Source activity data | 10 years | PostgreSQL + S3 backup |
| Emission factors used | 10 years | Versioned reference database |
| Calculation workpapers | 10 years | S3 with versioning |
| Audit trail | 10 years | Append-only database |
| Assurance reports | 10 years | S3 with versioning |
| Provenance hashes | 10 years | Append-only database |

---

## Pre-Assurance Control Testing

### Automated Control Tests

PACK-027 runs automated pre-assurance control tests before generating the workpaper package:

```python
control_tests = workflow.run_pre_assurance_tests(
    baseline=baseline,
    sample_size=60,
)

print(f"Tests run: {control_tests.total_tests}")
print(f"Passed: {control_tests.passed}")
print(f"Failed: {control_tests.failed}")
print(f"Warnings: {control_tests.warnings}")

for test in control_tests.results:
    status = "PASS" if test.passed else "FAIL"
    print(f"  [{status}] {test.category}: {test.description}")
    if not test.passed:
        print(f"         Finding: {test.finding}")
        print(f"         Remediation: {test.remediation}")
```

### Control Test Categories

| Category | Test | Pass Criteria | Remediation |
|----------|------|--------------|-------------|
| **Completeness** | All entities submitted data | 100% entity coverage | Escalation to entity data owners |
| **Completeness** | All material Scope 3 categories calculated | 100% material categories | Run calculations for missing categories |
| **Accuracy** | Recalculation of 60 sampled line items | Within +/-1% of original | Investigate and correct errors |
| **Accuracy** | Emission factor version check | All factors from current year | Update stale factors |
| **Consistency** | Year-over-year variance >10% per entity | All variances explained | Document variance drivers |
| **Consistency** | Scope 2 dual reporting reconciliation | Location >= market (or explained) | Verify contractual instruments |
| **Cut-off** | All data within reporting period | No out-of-period data | Correct period allocation |
| **Classification** | Scope 1/2/3 classification | 100% correctly classified | Reclassify misclassified items |
| **Existence** | Emission factor source verification | All factors traceable | Replace unsourced factors |
| **Valuation** | GWP values match IPCC AR6 | 100% match | Update GWP values |

---

## Provider Selection

### Selecting an Assurance Provider

| Provider Type | Expertise | Typical Fee Range | Best For |
|--------------|-----------|-------------------|----------|
| **Big 4** (Deloitte, EY, KPMG, PwC) | Broad sustainability + financial audit | $100K-$500K | SEC filers, CSRD reporters, integrated assurance |
| **Specialist** (SGS, Bureau Veritas, DNV) | Deep GHG/ISO expertise | $30K-$150K | ISO 14064-3, standalone GHG verification |
| **Boutique** (Carbon Trust, ERM CVS) | Climate-specific expertise | $25K-$100K | CDP assurance, voluntary verification |

### Selection Criteria

| Criterion | Weight | Assessment Method |
|-----------|--------|------------------|
| GHG verification experience | 25% | References, track record |
| Sector expertise | 20% | Prior engagements in same sector |
| Geographic coverage | 15% | Ability to cover all entity countries |
| Standard accreditation | 15% | ISAE 3410, ISO 14064-3, ISAE 3000 |
| Team composition | 10% | Named partner, GHG specialists |
| Fee and timeline | 10% | Competitive pricing, realistic timeline |
| Technology capability | 5% | Ability to work with PACK-027 outputs |

### PACK-027 Provider Integration

```python
from integrations.assurance_provider_bridge import AssuranceProviderBridge

bridge = AssuranceProviderBridge(config=config)

# Configure provider access
bridge.setup_provider_access(
    provider_name="Deloitte",
    contact_email="ghg-assurance@deloitte.com",
    access_level="auditor",  # Read-only role
    start_date="2026-03-01",
    end_date="2026-06-30",
)

# Share workpapers with provider
bridge.share_workpapers(
    workpapers=workpapers,
    provider_name="Deloitte",
    sharing_method="secure_portal",  # or "encrypted_email"
)
```

---

## Engagement Management

### Typical Engagement Flow

```
Phase 1: Planning (Week 1-2)
    |
    +-- Agree scope (limited vs. reasonable)
    +-- Define boundary (which entities, which scopes)
    +-- Set materiality threshold (typically 5%)
    +-- PACK-027: Generate WP-100, WP-200
    |
    v
Phase 2: Risk Assessment (Week 3-4)
    |
    +-- Assess areas of higher risk
    +-- Plan testing approach
    +-- PACK-027: Provide DQ matrix (WP-700), variance analysis (WP-1400)
    |
    v
Phase 3: Fieldwork (Week 5-8)
    |
    +-- Test calculations (recalculate samples)
    +-- Verify emission factors
    +-- Test data collection controls
    +-- Review intercompany eliminations
    +-- PACK-027: Provide WP-300 to WP-1100
    |
    v
Phase 4: Findings and Remediation (Week 9-10)
    |
    +-- Discuss findings with management
    +-- Remediate errors (if any)
    +-- Re-run calculations if needed
    +-- PACK-027: Update WP-1500, re-generate affected workpapers
    |
    v
Phase 5: Reporting (Week 11-12)
    |
    +-- Draft assurance report
    +-- Management representation letter
    +-- Finalize and issue assurance statement
    +-- PACK-027: Provide WP-1300 template, generate assurance_statement template
```

### Managing Auditor Queries

```python
# Track auditor queries
query = bridge.log_query(
    query_id="Q-001",
    from_provider="Deloitte",
    category="data_verification",
    question="Please provide supporting evidence for entity Sub-Beta Scope 1 stationary combustion of 8,200 tCO2e",
    assigned_to="analyst@globalmfg.com",
    due_date="2026-04-15",
)

# Respond with evidence
bridge.respond_to_query(
    query_id="Q-001",
    response="See WP-300, rows 15-22. Supporting evidence: SAP fuel invoices extracted via DATA-003, DEFRA 2024 emission factors (WP-600 row 8). Provenance hash: abc123...",
    attachments=["wp-300-beta-scope1.pdf", "sap-fuel-invoices-beta-2025.xlsx"],
)
```

---

## Timeline Planning

### Assurance Calendar

| Activity | Timing (relative to reporting year end) | Duration | PACK-027 Support |
|----------|----------------------------------------|----------|-----------------|
| Data collection finalized | +30 to +60 days | - | Entity data collection schedule |
| Annual inventory completed | +60 to +90 days | 2-4 weeks | Annual inventory workflow |
| Workpapers generated | +90 to +100 days | 1 week | External assurance workflow |
| Pre-assurance testing | +100 to +110 days | 1 week | Automated control tests |
| Provider engagement starts | +90 to +100 days | - | Provider bridge setup |
| Planning phase | +100 to +115 days | 2 weeks | WP-100, WP-200 |
| Fieldwork | +115 to +145 days | 4 weeks | WP-300 to WP-1100 |
| Findings remediation | +145 to +160 days | 2 weeks | WP-1500 updates |
| Assurance report issued | +160 to +175 days | 2 weeks | Assurance statement template |
| Filing deadline (CSRD) | +180 days (typical) | - | Regulatory filings template |
| Filing deadline (CDP) | July (typical) | - | CDP response template |

### Total Timeline: ~25 weeks from reporting year end to assurance report

---

## Management Assertion Letter

### Template Content

PACK-027 generates a management assertion letter template:

```python
assertion = workflow.generate_management_assertion(
    baseline=baseline,
    assurance_level="limited",
    reporting_year=2025,
)

# Template sections:
# 1. Management's responsibility for the GHG statement
# 2. Assertion of completeness (all material sources included)
# 3. Assertion of accuracy (methodology correctly applied)
# 4. Assertion of cut-off (data within reporting period)
# 5. Assertion of classification (correct scope classification)
# 6. Assertion of consistency (consistent with prior year)
# 7. Disclosure of known errors (if any)
# 8. Disclosure of changes (methodology, boundary, base year)
# 9. Confirmation of access to records
# 10. Signature block (CSO + CFO)
```

---

## Post-Assurance Activities

### After Receiving the Assurance Report

```python
# Record assurance outcome
outcome = bridge.record_assurance_outcome(
    assurance_level="limited",
    provider="Deloitte",
    conclusion="unqualified",  # or "qualified", "adverse", "disclaimer"
    findings_count=2,
    material_misstatements=0,
    report_date="2026-06-15",
    report_reference="DEL-2026-GHG-001",
)

# Update disclosure with assurance reference
from templates.regulatory_filings import RegulatoryFilings
filings = RegulatoryFilings()
filings.add_assurance_reference(
    framework="CSRD",
    assurance_level="limited",
    provider="Deloitte",
    report_date="2026-06-15",
)
```

### Continuous Improvement

| Year | Assurance Goal | Action Items |
|------|---------------|-------------|
| Year 1 | Limited assurance on Scope 1+2 | Establish controls, generate initial workpapers |
| Year 2 | Limited assurance on Scope 1+2+3 | Extend to Scope 3, improve DQ to Level 2-3 |
| Year 3 | Reasonable assurance on Scope 1+2, limited on Scope 3 | Strengthen controls, automate data collection |
| Year 4-5 | Reasonable assurance on Scope 1+2+3 | Best-in-class data quality, comprehensive evidence |

---

## Regulatory Assurance Requirements

### Requirements by Regulation

| Regulation | Scope | Level | When | Provider Requirements |
|------------|-------|-------|------|----------------------|
| CSRD | All ESRS disclosures | Limited (2025), Reasonable (2028) | Annual | Statutory auditor or accredited provider |
| SEC Climate Rule | Scope 1+2 (LAF only) | Attestation | FY2025 (LAF) | PCAOB-registered or equivalent |
| CA SB 253 | Scope 1+2+3 | Third-party verification | FY2026 (S1+S2), FY2027 (S3) | CARB-approved |
| ISO 14064 | GHG statement | Verification | Voluntary | ISO 14065 accredited |
| CDP | Climate questionnaire | Third-party verification | Voluntary | Any recognized provider |

### PACK-027 Regulatory Alignment

PACK-027 generates workpapers that satisfy all regulatory requirements simultaneously. A single assurance engagement can cover multiple regulations:

```python
# Multi-regulatory assurance package
package = workflow.generate_multi_regulatory_package(
    regulations=["CSRD", "SEC", "CA_SB253", "ISO_14064", "CDP"],
    baseline=baseline,
    assurance_level="limited",
)

print(f"Total workpapers: {package.total_workpapers}")
print(f"Regulatory requirements covered: {package.regulations_covered}")
print(f"Estimated auditor hours: {package.estimated_hours}")
```
