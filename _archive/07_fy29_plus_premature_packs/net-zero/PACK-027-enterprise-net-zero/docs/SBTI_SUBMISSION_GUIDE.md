# PACK-027 Enterprise Net Zero Pack -- SBTi Submission Guide

**Pack ID:** PACK-027-enterprise-net-zero
**Version:** 1.0.0
**Date:** 2026-03-19
**Author:** GreenLang Platform Engineering

---

## Table of Contents

1. [Introduction](#introduction)
2. [SBTi Corporate Standard Overview](#sbti-corporate-standard-overview)
3. [Enterprise vs. SME Pathway](#enterprise-vs-sme-pathway)
4. [Pre-Submission Requirements](#pre-submission-requirements)
5. [Step 1: Commit to SBTi](#step-1-commit-to-sbti)
6. [Step 2: Complete GHG Inventory](#step-2-complete-ghg-inventory)
7. [Step 3: Scope 3 Materiality Screening](#step-3-scope-3-materiality-screening)
8. [Step 4: Select Target Pathway](#step-4-select-target-pathway)
9. [Step 5: Define Targets](#step-5-define-targets)
10. [Step 6: Validate 42 Criteria](#step-6-validate-42-criteria)
11. [Step 7: Generate Submission Package](#step-7-generate-submission-package)
12. [Step 8: Submit and Track](#step-8-submit-and-track)
13. [Step 9: Annual Progress Reporting](#step-9-annual-progress-reporting)
14. [Step 10: Five-Year Review](#step-10-five-year-review)
15. [42 Criteria Reference](#42-criteria-reference)
16. [SDA Sector Pathways](#sda-sector-pathways)
17. [FLAG Targets](#flag-targets)
18. [Common Rejection Reasons](#common-rejection-reasons)
19. [Troubleshooting](#troubleshooting)

---

## Introduction

This guide walks enterprise users through the complete SBTi (Science Based Targets initiative) target submission process using PACK-027. Unlike the simplified SME pathway available in PACK-026, enterprise organizations must follow the full SBTi Corporate Manual V5.3 and Net-Zero Standard V1.3, which require:

- **28 near-term criteria** (C1-C28)
- **14 net-zero criteria** (NZ-C1 to NZ-C14)
- **42 total criteria** validated before submission
- **Coverage requirements**: 95% of Scope 1+2, 67%+ of Scope 3 for near-term, 90%+ for long-term
- **Pathway selection**: ACA (Absolute Contraction Approach), SDA (Sectoral Decarbonization Approach), or FLAG (Forest, Land and Agriculture)

PACK-027 automates the entire process from GHG inventory through submission package generation.

---

## SBTi Corporate Standard Overview

### Key Standards

| Standard | Version | Scope | Key Requirements |
|----------|---------|-------|-----------------|
| SBTi Corporate Manual | V5.3 (2024) | Near-term targets | 28 criteria (C1-C28); ACA/SDA pathways; 5-10 year timeframe |
| SBTi Net-Zero Standard | V1.3 (2024) | Long-term + net-zero | 14 criteria (NZ-C1 to NZ-C14); 90%+ reduction by 2050; permanent CDR |
| SBTi FLAG Guidance | V1.1 (2022) | Land use emissions | Required if >20% FLAG; 3.03%/yr; no-deforestation by 2025 |
| SBTi SDA Tool | V3.0 (2024) | Sector pathways | 12 sectors; intensity convergence; IEA NZE alignment |
| IPCC AR6 | 2021/2022 | Scientific basis | GWP-100 values; carbon budgets; 1.5C pathway constraints |

### Target Types

| Target Type | Ambition | Annual Reduction Rate | Timeline | Coverage |
|-------------|----------|----------------------|----------|----------|
| Near-term 1.5C (ACA) | 1.5C-aligned | >= 4.2%/yr absolute | 5-10 years | Scope 1+2: 95%, Scope 3: 67% |
| Near-term WB2C (ACA) | Well-below 2C | >= 2.5%/yr absolute | 5-10 years | Scope 1+2: 95%, Scope 3: 67% |
| Near-term SDA | Sector-specific | Intensity convergence | 5-10 years | Scope 1+2: 95% |
| Long-term (net-zero) | Net-zero by 2050 | 90%+ reduction | By 2050 | Scope 1+2: 95%, Scope 3: 90% |
| FLAG | 1.5C no-deforestation | 3.03%/yr FLAG | By 2030/2050 | FLAG emissions if >20% |

### SBTi Process Timeline

```
Month 0: Commit to SBTi
    |
    v
Month 1-3: Build GHG inventory (PACK-027 comprehensive_baseline_workflow)
    |
    v
Month 4-5: Screen Scope 3, select pathway, define targets (PACK-027 sbti_target_engine)
    |
    v
Month 6: Validate 42 criteria, generate submission package (PACK-027 sbti_submission_workflow)
    |
    v
Month 7: Submit to SBTi
    |
    v
Month 7-10: SBTi validation queue (4-12 weeks typical)
    |
    v
Month 10-11: Address SBTi queries (if any)
    |
    v
Month 11-12: Targets validated and published
    |
    v
Annually: Progress reporting (PACK-027 annual_inventory_workflow)
    |
    v
Every 5 years: Revalidation (PACK-027 sbti_submission_workflow)
```

---

## Enterprise vs. SME Pathway

| Feature | Corporate Standard (PACK-027) | SME Pathway (PACK-026) |
|---------|-------------------------------|------------------------|
| Eligibility | Any company (no size limit) | <500 employees |
| Scope 1+2 target | 4.2%/yr ACA (1.5C) or SDA | 50% by 2030 |
| Scope 3 target | 67%+ coverage, quantified | Measure and reduce (no formal target) |
| Long-term target | 90%+ by 2050, mandatory | Not required |
| Net-zero commitment | Residual neutralization via permanent CDR | Not applicable |
| Pathway options | ACA, SDA (12 sectors), FLAG | Simplified absolute contraction |
| Validation process | Queued review (4-12 weeks) | Immediate auto-validation |
| Cost | $2,500-$15,000+ (by revenue) | Free |
| Revalidation | Every 5 years | Annual self-assessment |
| Criteria count | 42 (C1-C28 + NZ-C1 to NZ-C14) | Simplified checklist |

**When to use the Corporate Standard:**
- Organization has >250 employees or >$50M revenue
- Operating in multiple countries
- Complex corporate structure (subsidiaries, JVs)
- External reporting obligations (SEC, CSRD, CDP)
- Pursuing investor-grade climate targets

---

## Pre-Submission Requirements

Before starting the SBTi submission process, ensure:

### Data Requirements

| Requirement | Description | PACK-027 Source |
|-------------|-------------|----------------|
| Complete Scope 1 inventory | All sources across all entities | MRV-001 to MRV-008 |
| Complete Scope 2 inventory | Dual reporting (location + market) | MRV-009 to MRV-013 |
| Complete Scope 3 screening | All 15 categories assessed | MRV-014 to MRV-028 |
| Base year data | Recent (within 2 years of submission) | Enterprise baseline engine |
| Entity hierarchy | Complete with ownership and control | Multi-entity orchestrator |
| Sector classification | GICS or NACE code for SDA | Config preset |
| Financial data | Revenue for intensity metrics | Financial integration engine |

### System Requirements

```python
# Verify readiness
from integrations.health_check import HealthCheck

hc = HealthCheck()
sbti_readiness = hc.check_sbti_readiness()

print(f"GHG Inventory Complete: {sbti_readiness.inventory_complete}")
print(f"Base Year Valid: {sbti_readiness.base_year_valid}")
print(f"Entity Coverage: {sbti_readiness.entity_coverage_pct}%")
print(f"Scope 3 Categories Assessed: {sbti_readiness.scope3_categories_assessed}/15")
print(f"Data Quality Score: {sbti_readiness.weighted_dq_score}")
print(f"Ready for Submission: {sbti_readiness.is_ready}")
```

---

## Step 1: Commit to SBTi

Register your commitment on the SBTi platform. This starts a 24-month window to submit targets.

```python
from integrations.sbti_bridge import SBTiBridge

bridge = SBTiBridge(config=config)

commitment = bridge.register_commitment(
    organization_name="GlobalMfg Corp",
    commitment_date="2026-01-15",
    contact_email="sustainability@globalmfg.com",
    sector="Manufacturing",
    country="US",
)

print(f"Commitment ID: {commitment.id}")
print(f"Submission deadline: {commitment.deadline}")  # 24 months from commitment
```

---

## Step 2: Complete GHG Inventory

Run the comprehensive baseline workflow to build your enterprise GHG inventory.

```python
from workflows.comprehensive_baseline_workflow import ComprehensiveBaselineWorkflow

workflow = ComprehensiveBaselineWorkflow(config=config)
baseline = workflow.execute()

# Phase 1: Entity Mapping
# Phase 2: Data Collection
# Phase 3: Quality Assurance
# Phase 4: Calculation (all 30 MRV agents)
# Phase 5: Consolidation
# Phase 6: Reporting

print(f"Scope 1 Total: {baseline.scope1_total_tco2e:,.0f} tCO2e")
print(f"Scope 2 Location: {baseline.scope2_location_tco2e:,.0f} tCO2e")
print(f"Scope 2 Market: {baseline.scope2_market_tco2e:,.0f} tCO2e")
print(f"Scope 3 Total: {baseline.scope3_total_tco2e:,.0f} tCO2e")
print(f"Grand Total: {baseline.grand_total_tco2e:,.0f} tCO2e")
```

---

## Step 3: Scope 3 Materiality Screening

Assess all 15 Scope 3 categories for materiality:

```python
from engines.enterprise_baseline_engine import EnterpriseBaselineEngine

engine = EnterpriseBaselineEngine(config=config)
materiality = engine.assess_scope3_materiality(baseline)

for cat in materiality.categories:
    status = "MATERIAL" if cat.pct_of_total > 1.0 else "IMMATERIAL"
    print(f"Cat {cat.number:2d} | {cat.name:40s} | {cat.tco2e:>12,.0f} tCO2e | "
          f"{cat.pct_of_total:5.1f}% | {status}")

print(f"\nTotal Scope 3: {materiality.scope3_total:,.0f} tCO2e")
print(f"Material categories (>1%): {materiality.material_count}/15")
print(f"Coverage of material categories: {materiality.material_coverage_pct:.1f}%")
print(f"Excluded categories total: {materiality.excluded_pct:.1f}% (must be <5%)")
```

**Materiality thresholds per PACK-027:**
- Categories > 1% of total emissions: full activity-based calculation required
- Categories 0.1-1%: average-data method acceptable
- Categories < 0.1%: may be excluded with documented justification
- Total exclusions must not exceed 5% of anticipated total Scope 3

---

## Step 4: Select Target Pathway

PACK-027 supports three target pathways:

### Absolute Contraction Approach (ACA)

Best for most enterprises. Requires absolute reduction of emissions over time.

```python
from engines.sbti_target_engine import SBTiTargetEngine

engine = SBTiTargetEngine(config=config)

# Evaluate ACA suitability
aca_assessment = engine.evaluate_pathway(
    baseline=baseline,
    pathway="aca_15c",
)

print(f"ACA 1.5C Suitability: {aca_assessment.suitability}")
print(f"Required annual reduction: {aca_assessment.annual_reduction_pct:.1f}%/yr")
print(f"Scope 1+2 target (2030): {aca_assessment.scope12_target_2030:,.0f} tCO2e")
print(f"Scope 3 target (2030): {aca_assessment.scope3_target_2030:,.0f} tCO2e")
```

### Sectoral Decarbonization Approach (SDA)

For specific sectors where intensity pathways are more appropriate:

| Sector | Intensity Metric | 2030 Benchmark | 2050 Benchmark |
|--------|-----------------|----------------|----------------|
| Power generation | tCO2/MWh | 0.14 | 0.00 |
| Cement | tCO2/t cement | 0.42 | 0.07 |
| Iron & steel | tCO2/t crude steel | 1.06 | 0.05 |
| Aluminium | tCO2/t aluminium | 3.10 | 0.20 |
| Pulp & paper | tCO2/t product | 0.22 | 0.04 |
| Chemicals | tCO2/t product | Varies | Varies |
| Aviation | gCO2/pkm | 62.0 | 8.0 |
| Maritime shipping | gCO2/tkm | 5.8 | 0.8 |
| Road transport | gCO2/vkm | 85.0 | 0.0 |
| Commercial buildings | kgCO2/sqm | 25.0 | 2.0 |
| Residential buildings | kgCO2/sqm | 12.0 | 1.0 |
| Food & beverage | tCO2/t product | Varies | Varies |

```python
sda_assessment = engine.evaluate_pathway(
    baseline=baseline,
    pathway="sda",
    sector="cement",
    current_intensity=0.65,  # tCO2/t cement
)

print(f"SDA Suitability: {sda_assessment.suitability}")
print(f"Current intensity: {sda_assessment.current_intensity}")
print(f"2030 target intensity: {sda_assessment.target_2030_intensity}")
print(f"2050 target intensity: {sda_assessment.target_2050_intensity}")
```

### FLAG Pathway

Required if FLAG (Forest, Land, and Agriculture) emissions exceed 20% of total.

```python
flag_assessment = engine.evaluate_flag(baseline=baseline)

print(f"FLAG emissions: {flag_assessment.flag_tco2e:,.0f} tCO2e")
print(f"FLAG as % of total: {flag_assessment.flag_pct:.1f}%")
print(f"FLAG target required: {flag_assessment.flag_required}")
if flag_assessment.flag_required:
    print(f"FLAG reduction rate: 3.03%/yr")
    print(f"No-deforestation deadline: 2025")
```

### Mixed Pathway (common for diversified enterprises)

```python
# For diversified companies: ACA for general + SDA for sector-specific divisions
mixed = engine.evaluate_pathway(
    baseline=baseline,
    pathway="mixed",
    divisions={
        "general_operations": {"pathway": "aca_15c"},
        "cement_division": {"pathway": "sda", "sector": "cement"},
        "power_division": {"pathway": "sda", "sector": "power"},
    }
)
```

---

## Step 5: Define Targets

### Near-Term Targets (5-10 years)

```python
near_term = engine.set_near_term_target(
    baseline=baseline,
    pathway="aca_15c",
    base_year=2023,
    target_year=2030,
    scope12_coverage=0.95,  # 95% of Scope 1+2
    scope3_coverage=0.67,   # 67% of Scope 3
)

print("Near-Term Target:")
print(f"  Base year: {near_term.base_year}")
print(f"  Target year: {near_term.target_year}")
print(f"  Scope 1+2 base: {near_term.scope12_base:,.0f} tCO2e")
print(f"  Scope 1+2 target: {near_term.scope12_target:,.0f} tCO2e")
print(f"  Scope 1+2 reduction: {near_term.scope12_reduction_pct:.1f}%")
print(f"  Scope 3 base: {near_term.scope3_base:,.0f} tCO2e")
print(f"  Scope 3 target: {near_term.scope3_target:,.0f} tCO2e")

# Annual milestones
for milestone in near_term.annual_milestones:
    print(f"  {milestone.year}: {milestone.scope12_target:,.0f} / "
          f"{milestone.scope3_target:,.0f} tCO2e")
```

### Long-Term Targets (by 2050)

```python
long_term = engine.set_long_term_target(
    baseline=baseline,
    target_year=2050,
    reduction_pct=0.90,  # 90%+ from base year
    scope12_coverage=0.95,
    scope3_coverage=0.90,  # 90% for long-term
)

print("Long-Term Target:")
print(f"  Scope 1+2 target 2050: {long_term.scope12_target_2050:,.0f} tCO2e")
print(f"  Scope 3 target 2050: {long_term.scope3_target_2050:,.0f} tCO2e")
print(f"  Residual emissions: {long_term.residual_tco2e:,.0f} tCO2e")
```

### Net-Zero Neutralization

```python
net_zero = engine.set_net_zero_target(
    long_term_target=long_term,
    neutralization_method="permanent_cdr",  # Required by SBTi
)

print("Net-Zero Target:")
print(f"  Residual to neutralize: {net_zero.residual_tco2e:,.0f} tCO2e")
print(f"  Neutralization method: {net_zero.method}")
print(f"  CDR volume required: {net_zero.cdr_volume_tco2e:,.0f} tCO2e/yr")
```

---

## Step 6: Validate 42 Criteria

PACK-027 automatically validates all 42 criteria:

```python
validation = engine.validate_all_criteria(
    baseline=baseline,
    near_term=near_term,
    long_term=long_term,
    net_zero=net_zero,
)

print(f"Total Criteria: {validation.total_criteria}")
print(f"Passed: {validation.passed_count}")
print(f"Failed: {validation.failed_count}")
print(f"Warning: {validation.warning_count}")
print(f"Submission Ready: {validation.is_submission_ready}")

# Review each criterion
for c in validation.criteria:
    icon = "PASS" if c.passed else "FAIL"
    print(f"  [{icon}] {c.id}: {c.description}")
    if not c.passed:
        print(f"         Current: {c.current_value}")
        print(f"         Required: {c.required_value}")
        print(f"         Fix: {c.remediation}")
```

---

## Step 7: Generate Submission Package

```python
from workflows.sbti_submission_workflow import SBTiSubmissionWorkflow

workflow = SBTiSubmissionWorkflow(config=config)
package = workflow.execute(
    baseline=baseline,
    near_term=near_term,
    long_term=long_term,
    net_zero=net_zero,
)

# Generate submission documents
from templates.sbti_target_submission import SBTiTargetSubmission

template = SBTiTargetSubmission()
submission = template.render(package, format="pdf")

# Package contents:
# 1. Target submission form (SBTi template format)
# 2. Criteria validation matrix (42 criteria with evidence)
# 3. GHG inventory summary
# 4. Scope 3 materiality screening
# 5. Pathway justification
# 6. Annual milestone pathway
# 7. Base year and recalculation policy
# 8. Coverage analysis
# 9. Supporting methodology documentation
```

---

## Step 8: Submit and Track

```python
from integrations.sbti_bridge import SBTiBridge

bridge = SBTiBridge(config=config)

submission_result = bridge.submit_targets(
    package=package,
    contact_email="sustainability@globalmfg.com",
)

print(f"Submission ID: {submission_result.submission_id}")
print(f"Submitted: {submission_result.submission_date}")
print(f"Expected validation: {submission_result.expected_validation_date}")

# Track validation status
status = bridge.check_status(submission_result.submission_id)
print(f"Current status: {status.status}")
# Possible: SUBMITTED, IN_REVIEW, QUERIES_RAISED, VALIDATED, REJECTED
```

---

## Step 9: Annual Progress Reporting

```python
from workflows.annual_inventory_workflow import AnnualInventoryWorkflow

workflow = AnnualInventoryWorkflow(config=config)
annual_result = workflow.execute(reporting_year=2026)

# Compare against target pathway
progress = engine.assess_progress(
    baseline=baseline,
    current_year=annual_result,
    target=near_term,
)

print(f"Year: {progress.year}")
print(f"Target pathway: {progress.target_pathway_tco2e:,.0f} tCO2e")
print(f"Actual emissions: {progress.actual_tco2e:,.0f} tCO2e")
print(f"On track: {progress.on_track}")
print(f"Gap to target: {progress.gap_tco2e:,.0f} tCO2e ({progress.gap_pct:+.1f}%)")
```

---

## Step 10: Five-Year Review

SBTi requires revalidation every 5 years:

```python
# Run revalidation workflow
revalidation = workflow.execute(
    mode="revalidation",
    original_submission=submission_result,
    current_baseline=annual_result,
)

print(f"Revalidation required by: {revalidation.deadline}")
print(f"Current targets still valid: {revalidation.targets_valid}")
print(f"Methodology changes since submission: {revalidation.methodology_changes}")
print(f"Recommendation: {revalidation.recommendation}")
```

---

## 42 Criteria Reference

### Near-Term Criteria (C1-C28)

| # | Criterion | Requirement | PACK-027 Check |
|---|-----------|-------------|---------------|
| C1 | Organizational boundary covers >= 95% Scope 1+2 | All material entities included | `sum(entity_scope12) / total_scope12 >= 0.95` |
| C2 | Boundary consistent with financial reporting | Consolidation approach aligns with GAAP/IFRS | Config consolidation approach validation |
| C3 | Scope 1+2 separation | Scope 1 and Scope 2 reported separately | Separate engine outputs |
| C4 | Scope 2 dual reporting | Location-based and market-based both reported | MRV-009 + MRV-010 + MRV-013 |
| C5 | Scope 3 screening | All 15 categories assessed for materiality | Materiality engine assessment |
| C6 | Base year recent | Within 2 most recent completed years | `config.base_year >= submission_year - 2` |
| C7 | Base year not too old | No older than target submission minus 2 | Date validation |
| C8 | Recalculation policy | Written policy for base year recalculation | Policy template generated |
| C9 | Recalculation triggers defined | 5% significance threshold documented | Trigger assessment logic |
| C10 | Scope 1+2 minimum ambition | ACA >= 4.2%/yr (1.5C) or >= 2.5%/yr (WB2C) | `annual_reduction_rate >= 0.042` |
| C11 | Scope 1+2 not intensity-only | Absolute target required (intensity allowed as supplementary) | Target type validation |
| C12 | ACA calculation correct | Linear annual reduction from base to target | `(base - target) / (target_yr - base_yr)` |
| C13 | SDA sector classification | Correct GICS/NACE mapping to SDA sector | Preset sector validation |
| C14 | SDA convergence validated | Intensity target meets SDA benchmark | Cross-reference SDA tool |
| C15 | SDA production metric correct | Correct physical output metric used | Sector-specific validation |
| C16 | Target timeframe 5-10 years | From date of submission | `5 <= (target_year - submission_year) <= 10` |
| C17 | Target year not > 10 years from base | Maximum 10-year gap | `target_year - base_year <= 10` |
| C18 | Single base year | Same base year for all targets | Config base year consistency |
| C19 | Scope 3 coverage >= 67% | Of total Scope 3 emissions | `covered_scope3 / total_scope3 >= 0.67` |
| C20 | Scope 3 material categories included | All categories > 5% individually included | Category materiality check |
| C21 | Scope 3 target quantified | Absolute or intensity with clear metric | Target type validation |
| C22 | Supplier engagement target | If supplier engagement approach chosen | Engagement metrics defined |
| C23 | Scope 3 target minimum ambition | >= 2.5%/yr for absolute targets | `scope3_annual_reduction >= 0.025` |
| C24 | Annual disclosure commitment | Commitment to annual progress reporting | Policy in submission |
| C25 | GHG Protocol methodology | GHG Protocol Corporate Standard used | Methodology reference |
| C26 | Gases included | CO2, CH4, N2O, HFCs, PFCs, SF6, NF3 | All 7 gases covered |
| C27 | Progress tracking defined | Methodology for annual tracking documented | Tracking workflow reference |
| C28 | Restatement policy | Conditions for restating progress documented | Policy template |

### Net-Zero Criteria (NZ-C1 to NZ-C14)

| # | Criterion | Requirement | PACK-027 Check |
|---|-----------|-------------|---------------|
| NZ-C1 | Long-term target set | 90%+ absolute reduction by 2050 | `long_term_reduction_pct >= 0.90` |
| NZ-C2 | Long-term target year | No later than 2050 | `long_term_target_year <= 2050` |
| NZ-C3 | Scope 1+2 long-term coverage | >= 95% of Scope 1+2 | Coverage validation |
| NZ-C4 | Scope 3 long-term coverage | >= 90% of Scope 3 | `scope3_lt_coverage >= 0.90` |
| NZ-C5 | Residual emissions defined | <= 10% of base year after reduction | `residual <= base_year * 0.10` |
| NZ-C6 | Neutralization plan | Plan for neutralizing residual emissions | Neutralization engine |
| NZ-C7 | Permanent CDR | Neutralization via permanent carbon dioxide removal | CDR type validation |
| NZ-C8 | Credit quality | Credits meet SBTi quality criteria | VCMI/Oxford principles |
| NZ-C9 | Near-term target set | C1-C28 criteria satisfied | Near-term validation result |
| NZ-C10 | Interim milestones | Milestones every 5 years from near-term to long-term | Milestone generation |
| NZ-C11 | Linear or front-loaded pathway | Pathway is at minimum linear | Pathway shape validation |
| NZ-C12 | Board-level oversight | Board or senior management oversight documented | Governance section |
| NZ-C13 | Annual progress reporting | Annual public reporting on progress | Reporting commitment |
| NZ-C14 | Five-year review | Commitment to revalidate every 5 years | Review schedule |

---

## SDA Sector Pathways

### Pathway Benchmarks

| Sector | Metric | 2025 | 2030 | 2035 | 2040 | 2050 | Source |
|--------|--------|------|------|------|------|------|--------|
| Power | tCO2/MWh | 0.28 | 0.14 | 0.07 | 0.03 | 0.00 | IEA NZE |
| Cement | tCO2/t | 0.52 | 0.42 | 0.32 | 0.20 | 0.07 | SBTi SDA |
| Steel | tCO2/t | 1.32 | 1.06 | 0.72 | 0.38 | 0.05 | SBTi SDA |
| Aluminium | tCO2/t | 4.80 | 3.10 | 2.10 | 1.20 | 0.20 | SBTi SDA |
| Pulp & paper | tCO2/t | 0.34 | 0.22 | 0.16 | 0.10 | 0.04 | SBTi SDA |
| Aviation | gCO2/pkm | 82 | 62 | 42 | 24 | 8 | SBTi SDA |
| Shipping | gCO2/tkm | 8.4 | 5.8 | 3.8 | 2.0 | 0.8 | SBTi SDA |
| Road transport | gCO2/vkm | 120 | 85 | 48 | 18 | 0 | SBTi SDA |
| Commercial bldg | kgCO2/sqm | 35 | 25 | 16 | 8 | 2 | SBTi SDA |
| Residential bldg | kgCO2/sqm | 18 | 12 | 8 | 4 | 1 | SBTi SDA |

---

## FLAG Targets

### When FLAG Targets Are Required

FLAG targets are mandatory when Forest, Land, and Agriculture emissions exceed 20% of total Scope 1+2+3 emissions. Common in:

- Food and beverage companies (agricultural supply chain)
- Consumer goods with agricultural inputs (palm oil, soy, cotton)
- Forestry and timber companies
- Agricultural enterprises

### FLAG Calculation

```python
flag_result = engine.calculate_flag(
    baseline=baseline,
    commodities=["palm_oil", "soy", "beef", "dairy", "timber"],
)

print(f"FLAG emissions: {flag_result.flag_tco2e:,.0f} tCO2e")
print(f"Non-FLAG emissions: {flag_result.non_flag_tco2e:,.0f} tCO2e")
print(f"FLAG percentage: {flag_result.flag_pct:.1f}%")
print(f"FLAG target required: {flag_result.flag_pct > 20}")
print(f"FLAG reduction rate: 3.03%/yr")
print(f"No-deforestation by: 2025")
```

---

## Common Rejection Reasons

Based on SBTi historical validation data, the most common reasons for target rejection:

| # | Reason | Frequency | PACK-027 Prevention |
|---|--------|-----------|---------------------|
| 1 | Scope 3 coverage below 67% | 25% | Automated coverage calculation and warning |
| 2 | Base year too old | 18% | Date validation in config |
| 3 | Ambition level below minimum | 15% | Annual reduction rate validation |
| 4 | Missing Scope 3 categories | 12% | 15-category materiality screening |
| 5 | Incorrect SDA sector classification | 10% | Preset sector mapping validation |
| 6 | Boundary inconsistent with financial | 8% | Consolidation approach cross-check |
| 7 | No recalculation policy | 5% | Auto-generated recalculation policy |
| 8 | FLAG assessment missing | 4% | Automatic FLAG screening |
| 9 | Scope 2 not dual-reported | 2% | Enforced by MRV-009/010/013 |
| 10 | Net-zero without near-term | 1% | Sequential validation enforcement |

PACK-027 prevents all 10 common rejection reasons through automated validation.

---

## Troubleshooting

### Criteria Failing

**Problem:** One or more of the 42 criteria fails validation.

**Solution:**
```python
# Get detailed remediation for failing criteria
for c in validation.criteria:
    if not c.passed:
        print(f"\n--- {c.id}: {c.description} ---")
        print(f"Current value: {c.current_value}")
        print(f"Required value: {c.required_value}")
        print(f"Remediation steps:")
        for step in c.remediation_steps:
            print(f"  {step.order}. {step.action}")
            print(f"     Effort: {step.effort}")
            print(f"     Timeline: {step.timeline}")
```

### Scope 3 Coverage Too Low

**Problem:** Scope 3 coverage below 67% for near-term target.

**Solution:** Expand Scope 3 calculation to additional categories:
```python
# Identify categories that would increase coverage
gap_analysis = engine.scope3_coverage_gap(
    baseline=baseline,
    target_coverage=0.67,
)

print(f"Current coverage: {gap_analysis.current_coverage:.1%}")
print(f"Gap to 67%: {gap_analysis.gap_tco2e:,.0f} tCO2e")
print(f"Categories to add for 67% coverage:")
for cat in gap_analysis.recommended_additions:
    print(f"  Cat {cat.number}: {cat.name} - {cat.estimated_tco2e:,.0f} tCO2e")
```

### Base Year Recalculation Needed

**Problem:** Structural change since base year triggers recalculation.

**Solution:**
```python
from engines.multi_entity_consolidation_engine import MultiEntityConsolidationEngine

engine = MultiEntityConsolidationEngine(config=config)
recalc = engine.assess_recalculation_trigger(
    base_year_result=baseline,
    trigger_event="acquisition",
    impact_tco2e=250_000,
)

print(f"Significance: {recalc.significance_pct:.1f}%")
print(f"Recalculation required: {recalc.required}")
if recalc.required:
    new_base = engine.recalculate_base_year(baseline, recalc)
    print(f"Old base year: {new_base.old_total:,.0f} tCO2e")
    print(f"New base year: {new_base.new_total:,.0f} tCO2e")
```

### SDA Sector Not Eligible

**Problem:** Organization's sector is not in the SDA-eligible list.

**Solution:** Use ACA pathway instead, or apply SDA to eligible divisions only:
```python
# Check if any divisions are SDA-eligible
for division in config.divisions:
    sda_check = engine.check_sda_eligibility(sector=division.sector)
    print(f"{division.name}: SDA eligible = {sda_check.eligible}")
    if sda_check.eligible:
        print(f"  SDA sector: {sda_check.sda_sector}")
        print(f"  Intensity metric: {sda_check.intensity_metric}")

# If no divisions eligible: use ACA for entire organization
```

---

## Appendix: SBTi Validation Fee Schedule

| Company Revenue | Commitment Fee | Validation Fee | Total |
|----------------|---------------|----------------|-------|
| <$1M | Free | $1,000 | $1,000 |
| $1M - $100M | Free | $2,500 | $2,500 |
| $100M - $1B | Free | $7,500 | $7,500 |
| $1B - $10B | Free | $9,500 | $9,500 |
| >$10B | Free | $14,500 | $14,500 |

Note: Fees as of 2025. Check SBTi website for current pricing.

---

## Appendix: Useful Links

- SBTi Corporate Manual V5.3: https://sciencebasedtargets.org/resources/files/SBTi-Corporate-Manual.pdf
- SBTi Net-Zero Standard V1.3: https://sciencebasedtargets.org/net-zero
- SBTi Target Setting Tool: https://sciencebasedtargets.org/target-setting-tool
- SBTi FLAG Guidance: https://sciencebasedtargets.org/sectors/forest-land-and-agriculture
- SBTi SDA Tool: https://sciencebasedtargets.org/sectors
- SBTi Dashboard: https://sciencebasedtargets.org/companies-taking-action
