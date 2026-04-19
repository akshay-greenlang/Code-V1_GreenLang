# Regulatory Compliance Guide: SBTi Corporate Net-Zero Standard

**Pack:** PACK-029 Interim Targets Pack
**Version:** 1.0.0
**Standard:** SBTi Corporate Net-Zero Standard v1.2
**Reference:** SBTi Corporate Manual v5.3

---

## Table of Contents

1. [Overview](#overview)
2. [SBTi Corporate Net-Zero Standard Summary](#sbti-corporate-net-zero-standard-summary)
3. [Near-Term Target Requirements](#near-term-target-requirements)
4. [Long-Term Target Requirements](#long-term-target-requirements)
5. [Interim Target Requirements](#interim-target-requirements)
6. [21-Criteria Validation Checklist](#21-criteria-validation-checklist)
7. [Scope Coverage Requirements](#scope-coverage-requirements)
8. [Ambition Level Thresholds](#ambition-level-thresholds)
9. [FLAG Sector Requirements](#flag-sector-requirements)
10. [Recalculation Triggers](#recalculation-triggers)
11. [Submission Process](#submission-process)
12. [Annual Disclosure Requirements](#annual-disclosure-requirements)
13. [Common Non-Compliance Issues](#common-non-compliance-issues)
14. [PACK-029 SBTi Mapping](#pack-029-sbti-mapping)

---

## Overview

The Science Based Targets initiative (SBTi) Corporate Net-Zero Standard provides a framework for companies to set emission reduction targets consistent with limiting global warming to 1.5 degrees Celsius above pre-industrial levels. PACK-029 implements the full SBTi validation framework, ensuring that all interim targets, monitoring, and reporting align with the Standard's requirements.

### Key Documents

| Document | Version | Purpose | PACK-029 Coverage |
|----------|---------|---------|-------------------|
| SBTi Corporate Net-Zero Standard | v1.2 (Oct 2023) | Target-setting framework | 100% |
| SBTi Corporate Manual | v5.3 (2024) | Detailed criteria and thresholds | 100% |
| SBTi FLAG Guidance | v1.1 (2023) | Forest, Land and Agriculture targets | 100% |
| SBTi Criteria Assessment Indicators | v5.3 | Validation criteria | 21/21 criteria |
| SBTi Target Validation Protocol | v3.0 | Submission process | Supported |

### SBTi Target Architecture

```
                        NET-ZERO TARGET
                     (90%+ by 2050 max)
                           |
              +------------+------------+
              |                         |
        NEAR-TERM TARGET         LONG-TERM TARGET
       (5-10 year horizon)      (by 2050 at latest)
       Min 42% (1.5C) or        Min 90% reduction
       25% (WB2C) reduction      + neutralization
              |
    +---------+---------+
    |         |         |
  5-year    Annual    Quarterly
  Interim   Review    Monitoring
  Targets   (PACK-029) (PACK-029)
```

---

## SBTi Corporate Net-Zero Standard Summary

### Core Pillars

1. **Near-Term Targets**: Rapid, deep emission cuts within 5-10 years
2. **Long-Term Targets**: Reduce emissions by at least 90% by no later than 2050
3. **Beyond Value Chain Mitigation (BVCM)**: Invest in mitigation outside the value chain
4. **Neutralization**: Neutralize residual emissions (max 10%) through permanent removals

### Key Principles

- **Science-based**: Targets must be consistent with 1.5 degrees Celsius pathways
- **Comprehensive**: Cover all material scopes and categories
- **Ambitious**: No backsliding; continuous year-on-year reduction
- **Transparent**: Annual public disclosure of progress
- **Time-bound**: Clear timelines with interim milestones

### Target Types

| Target Type | Metric | Timeframe | Minimum Ambition |
|-------------|--------|-----------|-----------------|
| Near-term (Scope 1+2) | Absolute or intensity | 5-10 years from submission | 42% absolute (1.5C) |
| Near-term (Scope 3) | Absolute or intensity | 5-10 years from submission | 25% absolute (WB2C) |
| Long-term (all scopes) | Absolute only | By 2050 at latest | 90% absolute |
| FLAG (if applicable) | Absolute, commodity-specific | 2030 (near), 2050 (long) | FLAG pathway |

---

## Near-Term Target Requirements

### Scope 1+2 Near-Term Targets

| Criterion | 1.5C Aligned | Well-Below 2C |
|-----------|-------------|---------------|
| Minimum reduction | 42% | 25% |
| Maximum timeframe | 10 years from submission | 10 years from submission |
| Minimum timeframe | 5 years from submission | 5 years from submission |
| Implied annual rate | ~4.2%/year linear | ~2.5%/year linear |
| Base year | No earlier than 2015 | No earlier than 2015 |
| Coverage | 95% of Scope 1+2 emissions | 95% of Scope 1+2 emissions |
| Metric type | Absolute or intensity | Absolute or intensity |

### Scope 3 Near-Term Targets

| Criterion | Requirement |
|-----------|-------------|
| Trigger | Required if Scope 3 >= 40% of total (Scope 1+2+3) |
| Minimum reduction | 25% absolute (WB2C minimum) |
| Maximum timeframe | 10 years from submission |
| Coverage | 67% of Scope 3 emissions |
| Categories | Must cover 2/3 of Scope 3 by emissions volume |
| Metric type | Absolute or physical/economic intensity |
| Engagement targets | Supplier/customer engagement targets allowed for up to 100% |

### Near-Term Target Validation in PACK-029

```python
# From sbti_validation_engine.py
def _validate_near_term(self, target: InterimTargetResult) -> list[ValidationResult]:
    checks = []

    # C1: Timeframe (5-10 years)
    years = target.target_year - target.base_year
    checks.append(ValidationResult(
        criterion="C1",
        name="Near-term timeframe",
        passed=5 <= years <= 10,
        detail=f"Timeframe: {years} years"
    ))

    # C2: Ambition level
    if target.ambition == ClimateAmbition.CELSIUS_1_5:
        min_reduction = Decimal("42.0")
    else:
        min_reduction = Decimal("25.0")

    actual_reduction = target.total_reduction_pct
    checks.append(ValidationResult(
        criterion="C2",
        name="Ambition level",
        passed=actual_reduction >= min_reduction,
        detail=f"Reduction: {actual_reduction}% (minimum: {min_reduction}%)"
    ))

    # C3: Scope coverage
    checks.append(ValidationResult(
        criterion="C3",
        name="Scope 1+2 coverage",
        passed=target.scope_1_2_coverage >= Decimal("95.0"),
        detail=f"Coverage: {target.scope_1_2_coverage}% (minimum: 95%)"
    ))

    return checks
```

---

## Long-Term Target Requirements

### Absolute Reduction Requirements

| Scope | Minimum Reduction | Deadline | Notes |
|-------|-------------------|----------|-------|
| Scope 1 | 90% | By 2050 | From base year |
| Scope 2 | 90% | By 2050 | From base year |
| Scope 3 | 90% | By 2050 | From base year |
| All scopes combined | 90% | By 2050 | Weighted average |

### Neutralization of Residual Emissions

After achieving 90%+ reduction, the remaining emissions (up to 10%) must be neutralized:

```
Residual emissions = Baseline * (1 - reduction_achieved / 100)

If reduction_achieved >= 90%:
    Neutralization required = Residual emissions
    Method = Permanent carbon dioxide removals (CDR)
    Acceptable CDR methods:
        - Direct Air Capture with Carbon Storage (DACCS)
        - Bioenergy with Carbon Capture and Storage (BECCS)
        - Enhanced weathering
        - Afforestation/reforestation (with permanence guarantees)
    NOT acceptable:
        - Avoided emissions offsets
        - Renewable energy certificates
        - Traditional carbon offsets (unless permanent removal)
```

### Long-Term Target Validation in PACK-029

```python
# C11: Long-term reduction level
checks.append(ValidationResult(
    criterion="C11",
    name="Long-term reduction >= 90%",
    passed=target.long_term_reduction_pct >= Decimal("90.0"),
    detail=f"Long-term reduction: {target.long_term_reduction_pct}%"
))

# C12: Long-term deadline
checks.append(ValidationResult(
    criterion="C12",
    name="Long-term target year <= 2050",
    passed=target.long_term_target_year <= 2050,
    detail=f"Target year: {target.long_term_target_year}"
))

# C13: Neutralization plan
checks.append(ValidationResult(
    criterion="C13",
    name="Neutralization plan for residual emissions",
    passed=target.has_neutralization_plan,
    detail="Neutralization plan present" if target.has_neutralization_plan else "Missing"
))
```

---

## Interim Target Requirements

### SBTi Guidance on Interim Targets

While SBTi does not mandate specific interim milestones between near-term and long-term targets, PACK-029 generates them based on SBTi-consistent pathways:

| Interim Period | Purpose | PACK-029 Feature |
|---------------|---------|-------------------|
| 5-year milestones | Track progress toward near-term target | Interim Target Engine |
| Annual pathway points | Year-by-year expected emissions | Annual Review Engine |
| Quarterly checkpoints | In-year progress monitoring | Quarterly Monitoring Engine |

### Pathway Consistency

All interim targets must satisfy:

1. **Monotonic reduction**: Each milestone must be lower than the previous one
2. **No backsliding**: Year-over-year targets must decrease
3. **Terminal consistency**: The final interim target must equal the near-term target
4. **Pathway linearity**: For 1.5C targets, the pathway must be at least as ambitious as linear (no back-loading that defers reductions)

### PACK-029 Interim Target Validation

```python
def _validate_interim_consistency(self, milestones: list[InterimMilestone]) -> bool:
    """Validate that interim targets form a valid reduction pathway."""
    for i in range(1, len(milestones)):
        # No backsliding
        if milestones[i].target_emissions >= milestones[i-1].target_emissions:
            return False
        # Reduction rate consistency
        annual_rate = (milestones[i-1].target_emissions - milestones[i].target_emissions) / \
                      milestones[i-1].target_emissions
        if annual_rate < self.minimum_annual_rate:
            self.warnings.append(
                f"Milestone {milestones[i].year}: annual rate {annual_rate:.1%} "
                f"below minimum {self.minimum_annual_rate:.1%}"
            )
    return True
```

---

## 21-Criteria Validation Checklist

PACK-029's SBTi Validation Engine checks all 21 criteria from the SBTi Corporate Manual v5.3:

### Boundary and Coverage Criteria (C1-C5)

| # | Criterion | Description | PACK-029 Check |
|---|-----------|-------------|----------------|
| C1 | Timeframe | Near-term: 5-10 years from submission | `target_year - submission_year in [5, 10]` |
| C2 | Base year | No earlier than 2015 | `base_year >= 2015` |
| C3 | Scope 1+2 coverage | >= 95% of Scope 1+2 emissions | `scope_1_2_coverage >= 95%` |
| C4 | Scope 3 screening | Complete Scope 3 screening performed | `scope_3_screening_complete == True` |
| C5 | Scope 3 coverage | >= 67% if Scope 3 >= 40% of total | `scope_3_coverage >= 67% if scope_3_material` |

### Ambition Criteria (C6-C10)

| # | Criterion | Description | PACK-029 Check |
|---|-----------|-------------|----------------|
| C6 | Scope 1+2 ambition | >= 42% (1.5C) or 25% (WB2C) absolute | `reduction_pct >= threshold` |
| C7 | Scope 3 ambition | >= 25% absolute (WB2C minimum) | `scope_3_reduction >= 25%` |
| C8 | No carbon credits | Offsets not counted toward target | `carbon_credits_excluded == True` |
| C9 | No backsliding | Year-over-year reduction required | `all(milestones monotonically decreasing)` |
| C10 | Annual rate | >= 4.2%/yr (1.5C) or 2.5%/yr (WB2C) | `annual_rate >= threshold` |

### Long-Term and Net-Zero Criteria (C11-C15)

| # | Criterion | Description | PACK-029 Check |
|---|-----------|-------------|----------------|
| C11 | Long-term reduction | >= 90% from base year | `long_term_reduction >= 90%` |
| C12 | Long-term deadline | By 2050 at latest | `long_term_year <= 2050` |
| C13 | Neutralization plan | Plan for residual emissions | `neutralization_plan_exists` |
| C14 | Absolute metric (LT) | Long-term must be absolute, not intensity | `long_term_metric == "absolute"` |
| C15 | All scopes (LT) | Long-term covers Scope 1, 2, and 3 | `long_term_scope_coverage == "all"` |

### FLAG Criteria (C16-C18)

| # | Criterion | Description | PACK-029 Check |
|---|-----------|-------------|----------------|
| C16 | FLAG applicability | FLAG target if sector revenue >= 20% from FLAG | `flag_required and flag_target_set` |
| C17 | FLAG pathway | FLAG commodities follow FLAG pathway | `flag_pathway_aligned` |
| C18 | FLAG separation | FLAG targets separate from non-FLAG | `flag_targets_separate` |

### Reporting and Governance Criteria (C19-C21)

| # | Criterion | Description | PACK-029 Check |
|---|-----------|-------------|----------------|
| C19 | Annual disclosure | Public annual progress report | `annual_disclosure_enabled` |
| C20 | Board oversight | Board/executive oversight of targets | `governance_oversight_documented` |
| C21 | Recalculation policy | Policy for when to recalculate base year | `recalculation_policy_exists` |

### Validation Output

```json
{
  "sbti_validation": {
    "overall_status": "PASS",
    "criteria_passed": 21,
    "criteria_failed": 0,
    "criteria_warning": 2,
    "details": [
      {"criterion": "C1", "name": "Timeframe", "status": "PASS", "detail": "8 years (5-10 range)"},
      {"criterion": "C2", "name": "Base year", "status": "PASS", "detail": "2021 (>= 2015)"},
      {"criterion": "C3", "name": "Scope 1+2 coverage", "status": "PASS", "detail": "98.5% (>= 95%)"},
      {"criterion": "C4", "name": "Scope 3 screening", "status": "PASS", "detail": "All 15 categories screened"},
      {"criterion": "C5", "name": "Scope 3 coverage", "status": "PASS", "detail": "72.3% (>= 67%)"},
      {"criterion": "C6", "name": "Scope 1+2 ambition", "status": "PASS", "detail": "46.5% (>= 42% for 1.5C)"},
      {"criterion": "C7", "name": "Scope 3 ambition", "status": "PASS", "detail": "28.0% (>= 25%)"},
      {"criterion": "C8", "name": "No carbon credits", "status": "PASS", "detail": "Credits excluded from target"},
      {"criterion": "C9", "name": "No backsliding", "status": "PASS", "detail": "All milestones monotonically decreasing"},
      {"criterion": "C10", "name": "Annual rate", "status": "PASS", "detail": "4.6%/yr (>= 4.2%)"}
    ],
    "recommendations": [
      "Consider increasing Scope 3 coverage from 72.3% to 80%+ for stronger SBTi submission",
      "Ensure annual disclosure process is established before first reporting cycle"
    ]
  }
}
```

---

## Scope Coverage Requirements

### Scope 1+2 Coverage

```
Required: >= 95% of combined Scope 1 + Scope 2 emissions

Calculation:
    Coverage_pct = (Scope_1_covered + Scope_2_covered) / (Scope_1_total + Scope_2_total) * 100

Exclusions allowed (up to 5%):
    - De minimis sources (< 1% of total individually)
    - Sources with no credible quantification method
    - Must be documented and justified
```

### Scope 3 Materiality Test

```
Scope_3_material = (Scope_3_total / (Scope_1 + Scope_2 + Scope_3)) >= 0.40

If Scope_3_material:
    Scope 3 target REQUIRED
    Coverage >= 67% of Scope 3 emissions
    Must cover at least 2/3 of emissions by category volume
```

### Scope 3 Category Prioritization

PACK-029 helps identify which Scope 3 categories to include for 67% coverage:

```python
def _identify_scope3_coverage(self, categories: list[Scope3Category]) -> CoverageResult:
    """Select Scope 3 categories to meet 67% coverage threshold."""
    # Sort by emissions volume (descending)
    sorted_cats = sorted(categories, key=lambda c: c.emissions, reverse=True)

    total_scope3 = sum(c.emissions for c in categories)
    target_coverage = total_scope3 * Decimal("0.67")

    selected = []
    cumulative = Decimal("0")
    for cat in sorted_cats:
        selected.append(cat)
        cumulative += cat.emissions
        if cumulative >= target_coverage:
            break

    return CoverageResult(
        selected_categories=selected,
        coverage_pct=(cumulative / total_scope3 * 100),
        remaining_categories=[c for c in sorted_cats if c not in selected]
    )
```

---

## Ambition Level Thresholds

### PACK-029 Threshold Constants

```python
SBTI_THRESHOLDS = {
    "CELSIUS_1_5": {
        "near_term_min_pct": Decimal("42.0"),
        "annual_rate_min": Decimal("4.2"),
        "scope_1_2_coverage_min": Decimal("95.0"),
        "scope_3_coverage_min": Decimal("67.0"),
        "scope_3_materiality_threshold": Decimal("40.0"),
        "long_term_min_pct": Decimal("90.0"),
        "long_term_max_year": 2050,
        "base_year_earliest": 2015,
        "near_term_min_years": 5,
        "near_term_max_years": 10,
    },
    "WELL_BELOW_2C": {
        "near_term_min_pct": Decimal("25.0"),
        "annual_rate_min": Decimal("2.5"),
        "scope_1_2_coverage_min": Decimal("95.0"),
        "scope_3_coverage_min": Decimal("67.0"),
        "scope_3_materiality_threshold": Decimal("40.0"),
        "long_term_min_pct": Decimal("90.0"),
        "long_term_max_year": 2050,
        "base_year_earliest": 2015,
        "near_term_min_years": 5,
        "near_term_max_years": 10,
    },
    "RACE_TO_ZERO": {
        "near_term_min_pct": Decimal("50.0"),
        "annual_rate_min": Decimal("7.0"),
        "scope_1_2_coverage_min": Decimal("95.0"),
        "scope_3_coverage_min": Decimal("67.0"),
        "scope_3_materiality_threshold": Decimal("40.0"),
        "long_term_min_pct": Decimal("90.0"),
        "long_term_max_year": 2050,
        "base_year_earliest": 2015,
        "near_term_min_years": 5,
        "near_term_max_years": 10,
    },
}
```

### Ambition Level Comparison

```
                 1.5C Aligned    WB2C         Race to Zero
Near-term min:   42%             25%          50%
Annual rate:     4.2%/yr         2.5%/yr      7.0%/yr
Long-term:       90%             90%          90%
Deadline:        2050            2050         2050
SBTi validation: Yes             Yes          Via SBTi
```

### Temperature Score

PACK-029 calculates an implied temperature score for the target:

```
Temperature score estimation (simplified):
    If annual_rate >= 7.0%:   T = 1.5C (Race to Zero)
    If annual_rate >= 4.2%:   T = 1.5C
    If annual_rate >= 2.5%:   T = 1.8C (Well-Below 2C)
    If annual_rate >= 1.5%:   T = 2.0C
    If annual_rate < 1.5%:    T = 2.5C+ (insufficient)
```

---

## FLAG Sector Requirements

### FLAG Applicability

FLAG (Forest, Land and Agriculture) targets are required when:

```
FLAG_required = (FLAG_sector_revenue / Total_revenue) >= 0.20

FLAG sectors include:
    - Agriculture
    - Forestry
    - Paper and forest products
    - Food production
    - Beverages
    - Tobacco
    - Textiles (natural fibers)
```

### FLAG Target Requirements

| Requirement | Details |
|-------------|---------|
| Separate targets | FLAG and non-FLAG targets must be set separately |
| FLAG pathway | Follow SBTi FLAG sector pathway |
| No netting | Cannot net FLAG removals against non-FLAG emissions |
| Land-based removals | Allowed only within FLAG target, not non-FLAG |
| Deforestation | Zero deforestation commitment required by 2025 |
| Near-term deadline | 2030 for FLAG near-term targets |

### PACK-029 FLAG Implementation

```python
def _calculate_flag_targets(self, input_data: InterimTargetInput) -> FLAGTargetResult:
    """Calculate FLAG-specific interim targets."""
    if not input_data.flag_applicable:
        return FLAGTargetResult(applicable=False)

    # FLAG uses its own pathway, separate from non-FLAG
    flag_baseline = input_data.baseline.flag_emissions
    flag_pathway = self._get_flag_pathway(input_data.flag_commodities)

    # Generate FLAG milestones
    flag_milestones = []
    for year in range(input_data.base_year, 2031):
        reduction = flag_pathway.get_reduction_pct(year)
        flag_milestones.append(InterimMilestone(
            year=year,
            target_emissions=flag_baseline * (1 - reduction / 100),
            reduction_from_baseline_pct=reduction,
        ))

    return FLAGTargetResult(
        applicable=True,
        flag_baseline=flag_baseline,
        flag_milestones=flag_milestones,
        zero_deforestation_year=2025,
        flag_commodities=input_data.flag_commodities,
    )
```

---

## Recalculation Triggers

### SBTi Recalculation Policy

The SBTi requires companies to recalculate their base year emissions and targets when significant changes occur:

| Trigger | Threshold | Action Required |
|---------|-----------|----------------|
| Acquisition | >= 5% change in base year emissions | Recalculate base year within 6 months |
| Divestment | >= 5% change in base year emissions | Recalculate base year within 6 months |
| Methodology change | Any material change to calculation method | Recalculate base year and targets |
| Emission factor update | Significant change in standard EFs | Recalculate if material impact |
| Structural change | Merger, spin-off, reorganization | Recalculate base year |
| Scope boundary change | Operational/financial control change | Recalculate affected scopes |
| Error correction | Discovery of material error | Recalculate and disclose |

### PACK-029 Recalculation Engine

```python
class RecalibrationTrigger(BaseModel):
    trigger_type: str        # acquisition, divestment, methodology, error, etc.
    description: str
    impact_tco2e: Decimal    # Estimated change in base year emissions
    impact_pct: Decimal      # Percentage change in base year
    requires_recalculation: bool
    deadline: date           # Date by which recalculation must be completed

def _evaluate_trigger(self, trigger: RecalibrationTrigger) -> RecalibrationResult:
    """Evaluate whether a trigger requires target recalculation."""
    threshold = self.config.recalibration_threshold  # Default: 5%

    if trigger.impact_pct >= threshold:
        return RecalibrationResult(
            recalculation_required=True,
            new_base_year_emissions=self._adjust_baseline(trigger),
            new_targets=self._recalculate_targets(trigger),
            deadline=trigger.deadline,
            disclosure_required=True,
        )
    else:
        return RecalibrationResult(
            recalculation_required=False,
            note=f"Impact ({trigger.impact_pct:.1f}%) below threshold ({threshold:.1f}%)"
        )
```

### Recalculation Methodology

```
Step 1: Adjust base year emissions
    New_baseline = Old_baseline + Acquisition_emissions (or - Divestment_emissions)

Step 2: Maintain ambition level
    New targets use the same percentage reduction as original targets
    Example: If original target was 42% reduction from 200,000 tCO2e = 116,000 tCO2e
             After acquisition adding 30,000 tCO2e:
             New baseline = 230,000 tCO2e
             New target = 230,000 * (1 - 0.42) = 133,400 tCO2e

Step 3: Recalculate interim milestones
    All interim targets are recalculated from the adjusted baseline
    The pathway shape and annual rate are preserved

Step 4: Update monitoring baselines
    All quarterly and annual monitoring comparisons use the new targets
```

---

## Submission Process

### SBTi Target Submission Workflow

```
Step 1: Commitment Letter
    - Sign the SBTi Commitment Letter
    - 24-month deadline to submit targets after commitment

Step 2: Target Development
    - Use PACK-029 Interim Target Engine
    - Run SBTi Validation Engine (21-criteria check)
    - Ensure all criteria pass

Step 3: Target Submission
    - Submit via SBTi Online Platform
    - Provide: base year data, target year, reduction %, scope coverage
    - PACK-029 SBTi Portal Bridge automates data preparation

Step 4: Validation
    - SBTi validates targets against criteria
    - May request clarifications or adjustments
    - Typical timeline: 8-12 weeks

Step 5: Approval and Publication
    - Targets published on SBTi website
    - Company added to SBTi Companies Taking Action database

Step 6: Annual Reporting
    - Disclose progress annually (CDP preferred channel)
    - PACK-029 Annual Reporting Workflow automates this
```

### PACK-029 Submission Support

```python
# SBTi Portal Bridge generates submission data
submission = sbti_portal_bridge.prepare_submission(
    interim_target_result=target_result,
    sbti_validation_result=validation_result,
    company_info=company_profile,
)

# Output: SBTi submission-ready data package
{
    "company_name": "Acme Manufacturing",
    "sector": "Manufacturing",
    "base_year": 2021,
    "near_term_target": {
        "target_year": 2030,
        "scope_1_2_reduction_pct": 46.5,
        "scope_3_reduction_pct": 28.0,
        "ambition_level": "1.5C",
        "metric": "absolute"
    },
    "long_term_target": {
        "target_year": 2050,
        "all_scopes_reduction_pct": 90.0,
        "neutralization_plan": true
    },
    "validation_status": "21/21 criteria passed"
}
```

---

## Annual Disclosure Requirements

### What to Disclose

| Item | Frequency | Channel | PACK-029 Template |
|------|-----------|---------|-------------------|
| Scope 1+2 actual emissions | Annual | CDP, annual report | Annual Progress Report |
| Scope 3 actual emissions | Annual | CDP, annual report | Annual Progress Report |
| Progress vs. near-term target | Annual | CDP C4.1/C4.2 | CDP Export Template |
| Progress vs. long-term target | Annual | CDP, sustainability report | TCFD Disclosure Template |
| Methodology changes | As needed | CDP, SBTi portal | Recalibration Report |
| Corrective actions (if off-track) | Annual | Sustainability report | Corrective Action Plan |
| Temperature score | Annual | TCFD | Executive Dashboard |

### CDP Alignment

PACK-029 maps SBTi targets to CDP Climate Change questionnaire:

| CDP Question | Content | PACK-029 Source |
|-------------|---------|-----------------|
| C4.1 | Emissions reduction targets | Interim Target Engine output |
| C4.1a | Details of emissions targets | CDP Export Template |
| C4.2 | Progress against targets | Annual Review Engine output |
| C4.2a | Target details and progress % | CDP Export Template |

### TCFD Alignment

PACK-029 maps to TCFD Metrics and Targets pillar:

| TCFD Recommendation | Content | PACK-029 Source |
|--------------------|---------|-----------------|
| Disclose Scope 1, 2, and 3 emissions | Actual emissions by scope | MRV Bridge data |
| Describe targets used | Near-term, long-term, interim | Interim Target Engine |
| Describe performance against targets | Progress, RAG status | Annual Review Engine |

---

## Common Non-Compliance Issues

### Frequent Failures in SBTi Validation

| Issue | Description | PACK-029 Prevention |
|-------|-------------|---------------------|
| Insufficient scope coverage | Scope 1+2 < 95% or Scope 3 < 67% | Coverage check in C3/C5 |
| Carbon credits counted | Offsets included in reduction calculation | C8 check excludes credits |
| Backsliding pathway | Some years show emission increases | C9 monotonic check |
| Base year too old | Base year before 2015 | C2 check |
| Timeframe out of range | Near-term target < 5 or > 10 years | C1 timeframe check |
| Intensity-only long-term | Long-term target uses intensity metric | C14 absolute-only check |
| Missing Scope 3 | Scope 3 material but no target set | C5 materiality check |
| FLAG not separated | FLAG emissions netted against non-FLAG | C18 separation check |
| No neutralization plan | Residual emissions plan missing | C13 check |
| Annual rate too low | Below minimum for chosen ambition | C10 rate check |

### PACK-029 Remediation Guidance

When a criterion fails, PACK-029 provides specific remediation steps:

```python
remediation_guidance = {
    "C1": "Adjust target year to be 5-10 years from submission date.",
    "C2": "Update base year to 2015 or later. Recalculate all targets.",
    "C3": "Expand Scope 1+2 boundary to cover >= 95% of emissions.",
    "C5": "Add more Scope 3 categories until coverage >= 67%.",
    "C6": "Increase Scope 1+2 reduction to >= 42% (1.5C) or 25% (WB2C).",
    "C8": "Remove carbon credit offsets from target reduction calculation.",
    "C9": "Ensure all milestones show year-over-year emission decreases.",
    "C10": "Increase annual reduction rate to >= 4.2%/yr (1.5C) or 2.5%/yr (WB2C).",
    "C11": "Set long-term reduction to >= 90% from base year.",
    "C13": "Develop a neutralization plan for residual emissions (CDR strategy).",
    "C14": "Convert long-term target to absolute metric (not intensity).",
    "C16": "Set separate FLAG targets if FLAG sector revenue >= 20%.",
}
```

---

## PACK-029 SBTi Mapping

### Engine-to-SBTi Mapping

| PACK-029 Component | SBTi Requirement | Coverage |
|--------------------|-----------------|----------|
| Interim Target Engine | Near-term target setting | 100% |
| Interim Target Engine | Long-term target setting | 100% |
| SBTi Validation Engine | 21-criteria compliance | 100% |
| Quarterly Monitoring Engine | Progress tracking | Goes beyond SBTi minimum |
| Annual Review Engine | Annual disclosure data | 100% |
| Variance Analysis Engine | Root cause analysis | Goes beyond SBTi minimum |
| Target Recalibration Engine | Recalculation triggers | 100% |
| Corrective Action Engine | Off-track remediation | Goes beyond SBTi minimum |
| Carbon Budget Tracker | Cumulative budget management | Goes beyond SBTi minimum |
| Alert Generation Engine | Early warning system | Goes beyond SBTi minimum |

### Workflow-to-SBTi Mapping

| PACK-029 Workflow | SBTi Process | Stage |
|-------------------|-------------|-------|
| Interim Target Setting | Target development | Pre-submission |
| Quarterly Monitoring | Progress tracking | Post-approval |
| Annual Progress Review | Annual disclosure | Post-approval |
| Variance Investigation | Root cause analysis | Post-approval |
| Corrective Action Planning | Off-track remediation | Post-approval |
| Annual Reporting | CDP/TCFD disclosure | Post-approval |
| Target Recalibration | Base year recalculation | As needed |

---

## References

- SBTi Corporate Net-Zero Standard v1.2: https://sciencebasedtargets.org/net-zero
- SBTi Corporate Manual v5.3: https://sciencebasedtargets.org/resources/files/SBTi-Corporate-Manual.pdf
- SBTi FLAG Guidance v1.1: https://sciencebasedtargets.org/sectors/forest-land-and-agriculture
- SBTi Criteria Assessment Indicators v5.3: https://sciencebasedtargets.org/resources/files/SBTi-Criteria.pdf
- IPCC AR6 WG3 Mitigation Pathways: https://www.ipcc.ch/report/ar6/wg3/

---

**End of SBTi Compliance Guide**
