# Regulatory Compliance Guide: CDP Climate Change Disclosure

**Pack:** PACK-029 Interim Targets Pack
**Version:** 1.0.0
**Standard:** CDP Climate Change 2025 Questionnaire
**Focus:** C4.1 and C4.2 -- Emissions Reduction Targets

---

## Table of Contents

1. [Overview](#overview)
2. [CDP Climate Change Questionnaire Structure](#cdp-climate-change-questionnaire-structure)
3. [C4.1 -- Emissions Reduction Targets](#c41----emissions-reduction-targets)
4. [C4.1a -- Target Details](#c41a----target-details)
5. [C4.2 -- Progress Against Targets](#c42----progress-against-targets)
6. [C4.2a -- Target Progress Details](#c42a----target-progress-details)
7. [CDP Scoring Methodology](#cdp-scoring-methodology)
8. [PACK-029 CDP Export Mapping](#pack-029-cdp-export-mapping)
9. [Data Quality Requirements](#data-quality-requirements)
10. [Common Disclosure Errors](#common-disclosure-errors)
11. [Best Practices for High Scores](#best-practices-for-high-scores)

---

## Overview

The CDP (formerly Carbon Disclosure Project) is the world's largest environmental disclosure platform. Over 23,000 companies disclose through CDP annually. PACK-029 automates the generation of CDP-compliant data for the emissions reduction targets module (C4), which is one of the highest-weighted sections for CDP scoring.

### CDP Alignment in PACK-029

| CDP Section | PACK-029 Source | Automation Level |
|-------------|-----------------|------------------|
| C4.1 Emissions reduction targets | Interim Target Engine | Fully automated |
| C4.1a Target details | CDP Export Template | Fully automated |
| C4.2 Progress against targets | Annual Review Engine | Fully automated |
| C4.2a Target progress details | CDP Export Template | Fully automated |
| C4.1b SBTi alignment | SBTi Validation Engine | Fully automated |

### Supported CDP Export Formats

- **JSON**: Machine-readable format for CDP API submission
- **XLSX**: Spreadsheet format matching CDP Online Response System (ORS) structure
- **CSV**: Flat file for data integration

---

## CDP Climate Change Questionnaire Structure

### Module C4: Targets and Performance

```
C4. Targets and performance
    |
    +-- C4.1  Did you have an emissions target active during the reporting year?
    |     |
    |     +-- C4.1a  Provide details of your absolute emissions target(s)
    |     +-- C4.1b  Provide details of your emissions intensity target(s)
    |     +-- C4.1c  Provide details of your net-zero target(s)
    |
    +-- C4.2  Did you have any other climate-related targets active?
    |     |
    |     +-- C4.2a  Provide details of your target(s) to increase low-carbon energy
    |     +-- C4.2b  Provide details of any other climate-related targets
    |
    +-- C4.3  Did you have emissions reduction initiatives active?
    |     |
    |     +-- C4.3a  Provide details of total emissions reduced
    |     +-- C4.3b  Provide details of emission reduction initiatives
    |
    +-- C4.4  Provide details of the initiatives that contributed most to emissions reduction
```

PACK-029 focuses on C4.1 and C4.2 (target-related sections), with supporting data for C4.3 and C4.4 from the Corrective Action Engine.

---

## C4.1 -- Emissions Reduction Targets

### Question C4.1

> "Did you have an emissions target that was active (ongoing or reached completion) during the reporting year?"

**Expected answer:** Yes (if interim targets are set)

### C4.1a -- Absolute Emissions Target Details

PACK-029 generates the following fields for each active target:

| CDP Field | Description | PACK-029 Source |
|-----------|-------------|-----------------|
| Target reference number | Unique target ID | `Abs {n}` (auto-generated) |
| Is this a science-based target? | SBTi alignment status | SBTi Validation Engine |
| Target ambition | 1.5C or WB2C | `InterimTargetInput.ambition` |
| Year target was set | Year of target creation | Current year |
| Target coverage | Company-wide or business unit | `InterimTargetInput.scope_coverage` |
| Scope(s) | Scope 1, 2, or 3 | `InterimTargetInput.scope_type` |
| Scope 2 accounting method | Location or market-based | Input configuration |
| Scope 3 categories | If applicable | Input configuration |
| Base year | The reference year | `InterimTargetInput.base_year` |
| Base year Scope 1 emissions | tCO2e | `BaselineData.scope_1_tco2e` |
| Base year Scope 2 emissions | tCO2e | `BaselineData.scope_2_tco2e` |
| Base year Scope 3 emissions | tCO2e | `BaselineData.scope_3_tco2e` |
| Base year total emissions | tCO2e | Calculated sum |
| Target year | Year to achieve target | `InterimTargetInput.target_year` |
| Targeted reduction (%) | Percentage reduction | `InterimTargetResult.total_reduction_pct` |
| % of target achieved | Progress | Annual Review Engine |
| Plan for achieving target | Narrative | Corrective Action Engine |

### C4.1a Export Format

```json
{
  "c4_1a_absolute_targets": [
    {
      "target_reference": "Abs 1",
      "is_science_based": true,
      "target_ambition": "1.5C aligned",
      "year_set": 2024,
      "target_coverage": "Company-wide",
      "scopes": ["Scope 1", "Scope 2"],
      "scope_2_method": "Market-based",
      "base_year": 2021,
      "base_year_emissions": {
        "scope_1": 50000,
        "scope_2": 30000,
        "total": 80000
      },
      "target_year": 2030,
      "targeted_reduction_pct": 46.5,
      "pct_achieved_to_date": 32.0,
      "plan_for_achieving": "Energy efficiency improvements (35%), renewable electricity procurement (40%), electrification of heating (15%), supply chain engagement (10%)"
    },
    {
      "target_reference": "Abs 2",
      "is_science_based": true,
      "target_ambition": "Well-below 2C",
      "year_set": 2024,
      "target_coverage": "Company-wide",
      "scopes": ["Scope 3"],
      "scope_3_categories": ["Cat 1: Purchased goods", "Cat 4: Upstream transport", "Cat 6: Business travel"],
      "base_year": 2021,
      "base_year_emissions": {
        "scope_3": 120000
      },
      "target_year": 2030,
      "targeted_reduction_pct": 28.0,
      "pct_achieved_to_date": 15.0,
      "plan_for_achieving": "Supplier engagement program (50%), logistics optimization (25%), travel policy changes (25%)"
    }
  ]
}
```

### C4.1b -- Intensity Target Details

If intensity targets are also set, PACK-029 generates:

```json
{
  "c4_1b_intensity_targets": [
    {
      "target_reference": "Int 1",
      "is_science_based": true,
      "metric": "Metric tonnes CO2e per unit revenue",
      "metric_numerator": "tCO2e",
      "metric_denominator": "million USD revenue",
      "base_year": 2021,
      "base_year_intensity": 160.0,
      "target_year": 2030,
      "targeted_reduction_pct": 55.0,
      "target_year_intensity": 72.0,
      "pct_achieved_to_date": 28.0,
      "current_intensity": 115.2
    }
  ]
}
```

### C4.1c -- Net-Zero Target Details

```json
{
  "c4_1c_net_zero": {
    "has_net_zero_target": true,
    "target_year": 2050,
    "covers_scopes": ["Scope 1", "Scope 2", "Scope 3"],
    "reduction_at_net_zero": 90,
    "neutralization_approach": "Permanent carbon dioxide removal (DACCS + BECCS)",
    "residual_emissions_pct": 10,
    "interim_target_2030": "46.5% reduction from 2021 baseline",
    "verified_by": "Science Based Targets initiative",
    "sbti_status": "Targets approved"
  }
}
```

---

## C4.2 -- Progress Against Targets

### Question C4.2

> "Provide details of your progress against each of your targets."

### C4.2a -- Target Progress Details

PACK-029 generates progress data for each active target:

| CDP Field | Description | PACK-029 Source |
|-----------|-------------|-----------------|
| Target reference | Matches C4.1a | Same target ID |
| % of target achieved | Progress percentage | Annual Review Engine |
| Target status | Underway, achieved, revised, etc. | Annual Review Engine |
| % emissions in reporting year vs base year | Year-on-year change | Annual Review Engine |
| Please explain | Narrative on progress | Variance Analysis Engine + narratives |

### Progress Calculation

```
% of target achieved = (Base_year_emissions - Current_year_emissions) /
                        (Base_year_emissions - Target_year_emissions) * 100

Example:
    Base year (2021): 80,000 tCO2e
    Target year (2030): 42,800 tCO2e (46.5% reduction)
    Current year (2024): 68,000 tCO2e

    Reduction achieved: 80,000 - 68,000 = 12,000 tCO2e
    Reduction needed: 80,000 - 42,800 = 37,200 tCO2e
    % achieved: 12,000 / 37,200 * 100 = 32.3%
```

### Progress Export Format

```json
{
  "c4_2a_progress": [
    {
      "target_reference": "Abs 1",
      "target_status": "Underway",
      "base_year_emissions": 80000,
      "target_year_emissions": 42800,
      "reporting_year_emissions": 68000,
      "reduction_from_base_year_pct": 15.0,
      "pct_of_target_achieved": 32.3,
      "on_track": true,
      "explanation": "Emissions reduced 15% from 2021 baseline through energy efficiency improvements and renewable electricity procurement. On track to meet 2030 near-term target of 46.5% reduction. Key drivers: LED lighting upgrade (-2,000 tCO2e), solar PV installation (-3,500 tCO2e), green electricity PPA (-6,500 tCO2e)."
    },
    {
      "target_reference": "Abs 2",
      "target_status": "Underway",
      "base_year_emissions": 120000,
      "target_year_emissions": 86400,
      "reporting_year_emissions": 115000,
      "reduction_from_base_year_pct": 4.2,
      "pct_of_target_achieved": 14.9,
      "on_track": false,
      "explanation": "Scope 3 reductions slower than planned. Supplier engagement program launched in 2023 is beginning to show results but needs acceleration. Business travel reductions on track (-20% from baseline). Purchased goods emissions reduced 3% through sustainable procurement policies."
    }
  ]
}
```

---

## CDP Scoring Methodology

### Scoring Overview

CDP scores companies on a scale from D- to A:

```
A    Leadership     (top ~2%)
A-   Leadership     (top ~5%)
B    Management     (top ~20%)
B-   Management     (top ~35%)
C    Awareness      (top ~55%)
C-   Awareness      (top ~70%)
D    Disclosure     (top ~90%)
D-   Disclosure     (all respondents)
F    Failure to disclose
```

### C4 Section Scoring Weight

The targets and performance section (C4) is one of the highest-weighted sections:

| CDP Category | Weight | C4 Contribution |
|-------------|--------|-----------------|
| Disclosure | ~25% | High (target existence) |
| Awareness | ~25% | High (target ambition) |
| Management | ~25% | Very High (progress tracking) |
| Leadership | ~25% | Very High (science-based, on-track) |

### Scoring Criteria for Targets

| Scoring Element | Points | How PACK-029 Helps |
|-----------------|--------|---------------------|
| Has emission reduction target | Base | Interim Target Engine |
| Target is science-based (SBTi validated) | Bonus | SBTi Validation Engine |
| Target is 1.5C aligned | Bonus | Ambition configuration |
| Covers Scope 1+2 and Scope 3 | Bonus | Scope-specific targets |
| Progress reported with detail | Points | Annual Review Engine |
| On track to meet target | Bonus | Quarterly Monitoring + trend |
| LMDI variance explanation | Bonus | Variance Analysis Engine |
| Corrective action plan (if off-track) | Recovery | Corrective Action Engine |

### Maximizing CDP Score with PACK-029

1. **Set SBTi-aligned targets** (1.5C preferred) -- earns Leadership points
2. **Cover all scopes** including material Scope 3 -- earns Management points
3. **Report progress with specifics** -- percentage achieved, absolute numbers
4. **Explain variances** using LMDI decomposition -- demonstrates sophistication
5. **Show corrective actions** when off-track -- demonstrates active management
6. **Include interim milestones** -- shows planned pathway

---

## PACK-029 CDP Export Mapping

### CDP Export Template

The CDP Export Template (`cdp_export_template.py`) maps PACK-029 engine outputs to CDP fields:

```python
class CDPExporter:
    """Export PACK-029 data to CDP-compatible formats."""

    def export(
        self,
        interim_targets: InterimTargetResult,
        annual_review: AnnualReviewResult,
        variance_analysis: VarianceAnalysisResult,
        corrective_actions: Optional[CorrectiveActionResult] = None,
        reporting_year: int = 2024,
    ) -> CDPExportPackage:
        """Generate complete CDP C4 export package."""

        package = CDPExportPackage()

        # C4.1 -- Target existence
        package.c4_1 = "Yes"

        # C4.1a -- Absolute target details
        for scope_timeline in interim_targets.scope_timelines:
            package.c4_1a.append(self._map_absolute_target(
                scope_timeline, interim_targets, reporting_year
            ))

        # C4.1c -- Net-zero target
        package.c4_1c = self._map_net_zero_target(interim_targets)

        # C4.2a -- Progress details
        for target in package.c4_1a:
            progress = self._calculate_progress(target, annual_review)
            narrative = self._generate_narrative(
                progress, variance_analysis, corrective_actions
            )
            package.c4_2a.append({
                "target_reference": target["target_reference"],
                "pct_achieved": progress.pct_achieved,
                "status": progress.status,
                "explanation": narrative,
            })

        return package
```

### Field Mapping Table

| CDP Field ID | CDP Field Name | PACK-029 Source Field |
|-------------|---------------|---------------------|
| C4.1a.col1 | Target reference number | Auto-generated "Abs {n}" |
| C4.1a.col2 | Year target was set | `datetime.now().year` |
| C4.1a.col3 | Target coverage | "Company-wide" (default) |
| C4.1a.col4 | Scope(s) | `scope_timeline.scope_type` |
| C4.1a.col5 | Scope 2 accounting method | `config.scope_2_method` |
| C4.1a.col6 | Base year | `interim_target.base_year` |
| C4.1a.col7 | Base year Scope 1 emissions | `baseline.scope_1_tco2e` |
| C4.1a.col8 | Base year Scope 2 emissions | `baseline.scope_2_tco2e` |
| C4.1a.col9 | Total base year emissions | Calculated |
| C4.1a.col10 | Target year | `scope_timeline.target_year` |
| C4.1a.col11 | Targeted reduction (%) | `scope_timeline.reduction_pct` |
| C4.1a.col12 | Is this a science-based target? | `sbti_validation.overall_pass` |
| C4.1a.col13 | Target status | From Annual Review |
| C4.1a.col14 | % of target achieved | Calculated from Annual Review |
| C4.1a.col15 | Plan for achieving target | From Corrective Action Engine |

---

## Data Quality Requirements

### CDP Data Quality Expectations

| Quality Dimension | Requirement | PACK-029 Assurance |
|-------------------|-------------|---------------------|
| Completeness | All scopes reported | Scope coverage validation |
| Accuracy | Within 5% of verified figures | Decimal arithmetic, cross-validation |
| Consistency | Year-over-year comparable | Base year recalculation policy |
| Timeliness | Current reporting year data | Annual Review Engine scheduling |
| Transparency | Methodology disclosed | Provenance hashing, audit trail |
| Verifiability | Third-party assurance | Assurance-ready output format |

### Data Validation Before Export

```python
def _validate_export_data(self, data: CDPExportPackage) -> list[ValidationWarning]:
    """Validate data quality before CDP export."""
    warnings = []

    # Check completeness
    if not data.c4_1a:
        warnings.append(ValidationWarning("No absolute targets found for C4.1a"))

    # Check consistency
    for target in data.c4_1a:
        if target["base_year_emissions"]["total"] == 0:
            warnings.append(ValidationWarning(
                f"Target {target['target_reference']}: base year emissions are zero"
            ))

        if target["pct_achieved_to_date"] > 100:
            warnings.append(ValidationWarning(
                f"Target {target['target_reference']}: progress exceeds 100%"
            ))

        if target["pct_achieved_to_date"] < 0:
            warnings.append(ValidationWarning(
                f"Target {target['target_reference']}: negative progress (backsliding)"
            ))

    return warnings
```

---

## Common Disclosure Errors

### Frequently Made Mistakes

| Error | Impact on Score | Prevention |
|-------|----------------|------------|
| Reporting only Scope 1+2, ignoring Scope 3 | Major deduction | PACK-029 checks materiality |
| Inconsistent base year across targets | Credibility loss | Single baseline source |
| Not disclosing off-track status | Scoring penalty | Honest RAG reporting |
| Missing plan for achieving target | Management score loss | Corrective Action Engine |
| Incorrect progress calculation | Data quality flag | Automated calculation |
| Carbon credits counted in progress | SBTi non-compliance | Carbon credits excluded |
| Mixing location and market Scope 2 | Inconsistency flag | Method locked per target |
| Not updating base year after acquisition | Comparability issue | Recalibration triggers |

### PACK-029 Error Prevention

```python
# Automatic checks before CDP export
pre_export_checks = [
    "scope_3_materiality_assessed",
    "base_year_consistent_across_targets",
    "progress_calculation_verified",
    "carbon_credits_excluded",
    "scope_2_method_consistent",
    "base_year_recalculation_current",
    "all_active_targets_included",
    "narratives_generated_for_all_targets",
]
```

---

## Best Practices for High Scores

### Leadership Score (A/A-) Requirements

1. **SBTi-validated 1.5C target** with near-term and long-term components
2. **Complete scope coverage** including material Scope 3 categories
3. **Detailed progress reporting** with quantified actions
4. **On-track performance** or credible corrective action plan
5. **LMDI variance analysis** explaining drivers of change
6. **Board-level governance** of climate targets
7. **Consistent year-over-year reporting** with base year updates
8. **Third-party verification** of emissions data

### PACK-029 Best Practice Workflow

```
1. Run Interim Target Setting Workflow (once)
   -> Generates SBTi-aligned targets for all scopes

2. Run SBTi Validation (once)
   -> Ensures 21/21 criteria pass

3. Run Quarterly Monitoring (4x per year)
   -> Tracks progress and generates alerts

4. Run Annual Progress Review (annually)
   -> Computes year-over-year comparison and budget status

5. Run Variance Investigation (annually or when off-track)
   -> LMDI decomposition for CDP narrative

6. Run Corrective Action Planning (if off-track)
   -> Generates action plan for CDP disclosure

7. Run Annual Reporting Workflow (annually, before CDP deadline)
   -> Exports CDP C4.1/C4.2 data package
```

### CDP Submission Timeline

| Month | Activity | PACK-029 Workflow |
|-------|----------|-------------------|
| January | Finalize previous year emissions | MRV Bridge data collection |
| February | Run annual review | Annual Progress Review |
| March | Run variance analysis | Variance Investigation |
| April | Generate CDP export | Annual Reporting Workflow |
| May-July | Submit CDP response | SBTi Portal + CDP Bridge |
| September | Receive CDP score | Review and plan improvements |
| October-December | Plan next year improvements | Corrective Action Planning |

---

## References

- CDP Climate Change 2025 Questionnaire: https://www.cdp.net/en/guidance/guidance-for-companies
- CDP Scoring Methodology: https://www.cdp.net/en/scores/cdp-scores-explained
- CDP Technical Note on Targets: https://www.cdp.net/en/guidance
- CDP-SBTi Partnership: https://www.cdp.net/en/climate/science-based-targets

---

**End of CDP Disclosure Guide**
