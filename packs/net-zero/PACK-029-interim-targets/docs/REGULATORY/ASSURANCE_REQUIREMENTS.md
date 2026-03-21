# Regulatory Compliance Guide: Assurance Requirements

**Pack:** PACK-029 Interim Targets Pack
**Version:** 1.0.0
**Standards:** ISO 14064-3:2019, ISAE 3000/3410, AA1000AS

---

## Table of Contents

1. [Overview](#overview)
2. [Assurance Standards Landscape](#assurance-standards-landscape)
3. [ISO 14064-3:2019 Requirements](#iso-14064-32019-requirements)
4. [ISAE 3410 Requirements](#isae-3410-requirements)
5. [Assurance Levels](#assurance-levels)
6. [Evidence Hierarchy](#evidence-hierarchy)
7. [PACK-029 Assurance Support Features](#pack-029-assurance-support-features)
8. [Audit Trail and Provenance](#audit-trail-and-provenance)
9. [Data Quality Assessment](#data-quality-assessment)
10. [Common Assurance Findings](#common-assurance-findings)
11. [Remediation Guidance](#remediation-guidance)
12. [Preparing for Assurance Engagement](#preparing-for-assurance-engagement)

---

## Overview

As climate disclosures become mandatory in multiple jurisdictions, third-party assurance of GHG emissions data and targets is increasingly required. PACK-029 is designed to be "assurance-ready" from the ground up, providing the evidence, audit trails, and data quality documentation that assurance providers need.

### Why Assurance Matters

| Driver | Requirement | Timeline |
|--------|-------------|----------|
| EU CSRD | Limited assurance on sustainability data | From 2024 (phased) |
| EU CSRD | Reasonable assurance on sustainability data | From 2028 (planned) |
| SEC Climate Rule | Attestation of Scope 1+2 emissions | From 2026 (for LAFs) |
| SBTi | Third-party validation of targets | At submission |
| CDP | Verification improves scoring | Annual |
| ISSB/IFRS S2 | Assurance expected by adopting jurisdictions | Varies |

### Assurance Scope for Interim Targets

| Subject Matter | Assurance Needed | PACK-029 Support |
|---------------|------------------|------------------|
| Base year emissions | Yes (foundational) | MRV Bridge data + provenance |
| Target calculation methodology | Yes (credibility) | Calculation docs + unit tests |
| Annual progress data | Yes (accuracy) | Annual Review Engine + audit trail |
| Variance analysis | Helpful (transparency) | LMDI decomposition + provenance |
| Corrective action plans | Limited (forward-looking) | Portfolio optimization docs |
| SBTi validation results | Yes (SBTi requires) | 21-criteria validation report |

---

## Assurance Standards Landscape

### Applicable Standards

| Standard | Issuer | Scope | PACK-029 Alignment |
|----------|--------|-------|---------------------|
| ISO 14064-3:2019 | ISO | GHG assertion validation/verification | Full |
| ISAE 3000 (Revised) | IAASB | Assurance on non-financial information | Full |
| ISAE 3410 | IAASB | Assurance on GHG statements | Full |
| AA1000AS v3 | AccountAbility | Stakeholder-centric sustainability assurance | Partial |
| PCAF Standard | PCAF | Financed emissions verification | Via PACK-028 |
| SBTi Target Validation Protocol | SBTi | Science-based target validation | Full |

### Standard Hierarchy

```
Broadest Scope
    |
    +-- ISAE 3000 (Revised)
    |     General assurance standard for non-financial information
    |     |
    |     +-- ISAE 3410
    |           GHG-specific assurance standard
    |           |
    |           +-- ISO 14064-3:2019
    |                 GHG verification/validation standard
    |
    +-- AA1000AS v3
    |     Stakeholder-centric assurance
    |
    +-- SBTi Target Validation Protocol
          Science-based target-specific validation
```

---

## ISO 14064-3:2019 Requirements

### Scope

ISO 14064-3 specifies principles and requirements for verifying GHG statements and validating GHG claims. It applies to both historical emissions (verification) and forward-looking targets (validation).

### Key Requirements

| Requirement | Description | PACK-029 Support |
|-------------|-------------|------------------|
| Materiality threshold | Define material misstatement level | Configurable (default 5%) |
| Organizational boundary | Clear scope definition | Operational control documentation |
| Quantification methodology | Documented calculation methods | CALCULATIONS/ guides |
| Emission factors | Traceable, current EFs | MRV Bridge (Agent-MRV EFs) |
| Data management | Controls over data collection | Audit trail + provenance hash |
| Uncertainty assessment | Quantified uncertainty ranges | Confidence intervals |
| Competence | Qualified verification team | N/A (external requirement) |
| Evidence | Sufficient, appropriate evidence | Evidence hierarchy (below) |

### Verification vs. Validation

| Activity | Subject | Time Orientation | PACK-029 Coverage |
|----------|---------|------------------|-------------------|
| Verification | Historical GHG assertion | Past | Annual Review, emissions data |
| Validation | GHG target/projection | Future | Interim targets, forecasts |

### ISO 14064-3 Verification Process

```
Step 1: Engagement Planning
    - Define scope, materiality, objectives
    - PACK-029 provides: scope definition, methodology docs

Step 2: Strategic Analysis
    - Understand the organization and its GHG context
    - PACK-029 provides: baseline data, sector info, target architecture

Step 3: Risk Assessment
    - Identify risks of material misstatement
    - PACK-029 provides: data quality scores, uncertainty flags

Step 4: Evidence Gathering
    - Test data, calculations, and controls
    - PACK-029 provides: audit trail, provenance hashes, raw data access

Step 5: Evaluation of Evidence
    - Assess sufficiency and appropriateness
    - PACK-029 provides: cross-validation results, accuracy metrics

Step 6: Verification Statement
    - Issue opinion (reasonable or limited assurance)
    - PACK-029 provides: assurance-ready data package
```

---

## ISAE 3410 Requirements

### Scope

ISAE 3410 (Assurance Engagements on Greenhouse Gas Statements) is the IAASB standard specifically designed for GHG assurance engagements.

### Key Requirements

| Requirement | Description | PACK-029 Support |
|-------------|-------------|------------------|
| GHG statement | Complete emissions disclosure | Annual Progress Report |
| Quantification criteria | GHG Protocol, ISO 14064-1 | Methodology documentation |
| Entity-level controls | Data governance and controls | Audit trail, provenance |
| Emissions data testing | Sample testing of calculations | Unit tests (1,342 tests) |
| Analytical procedures | Trend analysis, reasonableness | Variance Analysis Engine |
| Materiality assessment | Quantitative thresholds | Configurable materiality |
| Management representations | Entity responsibility | Template provided |

### ISAE 3410 Procedures Supported by PACK-029

```
Procedure                         PACK-029 Evidence
---------                         -----------------
Test organizational boundary      Scope configuration + coverage %
Test completeness of sources      Scope 3 screening results
Test emission factors              EF database with sources
Test activity data                 MRV Bridge raw data
Test calculations                  Provenance hash verification
Analytical review                  LMDI variance decomposition
Recalculation                      Deterministic Decimal arithmetic
Review of estimates                Uncertainty quantification
Test base year recalculations      Recalibration audit trail
Review of disclosures              Template output validation
```

---

## Assurance Levels

### Limited Assurance vs. Reasonable Assurance

| Aspect | Limited Assurance | Reasonable Assurance |
|--------|-------------------|---------------------|
| Conclusion form | "Nothing has come to our attention..." (negative) | "In our opinion..." (positive) |
| Evidence required | Moderate | Extensive |
| Procedures | Inquiry, analytical, limited testing | Detailed testing, corroboration |
| Cost | Lower | Higher |
| Confidence level | ~75% | ~95% |
| Current requirement | CSRD Phase 1, CDP | CSRD Phase 2 (2028+) |

### PACK-029 Support by Assurance Level

| Feature | Limited Assurance | Reasonable Assurance |
|---------|-------------------|---------------------|
| Audit trail | Sufficient | Sufficient |
| Provenance hashing | Sufficient | Sufficient |
| Calculation documentation | Sufficient | Sufficient |
| Unit test coverage (92.4%) | Sufficient | Sufficient |
| Cross-validation | Helpful | Essential (provided) |
| Data quality scores | Helpful | Essential (provided) |
| Raw data access | Usually not needed | Required (MRV Bridge) |
| Process documentation | Helpful | Essential (provided) |

### Transitioning from Limited to Reasonable Assurance

Organizations preparing for reasonable assurance (CSRD 2028+) should:

1. **Strengthen internal controls**: Use PACK-029 audit trail for all changes
2. **Improve data quality**: Reduce estimated data, increase measured data
3. **Document methodology**: Reference PACK-029 CALCULATIONS/ guides
4. **Automate reconciliation**: Use LMDI variance to explain all changes
5. **Establish governance**: Board oversight of climate data (SBTi C20)
6. **Engage early**: Begin reasonable assurance readiness assessment

---

## Evidence Hierarchy

### Types of Evidence (Strongest to Weakest)

| Rank | Evidence Type | Description | PACK-029 Source |
|------|-------------|-------------|-----------------|
| 1 | Direct measurement | Continuous emissions monitoring (CEMS) | MRV Bridge (where available) |
| 2 | Metered data | Utility bills, fuel receipts | MRV Bridge activity data |
| 3 | Calculated (primary) | Activity data x emission factors | MRV Agents (all 30) |
| 4 | Calculated (secondary) | Spend-based or estimated activity data | Spend categorizer (DATA-009) |
| 5 | Industry averages | Sector-average emission factors | PACK-028 sector data |
| 6 | Extrapolated | Scaled from partial data | Trend Extrapolation Engine |
| 7 | Expert judgment | Professional estimate | Manual input |

### Evidence Quality Scoring

PACK-029 tracks data quality for each emission source:

```python
class DataQualityScore(BaseModel):
    """Data quality assessment for assurance purposes."""
    source_id: str
    evidence_type: str              # From hierarchy above (1-7)
    completeness_pct: Decimal       # % of period covered by data
    accuracy_estimate: str          # "high", "medium", "low"
    temporal_representativeness: str # "current", "1-2 years old", "3+ years old"
    geographical_representativeness: str  # "site-specific", "country", "global"
    technological_representativeness: str # "process-specific", "sector-average"
    overall_score: str              # "A" (highest) to "E" (lowest)
```

### Minimum Evidence Requirements

| Scope | Minimum for Limited Assurance | Minimum for Reasonable Assurance |
|-------|-------------------------------|----------------------------------|
| Scope 1 | Rank 2-3 (metered/calculated) | Rank 1-3 (measured/metered/calculated) |
| Scope 2 | Rank 2-3 (utility data + EFs) | Rank 2 (metered utility data) |
| Scope 3 | Rank 3-5 (calculated/estimated) | Rank 3-4 (calculated from primary data) |
| Targets | Documented methodology | Documented + validated methodology |
| Progress | Calculated with audit trail | Calculated + independently verified |

---

## PACK-029 Assurance Support Features

### SHA-256 Provenance Hashing

Every PACK-029 engine output includes a SHA-256 hash of the complete calculation:

```python
def _compute_provenance_hash(self, result: InterimTargetResult) -> str:
    """Compute SHA-256 hash of all inputs and outputs."""
    hash_input = json.dumps({
        "engine": "interim_target_engine",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "inputs": result.input_data.dict(),
        "outputs": result.dict(exclude={"provenance_hash"}),
    }, sort_keys=True, default=str)

    return hashlib.sha256(hash_input.encode()).hexdigest()
```

**Assurance value:** Auditors can verify that results have not been tampered with since calculation. Any change to inputs or outputs would produce a different hash.

### Comprehensive Audit Trail

Every operation in PACK-029 is logged to the `gl_pack029_audit_trail` table:

```sql
CREATE TABLE gl_pack029_audit_trail (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id       UUID NOT NULL,
    operation       VARCHAR(100) NOT NULL,
    engine_name     VARCHAR(100),
    input_hash      VARCHAR(64),
    output_hash     VARCHAR(64),
    user_id         UUID,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    details         JSONB,
    CONSTRAINT fk_entity FOREIGN KEY (entity_id) REFERENCES gl_entities(id)
);
```

**Assurance value:** Complete history of all calculations, changes, and data flows. Auditors can trace any result back to its source data.

### Deterministic Decimal Arithmetic

All calculations use Python's `Decimal` type instead of `float`:

```python
# Every calculation is deterministic and reproducible
result = Decimal("200000") * (Decimal("1") - Decimal("0.465"))
# Always exactly 107000.000, never 106999.99999999999 or similar float error
```

**Assurance value:** Calculations are perfectly reproducible. Running the same inputs always produces the exact same outputs, facilitating independent recalculation by auditors.

### Cross-Validation Checks

PACK-029 performs internal cross-validation:

```python
# LMDI perfect decomposition check
assert abs(activity_effect + intensity_effect + structural_effect - total_change) < Decimal("1e-10")

# Carbon budget consistency check
assert abs(cumulative_budget_used + remaining_budget - total_budget) < Decimal("1e-10")

# Target pathway monotonicity check
for i in range(1, len(milestones)):
    assert milestones[i].target_emissions < milestones[i-1].target_emissions
```

**Assurance value:** Built-in integrity checks catch calculation errors before they reach the assurance stage.

---

## Data Quality Assessment

### GHG Protocol Data Quality Indicators

PACK-029 tracks the GHG Protocol's five data quality indicators:

| Indicator | Definition | Assessment Method |
|-----------|-----------|-------------------|
| Technological representativeness | How well the data reflects the actual technology | Source classification |
| Temporal representativeness | How recent the data is | Date stamps on EFs |
| Geographical representativeness | How well the data matches the location | Region tagging |
| Completeness | Whether all sources are covered | Coverage % calculation |
| Reliability | Measurement/estimation method quality | Evidence hierarchy rank |

### Quality Score Matrix

```
Score | Technological | Temporal | Geographical | Completeness | Reliability
------|---------------|----------|-------------|--------------|------------
  A   | Process-specific | Current year | Site-specific | 100% | Direct measurement
  B   | Sector-specific | 1-2 years | Country | 95-99% | Metered data
  C   | Similar process | 3-5 years | Region | 85-94% | Calculated (primary)
  D   | Generic | 5-10 years | Continent | 70-84% | Estimated (secondary)
  E   | Unknown | >10 years | Global | <70% | Expert judgment
```

### Aggregate Quality Score

```python
def _compute_aggregate_quality(self, scores: list[DataQualityScore]) -> str:
    """Compute weighted aggregate data quality score."""
    # Weight by emissions contribution
    total_emissions = sum(s.emissions for s in scores)
    weighted_score = sum(
        s.numeric_score * (s.emissions / total_emissions)
        for s in scores
    )

    if weighted_score >= 4.0: return "A"
    if weighted_score >= 3.0: return "B"
    if weighted_score >= 2.0: return "C"
    if weighted_score >= 1.0: return "D"
    return "E"
```

---

## Common Assurance Findings

### Frequent Issues Found During GHG Assurance

| Finding | Severity | Prevention in PACK-029 |
|---------|----------|------------------------|
| Incomplete Scope 3 screening | High | Scope 3 materiality check (SBTi C4) |
| Inconsistent base year | High | Single-source baseline with recalibration |
| Outdated emission factors | Medium | EF version tracking in MRV Bridge |
| Missing activity data for some months | Medium | Data completeness check |
| Calculation errors (float rounding) | Medium | Decimal arithmetic eliminates this |
| No audit trail for changes | High | gl_pack029_audit_trail table |
| Inconsistent organizational boundary | High | Scope configuration documentation |
| Double counting across scopes | Medium | Dual reporting reconciliation (MRV-013) |
| Methodology not documented | Medium | CALCULATIONS/ guides |
| Target recalculation not performed after acquisition | High | Recalibration triggers (5% threshold) |

### PACK-029 Preventive Controls

```
Control ID | Control Description                    | Frequency
-----------|---------------------------------------|----------
PC-001     | Provenance hash on all outputs         | Every calculation
PC-002     | Audit trail logging                    | Every operation
PC-003     | Perfect decomposition assertion        | Every LMDI run
PC-004     | Monotonic pathway validation           | Every target set
PC-005     | Scope coverage threshold check         | Every target set
PC-006     | Base year recalibration trigger check   | Every material change
PC-007     | Data completeness scoring              | Every data import
PC-008     | Cross-validation of totals             | Every aggregation
PC-009     | SBTi 21-criteria validation             | Every submission
PC-010     | Carbon budget balance assertion         | Every budget update
```

---

## Remediation Guidance

### When Assurance Findings Arise

| Finding Category | Remediation Steps | PACK-029 Tools |
|-----------------|-------------------|----------------|
| Data gaps | Fill missing periods, re-run calculations | Time Series Gap Filler (DATA-014) |
| Calculation errors | Correct methodology, re-calculate | Engine re-run with corrected inputs |
| Incomplete scope | Add missing sources, update boundary | MRV Bridge configuration |
| Outdated EFs | Update emission factors, recalculate | MRV Agent EF updates |
| Missing documentation | Generate from PACK-029 outputs | CALCULATIONS/ + API_REFERENCE |
| Base year issue | Run Target Recalibration Engine | Recalibration workflow |
| Control weakness | Enable additional audit trail fields | Audit trail configuration |
| Inconsistent reporting | Reconcile across frameworks | CDP/TCFD/SBTi bridge outputs |

### Remediation Workflow

```
1. Receive assurance finding
2. Classify severity (high/medium/low)
3. Identify root cause
4. Apply correction:
   a. Data issue -> Update source data, re-run engines
   b. Methodology issue -> Update configuration, re-run engines
   c. Documentation issue -> Generate from PACK-029 templates
   d. Control issue -> Enable additional PACK-029 controls
5. Re-run affected engines
6. Verify provenance hash changed (new calculation)
7. Document remediation in audit trail
8. Submit corrected data for re-assurance
```

---

## Preparing for Assurance Engagement

### Pre-Engagement Checklist

- [ ] Run full Health Check (20 categories, score >= 90/100)
- [ ] Verify all engine outputs have provenance hashes
- [ ] Confirm audit trail has no gaps
- [ ] Run SBTi Validation Engine (21/21 criteria pass)
- [ ] Run LMDI perfect decomposition verification
- [ ] Generate data quality assessment report
- [ ] Prepare base year documentation package
- [ ] Document all methodology choices (pathway shape, ambition, etc.)
- [ ] Reconcile Scope 2 dual reporting (location vs. market)
- [ ] Verify target recalibration triggers have been assessed
- [ ] Prepare management representation letter
- [ ] Compile emission factor sources and versions
- [ ] Generate complete CALCULATIONS/ guide set
- [ ] Prepare CDP and TCFD export for cross-reference

### Documentation Package for Auditors

| Document | Source | Purpose |
|----------|--------|---------|
| Interim Target Configuration | pack.yaml + presets | Methodology choices |
| Calculation Guides (4 docs) | docs/CALCULATIONS/ | Detailed formulas |
| SBTi Validation Report | SBTi Validation Engine | 21-criteria compliance |
| Annual Progress Report | Annual Review Engine | Performance data |
| Variance Analysis Report | Variance Analysis Engine | Change decomposition |
| Data Quality Assessment | Data quality scoring | Evidence classification |
| Audit Trail Extract | gl_pack029_audit_trail | Complete change history |
| Provenance Hash Registry | All engine outputs | Tamper detection |
| Base Year Documentation | Baseline configuration | Reference point |
| Recalibration Log | Recalibration Engine | Any base year changes |

### Recommended Assurance Engagement Timeline

| Month | Activity |
|-------|----------|
| January | Finalize reporting year data |
| February | Run Annual Review + Variance Analysis |
| March | Generate all reports and documentation |
| April | Internal review of assurance readiness |
| May | Engage assurance provider |
| June-July | Assurance fieldwork |
| August | Receive draft assurance report |
| September | Address any findings |
| October | Final assurance statement issued |
| November | Publish assured disclosure |

---

## References

- ISO 14064-3:2019: https://www.iso.org/standard/66455.html
- ISAE 3000 (Revised): https://www.iaasb.org/publications/isae-3000-revised
- ISAE 3410: https://www.iaasb.org/publications/isae-3410
- AA1000 Assurance Standard v3: https://www.accountability.org/standards/
- GHG Protocol Data Quality Guidelines: https://ghgprotocol.org/
- CSRD Assurance Requirements: https://eur-lex.europa.eu/

---

**End of Assurance Requirements Guide**
