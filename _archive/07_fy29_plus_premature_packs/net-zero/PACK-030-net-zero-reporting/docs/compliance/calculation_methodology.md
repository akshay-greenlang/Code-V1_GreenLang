# PACK-030: Calculation Methodology Documentation

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [Overview](#overview)
2. [GHG Accounting Methodology](#ghg-accounting-methodology)
3. [Data Aggregation Methodology](#data-aggregation-methodology)
4. [Reconciliation Methodology](#reconciliation-methodology)
5. [Completeness Scoring](#completeness-scoring)
6. [Consistency Scoring](#consistency-scoring)
7. [Framework Mapping Methodology](#framework-mapping-methodology)
8. [Narrative Generation Methodology](#narrative-generation-methodology)
9. [XBRL Tagging Methodology](#xbrl-tagging-methodology)
10. [Quality Scoring Methodology](#quality-scoring-methodology)
11. [Provenance Methodology](#provenance-methodology)

---

## 1. Overview

PACK-030 does not perform primary GHG calculations. Instead, it aggregates pre-calculated data from upstream systems (GL-GHG-APP, PACK-021, PACK-022, PACK-028, PACK-029, GL-SBTi-APP, GL-CDP-APP, GL-TCFD-APP) and transforms it into framework-compliant report outputs. All calculations within PACK-030 are limited to:

- Data reconciliation and gap detection
- Completeness and quality scoring
- Cross-framework consistency validation
- Narrative consistency scoring
- Framework metric mapping and translation
- XBRL taxonomy tag assignment

### Zero-Hallucination Principles

| Principle | Implementation |
|-----------|---------------|
| No LLM in calculation path | All quantitative operations use deterministic code |
| Decimal arithmetic | Python `Decimal` type for all numeric operations |
| Provenance tracking | SHA-256 hash on every output |
| Source attribution | Every number traced to source system |
| Human review | Narrative content flagged for review before publication |

---

## 2. GHG Accounting Methodology

### Standards Alignment

PACK-030 reports data aligned with:

| Standard | Version | Usage |
|----------|---------|-------|
| GHG Protocol Corporate Accounting and Reporting Standard | Revised (2015) | Scope 1, 2 accounting |
| GHG Protocol Corporate Value Chain (Scope 3) Standard | 2011 | Scope 3 accounting |
| GHG Protocol Technical Guidance for Calculating Scope 2 | 2015 | Dual reporting (location/market) |
| IPCC Guidelines for National GHG Inventories | 2006/2019 Refinement | Emission factor methodology |

### Emission Factor Sources

PACK-030 documents all emission factor sources used by upstream systems:

| Source | Publisher | Coverage | Update Frequency |
|--------|-----------|----------|-----------------|
| DEFRA GHG Conversion Factors | UK Government | UK-specific factors | Annual |
| EPA Emission Factors | US EPA | US-specific factors | Annual |
| IPCC Emission Factor Database | IPCC | Global default factors | Periodic |
| IEA Emission Factors | IEA | Grid electricity factors | Annual |
| ecoinvent | ecoinvent Association | Life cycle factors | Annual |

### Global Warming Potentials

| GHG | AR5 GWP (100-year) | AR6 GWP (100-year) |
|-----|-------------------|-------------------|
| CO2 | 1 | 1 |
| CH4 | 28 | 27.9 |
| N2O | 265 | 273 |
| HFC-134a | 1,300 | 1,526 |
| SF6 | 23,500 | 25,200 |

---

## 3. Data Aggregation Methodology

### Multi-Source Aggregation

```
For each required metric M:
  1. Collect M from all available source systems
  2. If only one source provides M: use that value
  3. If multiple sources provide M: apply reconciliation rules
  4. If no source provides M: flag as data gap
  5. Record lineage for M (source -> transformation -> report)
  6. Calculate provenance hash for M
```

### Aggregation Priority Rules

When the same metric is available from multiple sources, the priority order is:

1. **GL-GHG-APP** (latest calculated GHG inventory - highest accuracy)
2. **PACK-029** (interim targets pack - includes progress monitoring data)
3. **GL-SBTi-APP** (SBTi-validated data - highest authority for targets)
4. **PACK-021** (baseline data - definitive for base year)
5. **PACK-022** (initiative data - for reduction metrics)
6. **PACK-028** (sector data - for pathway/benchmark metrics)
7. **GL-CDP-APP** (historical CDP responses - reference only)
8. **GL-TCFD-APP** (scenario data - for qualitative assessments)

---

## 4. Reconciliation Methodology

### Reconciliation Process

```
For each metric M available from multiple sources (S1, S2, ...):
  1. Compare values: M_S1 vs M_S2
  2. If |M_S1 - M_S2| / max(M_S1, M_S2) < threshold: MATCH
  3. If mismatch: Apply priority rules to select authoritative value
  4. Log reconciliation decision with justification
  5. Flag for manual review if difference > 5%
```

### Reconciliation Thresholds

| Metric Type | Auto-Accept Threshold | Flag for Review | Reject |
|------------|----------------------|-----------------|--------|
| Scope 1 total | < 1% difference | 1-5% difference | > 5% difference |
| Scope 2 total | < 1% difference | 1-5% difference | > 5% difference |
| Scope 3 total | < 3% difference | 3-10% difference | > 10% difference |
| Base year emissions | < 0.1% difference | 0.1-1% difference | > 1% difference |
| Reduction targets | Exact match required | Any difference | N/A |

---

## 5. Completeness Scoring

### Scoring Formula

```
Completeness Score = (Metrics Provided / Metrics Required) * 100

Where:
  Metrics Required = Count of mandatory metrics for the selected framework
  Metrics Provided = Count of mandatory metrics with valid values
```

### Per-Framework Required Metrics

| Framework | Required Metrics | Optional Metrics | Total |
|-----------|-----------------|-----------------|-------|
| SBTi | 20 | 5 | 25 |
| CDP | 250 | 50 | 300 |
| TCFD | 11 | 8 | 19 |
| GRI | 25 | 10 | 35 |
| ISSB | 20 | 15 | 35 |
| SEC | 8 | 4 | 12 |
| CSRD | 40 | 20 | 60 |

---

## 6. Consistency Scoring

### Cross-Framework Consistency

```
Consistency Score = (Consistent Metrics / Total Comparable Metrics) * 100

A metric is "consistent" if:
  - Same value reported across all frameworks that require it
  - OR difference is within acceptable rounding threshold (< 0.01%)
  - OR difference is explained by legitimate framework-specific differences
```

### Legitimate Differences

Some differences across frameworks are expected and do not reduce the consistency score:

| Difference | Example | Reason |
|-----------|---------|--------|
| Scope 2 approach | TCFD uses dual, SEC uses location-only | Framework-specific requirements |
| Scope 3 coverage | CDP requires all 15, SBTi requires 67% | Different coverage thresholds |
| Reporting period | SEC uses fiscal year, CDP uses calendar year | Different period definitions |
| Currency | SEC uses USD, CSRD uses EUR | Jurisdiction-specific currency |
| Metric precision | GRI reports whole numbers, SEC reports to 1 decimal | Format conventions |

---

## 7. Framework Mapping Methodology

### Mapping Types

| Type | Description | Confidence | Example |
|------|-------------|-----------|---------|
| **Direct** | Same metric, same definition | 100% | Scope 1 total across frameworks |
| **Calculated** | Derivable with known formula | 95%+ | Intensity = Total / Revenue |
| **Approximate** | Similar but not identical | 80-94% | Different boundary definitions |
| **Manual** | Requires human interpretation | <80% | Qualitative narrative mappings |

### Mapping Confidence Scores

```
Confidence Score = f(mapping_type, data_quality, definition_alignment)

Where:
  mapping_type_weight: direct=1.0, calculated=0.95, approximate=0.85, manual=0.70
  data_quality_weight: tier1=1.0, tier2=0.95, tier3=0.85
  definition_alignment: exact=1.0, similar=0.90, different=0.75

  Confidence = mapping_type_weight * data_quality_weight * definition_alignment * 100
```

---

## 8. Narrative Generation Methodology

### Process

1. **Template selection**: Select framework-appropriate section template
2. **Data insertion**: Insert quantitative data from aggregated dataset
3. **Contextual expansion**: Generate surrounding narrative context
4. **Citation linking**: Link every quantitative claim to source
5. **Consistency check**: Validate against existing narratives
6. **Quality scoring**: Score narrative on completeness, clarity, citation density

### Quality Criteria

| Criterion | Weight | Measurement |
|-----------|--------|-------------|
| Completeness | 30% | All required topics covered |
| Citation density | 25% | Quantitative claims with sources |
| Consistency | 25% | No contradictions across frameworks |
| Clarity | 10% | Readability score |
| Accuracy | 10% | Numbers match source data |

---

## 9. XBRL Tagging Methodology

### Tag Assignment Process

1. **Metric identification**: Match report metric to XBRL element
2. **Context creation**: Create filing context (entity, period)
3. **Unit assignment**: Assign appropriate unit (tCO2e, USD, etc.)
4. **Precision**: Set decimals attribute based on metric type
5. **Validation**: Validate tag against official taxonomy

### Taxonomy Versions

| Framework | Taxonomy | Version | Elements |
|-----------|----------|---------|----------|
| SEC | SEC Climate Disclosure | 2024 | ~50 |
| CSRD | ESRS E1 Digital Taxonomy | 2024 | ~80 |
| ISSB | IFRS S2 Taxonomy | 2023 | ~40 |

---

## 10. Quality Scoring Methodology

### Overall Report Quality Score

```
Quality Score = weighted_average(
    schema_compliance * 0.30,
    completeness * 0.25,
    consistency * 0.25,
    narrative_quality * 0.10,
    provenance_coverage * 0.10
)
```

### Score Interpretation

| Score Range | Interpretation | Action |
|------------|----------------|--------|
| 95-100% | Excellent | Ready for publication |
| 85-94% | Good | Minor improvements recommended |
| 70-84% | Acceptable | Review recommended before publication |
| 50-69% | Needs work | Significant gaps to address |
| < 50% | Insufficient | Major data or content issues |

---

## 11. Provenance Methodology

### SHA-256 Provenance Hash

Every output metric receives a SHA-256 hash computed from:

```python
provenance_input = {
    "metric_name": "scope1_total_tco2e",
    "metric_value": str(Decimal("45000.00")),
    "source_system": "GL-GHG-APP",
    "source_record_ids": ["uuid1", "uuid2", ...],
    "calculation_method": "GHG Protocol Tier 2",
    "emission_factors": {"natural_gas": "0.18293", ...},
    "timestamp": "2026-03-20T10:00:00Z",
    "engine_version": "1.0.0",
}

canonical = json.dumps(provenance_input, sort_keys=True)
provenance_hash = hashlib.sha256(canonical.encode()).hexdigest()
```

### Hash Chain Integrity

Report-level provenance is computed as a hash of all metric provenance hashes:

```python
metric_hashes = [m.provenance_hash for m in report.metrics]
metric_hashes.sort()
report_hash = hashlib.sha256(
    ":".join(metric_hashes).encode()
).hexdigest()
```

This creates a Merkle-like hash chain where any modification to any metric invalidates the report-level hash, ensuring tamper detection.

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
