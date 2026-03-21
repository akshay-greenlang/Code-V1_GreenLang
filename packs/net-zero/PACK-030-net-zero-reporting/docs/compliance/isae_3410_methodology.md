# PACK-030: ISAE 3410 Assurance Methodology

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [Overview](#overview)
2. [ISAE 3410 Requirements](#isae-3410-requirements)
3. [Evidence Bundle Components](#evidence-bundle-components)
4. [Control Matrix](#control-matrix)
5. [Provenance Tracking](#provenance-tracking)
6. [Data Lineage Documentation](#data-lineage-documentation)
7. [Calculation Methodology Documentation](#calculation-methodology-documentation)
8. [Audit Trail](#audit-trail)
9. [Assurance Engagement Support](#assurance-engagement-support)
10. [ISAE 3000 Alignment](#isae-3000-alignment)

---

## 1. Overview

PACK-030 automates the preparation of assurance evidence packages aligned with ISAE 3410 (Assurance Engagements on Greenhouse Gas Statements) and ISAE 3000 (Revised) (Assurance Engagements Other than Audits or Reviews of Historical Financial Information).

### Assurance Levels Supported

| Level | Standard | Description |
|-------|----------|-------------|
| **Limited Assurance** | ISAE 3410 | Negative form conclusion ("nothing has come to our attention...") |
| **Reasonable Assurance** | ISAE 3410 | Positive form conclusion ("in our opinion, the GHG statement is...") |
| **Limited Assurance** | ISAE 3000 | For non-GHG climate disclosures (TCFD narratives, targets) |

### What PACK-030 Provides for Auditors

- Automated evidence bundle generation (ZIP file with structured contents)
- SHA-256 cryptographic provenance on every calculation
- Visual data lineage diagrams (source-to-report flow)
- Detailed calculation methodology documentation
- ISAE 3410 control matrix with requirement mapping
- Immutable audit trail of all data access and modifications
- Cross-framework consistency validation reports

---

## 2. ISAE 3410 Requirements

### Key Requirements Addressed

| ISAE 3410 Para | Requirement | PACK-030 Evidence |
|----------------|-------------|-------------------|
| 14 | Understanding the entity's methods for preparing GHG statement | Methodology documentation in evidence bundle |
| 17 | Evaluating the suitability of quantification methods | Emission factor sources, calculation method documentation |
| 19 | Evaluating the suitability of the reporting criteria | Framework schema validation results |
| 23 | Understanding the entity's information system | Data lineage diagrams, system architecture docs |
| 31 | Analytical procedures (limited assurance) | Cross-framework consistency checks, trend analysis |
| 37 | Tests of details (reasonable assurance) | Provenance hashes, source data verification |
| 44 | Evaluating the sufficiency and appropriateness of evidence | Evidence completeness checklist |
| 51 | Forming the assurance conclusion | Validation results, quality scores |

### GHG Statement Components

ISAE 3410 requires assurance over the following components, all of which PACK-030 generates with provenance:

| Component | PACK-030 Source | Provenance |
|-----------|----------------|-----------|
| Scope 1 emissions | GL-GHG-APP via Data Aggregation Engine | SHA-256 per calculation |
| Scope 2 emissions (location) | GL-GHG-APP via Data Aggregation Engine | SHA-256 per calculation |
| Scope 2 emissions (market) | GL-GHG-APP via Data Aggregation Engine | SHA-256 per calculation |
| Scope 3 emissions | GL-GHG-APP via Data Aggregation Engine | SHA-256 per calculation |
| Emission factors used | GL-GHG-APP emission factor database | Versioned, source-cited |
| Quantification methodology | Calculation Methodology docs | Per-engine documentation |
| Organizational boundary | PACK-021 configuration | Boundary definition record |
| Base year recalculation | PACK-029 recalibration engine | Recalculation audit log |

---

## 3. Evidence Bundle Components

### Bundle Structure

```
evidence_bundle_{org}_{year}.zip
|
+-- manifest.json
|   Description: Bundle contents, checksums, generation metadata
|
+-- 01_provenance/
|   +-- calculation_hashes.json        # SHA-256 for every metric calculation
|   +-- report_hashes.json             # SHA-256 for every generated report
|   +-- hash_verification_script.py    # Script to independently verify hashes
|
+-- 02_data_lineage/
|   +-- scope1_lineage.svg             # Visual lineage: source -> calc -> report
|   +-- scope2_lineage.svg
|   +-- scope3_lineage.svg
|   +-- full_lineage.json              # Machine-readable lineage graph
|   +-- lineage_summary.pdf            # Human-readable lineage summary
|
+-- 03_methodology/
|   +-- ghg_accounting_methodology.pdf # GHG Protocol alignment documentation
|   +-- emission_factor_sources.pdf    # All emission factor sources and versions
|   +-- calculation_methods.pdf        # Engine-by-engine calculation documentation
|   +-- scope_boundary_definition.pdf  # Organizational boundary documentation
|   +-- exclusions_justification.pdf   # Justification for any scope exclusions
|
+-- 04_controls/
|   +-- isae_3410_control_matrix.xlsx  # Control requirements mapped to PACK-030
|   +-- access_control_evidence.json   # RBAC configuration, RLS policies
|   +-- data_validation_rules.json     # All validation rules applied
|   +-- change_management_log.json     # Configuration changes during period
|
+-- 05_validation/
|   +-- schema_validation_results.json # Framework schema compliance
|   +-- completeness_report.json       # Data completeness per framework
|   +-- consistency_report.json        # Cross-framework consistency checks
|   +-- reconciliation_report.json     # Source data reconciliation results
|
+-- 06_audit_trail/
|   +-- report_lifecycle_log.json      # All report status changes
|   +-- data_access_log.json           # All data access events
|   +-- user_edit_log.json             # All manual narrative edits
|   +-- approval_records.json          # Signoff records with timestamps
|
+-- 07_source_data_summary/
    +-- data_sources_inventory.json    # List of all source systems used
    +-- data_freshness_report.json     # Age of source data at time of reporting
    +-- data_quality_scores.json       # Quality scores per data source
```

### Manifest Format

```json
{
  "bundle_id": "uuid",
  "organization_id": "uuid",
  "reporting_period": {"start": "2025-01-01", "end": "2025-12-31"},
  "generated_at": "2026-03-20T10:00:00Z",
  "generated_by": "PACK-030 v1.0.0",
  "frameworks_covered": ["SBTi", "CDP", "TCFD", "GRI", "ISSB", "SEC", "CSRD"],
  "file_count": 25,
  "total_size_bytes": 5242880,
  "bundle_checksum": "sha256:abc123...",
  "files": [
    {
      "path": "01_provenance/calculation_hashes.json",
      "checksum": "sha256:def456...",
      "size_bytes": 102400
    }
  ]
}
```

---

## 4. Control Matrix

The ISAE 3410 control matrix maps PACK-030 controls to ISAE 3410 requirements:

| Control Objective | ISAE 3410 Para | PACK-030 Control | Evidence |
|------------------|----------------|------------------|----------|
| Data completeness | 23-25 | Completeness scoring engine validates all required fields | `completeness_report.json` |
| Data accuracy | 31-37 | SHA-256 provenance on all calculations; Decimal arithmetic | `calculation_hashes.json` |
| Data timeliness | 23 | Data freshness monitoring; staleness alerts | `data_freshness_report.json` |
| Boundary completeness | 17-19 | Organizational boundary validation against config | `scope_boundary_definition.pdf` |
| Methodology consistency | 14-17 | Deterministic calculation engines; zero-hallucination | `calculation_methods.pdf` |
| Access control | 23 | RBAC with 15+ permissions; RLS on 15 tables | `access_control_evidence.json` |
| Change management | 23 | Immutable audit trail; version-controlled configs | `change_management_log.json` |
| Cross-validation | 31 | Cross-framework consistency validation | `consistency_report.json` |
| Source verification | 37 | Data lineage tracing to source transaction | `full_lineage.json` |
| Emission factor validity | 14, 17 | Emission factor version tracking and source citation | `emission_factor_sources.pdf` |

---

## 5. Provenance Tracking

### SHA-256 Hash Chain

Every calculation in PACK-030 produces a SHA-256 hash that captures:

```json
{
  "metric_name": "scope1_total_tco2e",
  "metric_value": "45000.00",
  "calculation_inputs": {
    "stationary_combustion": "28000.00",
    "mobile_combustion": "12000.00",
    "process_emissions": "3000.00",
    "fugitive_emissions": "2000.00"
  },
  "emission_factors": {
    "natural_gas": {"value": "0.18293", "source": "DEFRA 2025", "version": "1.2"},
    "diesel": {"value": "0.25301", "source": "DEFRA 2025", "version": "1.2"}
  },
  "calculation_method": "GHG Protocol Corporate Standard 2015",
  "calculated_at": "2026-03-20T10:00:00Z",
  "calculated_by": "DataAggregationEngine v1.0.0",
  "provenance_hash": "sha256:7a8b9c..."
}
```

### Independent Verification

The evidence bundle includes a Python script that auditors can run to independently verify provenance hashes:

```python
# hash_verification_script.py
import hashlib, json

def verify_hash(metric_record):
    """Independently verify SHA-256 provenance hash."""
    canonical = json.dumps({
        "metric_name": metric_record["metric_name"],
        "metric_value": metric_record["metric_value"],
        "calculation_inputs": metric_record["calculation_inputs"],
        "emission_factors": metric_record["emission_factors"],
    }, sort_keys=True)
    computed_hash = hashlib.sha256(canonical.encode()).hexdigest()
    return computed_hash == metric_record["provenance_hash"]
```

---

## 6. Data Lineage Documentation

### Visual Lineage Diagrams

PACK-030 generates SVG lineage diagrams showing the complete data flow:

```
[Source System]     [Transformation]      [Report Output]

GL-GHG-APP -------> Scope 1 Calc -------> SBTi Report
  |                    |                     |
  +-- Fuel data        +-- DEFRA EFs         +-- Progress table
  +-- Fleet data       +-- GHG Protocol      +-- Variance explanation
  +-- Process data     +-- Aggregation       +-- Target comparison

PACK-021 ----------> Baseline Import -----> SBTi Report
  |                    |                     |
  +-- Base year data   +-- Validation        +-- Base year column
                       +-- Normalization
```

### Machine-Readable Lineage

```json
{
  "metric": "scope1_total_tco2e",
  "report": "SBTi Progress Report 2025",
  "lineage_chain": [
    {
      "step": 1,
      "source": "GL-GHG-APP",
      "data": "fuel_consumption_records",
      "timestamp": "2026-03-20T09:00:00Z"
    },
    {
      "step": 2,
      "transformation": "stationary_combustion_calculation",
      "engine": "AGENT-MRV-001",
      "method": "GHG Protocol Tier 2"
    },
    {
      "step": 3,
      "aggregation": "scope1_total",
      "engine": "DataAggregationEngine",
      "provenance_hash": "sha256:abc..."
    },
    {
      "step": 4,
      "output": "SBTi Progress Report",
      "section": "progress_table",
      "format": "PDF"
    }
  ]
}
```

---

## 7. Calculation Methodology Documentation

### Per-Engine Documentation

Each engine produces methodology documentation describing:

1. **Calculation approach**: GHG Protocol methodology tier (1, 2, or 3)
2. **Emission factors**: Source, version, effective date
3. **Global Warming Potentials**: Source (IPCC AR5/AR6), time horizon (100-year)
4. **Organizational boundary**: Control approach or equity share
5. **Scope boundary**: Included and excluded sources with justification
6. **Uncertainty assessment**: Qualitative assessment of data quality

### Emission Factor Traceability

```json
{
  "emission_factor_id": "DEFRA-2025-natural-gas-gross-cv",
  "factor_name": "Natural Gas (Gross CV)",
  "value": 0.18293,
  "unit": "kgCO2e/kWh",
  "source": "UK Government GHG Conversion Factors 2025",
  "publisher": "DEFRA/BEIS",
  "version": "1.2",
  "effective_date": "2025-06-01",
  "gases_included": ["CO2", "CH4", "N2O"],
  "gwp_source": "IPCC AR6",
  "gwp_time_horizon": "100-year"
}
```

---

## 8. Audit Trail

### Immutable Event Log

All events are logged to the `gl_nz_audit_trail` table with the following properties:
- **Immutable**: No UPDATE or DELETE on audit trail records
- **Timestamped**: Sub-second precision with timezone
- **Attributed**: Every event linked to user, system, or API actor
- **Contextual**: Full event details in JSONB
- **Searchable**: Indexed by report_id, event_type, timestamp

### Event Types

| Event | Trigger | Details Captured |
|-------|---------|-----------------|
| `report_created` | Report generation initiated | Config used, data sources, user |
| `report_updated` | Report content modified | Changed fields, old/new values |
| `report_approved` | Report approved for publication | Approver, approval conditions |
| `report_published` | Report made available externally | Publication channel, audience |
| `narrative_edited` | Manual narrative edit | Before/after text, editor |
| `metric_updated` | Metric value changed | Old/new value, reason |
| `evidence_added` | Evidence file added to bundle | File type, checksum |
| `validation_run` | Validation executed | Results summary |
| `access_granted` | User granted report access | Permission, role |

---

## 9. Assurance Engagement Support

### Pre-Engagement Checklist

PACK-030 generates a pre-engagement checklist for auditors:

- [ ] Evidence bundle generated and verified
- [ ] All provenance hashes independently verifiable
- [ ] Data lineage diagrams cover all material metrics
- [ ] Methodology documentation complete for all scopes
- [ ] Control matrix maps all ISAE 3410 requirements
- [ ] Audit trail covers full reporting period
- [ ] Cross-framework consistency validated
- [ ] Source data reconciliation complete
- [ ] Access control evidence current

### Auditor Access

PACK-030 provides read-only auditor access:
- **API endpoint**: `/api/v1/assurance/evidence-bundle/{bundle_id}`
- **Dashboard view**: Auditor-specific dashboard with evidence drill-down
- **Export**: Full evidence bundle as ZIP download
- **Verification**: Independent hash verification capability

---

## 10. ISAE 3000 Alignment

For non-GHG climate disclosures (TCFD narratives, targets, strategy statements), PACK-030 also supports ISAE 3000 (Revised) requirements:

| ISAE 3000 Area | PACK-030 Support |
|----------------|-----------------|
| Subject matter information | Framework-specific report content with citations |
| Suitable criteria | Official framework schemas and requirements |
| Sufficient appropriate evidence | Provenance, lineage, methodology docs |
| Practitioner's expertise | Automated calculation reduces human error |
| Quality control | Validation engine, consistency checks |
| Communication | Structured evidence bundle format |

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
