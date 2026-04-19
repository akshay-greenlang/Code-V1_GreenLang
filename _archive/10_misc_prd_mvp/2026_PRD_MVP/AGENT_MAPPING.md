# GreenLang MVP 2026 - Agent Catalog Mapping

This document maps the CBAM MVP pipeline to the GreenLang 402-agent catalog.

---

## MVP Agent Chain Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                     CBAM MVP Pipeline (7 Agents)                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────┐                                                      │
│  │   INPUTS     │                                                      │
│  │ (CSV/YAML)   │                                                      │
│  └──────┬───────┘                                                      │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────────┐     ┌──────────────────┐                        │
│  │ GL-FOUND-X-001   │     │ GL-FOUND-X-002   │                        │
│  │ Orchestrator     │────▶│ Schema Validator │                        │
│  │ (FOUNDATION)     │     │ (FOUNDATION)     │                        │
│  └──────────────────┘     └────────┬─────────┘                        │
│                                    │                                   │
│                                    ▼                                   │
│                           ┌──────────────────┐                        │
│                           │ GL-FOUND-X-003   │                        │
│                           │ Unit Normalizer  │                        │
│                           │ (FOUNDATION)     │                        │
│                           └────────┬─────────┘                        │
│                                    │                                   │
│                    ┌───────────────┴───────────────┐                  │
│                    │                               │                   │
│                    ▼                               ▼                   │
│           ┌──────────────────┐            ┌──────────────────┐        │
│           │ GL-DATA-X-010    │            │ GL-CBAM-CALC-001 │        │
│           │ Factor Library   │───────────▶│ CBAM Calculator  │        │
│           │ (DATA)           │            │ (MRV) [NEW]      │        │
│           └──────────────────┘            └────────┬─────────┘        │
│                                                    │                   │
│                                                    ▼                   │
│                                           ┌──────────────────┐        │
│                                           │ GL-CBAM-XML-001  │        │
│                                           │ XML Exporter     │        │
│                                           │ (REPORTING) [NEW]│        │
│                                           └────────┬─────────┘        │
│                                                    │                   │
│                                                    ▼                   │
│                                           ┌──────────────────┐        │
│                                           │ GL-FOUND-X-005   │        │
│                                           │ Evidence Packager│        │
│                                           │ (FOUNDATION)     │        │
│                                           └────────┬─────────┘        │
│                                                    │                   │
│                                                    ▼                   │
│                                           ┌──────────────────┐        │
│                                           │    OUTPUTS       │        │
│                                           │ (XML/Bundle)     │        │
│                                           └──────────────────┘        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Catalog Mapping Table

### Existing Agents (From 402-Agent Catalog)

| MVP Role | Agent ID | Agent Name | Layer | Status | Notes |
|----------|----------|------------|-------|--------|-------|
| Orchestrator | GL-FOUND-X-001 | GreenLang Orchestrator | Foundation & Governance | Exists | Plans/executes pipelines |
| Validator | GL-FOUND-X-002 | Schema Compiler & Validator | Foundation & Governance | Exists | Input validation |
| Normalizer | GL-FOUND-X-003 | Unit & Reference Normalizer | Foundation & Governance | Exists | Unit conversions |
| Factor Library | GL-DATA-X-010 | Emission Factor Library Agent | Data & Connectors | Exists | Factor lookup |
| Evidence | GL-FOUND-X-005 | Citations & Evidence Agent | Foundation & Governance | Exists | Audit bundle |

### New Agents (Created for MVP)

| MVP Role | Agent ID | Agent Name | Layer | Status | Notes |
|----------|----------|------------|-------|--------|-------|
| Calculator | GL-CBAM-CALC-001 | CBAM Embedded Emissions Calculator | MRV / Accounting | **New** | Core CBAM calculation |
| XML Export | GL-CBAM-XML-001 | CBAM XML Export Agent | Reporting | **New** | Registry XML generation |

---

## Detailed Agent Specifications

### 1. GL-FOUND-X-001: GreenLang Orchestrator

**From Catalog:**
```
Layer: Foundation & Governance (FOUND)
Sector: Cross-cutting (X)
What it does: Plans and executes multi-agent pipelines; manages dependency graph,
              retries, timeouts, and handoffs; enforces deterministic run metadata.
Key Inputs: Pipeline YAML, agent registry, run configuration, credentials/permissions
Key Outputs: Execution plan, run logs, step-level artifacts, status and lineage
Methods/Tools: DAG orchestration, policy checks, observability hooks
Dependencies: OPS+DATA agents, audit trail
Maturity Target: MVP
```

**MVP Adaptation:**
- Executes 7-agent CBAM pipeline
- Manages agent dependencies (validator → normalizer → calculator → exporter)
- Records run metadata for determinism
- Handles retries on transient failures

---

### 2. GL-FOUND-X-002: Schema Compiler & Validator

**From Catalog:**
```
Layer: Foundation & Governance (FOUND)
Sector: Cross-cutting (X)
What it does: Validates input payloads against GreenLang schemas; pinpoints missing
              fields, unit inconsistencies, and invalid ranges; emits machine-fixable errors.
Key Inputs: YAML/JSON inputs, schema version, validation rules
Key Outputs: Validation report, normalized payload, fix suggestions
Methods/Tools: Schema validation, rule engines, linting
Maturity Target: MVP
```

**MVP Adaptation:**
- Validates import ledger against CBAM schema
- Validates config file against config schema
- Produces actionable error messages with row/column references
- Implements fail-fast behavior

---

### 3. GL-FOUND-X-003: Unit & Reference Normalizer

**From Catalog:**
```
Layer: Foundation & Governance (FOUND)
Sector: Cross-cutting (X)
What it does: Normalizes units, converts to canonical units, standardizes naming
              for fuels, processes, materials; maintains consistent reference IDs.
Key Inputs: Raw measurements, unit metadata, reference tables
Key Outputs: Canonical measurements, conversion audit log
Methods/Tools: Unit conversion, entity resolution, controlled vocabularies
Dependencies: Schema Validator
Maturity Target: MVP
```

**MVP Adaptation:**
- Converts kg ↔ tonnes
- Standardizes CN codes (validates 8-digit format)
- Normalizes country codes to ISO 3166-1 alpha-2
- Records all conversions in audit log

---

### 4. GL-DATA-X-010: Emission Factor Library Agent

**From Catalog:**
```
Layer: Data & Connectors (DATA)
Sector: Cross-cutting (X)
What it does: Curates and versions emission factors (combustion, refrigerants,
              electricity, upstream fuels); enforces citations and validity windows.
Key Inputs: Factor sources, effective dates, jurisdiction
Key Outputs: Factor manifest, applied factor log
Methods/Tools: Lookup tables, versioning, provenance
Dependencies: Assumptions registry
Maturity Target: MVP
```

**MVP Adaptation:**
- Provides EU Commission CBAM default factors for Steel and Aluminum
- Supports country-specific electricity emission factors
- Enforces factor validity windows
- Records factor selection in assumptions log

**CBAM-Specific Factors:**
- Direct emissions by product type and country of origin
- Indirect emissions (electricity factors by country)
- Default values from EU Implementing Regulation

---

### 5. GL-CBAM-CALC-001: CBAM Embedded Emissions Calculator (NEW)

**New Agent for MVP:**
```
Layer: MRV / Accounting
Sector: Cross-cutting
What it does: Computes direct and indirect embedded emissions per CBAM import line
              using appropriate methodology (supplier-specific or default).
Key Inputs: Normalized import lines, emission factors, method selection policy
Key Outputs: Emissions results per line, method notes, assumptions
Methods/Tools: CBAM calculation methodology, factor application, uncertainty
Dependencies: Unit Normalizer, Emission Factor Library
Maturity Target: MVP
```

**Calculation Logic:**
```python
# Direct emissions
if supplier_direct_emissions provided:
    direct = supplier_direct_emissions * quantity
    method_direct = "supplier_specific"
else:
    direct = default_direct_factor[cn_code][country] * quantity
    method_direct = "default"

# Indirect emissions
if supplier_indirect_emissions provided:
    indirect = supplier_indirect_emissions * quantity
    method_indirect = "supplier_specific"
else:
    indirect = default_electricity_factor[country] * estimated_electricity_use
    method_indirect = "default"

# Total
total = direct + indirect
```

**Aggregation:**
- Aggregates by CN code + country of origin
- Weight-averages emissions intensity if multiple installations

---

### 6. GL-CBAM-XML-001: CBAM XML Export Agent (NEW)

**New Agent for MVP:**
```
Layer: Reporting (REP)
Sector: Cross-cutting
What it does: Generates EU CBAM Transitional Registry XML format from aggregated
              emissions results; validates against EU XSD schema.
Key Inputs: Aggregated emissions results, declarant metadata, reporting period
Key Outputs: CBAM Registry XML, XSD validation result
Methods/Tools: XML generation, XSD validation
Dependencies: CBAM Calculator
Maturity Target: MVP
```

**XML Structure:**
- Header (reporting period, declarant info)
- ImportedGoods (per CN code + country aggregation)
- Summary (total emissions)

**XSD Validation:**
- Validates against EU Commission provided schema
- Fails run if validation errors

---

### 7. GL-FOUND-X-005: Citations & Evidence Agent

**From Catalog:**
```
Layer: Foundation & Governance (FOUND)
Sector: Cross-cutting (X)
What it does: Attaches sources, evidence files, and calculation notes to outputs;
              creates an evidence map tying every KPI to inputs and rules.
Key Inputs: Input datasets, factor sources, calculation graph
Key Outputs: Evidence map, citations list, traceability report
Methods/Tools: Lineage tracking, document linking
Dependencies: Audit Trail Agent
Maturity Target: v1
```

**MVP Adaptation:**
- Generates claims.json (claim graph)
- Generates lineage.json (provenance graph)
- Generates assumptions.json (assumptions registry)
- Generates gap_report.json (improvement opportunities)
- Generates run_manifest.json (version pinning)
- Copies input files to evidence/ folder with hashes

---

## Agent Family Mapping

From the 402-agent catalog, the MVP agents map to these families:

| Family | Count in Catalog | MVP Agents |
|--------|------------------|------------|
| OrchestrationFamily | ~10,000 variants | GL-FOUND-X-001 |
| SchemaFamily | ~1,500 variants | GL-FOUND-X-002 |
| NormalizationFamily | ~1,800 variants | GL-FOUND-X-003 |
| FactorFamily | ~90,000 variants | GL-DATA-X-010 |
| AssuranceFamily | ~12 variants | GL-FOUND-X-005 |
| **New: CBAMFamily** | TBD | GL-CBAM-CALC-001, GL-CBAM-XML-001 |

---

## Layer Distribution

| Layer | Full Catalog | MVP Agents | Percentage |
|-------|--------------|------------|------------|
| FOUND (Foundation) | 10 | 4 | 40% |
| DATA (Connectors) | 15 | 1 | 6.7% |
| MRV (Accounting) | 93 | 1 | 1.1% |
| REP (Reporting) | 12 | 1 | 8.3% |
| **Total** | 402 | 7 | 1.7% |

The MVP uses only 7 of 402 canonical agents (1.7%), demonstrating focused scope.

---

## Future Agent Expansion

### v1.1 (Cement + Fertilizers)

| New Agent | Based On | Purpose |
|-----------|----------|---------|
| GL-CBAM-CALC-002 | GL-CBAM-CALC-001 | Cement emissions calculation |
| GL-CBAM-CALC-003 | GL-CBAM-CALC-001 | Fertilizers emissions calculation |

### v2.0 (Full CBAM)

| New Agent | Based On | Purpose |
|-----------|----------|---------|
| GL-CBAM-CALC-004 | GL-CBAM-CALC-001 | Electricity embedded emissions |
| GL-CBAM-CALC-005 | GL-CBAM-CALC-001 | Hydrogen embedded emissions |
| GL-CBAM-CERT-001 | New | Certificate tracking (definitive phase) |

---

## Implementation Notes

### Existing Agent Reuse

The following agents from GL-CBAM-APP can be refactored into the catalog agents:

| Existing Code | Maps To | Action |
|---------------|---------|--------|
| `cbam_validator.py` | GL-FOUND-X-002 | Refactor to use standard schema validator |
| `unit_converter.py` | GL-FOUND-X-003 | Refactor to use standard normalizer |
| `emission_factors.py` | GL-DATA-X-010 | Integrate with factor library |
| `xml_generator.py` | GL-CBAM-XML-001 | Extract to new agent |
| `evidence_packager.py` | GL-FOUND-X-005 | Refactor to use standard evidence agent |

### New Agent Development

| Agent | Estimated Effort | Dependencies |
|-------|------------------|--------------|
| GL-CBAM-CALC-001 | Medium | Factor library integration |
| GL-CBAM-XML-001 | Medium | XSD schema integration |

---

## References

- [GreenLang_Agent_Catalog (2).xlsx](../GreenLang_Agent_Catalog%20(2).xlsx) - Full 402-agent catalog
- [GL-CBAM-APP](../GL-CBAM-APP/) - Existing CBAM application code
- [PRD Section 11](GreenLang_PRD_MVP_2026.md#11-agent-architecture--pipeline) - Agent architecture details
