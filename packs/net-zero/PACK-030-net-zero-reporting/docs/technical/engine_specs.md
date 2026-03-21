# PACK-030: Engine Specifications

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [Engine Overview](#engine-overview)
2. [Engine 1: Data Aggregation Engine](#engine-1-data-aggregation-engine)
3. [Engine 2: Narrative Generation Engine](#engine-2-narrative-generation-engine)
4. [Engine 3: Framework Mapping Engine](#engine-3-framework-mapping-engine)
5. [Engine 4: XBRL Tagging Engine](#engine-4-xbrl-tagging-engine)
6. [Engine 5: Dashboard Generation Engine](#engine-5-dashboard-generation-engine)
7. [Engine 6: Assurance Packaging Engine](#engine-6-assurance-packaging-engine)
8. [Engine 7: Report Compilation Engine](#engine-7-report-compilation-engine)
9. [Engine 8: Validation Engine](#engine-8-validation-engine)
10. [Engine 9: Translation Engine](#engine-9-translation-engine)
11. [Engine 10: Format Rendering Engine](#engine-10-format-rendering-engine)
12. [Cross-Engine Patterns](#cross-engine-patterns)

---

## 1. Engine Overview

| # | Engine | File | Lines | Key Functions | Performance Target |
|---|--------|------|-------|---------------|-------------------|
| 1 | Data Aggregation | `engines/data_aggregation_engine.py` | 1,800 | 5 | <3s full aggregation |
| 2 | Narrative Generation | `engines/narrative_generation_engine.py` | 2,000 | 5 | <2s per section |
| 3 | Framework Mapping | `engines/framework_mapping_engine.py` | 1,500 | 4 | <100ms per mapping |
| 4 | XBRL Tagging | `engines/xbrl_tagging_engine.py` | 1,600 | 4 | <3s full document |
| 5 | Dashboard Generation | `engines/dashboard_generation_engine.py` | 1,700 | 4 | <4s full dashboard |
| 6 | Assurance Packaging | `engines/assurance_packaging_engine.py` | 1,400 | 4 | <5s full bundle |
| 7 | Report Compilation | `engines/report_compilation_engine.py` | 2,200 | 4 | <2s per report |
| 8 | Validation | `engines/validation_engine.py` | 1,300 | 4 | <1s per report |
| 9 | Translation | `engines/translation_engine.py` | 1,100 | 4 | <3s per 1,000 words |
| 10 | Format Rendering | `engines/format_rendering_engine.py` | 1,900 | 5 | <5s for PDF |
| **Total** | | | **16,500** | **43** | |

All engines follow common patterns:
- Async/await for all I/O operations
- Pydantic v2 models for input/output validation
- SHA-256 provenance tracking on all outputs
- Structured logging with correlation IDs
- OpenTelemetry span instrumentation
- Deterministic Decimal arithmetic (no floating point in calculations)

---

## 2. Engine 1: Data Aggregation Engine

**File:** `engines/data_aggregation_engine.py` (~1,800 lines)

### Purpose

Collect and reconcile emissions data from all source packs (PACK-021/022/028/029) and GreenLang applications (GL-SBTi-APP, GL-CDP-APP, GL-TCFD-APP, GL-GHG-APP) with automated gap detection, completeness scoring, and data lineage tracking.

### Key Functions

| Function | Input | Output | Performance |
|----------|-------|--------|-------------|
| `aggregate_pack_data()` | Pack IDs, org ID, period | Aggregated pack data | <1.5s |
| `aggregate_app_data()` | App endpoints, org ID | Aggregated app data | <1.5s |
| `reconcile_sources()` | Aggregated datasets | Reconciliation report | <500ms |
| `generate_lineage()` | Aggregated data, report ID | Lineage graph (JSON) | <300ms |
| `calculate_completeness()` | Aggregated data, framework | Completeness score (0-100%) | <200ms |

### Data Flow

```
PACK-021 (baseline)     --+
PACK-022 (initiatives)  --+--> aggregate_pack_data() --+
PACK-028 (pathways)     --+                            |
PACK-029 (targets)      --+                            +--> reconcile_sources()
                                                       |         |
GL-SBTi-APP (targets)   --+                            |    reconciliation
GL-CDP-APP (history)    --+--> aggregate_app_data() ---+    report
GL-TCFD-APP (scenarios) --+                            |
GL-GHG-APP (inventory)  --+                            +--> Aggregated Dataset
                                                              |
                                                        generate_lineage()
                                                              |
                                                        calculate_completeness()
```

### Reconciliation Rules

| Rule | Source A | Source B | Resolution |
|------|---------|---------|------------|
| Scope 1 total | GL-GHG-APP | PACK-021 | Use GL-GHG-APP (latest inventory) |
| Base year emissions | PACK-021 | GL-SBTi-APP | Use PACK-021 (master baseline) |
| Reduction targets | PACK-029 | GL-SBTi-APP | Use GL-SBTi-APP (validated) |
| Scope 3 categories | GL-GHG-APP | PACK-021 | Use GL-GHG-APP (most granular) |

---

## 3. Engine 2: Narrative Generation Engine

**File:** `engines/narrative_generation_engine.py` (~2,000 lines)

### Purpose

Generate AI-assisted narrative drafts for qualitative disclosure sections with citation management, cross-framework consistency validation, and multi-language support.

### Key Functions

| Function | Input | Output | Performance |
|----------|-------|--------|-------------|
| `generate_narrative()` | Section type, framework, data | Draft narrative with citations | <2s |
| `add_citations()` | Narrative text, source data | Narrative with citation links | <500ms |
| `translate_narrative()` | Text, source lang, target lang | Translated text | <3s |
| `validate_consistency()` | Multiple narratives | Consistency report | <1s |
| `calculate_consistency_score()` | Narratives across frameworks | Score (0-100%) | <500ms |

### Narrative Generation Approach

1. **Template selection**: Choose section-appropriate template (governance, strategy, metrics, etc.)
2. **Data insertion**: Insert quantitative data points with references
3. **Contextual expansion**: Use AI to expand template with organization-specific context
4. **Citation linking**: Link every quantitative claim to source calculation
5. **Consistency check**: Validate against existing narratives in other frameworks
6. **Human review flag**: Mark for mandatory human review before publication

### Zero-Hallucination Safeguards

- All quantitative statements sourced from aggregated data (no generation)
- Every number cited with source system and calculation method
- Qualitative statements flagged for human review
- No forward-looking claims without explicit scenario basis
- Consistency score must exceed 95% before approval

---

## 4. Engine 3: Framework Mapping Engine

**File:** `engines/framework_mapping_engine.py` (~1,500 lines)

### Purpose

Map metrics, structures, and terminologies between 7 frameworks with bidirectional synchronization and confidence scoring.

### Key Functions

| Function | Input | Output | Performance |
|----------|-------|--------|-------------|
| `map_metric()` | Source framework, target, metric | Mapped metric | <100ms |
| `map_structure()` | Source framework, target, section | Mapped structure | <100ms |
| `bidirectional_sync()` | Framework pair, changes | Sync result | <200ms |
| `detect_conflicts()` | Mapping set | Conflict list | <100ms |

### Framework Mapping Matrix (42 Bidirectional Mappings)

| From \ To | SBTi | CDP | TCFD | GRI | ISSB | SEC | CSRD |
|-----------|------|-----|------|-----|------|-----|------|
| SBTi | - | C4 | M&T | 305-5 | S2.29 | 1505 | E1-4 |
| CDP | C4 | - | All | 305 | S2 | 1502-6 | E1 |
| TCFD | M&T | All | - | 305 | S2 | 1502-6 | E1 |
| GRI | 305-5 | 305 | 305 | - | S2.29 | 1505 | E1-6 |
| ISSB | S2.29 | S2 | S2 | S2.29 | - | 1502-6 | E1 |
| SEC | 1505 | 1502-6 | 1502-6 | 1505 | 1502-6 | - | E1-6 |
| CSRD | E1-4 | E1 | E1 | E1-6 | E1 | E1-6 | - |

---

## 5. Engine 4: XBRL Tagging Engine

**File:** `engines/xbrl_tagging_engine.py` (~1,600 lines)

### Purpose

Generate XBRL and inline XBRL (iXBRL) documents for SEC and CSRD digital reporting, with taxonomy validation.

### Key Functions

| Function | Input | Output | Performance |
|----------|-------|--------|-------------|
| `tag_metric()` | Metric, taxonomy | XBRL element | <50ms |
| `generate_xbrl()` | Report data, taxonomy | XBRL XML file | <3s |
| `generate_ixbrl()` | Report data, HTML template | iXBRL HTML file | <3s |
| `validate_taxonomy()` | XBRL document, taxonomy version | Validation result | <1s |

### Supported Taxonomies

| Taxonomy | Version | Framework | Elements |
|----------|---------|-----------|----------|
| SEC Climate Disclosure | 2024 | SEC | ~50 elements |
| CSRD ESRS E1 | 2024 | CSRD | ~80 elements |
| IFRS S2 | 2023 | ISSB | ~40 elements |

---

## 6. Engine 5: Dashboard Generation Engine

**File:** `engines/dashboard_generation_engine.py` (~1,700 lines)

### Purpose

Create interactive HTML5 dashboards with charts, framework coverage heatmaps, deadline timers, and stakeholder-specific views.

### Key Functions

| Function | Input | Output | Performance |
|----------|-------|--------|-------------|
| `generate_executive_dashboard()` | Aggregated data | HTML dashboard | <4s |
| `generate_framework_dashboard()` | Framework, data | Framework-specific HTML | <2s |
| `generate_stakeholder_view()` | Stakeholder type, data | Customized HTML | <2s |
| `add_interactivity()` | HTML, chart config | Interactive HTML | <1s |

### Dashboard Components

- **Framework Coverage Heatmap**: 7x1 grid showing completion percentage per framework
- **Deadline Countdown Timers**: Visual countdown for each framework deadline
- **Emissions Trend Chart**: Multi-year Scope 1/2/3 trend with targets
- **Consistency Score Gauge**: Cross-framework consistency percentage
- **Data Completeness Bar**: Per-framework data gap indicator
- **Report Status Table**: Status of all generated reports

---

## 7. Engine 6: Assurance Packaging Engine

**File:** `engines/assurance_packaging_engine.py` (~1,400 lines)

### Purpose

Package evidence bundles for ISAE 3410/3000 audits with calculation provenance, data lineage, methodology documentation, and control matrices.

### Key Functions

| Function | Input | Output | Performance |
|----------|-------|--------|-------------|
| `collect_provenances()` | Report IDs | SHA-256 hash collection | <1s |
| `generate_lineage_diagrams()` | Report ID, metrics | SVG/PNG diagrams | <2s |
| `package_methodology()` | Framework, engines used | Methodology PDF | <2s |
| `create_control_matrix()` | Framework, ISAE standard | Control matrix Excel | <1s |

### Evidence Bundle Contents

```
evidence_bundle_2025.zip
+-- manifest.json                    # Bundle contents and checksums
+-- provenance/
|   +-- calculation_hashes.json      # SHA-256 for every calculation
|   +-- report_hashes.json           # SHA-256 for every report output
+-- lineage/
|   +-- scope1_lineage.svg           # Visual lineage for Scope 1
|   +-- scope2_lineage.svg           # Visual lineage for Scope 2
|   +-- scope3_lineage.svg           # Visual lineage for Scope 3
+-- methodology/
|   +-- ghg_methodology.pdf          # GHG accounting methodology
|   +-- emission_factors.pdf         # Emission factor sources
|   +-- calculation_methods.pdf      # Calculation engine documentation
+-- controls/
|   +-- isae_3410_matrix.xlsx        # ISAE 3410 control requirements
|   +-- review_log.json              # Internal review activity log
|   +-- approval_records.json        # Approval signoff records
+-- source_data_summary/
|   +-- data_sources.json            # List of all source systems
|   +-- reconciliation_report.json   # Data reconciliation results
```

---

## 8. Engine 7: Report Compilation Engine

**File:** `engines/report_compilation_engine.py` (~2,200 lines)

### Purpose

Assemble final reports from individual sections, apply branding, generate table of contents, and create cross-references between related sections.

### Key Functions

| Function | Input | Output | Performance |
|----------|-------|--------|-------------|
| `compile_report()` | Sections, template | Compiled report | <2s |
| `apply_branding()` | Report, branding config | Branded report | <500ms |
| `generate_toc()` | Report sections | Table of contents | <200ms |
| `add_cross_references()` | Report, reference map | Cross-referenced report | <300ms |

---

## 9. Engine 8: Validation Engine

**File:** `engines/validation_engine.py` (~1,300 lines)

### Purpose

Validate reports against framework-specific JSON schemas, check completeness of required fields, and validate cross-framework consistency.

### Key Functions

| Function | Input | Output | Performance |
|----------|-------|--------|-------------|
| `validate_schema()` | Report data, framework schema | Schema validation result | <500ms |
| `validate_completeness()` | Report data, required fields | Completeness score | <300ms |
| `validate_consistency()` | Multiple reports | Consistency report | <1s |
| `calculate_quality_score()` | All validation results | Quality score (0-100%) | <200ms |

### Validation Checks

| Category | Checks | Severity |
|----------|--------|----------|
| Schema compliance | Field types, required fields, enum values | Critical |
| Completeness | Required metrics present, sections populated | High |
| Cross-framework consistency | Same metrics match across frameworks | High |
| Narrative quality | Citations present, no contradictions | Medium |
| XBRL compliance | Valid elements, correct taxonomy | Critical (SEC/CSRD) |
| Data freshness | Source data within acceptable age | Medium |

---

## 10. Engine 9: Translation Engine

**File:** `engines/translation_engine.py` (~1,100 lines)

### Purpose

Multi-language support for narrative content with climate-specific terminology glossary, citation preservation, and translation quality validation.

### Key Functions

| Function | Input | Output | Performance |
|----------|-------|--------|-------------|
| `translate_narrative()` | Text, source/target language | Translated text | <3s per 1K words |
| `validate_translation()` | Original, translated text | Quality score | <500ms |
| `maintain_terminology()` | Text, glossary | Corrected text | <200ms |
| `preserve_citations()` | Text with citations | Text with preserved citations | <100ms |

### Supported Languages

| Code | Language | Quality Target |
|------|----------|---------------|
| `en` | English | Native (source) |
| `de` | German | 98%+ |
| `fr` | French | 98%+ |
| `es` | Spanish | 98%+ |

### Climate Glossary (150+ terms)

Key terms maintained across languages:
- Net-zero / Netto-Null / Neutralite carbone / Cero neto
- Scope 1 emissions / Scope-1-Emissionen / Emissions du scope 1 / Emisiones de alcance 1
- Science-based target / Wissenschaftsbasiertes Ziel / Objectif fonde sur la science / Objetivo basado en la ciencia

---

## 11. Engine 10: Format Rendering Engine

**File:** `engines/format_rendering_engine.py` (~1,900 lines)

### Purpose

Render compiled reports into multiple output formats with format-specific optimizations.

### Key Functions

| Function | Input | Output | Performance |
|----------|-------|--------|-------------|
| `render_pdf()` | Compiled report, branding | PDF binary | <5s |
| `render_html()` | Compiled report, interactivity | HTML file | <2s |
| `render_excel()` | Compiled report, formatting | XLSX binary | <2s |
| `render_json()` | Compiled report, API version | JSON string | <500ms |
| `render_xbrl()` | Compiled report, taxonomy | XBRL XML (delegates to Engine 4) | <3s |

### Format Specifications

| Format | Engine | Features |
|--------|--------|----------|
| PDF | WeasyPrint 60+ | Custom branding, TOC, page numbers, hyperlinks, charts |
| HTML | Jinja2 + Plotly | Responsive, interactive charts, drill-down, export-to-PDF |
| Excel | openpyxl 3.1+ | Formatted tables, charts, data validation, protected sheets |
| JSON | stdlib json | Paginated, field selection, versioned, provenance-tagged |
| XBRL | lxml 5.1+ | SEC taxonomy, context/unit refs, validation |
| iXBRL | lxml + HTML | Human-readable + machine-readable combined |

---

## 12. Cross-Engine Patterns

### Provenance Tracking

Every engine tags its outputs with SHA-256 provenance:

```python
import hashlib
from decimal import Decimal

def calculate_provenance(data: dict) -> str:
    """Generate SHA-256 hash of calculation inputs and outputs."""
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
```

### Error Handling

All engines use structured error handling with classification:

```python
class EngineError(Exception):
    """Base exception for all engine errors."""
    def __init__(self, message: str, engine: str, recoverable: bool = False):
        self.message = message
        self.engine = engine
        self.recoverable = recoverable

class DataAggregationError(EngineError):
    """Raised when data aggregation fails."""
    pass

class ValidationError(EngineError):
    """Raised when validation fails."""
    pass
```

### Telemetry

All engine functions are instrumented with OpenTelemetry:

```python
from opentelemetry import trace

tracer = trace.get_tracer("pack030.engines")

class DataAggregationEngine:
    @tracer.start_as_current_span("aggregate_pack_data")
    async def aggregate_pack_data(self, ...):
        span = trace.get_current_span()
        span.set_attribute("pack.count", len(pack_ids))
        # ... implementation
```

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
