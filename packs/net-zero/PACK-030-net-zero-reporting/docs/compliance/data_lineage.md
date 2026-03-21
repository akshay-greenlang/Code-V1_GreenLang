# PACK-030: Data Lineage Documentation

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [Overview](#overview)
2. [Lineage Architecture](#lineage-architecture)
3. [Source Systems](#source-systems)
4. [Transformation Steps](#transformation-steps)
5. [Lineage Tracking Implementation](#lineage-tracking-implementation)
6. [Lineage Diagrams](#lineage-diagrams)
7. [Lineage Queries](#lineage-queries)
8. [Audit Requirements](#audit-requirements)

---

## 1. Overview

Data lineage in PACK-030 provides complete traceability from source system transactions to final report metrics. Every number in every report can be traced back through the transformation chain to its original source, enabling auditors to verify the integrity of the reporting pipeline.

### Lineage Principles

| Principle | Implementation |
|-----------|---------------|
| **Complete traceability** | Every report metric linked to source records |
| **Transformation transparency** | All data transformations documented with logic |
| **Immutability** | Lineage records are append-only, never modified |
| **Machine-readable** | JSON-based lineage graph for programmatic analysis |
| **Human-readable** | SVG/PNG visual diagrams for auditor review |
| **Performance** | Lineage tracking adds <5% overhead to report generation |

---

## 2. Lineage Architecture

### Lineage Data Model

```
Source Record (upstream system)
    |
    +-- Source Record ID (UUID)
    +-- Source System Name
    +-- Source Timestamp
    +-- Raw Value
    |
    v
Transformation Step (engine processing)
    |
    +-- Step Number
    +-- Engine Name
    +-- Transformation Type (aggregation, conversion, mapping)
    +-- Input Values
    +-- Output Values
    +-- Formula/Logic Applied
    +-- Provenance Hash
    |
    v
Report Metric (final output)
    |
    +-- Metric ID (UUID)
    +-- Report ID (UUID)
    +-- Metric Name
    +-- Metric Value
    +-- Unit
    +-- Framework
    +-- Provenance Hash (SHA-256)
```

### Database Schema

```sql
CREATE TABLE gl_nz_data_lineage (
    lineage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id),
    metric_name VARCHAR(200) NOT NULL,
    source_system VARCHAR(100) NOT NULL,
    transformation_steps JSONB NOT NULL DEFAULT '[]',
    source_records JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 3. Source Systems

### Source System Inventory

| # | Source System | Data Type | Integration Method | Data Points |
|---|-------------|-----------|-------------------|-------------|
| 1 | PACK-021 | Baseline emissions | REST API | Scope 1/2/3 base year totals, activity data |
| 2 | PACK-022 | Reduction initiatives | REST API | Initiative list, MACC curve, abatement potential |
| 3 | PACK-028 | Sector pathways | REST API | Sector-specific pathways, convergence data |
| 4 | PACK-029 | Interim targets | REST API | 5/10-year targets, progress, variance analysis |
| 5 | GL-SBTi-APP | SBTi targets | GraphQL | Validated targets, submission history |
| 6 | GL-CDP-APP | CDP history | GraphQL | Historical responses, scores |
| 7 | GL-TCFD-APP | TCFD scenarios | GraphQL | Scenario analysis, risk assessments |
| 8 | GL-GHG-APP | GHG inventory | GraphQL | Scope 1/2/3 emissions, emission factors |

### Source Record Format

```json
{
  "source_system": "GL-GHG-APP",
  "record_type": "ghg_inventory",
  "record_id": "uuid",
  "timestamp": "2026-03-15T14:30:00Z",
  "data": {
    "scope": "scope1",
    "category": "stationary_combustion",
    "emission_source": "natural_gas_boilers",
    "activity_data": {
      "value": 5000000,
      "unit": "kWh"
    },
    "emission_factor": {
      "value": 0.18293,
      "unit": "kgCO2e/kWh",
      "source": "DEFRA 2025"
    },
    "calculated_emissions": {
      "value": 914.65,
      "unit": "tCO2e"
    }
  }
}
```

---

## 4. Transformation Steps

### Transformation Types

| Type | Description | Example |
|------|-------------|---------|
| **Collection** | Raw data fetched from source system | Fetch Scope 1 from GL-GHG-APP |
| **Aggregation** | Multiple values combined into total | Sum all Scope 1 categories into Scope 1 total |
| **Conversion** | Unit conversion applied | Convert MWh to kWh |
| **Mapping** | Framework-specific metric mapping | Map "Scope 1 total" to GRI 305-1 |
| **Calculation** | Derived metric calculated | Intensity = Emissions / Revenue |
| **Reconciliation** | Multiple sources compared and resolved | Select GL-GHG-APP value over PACK-021 |
| **Formatting** | Value formatted for output | Round to 2 decimal places for PDF |

### Transformation Step Format

```json
{
  "step_number": 1,
  "transformation_type": "collection",
  "engine": "DataAggregationEngine",
  "description": "Fetch Scope 1 emissions from GL-GHG-APP",
  "inputs": {
    "source_system": "GL-GHG-APP",
    "endpoint": "/api/v1/ghg/inventory",
    "query_params": {"org_id": "uuid", "year": 2025, "scope": "scope1"}
  },
  "outputs": {
    "stationary_combustion_tco2e": "28000.00",
    "mobile_combustion_tco2e": "12000.00",
    "process_emissions_tco2e": "3000.00",
    "fugitive_emissions_tco2e": "2000.00"
  },
  "timestamp": "2026-03-20T09:00:01Z",
  "provenance_hash": "sha256:a1b2c3..."
}
```

---

## 5. Lineage Tracking Implementation

### Automatic Lineage Recording

Every engine operation automatically records lineage:

```python
class DataAggregationEngine:
    async def aggregate_pack_data(self, org_id, period):
        lineage_recorder = LineageRecorder(report_id=self.report_id)

        # Step 1: Fetch from PACK-021
        pack021_data = await self.pack021.fetch_baseline(org_id)
        lineage_recorder.add_step(
            source_system="PACK-021",
            transformation_type="collection",
            inputs={"org_id": org_id},
            outputs=pack021_data.to_dict(),
        )

        # Step 2: Fetch from GL-GHG-APP
        ghg_data = await self.ghg_app.fetch_inventory(org_id, period)
        lineage_recorder.add_step(
            source_system="GL-GHG-APP",
            transformation_type="collection",
            inputs={"org_id": org_id, "period": period},
            outputs=ghg_data.to_dict(),
        )

        # Step 3: Reconcile
        reconciled = self.reconcile(pack021_data, ghg_data)
        lineage_recorder.add_step(
            source_system="DataAggregationEngine",
            transformation_type="reconciliation",
            inputs={"pack021": pack021_data, "ghg_app": ghg_data},
            outputs=reconciled.to_dict(),
            decision="GL-GHG-APP selected (latest inventory)",
        )

        # Save lineage
        await lineage_recorder.save()
```

---

## 6. Lineage Diagrams

### Visual Diagram Format

PACK-030 generates SVG lineage diagrams showing the complete data flow for each scope:

#### Scope 1 Lineage Example

```
+------------------+     +---------------------+     +------------------+
| GL-GHG-APP       |     | DataAggregation     |     | Report Output    |
|                  |     | Engine              |     |                  |
| Stationary:      |---->|                     |     | SBTi Report:     |
|   28,000 tCO2e   |     | Collect + Aggregate |---->|   Scope 1 Total  |
| Mobile:          |---->|                     |     |   45,000 tCO2e   |
|   12,000 tCO2e   |     | Reconcile with      |     |                  |
| Process:         |---->| PACK-021 baseline   |     | CDP C6.1:        |
|   3,000 tCO2e    |     |                     |---->|   Scope 1 Total  |
| Fugitive:        |---->| Sum = 45,000 tCO2e  |     |   45,000 tCO2e   |
|   2,000 tCO2e    |     |                     |     |                  |
+------------------+     | SHA-256: a1b2c3...  |     | TCFD M&T:        |
                         +---------------------+---->|   Scope 1 Total  |
+------------------+              |              |   |   45,000 tCO2e   |
| PACK-021         |              |              |   +------------------+
| Base Year:       |-----> Reconciliation        |
|   42,000 tCO2e   |     (confirms GL-GHG-APP)  |
+------------------+                              |
                                                   +-> 7 framework reports
```

### Machine-Readable Lineage Graph

```json
{
  "metric": "scope1_total_tco2e",
  "final_value": "45000.00",
  "unit": "tCO2e",
  "provenance_hash": "sha256:a1b2c3...",
  "graph": {
    "nodes": [
      {"id": "src_ghg_stationary", "type": "source", "system": "GL-GHG-APP", "value": "28000.00"},
      {"id": "src_ghg_mobile", "type": "source", "system": "GL-GHG-APP", "value": "12000.00"},
      {"id": "src_ghg_process", "type": "source", "system": "GL-GHG-APP", "value": "3000.00"},
      {"id": "src_ghg_fugitive", "type": "source", "system": "GL-GHG-APP", "value": "2000.00"},
      {"id": "src_pack021", "type": "source", "system": "PACK-021", "value": "42000.00"},
      {"id": "agg_scope1", "type": "aggregation", "engine": "DataAggregationEngine"},
      {"id": "recon_scope1", "type": "reconciliation", "engine": "DataAggregationEngine"},
      {"id": "out_sbti", "type": "output", "framework": "SBTi", "value": "45000.00"},
      {"id": "out_cdp", "type": "output", "framework": "CDP", "value": "45000.00"},
      {"id": "out_tcfd", "type": "output", "framework": "TCFD", "value": "45000.00"}
    ],
    "edges": [
      {"from": "src_ghg_stationary", "to": "agg_scope1"},
      {"from": "src_ghg_mobile", "to": "agg_scope1"},
      {"from": "src_ghg_process", "to": "agg_scope1"},
      {"from": "src_ghg_fugitive", "to": "agg_scope1"},
      {"from": "src_pack021", "to": "recon_scope1"},
      {"from": "agg_scope1", "to": "recon_scope1"},
      {"from": "recon_scope1", "to": "out_sbti"},
      {"from": "recon_scope1", "to": "out_cdp"},
      {"from": "recon_scope1", "to": "out_tcfd"}
    ]
  }
}
```

---

## 7. Lineage Queries

### API Endpoints

```http
# Get lineage for a specific metric in a report
GET /api/v1/data/lineage/{report_id}?metric_name=scope1_total_tco2e

# Get all lineage for a report
GET /api/v1/data/lineage/{report_id}

# Get lineage diagram (SVG)
GET /api/v1/data/lineage/{report_id}/diagram?format=svg&metric=scope1_total

# Get lineage summary
GET /api/v1/data/lineage/{report_id}/summary
```

### Database Queries

```sql
-- Find all source systems for a specific metric
SELECT
    l.metric_name,
    l.source_system,
    jsonb_array_length(l.transformation_steps) AS step_count,
    l.created_at
FROM gl_nz_data_lineage l
WHERE l.report_id = 'report-uuid'
    AND l.metric_name = 'scope1_total_tco2e';

-- View lineage summary across all metrics
SELECT * FROM gl_nz_lineage_summary
WHERE report_id = 'report-uuid';
```

---

## 8. Audit Requirements

### ISAE 3410 Lineage Requirements

| Requirement | PACK-030 Evidence |
|-------------|-------------------|
| Source identification | `source_records` in lineage table |
| Transformation documentation | `transformation_steps` in lineage table |
| Calculation verification | SHA-256 provenance hashes |
| System documentation | Architecture docs, engine specs |
| Access control | RLS policies, audit trail |
| Change management | Immutable audit log |

### Lineage Completeness Check

```python
async def verify_lineage_completeness(report_id):
    """Verify that all report metrics have complete lineage."""
    metrics = await db.get_report_metrics(report_id)
    lineage_records = await db.get_lineage(report_id)

    lineage_metrics = {l.metric_name for l in lineage_records}
    report_metrics = {m.metric_name for m in metrics}

    missing = report_metrics - lineage_metrics
    if missing:
        raise LineageIncompleteError(
            f"Missing lineage for {len(missing)} metrics: {missing}"
        )
    return True
```

### Retention Policy

| Data Type | Retention Period | Storage |
|-----------|-----------------|---------|
| Lineage records | 7 years | PostgreSQL |
| Lineage diagrams | 7 years | S3 archive |
| Source record references | 7 years | PostgreSQL |
| Transformation logs | 7 years | PostgreSQL |
| Provenance hashes | Permanent | PostgreSQL + S3 |

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
