---
title: "Concept: factor record"
description: What every field of a Canonical Factor Record means.
---

# The Canonical Factor Record

Every Factors API response returns one or more **Canonical Factor Records**. This page is the field-by-field reference.

## Top-level fields

| Field                  | Type            | Meaning                                                        |
|------------------------|-----------------|----------------------------------------------------------------|
| `factor_id`            | string          | Stable, edition-scoped id (`ef:co2:diesel:us:2026`).           |
| `edition_id`           | string          | The catalog edition this record came from.                     |
| `co2e_per_unit`        | number          | kg CO2-equivalent per `unit`.                                  |
| `unit`                 | string          | Activity unit (e.g. `therm`, `kWh`, `tonne_km`, `gal`).        |
| `method_profile`       | string          | Method family (Scope 1, Scope 2 LB/MB, freight ISO 14083, ...).|
| `factor_status`        | string          | `certified`, `preview`, or `connector_only`.                   |
| `redistribution_class` | string          | `redistribute_open`, `restricted`, `connector_only`, `customer_private`, `internal_only`. |
| `source`               | object          | See [Source](#source) below.                                   |
| `gas_breakdown`        | object          | Per-gas split (CO2 / CH4 / N2O / F-gases).                     |
| `uncertainty`          | object          | Lower / upper bounds and distribution shape.                   |
| `quality`              | object          | Data Quality Score (DQS) per the Pedigree matrix.              |
| `provenance`           | object          | Lineage hash, ingestion timestamps, approval chain.            |

## Source

```json
{
  "source_id": "epa-ghg-2026",
  "publisher": "US Environmental Protection Agency",
  "publication_year": 2026,
  "license": {
    "name": "Public Domain (US Govt)",
    "redistribution": "redistribute_open"
  },
  "url": "https://www.epa.gov/...",
  "doi": null
}
```

## Gas breakdown

```json
{
  "co2": 9.32,
  "ch4": 0.04,
  "n2o": 0.001,
  "co2e": 10.21
}
```

## Uncertainty

```json
{
  "low": 9.6,
  "high": 10.8,
  "confidence_pct": 95,
  "distribution": "lognormal"
}
```

## Quality (Pedigree matrix DQS)

```json
{
  "dqs": 4.2,
  "axes": {
    "reliability": 5,
    "completeness": 4,
    "temporal": 4,
    "geographic": 5,
    "technological": 3
  }
}
```

## Provenance

```json
{
  "lineage_hash": "sha256:abc123...",
  "ingested_at": "2026-04-01T00:00:00Z",
  "approved_by": "factor-ops@greenlang.io",
  "approved_at": "2026-04-02T12:30:00Z",
  "review_log_id": "review-2026-q2-127"
}
```

## How to use

* Use **`factor_id`** as the primary key when storing or referencing a factor.
* Use **`edition_id`** as the row in your audit trail; pin to it for reproducibility.
* Use **`provenance.lineage_hash`** to prove the factor has not been modified.
* Use **`uncertainty`** to drive Monte Carlo simulations or sensitivity analyses.
* Use **`redistribution_class`** to decide whether you can publish the factor in a downstream report.
