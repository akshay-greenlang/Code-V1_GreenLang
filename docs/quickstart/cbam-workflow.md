# CBAM Flagship Workflow - Quickstart Guide

**Time to Complete:** 5 minutes
**Difficulty:** Beginner
**Prerequisites:** GreenLang CLI installed

## Overview

This guide demonstrates the complete CBAM (Carbon Border Adjustment Mechanism) workflow from data intake to verified output. This is the "impossible to argue with" demo that shows GreenLang's core value proposition.

**What You'll Learn:**
- How to run a deterministic climate calculation
- How to verify calculation provenance
- How to trace outputs back to emission factors

## The CBAM Use Case

The EU Carbon Border Adjustment Mechanism requires importers to report the embedded carbon emissions in goods imported into the EU. This workflow calculates those emissions for a sample shipment.

## Workflow Diagram

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│   Import    │───>│   Normalize  │───>│  Calculate  │───>│   Generate   │───>│    Verify    │
│   Data      │    │   Units      │    │  Emissions  │    │   Report     │    │   Artifacts  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └──────────────┘
     │                   │                   │                   │                   │
     v                   v                   v                   v                   v
  input.json        normalized.json      emissions.json      report.xml         run.json
                                         + provenance        + SBOM
```

## Step 1: Create Sample Data

Create a file called `cbam_import.json` with a sample shipment:

```json
{
  "shipment_id": "CBAM-2026-001",
  "importer": {
    "name": "EcoImports GmbH",
    "country": "DE",
    "eori": "DE123456789012345"
  },
  "goods": [
    {
      "cn_code": "7208",
      "description": "Hot-rolled steel plates",
      "quantity": 100,
      "unit": "tonnes",
      "country_of_origin": "CN",
      "installation_id": "CN-STEEL-001",
      "production_method": "blast_furnace"
    }
  ],
  "reporting_period": {
    "start": "2026-01-01",
    "end": "2026-03-31"
  }
}
```

## Step 2: Run the Calculation

```bash
# Run the CBAM calculation pipeline
gl run cbam-embedded-emissions --input cbam_import.json --output results/

# Expected output:
# [INFO] Loading input data...
# [INFO] Normalizing units...
# [INFO] Calculating embedded emissions...
# [INFO]   Using factor: CBAM-2024-steel-blast_furnace (1.85 tCO2e/t)
# [INFO] Generating report...
# [INFO] Pipeline completed successfully
# [INFO] Artifacts written to: results/
```

## Step 3: Examine the Results

### Emissions Output (results/emissions.json)

```json
{
  "shipment_id": "CBAM-2026-001",
  "reporting_period": "2026-Q1",
  "goods": [
    {
      "cn_code": "7208",
      "quantity_tonnes": 100,
      "embedded_emissions": {
        "direct": 150.0,
        "indirect": 35.0,
        "total": 185.0,
        "unit": "tCO2e"
      },
      "emission_intensity": 1.85,
      "intensity_unit": "tCO2e/t",
      "factor_citation": {
        "factor_id": "CBAM-2024-steel-blast_furnace",
        "source": "EU CBAM Default Values",
        "vintage": 2024,
        "methodology": "CBAM Regulation Annex III"
      }
    }
  ],
  "totals": {
    "direct_emissions": 150.0,
    "indirect_emissions": 35.0,
    "total_emissions": 185.0,
    "unit": "tCO2e"
  }
}
```

### Run Artifact (results/run.json)

```json
{
  "schema_version": "1.0",
  "run_id": "cbam_2026_001_abc123",
  "pipeline": {
    "name": "cbam-embedded-emissions",
    "version": "1.0.0",
    "hash": "sha256:9f86d08..."
  },
  "status": "completed",
  "success": true,
  "started_at": "2026-02-03T12:00:00Z",
  "completed_at": "2026-02-03T12:00:05Z",
  "provenance": {
    "hash": "sha256:abc123...",
    "factors_used": [
      "CBAM-2024-steel-blast_furnace"
    ],
    "input_hash": "sha256:def456...",
    "output_hash": "sha256:ghi789..."
  }
}
```

## Step 4: Verify the Calculation

The verification step proves that the calculation is:
1. **Reproducible** - Same inputs produce identical outputs
2. **Auditable** - Every number traces to a citable source
3. **Signed** - Artifacts are cryptographically verified

```bash
# Verify the run artifacts
gl verify results/run.json

# Output:
# ============================================
# GreenLang Verification Report
# ============================================
#
# Run ID: cbam_2026_001_abc123
# Pipeline: cbam-embedded-emissions@1.0.0
# Status: VERIFIED
#
# Signer: GreenLang Official Packs
# Signed: 2026-02-03T12:00:05Z
# Key ID: GL-PACK-2026-001
#
# SBOM Summary:
#   - Dependencies: 12
#   - Known Vulnerabilities: 0
#   - License: Apache-2.0, MIT
#
# Provenance:
#   - Input Hash: sha256:def456... MATCH
#   - Output Hash: sha256:ghi789... MATCH
#   - Factor Citations: 1 (all verified)
#
# Factor Audit:
#   - CBAM-2024-steel-blast_furnace
#     Source: EU CBAM Regulation
#     Vintage: 2024
#     Citation: COMPLETE
#
# Result: ALL CHECKS PASSED
# ============================================
```

## Step 5: Re-run for Determinism Verification

```bash
# Run again with the same inputs
gl run cbam-embedded-emissions --input cbam_import.json --output results2/

# Compare hashes
gl verify --compare results/run.json results2/run.json

# Output:
# Comparing runs...
# Input hash:  IDENTICAL
# Output hash: IDENTICAL
# Provenance:  IDENTICAL
#
# Determinism verification: PASSED
# Both runs produced byte-identical outputs.
```

## Understanding the Output

### Why This Matters for Compliance

1. **Auditability**: Every number in the report can be traced back to:
   - The input data (with hash)
   - The emission factor used (with full citation)
   - The calculation methodology (documented in provenance)

2. **Reproducibility**: Anyone can re-run this calculation and get identical results:
   - Same inputs + same factor version = same outputs
   - Hash verification proves no tampering

3. **Transparency**: The SBOM shows:
   - All dependencies used in the calculation
   - No known security vulnerabilities
   - Compliant licenses

## Common Operations

### Use Specific Factor Vintage

```bash
gl run cbam-embedded-emissions \
  --input cbam_import.json \
  --config factor_vintage=2024 \
  --output results/
```

### Generate XML Report for EU Submission

```bash
gl run cbam-embedded-emissions \
  --input cbam_import.json \
  --output results/ \
  --format xml
```

### Batch Processing

```bash
# Process multiple shipments
gl run cbam-embedded-emissions \
  --input shipments/*.json \
  --output results/ \
  --batch
```

### Export for External Audit

```bash
# Create audit package with all artifacts
gl export-audit results/run.json --output audit_package.zip
```

## Troubleshooting

### "Factor not found"

```bash
# Check available factors
gl factors list --source CBAM --vintage 2024

# Use fallback factor
gl run cbam-embedded-emissions --input input.json --config use_default_factors=true
```

### "Verification failed"

```bash
# Check what's different
gl verify --verbose results/run.json

# Re-run with same environment
gl run cbam-embedded-emissions --input input.json --deterministic
```

## Next Steps

- [CBAM API Reference](../api/cbam.md)
- [Custom Pipelines](../guides/custom-pipelines.md)
- [Factor Database](../reference/emission-factors.md)
- [Regulatory Compliance](../compliance/cbam.md)

## Summary

This workflow demonstrates GreenLang's core value proposition:

| Feature | Benefit |
|---------|---------|
| Deterministic execution | Same inputs always produce same outputs |
| Full provenance | Every number traces to a source |
| Factor citations | Auditors can verify methodology |
| SBOM/signing | Supply chain transparency |
| Verification | Cryptographic proof of integrity |

**GreenLang makes climate compliance calculations that auditors and regulators can trust.**
