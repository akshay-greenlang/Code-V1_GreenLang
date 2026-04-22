# Changelog

Published editions of the GreenLang Factors catalog. Each edition is immutable and carries a deterministic fingerprint (see [version-pinning](../concepts/version-pinning.md)).

**Binding cut-list:** [docs/editions/v1-certified-cutlist.md](../../editions/v1-certified-cutlist.md) — the founder-approved promotion manifest that locks the v1.0 Certified contents.

---

## v1.0 Certified (seed)

The v1.0 Certified edition is delivered as **7 thin vertical slices**, each promoted as its own Certified edition ID. See `docs/editions/v1-certified-cutlist.md` for the full founder-approved promotion manifest.

### Slice 1 — `2027.Q1-electricity`

- **Scope:** Location-based + market-based Scope 2 for India, EU, UK, US, AU, JP, CA.
- **Sources:** eGRID, EPA GHG Emission Factors Hub, DESNZ GHG conversion, DEFRA, Australian NGA, Japan METI, Green-e Residual Mix, Electricity Maps (connector-only), IEA (connector-only), GreenLang built-in, India CEA (pending registry row).
- **Method profiles:** `corporate_scope2_location_based`, `corporate_scope2_market_based`.
- **FQS gate:** Certified minima (75/100 composite; all five 1-5 components >= 3).
- **Rollback target:** n/a (first slice).
- **Highlights:**
  - eGRID 2024 subregions (SERC, NPCC, RFC, ...).
  - DESNZ 2026 annual conversion factors.
  - Green-e Energy Residual Mix 2025 (restricted redistribution).
  - AIB / EECS Guarantee of Origin integration for EU market-based Scope 2.
- **Changelog-draft shape:** see `greenlang/factors/watch/changelog_draft.py`.

### Slice 2 — `2027.Q1-corporate-scope1` (planned)

- **Scope:** Stationary combustion, mobile combustion, fugitive emissions (refrigerants, SF6), process emissions.
- **Sources:** EPA GHG Emission Factors Hub, IPCC 2019 Refinement, DEFRA, IEA (connector).
- **Method profiles:** `corporate_scope1`.

### Slice 3 — `2027.Q2-corporate-scope3` (planned)

- **Scope:** All 15 GHG Protocol Scope 3 categories.
- **Sources:** ecoinvent (connector), exiobase, EEIO tables (US EPA, UK ONS, JRC EU), supplier PCF intake.
- **Method profiles:** `corporate_scope3`.

### Slice 4 — `2027.Q2-freight` (planned)

- **Scope:** Freight ISO 14083 / GLEC for road, rail, sea, air, inland waterway, pipeline.
- **Sources:** GLEC default values, Smart Freight Centre, IMO EEDI, ICAO CORSIA.
- **Method profiles:** `freight_iso_14083`.

### Slice 5 — `2027.Q3-product-carbon` (planned)

- **Scope:** Cradle-to-gate and cradle-to-grave PCF for priority CPC categories.
- **Sources:** ecoinvent (connector), US LCI, Sphera GaBi (connector).
- **Method profiles:** `product_carbon` with PACT / ISO 14067 / PEF output variants.

### Slice 6 — `2027.Q3-cbam` (planned)

- **Scope:** CBAM goods (cement, iron & steel, aluminium, fertilisers, hydrogen, electricity) per Implementing Act 2023/1773.
- **Sources:** JRC CBAM default values 2024 + supplier primary-data intake via tenant overlay.
- **Method profiles:** `eu_cbam`.

### Slice 7 — `2027.Q4-pcaf` (planned)

- **Scope:** PCAF Scope 3 Cat 15 asset classes (listed equity, corporate bonds/loans, project finance, CRE, mortgages, motor vehicle loans, sovereign debt).
- **Sources:** CDP disclosures, EEIO, property-energy tables per country.
- **Method profiles:** `finance_proxy`.

---

## Edition naming convention

```
v<major>.<minor>.<patch>[-<slice>]          # semantic for platform tools
<YYYY>.Q<n>[-<slice>]                        # calendar-quarter anchored editions (common)
```

Examples seen in the wild:

- `2027.Q1-electricity` (slice 1)
- `2026.Q4` (multi-slice quarterly bundle)
- `v1.0.0-certified`
- `2027.Q2-cbam-hotfix-1`

The calendar-quarter convention aligns with source publication cadences (eGRID annual, DESNZ annual, IEA annual).

---

## How to subscribe to changes

Use webhooks to get notified:

```bash
curl -sS -X POST "$GL_API_BASE/api/v1/webhooks/subscriptions" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -d '{
    "target_url": "https://hooks.acme.com/greenlang",
    "event_types": ["edition.published", "factor.deprecated", "factor.updated"]
  }'
```

See [webhooks](../api-reference/webhooks.md).

Or poll:

```bash
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
     "$GL_API_BASE/api/v1/editions?status=stable&updated_since=2027-01-01"
```

---

## Promotion gate (S1..S9)

Every slice promotion must pass the 9 required items from the release-signoff pipeline (`greenlang/factors/quality/release_signoff.py`). v1 policy locks all 9 to `severity = required`:

| # | Check |
|---|---|
| S1 | Q1..Q6 QA gates pass for all factors (`total_failed == 0 AND total_factors > 0`). |
| S2 | No unresolved duplicate pairs (`human_review == 0` on DedupReport). |
| S3 | Cross-source consistency reviewed (`total_reviews == 0` on ConsistencyReport). |
| S4 | Changelog reviewed and approved. |
| S5 | Methodology lead signed off. |
| S6 | Legal confirmed source licences. |
| S7 | Regression test (`compare_editions`) passed. |
| S8 | Load test passed (p95 < 500ms). |
| S9 | Gold-eval precision@1 >= 0.85. |

`approve_release(force=False)` raises `ValueError` if any required item fails.

---

## Rollback history

Published rollbacks appear here. Each entry links to a `RollbackResult` record accessible via `/api/v1/factors/rollback/{rollback_id}`.

- (none published yet — rollbacks will be appended here when they occur)

---

## See also

- [Version pinning](../concepts/version-pinning.md) — pin semantics and rollback protocol.
- [Quality scores](../concepts/quality-scores.md) — S1..S9 and FQS minima.
- [Cut-list anchor (binding)](../../editions/v1-certified-cutlist.md).

---

## File citations

| Piece | File |
|---|---|
| Cut-list / promotion manifest | `docs/editions/v1-certified-cutlist.md` |
| Edition manifest dataclass | `greenlang/factors/edition_manifest.py` |
| Release signoff gate (S1..S9) | `greenlang/factors/quality/release_signoff.py` |
| Release orchestrator | `greenlang/factors/watch/release_orchestrator.py` |
| Changelog drafting | `greenlang/factors/watch/changelog_draft.py`, `cross_edition_changelog.py` |
| Source registry | `greenlang/factors/data/source_registry.yaml` |
