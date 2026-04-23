# API — Quality surface

Expose the Factor Quality Score (FQS) and its five components for a given factor. Used by the three-label dashboard and by callers building their own data-quality gates.

**Authoritative spec:** [`docs/api/factors-v1.yaml`](../../api/factors-v1.yaml).

See [`concepts/quality_score.md`](../concepts/quality_score.md) for the scoring model.

---

## `GET /v1/quality/{factor_id}`

```bash
FID=$(python -c "import urllib.parse; print(urllib.parse.quote('EF:IN:grid:CEA:FY2024-25:v1', safe=''))")

curl "https://api.greenlang.io/v1/quality/$FID?version=1.0.0" \
  -H "Authorization: Bearer $GL_API_KEY"
```

### Response

```json
{
  "factor_id": "EF:IN:grid:CEA:FY2024-25:v1",
  "factor_version": "1.0.0",
  "composite_fqs_0_100": 82.0,
  "components": {
    "temporal_score": 4,
    "geographic_score": 5,
    "technology_score": 4,
    "verification_score": 4,
    "completeness_score": 4
  },
  "weights": {
    "temporal_score": 0.25,
    "geographic_score": 0.25,
    "technology_score": 0.20,
    "verification_score": 0.15,
    "completeness_score": 0.15
  },
  "band": "medium",
  "rationale_per_component": {
    "temporal_score": "Source valid_from 2024-04-01 .. valid_to 2025-03-31; reporting period 2026-12-31 is 1 year outside validity. Score=4.",
    "geographic_score": "Factor jurisdiction=IN matches activity jurisdiction=IN exactly. Score=5.",
    "technology_score": "Grid-average factor; no subregion granularity for India. Score=4.",
    "verification_score": "Publisher-verified by CEA. Score=4.",
    "completeness_score": "CO2 + CH4 + N2O covered; biogenic + F-gases not applicable. Score=4."
  }
}
```

---

## `GET /v1/quality/{factor_id}/distribution`

Return the FQS distribution across all factor_versions of this ID — used to detect score regressions over time.

```json
{
  "factor_id": "EF:IN:grid:CEA:FY2024-25:v1",
  "versions": [
    { "factor_version": "0.9.0", "composite_fqs_0_100": 74.0, "valid_from": "2023-04-01" },
    { "factor_version": "1.0.0", "composite_fqs_0_100": 82.0, "valid_from": "2024-04-01" }
  ]
}
```

---

## `POST /v1/quality/bulk`

Fetch FQS for up to 1,000 factor IDs at once.

```bash
curl -X POST "https://api.greenlang.io/v1/quality/bulk" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "factor_ids": [
      {"factor_id": "EF:IN:grid:CEA:FY2024-25:v1", "factor_version": "1.0.0"},
      {"factor_id": "EF:US:grid:eGRID-SERC:2024:v1", "factor_version": "1.0.0"}
    ]
  }'
```

---

## Bands

| Band | Range |
|---|---|
| `high` | `composite_fqs_0_100 >= 80` |
| `medium` | `60 <= composite_fqs_0_100 < 80` |
| `low` | `composite_fqs_0_100 < 60` |

The operator console's three-label dashboard groups by band; customers building internal gates typically require `high` for regulated disclosures (CBAM, CSRD) and accept `medium` for exploratory reporting.

---

## Python / TypeScript

```python
q = client.get_quality("EF:IN:grid:CEA:FY2024-25:v1", version="1.0.0")
print(q.composite_fqs_0_100, q.band)
```

```ts
const q = await client.getQuality({
  factorId: "EF:IN:grid:CEA:FY2024-25:v1",
  factorVersion: "1.0.0",
});
```

---

## Errors

| Status | Code | When |
|---|---|---|
| 401 | `unauthorized` | Missing token. |
| 404 | `not_found` | Unknown factor ID / version. |

## Related

- [`concepts/quality_score.md`](../concepts/quality_score.md), [`concepts/factor.md`](../concepts/factor.md).
- [`/resolve`](resolve.md), [`/explain`](explain.md).
