# API — `POST /v1/factors/{factor_id}/explain`

Return the full derivation trace for a factor: every tier the resolver walked, every candidate considered, why the chosen factor won, the assumptions applied, the method pack that bound the selection, and an auditor-grade narrative paragraph.

**Authoritative spec:** [`docs/api/factors-v1.yaml`](../../api/factors-v1.yaml) (`operationId: explainFactor`).

This endpoint is first-class (CTO non-negotiable #3): fallback logic is never hidden; every `/resolve` call can be re-run via `/explain` to expose its internals.

---

## Endpoint

```
POST https://api.greenlang.io/v1/factors/{factor_id}/explain
Authorization: Bearer <key>
Content-Type: application/json
```

`{factor_id}` is the chosen factor ID from a prior `/resolve` response, URL-encoded.

---

## Request body

Same shape as `/resolve` — the explain endpoint re-runs the resolution under the same inputs so the trace reflects the exact candidate pool.

```json
{
  "method_profile": "corporate_scope2_location_based",
  "quantity": 12500,
  "unit": "kWh",
  "jurisdiction": "IN",
  "valid_at": "2026-12-31"
}
```

---

## curl

```bash
FID=$(python -c "import urllib.parse; print(urllib.parse.quote('EF:IN:grid:CEA:FY2024-25:v1', safe=''))")

curl -X POST "https://api.greenlang.io/v1/factors/$FID/explain" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "method_profile": "corporate_scope2_location_based",
    "quantity": 12500,
    "unit": "kWh",
    "jurisdiction": "IN",
    "valid_at": "2026-12-31"
  }'
```

## Python

```python
explanation = client.explain(
    factor_id="EF:IN:grid:CEA:FY2024-25:v1",
    method_profile="corporate_scope2_location_based",
    quantity=12500,
    unit="kWh",
    jurisdiction="IN",
    valid_at="2026-12-31",
)

for tier in explanation.cascade:
    print(tier.rank, tier.label, tier.candidates_evaluated)

print(explanation.audit_text)
```

## TypeScript

```ts
const explanation = await client.explain({
  factorId: "EF:IN:grid:CEA:FY2024-25:v1",
  methodProfile: "corporate_scope2_location_based",
  quantity: 12500,
  unit: "kWh",
  jurisdiction: "IN",
  validAt: "2026-12-31",
});
```

---

## Response (200 OK) — key fields

```json
{
  "chosen_factor": { "factor_id": "...", "factor_version": "1.0.0" },
  "cascade": [
    { "rank": 1, "label": "customer_override", "candidates_evaluated": 0, "outcome": "no_candidates" },
    { "rank": 2, "label": "supplier_specific",  "candidates_evaluated": 0, "outcome": "no_candidates" },
    { "rank": 3, "label": "facility_specific",  "candidates_evaluated": 0, "outcome": "no_candidates" },
    { "rank": 4, "label": "utility_or_grid_subregion", "candidates_evaluated": 1,
      "outcome": "chosen", "winner_factor_id": "EF:IN:grid:CEA:FY2024-25:v1",
      "reason": "CEA is the authoritative publisher for India grid intensity; subregion granularity not available" },
    { "rank": 5, "label": "country_or_sector_average", "candidates_evaluated": 0, "outcome": "skipped_chosen_upstream" }
  ],
  "assumptions": [
    "FY2024-25 CEA baseline used as best-available proxy for FY2026-27 reporting",
    "T&D losses NOT included (busbar basis)",
    "AR6 100-yr GWPs"
  ],
  "pack_rules_applied": [
    "selection.allowed_families includes 'electricity'",
    "boundary.include_transmission_losses = false",
    "fallback.cannot_resolve_action = raise_no_safe_match"
  ],
  "audit_text": "Under the GHG Protocol Scope 2 Guidance (2015) §6.1 location-based method, the chosen factor is the India national grid intensity published by the Central Electricity Authority (Government of India), CO2 Baseline Database v20.0 (FY2024-25). No supplier-specific, facility-specific, or utility-level data was provided, so the resolver fell back to the country-level factor (tier 4). Transmission and distribution losses are not included in this busbar factor.",
  "audit_text_draft": false,
  "signed_receipt": { "...": "..." }
}
```

The `cascade[]` array is the full 7-step trace (tiers 1 through 7). Tiers after the winner are marked `skipped_chosen_upstream`. Tiers with zero candidates show `no_candidates`.

---

## When to use `/explain` vs `/resolve`

| Use `/resolve` when | Use `/explain` when |
|---|---|
| You just need the number and a receipt | You need the cascade trace for audit |
| Streaming resolution in a pipeline | Building an audit bundle for third-party assurance |
| CLI / UI where receipt verification is enough | Regulator asked "why did you pick this factor" |

`/resolve` already carries `assumptions[]`, `fallback_rank`, `audit_text`, and `signed_receipt`. `/explain` adds the full per-tier cascade.

---

## Related

- [`/resolve`](resolve.md) — runs the cascade and returns the winner.
- [`concepts/method_pack.md`](../concepts/method_pack.md), [`concepts/signed_receipt.md`](../concepts/signed_receipt.md).
- [`docs/specs/audit_text_template_policy.md`](../../specs/audit_text_template_policy.md) — draft-banner behaviour for unapproved templates.
