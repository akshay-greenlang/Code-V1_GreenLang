# Migrating from ecoinvent (and operating the ecoinvent connector)

**ecoinvent** is the most widely used commercial LCA database (cut-off, APOS, consequential). It is a **commercial_connector** source — GreenLang provides an integration, but each tenant must bring its own ecoinvent license. This page covers how the connector works, how licensing is enforced, and how to migrate workflows that previously read ecoinvent directly (from a SimaPro project, an OpenLCA file, or the ecoinvent web API).

**Connector implementation:** `greenlang/factors/connectors/ecoinvent.py`.
**License manager:** `greenlang/factors/connectors/license_manager.py`.
**Catalog of licensed sources:** `greenlang/factors/connectors/LICENSED_CONNECTORS.md`.

---

## The big shift

If you were reading ecoinvent **directly** (v3.10 APOS via `.spold` files or the REST API):

- The numbers are the same (ecoinvent activity dataset CO2e totals).
- The **shape** becomes a GreenLang `EmissionFactorRecord` with a gas breakdown, provenance, DQS, and a `license_class = "commercial_connector"`.
- The **methodology guardrail** moves from "whatever you chose in SimaPro/OpenLCA" to "`method_profile` on every call."
- The **output is signed and edition-pinned** — the same activity resolved six months apart is still reproducible.

---

## Licensing flow

1. You hold an ecoinvent license directly (Association or reseller). Confirm the license permits third-party caching or on-the-fly retrieval — ecoinvent's terms vary by subscription tier.
2. Register your license with GreenLang:

   ```bash
   # Per-tenant env var (preferred).
   export GL_FACTORS_LICENSE_ECOINVENT="<your-ecoinvent-api-key>"
   ```

   Or via the dashboard (encrypted at rest via AES-256-GCM through SEC-003; only the SHA-256 hash is in the audit log).
3. The `LicenseManager.resolve_key()` flow (`greenlang/factors/connectors/license_manager.py`) checks, in order:
   1. In-memory cache of prior rotations.
   2. `GL_FACTORS_LICENSE_ECOINVENT_<TENANT>`.
   3. `GL_FACTORS_LICENSE_ECOINVENT`.
4. On a request that routes to ecoinvent without a key, the API returns **HTTP 451 Unavailable For Legal Reasons**. See [licensing-classes](../concepts/licensing-classes.md) for the full 451 shape.

### Key rotation

```python
from greenlang.factors.connectors.license_manager import LicenseManager

mgr = LicenseManager()
mgr.register_key(connector_id="ecoinvent", tenant_id="acme-climate",
                 key="<new-key>", expires_at="2027-12-31T23:59:59Z")
```

Old key stays verifiable for grace-period audit replay; audit log records the rotation event.

---

## What the connector returns

When you resolve an activity that maps to an ecoinvent dataset:

```bash
curl -sS -X POST "$GL_API_BASE/api/v1/factors/resolve-explain" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "X-Factors-Edition: 2027.Q3-product-carbon" \
  -d '{
    "activity": "aluminium_primary_production_cradle_to_gate",
    "method_profile": "product_carbon",
    "jurisdiction": "GLOBAL",
    "reporting_date": "2026-06-01",
    "preferred_sources": ["ecoinvent"],
    "extras": {"ecoinvent_system_model": "APOS"}
  }'
```

Response (truncated):

```json
{
  "factor_id": "EF:ECOINVENT:aluminium_primary_production_rer:v3.10:apos",
  "co2e_per_unit": 16.4,
  "unit": "kg_CO2e_per_kg",
  "source": {
    "organization": "ecoinvent",
    "publication": "ecoinvent v3.10 APOS",
    "year": 2023,
    "methodology": "LCA_ECOINVENT_APOS",
    "ecoinvent_activity_uuid": "abc12345-...",
    "ecoinvent_reference_product": "aluminium, primary, ingot"
  },
  "gas_breakdown": {"co2": 15.9, "ch4": 0.3, "n2o": 0.08, "hfcs": null, "pfcs": 0.12, "sf6": null, "nf3": null},
  "data_quality": {"overall_score": 84, "rating": "good"},
  "license_class": "commercial_connector",
  "redistribution_allowed": false,
  "attribution_required": ["ecoinvent v3.10 APOS (c) ecoinvent Association, Zurich."]
}
```

Key differences vs reading the `.spold` directly:

- **Gas breakdown is exposed.** ecoinvent's internal flows are aggregated by species; the connector lifts the seven CTO-mandated slots.
- **Data quality score is GreenLang-native.** The connector maps ecoinvent's internal uncertainty and Pedigree scores onto the 1-5 DQS dimensions.
- **PFC slot populated** for aluminium-specific C2F6 / CF4 emissions. Do not double-count by also adding a separate PFC refrigerant call.
- `redistribution_allowed: false`. You can compute your product's footprint and publish that aggregate; you cannot publish the raw ecoinvent unit-process value.

---

## System model selection (APOS vs cut-off vs consequential)

Ecoinvent publishes three parallel datasets; the connector honours your choice via `extras.ecoinvent_system_model`:

- `APOS` (default) — Allocation at the Point of Substitution. Balanced approach.
- `CUTOFF` — Cut-off by classification. Recycled content burden-free.
- `CONSEQUENTIAL` — Market-mix, change-oriented.

```json
{"extras": {"ecoinvent_system_model": "CUTOFF"}}
```

If you previously pinned a system model in SimaPro/OpenLCA project files, carry that choice across explicitly.

---

## Version pinning ecoinvent

Each GreenLang edition pins to a specific ecoinvent release (v3.9, v3.10, v3.11). You do not control that pin — the edition does.

Check which ecoinvent version an edition uses:

```bash
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
     "$GL_API_BASE/api/v1/editions/2027.Q3-product-carbon" \
     | jq '.per_source_hashes.ecoinvent, .changelog'
```

When ecoinvent publishes v3.11, a new GreenLang edition ingests it and the old edition stays pinned to v3.10 forever. Your restatements are reproducible.

---

## Migration checklist

1. **Inventory your ecoinvent usage.** Which activities, which system model, which version.
2. **Obtain and register the license key.** `GL_FACTORS_LICENSE_ECOINVENT` or dashboard.
3. **Map activities.** Most ecoinvent activity UUIDs have a GreenLang activity label. For the rest, use the connector's search:

   ```bash
   curl -sS -H "Authorization: Bearer $GL_API_KEY" \
        "$GL_API_BASE/api/v1/factors/search?q=aluminium+primary+production&source=ecoinvent"
   ```

4. **Pick a `method_profile`.** For most ecoinvent usage this is `product_carbon`; for spend-based Scope 3 it is `corporate_scope3`.
5. **Pin an edition.** Confirm the pinned edition uses the ecoinvent release you want (v3.10 APOS, etc.).
6. **Update response parsing.** You get gas breakdowns now — do not rely on CO2e-only; you might be combining with non-ecoinvent factors that have different GWP sets baked in.
7. **Handle 451.** If someone else in your org has a different tenant, they will 451 without their own key.
8. **Respect `redistribution_allowed: false`.** Do not embed raw ecoinvent factor rows in products you sell.

---

## What NOT to do

- **Do not** cache ecoinvent numbers outside the connector and serve them to tenants without keys. This is a license violation ecoinvent takes seriously.
- **Do not** aggregate different system models in one inventory. Pick APOS OR cut-off OR consequential and stick with it.
- **Do not** mix ecoinvent v3.9 and v3.10 within the same reporting period. The underlying LCI changes subtly; the aggregate is not defensible.
- **Do not** apply GWP AR5 to ecoinvent gas flows after they have already been rolled up to CO2e under AR4 (older ecoinvent versions). Use `gas_breakdown` + `gwp_set` instead.

---

## Common pitfalls

- **Missing license key in CI environment.** Your local dev has `GL_FACTORS_LICENSE_ECOINVENT` set; CI does not. Batch jobs in CI hit 451 at 3am. Fix: store the key in CI secrets.
- **Cross-tenant contamination.** A staff user querying another tenant's overlay without realising. `tenant_id` on every request.
- **Assuming ecoinvent is the best source.** Sometimes a country-specific sector average (e.g. a jurisdiction's published Scope 3 tables) is more defensible than a global ecoinvent average. Let the cascade decide; only use `preferred_sources=["ecoinvent"]` when you have a policy reason.

---

## Other licensed connectors

The same flow applies to:

- **IEA** — statistics and country-level electricity factors (`license_class = commercial_connector`).
- **Electricity Maps** — daily grid mixes (`license_class = commercial_connector`).
- **Sphera GaBi** — alternative LCA database (`license_class = commercial_connector`).
- **Thinkstep** — sustainability databases (`license_class = commercial_connector`).

Env vars: `GL_FACTORS_LICENSE_IEA`, `GL_FACTORS_LICENSE_ELECTRICITY_MAPS`, etc. Full inventory in `greenlang/factors/connectors/LICENSED_CONNECTORS.md`.

---

## See also

- [Licensing classes](../concepts/licensing-classes.md)
- [Method packs](../concepts/method-packs.md)
- [Errors — 451 license_required](../api-reference/errors.md)
- [Changelog](./CHANGELOG.md)

---

## File citations

| Piece | File |
|---|---|
| ecoinvent connector | `greenlang/factors/connectors/ecoinvent.py` |
| License manager (resolve / register / revoke / rotate) | `greenlang/factors/connectors/license_manager.py` |
| Licensed connectors inventory | `greenlang/factors/connectors/LICENSED_CONNECTORS.md` |
| IEA / Electricity Maps connectors | `greenlang/factors/connectors/iea.py`, `electricity_maps.py` |
| Audit log of license-key access events | `greenlang/factors/connectors/audit_log.py` |
| Source registry | `greenlang/factors/data/source_registry.yaml` |
