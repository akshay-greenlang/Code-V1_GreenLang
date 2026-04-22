# Licensing Classes

Every factor in the catalog carries a `license_class` that determines whether you can see it, whether you can redistribute it, and whether a specific response may mix it with other factors. Licensing is enforced at **both** the connector layer (ingest time) and the serving layer (response time) — there is no single enforcement point that a bug or misconfiguration can bypass.

**Connector-side enforcement:** `greenlang/factors/connectors/license_manager.py`
**Serving-side enforcement:** `greenlang/factors/quality/license_scanner.py`
**Homogeneity rule:** `greenlang/data/canonical_v2.py::enforce_license_class_homogeneity`
**Tier gates:** `greenlang/factors/tier_enforcement.py`

---

## The four classes

| `license_class` (plus category family) | Can you read it? | Can you redistribute the numbers? | Typical sources |
|---|---|---|---|
| **open** — `open`, `public_us_government`, `uk_open_government`, `greenlang_terms` | Yes, on every tier. | Yes, with attribution. | EPA eGRID & GHG Hub, DESNZ / DEFRA, Australian NGA, Japan METI, India CEA, IPCC, GreenLang curated built-ins. |
| **restricted** — `restricted` | Yes on Pro+; visible but gated on Community. | No — summaries and sector aggregates OK, but individual factor rows cannot appear in redistributable product. | Green-e Residual Mix, some ISO / industry consortium publications. |
| **licensed** — `commercial_connector` | Only when your tenant has a valid license key for the source. The connector returns 402 if no key. | No — output bound to your tenant, cannot be redistributed. | ecoinvent, IEA statistics, Electricity Maps, Sphera, Thinkstep. |
| **customer_private** | Only within the owning tenant's scope. | No. | Supplier-provided PCFs you loaded via `tenant_overlay`. |

The full set of `license_class` strings registered in `source_registry.yaml` includes: `open`, `public_us_government`, `uk_open_government`, `greenlang_terms`, `restricted`, `commercial_connector`, `customer_private`. They map onto the four families above — developers usually reason about the four families, methodology leads reason about the specific string.

---

## Licensed sources and the 451 response

When you resolve a factor that routes to a `commercial_connector` source and your tenant does not have a valid license key configured for that connector, the API returns **HTTP 451 Unavailable For Legal Reasons**.

Example:

```bash
curl -X POST "$GL_API_BASE/api/v1/factors/resolve-explain" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "activity": "concrete CEM III cradle-to-gate",
    "method_profile": "product_carbon",
    "jurisdiction": "DE",
    "reporting_date": "2026-06-01",
    "preferred_sources": ["ecoinvent"]
  }'
```

```http
HTTP/1.1 451 Unavailable For Legal Reasons
Content-Type: application/json
X-GreenLang-License-Required: ecoinvent

{
  "error": "license_required",
  "message": "Resolution routed to licensed source 'ecoinvent' but no active license key is configured for tenant 'acme-climate'.",
  "details": {
    "connector_id": "ecoinvent",
    "license_class": "commercial_connector",
    "resolution": "Configure GL_FACTORS_LICENSE_ECOINVENT env var, or contact sales@greenlang.io for a connector key.",
    "alternate_resolution": "Re-run with 'preferred_sources' omitted to fall back to open-licensed sources."
  }
}
```

### How to upgrade

1. **Acquire a license directly** from the source (ecoinvent, IEA, Sphera, ...).
2. **Register it with GreenLang** via the dashboard or env var: `GL_FACTORS_LICENSE_ECOINVENT=<your-key>`.
3. The connector will resolve it on first request (see `LicenseManager.resolve_key` in `greenlang/factors/connectors/license_manager.py`). Key hashes (never plaintext) land in the audit log.
4. Rotate with `LicenseManager.register_key` + `revoke_key`.

### Community vs Pro vs Enterprise on licensed sources

- **Community** — Licensed sources are invisible; they are never candidates in the cascade. `include_connector=true` is force-clamped to `false`.
- **Pro** — Licensed sources visible but require a connector key. Without a key, 451 Unavailable For Legal Reasons.
- **Consulting / Enterprise** — Same as Pro plus `include_connector=true` is respected (see `tier_enforcement.enforce_tier_on_request`).

---

## Redistribution rules

Every factor response includes `redistribution_allowed: true | false`. The responsibility for honouring it is on the consumer. The API will tell you, but it will not enforce it downstream.

Rule of thumb:

- **`redistribution_allowed: true`** — you can embed the numeric factor in a product you sell. Attribution requirement lives in `license_info.attribution_text`.
- **`redistribution_allowed: false`** — you can use the number internally to compute your own emissions but cannot ship the raw factor row to a third party. Downstream derived aggregates (e.g. your company's total Scope 2 in kgCO2e) are fine.

### License-class homogeneity (cross-factor rule)

A single resolution cannot mix `open` and `restricted`/`commercial_connector` factors across cascade steps. Enforced in `enforce_license_class_homogeneity` (`greenlang/data/canonical_v2.py`). If step 5 returns an open factor and step 2 returns a restricted one, the restricted one wins only if the downstream result fully honours the restricted terms — otherwise the engine drops the restricted alternate.

This matters for **redistributable computation bundles** where the final output must be uniformly redistributable.

---

## The license scanner (serving side)

`greenlang/factors/quality/license_scanner.py` runs on every response and:

1. Checks that every cited factor carries a `license_class`.
2. Verifies the tenant's tier is eligible for that class.
3. Verifies any `commercial_connector` factor has an active key.
4. Attaches a `license_summary` block to the response:

```json
"license_summary": {
  "classes_present": ["open", "public_us_government"],
  "redistribution_allowed": true,
  "attribution_required": [
    "EPA eGRID 2024, US EPA (public domain).",
    "DESNZ GHG conversion factors 2026 (UK Open Government Licence v3)."
  ]
}
```

Always surface `attribution_required` in any artefact you publish. That is the single most common licence-breach reported to us.

---

## Denied-license audit trail

Every time a licensed source is requested without a key, or a restricted row is emitted to a Community tenant, the `license_manager` appends an immutable audit entry (see `LicenseManager._audit_log`). This is what you hand to a customer's procurement team during a vendor review: "Yes, we tried to query ecoinvent on 2026-11-03 at 09:14:22Z; no key was present; 451 returned; fallback path selected open-licensed alternative."

---

## See also

- [Authentication](../api-reference/authentication.md) — JWT claims carry tier.
- [Errors](../api-reference/errors.md) — full 401 / 402 / 403 / 451 semantics.
- [Migration from ecoinvent](../migration/from-ecoinvent.md) — connector walkthrough.
- [Resolution cascade](./resolution-cascade.md) — how license-class homogeneity is enforced per-resolution.

---

## File citations

| Piece | File |
|---|---|
| License class enforcement at ingest | `greenlang/factors/connectors/license_manager.py` |
| Licensed connectors inventory | `greenlang/factors/connectors/LICENSED_CONNECTORS.md` |
| ecoinvent / IEA / Electricity Maps connectors | `greenlang/factors/connectors/ecoinvent.py`, `iea.py`, `electricity_maps.py` |
| Serving-side license scanner | `greenlang/factors/quality/license_scanner.py` |
| License-class homogeneity | `greenlang/data/canonical_v2.py::enforce_license_class_homogeneity` |
| Tier gate / visibility clamp | `greenlang/factors/tier_enforcement.py` |
| Source registry (one row per source) | `greenlang/factors/data/source_registry.yaml` |
