# Concept — License Class

Every factor record carries exactly one `licensing.redistribution_class`. This value governs whether the factor can appear in a bulk export, which tenants are entitled to see it, and what attribution text the API must render alongside it. A single API response MUST NOT mix classes — this is CTO non-negotiable #4.

## The four classes

| Class | Storage | Export | Example sources |
|---|---|---|---|
| `open` | Public bucket. No entitlement check. | Included in Community-tier AND Certified bulk exports. | `epa_hub`, `egrid`, `desnz_ghg_conversion`, `india_cea_co2_baseline`, `australia_nga_factors` |
| `licensed_embedded` | Separate namespace with per-pack entitlement. | API-only. Never in bulk Certified export. | `ghgp_method_refs`, `pact_pathfinder`, `glec_framework`, `pcaf_global_std_v2`, `green_e_residual_mix` |
| `customer_private` | Tenant-scoped namespace. Zero cross-tenant visibility. | Never leaves the tenant. | Tenant overlay factors uploaded via `/v1/sources/tenant` |
| `oem_redistributable` | Separate OEM namespace. | OEM tenants may redistribute to sub-tenants under their upstream contract with GreenLang. | Premium packs with OEM addendum (post-contract) |

## BYO-credentials posture (operationally critical)

Several commercial publishers (ecoinvent, IEA, Electricity Maps, EC3, Green-e pre-contract, GLEC pre-SFBG, TCR) **do not permit GreenLang redistribution** at v1 launch. For these sources GreenLang does NOT bundle factor values into Certified editions. Instead the tenant registers their own license key, and GreenLang resolves through a connector at query time:

- Tenant provides credentials via `factors connector add --source <id>`.
- Factors fetched AT RUNTIME; not persisted in the shared catalog.
- The tenant bears the upstream license; GreenLang is a transport, not a redistributor.
- API response still carries the publisher's required attribution string.
- Audit bundle records the connector call plus tenant credential ID (not the secret).

Contracts in progress upgrade specific sources from BYO to `licensed_embedded` without breaking existing flows. See [`licensing.md`](../licensing.md) and the [source rights matrix](../../legal/source_rights_matrix.md).

## Entitlements

For `licensed_embedded` sources the server checks the caller's entitlement token against the record's `source_id` at every call. Mismatches return `402 Payment Required` (no SKU) or `403 Forbidden` (SKU held but not entitled for this specific source). See [`error-codes.md`](../error-codes.md).

## Attribution

When a record is returned, the server inserts the required attribution string into the response envelope (`licensing.attribution_text`). SDKs surface this as `result.licensing.attribution_text`. Clients MUST render attribution in any human-readable report. Exact strings are reviewed by Legal every 12 months or on source version bump; see [`source_contracts_outreach.md`](../../legal/source_contracts_outreach.md) Part 4.

## What NEVER happens

- An `open` factor is never returned inside a response that also contains a `licensed_embedded` factor; the server splits the call.
- A `customer_private` factor is never visible to another tenant, OEM parent, or GreenLang staff in a query response.
- A `licensed_embedded` factor is never written to a bulk export CSV, Parquet, or signed bundle.

**See also:** [`source`](source.md), [`signed_receipt`](signed_receipt.md), [Licensing](../licensing.md), [Legal binder](../../launch/legal_source_rights_binder.md).
