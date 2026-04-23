# Concept — Source

A **source** is an upstream publisher of emission factor data. Every factor record in the catalog is stamped with a `source_id` and a `source_version` that pin the exact release of that publisher's dataset used to derive the row.

## Why sources matter

Auditors verify factors by reference to the publisher; regulators require the publisher cited by name and version. The `source_id` + `source_version` pair is what makes a filing reproducible years later. A record without both is rejected at write time.

## Source catalog

The live source catalog is available at `GET /v1/sources` (see [`api-reference/sources.md`](../api-reference/sources.md)). Each source has:

- `source_id` — unique slug (e.g., `epa_hub`, `egrid`, `desnz_ghg_conversion`, `india_cea_co2_baseline`).
- `authority` — publishing organization (US EPA, UK DESNZ, India CEA, IPCC, SFC GLEC, ...).
- `jurisdiction` — the publisher's geographic coverage.
- `current_version` — latest release tracked (e.g., `2024.1`, `v3.0`, `FY2023-24`).
- `license_name` + `license_url` — e.g., `OGL-UK-v3`, `CC-BY-4.0`, `US-Gov-PD`.
- `redistribution_class` — see [`license_class.md`](license_class.md).
- `attribution_required` + `attribution_text` — the exact string the API response must render when a factor from this source is returned.

## Customer-supplied sources

When a tenant uploads their own primary data (facility-specific emission factor, supplier PCF), the resulting record carries `source_id = tenant:<uuid>`, `redistribution_class = customer_private`, and is visible only to that tenant. See [`license_class.md`](license_class.md) and [`api-reference/sources.md`](../api-reference/sources.md#creating-a-tenant-source).

## Derived sources

Some factors are derived from multiple upstream sources (e.g., a residual-mix factor netted from grid average + certificate surrender statistics). In that case `source_id` names the derivation authority (e.g., `aib_residual_mix_eu`, `beis_uk_residual`) and the provenance chain is recorded in `lineage.raw_record_ref`.

## BYO-credentials sources

Certain commercial publishers (ecoinvent, IEA, Electricity Maps, EC3) do not permit redistribution. GreenLang Factors exposes these through a **connector-only, bring-your-own-credentials** path: the tenant registers their own license key in the dashboard, and GreenLang acts as a transport only. Factor values are not persisted in the shared catalog. See [`licensing.md`](../licensing.md) for the authoritative BYO posture.

## Source versioning

Every new source release (e.g., EPA eGRID 2025 Q1 replaces 2024.1) is ingested as a **new generation of factor records** — never an overwrite. Old `factor_version`s remain retrievable via pinned editions; new resolutions default to the latest `active` row. See [`edition.md`](edition.md).

**See also:** [`factor`](factor.md), [`license_class`](license_class.md), [Source rights matrix](../../legal/source_rights_matrix.md).
