# Concept — Edition

An **edition** is an immutable, signed snapshot of the entire factor catalog at a point in time. Editions exist so an organization that filed its FY2026 inventory using edition `builtin-v1.2.0` can re-run the same inventory in FY2031 and get bit-identical numbers — even after the underlying EPA / DEFRA / IEA sources have released newer data.

## What an edition contains

- Every `active`, `deprecated`, and `retired` factor record as of the cut date.
- The method pack registry (every certified pack, pinned at its version).
- The source catalog (every `source_id` and the `source_version` it was pinned to).
- The GWP registry (coefficients for every supported `gwp_set`).
- An Ed25519 signature over a Merkle root of the catalog contents.

## Edition identifiers

Editions are named `<channel>-v<semver>`. Examples:

| `edition_id` | Meaning |
|---|---|
| `builtin-v1.0.0` | First Certified GA edition. |
| `builtin-v1.1.0` | Minor: added jurisdictions, new sources, backward-compatible factor changes. |
| `builtin-v1.0.1` | Patch: audit-text and doc fixes only; no factor value changes. |
| `preview-v1.1.0-rc.1` | Release candidate published for partner validation. |
| `tenant:<uuid>-v12` | Tenant-scoped edition (customer overlays). |

## Pinning in calls

Callers pin an edition three ways:

1. **Header (preferred):** `X-GreenLang-Edition: builtin-v1.0.0`
2. **Query / body:** `?edition=builtin-v1.0.0` or `"edition": "builtin-v1.0.0"` in the request body.
3. **SDK client pin:** `client.pin_edition("builtin-v1.0.0")` returns a new client pinned for every call.

If no pin is supplied, the server returns the latest Certified edition and echoes it in `X-GreenLang-Edition`. Clients SHOULD record the echoed edition for reproducibility.

## Drift rejection

If the caller pins `builtin-v1.0.0` and the server cannot serve that edition (retired / not yet synced / wrong cluster), the server returns `409 Conflict` with `error_code: edition_mismatch`. The SDKs map this to `EditionMismatchError`.

## Certified, Preview, Connector-only

Editions carry a channel:

- **Certified** — full SLA; eligible for regulated disclosures; immutable after cut.
- **Preview** — same shape but includes `preview`-status factors and licensed-connector sources that have not cleared Certified gating. Clients MUST NOT file regulated disclosures against a preview edition.
- **Connector-only** — no bundled factor values; resolution routes through live BYO-credentials connectors (e.g., ecoinvent, IEA). Reproducibility depends on the customer's upstream subscription.

See [`licensing.md`](../licensing.md) for the carve-out posture and [`coverage.md`](../coverage.md) for what each channel counts.

## Reproducibility guarantee

Two identical `/resolve` calls against the same `edition_id` with the same inputs MUST return byte-identical numerator / denominator / emissions / `chosen_factor.factor_id` / `chosen_factor.factor_version`. Non-deterministic fields (request ID, receipt timestamp) are excluded from the guarantee. This is tested by the gold-set evaluation gate in CI.

**See also:** [`factor`](factor.md), [`signed_receipt`](signed_receipt.md), [API `/editions`](../api-reference/releases.md).
