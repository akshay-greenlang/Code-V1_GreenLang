# API — `GET /v1/editions` and bulk export

Editions are immutable, signed catalog snapshots. This endpoint lets callers enumerate them, fetch metadata, and stream a bulk export of the Open-class subset.

**Authoritative spec:** [`docs/api/factors-v1.yaml`](../../api/factors-v1.yaml) (`operationId: listEditions`, `getCurrentEdition`, `bulkExport`).

---

## List — `GET /v1/editions`

```bash
curl "https://api.greenlang.io/v1/editions" \
  -H "Authorization: Bearer $GL_API_KEY"
```

### Response

```json
{
  "items": [
    {
      "edition_id": "builtin-v1.0.0",
      "channel": "certified",
      "cut_at": "2026-05-01T00:00:00Z",
      "factor_count": 48213,
      "source_count": 24,
      "method_pack_count": 14,
      "signature": "base64url(Ed25519-sig-over-merkle-root)",
      "verification_key_hint": "jwk-2026Q2-primary"
    },
    {
      "edition_id": "builtin-v1.1.0",
      "channel": "certified",
      "cut_at": "2026-08-01T00:00:00Z",
      "factor_count": 54107
    },
    {
      "edition_id": "preview-v1.2.0-rc.1",
      "channel": "preview",
      "cut_at": "2026-10-15T00:00:00Z"
    }
  ]
}
```

Channels: `certified`, `preview`, `connector_only`. See [`concepts/edition.md`](../concepts/edition.md).

---

## Current — `GET /v1/editions/current`

```bash
curl "https://api.greenlang.io/v1/editions/current" \
  -H "Authorization: Bearer $GL_API_KEY"
```

Returns the edition the server serves when no pin is supplied. Latest Certified by default.

---

## Pinning in other calls

- Header: `X-GreenLang-Edition: builtin-v1.0.0`
- Body: `"edition": "builtin-v1.0.0"`
- SDK: `client.pin_edition("builtin-v1.0.0")`

See [`concepts/edition.md`](../concepts/edition.md#drift-rejection) for how drift is handled.

---

<a id="bulk-export"></a>
## Bulk export — `GET /v1/editions/{edition_id}/bulk-export`

Streams a compressed Parquet / JSON Lines archive containing every **`open`**-class factor in the edition. `licensed_embedded`, `oem_redistributable`, and `customer_private` factors are NEVER included.

```bash
curl -o builtin-v1.0.0.open.parquet \
  "https://api.greenlang.io/v1/editions/builtin-v1.0.0/bulk-export?format=parquet" \
  -H "Authorization: Bearer $GL_API_KEY"
```

### Query parameters

| Param | Default | Notes |
|---|---|---|
| `format` | `parquet` | One of `parquet`, `jsonl`, `csv`. |
| `families` | all | Comma-separated `factor_family` filter. |

### Output manifest

The export ships with a sibling `manifest.json` that records:

- Edition ID + Ed25519 signature
- SHA-256 per output file
- Attribution block covering every unique `source_id` in the export

Re-derivation test: re-parsing the manifest + the bulk file reproduces the hash in `/v1/editions/{id}`.signature field. See [`concepts/signed_receipt.md`](../concepts/signed_receipt.md).

---

## Python / TypeScript

```python
for ed in client.list_editions():
    print(ed.edition_id, ed.channel, ed.factor_count)

latest = client.get_current_edition()
```

```ts
const editions = await client.listEditions();
```

---

## Errors

| Status | Code | When |
|---|---|---|
| 401 | `unauthorized` | Missing token. |
| 403 | `forbidden` | Bulk export scope missing from key. |
| 404 | `not_found` | Unknown edition. |
| 410 | `edition_retired` | Retired edition; retrievable only for audit reconstruction. |

## Related

- [`concepts/edition.md`](../concepts/edition.md), [`concepts/signed_receipt.md`](../concepts/signed_receipt.md).
- [Licensing](../licensing.md) for why `licensed_embedded` is never in a bulk export.
