# API — `GET /v1/sources` and `GET /v1/sources/{source_id}`

Enumerate upstream publishers and their rights posture. This catalog is what the resolver binds every factor record to via `source_id` + `source_version`.

**Authoritative spec:** [`docs/api/factors-v1.yaml`](../../api/factors-v1.yaml) (`operationId: listSources`, `getSource`).

---

## List — `GET /v1/sources`

```bash
curl "https://api.greenlang.io/v1/sources" \
  -H "Authorization: Bearer $GL_API_KEY"
```

### Query parameters

| Param | Notes |
|---|---|
| `jurisdiction` | Filter by ISO-3166 alpha-2 (`US`, `IN`) or region (`EU`, `XX` for global). |
| `redistribution_class` | `open`, `licensed_embedded`, `customer_private`, `oem_redistributable`. |
| `status` | `active`, `deprecated`. |
| `authority` | Free-text substring match against publisher name. |

### Response

```json
{
  "items": [
    {
      "source_id": "epa_hub",
      "authority": "US EPA",
      "dataset_name": "EPA GHG Emission Factors Hub",
      "jurisdiction": "US",
      "current_version": "2024",
      "publication_year": 2024,
      "license_name": "US-Gov-PD",
      "license_url": "https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
      "redistribution_class": "open",
      "attribution_required": true,
      "attribution_text": "U.S. Environmental Protection Agency, GHG Emission Factors Hub.",
      "v1_gate_status": "Safe-to-Certify"
    },
    { "source_id": "india_cea_co2_baseline", "...": "..." }
  ]
}
```

`v1_gate_status` comes from [`docs/legal/source_rights_matrix.md`](../../legal/source_rights_matrix.md) and tells a partner whether a source is Safe-to-Certify, Needs-Legal-Review, or Blocked-Contract-Required for a Certified edition.

---

## Get — `GET /v1/sources/{source_id}`

```bash
curl "https://api.greenlang.io/v1/sources/india_cea_co2_baseline" \
  -H "Authorization: Bearer $GL_API_KEY"
```

Single source document, including the full license chain and the attribution string to render in UIs and reports.

---

## Python / TypeScript

```python
for source in client.list_sources(redistribution_class="open"):
    print(source.source_id, source.current_version)
```

```ts
const { items } = await client.listSources({ redistributionClass: "open" });
```

---

## Creating a tenant source

Tenants uploading primary data (facility-specific factor, supplier PCF) create a tenant-scoped source:

```bash
curl -X POST "https://api.greenlang.io/v1/sources/tenant" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "Acme Cement - Plant 3 clinker factor",
    "authority": "Acme Cement internal metrology",
    "redistribution_class": "customer_private",
    "license_name": "tenant_private",
    "current_version": "2026.Q1"
  }'
```

Returns `source_id` of the form `tenant:<uuid>`. The record can now be used as the `source_id` on new factor records created via the catalog admin API. These records carry `licensing.redistribution_class == "customer_private"` and are never visible cross-tenant. See [`concepts/license_class.md`](../concepts/license_class.md).

---

## BYO-credentials sources

For publishers that do not permit redistribution at v1 launch (ecoinvent, IEA, Electricity Maps, EC3, pre-contract Green-e / GLEC / TCR), the tenant registers their own credentials:

```bash
gl-factors connector add --source ecoinvent \
  --credential-id "$ECOINVENT_LICENSE_ID"
```

The connector forwards resolution requests at query time; no factor values are persisted in the shared catalog. See [`licensing.md`](../licensing.md) and [CLI reference](../sdks/cli.md).

---

## Errors

| Status | Code | When |
|---|---|---|
| 401 | `unauthorized` | Missing token. |
| 403 | `forbidden` | Attempting to read another tenant's `tenant:<uuid>` source. |
| 404 | `not_found` | Unknown `source_id`. |

See [`error-codes.md`](../error-codes.md).

## Related

- [`concepts/source.md`](../concepts/source.md), [`concepts/license_class.md`](../concepts/license_class.md).
- [Source rights matrix](../../legal/source_rights_matrix.md), [Contract outreach](../../legal/source_contracts_outreach.md).
