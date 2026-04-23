# API — GraphQL

GreenLang Factors exposes a GraphQL surface at `POST /v1/graphql` for callers who need flexible projections over the catalog. The REST API remains the canonical surface for `/resolve`, `/explain`, bulk export, and webhooks; GraphQL is the right choice for dashboards, Explorer UIs, and federated queries.

**Authoritative spec:** [`docs/api/factors-v1.yaml`](../../api/factors-v1.yaml).

---

## Endpoint

```
POST https://api.greenlang.io/v1/graphql
Authorization: Bearer <key>
Content-Type: application/json
```

Introspection is enabled for authenticated callers. Fetch the schema via:

```bash
curl -X POST "https://api.greenlang.io/v1/graphql" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ __schema { types { name } } }"}'
```

---

## Example — Search + project

```graphql
query FactorsByJurisdiction($country: String!) {
  factors(filters: { country: $country, family: ELECTRICITY, status: ACTIVE }) {
    totalCount
    edges {
      node {
        factorId
        factorVersion
        factorName
        jurisdiction { country region gridRegion }
        quality { compositeFqs0_100 components { temporalScore geographicScore } }
        source { sourceId currentVersion attributionText }
      }
    }
  }
}
```

**curl:**

```bash
curl -X POST "https://api.greenlang.io/v1/graphql" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "query($country: String!) { factors(filters:{country:$country,family:ELECTRICITY,status:ACTIVE}){ totalCount edges { node { factorId factorName } } } }",
    "variables": { "country": "IN" }
  }'
```

---

## Example — Resolve with full envelope

```graphql
mutation Resolve($input: ResolveInput!) {
  resolve(input: $input) {
    chosenFactor { factorId factorVersion releaseVersion }
    emissions { co2eKg gwpBasis }
    quality { compositeFqs0_100 }
    fallbackRank
    assumptions
    auditText
    signedReceipt { receiptId signature verificationKeyHint alg payloadHash }
  }
}
```

Variables:

```json
{
  "input": {
    "factorFamily": "ELECTRICITY",
    "quantity": 12500,
    "unit": "kWh",
    "methodProfile": "CORPORATE_SCOPE2_LOCATION_BASED",
    "jurisdiction": "IN",
    "validAt": "2026-12-31"
  }
}
```

---

## Schema highlights

| Top-level type | REST equivalent |
|---|---|
| `Query.factors(filters, sort, offset, limit)` | `POST /v1/factors/search` |
| `Query.factor(id, version)` | `GET /v1/factors/{id}` |
| `Query.sources(filters)` | `GET /v1/sources` |
| `Query.methodPacks()` | `GET /v1/method-packs` |
| `Query.editions()` / `Query.currentEdition` | `GET /v1/editions` / `/editions/current` |
| `Mutation.resolve(input)` | `POST /v1/factors/resolve` |
| `Mutation.batchResolve(items)` | `POST /v1/factors/batch-resolve` |

Enums (`FactorFamily`, `MethodProfile`, `RedistributionClass`, `GwpSet`) mirror the REST schemas verbatim. See [`schema.md`](../schema.md) for the canonical record shape.

---

## Edition pinning

Pass `X-GreenLang-Edition` in request headers, OR set `editionId` on the `resolve` input. The response object always carries `editionId`, echoing what was served.

---

## Rate limiting

GraphQL calls count against the same 100 req/min per key budget as REST. The server applies a per-query complexity budget (max 10,000 cost units). Deep nested projections can exceed it; use pagination and split queries where needed.

---

## Errors

GraphQL errors follow the spec: non-200 HTTP only on auth/transport failures; per-field errors appear in the `errors[]` array.

```json
{
  "data": { "resolve": null },
  "errors": [
    {
      "message": "No candidate satisfies pack rules for method_profile=eu_cbam, jurisdiction=XX",
      "extensions": { "code": "factor_cannot_resolve_safely",
                      "http_status": 422, "pack_id": "eu_cbam" }
    }
  ]
}
```

See [`error-codes.md`](../error-codes.md) for the mapping.

## Related

- REST endpoints: [`/resolve`](resolve.md), [`/explain`](explain.md), [`/factors`](factors.md).
- [`sdks/python.md`](../sdks/python.md), [`sdks/typescript.md`](../sdks/typescript.md).
