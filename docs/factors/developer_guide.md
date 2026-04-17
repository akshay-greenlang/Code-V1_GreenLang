# GreenLang Factors API — Developer Guide

## Quickstart (5-Minute Integration)

### 1. Get an API Key

Request an API key from your GreenLang account dashboard or contact sales@greenlang.io.

### 2. Install the SDK

**Python:**
```bash
pip install greenlang-factors-sdk
```

**TypeScript/JavaScript:**
```bash
npm install @greenlang/factors-sdk
```

### 3. Make Your First Query

**Python:**
```python
from greenlang.factors.sdk import FactorsClient, FactorsConfig

client = FactorsClient(FactorsConfig(
    base_url="https://api.greenlang.io/api/v1",
    api_key="gl_your_api_key_here",
))

# Search for a diesel emission factor
results = client.search("diesel combustion US Scope 1")
for f in results.get("factors", []):
    print(f"{f['factor_id']}: {f['co2e_per_unit']} {f.get('unit', 'kg CO2e')}")
```

**TypeScript:**
```typescript
import { FactorsClient } from '@greenlang/factors-sdk';

const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io/api/v1',
  apiKey: 'gl_your_api_key_here',
});

const results = await client.search('diesel combustion US Scope 1');
console.log(results);
```

**cURL:**
```bash
curl -H "Authorization: Bearer gl_your_api_key_here" \
     "https://api.greenlang.io/api/v1/factors/search?q=diesel+US+scope+1&limit=5"
```

---

## Authentication

### API Key (Recommended for server-to-server)

Include your API key in the `Authorization` header:
```
Authorization: Bearer gl_xxxxxxxxxx
```

### JWT Token (For interactive/frontend apps)

Obtain a JWT from the GreenLang auth endpoint:
```bash
POST /auth/token
Content-Type: application/json
{"client_id": "...", "client_secret": "..."}
```

Include the token:
```
Authorization: Bearer eyJhbG...
```

### Tier Entitlements

| Tier | Certified | Preview | Connector | Export | Audit Bundle |
|------|-----------|---------|-----------|--------|--------------|
| Community | Yes | No | No | 1,000 rows | No |
| Pro | Yes | Yes | No | 10,000 rows | No |
| Enterprise | Yes | Yes | Yes | 100,000 rows | Yes |

Your tier is determined by your API key or JWT claims. Requesting visibility
beyond your tier is silently clamped (no error, just filtered results).

---

## Edition Pinning (Reproducibility)

Every factor query is scoped to an **edition** — an immutable snapshot of the catalog.

### Why Pin Editions?

- **Audit reproducibility**: Same query + same edition = same result, always
- **Regulatory defense**: Prove which factors you used and when
- **Change management**: Upgrade editions on your schedule, not ours

### How to Pin

**HTTP header (recommended):**
```
X-Factors-Edition: 2026.04.0
```

**Query parameter:**
```
GET /api/v1/factors?edition=2026.04.0&q=diesel
```

**SDK config:**
```python
client = FactorsClient(FactorsConfig(
    base_url="https://api.greenlang.io/api/v1",
    api_key="gl_...",
    edition="2026.04.0",  # Pin to this edition
))
```

### Resolution Order

1. `GL_FACTORS_FORCE_EDITION` env var (rollback override)
2. `X-Factors-Edition` HTTP header
3. `?edition=` query parameter
4. Default (latest stable edition)

### Listing Available Editions

```bash
GET /api/v1/editions
```

Returns: `[{edition_id, status, label, manifest_hash}, ...]`

### Comparing Editions

```bash
GET /api/v1/editions/compare?left=2026.03.0&right=2026.04.0
```

Returns: `{added_factor_ids, removed_factor_ids, changed_factor_ids, unchanged_count}`

---

## Factor Search Cookbook

### Basic Text Search
```
GET /api/v1/factors/search?q=natural+gas+combustion&limit=10
```

### Filter by Geography
```
GET /api/v1/factors/search?q=electricity&geography=US-CA&limit=10
```

### Advanced Search (v2 POST)
```bash
POST /api/v1/factors/search/v2
Content-Type: application/json

{
  "query": "diesel",
  "geography": "US",
  "scope": "1",
  "dqs_min": 3.5,
  "sort_by": "dqs_score",
  "sort_order": "desc",
  "offset": 0,
  "limit": 20
}
```

### Get Facets (for filter UIs)
```
GET /api/v1/factors/search/facets?include_preview=false
```

Returns: `{facets: {geography: {"US": 150, ...}, scope: {"1": 200, ...}, ...}}`

### Common Query Patterns

| Use Case | Query |
|----------|-------|
| Scope 1 fuels | `q=combustion scope 1&scope=1` |
| US electricity grid | `q=electricity grid&geography=US` |
| UK conversion factors | `q=DESNZ&source_id=desnz_uk` |
| CBAM products | `q=steel cement aluminum&source_id=eu_cbam` |
| Business travel | `q=flight business travel&scope=3` |
| Freight transport | `q=freight tonne-km&scope=3` |

---

## Match API (Activity-to-Factor)

The Match API takes a natural language activity description and returns the best matching emission factors.

```bash
POST /api/v1/factors/match
Content-Type: application/json

{
  "activity_description": "Burned 10,000 gallons of diesel fuel at our manufacturing plant in Texas",
  "geography": "US-TX",
  "scope": "1",
  "limit": 5
}
```

**Response includes:**
- `matches`: Ranked list of factors with match scores
- `best_match`: Top recommendation
- `confidence`: How confident the system is in the match

### How Matching Works

1. **Facet filter**: Narrow by geography, scope, fuel_type
2. **Lexical search**: Token overlap scoring
3. **Semantic search**: Vector similarity (pgvector, 384-dim embeddings)
4. **RRF fusion**: Combine lexical (40%) + semantic (60%) rankings
5. **DQS boost**: Prefer higher-quality factors
6. **LLM rerank** (enterprise only): Optional Claude-assisted reranking

---

## Audit Bundle (Enterprise)

For auditors who need full provenance documentation:

```
GET /api/v1/factors/{factor_id}/audit-bundle
```

**Returns:**
```json
{
  "factor_id": "EF:EPA:diesel:US:2024:v1",
  "edition_id": "2026.04.0",
  "content_hash": "sha256:abc123...",
  "normalized_record": { ... },
  "provenance": {
    "source_org": "US EPA",
    "source_publication": "GHG Emission Factors Hub",
    "source_year": 2024,
    "methodology": "measured",
    "citation": "..."
  },
  "license_info": { ... },
  "quality": {
    "dqs_overall": 4.2,
    "dqs_rating": "high"
  },
  "verification_chain": {
    "content_hash": "sha256:abc123...",
    "payload_sha256": "sha256:def456...",
    "algorithm": "SHA-256"
  }
}
```

---

## Rate Limiting

| Tier | Requests/min | Burst | Export/15min |
|------|-------------|-------|-------------|
| Community | 60 | 10 | 1 |
| Pro | 600 | 50 | 5 |
| Enterprise | 6,000 | 200 | 20 |

**Rate limit headers:**
```
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 599
X-RateLimit-Reset: 1714435200
```

On 429 Too Many Requests, retry after `Retry-After` seconds.
The SDK handles this automatically with exponential backoff.

---

## Error Code Reference

| Status | Code | Description |
|--------|------|-------------|
| 400 | `invalid_query` | Malformed search query or request body |
| 400 | `invalid_edition` | Unknown edition_id |
| 401 | `unauthorized` | Missing or invalid API key/JWT |
| 403 | `tier_insufficient` | Requested visibility exceeds tier entitlement |
| 404 | `factor_not_found` | Factor ID does not exist in edition |
| 404 | `edition_not_found` | Edition ID does not exist |
| 429 | `rate_limited` | Too many requests |
| 500 | `internal_error` | Server error (contact support) |

**Error response format:**
```json
{
  "error": {
    "code": "factor_not_found",
    "message": "Factor EF:UNKNOWN:xxx not found in edition 2026.04.0",
    "request_id": "req_abc123"
  }
}
```

---

## Changelog Format

Each edition includes a changelog accessible via:
```
GET /api/v1/editions/{edition_id}/changelog
```

**Format:**
```
edition diff 2026.03.0 -> 2026.04.0
added: 1,234
removed: 12
changed: 89

Sources updated:
- EPA GHG Hub 2025 -> 2026 (annual refresh)
- DESNZ UK 2025 -> 2026 (annual refresh)

Numeric corrections:
- EF:EPA:propane:US:2024:v1: 1.532 -> 1.548 kg CO2e/gallon

Deprecations:
- EF:OLD:coal:US:2020:v1 -> replaced by EF:EPA:coal:US:2024:v1
```

---

## Factor Diff (Edition Comparison)

Compare a specific factor between two editions:

```
GET /api/v1/factors/{factor_id}/diff?left_edition=2026.03.0&right_edition=2026.04.0
```

**Response:**
```json
{
  "factor_id": "EF:EPA:diesel:US:2024:v1",
  "status": "changed",
  "changes": [
    {"field": "gwp_100yr.CO2", "type": "changed", "old_value": 2.681, "new_value": 2.693},
    {"field": "provenance.source_year", "type": "changed", "old_value": 2024, "new_value": 2025}
  ]
}
```

---

## Bulk Export

Export full factor datasets (Pro/Enterprise):

```
GET /api/v1/factors/export?format=json&status=certified&geography=US
```

**Response headers:**
```
X-Factors-Edition: 2026.04.0
X-Factors-Manifest-Hash: sha256:...
Content-Type: application/jsonlines
```

Filter parameters: `status`, `geography`, `fuel_type`, `scope`, `source_id`

---

## ETag Caching

Factor responses include ETags for efficient caching:

```
GET /api/v1/factors/EF:EPA:diesel:US:2024:v1
ETag: "sha256:abc123..."
Cache-Control: public, max-age=3600
```

On subsequent requests:
```
GET /api/v1/factors/EF:EPA:diesel:US:2024:v1
If-None-Match: "sha256:abc123..."
```

Returns `304 Not Modified` if unchanged, saving bandwidth.

| Status | Cache-Control |
|--------|---------------|
| certified | `public, max-age=3600` (1 hour) |
| preview | `public, max-age=600` (10 minutes) |
| connector_only | `private, max-age=600` |
| deprecated | `no-cache` |
