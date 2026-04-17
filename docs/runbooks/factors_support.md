# GreenLang Factors — support and operations

## Editions and pinning

- Clients SHOULD send `X-Factors-Edition` with a value from `GET /api/v1/editions` so audits reproduce the same catalog snapshot.
- Query override: `?edition=<edition_id>` (ignored when the header is set).
- Unknown edition returns HTTP 404 with a clear message.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `GL_FACTORS_SQLITE_PATH` | When set to an existing SQLite file produced by `gl factors ingest-builtin`, the API serves that catalog instead of the in-memory built-ins for list/search/stats. |
| `GL_FACTORS_BUILTIN_EDITION` | Synthetic edition label for the in-memory adapter (default `builtin-v1.0.0`). |
| `GL_API_KEYS` | Comma-separated API keys (`gl_…`, min length 32). When non-empty in production-like environments, only listed keys are accepted. |
| API key rotation (A4) | Rotate by appending a new key to `GL_API_KEYS`, redeploying clients to use it, then removing the old key on the next maintenance window. JWT signing secret rotation uses `GL_JWT_SECRET` with coordinated token invalidation. |
| `GL_FACTORS_TIER` | `community` (default), `pro`, or `enterprise` — attached to JWT/API-key auth context for future rate-limit tiers. |
| `GL_JWT_SECRET` | HS256 secret for JWTs (existing API auth). |
| `GL_ENV` | `development` / `staging` / `test` allow anonymous access with warnings; `production` requires credentials. |
| `GL_FACTORS_FORCE_EDITION` | Emergency pin: override edition resolution for all factor routes (hotfix / rollback, U6). |
| `GL_FACTORS_USAGE_SQLITE` | When set, append-only `api_usage_events` table for billing hooks (C5). |

## Enterprise SLA and procurement (C2 / C4)

- **SLA:** enterprise contracts SHOULD reference response targets from `docs/factors/support_boundaries_and_severity.md` (severity matrix).
- **Procurement:** data residency stays with the deployer; connector-only sources (IEA, ecoinvent, Electricity Maps) never imply open redistribution—cite `license_class` from `GET /api/v1/factors/source-registry`.
- **API freeze / SDK:** generate OpenAPI client only after `A6` freeze; contract tests live under `tests/factors/`.

## SQLite ingest (local / air-gapped)

```bash
gl factors ingest-builtin --sqlite ./data/factors_catalog.sqlite --edition-id 2026.04.0-builtin
export GL_FACTORS_SQLITE_PATH=./data/factors_catalog.sqlite
```

Optional JSON bundles (DEFRA scope1, CBAM defaults):

```bash
gl factors ingest-paths --sqlite ./data/factors_catalog.sqlite --edition-id 2026.04.0-mix \
  cbam-pack-mvp/data/emission_factors/cbam_defaults_2024.json \
  applications/GL-Agent-Factory/backend/data/emission_factors/defra/scope1_fuels.json
```

## Pending edition workflow (policy-driven)

1. When a methodology change affects factors, write `*.pending_edition.json` next to the SQLite DB using `greenlang.factors.policy_mapping.write_pending_edition` (or internal tooling).
2. SMEs review the proposed `factor_ids` and `policy_rule_ids`.
3. After ingest and manifest sign-off, promote the SQLite `editions.status` row to `stable` and delete the pending sidecar.

## Metering

- `GET /api/v1/system/factors-metering` returns in-process path counters (development aid; replace with Prometheus in hosted deployments).

## Wrong factor triage

1. Confirm `X-Factors-Edition` and `content_hash` from `GET /api/v1/factors/{id}/provenance`.
2. Compare against upstream source publication cited in `provenance`.
3. If data is wrong, publish a new edition; never mutate rows in-place for stable editions.

## Postgres

- Flyway migration: `deployment/database/migrations/sql/V426__factors_catalog.sql` defines `factors_catalog` schema for hosted parity with SQLite semantics.
- `V427__factors_cto_extensions.sql` adds governance columns, lineage, QA reviews, usage events, and policy applicability tables.

## Load testing (GA)

- Target 100k+ rows in SQLite or Postgres, then exercise `GET /api/v1/factors` and `GET /api/v1/factors/search` with realistic `limit` and pinning headers. Record p95 latency and error rate.
