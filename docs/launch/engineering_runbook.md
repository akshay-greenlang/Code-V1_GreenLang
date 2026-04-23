# GreenLang Factors — Engineering Runbook

Onboarding guide for a new engineer joining the Factors team. Covers local dev setup, running tests, building editions, cutting releases, and rollback procedures.

**Repo root:** `C:\Users\aksha\Code-V1_GreenLang`
**Factors module:** `greenlang/factors/`
**Tests:** `tests/factors/`
**Architecture deck:** [`cto_architecture_deck.md`](cto_architecture_deck.md).

---

## 1. Local dev setup (30 min)

### Prereqs

- Python 3.11+ (3.12 recommended).
- Node 20+ for the frontend (Explorer + Operator Console).
- Docker 24+ (for Postgres + pgvector + Redis).
- `make` and `just` installed.

### Steps

```bash
# 1. Clone
git clone git@github.com:greenlang/code-v1.git
cd code-v1

# 2. Create venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install with factors + test + dev extras
pip install -e ".[server,security,test,dev]"

# 4. Start infra (Postgres + pgvector + Redis + Kong)
docker compose -f deployment/docker/dev-compose.yml up -d

# 5. Run DB migrations
python -m greenlang.migrations.apply --up

# 6. Bootstrap the catalog (ingests open-source parsers)
python -m greenlang.factors.bootstrap --env dev

# 7. Start the API
uvicorn greenlang.factors.api:app --reload --port 8000

# 8. Smoke test
curl http://localhost:8000/v1/health
```

You should see `{"status": "ok", "edition": "builtin-dev-...", "db": "up", "redis": "up"}`.

---

## 2. Running tests

Three layers:

### 2.1 Unit + integration (pytest)

```bash
pytest tests/factors/ -v
```

Covers parsers, resolution engine, method pack rules, schema validation, signed-receipt verification, SDK transport. ~4,800 tests at current count; full run under 4 minutes.

Subsuites:

```bash
pytest tests/factors/resolution/           # cascade
pytest tests/factors/method_packs/         # pack rules
pytest tests/factors/ingestion/parsers/    # per-publisher parsers
pytest tests/factors/schema/               # canonical record validation
pytest tests/factors/sdk/                  # Python SDK
```

### 2.2 Static gates

```bash
make factors-gates
```

Runs:
- `ruff check` + `mypy` (Python 3.11 strict).
- `audit_text_gate.yml` — prevents Certified packs shipping with `approved: false` templates.
- `schema_gate.yml` — confirms every catalog row validates against `factor_record_v1.schema.json`.
- `license_scanner.yml` — fails if a Certified-cut row is tagged `Blocked-Contract-Required` without a `legal_signoff_artifact`.

### 2.3 Gold-set gate

```bash
pytest tests/factors/gold_set/ -v
# or
make factors-gold-set
```

The gold set is a frozen list of ~150 resolution requests with expected chosen factors and assumptions. Any edition promotion that fails the gold set is blocked. Source: `tests/factors/gold_set/canonical_requests.yaml`.

CI workflow: `.github/workflows/factors_gold_eval.yml`.

---

## 3. Running the ingestion bootstrap

Ingest every parser under `greenlang/factors/ingestion/parsers/` and populate the dev catalog:

```bash
python -m greenlang.factors.bootstrap --env dev --verbose
```

Options:

| Flag | Notes |
|---|---|
| `--sources <id1,id2>` | Restrict to specific source IDs. |
| `--skip-connectors` | Skip BYO-credentials connectors (default in dev). |
| `--dry-run` | Validate + count rows without writing. |
| `--reset` | Drop catalog tables before re-populating. |

The bootstrap prints a summary per source: `rows_inserted`, `rows_updated`, `validation_errors`, `license_class_distribution`.

---

## 4. Running the canonical-record migration

The v1 migration (one-time, already executed on production 2026-04-22) transforms legacy `EmissionFactorRecord` rows into `factor_record_v1` shape:

```bash
python -m greenlang.factors.migrations.canonical_v1 --apply
```

Script source: `scripts/migrate_to_canonical_v1.py`. See [`docs/specs/factor_record_v1.md`](../specs/factor_record_v1.md) §4 for the drift table.

To re-run the migration on a dev database (after `--reset` bootstrap):

```bash
python -m greenlang.factors.migrations.canonical_v1 --apply --dev
```

Reversible via branch within 72 hours of merge; beyond that, forward-only.

---

## 5. Cutting a dev edition

```bash
# Validate the catalog
gl-factors edition validate --channel dev

# Cut an unsigned dev edition (for local testing)
gl-factors edition cut --channel dev --name "dev-2026-04-23-01"

# List editions
gl-factors edition list
```

The command produces an edition manifest, signs it (with a dev key if `--dev`), and writes the artifacts to `$GL_EDITION_STORE` (default `./artifacts/editions/`).

For a **Certified** edition, the manifest goes through the methodology WG review board and is signed with the production Ed25519 key held in Vault. Do not cut Certified editions locally.

---

## 6. Rollback procedure

If a bad edition reaches production:

### 6.1 Immediate rollback (no data change)

```bash
gl-factors edition rollback --to builtin-v1.0.0
```

This updates the default edition pointer. In-flight callers pinning the bad edition get `409 edition_mismatch`; callers on the default immediately start receiving the prior edition.

### 6.2 Catalog rollback (rare)

If a factor value was ingested wrong:

1. Identify the affected rows via the admin dashboard or `SELECT * FROM factor_catalog WHERE ingested_at BETWEEN ... AND ...`.
2. Bump affected `factor_version` with a `change_reason` describing the corrected value.
3. Mark the old version `deprecated` with `rationale` pointing to the corrected `factor_id:version`.
4. Cut a new patch edition (`builtin-v1.0.1`).
5. Emit `factor.deprecated` and `factor.released` webhook events.

**Never overwrite a `(factor_id, factor_version)` pair.** That breaks CTO non-negotiable #2.

### 6.3 Signed-receipt key rotation

If a signing key is compromised:

1. Rotate the key in Vault.
2. Publish the new public key to the JWKS at `https://api.greenlang.io/.well-known/jwks.json`.
3. Retain the old public key in the JWKS for 18 months so historical receipts still verify (CTO reproducibility commitment).
4. Notify tenants via `receipt.key_rotated` webhook (if subscribed).

---

## 7. On-call operations

- **Health endpoints:** `GET /v1/health`, `GET /v1/readiness`, `GET /v1/metrics` (Prometheus).
- **Dashboards:** Grafana `factors-api` and `factors-resolver` — p50/p95/p99 latency, error rate by code, cache hit ratio.
- **Alerts:** PagerDuty routes; runbook per alert in `deployment/runbooks/factors/`.
- **Incident logs:** Loki index `service=factors-api`.

---

## 8. Common tasks

| Task | Command |
|---|---|
| Run the Factors test suite | `pytest tests/factors/ -v` |
| Run the gold-set gate | `make factors-gold-set` |
| Rebuild the semantic index | `python -m greenlang.factors.matching.semantic_index rebuild` |
| Refresh entitlements from Stripe | `python -m greenlang.factors.billing.sync` |
| Generate a signed receipt (test) | `python scripts/gen_test_receipt.py` |
| Validate a factor row against the schema | `gl-factors schema validate ./row.json` |

---

## 9. Further reading

- [`cto_architecture_deck.md`](cto_architecture_deck.md) — system overview.
- [`methodology_manual.md`](methodology_manual.md) — methodology lead's reference.
- [`legal_source_rights_binder.md`](legal_source_rights_binder.md) — legal evidence pack.
- [`docs/specs/factor_record_v1.md`](../specs/factor_record_v1.md) — canonical record schema.
- [`docs/specs/method_pack_template.md`](../specs/method_pack_template.md) — method pack spec.
- [`docs/specs/audit_text_template_policy.md`](../specs/audit_text_template_policy.md) — audit-text policy.
- [`docs/legal/source_rights_matrix.md`](../legal/source_rights_matrix.md) — source rights audit.
