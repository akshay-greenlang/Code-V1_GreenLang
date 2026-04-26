# Epic: v0.5 Closed Beta (FY27 Q2)

| Field            | Value                                                    |
| ---------------- | -------------------------------------------------------- |
| Release profile  | `beta-v0.5`                                              |
| Target quarter   | FY27 Q2                                                  |
| Owner            | Backend/API Lead                                         |
| Status           | Planned                                                  |

## Scope

Closed beta with 5–10 design partners. Activates the resolution +
explain stack on top of the v0.1 read-only catalog.

* New endpoints (all gated by `feature_enabled('<name>')`):
  `/v1/resolve`, `/v1/explain`, `/v1/batch`, `/v1/coverage`,
  `/v1/quality/fqs`, `/v1/editions`.
* Method-pack coverage routes.
* Signed receipts (`signed_receipts` feature) — content-addressed
  resolution receipts with edition + parser commit pinning.
* Admin console (read-only).
* TypeScript SDK (parity with Python read + resolve).
* Extended CLI (`gl factors resolve|explain|batch|coverage`).

## Out of scope

* GraphQL, ML resolve — defer to v0.9.
* Billing, OEM, SQL-over-HTTP, commercial packs, real-time grid —
  defer to v1.0.

## Deliverables

* Resolution engine + ranker + explain trace.
* Method-pack registry (GHG Protocol, ISO 14064, GRI, CDP method packs).
* FQS (Factor Quality Score) computation + per-factor scoreboard.
* Edition-cut publisher promoted to GA-quality.
* Signed-receipt verifier and SDK helpers.
* Admin console (read-only) at `/admin/factors/`.

## Acceptance criteria

* Gold-set evaluation: P@1 ≥ 70%, R@3 ≥ 80% on the
  `tests/factors/gold_set_v0_5/` corpus (to be created).
* p95 `/v1/resolve` ≤ 500ms; batch resolve ≥ 200 req/s steady-state.
* Receipt signature verifies offline using the published Ed25519
  pubkey.
* TypeScript SDK ships with parity tests vs Python SDK.

## Source coverage

* All 6 alpha sources, plus:
  * GLEC v3.1 (transport),
  * Climate TRACE (sector inventories),
  * EU ETS Phase 4 emissions registry (sector benchmarks),
  * GHG Protocol Scope 3 calculation tool defaults.

## API / SDK expectations

| Surface                | v0.1 | v0.5 |
| ---------------------- | ---- | ---- |
| `/v1/factors*`         | ON   | ON   |
| `/v1/sources`,`/packs` | ON   | ON   |
| `/v1/resolve`          | OFF  | ON   |
| `/v1/explain`          | OFF  | ON   |
| `/v1/batch`            | OFF  | ON   |
| `/v1/coverage`         | OFF  | ON   |
| `/v1/quality/fqs`      | OFF  | ON   |
| `/v1/editions`         | OFF  | ON   |
| Admin console          | OFF  | ON (read-only) |
| TS SDK                 | —    | ON   |

## Security / compliance gates

* All beta partners under MSA + DPA.
* Per-tenant rate limits configured at API gateway.
* Audit log emits one event per `/v1/resolve` call (tenant, query,
  result urn, edition, latency).

## Tickets

* [ ] Resolver pipeline + ranker (`greenlang/factors/resolve/`).
* [ ] Explain trace generator (DAG-of-decisions JSON).
* [ ] Batch endpoint (`/v1/batch`) with parquet I/O.
* [ ] FQS scoreboard + per-source quality dashboard.
* [ ] Method-pack manifest schema + registry seeding.
* [ ] Signed-receipt issuer + verifier.
* [ ] TS SDK package + npm publish pipeline.
* [ ] Admin console (read-only Next.js app).

## Dependencies

* v0.1 Alpha shipped + 1+ design partner live.
* OBS-005 (SLOs) defined for resolve + batch endpoints.
* INFRA-005 (pgvector) populated with method-pack embeddings.

## Release risks

* Gold-set construction is a long-lead item; need methodology lead +
  1 SME per major source family.
* TS SDK parity — risk of API drift between languages without a
  shared OpenAPI spec.
