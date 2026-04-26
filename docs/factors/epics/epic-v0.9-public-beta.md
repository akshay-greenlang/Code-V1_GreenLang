# Epic: v0.9 Public Beta / RC (FY27 Q3)

| Field            | Value                                                    |
| ---------------- | -------------------------------------------------------- |
| Release profile  | `rc-v0.9`                                                |
| Target quarter   | FY27 Q3                                                  |
| Owner            | Developer Experience Lead                                |
| Status           | Planned                                                  |

## Scope

Open public beta. Adds GraphQL and ML-driven resolution.

* GraphQL endpoint (read + resolve subset).
* ML resolve (LLM + retrieval) for free-text queries.
* All v0.5 features locked at GA-readiness.

## Out of scope

* Billing, OEM, commercial packs, SQL-over-HTTP, real-time grid —
  ship at v1.0 GA.

## Deliverables

* GraphQL schema + resolver layer at `/graphql`.
* ML resolve service (`greenlang/factors/ml_resolve/`) backed by
  pgvector + LLM with strict zero-hallucination guardrails (every
  ML answer carries the deterministic urn it resolved to).
* Self-serve sign-up flow (no billing — free tier).
* Public docs site at `factors.greenlang.io/docs`.

## Acceptance criteria

* GraphQL schema lints clean; persisted-query allowlist enforced in
  production.
* ML resolve P@1 ≥ 80% on extended gold set; 0 hallucinated URNs in
  100k-query soak test.
* Public docs cover every v0.9 endpoint with worked examples.
* Public sign-up open behind Cloudflare Turnstile.

## Source coverage

* v0.5 set + Ecoinvent 3.11 (subset under EULA), GLEC 4.0 update,
  ENTSO-E day-ahead grid factors (hourly, EU bidding zones).

## API / SDK expectations

| Surface                | v0.5 | v0.9 |
| ---------------------- | ---- | ---- |
| GraphQL `/graphql`     | OFF  | ON   |
| ML resolve             | OFF  | ON   |
| Persisted queries      | —    | ON   |

## Security / compliance gates

* GraphQL persisted-query allowlist on (no arbitrary GQL in prod).
* ML resolve PII-detector pass on every prompt + completion (SEC-011).
* Penetration test report on file before opening public sign-up.

## Tickets

* [ ] GraphQL schema + resolvers.
* [ ] Persisted-query store + signing.
* [ ] ML resolve pipeline (retrieve → LLM → urn-bind → validate).
* [ ] Hallucination-detector (URN-set diff vs catalog).
* [ ] Public docs site + worked examples.
* [ ] Self-serve sign-up + Turnstile + free-tier rate limits.

## Dependencies

* v0.5 Beta shipped.
* SEC-011 (PII detection) integrated into ML pipeline.
* OBS-003 (OpenTelemetry tracing) emits per-GraphQL-field spans.

## Release risks

* ML resolve cost per query — needs caching + retrieval tuning to
  stay below free-tier unit economics.
* GraphQL schema evolution — need durable deprecation policy.
