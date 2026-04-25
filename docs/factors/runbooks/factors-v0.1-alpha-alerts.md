# Factors v0.1 Alpha — Dashboard & Alert Runbook

This runbook is the operational companion to the Grafana dashboard
[`factors-v0.1-alpha.json`](../../../deployment/observability/grafana/dashboards/factors-v0.1-alpha.json)
and the Prometheus alert rules
[`factors-v0.1-alpha-alerts.yaml`](../../../deployment/observability/prometheus/factors-v0.1-alpha-alerts.yaml).

Scope: **alpha API ingestion-health only** (CTO doc §19.1). Anything
related to resolve/explain/edition selection, billing, or entitlements
is intentionally out of scope here — see the production runbooks for
those.

## Table of contents
1. [Panel 1 — Alpha API request rate](#panel-1)
2. [Panel 2 — Alpha API p50 / p95 / p99 latency](#panel-2)
3. [Panel 3 — Alpha API error rate (4xx / 5xx)](#panel-3)
4. [Panel 4 — Edition served (current edition_id)](#panel-4)
5. [Panel 5 — Schema validation failure rate](#panel-5)
6. [Panel 6 — Provenance gate rejection rate](#panel-6)
7. [Panel 7 — Ingestion success rate per source](#panel-7)
8. [Panel 8 — Parser error rate](#panel-8)
9. [Quick diagnostics cheatsheet](#cheatsheet)

---

<a id="panel-1"></a>
## Panel 1 — Alpha API request rate

**Description.** Per-path request rate (req/s) for the alpha API, filtered
to `release_profile="alpha-v0.1"`.

**PromQL.**
```promql
sum(rate(http_requests_total{job="factors",release_profile="alpha-v0.1"}[5m])) by (path)
```

**Why it matters.** Anchor for traffic shape. Sudden drops to zero on
all paths mean the alpha pod stopped serving; a sudden spike means a
partner has gone live or a load test is running.

**Healthy.** Steady curve; alpha typically serves ~0.1–10 req/s during
the design-partner window.

**Unhealthy.** Flat-zero across all paths for > 5m **and** the alpha pod
is `Running` per `kubectl`. This means the deployment is up but routing
is broken (Kong / ingress / release_profile label mismatch).

**Common causes when red.**
- Ingress changed and the alpha host stopped routing.
- The `release_profile` label was renamed in the deployment without
  re-deploying the ServiceMonitor.
- Prometheus relabel config was modified.

**Quick diagnostics.**
```bash
kubectl get pods -n greenlang-factors -l release_profile=alpha-v0.1
kubectl logs -n greenlang-factors -l release_profile=alpha-v0.1 --tail=100
curl -sS https://alpha.factors.greenlang.io/api/v1/factors/health
```

**Severity / escalation.** Informational only. No alert is wired.

---

<a id="panel-2"></a>
## Panel 2 — Alpha API p50 / p95 / p99 latency

**Description.** Per-endpoint latency quantiles. CTO doc §19.1 acceptance
threshold: **p95 < 100 ms**.

**PromQL.**
```promql
histogram_quantile(
  0.95,
  sum(rate(http_request_duration_seconds_bucket{job="factors",release_profile="alpha-v0.1"}[5m])) by (le, path)
)
```

**Why it matters.** Latency is the headline alpha SLO; partners have
contractual 100 ms ceilings.

**Healthy.** p95 < 100 ms across every path; p99 < 250 ms.

**Unhealthy.** p95 > 100 ms for 10 minutes — `FactorsAlphaP95LatencyHigh`
**pages** on-call.

**Common causes when red.**
- Cold pgvector cache after a deploy or pod restart.
- New edition cut shipped a larger HNSW index.
- DB connection pool saturation (check `pgbouncer` queues).
- A parser landed an O(n²) regression.

**Quick diagnostics.**
```bash
kubectl top pods -n greenlang-factors
kubectl exec -n greenlang-factors deploy/factors-alpha -- \
  curl -sS http://localhost:8080/metrics | grep http_request_duration
psql "$FACTORS_DB_URL" -c "SELECT now() - pg_stat_activity.query_start AS age, query \
  FROM pg_stat_activity WHERE state = 'active' ORDER BY age DESC LIMIT 10;"
```

**Severity / escalation.** Page → on-call data-platform. If sustained
for > 30m and traffic is normal, consider rolling back the most recent
release with `kubectl rollout undo deploy/factors-alpha -n greenlang-factors`.

---

<a id="panel-3"></a>
## Panel 3 — Alpha API error rate (4xx / 5xx)

**Description.** Errored alpha API responses split by HTTP status.

**PromQL.**
```promql
sum(rate(http_requests_total{job="factors",release_profile="alpha-v0.1",status=~"4..|5.."}[5m])) by (status)
```

**Why it matters.** 5xx = server bug; 4xx = caller bug or auth issue.

**Healthy.** Combined 4xx+5xx ratio < 1%; 5xx < 0.1%.

**Unhealthy.**
- Combined > 1% for 15m → `FactorsAlphaErrorRateWarning` (Slack).
- 5xx > 5% for 5m → `FactorsAlphaErrorRateHigh` (page).

**Common causes when red.**
- 5xx: Database unreachable, pgvector OOM, unhandled parser exception
  bubbling into the response path.
- 4xx: A partner integration started sending bad payloads (often after a
  deploy on their side); also expired JWTs.

**Quick diagnostics.**
```bash
kubectl logs -n greenlang-factors -l release_profile=alpha-v0.1 --tail=500 \
  | grep -E '"status":(4[0-9]{2}|5[0-9]{2})'
kubectl exec -n greenlang-factors deploy/factors-alpha -- \
  curl -sS http://localhost:8080/metrics | grep http_requests_total
```

**Severity / escalation.** Warning → triage; page → on-call. If 5xx
spike correlates with a deploy, roll back first, debug second.

---

<a id="panel-4"></a>
## Panel 4 — Edition served (current edition_id)

**Description.** Stat panel showing the currently-active edition id.
Backed by `factors_current_edition_id_info{edition}` (gauge, value=1.0
for the active edition).

**PromQL.**
```promql
max(factors_current_edition_id_info) by (edition)
```

**Why it matters.** Every signed receipt embeds the edition; if the
served edition is the wrong one, every receipt issued in that window is
wrong.

**Healthy.** Exactly one edition shows value 1.0; matches the value in
`docs/factors/release_policy.yaml`.

**Unhealthy.** Multiple editions = 1.0 (split-brain after a partial
deploy) OR the published edition does not match the policy file.

**Common causes when red.**
- Half-rolled deploy (`kubectl rollout` paused).
- Manual edition switch via SDK without updating the gauge.

**Quick diagnostics.**
```bash
kubectl rollout status deploy/factors-alpha -n greenlang-factors
psql "$FACTORS_DB_URL" -c "SELECT edition_id, status FROM factors_editions \
  WHERE status = 'active';"
```

**Severity / escalation.** No automated alert; release manager owns this
during a cut. Surface manually if mismatch is detected during release.

---

<a id="panel-5"></a>
## Panel 5 — Schema validation failure rate

**Description.** Rate of `factor_record_v0_1` schema-validation failures
emitted by `AlphaProvenanceGate.validate()`.

**PromQL.**
```promql
sum(rate(factors_schema_validation_failures_total{schema="factor_record_v0_1"}[5m])) by (source)
```

**Why it matters.** Schema is the contract between ETL and the catalogue;
any non-zero rate means upstream is producing records that should never
be merged.

**Healthy.** Flat zero.

**Unhealthy.** Any sustained > 0 for 5m fires
`FactorsAlphaSchemaValidationFailure` (page). The reason is virtually
always a parser regression.

**Common causes when red.**
- A new parser version drops a required field.
- Source schema changed upstream and the parser was not updated.
- A hand-edited fixture file was committed.

**Quick diagnostics.**
```bash
ls -lt /var/lib/greenlang/factors/validation_report.json
jq '.failures[] | .failures[]' /var/lib/greenlang/factors/validation_report.json | head -20
git log --oneline -- greenlang/factors/etl/parsers/ | head -10
```

**Severity / escalation.** Page → data-eng on-call. Pin the previous
parser version in `pyproject.toml` until fixed.

---

<a id="panel-6"></a>
## Panel 6 — Provenance gate rejection rate

**Description.** Rate of `AlphaProvenanceGate.assert_valid()` rejections.
Labels: `source` and bucketed `reason`
(`schema_violation`, `extraction_metadata`, `review_metadata`,
`gwp_basis`, `other`).

**PromQL.**
```promql
sum(rate(factors_alpha_provenance_gate_rejections_total[5m])) by (source, reason)
```

**Why it matters.** Provenance gate is the last line of defence before a
record gets a signed receipt; any rejection means we caught a problem,
but a sustained rate means upstream is producing systematically bad
records.

**Healthy.** Flat zero (or rare single rejections during an ETL retry).

**Unhealthy.** > 1/min for 10m fires
`FactorsAlphaProvenanceGateRejecting` (page).

**Common causes when red.**
- `schema_violation`: parser regression (also fires panel 5).
- `extraction_metadata`: parser stopped emitting `parser_commit` /
  `parser_version` / `operator` correctly.
- `review_metadata`: a reviewer marked a record `approved` without
  filling `approved_by` / `approved_at`.
- `gwp_basis`: someone tried to ingest an AR5 record into the alpha
  catalogue (alpha is AR6-only).

**Quick diagnostics.**
```bash
jq '.failures[] | {idx: .record_index, first: .failures[0]}' \
  /var/lib/greenlang/factors/validation_report.json | head -30
```

**Severity / escalation.** Page → on-call. Triage by `reason` label.

---

<a id="panel-7"></a>
## Panel 7 — Ingestion success rate per source

**Description.** Successful ingestion runs per hour for each whitelisted
alpha source. The set is locked to:
`ipcc-ar6, defra-2025, epa-ghg-hub, epa-egrid, india-cea-baseline,
eu-cbam-defaults`.

**PromQL.**
```promql
sum(rate(factors_ingestion_runs_total{status="success",
  source=~"ipcc-ar6|defra-2025|epa-ghg-hub|epa-egrid|india-cea-baseline|eu-cbam-defaults"}[1h])) by (source)
```

**Why it matters.** This is the alpha catalogue freshness signal. If a
source's success rate falls below 95%, factors from that source will
silently age out.

**Healthy.** Each source ≥ 0.99 (≥ 99% of runs succeed in the past hour).

**Unhealthy.** Per-source success ratio < 0.95 for 30m fires
`FactorsAlphaIngestionSuccessRateLow` (warning).

**Common causes when red.**
- Upstream provider returning 5xx (DEFRA/EPA endpoints flake).
- Network egress policy blocking the source-watch pod.
- Credentials expired (e.g. EPA key rotation).
- Disk full on the staging volume.

**Quick diagnostics.**
```bash
kubectl logs -n greenlang-factors -l app=factors-source-watch --tail=200
kubectl exec -n greenlang-factors deploy/factors-source-watch -- \
  curl -sS https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2025 -o /dev/null -w "%{http_code}\n"
df -h /var/lib/greenlang/factors
```

**Severity / escalation.** Warning → Slack `#factors-alpha`. If the same
source flat-lines for > 4h, escalate to data-eng on-call.

---

<a id="panel-8"></a>
## Panel 8 — Parser error rate

**Description.** Rate of parser exceptions raised by ETL parsers
(CBAM/DEFRA/IPCC/EPA/CEA), labelled by `source` and `error_type`
(exception class name).

**PromQL.**
```promql
sum(rate(factors_parser_errors_total[5m])) by (source, error_type)
```

**Why it matters.** Distinct from schema/provenance failures: this is
"the parser threw before it could even produce a record". A non-zero
rate means we are losing rows, not just rejecting them.

**Healthy.** Flat zero.

**Unhealthy.** Any sustained > 0 for 5m fires `FactorsAlphaParserErrors`
(page).

**Common causes when red.**
- Source page HTML changed (CBAM scraper regressions are common).
- A new edition of the source dataset has a column the parser does not
  handle.
- Encoding issues (Windows-1252 vs UTF-8).
- Network timeouts wrapped as `JSONDecodeError`.

**Quick diagnostics.**
```bash
kubectl logs -n greenlang-factors -l app=factors-source-watch --tail=200 \
  | grep -E "Traceback|Error"
python -c "from greenlang.factors.etl.normalize import iter_cbam_factor_dicts; \
  import json; list(iter_cbam_factor_dicts(json.load(open('latest_cbam.json'))))"
```

**Severity / escalation.** Page → data-eng on-call. Pin the previous
parser commit and re-run ingest with the pinned version while the fix is
authored.

---

<a id="cheatsheet"></a>
## Quick diagnostics cheatsheet

| What | Command |
|---|---|
| List alpha pods | `kubectl get pods -n greenlang-factors -l release_profile=alpha-v0.1` |
| Tail alpha logs | `kubectl logs -n greenlang-factors -l release_profile=alpha-v0.1 --tail=200 -f` |
| Active editions | `psql "$FACTORS_DB_URL" -c "SELECT edition_id, status FROM factors_editions WHERE status='active';"` |
| Last validation report | `jq '.summary' /var/lib/greenlang/factors/validation_report.json` |
| Roll back alpha deploy | `kubectl rollout undo deploy/factors-alpha -n greenlang-factors` |
| Disable provenance gate (emergency) | `kubectl set env deploy/factors-alpha GL_FACTORS_ALPHA_PROVENANCE_GATE=0 -n greenlang-factors` |

> **Reminder.** Disabling the provenance gate is an emergency-only
> mitigation; every record ingested while it is off is unsigned and
> must be re-validated retrospectively. Re-enable inside the same shift.
