# GreenLang Factors v0.1 Alpha — Performance Budget

**Wave D / TaskCreate #28 / WS10-T2**
**Owner:** GL-Platform / SRE on-call
**Status:** Active (gate for v0.1 alpha launch)
**Last reviewed:** 2026-04-25

---

## 1. The Budget

| Endpoint | Metric | Ceiling | Source |
|---|---|---|---|
| `GET /v1/factors/{urn}` | **p95** | **< 100 ms** | CTO doc §19.1 (verbatim acceptance criterion) |
| `GET /v1/factors/{urn}` | p99 | < 300 ms | Tail-latency sanity (3x p95) |
| `GET /v1/factors?limit=50` | p95 | < 100 ms | Listing must keep parity with lookup |
| `GET /v1/healthz` | p95 | < 75 ms | Kube-probe + SDK pre-flight; must beat the auth'd lookup gate (test name preserves `_under_50ms` for stable CI reference) |

The single most important number is **p95(lookup) < 100ms**. That is the
clause the CTO doc cites; everything else is a guard around it.

---

## 2. Conditions Under Which the Budget Holds

The CTO doc's verbatim qualifier is **"single-region; no caching tier yet"**.
The test environment that gates the budget honours both clauses literally:

1. **Single-region (in-process).** The test boots the FastAPI app via
   Starlette's `TestClient` — no DNS, no TCP handshake, no TLS, no
   load balancer, no Kubernetes ingress hop. This is the **floor** of
   request-pipeline latency; production with a network round-trip will
   add tens of milliseconds on top, not subtract any.
2. **No caching tier.** The alpha app does not mount Redis or any
   request-level cache; the test seeds factors via the e2e shim's
   `_FakeRepo`, which answers every GET from a plain `dict`. No cache
   warm-up is benefiting the measurement.
3. **No Postgres.** Wave D's Postgres DDL is in flight (`V500__factors_*.sql`),
   but the alpha API contract under `GL_FACTORS_RELEASE_PROFILE=alpha-v0.1`
   does not require a live database; the e2e shim is the canonical
   alpha backing store for tests.
4. **Single-process, single-thread.** GIL contention from a busy
   worker pool can move p95 by 10-30%; the perf workflow runs without
   `pytest-xdist` (`-n` is intentionally NOT set).
5. **Cold-start excluded.** Each test does a 100-200 call warm-up
   pass before the measured loop, so import / interpreter-cache
   one-shots are not folded into p95.

If any of these conditions change (e.g. you add Redis to the alpha
app, or you switch the perf job to xdist), **rebaseline the budget
before merging**.

---

## 3. How to Reproduce

### Local

```bash
# Opt into the perf-marker auto-skip:
pytest -m perf tests/factors/v0_1_alpha/ -v

# Or via env var:
GL_RUN_PERF=1 pytest tests/factors/v0_1_alpha/test_perf_p95_lookup.py -v
```

The acceptance test writes a JSON report to:

```
out/factors/v0_1_alpha/perf_p95_report.json
```

Sibling reports for the other three perf tests live next to it
(`perf_test_p95_factor_list_under_100ms.json`,
`perf_test_p95_healthz_under_50ms.json`,
`perf_test_p99_factor_lookup_under_300ms.json`).

### CI (nightly + manual dispatch)

```
.github/workflows/factors-alpha-perf.yml
```

- Trigger: `schedule` (03:17 UTC nightly), `workflow_dispatch` (manual),
  `pull_request` on perf-relevant paths only.
- The job uploads `out/factors/v0_1_alpha/perf_*.json` and
  `junit-factors-perf.xml` as artefacts.

---

## 4. Method (What the Test Actually Does)

The canonical acceptance test (`test_p95_factor_lookup_under_100ms`):

1. Boots the alpha app under `GL_FACTORS_RELEASE_PROFILE=alpha-v0.1`.
2. Seeds **N=1000** distinct v0.1-shape factors via `install_alpha_e2e_shim`.
3. Samples **100** random URNs (deterministic seed `20260425`) — diverse
   URN strings prevent a single-URN dict-lookup hot path from
   monopolising the CPU cache.
4. Warms up with 200 GET calls (~2% of corpus, well below noise floor).
5. Issues **10,000** `GET /v1/factors/{urn}` calls (100 URNs × 100 reps),
   timing each call with `time.perf_counter()` in milliseconds.
6. Computes p50 / p95 / p99 / max / min / mean / stddev.
7. Writes the JSON report.
8. **Asserts p95 < 100ms.**

The percentile method is **nearest-rank** (no interpolation): "p95 < 100ms"
means at least 95% of real calls completed under 100ms, not "the
linearly-interpolated 95th-percentile bucket centre".

---

## 5. What Would Make Us Miss the Budget

In rough order of likelihood:

| # | Cause | Symptom in the report | First-look diagnostic |
|---|---|---|---|
| 1 | **Postgres index miss** when the real DB lands (Wave D #31) | p95 jumps from ~5ms to 80-150ms when `factors_catalog.factors` table is queried | `EXPLAIN ANALYZE SELECT ... WHERE urn = $1` — confirm the URN B-tree index is hit |
| 2 | **JSONB scan** in the citations / extraction columns | p95 drifts upward as catalog grows; p99 climbs faster | Add a generated column or a partial JSONB index; or denormalise hot fields |
| 3 | **GIL contention** from a busy uvicorn worker | p95 stable but p99 spikes | Profile with `py-spy dump`; consider gunicorn + multiple workers, NOT threads |
| 4 | **Auth-middleware regression** (auth path on `/v1/healthz`) | `/v1/healthz` p95 climbs above 50ms | Verify `PUBLIC_PATHS` list still short-circuits before the keyring lookup |
| 5 | **Pydantic v2 model rebuild** on hot path | p95 stable but mean climbs; cold p95 shows a 1-call cliff | Pin `model_config` and avoid `model_rebuild()` per request |
| 6 | **Logging amplification** (DEBUG enabled in prod) | Stable curve shifted ~10-30ms | Confirm `LOG_LEVEL=INFO` and that f-string log calls are %-format (logging-standardization, see MEMORY) |
| 7 | **Slow JSON encoder** (default `json` vs `orjson`) | Reproducible p95 ~30ms over baseline | Switch to `orjson` via `ORJSONResponse` for the alpha routes |

---

## 6. Escalation Path When Missed

1. **Workflow goes red on schedule** → SRE on-call is paged via the
   PagerDuty / Slack webhook stub in `factors-alpha-perf.yml` (currently
   commented out — wire up before launch).
2. **First responder downloads `factors-alpha-perf-report` artefact** and
   inspects:
   - `out/factors/v0_1_alpha/perf_p95_report.json` — the headline number
   - the runner profile dump (`/proc/cpuinfo`, `free -h`, `uptime`) — rule
     out a noisy GitHub-hosted runner; if `uptime` shows load > 4 on a
     2-vCPU runner, re-run via `workflow_dispatch` and the false alarm
     usually clears.
3. **If the regression reproduces locally:** bisect against the last
   green nightly. The blast radius is normally one of:
   - `greenlang/factors/api_v0_1_alpha_routes.py`
   - `greenlang/factors/api_v0_1_alpha_models.py`
   - `greenlang/factors/factors_app.py` (middleware order)
   - the `GL_FACTORS_RELEASE_PROFILE=alpha-v0.1` gate.
4. **If the regression is real and we cannot fix in 24h:**
   - Open a release-blocker issue tagged `v0.1-alpha`.
   - Notify the CTO (the budget is in their doc).
   - Either patch the perf hot spot OR — if the regression is a deliberate
     trade-off (e.g. we added auth that we MUST keep) — file a budget-revision
     RFC against this doc and re-baseline the ceiling.
5. **Do NOT silently widen the assertion.** If `p95 < 100.0` becomes
   `p95 < 150.0` in `test_perf_p95_lookup.py`, that is a CTO-doc-level
   change and needs sign-off, not a one-line PR.

---

## 7. Production Budget (Beta+, Post-Cache Tier)

When the Wave E+ network + Redis cache lands, the budget is expected to
shift roughly as follows. These numbers are **forecasts**, not gates —
re-measure on the real prod stack before promoting them into a CI gate.

| Endpoint | Budget (in-process, today) | Budget (prod w/ network + cache) |
|---|---|---|
| `GET /v1/factors/{urn}` (cache hit) | < 100 ms p95 | < 50 ms p95 |
| `GET /v1/factors/{urn}` (cache miss → Postgres) | n/a | < 250 ms p95 |
| `GET /v1/factors?limit=50` | < 100 ms p95 | < 200 ms p95 |
| `GET /v1/healthz` | < 75 ms p95 | < 100 ms p95 |

The widening on cache miss accounts for Postgres round-trip plus the
network hop; the in-process number does NOT include either.

---

## 8. References

- **CTO doc §19.1** — verbatim acceptance criterion: "Metrics show p95
  lookup latency under 100 ms (single-region; no caching tier yet)."
- **Test file:** `tests/factors/v0_1_alpha/test_perf_p95_lookup.py`
- **Workflow:** `.github/workflows/factors-alpha-perf.yml`
- **Marker registration:** `tests/factors/v0_1_alpha/conftest.py`
  (`perf` and `alpha_v0_1_acceptance`)
- **E2E shim:** `tests/factors/v0_1_alpha/_e2e_helpers.py`
- **Report artefacts:** `out/factors/v0_1_alpha/perf_*.json`
