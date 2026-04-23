# Factors Test Environment Runbook

> **Status:** authoritative as of 2026-04-23.
> **Owner:** GL-DevOpsEngineer.
> **Resolves:** CTO-flagged "no reproducible test environment" gap for
> the GreenLang Factors FY27 launch.
>
> The full Factors test suite under `tests/factors/` (92 test files) requires
> pytest, FastAPI, Postgres-with-pgvector, Redis, and a few signing /
> billing dependencies. Before this runbook existed, the only way to run
> them locally on the founder's machine was through a broken CBAM venv with
> mismatched packages — hence the CTO's red flag. The three paths below
> guarantee a green run from a cold checkout.

---

## TL;DR

```bash
# Fastest path — Docker Desktop required.
make factors-test

# Or the equivalent shell / PowerShell invocations:
bash scripts/run_factors_tests.sh           # macOS / Linux / WSL
pwsh -File scripts/run_factors_tests.ps1    # Windows native

# Local-venv fallback (no Docker):
make factors-test-local
```

Exit code 0 means every test passed and coverage XML was emitted to
`./coverage-factors.xml` (Docker path) or `./.coverage` (local path).

---

## The three supported paths

### 1. Docker Compose (canonical, what CI runs)

Bring up Postgres-with-pgvector + Redis + the pytest runner in one shot:

```bash
docker compose -f deployment/docker/docker-compose.factors-test.yml \
  up --build --abort-on-container-exit --exit-code-from factors-test
```

The `factors-test` container exits with the pytest exit code, and
`--abort-on-container-exit` ensures the script returns it. Tear down with
`docker compose -f deployment/docker/docker-compose.factors-test.yml down -v`.

**Why this is the canonical path:**

- Reproduces CI exactly (`.github/workflows/factors_ci.yml` uses the same
  service definitions for Postgres + Redis).
- pgvector image, so `tests/factors/matching/test_pgvector_matching.py`
  exercises real vector search, not a stub.
- Source tree is bind-mounted, so test edits do **not** require a rebuild.
  Only changes to `pyproject.toml` invalidate the cached dep layer.

### 2. Local virtualenv

Use this when Docker Desktop is unavailable (e.g. an air-gapped machine
or a fresh corporate laptop where Docker isn't approved yet):

```bash
python -m venv .venv
source .venv/bin/activate                         # or .venv\Scripts\activate
pip install -e ".[factors-test]"
pytest tests/factors -v --cov=greenlang.factors
```

The `make factors-test-local` target wraps this. Tests that strictly
need Postgres / Redis will skip cleanly with a clear message rather than
fail; everything else (resolution, signing, billing, parsers, mapping,
matching with `fakeredis`, etc.) runs from the venv alone.

### 3. CI (GitHub Actions)

`.github/workflows/factors_ci.yml` is triggered automatically on PRs that
touch `greenlang/factors/**` or `tests/factors/**`. It:

1. Spins up `pgvector/pgvector:pg16` and `redis:7-alpine` as services.
2. Runs `pip install -e ".[factors-test]"`.
3. Runs `pytest tests/factors -v --cov`.
4. Uploads `coverage-factors.xml` and `junit-factors.xml` as artifacts.

You should never need to run anything from CI manually — push the branch
and read the Actions tab.

---

## Required vs optional environment variables

| Variable | Required? | Purpose |
|---|---|---|
| `DATABASE_URL` | Optional (auto-set in Docker) | Real Postgres DSN; if unset, `mock_pg` tries `testcontainers`, then skips. |
| `GL_FACTORS_PG_DSN` | Optional | Alias of `DATABASE_URL` — checked second. |
| `REDIS_URL` | Optional (auto-set in Docker) | Real Redis URL; if unset, `mock_redis` falls back to `fakeredis`. |
| `STRIPE_API_KEY` | Optional | Tests **never** call real Stripe — `mock_stripe` monkeypatches the SDK. Set to `test_key_mock` to silence boot-time validation. |
| `STRIPE_WEBHOOK_SECRET` | Optional | Same — defaults to `test_secret_mock` in compose. |
| `GL_FACTORS_SIGNING_SECRET` | Optional | Used by HMAC signed-receipt tests. The Ed25519 path uses `signing_key_pair` (ephemeral). |
| `GL_JWT_SECRET` | Optional | Used by `tests/factors/test_api_auth.py`. Compose injects a 32-char dev value. |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Optional | LLM-rerank tests skip when absent. |
| `GL_ENV` | Set automatically | Always `test` in the runner container. |

---

## Seeding the catalog for E2E tests

Most unit tests can use the `seed_factors` fixture (10 representative
factors across all 7 method packs — see `tests/factors/conftest.py`).
That is fast (microseconds) and stable.

For tests that hit a *real* HTTP surface (the `factors_app` /
`factors_client` fixtures), the conftest pre-seeds a SQLite catalog by
calling `ingest_builtin_database()` into a `tmp_path`. No manual setup.

If you want a *full* ingest into the compose Postgres for ad-hoc
exploration:

```bash
docker compose -f deployment/docker/docker-compose.factors-test.yml \
  run --rm factors-test \
  python -m greenlang.factors.cli ingest \
    --target postgres --edition dev-full
```

---

## Running a single family's coverage matrix

```bash
# Just the billing tests:
docker compose -f deployment/docker/docker-compose.factors-test.yml \
  run --rm factors-test \
  pytest tests/factors/billing -v --cov=greenlang.factors.billing

# Just the signing / Ed25519 rotation tests:
docker compose -f deployment/docker/docker-compose.factors-test.yml \
  run --rm factors-test \
  pytest tests/factors/signing -v --cov=greenlang.factors.signing

# Just the resolution + matching layer:
docker compose -f deployment/docker/docker-compose.factors-test.yml \
  run --rm factors-test \
  pytest tests/factors/resolution tests/factors/matching -v
```

Locally (no Docker):

```bash
pytest tests/factors/<sub-package> -v --cov=greenlang.factors.<sub>
```

The shell / PowerShell wrappers (`scripts/run_factors_tests.{sh,ps1}`)
forward extra args straight to pytest:

```bash
bash scripts/run_factors_tests.sh tests/factors/billing -k credits_for
```

---

## Debugging a failing gold-set case

The gold-set lives at `tests/factors/fixtures/gold_eval_smoke.json` and
is exposed via the `gold_eval_cases` fixture. To debug a single case
interactively:

```bash
make factors-test-shell                # drops you into the runner
# inside the container:
pytest tests/factors/test_evaluation.py::test_gold_eval_smoke \
  -v -s --pdb -k "<case-id>"
```

`--pdb` opens the Python debugger on the first failure. Combine with
`-s` to see prints / logs immediately.

---

## Pyproject group reference

The new `[factors-test]` group in `pyproject.toml` is the single source
of truth for the toolchain — it pulls in pytest, asyncio plumbing, an
HTTP client, FastAPI/Starlette, Pydantic v2, real Postgres + Redis
drivers, `pgvector`, `fakeredis`, `stripe` (mocked at runtime),
`python-jose` + `cryptography` for offline signed-receipt verification,
and `respx` / `responses` for HTTP-level mocking of SDK / connector
tests. It is also bundled into the `[all]` aggregator group so
`pip install -e ".[all]"` is a strict superset of what CI runs.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pytest: command not found` | Run `pip install -e ".[factors-test]"` first, or use `make factors-test`. |
| `psycopg.OperationalError: could not connect` | Postgres container not healthy yet — `docker compose ps` and wait, or rerun. The healthcheck retries 30 times. |
| `SkipTest: No Postgres available` (local path) | Expected when `DATABASE_URL` is unset and `testcontainers` not installed — switch to the Docker path. |
| `ModuleNotFoundError: pgvector` | You're on a Python interpreter that wasn't installed via `[factors-test]`. Recreate your venv. |
| Tests pass locally but fail in CI | Almost always one of: (a) hardcoded `tmp_path` assumption, (b) missing `monkeypatch.setenv` for an env var the code depends on, (c) ordering dependency on a session fixture — try `pytest -p no:randomly` to eliminate ordering as a suspect. |
| `stripe.error.AuthenticationError` in a test | Test isn't using `mock_stripe` — request the fixture and the SDK is monkeypatched. Real Stripe must never be called from tests. |

---

## CTO acceptance checklist

- [x] `pip install -e ".[factors-test]"` resolves on a clean Python 3.11 / 3.12 venv.
- [x] `make factors-test` brings up PG + Redis + runs the full suite.
- [x] `bash scripts/run_factors_tests.sh` works on macOS / Linux / WSL.
- [x] `pwsh -File scripts/run_factors_tests.ps1` works on Windows.
- [x] `.github/workflows/factors_ci.yml` reproduces the exact same
      toolchain in CI on every PR to `greenlang/factors/**` or
      `tests/factors/**`.
- [x] Conftest provides `factors_app`, `factors_client`, `mock_pg`,
      `mock_redis`, `mock_stripe`, `seed_factors`, `signing_key_pair`
      so future test authors do not have to reinvent the boundary.
