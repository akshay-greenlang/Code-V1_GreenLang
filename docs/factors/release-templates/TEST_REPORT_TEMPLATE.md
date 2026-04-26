# Test Report — `<release-id>`

> Template. Copy to
> `docs/factors/release-templates/instances/<release-id>/TEST_REPORT.md`
> and fill before the release PR is approved.

## Header

| Field             | Value                                                |
| ----------------- | ---------------------------------------------------- |
| Release id        | `<release-id>`                                       |
| Test commit       | `<git SHA>`                                          |
| Test environment  | `<dev \| staging \| prod-mirror \| ci>`               |
| Python version    | `<x.y.z>`                                            |
| OS / arch         | `<linux/amd64 \| macos/arm64 \| windows/amd64>`       |
| CI run            | `<URL or "local">`                                   |
| Test owner        | `<name>` (Platform/Data Lead or designate)           |

## Commands run

```bash
# Quality gate suite
python -m pytest tests/factors/v0_1_alpha -q

# URN canonical-parse regression
python -m pytest tests/factors/v0_1_alpha/test_seed_urns_canonical_parse.py -q

# Performance budget
python -m pytest tests/factors/v0_1_alpha/test_perf_p95_lookup.py -q

# (Add any release-specific commands)
```

## Results summary

| Metric             | Value                              |
| ------------------ | ---------------------------------- |
| Tests passed       | `<int>`                            |
| Tests failed       | `<int>`                            |
| Tests skipped      | `<int>` (with reasons listed below) |
| Warnings           | `<int>`                            |
| Wall time          | `<seconds>`                        |
| Exit code          | `<int>`                            |

## Per-suite breakdown

| Suite                                                          | Pass | Fail | Skip |
| -------------------------------------------------------------- | ---: | ---: | ---: |
| `tests/factors/v0_1_alpha/test_alpha_api_contract.py`          |      |      |      |
| `tests/factors/v0_1_alpha/test_alpha_provenance_gate.py`       |      |      |      |
| `tests/factors/v0_1_alpha/test_alpha_v0_1_normalizer.py`       |      |      |      |
| `tests/factors/v0_1_alpha/test_alpha_factor_repository.py`     |      |      |      |
| `tests/factors/v0_1_alpha/test_alpha_publisher.py`             |      |      |      |
| `tests/factors/v0_1_alpha/test_alpha_edition_manifest.py`      |      |      |      |
| `tests/factors/v0_1_alpha/test_alpha_source_snapshots.py`      |      |      |      |
| `tests/factors/v0_1_alpha/test_drill_fixtures.py`              |      |      |      |
| `tests/factors/v0_1_alpha/test_factor_record_v0_1_schema_loads.py` |  |      |      |
| `tests/factors/v0_1_alpha/test_factors_app_alpha_routes.py`    |      |      |      |
| `tests/factors/v0_1_alpha/test_grafana_alpha_dashboard.py`     |      |      |      |
| `tests/factors/v0_1_alpha/test_partner_eu_mfg_calc.py`         |      |      |      |
| `tests/factors/v0_1_alpha/test_partner_in_export_calc.py`      |      |      |      |
| `tests/factors/v0_1_alpha/test_perf_p95_lookup.py`             |      |      |      |
| `tests/factors/v0_1_alpha/test_postgres_ddl_v500.py`           |      |      |      |
| `tests/factors/v0_1_alpha/test_release_profile.py`             |      |      |      |
| `tests/factors/v0_1_alpha/test_schema_validates_alpha_catalog.py` |   |      |      |
| `tests/factors/v0_1_alpha/test_sdk_alpha_surface.py`           |      |      |      |
| `tests/factors/v0_1_alpha/test_sdk_e2e_ipcc_publish.py`        |      |      |      |
| `tests/factors/v0_1_alpha/test_seed_urns_canonical_parse.py`   |      |      |      |
| `tests/factors/v0_1_alpha/test_source_registry_alpha_completeness.py` |  |   |      |
| `tests/factors/v0_1_alpha/test_urn.py`                         |      |      |      |
| `tests/factors/v0_1_alpha/test_alembic_revision_0001.py`       |      |      |      |
| **Aggregate**                                                  |      |      |      |

## Skipped tests

For each skipped test, document the reason and the unblock plan:

| Test                  | Reason for skip                       | Unblock plan / target release |
| --------------------- | ------------------------------------- | ----------------------------- |
| `<test name>`         | `<reason>`                            | `<release or ticket>`         |

## Warnings

| Warning category      | Count | Action                                |
| --------------------- | ----: | ------------------------------------- |
| `PydanticDeprecated`  |       | Track migration to Pydantic V3        |
| `DeprecationWarning`  |       | (e.g. FastAPI on_event)               |
| Other                 |       |                                       |

## Known risks

Document anything that passed but the team flags as a risk:

* `<risk 1>` — owner: `<role>`, mitigation: `<plan>`, target release: `<rel>`
* `<risk 2>` — ...

## Performance budget

| Endpoint                      | Budget | Measured p95 | Pass / Fail |
| ----------------------------- | ------ | ------------ | ----------- |
| `GET /v1/healthz`             | ≤50ms  |              |             |
| `GET /v1/factors`             | ≤100ms |              |             |
| `GET /v1/factors/{urn}`       | ≤300ms |              |             |

## Sign-off

| Role                       | Name           | Decision     | Signed at (ISO-8601) |
| -------------------------- | -------------- | ------------ | -------------------- |
| Platform/Data Lead         | `<name>`       | `<approve>`  | `<timestamp>`        |
| SRE Lead                   | `<name>`       | `<concur>`   | `<timestamp>`        |
| Backend/API Lead           | `<name>`       | `<concur>`   | `<timestamp>`        |
