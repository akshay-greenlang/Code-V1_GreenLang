# Phase 2 - CTO §2.7 Acceptance Test Coverage Matrix

> **Authority**: CTO Phase 2 brief Section 2.7 (`docs/factors/PHASE_2_PLAN.md`).
> **Owner**: WS10 (gl-test-engineer).
> **Scope**: This document maps every CTO §2.7 minimum required test suite to a
> concrete file path in the test tree, classifying each as `PASS`, `IN-FLIGHT`,
> or `GAP`.
> **Last refreshed**: 2026-04-27 — Phase 2 acceptance build.

---

## CTO §2.7 minimum required suites (10)

| # | CTO §2.7 required suite | Tree path of the covering suite | Workstream owner | Status |
|---|---|---|---|---|
| 1 | `test_schema_validates_alpha_catalog` | [`tests/factors/v0_1_alpha/phase2/test_schema_validates_alpha_catalog.py`](../../tests/factors/v0_1_alpha/phase2/test_schema_validates_alpha_catalog.py) | WS1 | PASS |
| 2 | `test_urn_parse_build_roundtrip` | [`tests/factors/v0_1_alpha/phase2/test_urn_property_roundtrip.py`](../../tests/factors/v0_1_alpha/phase2/test_urn_property_roundtrip.py) | WS2 | PASS |
| 3 | `test_urn_uniqueness_db` | [`tests/factors/v0_1_alpha/phase2/test_urn_uniqueness_db.py`](../../tests/factors/v0_1_alpha/phase2/test_urn_uniqueness_db.py) | WS2 | PASS |
| 4 | `test_ontology_fk_enforcement` | [`tests/factors/v0_1_alpha/phase2/test_ontology_fk_enforcement.py`](../../tests/factors/v0_1_alpha/phase2/test_ontology_fk_enforcement.py) | WS10 | PASS (newly added) |
| 5 | `test_alembic_up_down` | [`tests/factors/v0_1_alpha/phase2/test_alembic_up_down.py`](../../tests/factors/v0_1_alpha/phase2/test_alembic_up_down.py) | WS7 | PASS |
| 6 | `test_seed_load` | [`tests/factors/v0_1_alpha/phase2/test_seed_load.py`](../../tests/factors/v0_1_alpha/phase2/test_seed_load.py) + [`test_activity_seed_load.py`](../../tests/factors/v0_1_alpha/phase2/test_activity_seed_load.py) | WS3+4+5+6 | PASS |
| 7 | `test_publish_rejection_matrix` | [`tests/factors/v0_1_alpha/phase2/test_publish_rejection_matrix.py`](../../tests/factors/v0_1_alpha/phase2/test_publish_rejection_matrix.py) | WS8 | IN-FLIGHT |
| 8 | `test_api_query_factor_by_urn` | [`tests/factors/v0_1_alpha/phase2/test_api_query_factor_by_urn.py`](../../tests/factors/v0_1_alpha/phase2/test_api_query_factor_by_urn.py) | WS10 | PASS (newly added) |
| 9 | `test_sdk_fetch_by_urn` | [`tests/factors/v0_1_alpha/phase2/test_sdk_fetch_by_urn.py`](../../tests/factors/v0_1_alpha/phase2/test_sdk_fetch_by_urn.py) | WS10 | PASS (newly added) |
| 10 | `test_provenance_checksum` | [`tests/factors/v0_1_alpha/phase2/test_provenance_checksum.py`](../../tests/factors/v0_1_alpha/phase2/test_provenance_checksum.py) | WS10 | PASS (newly added) |

---

## Status legend

- **PASS**: Suite exists in tree, passes locally on a clean checkout, gates
  one or more CTO §2.7 KPIs.
- **IN-FLIGHT**: Suite is owned by another workstream and currently being
  authored. WS10 does NOT touch it; the aggregate runner records its
  status as `IN-FLIGHT` until WS8 marks it `PASS`.
- **GAP**: No suite exists. (Zero rows in this state at Phase 2 acceptance
  ship.)

---

## Pre-existing Wave 1 suites (context, not required by §2.7)

These ship as part of Wave 1 and are exercised by the aggregate runner but
are not enumerated by CTO §2.7:

| Tree path | Workstream | Notes |
|---|---|---|
| `test_pydantic_mirrors_jsonschema.py` | WS1 | Phase 2 Block 1 gate (mirror drift). |
| `test_urn_lowercase_sweep.py` | WS2 | Lowercase namespace enforcement. |
| `test_alias_backfill_idempotency.py` | WS2 | V501 alias backfill is idempotent. |
| `test_api_urn_primary.py` | WS2 | URN-as-primary on every API/SDK surface. |
| `test_activity_urn_parser.py` | WS5 | Activity URN parser/builder. |
| `test_activity_table_constraints.py` | WS5 | V502 activity table constraints. |
| `test_storage_tables.py` | WS7 | V501-V504 table presence + indexes. |
| `test_repository_extensions.py` | WS7 | `find_by_methodology`, `find_by_alias`. |
| `test_schema_evolution_gate.py` | WS9 | CI gate for schema evolution policy. |
| `test_version_registry.py` | WS9 | Schema version registry classification. |

---

## Aggregate acceptance runner

The full Phase 2 acceptance suite is executed by:

```bash
python scripts/factors/run_phase2_acceptance.py
```

The runner enumerates the 10 CTO §2.7 suites, runs each via
`pytest --tb=line --no-header`, and renders a colour-coded matrix to stdout.
Exit code is `0` only when every CTO row is `PASS`.

CI wiring: see job `phase2-acceptance` in `.github/workflows/factors_ci.yml`.

---

## Coverage targets per WS10 deliverable

The new modules / surfaces delivered by WS10 carry an `>= 85%` line-coverage
target on the following packages:

- `greenlang/factors/quality/` (publish gates, validators)
- `greenlang/factors/data/ontology/loaders/` (geo / unit / activity / methodology loaders)
- `greenlang/factors/schemas/` (Pydantic mirror modules)

Coverage is measured per-job in CI and uploaded as
`factors-phase2-coverage-py{python-version}.xml` for review.

---

## Gap items resolved by this delivery

The four GAP rows in CTO §2.7 (`test_ontology_fk_enforcement`,
`test_api_query_factor_by_urn`, `test_sdk_fetch_by_urn`,
`test_provenance_checksum`) were authored in this deliverable. No outstanding
GAP rows remain at Phase 2 acceptance ship.

The single `IN-FLIGHT` row (`test_publish_rejection_matrix`) is owned by WS8.
WS10 does NOT modify the file. The aggregate runner reflects its live status:
once WS8 ships the suite, the runner upgrades the row to `PASS` automatically.
