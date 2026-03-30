# GreenLang V2 Migration Playbooks

## Purpose

Provide deterministic, low-risk migration steps from legacy or partial V1 implementations to V2 contracts, release trains, and governance.

## Command Standard

All playbooks use the canonical runtime invocation form:
- `python -m greenlang.cli.main v2 <command>`

## Playbook 1: App Runtime Migration (V1 -> V2)

1. Add `applications/<app>/v2/pack.yaml` and `applications/<app>/v2/gl.yaml`.
2. Set `contract_version: "2.0"` and `runtime: greenlang-v2`.
3. Register profile in `greenlang/v2/profiles.py`.
4. Add backend adapter support in `greenlang/v2/backends.py` and CLI mapping in `greenlang/cli/main.py`.
5. Validate:
   - `python -m greenlang.cli.main v2 validate-contracts`
   - `python -m greenlang.cli.main v2 runtime-checks`
   - `python -m greenlang.cli.main v2 gate`

## Playbook 2: Pack Tier Migration

1. Add pack entry to `greenlang/ecosystem/packs/v2_tier_registry.yaml`.
2. Ensure owner/support metadata and required evidence exist.
3. Promote tier only when evidence requirements pass:
   - candidate: docs evidence
   - supported: signed + security evidence
   - regulated-critical: determinism evidence
4. Validate:
   - `pytest -q tests/v2/test_pack_tiers.py tests/v2/test_pack_tier_lifecycle.py`
   - `python -c "from pathlib import Path; from greenlang.v2.pack_tiers import validate_tier_registry; p=Path('greenlang/ecosystem/packs/v2_tier_registry.yaml'); raise SystemExit(1 if validate_tier_registry(p) else 0)"`

## Playbook 3: Agent Lifecycle Migration

1. Register/normalize agents in `greenlang/agents/v2_agent_registry.yaml`.
2. For deprecated agents, set `deprecation_date` and `replacement_agent_id`.
3. Ensure no retired agent remains in active runtime paths.
4. Validate:
   - `python -m greenlang.cli.main v2 agent-checks`
   - `pytest -q tests/v2/test_agent_lifecycle_phase3.py`

## Playbook 4: Connector Reliability Migration

1. Add connector profile in `applications/connectors/v2_connector_registry.yaml`.
2. Map retry, timeout, circuit-breaker, idempotency settings.
3. Verify runtime binding through `BaseConnector` path.
4. Validate:
   - `python -m greenlang.cli.main v2 connector-checks`
   - `pytest -q tests/v2/test_connector_reliability_acceptance.py`

## Playbook 5: Release-Train Adoption

1. Ensure `.github/workflows/greenlang-v2-platform-ci.yml` and `v2-release-train.yml` are enabled.
2. Ensure UX quality blockers run via `.github/workflows/v2-frontend-ux-ci.yml`.
3. Run two consecutive green cycles and log in `docs/v2/RELEASE_TRAIN_CYCLE_LOG.md`.
4. Record RC soak and final board decision:
   - `docs/v2/RC_SOAK_LOG.md`
   - `docs/v2/GO_NO_GO_RECORD.md`

## Playbook 6: Regulated App Backend Parity (EUDR/GHG/ISO14064)

1. ensure each app has `v2/runtime_backend.py` and `v2/smoke_input.json`.
2. verify CLI execution with strict native mode:
   - `python -m greenlang.cli.main run eudr applications/GL-EUDR-APP/v2/smoke_input.json out/eudr`
   - `python -m greenlang.cli.main run ghg applications/GL-GHG-APP/v2/smoke_input.json out/ghg`
   - `python -m greenlang.cli.main run iso14064 applications/GL-ISO14064-APP/v2/smoke_input.json out/iso14064`
3. verify web execution:
   - `POST /api/v1/apps/eudr/run`
   - `POST /api/v1/apps/ghg/run`
   - `POST /api/v1/apps/iso14064/run`
4. verify parity tests:
   - `pytest -q tests/v2/test_v2_profile_backend_parity.py`
   - `pytest -q cbam-pack-mvp/tests/test_web_v2_eudr_ghg_iso_endpoints.py`

## Exit Condition

Migration is complete only when `python -m greenlang.cli.main v2 gate` is green and immutable evidence hashes are recorded in `docs/v2/IMMUTABLE_EVIDENCE_MANIFEST.json`.
