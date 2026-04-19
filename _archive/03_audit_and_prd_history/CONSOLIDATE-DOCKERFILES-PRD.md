# CONSOLIDATE-DOCKERFILES-PRD

Validation PRD for the Dockerfile template consolidation work. The GreenLang project
consolidated 100+ Dockerfiles into 3 parameterized multi-stage templates, deleted 40+
redundant files, and introduced a centralized build script.

**Before**: 126 Dockerfile* files scattered across agents, apps, generated artifacts,
GreenLang Development mirrors, deployment variants, and docs/planning references.

**After**: 3 parameterized templates + build script + surviving Dockerfiles that serve
distinct, non-redundant purposes (app frontends, normalizer infrastructure, etc.).

---

## Task 1: Verify template files exist and have correct structure

- Check `deployment/docker/templates/Dockerfile.agent` exists
- Check `deployment/docker/templates/Dockerfile.cli` exists
- Check `deployment/docker/templates/Dockerfile.api` exists
- Check `deployment/docker/templates/README.md` exists
- Check `deployment/docker/templates/.dockerignore.template` exists
- Verify `Dockerfile.agent` has multi-stage build structure: `FROM ... AS base`, `FROM ... AS builder`, `FROM ... AS production`, `FROM ... AS development`, `FROM ... AS test`
- Verify `Dockerfile.agent` contains `ARG AGENT_ID`, `ARG AGENT_NAME`, `ARG APP_MODULE`, `ARG HEALTH_PATH`
- Verify `Dockerfile.agent` contains `HEALTHCHECK`, `ENTRYPOINT`, `USER glagent` (non-root), `LABEL org.opencontainers`
- Verify `Dockerfile.cli` has multi-stage build structure with `FROM ... AS base` and `FROM ... AS production`
- Verify `Dockerfile.cli` contains `ARG GL_VERSION` or equivalent version parameterization
- Verify `Dockerfile.api` has multi-stage build structure with `FROM ... AS base` and `FROM ... AS production`
- Verify `Dockerfile.api` contains `ARG APP_NAME` or equivalent application parameterization
- Verify all three templates use `tini` as PID 1 entrypoint
- Verify all three templates use non-root users
- Verify all three templates have `PIP_NO_CACHE_DIR=1` for reproducible builds

## Task 2: Verify build script exists and works

- Check `scripts/docker_build.py` exists
- Run: `python scripts/docker_build.py --help`
- Verify help output shows supported subcommands: `agent`, `cli`, `app`, `list`, `audit`
- Run: `python scripts/docker_build.py list`
- Verify all GL agents are listed (GL-001 through GL-017, excluding GL-013/GL-015/GL-016 if not present)
- Verify DATA agents are listed (duplicate-detector, missing-value-imputer, outlier-detector, time-series-gap-filler, cross-source-reconciliation, data-freshness-monitor, schema-migration, data-lineage-tracker, validation-rule-engine, climate-hazard)
- Verify CLI variants are listed
- Verify application services are listed (CSRD, CBAM, VCCI backend/worker)
- Verify the script maps each service to the correct template (Dockerfile.agent, Dockerfile.cli, or Dockerfile.api)

## Task 3: Verify GL Agent Dockerfiles deleted

These per-agent Dockerfiles are replaced by `Dockerfile.agent` with `--build-arg AGENT_ID=...`:

- Verify `applications/GL Agents/GL-001_Thermalcommand/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-002_Flameguard/deployment/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-003_UnifiedSteam/deployment/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-003_UnifiedSteam/deployment/Dockerfile.production` does NOT exist
- Verify `applications/GL Agents/GL-004_Burnmaster/deployment/docker/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-005_Combusense/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-006_HEATRECLAIM/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-007_FurnacePulse/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-008_Trapcatcher/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-008_Trapcatcher/deployment/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-009_ThermalIQ/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-010_EmissionGuardian/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-011_FuelCraft/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-012_SteamQual/deploy/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-014_Exchangerpro/deploy/Dockerfile` does NOT exist
- Verify `applications/GL Agents/GL-017_Condensync/Dockerfile` does NOT exist
- Verify `applications/GL Agents/Framework_GreenLang/deployment/docker/Dockerfile.template` does NOT exist
- Count remaining files under `applications/GL Agents/` -- zero Dockerfile* files expected

## Task 4: Verify DATA agent Dockerfiles deleted from deployment/docker

These per-agent Dockerfiles are replaced by `Dockerfile.agent` with `--build-arg AGENT_NAME=...`:

- Verify `deployment/docker/Dockerfile.duplicate-detector` does NOT exist
- Verify `deployment/docker/Dockerfile.missing-value-imputer` does NOT exist
- Verify `deployment/docker/Dockerfile.outlier-detector` does NOT exist
- Verify `deployment/docker/Dockerfile.time-series-gap-filler` does NOT exist
- Verify `deployment/docker/Dockerfile.cross-source-reconciliation` does NOT exist
- Verify `deployment/docker/Dockerfile.data-freshness-monitor` does NOT exist
- Verify `deployment/docker/Dockerfile.schema-migration` does NOT exist
- Verify `deployment/docker/Dockerfile.data-lineage-tracker` does NOT exist
- Verify `deployment/docker/Dockerfile.validation-rule-engine` does NOT exist
- Verify `deployment/docker/Dockerfile.climate-hazard` does NOT exist
- Verify `deployment/docker/Full.Dockerfile` does NOT exist (consolidated into templates)
- Verify `deployment/docker/Runner.Dockerfile` does NOT exist (consolidated into Dockerfile.cli)
- Count remaining files under `deployment/docker/` -- only `base/Dockerfile.base`, `base/requirements-base.txt`, `.dockerignore`, `weaviate/` dir, and `templates/` dir expected

## Task 5: Verify generated artifact Dockerfiles deleted

These are auto-generated artifacts that should not exist as standalone Dockerfiles:

- Verify `reports/results/artifacts/generated/fuel_analyzer_agent/Dockerfile` does NOT exist
- Verify `reports/results/artifacts/generated/carbon_intensity_v1/Dockerfile` does NOT exist
- Verify `reports/results/artifacts/generated/energy_performance_v1/Dockerfile` does NOT exist
- Verify `reports/results/artifacts/generated/eudr_compliance_v1/Dockerfile` does NOT exist
- Count Dockerfiles under `reports/results/artifacts/generated/` -- should be 0

## Task 6: Verify GreenLang Development mirror Dockerfiles deleted

The `GreenLang Development/` directory contained exact mirrors of Dockerfiles already
present under `applications/`, `deployment/`, and `greenlang/`. All are redundant:

- Verify no `Dockerfile*` files exist under `GreenLang Development/02-Applications/GL Agents/`
- Verify no `Dockerfile*` files exist under `GreenLang Development/02-Applications/GL-Agent-Factory/`
- Verify no `Dockerfile*` files exist under `GreenLang Development/08-Deployment/`
- Verify no `Dockerfile*` files exist under `GreenLang Development/01-Core-Platform/`
- Count all `Dockerfile*` files under `GreenLang Development/` -- should be 0

## Task 7: Verify runner variant Dockerfiles consolidated

Multiple runner variants existed across the project. They should be consolidated into
`Dockerfile.cli` or a single runner Dockerfile:

- Verify `deployment/Dockerfile.runner` and `deployment/Dockerfile.runner.optimized` are either deleted or one is the canonical survivor
- If both still exist, verify one references the other or document why both are needed
- Verify `deployment/Dockerfile.core`, `deployment/Dockerfile.full`, `deployment/Dockerfile.secure` are either deleted or documented as intentionally distinct (different dep profiles)
- Verify `deployment/Dockerfile.api` either deleted (replaced by template) or is the template itself
- Verify `deployment/Dockerfile.registry` either deleted (replaced by template) or documented

## Task 8: Verify deployment-level Dockerfiles rationalized

The root and deployment directories had overlapping Dockerfiles:

- Check if root `Dockerfile` still exists -- if so, verify it delegates to or references a template
- Verify `deployment/kubernetes/Dockerfile` either deleted or documented as K8s-specific
- Verify `greenlang/integration/api/Dockerfile` either deleted or documented
- Verify `greenlang/config/greenlang_registry/Dockerfile` either deleted or documented
- Verify `greenlang/tests/templates/data-intake-app/Dockerfile` kept (test fixture, not production)

## Task 9: Verify docs/planning Dockerfiles are NOT deleted

These are reference/planning documents, not production Dockerfiles. They should survive:

- Verify `docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-*/Dockerfile` files still exist (11+ reference Dockerfiles)
- Verify `docs/planning/greenlang-2030-vision/agent_foundation/deployment/dockerfiles/Dockerfile.*` files still exist (5 reference Dockerfiles)
- These are vision documents and must NOT be deleted during consolidation

## Task 10: Verify application Dockerfiles rationalized

Application Dockerfiles serve distinct frontend/backend/worker roles and may not be
fully replaceable by templates. Validate the decision for each:

- Verify `applications/GL-CSRD-APP/CSRD-Reporting-Platform/Dockerfile` either migrated to Dockerfile.api template or documented as app-specific
- Verify `applications/GL-CBAM-APP/CBAM-Importer-Copilot/Dockerfile` either migrated to Dockerfile.api template or documented as app-specific
- Verify `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend/Dockerfile` either migrated to Dockerfile.api template or documented
- Verify `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/frontend/Dockerfile` kept (frontend, not covered by Python templates)
- Verify `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/worker/Dockerfile` either migrated to Dockerfile.agent template or documented
- Verify `applications/GL-Agent-Factory/Dockerfile` either migrated or documented
- Verify `applications/GL-Agent-Factory/docker/` subdirectory Dockerfiles either deleted or consolidated
- Verify `applications/GL-Agent-Factory/cli/templates/agent/{{cookiecutter.agent_id}}/Dockerfile.j2` kept (cookiecutter template, distinct purpose)
- Verify `applications/GL-016/Dockerfile` and `applications/GL-017/Dockerfile` either deleted or migrated

## Task 11: Verify greenlang-normalizer Dockerfiles rationalized

The normalizer has its own infrastructure Dockerfiles:

- Check if `greenlang-normalizer/infrastructure/docker/Dockerfile.core` still exists
- Check if `greenlang-normalizer/infrastructure/docker/Dockerfile.service` still exists
- Check if `greenlang-normalizer/infrastructure/docker/Dockerfile.review-console` still exists
- These may be legitimately distinct (separate project) -- document decision either way

## Task 12: Verify templates are syntactically valid

Run quick validation on each template to confirm they are parseable and contain
required directives:

- Run: `python -c "with open('deployment/docker/templates/Dockerfile.agent') as f: c=f.read(); assert 'FROM' in c; assert 'ARG AGENT_ID' in c; assert 'HEALTHCHECK' in c; assert 'USER glagent' in c; assert 'ENTRYPOINT' in c; print('Dockerfile.agent: PASS')"`
- Run: `python -c "with open('deployment/docker/templates/Dockerfile.cli') as f: c=f.read(); assert 'FROM' in c; assert 'ENTRYPOINT' in c; print('Dockerfile.cli: PASS')"`
- Run: `python -c "with open('deployment/docker/templates/Dockerfile.api') as f: c=f.read(); assert 'FROM' in c; assert 'ENTRYPOINT' in c or 'CMD' in c; print('Dockerfile.api: PASS')"`
- Run: `docker build --check -f deployment/docker/templates/Dockerfile.agent .` (if Docker BuildKit available, validates syntax without building)

## Task 13: Count final Dockerfile inventory

After all deletions, the total Dockerfile count should be significantly reduced:

- Count all `Dockerfile*` files in the entire repo
- Expected: fewer than 50 (down from 126 pre-consolidation)
- Breakdown expected:
  - `deployment/docker/templates/`: 3 (agent, cli, api)
  - `deployment/docker/base/`: 1 (Dockerfile.base)
  - `deployment/`: 0-3 (runner, core, secure -- if any survive)
  - `applications/`: 5-10 (app-specific frontends, cookiecutter template, Agent Factory)
  - `docs/planning/`: ~21 (reference docs, untouched)
  - `greenlang-normalizer/`: 3 (separate project)
  - `greenlang/`: 0-2 (test fixtures)
  - Root `Dockerfile`: 0-1
  - `reports/`: 0
  - `GreenLang Development/`: 0
- Print the full list and verify each surviving Dockerfile has a documented reason to exist

## Task 14: Run audit to check coverage

- Run: `python scripts/docker_build.py audit`
- Verify audit identifies all remaining Dockerfiles in the repo
- Verify audit classifies each as: `template`, `consolidated`, `app-specific`, `reference`, `test-fixture`
- Verify no Dockerfiles are classified as `replaceable` (meaning they should have been consolidated but were missed)
- Verify audit reports the consolidation ratio (e.g., "126 -> 45 Dockerfiles, 64% reduction")

## Task 15: Verify docker-compose integration

- Check if `docker-compose.yml` or `docker-compose.yaml` exists at project root or `deployment/`
- If it exists, verify it references the template Dockerfiles (not deleted per-agent Dockerfiles)
- Verify build contexts point to correct paths
- Verify build args are passed correctly for agent services
- Run: `docker compose config` (dry-run validation, no actual build)
