# GreenLang Codebase Reorganization PRD

## Overview
Strategic reorganization of the GreenLang Climate Operating System codebase to improve maintainability, discoverability, and developer experience.

## CRITICAL CONSTRAINTS
- **DO NOT MODIFY**: `2026_PRD_MVP/` directory
- **DO NOT MODIFY**: `cbam-pack-mvp/` directory
- All changes must preserve existing functionality
- Update all imports after moving files
- Run tests after each major reorganization phase

---

## Phase 1: Clean Root Directory Pollution

### Task 1.1: Remove Malformed Version Files
Delete malformed files at root that appear to be pip/setuptools artifacts:
- `=2.2.0`
- `=22.2.0`
- `=4.12`
- `=4.30.0`
- Any other files starting with `=`

### Task 1.2: Create scripts/ Directory Structure
Create a `scripts/` directory and move all root-level Python utility scripts:
```
scripts/
├── dev/           # Development utilities
├── maintenance/   # Fix scripts
└── testing/       # Test runners
```

Move these files:
- `fix_*.py` files -> `scripts/maintenance/`
- `generate_*.py` files -> `scripts/dev/`
- `run_*.py` files -> `scripts/dev/`
- Root-level `test_*.py` files -> `scripts/testing/`

### Task 1.3: Organize Root Documentation
Create `docs/reports/` and `docs/status/` directories. Move:
- All `*_STATUS.md` files -> `docs/status/`
- All `*_REPORT.md` and `*_REPORT.txt` files -> `docs/reports/`
- All `*_LOG.md` files -> `docs/reports/`
- Keep only essential docs at root: README.md, CONTRIBUTING.md, LICENSE, CHANGELOG.md, SECURITY.md

### Task 1.4: Consolidate GL Status Files
Move GL-prefixed status/plan markdown files to appropriate locations:
- `GL_5_YEAR_PLAN.md` -> `docs/strategy/`
- `GL-*-STATUS.md` files -> `docs/status/`
- `GL-*-SUMMARY.md` files -> `docs/status/`

---

## Phase 2: Consolidate Duplicate Modules in greenlang/

### Task 2.1: Merge Calculation Modules
Consolidate the three calculation-related directories into one:
```
greenlang/calculations/  (keep this as primary)
├── engines/             # Core calculation engines
├── emission_factors/    # Factor databases
├── carbon/              # Carbon-specific calcs
├── energy/              # Energy calcs
└── utilities/           # Calculation helpers
```

Steps:
1. Audit contents of `greenlang/calculation/`, `greenlang/calculations/`, `greenlang/calculators/`
2. Create unified structure under `greenlang/calculations/`
3. Move unique code from `calculation/` and `calculators/` to `calculations/`
4. Update all imports across the codebase
5. Remove empty/duplicate directories
6. Add deprecation notices if needed for backward compatibility

### Task 2.2: Merge Configuration Directories
Consolidate into single config module:
```
greenlang/config/  (keep this as primary)
├── settings/      # App settings
├── schemas/       # Config schemas/validation
└── defaults/      # Default configurations
```

Steps:
1. Audit `greenlang/config/` and `greenlang/configs/`
2. Merge into `greenlang/config/`
3. Update all imports
4. Remove `greenlang/configs/`

### Task 2.3: Consolidate Database Directories
Merge `greenlang/database/` and `greenlang/db/` into one:
```
greenlang/database/  (keep as primary)
├── models/         # ORM models
├── migrations/     # Alembic migrations
├── repositories/   # Data access layer
└── connections/    # Connection management
```

### Task 2.4: Rationalize Data Directories
Create clear hierarchy for data-related directories:
```
greenlang/data/  (primary for runtime data handling)
├── engineering/   # Merge from data_engineering/
├── loaders/       # Data loading utilities
├── transformers/  # Data transformation
└── validators/    # Data validation

datasets/  (root level - static reference data)
├── emission_factors/
├── knowledge_base/
└── examples/
```

Remove/merge:
- `greenlang/datasets/` -> merge into root `datasets/`
- `greenlang/data_engineering/` -> merge into `greenlang/data/engineering/`
- Root `data/` -> merge into `datasets/` if static, or `greenlang/data/` if runtime

---

## Phase 3: Consolidate Core Modules

### Task 3.1: Resolve Core Module Duplication
Audit and consolidate:
- `core/greenlang/` (legacy)
- `greenlang/core/` (current)

Steps:
1. Identify unique functionality in `core/greenlang/`
2. Migrate any unique code to `greenlang/core/`
3. Update imports throughout codebase
4. Create backward-compatible re-exports if needed
5. Deprecate `core/greenlang/` with clear migration path

### Task 3.2: Consolidate Examples
Merge example directories:
```
examples/  (root level - primary location)
├── quickstart/
├── tutorials/
├── integrations/
├── applications/
└── advanced/
```

Move `greenlang/examples/` content to root `examples/` directory.

---

## Phase 4: Organize Test Infrastructure

### Task 4.1: Consolidate Test Directories
Organize test structure:
```
tests/  (primary test directory)
├── unit/
├── integration/
├── e2e/
├── fixtures/
├── packs/          # Move from test-pack/, test-gpl-pack/, test-mit-pack/
└── load/           # Move from load-tests/

test-reports/       # Keep separate - generated output
test-results/       # Keep separate - generated output
```

### Task 4.2: Clean Test Artifacts
- Move `test-pack/`, `test-gpl-pack/`, `test-mit-pack/`, `test-scaffold-pack/` contents to `tests/packs/`
- Move `load-tests/` to `tests/load/`
- Ensure test output directories are in `.gitignore`

---

## Phase 5: Deployment Consolidation

### Task 5.1: Unify Deployment Directories
Create clear deployment hierarchy:
```
deployment/  (primary)
├── docker/         # Move from root docker/
├── kubernetes/     # Consolidate k8s/ and kubernetes/
├── helm/           # Keep as-is
├── terraform/      # Keep as-is
├── kustomize/      # Keep as-is
└── cloud/          # Move from cc_deployment/
```

### Task 5.2: Consolidate Docker Files
Move all Dockerfiles to `deployment/docker/`:
- `Dockerfile.*` variants -> `deployment/docker/`
- Update docker-compose files to reference new paths
- Keep `docker-compose.yml` at root for convenience (update paths)

---

## Phase 6: Establish Naming Conventions

### Task 6.1: Standardize Module Naming
Establish convention: **Use singular names for modules** (Python standard)
- Keep: `benchmark`, `config`, `adapter`
- Rename where appropriate: `benchmarks/` -> `benchmark/` (if both don't exist)

### Task 6.2: Document Structure
Create/update `docs/architecture/DIRECTORY_STRUCTURE.md` documenting:
- Purpose of each top-level directory
- Naming conventions
- Where to add new code
- Import patterns

---

## Phase 7: Final Cleanup

### Task 7.1: Update All Imports
Run automated import updater across entire codebase:
1. Use tools like `rope` or `autoflake` to update imports
2. Run `isort` to organize imports
3. Verify no broken imports with `python -c "import greenlang"`

### Task 7.2: Update CI/CD Paths
Update any hardcoded paths in:
- `.github/workflows/*.yml`
- `Makefile`
- `pyproject.toml`
- Docker files

### Task 7.3: Validate and Test
1. Run full test suite: `pytest tests/`
2. Run linting: `ruff check .`
3. Run type checking: `mypy greenlang/`
4. Build Docker images to verify paths
5. Run any integration tests

### Task 7.4: Create Migration Guide
Document all changes in `docs/MIGRATION_GUIDE.md`:
- Old path -> New path mappings
- Updated import statements
- Any deprecated modules
- Timeline for removing backward compatibility shims

---

## Success Criteria
- [ ] Root directory has < 30 files/directories
- [ ] No duplicate module directories
- [ ] All tests pass
- [ ] All imports resolve correctly
- [ ] Documentation updated
- [ ] No breaking changes to public API

## Rollback Plan
- Create git branch `pre-reorganization` before starting
- Tag current state as `v0.3.0-pre-reorg`
- Each phase should be a separate commit for easy rollback
