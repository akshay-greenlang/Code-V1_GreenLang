# GreenLang: Fix Remaining A-Grade Issues

## Current Status Assessment

### Already Fixed ✅
- pytest.ini paths - FIXED
- calculations/ module - Already consolidated (doesn't exist, just broken imports)
- calculators/ module - Already a deprecation shim
- database/ module - Already a deprecation shim pointing to db/
- configs/ module - Already doesn't exist

### Remaining Issues ❌
1. 77 subdirectories in greenlang/ (target: 10-15)
2. 3 broken CI/CD workflows
3. 80+ broken imports in industry module files
4. 96 TODO/FIXME markers
5. 72 GitHub workflows (consolidate to 15-20)

## CRITICAL CONSTRAINTS
- **DO NOT MODIFY**: `2026_PRD_MVP/` directory
- **DO NOT MODIFY**: `cbam-pack-mvp/` directory

---

## Task 1: Fix Broken Imports in Industry Module

The industry module has 80+ broken imports referencing non-existent `greenlang.calculators.*` paths.

### Files to Fix:
1. `greenlang/calculation/industry/industry/__init__.py` - Fix imports
2. `greenlang/calculation/industry/sb253/__init__.py` - Fix imports
3. `greenlang/calculation/industry/sb253/scope3/__init__.py` - Fix imports
4. All 15 category files in `greenlang/calculation/industry/sb253/scope3/`

### Change Pattern:
```python
# OLD (broken):
from greenlang.calculators.industry.steel_calculator import SteelEmissionCalculator

# NEW (correct - use relative imports):
from .steel_calculator import SteelEmissionCalculator
```

---

## Task 2: Fix Broken Test Imports

### Files to Fix:
1. `tests/unit/test_api_530_creep.py`
   - OLD: `from greenlang.calculations.api.api_530_creep import CreepLifeAssessor`
   - NEW: `from greenlang.calculation.physics.api.api_530_creep import CreepLifeAssessor`

2. `tests/unit/test_b31_1_pipe_stress.py`
   - OLD: `from greenlang.calculations.asme.b31_1_pipe_stress import ASMEB311PipeStress`
   - NEW: `from greenlang.calculation.physics.asme.b31_1_pipe_stress import ASMEB311PipeStress`

3. `tests/golden/test_steam_tables_golden.py`
   - OLD: `from greenlang.calculations.steam_tables import ...`
   - NEW: `from greenlang.calculation.physics.steam_tables import ...`

---

## Task 3: Fix 3 Broken CI/CD Workflows

### File 1: `.github/workflows/gl-001-ci.yaml`
- Change: `greenlang_2030/agent_foundation/agents/GL-001` → `applications/GL Agents/GL-001_Thermalcommand`

### File 2: `.github/workflows/gl-002-ci.yaml`
- Change: `greenlang_2030/agent_foundation/agents/GL-002` → `applications/GL Agents/GL-002*`

### File 3: `.github/workflows/vcci_production_deploy.yml`
- Change all: `GL-VCCI-Carbon-APP/` → `applications/GL-VCCI-Carbon-APP/`

---

## Task 4: Consolidate 77 Subdirectories to 15

Group the 77 greenlang/ subdirectories into 15 domain areas:

### Target Structure:
```
greenlang/
├── core/           # Merge: messaging, provenance, events, exceptions, runtime, sdk, compat
├── agents/         # Keep as-is (well organized)
├── api/            # Merge: middleware, schemas, services, routes
├── auth/           # Keep as-is
├── calculation/    # Keep as-is (primary calculation module)
├── compliance/     # Merge: standards, reporting, supply_chain
├── connectors/     # Merge: adapters, integrations
├── data/           # Merge: data_engineering, models, datasets
├── db/             # Keep as-is (primary database module)
├── intelligence/   # Merge: llm, ml, embeddings, vector_db
├── packs/          # Merge: cards, templates
├── pipeline/       # Merge: workflows
├── registry/       # Merge: hub
├── security/       # Merge: policy
├── cli/            # Keep as-is
└── utils/          # Merge: logging, cache, monitoring, observability
```

### Directories to Merge:
- `adapters/` → `connectors/adapters/`
- `benchmarks/` → Move to `tests/benchmarks/`
- `business/` → `compliance/business/`
- `cache/` → `utils/cache/`
- `cards/` → `packs/cards/`
- `compat/` → `core/compat/`
- `data_engineering/` → `data/engineering/`
- `datasets/` → `data/datasets/`
- `docs/` → Remove (use root docs/)
- `embeddings/` → `intelligence/embeddings/`
- `emission_factors/` → `calculation/emission_factors/`
- `events/` → `core/events/`
- `examples/` → Remove (use root examples/)
- `greenlang_registry/` → `registry/`
- `hub/` → `registry/hub/`
- `integrations/` → `connectors/integrations/`
- `llm/` → `intelligence/llm/`
- `logging/` → `utils/logging/`
- `messaging/` → `core/messaging/`
- `middleware/` → `api/middleware/`
- `ml/` → `intelligence/ml/`
- `models/` → `data/models/`
- `monitoring/` → `utils/monitoring/`
- `observability/` → `utils/observability/`
- `policy/` → `security/policy/`
- `provenance/` → `core/provenance/`
- `reporting/` → `compliance/reporting/`
- `runtime/` → `core/runtime/`
- `schemas/` → `api/schemas/`
- `services/` → `api/services/`
- `src/` → Distribute appropriately
- `standards/` → `compliance/standards/`
- `supply_chain/` → `compliance/supply_chain/`
- `templates/` → `packs/templates/`
- `tests/` → Move to root tests/
- `vector_db/` → `intelligence/vectordb/`
- `workflows/` → `pipeline/workflows/`

### For Each Move:
1. Create target directory if needed
2. Move files
3. Update ALL imports across entire codebase
4. Add deprecation re-export in old location (temporary)
5. Remove old directory after verification

---

## Task 5: Consolidate GitHub Workflows (72 → 15-20)

### Core Workflows to Keep:
1. `ci.yml` - Main CI pipeline
2. `tests.yml` - Test execution
3. `security-scan.yml` - Security scanning
4. `lint.yml` - Code quality
5. `release.yml` - Release management
6. `deploy-staging.yml` - Staging deployment
7. `deploy-production.yml` - Production deployment
8. `docs.yml` - Documentation build
9. `dependency-review.yml` - Dependency audit
10. `codeql.yml` - Code analysis

### Workflows to Archive:
Move obsolete/duplicate workflows to `.github/workflows/archive/`:
- All `gl-001-ci.yaml`, `gl-002-ci.yaml` etc. (merge into main ci.yml)
- Duplicate security scans
- Old deployment workflows

---

## Task 6: Address TODO/FIXME Markers

1. Find all markers: `grep -r "TODO\|FIXME\|HACK\|XXX" greenlang/ --include="*.py"`
2. Categorize by priority:
   - Security-related: Fix immediately
   - Import-related: Fix as part of consolidation
   - Feature TODOs: Create GitHub issues
   - Deprecated code: Remove or document
3. Create `docs/TECHNICAL_DEBT.md` for tracking remaining items
4. Target: Reduce from 96 to <20

---

## Success Criteria

- [ ] greenlang/ subdirectories: ≤15 (currently 77)
- [ ] Broken imports: 0 (currently 80+)
- [ ] Broken CI workflows: 0 (currently 3)
- [ ] GitHub workflows: 15-20 (currently 72)
- [ ] TODO markers: <20 (currently 96)
- [ ] All tests pass
- [ ] All imports resolve correctly

## Target Grade: A
