# GreenLang A-Grade Codebase Fix PRD

## Overview
Fix all remaining issues to achieve A-grade professional codebase status.

## CRITICAL CONSTRAINTS
- **DO NOT MODIFY**: `2026_PRD_MVP/` directory
- **DO NOT MODIFY**: `cbam-pack-mvp/` directory
- All changes must preserve existing functionality
- Update ALL imports after moving/merging files
- Run tests after each major change
- Create backward-compatible re-exports where needed

---

## Phase 1: Fix Module Duplication (CRITICAL)

### Task 1.1: Consolidate Calculation Modules
Merge three calculation modules into ONE authoritative module:

**Current State:**
- `greenlang/calculation/` (13 files) - KEEP AS PRIMARY
- `greenlang/calculations/` (9 directories) - MERGE INTO calculation/
- `greenlang/calculators/` (2 directories) - MERGE INTO calculation/

**Steps:**
1. Audit all three directories for unique content
2. Create unified structure under `greenlang/calculation/`:
   ```
   greenlang/calculation/
   ├── __init__.py          # Re-export all public APIs
   ├── core/                 # Core calculation engines
   ├── emission_factors/     # Factor databases
   ├── carbon/               # Carbon calculations
   ├── energy/               # Energy calculations
   ├── physics/              # Physics calculations (steam tables, etc.)
   ├── thermal/              # Thermal calculations
   └── utils/                # Calculation utilities
   ```
3. Move unique code from `calculations/` to `calculation/`
4. Move unique code from `calculators/` to `calculation/`
5. Update ALL imports across the entire codebase:
   - `from greenlang.calculations import X` → `from greenlang.calculation import X`
   - `from greenlang.calculators import X` → `from greenlang.calculation import X`
6. Create deprecation re-exports in old locations (temporary backward compatibility)
7. Delete empty/duplicate directories after verification
8. Run tests to verify nothing broke

### Task 1.2: Consolidate Config Modules
Merge config modules into ONE:

**Current State:**
- `greenlang/config/` (7 files) - KEEP AS PRIMARY
- `greenlang/configs/` - MERGE INTO config/

**Steps:**
1. Audit both directories
2. Move unique content from `configs/` to `config/`
3. Update all imports: `from greenlang.configs import X` → `from greenlang.config import X`
4. Add deprecation re-export in `configs/__init__.py`
5. Remove `configs/` directory
6. Run tests

### Task 1.3: Consolidate Database Modules
Merge database modules into ONE:

**Current State:**
- `greenlang/database/` (4 files) - KEEP AS PRIMARY
- `greenlang/db/` (7 files) - MERGE INTO database/

**Steps:**
1. Audit both directories
2. Move unique content from `db/` to `database/`
3. Update all imports: `from greenlang.db import X` → `from greenlang.database import X`
4. Add deprecation re-export in `db/__init__.py`
5. Remove `db/` directory
6. Run tests

---

## Phase 2: Reduce Subdirectories (76 → 15)

### Task 2.1: Group Related Modules into Domain Areas
Reorganize greenlang/ from 76 subdirectories to ~15 domain areas:

**Target Structure:**
```
greenlang/
├── __init__.py
├── _version.py
│
├── core/                    # Core runtime & messaging
│   ├── messaging/           # (merge from greenlang/messaging/)
│   ├── provenance/          # (merge from greenlang/provenance/)
│   ├── events/              # (merge from greenlang/events/)
│   └── exceptions/          # (merge from greenlang/exceptions/)
│
├── agents/                  # Agent system (keep as-is, well organized)
│
├── api/                     # API layer
│   ├── routes/              # (keep from api/)
│   ├── middleware/          # (merge from greenlang/middleware/)
│   └── schemas/             # (merge from greenlang/schemas/)
│
├── auth/                    # Authentication & authorization (keep)
│
├── calculation/             # Unified calculations (from Phase 1)
│
├── compliance/              # Regulatory compliance
│   ├── standards/           # (merge from greenlang/standards/)
│   ├── reporting/           # (merge from greenlang/reporting/)
│   └── frameworks/          # CSRD, CBAM, etc.
│
├── connectors/              # Data connectors
│   ├── adapters/            # (merge from greenlang/adapters/)
│   ├── integrations/        # (merge from greenlang/integrations/)
│   └── sources/             # Data source connectors
│
├── data/                    # Data handling
│   ├── engineering/         # (merge from greenlang/data_engineering/)
│   ├── models/              # (merge from greenlang/models/)
│   ├── loaders/             # Data loading
│   └── validators/          # Data validation
│
├── database/                # Database layer (unified from Phase 1)
│
├── intelligence/            # AI/ML capabilities
│   ├── llm/                 # (merge from greenlang/llm/)
│   ├── ml/                  # (merge from greenlang/ml/)
│   ├── embeddings/          # (merge from greenlang/embeddings/)
│   └── vectordb/            # (merge from greenlang/vector_db/)
│
├── packs/                   # Pack system (keep as-is)
│
├── pipeline/                # Pipeline orchestration (keep)
│
├── registry/                # Agent/component registry (keep)
│
├── security/                # Security layer (keep)
│
├── cli/                     # Command line interface (keep)
│
└── utils/                   # Utilities
    ├── logging/             # (merge from greenlang/logging/)
    ├── cache/               # (merge from greenlang/cache/)
    └── helpers/             # General helpers
```

**Directories to Merge/Relocate:**
- `adapters/` → `connectors/adapters/`
- `benchmarks/` → `tests/benchmarks/` (move to tests)
- `business/` → `compliance/business/`
- `cache/` → `utils/cache/`
- `cards/` → `packs/cards/`
- `compat/` → `core/compat/`
- `data_engineering/` → `data/engineering/`
- `datasets/` → `data/datasets/`
- `docs/` → Remove (internal docs should be in root docs/)
- `embeddings/` → `intelligence/embeddings/`
- `emission_factors/` → `calculation/emission_factors/`
- `events/` → `core/events/`
- `examples/` → Remove (examples should be in root examples/)
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
- `sdk/` → `core/sdk/`
- `services/` → `api/services/`
- `src/` → Distribute contents appropriately
- `standards/` → `compliance/standards/`
- `supply_chain/` → `compliance/supply_chain/`
- `templates/` → `packs/templates/`
- `tests/` → Move to root tests/
- `vector_db/` → `intelligence/vectordb/`
- `workflows/` → `pipeline/workflows/`

### Task 2.2: Update All Imports
After reorganization, update ALL imports across the codebase:
1. Use automated tools (rope, autoflake) where possible
2. Search and replace common import patterns
3. Verify with `python -c "import greenlang"`
4. Run full test suite

---

## Phase 3: Fix CI/CD Workflows

### Task 3.1: Fix Broken Workflow Paths
Update these workflow files with correct paths:

**File: `.github/workflows/gl-001-ci.yaml`**
- Change: `greenlang_2030/agent_foundation/agents/GL-001` → `applications/GL Agents/GL-001_Thermalcommand`

**File: `.github/workflows/gl-002-ci.yaml`**
- Change: `greenlang_2030/agent_foundation/agents/GL-002` → `applications/GL Agents/GL-002_*`

**File: `.github/workflows/vcci_production_deploy.yml`**
- Change: `GL-VCCI-Carbon-APP/` → `applications/GL-VCCI-Carbon-APP/`

### Task 3.2: Consolidate Redundant Workflows
Audit and consolidate 72 workflows to ~15-20:

1. Identify duplicate/similar workflows
2. Merge workflows by function:
   - `ci.yml` - Main CI for all code
   - `tests.yml` - Test execution
   - `security.yml` - Security scanning
   - `lint.yml` - Code quality
   - `deploy-*.yml` - Deployment workflows
   - `release.yml` - Release management
3. Archive obsolete workflows (move to `.github/workflows/archive/`)
4. Update workflow references

---

## Phase 4: Reduce Technical Debt

### Task 4.1: Address Critical TODO/FIXME Markers
Find and resolve high-priority TODO/FIXME comments:

```bash
# Find all TODO/FIXME markers
grep -r "TODO\|FIXME\|HACK\|XXX" greenlang/ --include="*.py" | wc -l
```

Priority order:
1. Security-related TODOs
2. Import-related TODOs
3. Deprecated code TODOs
4. Performance TODOs
5. Feature TODOs (create issues for these)

### Task 4.2: Create Issues for Remaining TODOs
For TODOs that can't be immediately resolved:
1. Document in `docs/TECHNICAL_DEBT.md`
2. Categorize by priority
3. Assign target versions for resolution

---

## Phase 5: Update Makefile and Build Configs

### Task 5.1: Update Makefile
Fix references to moved files:
- Update script paths to `scripts/` directory
- Update test paths
- Verify all targets work

### Task 5.2: Update pyproject.toml
Ensure package discovery works with new structure:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["greenlang*"]
```

### Task 5.3: Standardize Line Length
Choose ONE line length standard (recommend 88):
- Update `.flake8` to use 88
- Update `pyproject.toml` black config
- Run formatters to apply

---

## Phase 6: Final Validation

### Task 6.1: Run Full Test Suite
```bash
pytest tests/ -v --tb=short
```
- All tests must pass
- Document any expected failures

### Task 6.2: Verify Import Resolution
```bash
python -c "import greenlang; print('OK')"
python -c "from greenlang.calculation import *; print('OK')"
python -c "from greenlang.database import *; print('OK')"
python -c "from greenlang.config import *; print('OK')"
```

### Task 6.3: Count Subdirectories
```bash
ls -d greenlang/*/ | wc -l
# Target: ≤15
```

### Task 6.4: Count TODO Markers
```bash
grep -r "TODO\|FIXME" greenlang/ --include="*.py" | wc -l
# Target: <20
```

---

## Success Criteria

- [ ] Root directory: <30 items ✓ (already achieved)
- [ ] greenlang/ subdirectories: ≤15 (currently 76)
- [ ] Duplicate modules: 0 (currently 3 sets)
- [ ] Broken CI workflows: 0 (currently 3)
- [ ] TODO/FIXME markers: <20 (currently 96)
- [ ] All tests pass
- [ ] All imports resolve
- [ ] Package builds successfully

## Target Grade: A

## Rollback Plan
- Each phase should be a separate commit
- Tag before starting: `git tag v0.3.0-pre-a-grade`
- Can revert individual phases if needed
