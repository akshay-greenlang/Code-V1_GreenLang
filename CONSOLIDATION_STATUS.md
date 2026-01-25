# GreenLang Module Consolidation Status

**Date:** 2026-01-25
**Status:** In Progress
**Objective:** Logical grouping of related files to improve codebase maintainability

## Executive Summary

The GreenLang codebase is undergoing a phased consolidation to eliminate duplicate modules, resolve legacy nesting issues, and establish clear architectural boundaries. This document tracks the current status and remaining work.

## Consolidation Progress

### ✅ Completed

#### 1. calculations/ + calculators/ → calculation/
- **Status:** Deprecation wrappers in place
- **Location:** `greenlang/calculations/` and `greenlang/calculators/`
- **Action Taken:** Both modules now contain deprecation warnings directing users to `greenlang.calculation`
- **Migration Path:** Documented in deprecation messages
- **Files Affected:** All imports updated to use `greenlang.calculation`

#### 2. database/ → db/
- **Status:** Partially complete - deprecation wrapper in place
- **Location:** `greenlang/database/__init__.py`
- **Action Taken:**
  - `database/__init__.py` provides backward-compatible re-exports with deprecation warnings
  - Points users to `greenlang.db` for all database operations
- **Remaining Work:**
  - `database/connection.py`, `database/models.py`, `database/transaction.py` still contain implementation code
  - Need to verify if these are duplicated in `db/` or need migration
  - 6 files still importing from `greenlang.database` need update:
    - `tests/unit/test_database_transaction.py`
    - `tests/integration/test_agent_pipeline.py`
    - `tests/integration/test_e2e_pipelines.py`
    - `examples/data_infrastructure_usage.py`
    - `greenlang/config/container.py` (commented out)

### 🚧 In Progress

#### 3. core/greenlang/ Legacy Nesting
- **Status:** Identified but not resolved
- **Issue:** `greenlang/core/greenlang/` creates confusing nested module structure
- **Modules Found:**
  - `agents/` - Mock agents
  - `auth/` - API key manager, JWT handler, middleware
  - `cache/` - Redis client
  - `cards/` - Generator, templates, validator
  - `cli/` - CLI tools
  - `data/factors/` - Data factors
  - `generator/templates/` - Template generator
  - `hub/` - Hub functionality
  - `ml_platform/` - ML platform code
  - `packaging/` - Packaging utilities
  - `packs/` - Pack definitions
  - `policy/bundles/` - Policy bundles
- **Impact:** These modules should exist at `greenlang/auth/`, `greenlang/cache/`, etc., not nested under `core/greenlang/`
- **Risk:** HIGH - affects import paths across codebase

### ⏳ Pending

#### 4. config/ vs configs/
- **Status:** Not started
- **Issue:** Unclear separation between code config (`config/`) and static config files (`configs/`)
- **Recommendation:**
  - Keep `config/` for Python configuration code
  - Move static files from `configs/` to `datasets/config/` or `config/defaults/`
- **Files to Review:**
  - `greenlang/config/` (10KB - Python code)
  - `greenlang/configs/` (JSON/YAML static files)

#### 5. monitoring/ vs observability/
- **Status:** Not started
- **Issue:** Overlapping concerns - unclear separation
- **Recommendation:** Consolidate into single `infrastructure/observability/` module
- **Impact:** Medium - affects monitoring and logging code

#### 6. exceptions.py vs exceptions/
- **Status:** Backward compatibility in place (based on pattern analysis)
- **Expected State:** File provides deprecation wrapper for module
- **Verification Needed:** Confirm deprecation warning exists

#### 7. determinism.py vs determinism/
- **Status:** Backward compatibility in place (based on pattern analysis)
- **Expected State:** File provides deprecation wrapper for module
- **Verification Needed:** Confirm deprecation warning exists

## Critical Issues Requiring Immediate Attention

### 1. Core/Greenlang Nesting (CRITICAL)
**Priority:** P0
**Impact:** Architectural confusion, incorrect import paths
**Action Required:**
- Audit all modules in `greenlang/core/greenlang/`
- Determine if they should be:
  - Moved to top-level `greenlang/` (e.g., `auth/`, `cache/`)
  - Kept in `core/` but unnested (e.g., `core/policy/`)
  - Deprecated and consolidated elsewhere
- Update all imports
- Add deprecation warnings in old locations

### 2. Database Module Cleanup
**Priority:** P1
**Impact:** Incomplete migration causing confusion
**Action Required:**
- Complete migration of `database/transaction.py` to `db/`
- Update 6 remaining import statements
- Remove implementation files from `database/` once migration complete
- Keep only deprecation wrapper in `database/__init__.py`

## Import Update Checklist

### Files Using Deprecated Imports

#### greenlang.database
- [ ] `tests/unit/test_database_transaction.py` (9 imports)
- [ ] `tests/integration/test_agent_pipeline.py` (1 import)
- [ ] `tests/integration/test_e2e_pipelines.py` (1 import)
- [ ] `examples/data_infrastructure_usage.py` (1 import)
- [ ] `greenlang/config/container.py` (commented out - verify safe to remove)

## Testing Requirements

After each consolidation phase:
1. ✅ Run full test suite: `pytest tests/`
2. ✅ Run linting: `flake8 greenlang/`
3. ✅ Verify imports: `python -m greenlang --help` (basic smoke test)
4. ✅ Check backward compatibility: Ensure deprecation warnings appear but code still works
5. ✅ Update documentation: Reflect new import paths

## Migration Timeline

### Phase 1: Quick Wins (Current)
- [x] Document current status (this file)
- [ ] Verify exceptions.py and determinism.py deprecation wrappers
- [ ] Update 6 files importing from greenlang.database

### Phase 2: Core Restructuring
- [ ] Audit `core/greenlang/` modules
- [ ] Move or consolidate nested modules
- [ ] Add deprecation warnings
- [ ] Update imports across codebase

### Phase 3: Config & Monitoring
- [ ] Consolidate config/ and configs/
- [ ] Merge monitoring/ and observability/
- [ ] Update related imports

### Phase 4: Final Cleanup
- [ ] Remove deprecated code (scheduled for v2.0.0)
- [ ] Update CI/CD paths
- [ ] Create migration guide for external users

## Naming Conventions

**Established Standards:**
- ✅ Use singular names for modules: `calculation` not `calculations`
- ✅ Use `Base` class prefix for abstract classes
- ✅ Follow PEP 8 naming: `module_name`, `ClassName`, `function_name`
- ✅ Package structure: `greenlang/module_name/submodule.py`

## Backward Compatibility Strategy

All consolidations must maintain backward compatibility until v2.0.0:

1. **Deprecation Warnings:** All old import paths must issue `DeprecationWarning`
2. **Re-exports:** Old modules should re-export from new locations
3. **Documentation:** Update docs to show both old (deprecated) and new paths
4. **Migration Period:** Minimum 6 months before removal (until v2.0.0)

## Architecture Vision

### Target Structure (Post-Consolidation)

```
greenlang/
├── core/                    # Core orchestration only
│   ├── orchestration/
│   ├── messaging/
│   └── execution/
├── calculation/             # All calculation logic (consolidated)
├── data/                    # Data layer
│   ├── pipeline/
│   ├── engineering/
│   └── database/            # Consolidated db module
├── api/                     # REST & GraphQL APIs
├── agents/                  # Agent framework
├── intelligence/            # AI/ML (llm + ml consolidated)
├── compliance/              # Regulatory frameworks
├── infrastructure/          # Consolidated monitoring/observability
├── config/                  # Configuration (static + code)
├── auth/                    # Authentication (moved from core/greenlang/)
├── cache/                   # Caching (moved from core/greenlang/)
└── utils/                   # Shared utilities
```

## Excluded from Consolidation

Per project rules, these directories are **NEVER** modified:
- `2026_PRD_MVP/**`
- `cbam-pack-mvp/**`
- `.git/**`
- `node_modules/**`
- `__pycache__/**`

## Metrics

- **Total modules analyzed:** 60+
- **Duplicates identified:** 7 major duplicates
- **Deprecation wrappers in place:** 2 confirmed (calculations, calculators)
- **Files requiring import updates:** 6
- **Completion percentage:** ~30%

## Next Actions

1. **Immediate:**
   - Verify and test existing deprecation wrappers
   - Update 6 files with deprecated database imports
   - Run full test suite to ensure no regressions

2. **This Sprint:**
   - Resolve `core/greenlang/` nesting issue
   - Consolidate config/ and configs/
   - Create architecture documentation

3. **Next Sprint:**
   - Merge monitoring/observability
   - Complete database/ cleanup
   - Update CI/CD paths

## References

- **Related Documents:**
  - `VALIDATION_PRD.md` - Product requirements (DO NOT MODIFY)
  - `.ralphy/progress.txt` - Progress tracking (DO NOT MODIFY)
  - Architecture exploration report - Agent a00e335

- **Key Commits:**
  - 75f590ec - Separation of concerns refactor
  - 8d691ee2 - Import error fixes

## Sign-off

**Status:** In Progress
**Last Updated:** 2026-01-25
**Next Review:** After Phase 1 completion

---

*This document is a living record of the consolidation effort. Update after each phase completion.*
