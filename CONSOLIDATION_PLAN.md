# GreenLang Directory Consolidation Plan
**Target: Reduce greenlang/ subdirectories from 77 to ≤15**

## Current State
- Current subdirectories: 77 (excluding __pycache__)
- Many directories have <3 Python files
- Significant code duplication and overlap
- Empty or near-empty directories

## Target Structure (15 Directories)

### 1. **agents/** - Keep as-is
Core agent framework and domain experts

### 2. **data/** - Keep as-is
Emission factors, unit conversion, data models

### 3. **config/** - Keep as-is
Application configuration and specs

### 4. **auth/** - Keep as-is
Authentication & authorization

### 5. **cache/** - Keep as-is
Caching layer

### 6. **cli/** - Keep as-is
Command-line interface

### 7. **persistence/** ← MERGE
- database/
- db/
- datasets/
- models/

### 8. **integration/** ← MERGE
- api/
- sdk/
- integrations/
- connectors/
- services/

### 9. **observability/** ← MERGE
- monitoring/
- telemetry/
- observability/
- sandbox/

### 10. **governance/** ← MERGE
- safety/
- policy/
- compliance/
- validation/
- security/

### 11. **utilities/** ← MERGE
- utils/
- io/
- serialization/
- determinism/
- lineage/
- provenance/

### 12. **ecosystem/** ← MERGE
- marketplace/
- packs/
- hub/
- partners/
- adapters/
- whitelabel/

### 13. **execution/** ← MERGE
- core/
- pipeline/
- runtime/
- infrastructure/
- intelligence/
- resilience/

### 14. **extensions/** ← MERGE
- ml/
- ml_platform/
- llm/
- simulation/
- business/
- benchmarks/
- middleware/
- generator/
- compat/

### 15. **tests/** - Keep as-is
Testing infrastructure

## Directories to DELETE (empty/minimal)
- calculators/
- calculation/
- satellite/
- regulations/
- emission_factors/
- data_engineering/
- supply_chain/
- schemas/
- docs/
- frontend/
- src/
- i18n/
- exceptions/ (merge into core or utilities)
- tools/ (merge into utilities)
- cards/ (merge into visualization → utilities)
- visualization/ (merge into utilities)
- templates/ (merge into ecosystem or examples)
- examples/ (merge into ecosystem)
- greenlang_registry/ (merge into registry → config)
- registry/ (merge into config)
- factory/ (merge into agents or core)
- testing/ (merge into tests/)

## Migration Strategy

### Phase 1: Database Consolidation
```
greenlang/persistence/
├── __init__.py (backward compat re-exports)
├── connection.py (from database/)
├── models.py (from db/ + database/ + models/)
├── schema.py (from db/)
├── transactions.py (from database/)
├── datasets.py (from datasets/)
└── emission_factors/ (submodule)
```

### Phase 2: Integration Layer
```
greenlang/integration/
├── __init__.py
├── api/ (from api/)
├── sdk/ (from sdk/)
├── connectors/ (from connectors/ + integrations/)
└── services/ (consolidated)
```

### Phase 3: Observability
```
greenlang/observability/
├── __init__.py
├── monitoring/ (from monitoring/)
├── telemetry/ (from telemetry/)
├── health/ (consolidated)
└── sandbox/ (from sandbox/)
```

### Phase 4: Governance
```
greenlang/governance/
├── __init__.py
├── auth/ (reference - kept separate)
├── safety/ (from safety/)
├── policy/ (from policy/)
├── compliance/ (from compliance/)
├── validation/ (from validation/)
└── security/ (from security/)
```

### Phase 5: Utilities
```
greenlang/utilities/
├── __init__.py
├── io/ (from io/ + serialization/)
├── determinism/ (from determinism/)
├── lineage/ (from lineage/)
├── provenance/ (from provenance/)
├── visualization/ (from visualization/ + cards/)
└── common/ (from utils/)
```

### Phase 6: Ecosystem
```
greenlang/ecosystem/
├── __init__.py
├── marketplace/ (from marketplace/)
├── packs/ (from packs/)
├── hub/ (from hub/)
├── partners/ (from partners/)
└── adapters/ (from adapters/)
```

### Phase 7: Execution
```
greenlang/execution/
├── __init__.py
├── runtime/ (from runtime/)
├── pipeline/ (from pipeline/)
├── infrastructure/ (from infrastructure/)
├── intelligence/ (from intelligence/)
└── resilience/ (from resilience/)
```

### Phase 8: Extensions
```
greenlang/extensions/
├── __init__.py
├── ml/ (from ml/ + ml_platform/)
├── llm/ (from llm/)
├── simulation/ (from simulation/)
├── business/ (from business/)
├── benchmarks/ (from benchmarks/)
└── middleware/ (from middleware/)
```

## Backward Compatibility

Each consolidated module will have:
1. Top-level `__init__.py` with re-exports
2. Deprecation warnings for old import paths
3. Clear migration guide in docstrings

Example:
```python
# greenlang/persistence/__init__.py
"""
Persistence layer - consolidated from db, database, datasets, models.

Deprecated imports (will be removed in v2.0):
- from greenlang.database import Connection → from greenlang.persistence import Connection
- from greenlang.db import Schema → from greenlang.persistence import Schema
"""
import warnings

def _deprecated_import(old_path, new_path):
    warnings.warn(
        f"{old_path} is deprecated. Use {new_path} instead.",
        DeprecationWarning,
        stacklevel=2
    )

# Re-exports for backward compatibility
from .connection import Connection
from .models import EmissionFactorModel
from .schema import Schema
from .transactions import Transaction
```

## Testing Strategy

1. Create comprehensive import tests
2. Run existing test suite after each phase
3. Verify no breaking changes to public API
4. Update internal imports progressively
5. Document any unavoidable breaking changes

## Rollback Plan

Each phase is committed separately with:
- Clear commit message
- Atomic changes
- Ability to revert individual phases

## Success Criteria

✓ greenlang/ subdirectories ≤ 15
✓ All tests passing
✓ No breaking changes to public API
✓ Clear deprecation warnings
✓ Updated documentation
✓ Improved code organization
