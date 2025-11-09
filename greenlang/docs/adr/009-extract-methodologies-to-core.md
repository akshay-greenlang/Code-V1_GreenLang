# ADR 009: Extract Methodologies to GreenLang Core

**Date:** 2025-11-09
**Status:** Accepted
**Context:** GL-VCCI Enhancement to 55% Custom Code

## Context

The Methodologies service (7,007 lines) provides Pedigree Matrix, Monte Carlo uncertainty,
and DQI calculations. These are universally applicable to ALL climate data quality needs.

## Decision

Extract `services/methodologies/` from GL-VCCI-Carbon-APP to `greenlang.services.methodologies`
as a core GreenLang service.

## Rationale

1. **Universal Applicability**: All climate apps need data quality assessment
   - GL-CSRD: ESRS data quality requirements
   - GL-LCA: ISO 14040/14044 uncertainty
   - GL-TCFD: Scenario uncertainty quantification

2. **Standard Methods**: Pedigree Matrix is ecoinvent/ILCD standard
   - Not specific to Scope 3
   - Required by multiple reporting frameworks
   - Industry-standard approach

3. **Code Reduction**: 7,007 lines moved from app to framework
   - Reduces GL-VCCI custom code by 9.6%
   - Available for all apps without duplication

4. **Strategic Value**: Methods are infrastructure, not domain logic
   - Pedigree Matrix (ecoinvent/ILCD standard)
   - Monte Carlo simulation (10K iterations)
   - DQI calculation (ISO 14040 compliant)

## Implementation

### Components Extracted
- `pedigree_matrix.py`: Pedigree scoring (ecoinvent/ILCD)
- `monte_carlo.py`: Uncertainty propagation (10K iterations)
- `dqi_calculator.py`: Data Quality Indicator calculation
- `uncertainty.py`: Unified uncertainty engine
- `models.py`, `config.py`, `constants.py`

### Source Structure
```
GL-VCCI-Carbon-APP/services/methodologies/
├── __init__.py
├── pedigree_matrix.py
├── monte_carlo.py
├── dqi_calculator.py
├── uncertainty.py
├── models.py
├── config.py
└── constants.py
```

### Destination Structure
```
greenlang/services/methodologies/
├── __init__.py (exports PedigreeMatrix, MonteCarloSimulator, DQICalculator)
├── pedigree_matrix.py
├── monte_carlo.py
├── dqi_calculator.py
├── uncertainty.py
├── models.py
├── config.py
└── constants.py
```

### GL-VCCI Integration
```python
# Before
from services.methodologies import PedigreeMatrix, MonteCarloSimulator

# After
from greenlang.services.methodologies import PedigreeMatrix, MonteCarloSimulator
```

## Consequences

### Positive
- ✅ 7,007 lines moved from GL-VCCI to core (9.6% reduction)
- ✅ Standard methods available for all apps
- ✅ Single source of truth for DQI/uncertainty
- ✅ ISO 14040/14044 compliant uncertainty quantification
- ✅ ecoinvent/ILCD Pedigree Matrix implementation

### Negative
- ⚠️ Import path changes (breaking change)
- ⚠️ GL-VCCI depends on greenlang.services.methodologies

### Mitigation
- Update gl.yaml dependencies
- Update imports in GL-VCCI
- Provide migration guide in CHANGELOG
- Maintain backward compatibility wrappers for one release

## Alternatives Considered

1. **Keep in GL-VCCI**: Rejected - duplicates code across apps
2. **Split into separate modules**: Rejected - cohesive methodology package
3. **Use external library**: Rejected - no existing library meets requirements

## References

- ecoinvent Pedigree Matrix Documentation
- ISO 14040/14044 (LCA Standards)
- ILCD Data Quality Guidelines
- GreenLang Framework Architecture
- GL-VCCI Phase 5 Enhancement Plan

## Migration Path

### Phase 1: Extraction (Week 1)
- Copy services/methodologies/ to greenlang/services/methodologies/
- Update greenlang __init__.py exports
- Create tests in greenlang/tests/services/methodologies/

### Phase 2: GL-VCCI Update (Week 2)
- Update imports in GL-VCCI Calculator Agent
- Update gl.yaml configuration
- Update pack.yaml dependencies
- Run integration tests

### Phase 3: Documentation (Week 2)
- Update API documentation
- Create migration guide
- Update CHANGELOG

## Rollback Plan

If issues arise during migration:
1. Revert GL-VCCI imports to local services/methodologies/
2. Keep greenlang.services.methodologies for future use
3. No data loss risk (read-only extraction)
