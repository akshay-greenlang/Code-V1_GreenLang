# ADR 008: Extract Factor Broker to GreenLang Core

**Date:** 2025-11-09
**Status:** Accepted
**Context:** GL-VCCI Enhancement to 55% Custom Code

## Context

The Factor Broker service (5,530 lines) in GL-VCCI-Carbon-APP provides runtime emission
factor resolution with license compliance. This service is universally applicable to ALL
carbon accounting applications, not just Scope 3 VCCI.

## Decision

Extract `services/factor_broker/` from GL-VCCI-Carbon-APP to `greenlang.services.factor_broker`
as a core GreenLang service.

## Rationale

1. **Reusability**: Every carbon app needs emission factors
   - GL-CSRD-APP needs factors for E1 climate calculations
   - GL-LCA-APP needs factors for product lifecycle
   - GL-TCFD-APP needs factors for scenario modeling

2. **License Compliance**: Factor Broker handles ecoinvent licensing correctly
   - Runtime API access (no bulk redistribution)
   - Caching within license terms
   - Audit trail for factor usage

3. **Code Reduction**: 5,530 lines moved from app to framework
   - Reduces GL-VCCI custom code by 7.6%
   - Available for all apps without duplication

4. **Strategic Value**: Factor Broker is infrastructure, not domain logic
   - Version control (GWP AR5/AR6)
   - Regional adaptation
   - Multi-source aggregation (DESNZ, EPA, ecoinvent)

## Implementation

### Source Structure
```
GL-VCCI-Carbon-APP/services/factor_broker/
├── __init__.py
├── broker.py (main service)
├── cache.py
├── config.py
├── models.py
├── exceptions.py
└── sources/
    ├── ecoinvent.py
    ├── desnz.py
    ├── epa.py
    └── proxy.py
```

### Destination Structure
```
greenlang/services/factor_broker/
├── __init__.py (exports FactorBroker, models, config)
├── broker.py
├── cache.py
├── config.py
├── models.py
├── exceptions.py
└── sources/
    ├── ecoinvent.py
    ├── desnz.py
    ├── epa.py
    └── proxy.py
```

### GL-VCCI Integration
```python
# Before
from services.factor_broker import FactorBroker

# After
from greenlang.services.factor_broker import FactorBroker
```

## Consequences

### Positive
- ✅ 5,530 lines moved from GL-VCCI to core (7.6% reduction)
- ✅ Available for GL-CSRD, GL-LCA, GL-TCFD without duplication
- ✅ Centralized license compliance management
- ✅ Single source of truth for emission factors

### Negative
- ⚠️ GL-VCCI now depends on greenlang.services
- ⚠️ Breaking change for existing deployments (import path changes)

### Mitigation
- Update gl.yaml dependencies
- Update imports in GL-VCCI
- Provide migration guide in CHANGELOG

## Alternatives Considered

1. **Keep in GL-VCCI**: Rejected - duplicates code across apps
2. **Create separate package**: Rejected - overhead, belongs in framework
3. **Hard-code in each app**: Rejected - license compliance nightmare

## References

- GreenLang Framework Architecture
- ecoinvent License Terms
- GL-VCCI Phase 5 Enhancement Plan
