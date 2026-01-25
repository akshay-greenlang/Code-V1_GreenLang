# GreenLang Module Organization Guide

**Version:** 2.0
**Last Updated:** 2026-01-25
**Status:** Post-Consolidation

## Overview

This document describes the logical organization of GreenLang modules after the 2026-01-25 consolidation effort. It provides guidelines for where new code should be placed and how to navigate the codebase.

---

## Quick Reference

### Where Should I Put My Code?

| Type of Code | Location | Example |
|-------------|----------|---------|
| Emissions calculations | `greenlang/calculation/` | Scope 1/2/3 calculators |
| Database models | `greenlang/db/models_*.py` | User, Role, AuditLog |
| API endpoints | `greenlang/api/routes/` | REST endpoints |
| Authentication | `greenlang/auth/` | JWT, API keys, SAML |
| Validation rules | `greenlang/validation/` | Data validators |
| Policy enforcement | `greenlang/policy/` | RBAC, OPA |
| CLI commands | `greenlang/cli/` | Typer commands |
| Agent implementations | `greenlang/agents/` | Custom agents |
| Data pipelines | `greenlang/data/pipeline/` | ETL logic |
| Testing utilities | `greenlang/testing/` | Test helpers |
| Configuration | `greenlang/config/` | Settings, schemas |
| Metrics/logging | `greenlang/telemetry/` | Prometheus, logging |

---

## Module Structure

### Primary Modules

#### 1. `greenlang/calculation/`
**Purpose:** Zero-hallucination emissions calculations

**Structure:**
```
calculation/
├── __init__.py
├── emissions/           # Scope 1/2/3 calculators
│   ├── scope1.py
│   ├── scope2.py
│   └── scope3.py
├── physics/             # Thermodynamic calculations
└── industry/            # Industry-specific logic
```

**When to use:** Any emissions calculation logic, including custom calculators.

**Example:**
```python
from greenlang.calculation.emissions import Scope1Calculator

calculator = Scope1Calculator()
result = calculator.calculate(activity_data, emission_factors)
```

---

#### 2. `greenlang/db/` (Primary Database Module)
**Purpose:** Database access, models, connections

**Structure:**
```
db/
├── __init__.py
├── base.py              # SQLAlchemy base, sessions
├── models_auth.py       # Auth models (User, Role, etc.)
├── models_analytics.py  # Analytics models
├── connection.py        # Connection pooling
└── query_optimizer.py   # Query optimization
```

**When to use:** Database models, queries, connections.

**Example:**
```python
from greenlang.db import get_session, Base
from greenlang.db.models_auth import User

session = get_session()
user = session.query(User).filter_by(email="user@example.com").first()
```

**Note:** `greenlang/database/` is DEPRECATED. Use `greenlang/db/` instead.

---

#### 3. `greenlang/api/`
**Purpose:** REST and GraphQL APIs

**Structure:**
```
api/
├── __init__.py
├── routes/              # REST endpoints
│   ├── calculations.py
│   ├── agents.py
│   └── auth.py
├── graphql/             # GraphQL schema
└── security/            # API security
```

**When to use:** Adding new API endpoints or GraphQL resolvers.

**Example:**
```python
from fastapi import APIRouter
from greenlang.api.routes import router

@router.post("/calculate")
async def calculate_endpoint(data: dict):
    return {"result": ...}
```

---

#### 4. `greenlang/agents/`
**Purpose:** Agent framework and implementations

**Structure:**
```
agents/
├── __init__.py
├── core/                # Base agent classes
├── templates/           # Agent templates
└── tools/               # Agent tools
```

**When to use:** Building new agents or extending the agent framework.

**Example:**
```python
from greenlang.agents.core import BaseAgent

class MyCustomAgent(BaseAgent):
    def run(self, inputs):
        return self.calculate(inputs)
```

---

#### 5. `greenlang/auth/`
**Purpose:** Authentication and authorization

**Structure:**
```
auth/
├── __init__.py
├── backends/            # Auth backends
│   ├── saml.py
│   ├── oauth.py
│   └── ldap.py
├── api_key_manager.py   # API key management
├── jwt_handler.py       # JWT tokens
└── middleware.py        # Auth middleware
```

**When to use:** Adding new auth methods or managing credentials.

**Example:**
```python
from greenlang.auth.jwt_handler import JWTHandler

handler = JWTHandler()
token = handler.create_token(user_id="123", role="admin")
```

---

#### 6. `greenlang/policy/`
**Purpose:** Policy enforcement and RBAC

**Structure:**
```
policy/
├── __init__.py
├── enforcer.py          # Policy enforcer
├── agent_rbac.py        # Role-based access control
└── opa.py               # Open Policy Agent
```

**When to use:** Implementing access control or policy rules.

**Example:**
```python
from greenlang.policy.agent_rbac import AgentPermission, PREDEFINED_ROLES

permission = AgentPermission.EXECUTE_AGENT
roles = PREDEFINED_ROLES
```

---

#### 7. `greenlang/validation/`
**Purpose:** Data validation and quality checks

**Structure:**
```
validation/
├── __init__.py
├── hooks.py             # Validation hooks
├── emission_factors.py  # EF validation
└── schema.py            # Schema validation
```

**When to use:** Adding validation rules for climate data.

**Example:**
```python
from greenlang.validation.hooks import EmissionFactorValidator

validator = EmissionFactorValidator()
is_valid = validator.validate(emission_factor=2.5, unit="kgCO2/kWh")
```

---

#### 8. `greenlang/config/`
**Purpose:** Configuration management

**Structure:**
```
config/
├── __init__.py
├── defaults/            # Static config files (JSON/YAML)
│   ├── boiler_efficiencies.json
│   └── fuel_properties.json
├── settings.py          # Application settings
└── schemas.py           # Config schemas
```

**When to use:** Managing application configuration or defaults.

**Example:**
```python
from greenlang.config import get_settings

settings = get_settings()
database_url = settings.DATABASE_URL
```

---

#### 9. `greenlang/telemetry/` (Implementation)
**Purpose:** Metrics, logging, tracing

**Structure:**
```
telemetry/
├── __init__.py
├── metrics.py           # Prometheus metrics
├── logging.py           # Structured logging
├── tracing.py           # OpenTelemetry tracing
└── health.py            # Health checks
```

**When to use:** Implementing observability features.

**Example:**
```python
from greenlang.telemetry.metrics import track_execution

@track_execution("my_function")
def my_function():
    pass
```

---

#### 10. `greenlang/observability/` (Facade)
**Purpose:** Convenience facade for telemetry

**Structure:**
```
observability/
└── __init__.py          # Re-exports from telemetry
```

**When to use:** Importing observability features (preferred over direct telemetry imports).

**Example:**
```python
# ✅ PREFERRED
from greenlang.observability import get_logger, track_execution

# ⚠️ ALSO WORKS (but use observability facade instead)
from greenlang.telemetry.logging import get_logger
```

---

## Deprecated Modules

### DO NOT USE These Modules

| Module | Status | Use Instead |
|--------|--------|-------------|
| `greenlang/database/` | **DEPRECATED** | `greenlang/db/` |
| `greenlang/calculations/` | **DEPRECATED** | `greenlang/calculation/` |
| `greenlang/calculators/` | **DEPRECATED** | `greenlang/calculation/` |
| `greenlang/core/greenlang/` | **REMOVED** | Top-level modules |
| `greenlang/configs/` (static files) | **MOVED** | `greenlang/config/defaults/` |

---

## Import Guidelines

### Standard Imports

```python
# ✅ CORRECT: Import from primary modules
from greenlang.db import get_session
from greenlang.calculation import calculate_emissions
from greenlang.auth.jwt_handler import JWTHandler
from greenlang.policy.agent_rbac import AgentPermission

# ✅ CORRECT: Use observability facade
from greenlang.observability import get_logger, track_execution

# ❌ WRONG: Don't import from deprecated modules
from greenlang.database import DatabaseConnection  # DEPRECATED
from greenlang.calculations import calculate_emissions  # DEPRECATED
from greenlang.core.greenlang.policy import ...  # REMOVED
```

### Relative Imports

Within a module, use relative imports:
```python
# In greenlang/calculation/emissions/scope1.py
from ..physics import thermodynamic_calculator  # ✅ Good
from greenlang.calculation.physics import thermodynamic_calculator  # ⚠️ Also OK
```

---

## Naming Conventions

### Module Names
- Use **singular** names: `calculation` not `calculations`
- Use **lowercase** with underscores: `emission_factors` not `EmissionFactors`
- Be **descriptive**: `api_key_manager` not `keys`

### Class Names
- Use **PascalCase**: `EmissionFactorValidator`
- Include context in name: `Scope1Calculator` not just `Calculator`

### Function Names
- Use **snake_case**: `get_session()`, `calculate_emissions()`
- Use verbs: `calculate_`, `validate_`, `fetch_`

---

## Directory Layout Rules

### File Organization

1. **`__init__.py`** - Package initialization, exports
2. **Implementation files** - Core logic (e.g., `scope1.py`)
3. **Subdirectories** - Group related functionality
4. **Tests** - ONLY in root `/tests` directory, NOT in source

### Example: Well-Organized Module

```
greenlang/calculation/
├── __init__.py              # Exports: calculate_emissions, Scope1Calculator, etc.
├── emissions/               # Subdirectory for emission calculations
│   ├── __init__.py
│   ├── scope1.py            # Scope 1 calculator
│   ├── scope2.py            # Scope 2 calculator
│   └── scope3.py            # Scope 3 calculator
├── physics/                 # Subdirectory for physics calculations
│   ├── __init__.py
│   └── thermodynamics.py
└── industry/                # Industry-specific calculators
    ├── __init__.py
    ├── steel.py
    └── cement.py
```

---

## Adding New Features

### Checklist for New Code

1. **Determine module placement**
   - Use the "Where Should I Put My Code?" table above
   - Follow existing patterns in that module

2. **Create implementation file**
   - Use descriptive filename: `boiler_calculator.py` not `calc.py`
   - Add docstring at top of file

3. **Update `__init__.py`**
   - Export public classes/functions
   - Add to `__all__` list

4. **Add tests**
   - Tests go in `/tests`, NOT in source directory
   - Mirror source structure: `tests/unit/calculation/test_boiler.py`

5. **Update documentation**
   - Add docstrings to classes/functions
   - Update relevant docs in `/docs`

### Example: Adding a New Calculator

```python
# File: greenlang/calculation/industry/boiler.py
"""
Boiler Efficiency Calculator

Calculates emissions from industrial boilers accounting for efficiency losses.
"""

from greenlang.calculation.emissions import Scope1Calculator

class BoilerCalculator(Scope1Calculator):
    """Calculate boiler emissions with efficiency adjustments."""

    def __init__(self, efficiency: float = 0.85):
        super().__init__()
        self.efficiency = efficiency

    def calculate(self, fuel_consumption, emission_factor):
        """Calculate emissions adjusted for boiler efficiency."""
        return (fuel_consumption / self.efficiency) * emission_factor
```

Then update `greenlang/calculation/industry/__init__.py`:
```python
from .boiler import BoilerCalculator

__all__ = ["BoilerCalculator"]
```

---

## Module Dependencies

### Dependency Rules

1. **No circular dependencies** - If A imports B, B cannot import A
2. **Layer dependencies flow downward** - API → Agents → Calculation → DB
3. **Use dependency injection** - Don't hardcode dependencies

### Allowed Dependencies

```
api/
├── Can import: agents/, calculation/, db/, auth/, policy/
└── Cannot import: Nothing (top layer)

agents/
├── Can import: calculation/, db/, validation/, policy/
└── Cannot import: api/

calculation/
├── Can import: db/, validation/
└── Cannot import: api/, agents/

db/
├── Can import: utils/
└── Cannot import: api/, agents/, calculation/
```

---

## Testing Module Organization

Tests are organized to mirror source structure:

```
tests/
├── unit/                    # Unit tests (isolated)
│   ├── calculation/
│   │   ├── test_scope1.py
│   │   └── test_boiler.py
│   ├── db/
│   │   └── test_models.py
│   └── api/
│       └── test_routes.py
├── integration/             # Integration tests
│   ├── test_agent_pipeline.py
│   └── test_e2e_pipelines.py
└── golden/                  # Expert-validated tests
    ├── test_emission_factors.py
    └── test_validation_hooks.py
```

**Rule:** Tests go in `/tests`, NEVER in source directories.

---

## Migration Guide for Developers

### Updating Imports After Consolidation

If you encounter import errors after the consolidation:

1. **Check deprecation warnings** - They tell you the new path
2. **Update imports** - Use table below
3. **Run tests** - Ensure nothing breaks
4. **Commit** - Document the import update

### Common Import Updates

| Old Import | New Import |
|------------|------------|
| `from greenlang.database import DatabaseConnection` | `from greenlang.db import get_session` |
| `from greenlang.calculations import calculate_emissions` | `from greenlang.calculation import calculate_emissions` |
| `from greenlang.core.greenlang.policy.agent_rbac import ...` | `from greenlang.policy.agent_rbac import ...` |
| `from greenlang.core.greenlang.validation.hooks import ...` | `from greenlang.validation.hooks import ...` |

---

## Frequently Asked Questions

### Q: Where do I find emission factor data?
**A:** `greenlang/data/emission_factor_db.py` or `greenlang/validation/emission_factors.py`

### Q: How do I add a new API endpoint?
**A:** Create a new route in `greenlang/api/routes/` and register it in `__init__.py`

### Q: Where are database models defined?
**A:** In `greenlang/db/models_*.py` (e.g., `models_auth.py`, `models_analytics.py`)

### Q: Can I import from `greenlang.database`?
**A:** Yes, but it's deprecated. Use `greenlang.db` instead to avoid warnings.

### Q: Where should I put utility functions?
**A:** If shared across modules: `greenlang/utils/`. If specific to one module: in that module.

### Q: How do I add logging?
**A:** `from greenlang.observability import get_logger`

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - High-level system architecture
- [CONSOLIDATION_STATUS.md](../CONSOLIDATION_STATUS.md) - Consolidation progress tracking
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines

---

**Last Updated:** 2026-01-25
**Maintained By:** GreenLang Core Team
