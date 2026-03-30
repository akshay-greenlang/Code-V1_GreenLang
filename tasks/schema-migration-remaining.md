# Shared Schema Migration - Remaining Application Models

These tasks migrate application-level model files to use `greenlang.schemas.GreenLangBase`
instead of raw `pydantic.BaseModel`. The core agents (greenlang/agents/) are already done.

## Task 1: Migrate GL-Agent-Factory backend models
Migrate all 5 Agent Factory model files from `from pydantic import BaseModel` to
`from greenlang.schemas import GreenLangBase`. Replace `class Foo(BaseModel):` with
`class Foo(GreenLangBase):`. Files:
- applications/GL-Agent-Factory/backend/agents/gl_022_superheater_control/models.py
- applications/GL-Agent-Factory/backend/agents/gl_023_heat_load_balancer/models.py
- applications/GL-Agent-Factory/backend/agents/gl_031_furnace_guardian/models.py
- applications/GL-Agent-Factory/backend/agents/gl_032_refractory_monitor/models.py
- applications/GL-Agent-Factory/backend/agents/gl_033_burner_balancer/models.py

## Task 2: Migrate GL-VCCI-Carbon-APP service models
Migrate all 5 VCCI service model files from `from pydantic import BaseModel` to
`from greenlang.schemas import GreenLangBase`. Replace `class Foo(BaseModel):` with
`class Foo(GreenLangBase):`. Files:
- applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/calculator/models.py
- applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/engagement/models.py
- applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/hotspot/models.py
- applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/intake/models.py
- applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/reporting/models.py

## Task 3: Migrate GreenLang Development copies
Migrate all model files under GreenLang Development/ that mirror the application models.
These are development/staging copies. Files:
- GreenLang Development/02-Applications/GL-Agent-Factory/backend/agents/*/models.py (5 files)
- GreenLang Development/02-Applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/services/agents/*/models.py (5 files)
- GreenLang Development/01-Core-Platform/agents/formulas/models.py
- GreenLang Development/01-Core-Platform/agents/intelligence/rag/models.py

## Task 4: Add linting rule to prevent BaseModel regression
Add a pre-commit hook or CI check that flags any new `from pydantic import BaseModel` in
greenlang/agents/**/*.py files. This prevents regression of the migration work.
