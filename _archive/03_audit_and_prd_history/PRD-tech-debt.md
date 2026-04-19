# GreenLang Technical Debt Cleanup - PRD

## Context

GreenLang has shared schema base classes in `greenlang/schemas/` (base.py, enums.py, fields.py).
Only ~3 files have been migrated so far. ~2,770 files still use plain `pydantic.BaseModel`.

### Shared schemas provide:
- `GreenLangBase` - root base with `extra="forbid"`, `validate_default=True`, ORM compat
- `GreenLangRecord` - Base + timestamps + tenant + provenance (for stored entities)
- `GreenLangRequest` - Base only (for API inputs)
- `GreenLangResponse` - Base + timestamps + provenance (for API outputs)
- `GreenLangResult` - Base + timestamps + provenance + metadata (for calc results)
- `GreenLangConfig` - Base with `extra="ignore"` (for config models)
- `GreenLangAuditRecord` - Base + audit actor + tenant + provenance
- `utcnow()`, `new_uuid()`, `prefixed_uuid()`, `compute_provenance_hash()`

### Rules for migration:
1. NEVER modify `greenlang/schemas/` files - those are the source of truth
2. Replace `def _utcnow()` definitions with `from greenlang.schemas import utcnow`
3. Replace duplicate enum definitions with `from greenlang.schemas.enums import ...`
4. For BaseModel migration: only change the base class, do NOT add/remove fields
5. Run `python -c "from <module> import *"` after each change to verify imports work
6. Do NOT change test files - only source files in greenlang/agents/ and packs/

## Tasks

- [ ] Migrate greenlang/agents/foundation/ models from BaseModel to GreenLangBase (10 agents)
- [ ] Migrate greenlang/agents/data/ models from BaseModel to GreenLangBase (20 agents)
- [ ] Migrate greenlang/agents/mrv/ models from BaseModel to GreenLangBase (31 agents)
- [ ] Migrate greenlang/agents/eudr/ models from BaseModel to GreenLangBase (40 agents)
- [ ] Migrate packs/eu-compliance/ engines and workflows from BaseModel to GreenLangBase
- [ ] Migrate packs/net-zero/ engines and workflows from BaseModel to GreenLangBase
- [ ] Migrate packs/energy-efficiency/ engines and workflows from BaseModel to GreenLangBase
- [ ] Migrate packs/ghg-accounting/ engines and workflows from BaseModel to GreenLangBase
