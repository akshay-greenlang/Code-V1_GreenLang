# Policy Resolution Pattern

**Audience:** Pack developers building `applications/GL-*-APP/` policy
workflows (CBAM, CSRD, VCCI, EUDR, SBTi, ISO 14064, …).

**Mandate:** CTO non-negotiable #6 — *"Policy workflows must call method
profiles, not raw factors."*

---

## TL;DR

Pack code MUST NOT call the catalog repository directly to fetch a factor.
It MUST go through `ResolutionEngine.resolve(...)` with an explicit
`method_profile`. The 7-step cascade and the method-pack `SelectionRule`
do the gatekeeping; bypassing them silently makes the policy workflow
non-compliant by construction.

```python
# DO NOT DO THIS (raw catalog access — bypasses methodology constraints)
factor = catalog_repository.get(factor_id="glf_steel_eu")
factor = ef_client.get_fuel_factor("natural_gas")
factor = ef.get_emission_factor_by_cn_code("7208")
```

```python
# DO THIS (method-profile-bound resolution)
from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.resolution.engine import ResolutionEngine
from greenlang.factors.resolution.request import ResolutionRequest

engine: ResolutionEngine = ...  # injected at app boot time
result = engine.resolve(ResolutionRequest(
    activity="hot-rolled steel coil import",
    method_profile=MethodProfile.EU_CBAM,
    jurisdiction="EU",
    extras={"cn_code": "7208"},
))
factor_id  = result.chosen_factor_id
co2e_per_t = result.gas_breakdown.co2e_total_kg
```

---

## Why this matters

A *method profile* is a contract about HOW a factor is allowed to be
used.  It encodes:

| Constraint enforced by the profile  | Example (`EU_CBAM`)                                                    |
|-------------------------------------|------------------------------------------------------------------------|
| Acceptable scope                    | only direct + indirect embedded emissions of in-scope CBAM goods       |
| Acceptable boundary                 | cradle-to-gate of the installation                                     |
| Acceptable GWP basis                | IPCC AR6 100-year                                                      |
| Acceptable verification status      | verified or default-CBAM only                                          |
| Acceptable redistribution class     | EU Commission default values when supplier-specific is unavailable     |
| Acceptable license class            | distributable to importers + EU Commission                             |

The catalog contains factors that satisfy MANY profiles.  If a CBAM
workflow grabbed an arbitrary `glf_steel_eu` row by id, it might pick a
factor whose verification status or redistribution class is not legal
for CBAM filings.  The `ResolutionEngine` ensures that only factors
the CBAM `SelectionRule` accepts can win.

The same mechanism guarantees:

- explainability — every selection ships an `alternates` list (why-not),
- determinism — tie-breaks are explicit, not driven by SQL ordering,
- auditability — fallback step (1..7) is always recorded.

These are CTO non-negotiables #1, #3, and #6.

---

## Canonical call pattern

### 1. Choose the right method profile

| Workflow                              | Profile                                  |
|---------------------------------------|------------------------------------------|
| CBAM emissions report                 | `MethodProfile.EU_CBAM`                  |
| CSRD/ESRS Scope 1                     | `MethodProfile.CORPORATE_SCOPE1`         |
| CSRD/ESRS Scope 2 location-based      | `MethodProfile.CORPORATE_SCOPE2_LOCATION`|
| CSRD/ESRS Scope 2 market-based        | `MethodProfile.CORPORATE_SCOPE2_MARKET`  |
| CSRD/ESRS Scope 3 (any category)      | `MethodProfile.CORPORATE_SCOPE3`         |
| Product carbon footprint (ISO 14067)  | `MethodProfile.PRODUCT_CARBON`           |
| Freight carbon (GLEC / ISO 14083)     | `MethodProfile.FREIGHT_ISO_14083`        |
| Land-use removals                     | `MethodProfile.LAND_REMOVALS`            |
| PCAF financed emissions               | `MethodProfile.FINANCE_PROXY`            |
| EU Digital Product Passport           | `MethodProfile.EU_DPP`                   |
| EU Battery Regulation passport        | `MethodProfile.EU_DPP_BATTERY`           |
| India CCTS                            | `MethodProfile.INDIA_CCTS`               |
| VCCI / voluntary carbon credit issuance | `MethodProfile.LAND_REMOVALS` (as default) |

If the workflow legitimately spans multiple profiles, the pack should
make ONE `resolve()` call per profile and combine the structured
results — never fall back to a raw catalog hit.

### 2. Build a `ResolutionRequest`

```python
from greenlang.factors.resolution.request import ResolutionRequest

req = ResolutionRequest(
    activity="diesel combustion stationary",   # required, free text or canonical id
    method_profile=MethodProfile.CORPORATE_SCOPE1,
    jurisdiction="IN",                          # ISO country / region code
    reporting_date="2026-04-22",                # ISO-8601, defaults to today
    activity_id="erp.scope1.diesel.boiler_42",  # optional, used for tenant overrides
    tenant_id="acme-corp",                      # optional, used for step 1
    supplier_id="shell-india",                  # optional, used for step 2
    facility_id="mumbai-plant",                 # optional, used for step 3
    utility_or_grid_region="CEA-W",             # optional, used for step 4
    target_unit="kgCO2e/L",                     # optional, engine converts
    extras={"fuel_type": "diesel", "cn_code": "2710.19"},  # passthrough hints
)
```

Every field except `activity` and `method_profile` is optional.  The
engine honours whatever context you supply — more context produces a
better-ranked match.

### 3. Call `engine.resolve(req)`

```python
from greenlang.factors.resolution.engine import ResolutionError

try:
    resolved = engine.resolve(req)
except ResolutionError:
    # No factor at any of the 7 cascade steps satisfied the
    # method-pack selection rule.  Surface this to the user — DO NOT
    # silently fall back to a raw catalog lookup.
    raise
```

The returned `ResolvedFactor` is the only object pack code should
ever touch — it carries the chosen factor + the explainability payload
required by `/explain`.

### 4. Use the structured result

```python
factor_id      = resolved.chosen_factor_id
source         = resolved.source_id
vintage        = resolved.vintage
quality        = resolved.quality_score
co2e_per_unit  = resolved.gas_breakdown.co2e_total_kg
denominator    = resolved.factor_unit_denominator
fallback_step  = resolved.fallback_rank        # 1..7
why_chosen     = resolved.why_chosen
alternates     = resolved.alternates           # top-9 with "why not chosen"
assumptions    = resolved.assumptions
```

Pack reports must persist `factor_id`, `fallback_rank`, `why_chosen`,
`source_id`, `vintage`, and the full `alternates` list for audit
reproducibility.

---

## Migration recipe

If you are touching legacy code that holds a raw catalog handle:

### Before

```python
from greenlang.factors.catalog_repository import FileCatalogRepository

class CBAMCalculator:
    def __init__(self, edition_id: str):
        self.catalog = FileCatalogRepository(edition_id=edition_id)

    def emissions_for(self, cn_code: str, mass_t: float) -> float:
        factor = self.catalog.get(factor_id=f"glf_cbam_{cn_code}")
        return mass_t * factor.gwp_100yr.co2e_total
```

### After

```python
from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.resolution.engine import ResolutionEngine
from greenlang.factors.resolution.request import ResolutionRequest

class CBAMCalculator:
    def __init__(self, engine: ResolutionEngine):
        # Engine is composed once per process at app boot,
        # backed by the catalog repository.  Pack code never sees the
        # repository directly.
        self.engine = engine

    def emissions_for(self, cn_code: str, mass_t: float) -> float:
        result = self.engine.resolve(ResolutionRequest(
            activity=f"cbam-good-{cn_code}",
            method_profile=MethodProfile.EU_CBAM,
            jurisdiction="EU",
            extras={"cn_code": cn_code},
            target_unit="tCO2e/t",
        ))
        # converted_co2e_per_unit is populated when target_unit is set
        return mass_t * (
            result.converted_co2e_per_unit
            or result.gas_breakdown.co2e_total_kg
        )
```

The engine is constructed once at app boot — see
`greenlang/factors/api_endpoints.py::_build_repo_engine` for the
production wiring backed by `FactorCatalogRepository`.

---

## What is allowed to call the catalog directly

Only **bootstrap / admin / ops** code may construct a
`FileCatalogRepository`, `PgCatalogRepository`, or call
`catalog_repository.get(...)` directly.  Examples:

- `greenlang/factors/cli.py` — operator CLI
- `greenlang/factors/etl/ingest.py` — ingestion pipeline
- `greenlang/factors/watch/*.py` — release / rollback machinery
- `greenlang/factors/api_endpoints.py` — composes the engine
- `applications/GL-*-APP/scripts/seed_*.py` — pack bootstrap scripts

Every such file is on the explicit allow-list in
`tests/factors/test_no_direct_catalog_calls_in_packs.py`.  Adding a
new entry requires a code-owner sign-off; the default answer is
"refactor to call `ResolutionEngine.resolve()`".

---

## Guard test

A CI guard at
`tests/factors/test_no_direct_catalog_calls_in_packs.py` walks every
`applications/GL-*-APP/` directory and fails the build if it finds:

- `catalog_repository.get(`
- `catalog.get_factor(`
- `FileCatalogRepository(`
- `PgCatalogRepository(`
- `ef.get_emission_factor_by_cn_code(`  *(legacy CBAM helper)*
- `EmissionFactorClient(...).get_factor(` or `.get_fuel_factor(` *(legacy SDK)*

outside the allow-list.  The failure message names the offending
`file:line` and points the developer back to this document.

---

## FAQ

**Q: My workflow needs a factor for an activity that has no method
pack yet.**

Either (a) add a method pack — see
`greenlang/factors/method_packs/` for examples — or (b) reuse the
closest-fit profile and add a pack-level assumption.  Never bypass the
engine.

**Q: I just need to display all factors in a UI.**

That's a catalog-browse use case, not a policy workflow — use
`/v1/factors` or the `FactorCatalogService` directly.  The guard test
only checks `applications/GL-*-APP/`.

**Q: The engine raised `ResolutionError`. What do I do?**

Surface it.  This means no factor in the catalog satisfies the
method-pack `SelectionRule` for the supplied context.  The correct fix
is upstream (richer activity description, more jurisdiction context, or
a missing method-pack mapping), not a silent fall-through to a raw
catalog id.

---

*Last updated: FY27.Q2 — owner: Factors squad.*
