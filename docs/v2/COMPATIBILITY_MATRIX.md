# GreenLang V2 Compatibility Matrix

## Matrix Dimensions

- app profile (`cbam`, `csrd`, `vcci`, `eudr`, `ghg`, `iso14064`)
- runtime contract version (`greenlang-v2`)
- pack quality tier (`experimental`, `candidate`, `supported`, `regulated-critical`)
- release lane (`smoke`, `full`, `rc`)

## App x Runtime x Pipeline Contract

| App | V2 Profile Path | Contract Version | Runtime |
| --- | --- | --- | --- |
| GL-CBAM-APP | `applications/GL-CBAM-APP/v2` | `2.0` | `greenlang-v2` |
| GL-CSRD-APP | `applications/GL-CSRD-APP/CSRD-Reporting-Platform/v2` | `2.0` | `greenlang-v2` |
| GL-VCCI-Carbon-APP | `applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/v2` | `2.0` | `greenlang-v2` |
| GL-EUDR-APP | `applications/GL-EUDR-APP/v2` | `2.0` | `greenlang-v2` |
| GL-GHG-APP | `applications/GL-GHG-APP/v2` | `2.0` | `greenlang-v2` |
| GL-ISO14064-APP | `applications/GL-ISO14064-APP/v2` | `2.0` | `greenlang-v2` |

## Pack Tier Compatibility Rules

| Tier | Runtime Allowed | Signature Required | Release Lane |
| --- | --- | --- | --- |
| experimental | `greenlang-v2` | No | smoke |
| candidate | `greenlang-v2` | Optional | full |
| supported | `greenlang-v2` | Yes | full + rc |
| regulated-critical | `greenlang-v2` | Yes | full + rc (blocking) |

## Exit Rules

1. all selected V2 profiles must pass `gl v2 validate-contracts`.
2. all selected V2 profiles must pass `gl v2 runtime-checks`.
3. `gl v2 gate` must be green in release candidate lanes.

## Enforcement Commands (CI and Runtime)

- `python -m greenlang.cli.main v2 validate-contracts`
- `python -m greenlang.cli.main v2 runtime-checks`
- `python -m greenlang.cli.main v2 gate`
- `python -c "from pathlib import Path; from greenlang.v2.pack_tiers import validate_tier_registry; p=Path('greenlang/ecosystem/packs/v2_tier_registry.yaml'); errs=validate_tier_registry(p); raise SystemExit(1 if errs else 0)"`
- `pytest -q tests/v2/test_pack_tiers.py tests/v2/test_pack_tier_lifecycle.py`

## Pilot Cohort Mapping

- Human-readable cohort: `docs/v2/PILOT_PACK_COHORT.md`
- Machine-readable cohort: `greenlang/ecosystem/packs/v2_tier_registry.yaml`
- Runtime enforcement entrypoint: `greenlang/ecosystem/packs/installer.py` (`_enforce_v2_tier_lifecycle`)
