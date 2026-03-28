# GreenLang V2 Pilot Pack Cohort

This pilot cohort is the Phase 2 control set used to enforce V2 tier lifecycle checks in CI and runtime.

## Cohort Summary

- total pilot packs: 12
- tiers covered: `experimental`, `candidate`, `supported`, `regulated-critical`
- source of truth (machine-readable): `greenlang/ecosystem/packs/v2_tier_registry.yaml`

## Pilot Cohort Table

| Pack Slug | App Scope | Tier | Promotion Status | Required Evidence |
| --- | --- | --- | --- | --- |
| demo | GL-CBAM-APP | experimental | pilot-approved | owner/support |
| demo-test | GL-CBAM-APP | experimental | pilot-approved | owner/support |
| boiler_replacement | GL-CBAM-APP | experimental | pilot-approved | owner/support |
| fuel_ai | GL-GHG-APP | candidate | candidate-approved | owner/support, docs_contract |
| report_ai | GL-CSRD-APP | candidate | candidate-approved | owner/support, docs_contract |
| forecast_sarima_ai | GL-GHG-APP | candidate | candidate-approved | owner/support, docs_contract |
| emissions-core | GL-VCCI-Carbon-APP | supported | supported-approved | owner/support, docs_contract, signed, security_scan |
| hvac-measures | GL-ISO14064-APP | supported | supported-approved | owner/support, docs_contract, signed, security_scan |
| cement-lca | GL-CSRD-APP | supported | supported-approved | owner/support, docs_contract, signed, security_scan |
| boiler-solar | GL-EUDR-APP | regulated-critical | regulated-approved | owner/support, docs_contract, signed, security_scan, determinism |
| waste_heat_recovery_ai | GL-GHG-APP | regulated-critical | regulated-approved | owner/support, docs_contract, signed, security_scan, determinism |
| industrial_process_heat_ai | GL-VCCI-Carbon-APP | regulated-critical | regulated-approved | owner/support, docs_contract, signed, security_scan, determinism |

## Evidence Requirements by Tier

- `experimental`: owner + support channel must be defined.
- `candidate`: experimental requirements + docs contract evidence.
- `supported`: candidate requirements + signed artifact + security scan evidence.
- `regulated-critical`: supported requirements + determinism evidence for regulated workflow.
