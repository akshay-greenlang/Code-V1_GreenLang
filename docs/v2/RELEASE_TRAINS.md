# GreenLang V2 Release Trains

## Cadence

- **PR train**: fast smoke and contract checks.
- **Nightly train**: full V2 matrix + security governance.
- **RC train (weekly)**: full gate bundle + determinism + soak evidence generation.
- **Stable train (biweekly/monthly)**: promoted only from successful RC.

## Required Tiers

1. Tier-0 smoke (contracts, docs, lightweight tests)
2. Tier-1 conformance (`gl v2 gate`)
3. Tier-2 security governance
4. Tier-3 determinism replay checks
5. Tier-4 RC soak and go/no-go record

## Promotion Criteria

- no failing blocking jobs
- no unresolved critical security findings
- determinism parity green for regulated-critical workflows
- go/no-go approval recorded
- EUDR/GHG/ISO14064 backend parity tests green in PR/nightly/RC lanes

## Cycle Evidence

- `docs/v2/RELEASE_TRAIN_CYCLE_LOG.md` records cycle outcomes.
- Phase 5 exit requires two consecutive green cycles in this log.
