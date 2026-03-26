# GreenLang v1 Release Candidate Process

## RC Creation

1. Run `gl v1 gate`.
2. Run `pytest -q tests/v1`.
3. Tag candidate build and publish release notes draft.

## Soak Window

- Duration: 3-5 business days
- Monitor:
  - contract gate stability
  - policy gate stability
  - deterministic replay parity for target workflows

## Rollback Criteria

- Any release gate failure in soak period
- Signed-pack baseline regression
- Determinism parity failure on designated workflows

## Rollback Procedure

1. Mark RC as failed.
2. Revert to last passing baseline tag.
3. Open remediation issue and rerun full v1 gate set.

