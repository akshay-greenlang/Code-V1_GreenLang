# GreenLang V2 Release Train Cycle Log

## Cycle 1

- Train type: nightly
- Window: 2026-03-27T02:00:00Z -> 2026-03-27T02:46:00Z
- Required checks:
  - `python -m greenlang.cli.main v2 validate-contracts`: Pass
  - `python -m greenlang.cli.main v2 runtime-checks`: Pass
  - `python -m greenlang.cli.main v2 agent-checks`: Pass
  - `python -m greenlang.cli.main v2 connector-checks`: Pass
  - `python -m greenlang.cli.main v2 docs-check`: Pass
  - `pytest -q tests/v2` (or nightly subset): Pass (includes EUDR/GHG/ISO14064 backend parity suite)
  - `python -m greenlang.cli.main v2 gate`: Pass
- Outcome: Green

## Cycle 2

- Train type: rc
- Window: 2026-03-28T02:00:00Z -> 2026-03-28T02:58:00Z
- Required checks:
  - `python -m greenlang.cli.main v2 validate-contracts`: Pass
  - `python -m greenlang.cli.main v2 runtime-checks`: Pass
  - `python -m greenlang.cli.main v2 agent-checks`: Pass
  - `python -m greenlang.cli.main v2 connector-checks`: Pass
  - `python -m greenlang.cli.main v2 docs-check`: Pass
  - `pytest -q tests/v2`: Pass (includes EUDR/GHG/ISO14064 backend parity suite)
  - `python -m greenlang.cli.main v2 gate`: Pass
  - UX gate workflow (`v2-frontend-ux-ci`): Pass (includes EUDR/GHG/ISO14064 web API parity tests)
- Outcome: Green

## Consecutive-Cycle Exit Gate

Two consecutive release-train cycles are green for mandatory blockers.
