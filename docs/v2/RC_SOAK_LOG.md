# GreenLang V2 RC Soak Log

## Candidate

- Candidate version: v2.0-rc1
- Start: 2026-03-28T01:30:00Z
- End: 2026-03-28T02:00:00Z
- Evidence scope: V2 contracts, policy, determinism, release-train, UX blockers
- Regulated app parity scope: EUDR, GHG, ISO14064 native runtime + web API paths

## Soak Checks

| Check | Result | Notes |
| --- | --- | --- |
| `python -m greenlang.cli.main v2 validate-contracts` | Pass | V2 profile contracts valid. |
| `python -m greenlang.cli.main v2 runtime-checks` | Pass | Command grammar and runtime conventions aligned. |
| `python -m greenlang.cli.main v2 docs-check` | Pass | Required V2 docs present. |
| `python -m greenlang.cli.main v2 gate` | Pass | Combined release gate checks passed. |
| `pytest -q tests/v2` | Pass | V2 policy/reliability/determinism tests passed (includes regulated app backend parity). |
| UX visual baseline tests | Pass | Shell visual baseline + regulated app API tests green. |

## Conclusion

V2 RC soak acceptance met for go/no-go review.
